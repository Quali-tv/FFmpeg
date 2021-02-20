/*
 * Copyright (c) 2021 Ernest Wilkerson
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Video duplicate range detector
 */

#include <float.h>
#include <math.h>

#include "avfilter.h"
#include "internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/timestamp.h"
#include "libavutil/tree.h"

#define FRAME_PIXEL_COMPRESSION_LEVEL 32
#define MAX_FRAMES 32
#define SCAN_DEPTH 13

#define FORCE_INLINE inline __attribute__((always_inline))

inline uint32_t rotl32(uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}

inline uint64_t rotl64(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

#define ROTL32(x, y) rotl32(x, y)
#define ROTL64(x, y) rotl64(x, y)

#define BIG_CONSTANT(x) (x##LLU)

//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

FORCE_INLINE uint32_t getblock32(const uint32_t *p, int i) { return p[i]; }

FORCE_INLINE uint64_t getblock64(const uint64_t *p, int i) { return p[i]; }

//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche

FORCE_INLINE uint32_t fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

//----------

FORCE_INLINE uint64_t fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;

  return k;
}

typedef struct hash_node {
  int64_t value[2];
  struct hash_node *next;
} hash_node;

typedef struct hash_list {
  hash_node *head;
  hash_node *tail;
} hash_list;

typedef struct checksum_ctx {
  hash_list *list[SCAN_DEPTH];
  int64_t frame_count;
  int64_t frame_depth;
  int64_t frame_pts_offset;
  struct AVTreeNode *lut[SCAN_DEPTH];
} checksum_ctx;

typedef struct DuplicateDetectContext {
  const AVClass *class;
  double duplicate_min_duration_time;  ///< minimum duration of duplicate in
                                       ///< seconds
  int64_t duplicate_min_duration;  ///< minimum duration of duplicate, expressed
                                   ///< in timebase units
  double duplicate_max_duration_time;  ///< maximum duration of duplicate in
                                       ///< seconds
  int64_t duplicate_max_duration;  ///< maximum duration of duplicate, expressed
  ///< in timebase units

  int64_t duplicate_fail_distance;
  int64_t duplicate_fail_allowance;

  int64_t duplicate_index;
  int64_t duplicate_source_start;
  int64_t duplicate_start;  ///< pts start time of the first black picture
  int64_t duplicate_end;    ///< pts end time of the last black picture
  int64_t last_picref_pts;  ///< pts of the last input picture
  int duplicate_started;

  unsigned int pixel_fingerprint_th;

  AVRational time_base;
  AVRational frame_rate;
  checksum_ctx *counter;
  hash_list *blacklist;
  double *weights;

  int frame_size;
  double *frame;
  int nb_frames;
  int frame_index;
} DuplicateDetectContext;

#define OFFSET(x) offsetof(DuplicateDetectContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption dupdetect_options[] = {
    {"fd",
     "set allowed duplicate failures in frames. this allows misses while "
     "matching",
     OFFSET(duplicate_fail_allowance),
     AV_OPT_TYPE_INT,
     {.i64 = 5},
     0,
     INT_MAX,
     FLAGS},
    {"wd",
     "set window duration in frames",
     OFFSET(nb_frames),
     AV_OPT_TYPE_INT,
     {.i64 = 5},
     0,
     INT_MAX,
     FLAGS},
    {"duplicate_window_duration",
     "set window duration in frames",
     OFFSET(nb_frames),
     AV_OPT_TYPE_INT,
     {.i64 = 5},
     0,
     INT_MAX,
     FLAGS},
    {"mind",
     "set minimum detected black duration in seconds",
     OFFSET(duplicate_min_duration_time),
     AV_OPT_TYPE_DOUBLE,
     {.dbl = 10},
     0,
     DBL_MAX,
     FLAGS},
    {"duplicate_min_duration",
     "set minimum detected black duration in seconds",
     OFFSET(duplicate_min_duration_time),
     AV_OPT_TYPE_DOUBLE,
     {.dbl = 10},
     0,
     DBL_MAX,
     FLAGS},
    {"maxd",
     "set maximum detected black duration in seconds",
     OFFSET(duplicate_max_duration_time),
     AV_OPT_TYPE_DOUBLE,
     {.dbl = 30},
     0,
     DBL_MAX,
     FLAGS},
    {"duplicate_max_duration",
     "set maximum detected black duration in seconds",
     OFFSET(duplicate_max_duration_time),
     AV_OPT_TYPE_DOUBLE,
     {.dbl = 30},
     0,
     DBL_MAX,
     FLAGS},
    {"pixel_fingerprint_th",
     "set the pixel fingerprint threshold",
     OFFSET(pixel_fingerprint_th),
     AV_OPT_TYPE_INT,
     {.i64 = 10},
     0,
     20,
     FLAGS},
    {"pix_th",
     "set the pixel fingerprint threshold",
     OFFSET(pixel_fingerprint_th),
     AV_OPT_TYPE_INT,
     {.i64 = 10},
     0,
     20,
     FLAGS},
    {NULL}};

AVFILTER_DEFINE_CLASS(dupdetect);

static int query_formats(AVFilterContext *ctx) {
  static const enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_YUV420P,
                                                AV_PIX_FMT_NONE};

  AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
  if (!fmts_list) return AVERROR(ENOMEM);
  return ff_set_common_formats(ctx, fmts_list);
}

static int64_t round_to_nearest(int64_t n, int64_t nearest) {
  const int64_t a = (n / nearest) * nearest;
  const int64_t b = a + nearest;
  return (n - a > b - n) ? b : a;
}

static int interval_intersects(int64_t a, int64_t b, int64_t c, int64_t d) {
  if (a >= c && a <= d) return 1;
  if (c >= a && c <= b) return 1;
  return 0;
}

static int config_input(AVFilterLink *inlink) {
  AVFilterContext *ctx = inlink->dst;
  DuplicateDetectContext *s = ctx->priv;
  const int w2 = inlink->w >> 1;
  const int h2 = inlink->h >> 1;

  s->weights = av_mallocz(inlink->w * inlink->h * sizeof(*s->weights));
  if (!s->weights) return AVERROR(ENOMEM);

  for (int y = 0, i = 0; y < inlink->h; y++) {
    const double y_weight = (h2 - fabs((1.0 * h2) - y)) / h2;
    for (int x = 0; x < inlink->w; x++, ++i) {
      const double x_weight = (w2 - fabs((1.0 * w2) - x)) / w2;
      s->weights[i] = y_weight * x_weight;
    }
  }

  s->time_base = inlink->time_base;
  s->frame_rate = inlink->frame_rate;

  s->duplicate_min_duration =
      s->duplicate_min_duration_time / av_q2d(s->time_base);
  s->duplicate_max_duration =
      s->duplicate_max_duration_time / av_q2d(s->time_base);

  s->counter = av_mallocz(sizeof(*s->counter));
  if (!s->counter) return AVERROR(ENOMEM);

  for (int i = 0; i < SCAN_DEPTH; ++i) {
    s->counter->list[i] = av_mallocz(sizeof(*s->counter->list[i]));
    if (!s->counter->list[i]) return AVERROR(ENOMEM);
  }

  s->blacklist = av_mallocz(sizeof(*s->blacklist));
  if (!s->blacklist) return AVERROR(ENOMEM);

  s->frame_size = FRAME_PIXEL_COMPRESSION_LEVEL;
  s->frame = av_calloc(s->frame_size * s->nb_frames, sizeof(*s->frame));
  if (!s->frame) return AVERROR(ENOMEM);

  s->counter->frame_depth = (uint64_t)s->nb_frames;
  s->counter->frame_pts_offset =
      (uint64_t)(s->nb_frames / av_q2d(s->frame_rate) / av_q2d(s->time_base));

  av_log(s, AV_LOG_INFO,
         "duplicate_min_duration:%s duplicate_max_duration:%s "
         "pixel_fingerprint_th:%d frame_size:%d nb_frames:%d\n",
         av_ts2timestr(s->duplicate_min_duration, &s->time_base),
         av_ts2timestr(s->duplicate_max_duration, &s->time_base),
         s->pixel_fingerprint_th, s->frame_size, s->nb_frames);

  return 0;
}

static void check_duplicate_end(AVFilterContext *ctx) {
  DuplicateDetectContext *s = ctx->priv;
  int should_blacklist = 0;

  const int64_t duplicate_duration = (s->duplicate_end - s->duplicate_start);
  if (duplicate_duration >= s->duplicate_min_duration) {
    // local loop or freeze
    if (((duplicate_duration << 1) + s->duplicate_source_start) >=
        s->duplicate_start) {
      should_blacklist = 1;
      av_log(
          s, AV_LOG_VERBOSE,
          "BLACKLIST: Reason: Local loop detected. duplicate_start:%s "
          "duplicate_end:%s duplicate_duration:%s source:%s\n",
          av_ts2timestr(s->duplicate_start, &s->time_base),
          av_ts2timestr(s->duplicate_end, &s->time_base),
          av_ts2timestr(s->duplicate_end - s->duplicate_start, &s->time_base),
          av_ts2timestr(s->duplicate_source_start, &s->time_base));
    } else {
      // check previous intervals
      hash_node *p = s->blacklist->head;
      while (p) {
        if (interval_intersects(s->duplicate_start, s->duplicate_end,
                                p->value[0], p->value[1]) ||
            interval_intersects(s->duplicate_source_start,
                                s->duplicate_source_start + duplicate_duration,
                                p->value[0], p->value[1])) {
          should_blacklist = 1;
          av_log(s, AV_LOG_VERBOSE,
                 "BLACKLIST: Reason: Found in previous blacklist interval. "
                 "duplicate_start:%s duplicate_end:%s duplicate_duration:%s "
                 "source:%s\n",
                 av_ts2timestr(s->duplicate_start, &s->time_base),
                 av_ts2timestr(s->duplicate_end, &s->time_base),
                 av_ts2timestr(s->duplicate_end - s->duplicate_start,
                               &s->time_base),
                 av_ts2timestr(s->duplicate_source_start, &s->time_base));

          // shouldn't be possible; here just in case
          const int64_t interval_duplicate_duration =
              duplicate_duration - (s->duplicate_start - p->value[0]);
          if (interval_duplicate_duration > (p->value[1] - p->value[0])) {
            av_log(s, AV_LOG_VERBOSE,
                   "BLACKLIST: Extending interval duration from %lld to %lld\n",
                   (p->value[1] - p->value[0]), interval_duplicate_duration);
            p->value[1] = p->value[0] + interval_duplicate_duration;
          }
          break;
        }
        p = p->next;
      }
    }
  }

  if (should_blacklist) {
    hash_node *node = av_mallocz(sizeof(hash_node));

    node->value[0] = s->duplicate_start;
    node->value[1] = s->duplicate_end;

    if (!s->blacklist->head) {
      s->blacklist->head = node;
      s->blacklist->tail = node;
    } else {
      s->blacklist->tail->next = node;
      s->blacklist->tail = node;
    }
  } else {
    if (duplicate_duration >= s->duplicate_min_duration &&
        duplicate_duration <= s->duplicate_max_duration) {
      av_log(
          s, AV_LOG_INFO,
          "index:%lld duplicate_start:%s duplicate_end:%s "
          "duplicate_duration:%s "
          "source:%s\n",
          s->duplicate_index++,
          av_ts2timestr(s->duplicate_start, &s->time_base),
          av_ts2timestr(s->duplicate_end, &s->time_base),
          av_ts2timestr(s->duplicate_end - s->duplicate_start, &s->time_base),
          av_ts2timestr(s->duplicate_source_start, &s->time_base));
    } else if (duplicate_duration > s->duplicate_max_duration) {
      av_log(
          s, AV_LOG_ERROR,
          "LARGE DUPLICATE: duplicate_start:%s duplicate_end:%s "
          "duplicate_duration:%s "
          "source:%s\n",
          av_ts2timestr(s->duplicate_start, &s->time_base),
          av_ts2timestr(s->duplicate_end, &s->time_base),
          av_ts2timestr(s->duplicate_end - s->duplicate_start, &s->time_base),
          av_ts2timestr(s->duplicate_source_start, &s->time_base));
    }
  }
}

static int tree_cmp(const void *a, const void *b) {
  uint64_t *a_val = (uint64_t *)a;
  uint64_t *b_val = (uint64_t *)b;

  if (!a_val && !b_val) return 0;
  if (!a_val) return 1;
  if (!b_val) return -1;

  return memcmp(a_val, b_val, 2 * sizeof(uint64_t));
}

static int tree_insert(DuplicateDetectContext *s, struct AVTreeNode **rootp,
                       uint64_t key1[2], uint64_t key2[2], int64_t *pts) {
  uint64_t *existing = NULL;
  uint64_t *keys = av_mallocz(5 * sizeof(uint64_t));
  keys[0] = key1[0];
  keys[1] = key1[1];
  keys[2] = key2[0];
  keys[3] = key2[1];
  keys[4] = *((uint64_t *)pts);

  if (existing = av_tree_find(*rootp, keys, tree_cmp, NULL)) {
    *pts = ((int64_t *)existing)[4];
    av_free(keys);
    return 0;
  }

  struct AVTreeNode *next = av_mallocz(av_tree_node_size);
  const void *result = av_tree_insert(rootp, keys, tree_cmp, &next);
  if (result == NULL) {
    return 1;
  } else if (result == keys) {
    return 1;
  }

  av_free(keys);
  av_free(next);
  return 0;
}

static void MurmurHash3_x64_128(const void *key, const int len,
                                const uint32_t seed, void *out) {
  const uint8_t *data = (const uint8_t *)key;
  const int nblocks = len / 16;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  //----------
  // body

  const uint64_t *blocks = (const uint64_t *)(data);

  for (int i = 0; i < nblocks; i++) {
    uint64_t k1 = getblock64(blocks, i * 2 + 0);
    uint64_t k2 = getblock64(blocks, i * 2 + 1);

    k1 *= c1;
    k1 = ROTL64(k1, 31);
    k1 *= c2;
    h1 ^= k1;

    h1 = ROTL64(h1, 27);
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729;

    k2 *= c2;
    k2 = ROTL64(k2, 33);
    k2 *= c1;
    h2 ^= k2;

    h2 = ROTL64(h2, 31);
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5;
  }

  //----------
  // tail

  const uint8_t *tail = (const uint8_t *)(data + nblocks * 16);

  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch (len & 15) {
    case 15:
      k2 ^= ((uint64_t)tail[14]) << 48;
    case 14:
      k2 ^= ((uint64_t)tail[13]) << 40;
    case 13:
      k2 ^= ((uint64_t)tail[12]) << 32;
    case 12:
      k2 ^= ((uint64_t)tail[11]) << 24;
    case 11:
      k2 ^= ((uint64_t)tail[10]) << 16;
    case 10:
      k2 ^= ((uint64_t)tail[9]) << 8;
    case 9:
      k2 ^= ((uint64_t)tail[8]) << 0;
      k2 *= c2;
      k2 = ROTL64(k2, 33);
      k2 *= c1;
      h2 ^= k2;

    case 8:
      k1 ^= ((uint64_t)tail[7]) << 56;
    case 7:
      k1 ^= ((uint64_t)tail[6]) << 48;
    case 6:
      k1 ^= ((uint64_t)tail[5]) << 40;
    case 5:
      k1 ^= ((uint64_t)tail[4]) << 32;
    case 4:
      k1 ^= ((uint64_t)tail[3]) << 24;
    case 3:
      k1 ^= ((uint64_t)tail[2]) << 16;
    case 2:
      k1 ^= ((uint64_t)tail[1]) << 8;
    case 1:
      k1 ^= ((uint64_t)tail[0]) << 0;
      k1 *= c1;
      k1 = ROTL64(k1, 31);
      k1 *= c2;
      h1 ^= k1;
  };

  //----------
  // finalization

  h1 ^= len;
  h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  ((uint64_t *)out)[0] = h1;
  ((uint64_t *)out)[1] = h2;
}

static int update_rolling_sum(DuplicateDetectContext *s, checksum_ctx *c,
                              int64_t *pts, double *frame, double *prev_frame,
                              int frame_len) {
  int result = 0;
  int64_t temp_frame[MAX_FRAMES];

  for (int i = 0; i < SCAN_DEPTH; ++i) {
    int look_for_hash = 0;
    struct hash_node *node = NULL;
    uint64_t hash[2];
    uint64_t total_hash[2];

    for (int j = 0; j < frame_len; ++j) {
      const int64_t clamped_frame_value =
          round_to_nearest((int64_t)frame[j], 1 << i);
      temp_frame[j] = clamped_frame_value;
    }

    MurmurHash3_x64_128(temp_frame, frame_len * sizeof(*temp_frame), 0xdeadbeef,
                        hash);

    if (prev_frame) {
      node = c->list[i]->head;
      if (node) {
        c->list[i]->head = node->next;
        node->next = NULL;
      } else {
        c->list[i]->head = NULL;
        c->list[i]->tail = NULL;
      }
      look_for_hash = 1;
    }

    if (!node) node = av_mallocz(sizeof(hash_node));

    node->value[0] = hash[0];
    node->value[1] = hash[1];

    if (!c->list[i]->head) {
      c->list[i]->head = node;
      c->list[i]->tail = node;
    } else {
      c->list[i]->tail->next = node;
      c->list[i]->tail = node;
    }

    if (look_for_hash) {
      total_hash[0] = 0x13;
      total_hash[1] = 0x7fffffff;

      struct hash_node *p = c->list[i]->head;
      while (p) {
        total_hash[0] = total_hash[0] * 524287 + p->value[0];
        total_hash[1] = total_hash[1] * 524287 + p->value[1];
        p = p->next;
      }

      if (!tree_insert(s, &c->lut[i], total_hash, hash, pts)) {
        result = 1;
      }
    }
  }

  c->frame_count++;
  return result;
}

static int duplicate_counter(AVFilterContext *ctx, AVFrame *in,
                             int *duplicate_detected,
                             int *duplicate_frame_pts_offset,
                             int64_t *duplicate_source_pts) {
  DuplicateDetectContext *s = ctx->priv;
  const int linesize = in->linesize[0];
  const int w = in->width;
  const int h = in->height;
  const int frame_index = s->frame_index % s->nb_frames;
  double *frame = &s->frame[frame_index * s->frame_size];

  memset(frame, 0x0, s->frame_size * sizeof(*frame));

  const uint8_t *p = in->data[0];
  for (int y = 0, i = 0; y < h; y++) {
    for (int x = 0; x < w; x++, ++i) {
      frame[i % FRAME_PIXEL_COMPRESSION_LEVEL] += s->weights[i] * p[x];
    }
    p += linesize;
  }

  {
    checksum_ctx *c = s->counter;
    double *prev_frame = NULL;
    int64_t pts = in->pts;

    if (c->frame_count >= c->frame_depth) {
      const int prev_frame_index = (s->frame_index - c->frame_depth);
      prev_frame = &s->frame[prev_frame_index % s->nb_frames];
    }

    if (update_rolling_sum(s, c, &pts, frame, prev_frame, s->frame_size)) {
      *duplicate_detected = 1;
      *duplicate_frame_pts_offset = c->frame_pts_offset;
      *duplicate_source_pts = pts;
    }
  }

  s->frame_index++;
  return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *picref) {
  AVFilterContext *ctx = inlink->dst;
  DuplicateDetectContext *s = ctx->priv;
  int duplicate_detected = 0;
  int duplicate_frame_pts_offset = 0;
  int64_t duplicate_source_pts = 0;

  duplicate_counter(ctx, picref, &duplicate_detected,
                    &duplicate_frame_pts_offset, &duplicate_source_pts);

  if (duplicate_detected) {
    if (!s->duplicate_started) {
      if ((picref->pts - duplicate_source_pts) >= s->duplicate_min_duration) {
        s->duplicate_started = 1;
        s->duplicate_start = picref->pts - duplicate_frame_pts_offset;
        s->duplicate_source_start =
            duplicate_source_pts - duplicate_frame_pts_offset;

        av_dict_set(&picref->metadata, "lavfi.duplicate_start",
                    av_ts2timestr(s->duplicate_start, &s->time_base), 0);

        av_log(s, AV_LOG_VERBOSE, "duplicate pts:%s start:%s source:%s\n",
               av_ts2timestr(picref->pts, &s->time_base),
               av_ts2timestr(s->duplicate_start, &s->time_base),
               av_ts2timestr(s->duplicate_source_start, &s->time_base));
      }
    }

    s->duplicate_fail_distance = s->duplicate_fail_allowance;
  } else if (s->duplicate_started) {
    if (s->duplicate_fail_distance-- == 0) {
      s->duplicate_started = 0;
      s->duplicate_end =
          picref->pts - (s->duplicate_fail_allowance / av_q2d(s->frame_rate) /
                         av_q2d(s->time_base));

      check_duplicate_end(ctx);
      av_dict_set(&picref->metadata, "lavfi.duplicate_end",
                  av_ts2timestr(s->duplicate_end, &s->time_base), 0);
    }
  }

  s->last_picref_pts = picref->pts;
  return ff_filter_frame(inlink->dst->outputs[0], picref);
}

static av_cold void uninit(AVFilterContext *ctx) {
  DuplicateDetectContext *s = ctx->priv;

  av_freep(&s->counter);

  if (s->duplicate_started) {
    s->duplicate_end = s->last_picref_pts;
    check_duplicate_end(ctx);
  }
}

static const AVFilterPad dupdetect_inputs[] = {{
                                                   .name = "default",
                                                   .type = AVMEDIA_TYPE_VIDEO,
                                                   .config_props = config_input,
                                                   .filter_frame = filter_frame,
                                               },
                                               {NULL}};

static const AVFilterPad dupdetect_outputs[] = {{
                                                    .name = "default",
                                                    .type = AVMEDIA_TYPE_VIDEO,
                                                },
                                                {NULL}};

AVFilter ff_vf_dupdetect = {
    .name = "dupdetect",
    .description = NULL_IF_CONFIG_SMALL("Detect duplicate video intervals."),
    .priv_size = sizeof(DuplicateDetectContext),
    .query_formats = query_formats,
    .inputs = dupdetect_inputs,
    .outputs = dupdetect_outputs,
    .uninit = uninit,
    .priv_class = &dupdetect_class,
};
