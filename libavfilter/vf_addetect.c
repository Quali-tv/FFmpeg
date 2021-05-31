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
 * Video ad range detector
 */

#include <float.h>
#include <koku/koku.h>
#include <math.h>
#include <time.h>

#include "avfilter.h"
#include "internal.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/time.h"
#include "libavutil/timestamp.h"
#include "libavutil/tree.h"
#include "scene_sad.h"

#define NUM_AD_DETECT_INFO 1000
#define MIP_MAP_SIZE 4

typedef struct AudioLevelContext {
  long count;
  double mean;
  double m2;
  double delta;
  double delta2;
} AudioLevelContext;

typedef struct AdDetectContext {
  const AVClass *class;

  // configuration
  const char *server_address;
  int server_port;
  const char *application_id;
  const char *session_id;
  const char *context_id;
  int overwrite_existing_data;
  double black_threshold;
  double white_threshold;

  void *koku_ctx;
  int koku_index;

  AVRational video_time_base;

  // silence detection
  double current_silence_duration;
  double noise;
  int64_t nb_samples;
  int nb_channels;

  AudioLevelContext *audio_levels;
  int audio_levels_need_reset;

  int64_t *silence_start;
  int64_t *silence_duration;

  int64_t frame_end;

  void (*silence_detect)(struct AdDetectContext *s, AVFrame *frame,
                         int nb_samples, int sample_rate, AVRational time_base);

  // scene detection
  int bit_depth;
  int linesize;
  int width;
  int height;
  ff_scene_sad_fn sad;

  int w, h;
  AVRational frame_rate;
  int audio_format, video_format;

  uint8_t *frame;
  uint8_t *prev_frame;
  uint8_t *free_frame;
  uint16_t *acc_frame;
  double prev_mafd;
} AdDetectContext;

typedef int (*ad_detect_pixel_score_fn)(const AdDetectContext *s,
                                        const uint8_t pixel);

#define OFFSET(x) offsetof(AdDetectContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption addetect_options[] = {
    {"server_address",
     "set koku application server address",
     OFFSET(server_address),
     AV_OPT_TYPE_STRING,
     {.str = "api.spotz.ai"},
     0,
     0,
     FLAGS},
    {"server_port",
     "set koku application server port",
     OFFSET(server_port),
     AV_OPT_TYPE_INT,
     {.i64 = 443},
     1,
     64000,
     FLAGS},
    {"application_id",
     "set koku application id",
     OFFSET(application_id),
     AV_OPT_TYPE_STRING,
     {.str = NULL},
     0,
     0,
     FLAGS},
    {"session_id",
     "set koku session id",
     OFFSET(session_id),
     AV_OPT_TYPE_STRING,
     {.str = NULL},
     0,
     0,
     FLAGS},
    {"context_id",
     "set koku context id",
     OFFSET(context_id),
     AV_OPT_TYPE_STRING,
     {.str = NULL},
     0,
     0,
     FLAGS},
    {"overwrite_existing_data",
     "overwrite existing data if found",
     OFFSET(overwrite_existing_data),
     AV_OPT_TYPE_BOOL,
     {.i64 = 0},
     0,
     1,
     FLAGS},
    {"black_threshold",
     "set blackness threshold",
     OFFSET(black_threshold),
     AV_OPT_TYPE_DOUBLE,
     {.dbl = .1},
     0,
     1.0,
     FLAGS},
    {"white_threshold",
     "set whiteness threshold",
     OFFSET(white_threshold),
     AV_OPT_TYPE_DOUBLE,
     {.dbl = .9},
     0,
     1.0,
     FLAGS},
    {"noise",
     "set silence noise tolerance",
     OFFSET(noise),
     AV_OPT_TYPE_DOUBLE,
     {.dbl = 0.01},
     0,
     DBL_MAX,
     FLAGS},
    {NULL}};

AVFILTER_DEFINE_CLASS(addetect);

static av_always_inline double simple_encode(const double pts, int *index,
                                             double val) {
  int64_t xor_key = (int64_t)(pts * 0x1fff * (++*index));
  unsigned char *a = (unsigned char *)&xor_key;
  unsigned char *b = (unsigned char *)&val;
  for (int i = 0; i < sizeof(val); ++i) {
    b[i] ^= a[i];
  }
  return val;
}

static av_always_inline void audio_level_reset(AdDetectContext *s) {
  for (int i = 0; i < s->nb_channels; ++i) {
    AudioLevelContext *a = &s->audio_levels[i];
    a->count = 0;
    a->delta = 0;
    a->delta2 = 0;
    a->mean = 0;
    a->m2 = 0;
  }
}

static av_always_inline void audio_level_update(AdDetectContext *s,
                                                const double sample,
                                                const int current_sample) {
  int channel = current_sample % s->nb_channels;
  AudioLevelContext *a = &s->audio_levels[channel];
  ++a->count;
  a->delta = sample - a->mean;
  a->mean += a->delta / a->count;
  a->delta2 = sample - a->mean;
  a->m2 += a->delta * a->delta2;
}

static av_always_inline void silence_update(AdDetectContext *s, AVFrame *frame,
                                            int is_silence, int current_sample,
                                            int sample_rate,
                                            AVRational time_base) {
  int channel = current_sample % s->nb_channels;
  if (is_silence) {
    if (s->silence_start[channel] == INT64_MIN) {
      s->silence_start[channel] =
          frame->pts + av_rescale_q(current_sample / s->nb_channels + 1,
                                    (AVRational){1, sample_rate}, time_base);
      s->silence_duration[channel] = INT64_MIN;
    } else {
      const int64_t end_pts =
          frame ? frame->pts + av_rescale_q(current_sample / s->nb_channels,
                                            (AVRational){1, sample_rate},
                                            time_base)
                : s->frame_end;
      s->silence_duration[channel] = end_pts - s->silence_start[channel];
    }
  } else {
    s->silence_start[channel] = INT64_MIN;
    s->silence_duration[channel] = INT64_MIN;
  }
}

#define SILENCE_DETECT(name, type)                                       \
  static void silence_detect_##name(AdDetectContext *s, AVFrame *frame,  \
                                    int nb_samples, int sample_rate,     \
                                    AVRational time_base) {              \
    const type *p = (const type *)frame->data[0];                        \
    const type noise = s->noise;                                         \
    int i;                                                               \
                                                                         \
    if (s->audio_levels_need_reset) {                                    \
      audio_level_reset(s);                                              \
      s->audio_levels_need_reset = 0;                                    \
    }                                                                    \
                                                                         \
    for (i = 0; i < nb_samples; i++, p++) {                              \
      audio_level_update(s, (double)*p, i);                              \
      silence_update(s, frame, *p<noise && * p> - noise, i, sample_rate, \
                     time_base);                                         \
    }                                                                    \
  }

SILENCE_DETECT(dbl, double)
SILENCE_DETECT(flt, float)
SILENCE_DETECT(s32, int32_t)
SILENCE_DETECT(s16, int16_t)

static int query_formats(AVFilterContext *ctx) {
  static const enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_YUV420P,
                                                AV_PIX_FMT_NONE};
  static const enum AVSampleFormat sample_fmts[] = {AV_SAMPLE_FMT_FLTP,
                                                    AV_SAMPLE_FMT_NONE};

  AVFilterLink *inlink = ctx->inputs[0];
  AVFilterFormats *formats = NULL;
  AVFilterChannelLayouts *layouts = NULL;
  int ret = AVERROR(EINVAL);

  formats = ff_make_format_list(sample_fmts);
  if ((ret = ff_formats_ref(formats, &inlink->outcfg.formats)) < 0 ||
      (layouts = ff_all_channel_counts()) == NULL ||
      (ret = ff_channel_layouts_ref(layouts, &inlink->outcfg.channel_layouts)) <
          0)
    return ret;

  formats = ff_all_samplerates();
  if ((ret = ff_formats_ref(formats, &inlink->outcfg.samplerates)) < 0)
    return ret;

  AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
  if (!fmts_list) return AVERROR(ENOMEM);
  return ff_set_common_formats(ctx, fmts_list);
}

static int config_audio_input(AVFilterLink *inlink) {
  AVFilterContext *ctx = inlink->dst;
  AdDetectContext *s = ctx->priv;

  s->nb_samples = FFMAX(1, inlink->sample_rate);
  s->nb_channels = inlink->channels;
  s->audio_format = inlink->format;

  s->audio_levels = av_malloc_array(sizeof(*s->audio_levels), s->nb_channels);
  if (!s->audio_levels) return AVERROR(ENOMEM);

  s->silence_start = av_malloc_array(sizeof(*s->silence_start), s->nb_channels);
  if (!s->silence_start) return AVERROR(ENOMEM);

  s->silence_duration =
      av_malloc_array(sizeof(*s->silence_duration), s->nb_channels);
  if (!s->silence_duration) return AVERROR(ENOMEM);

  for (int c = 0; c < s->nb_channels; c++) {
    s->silence_start[c] = INT64_MIN;
    s->silence_duration[c] = INT64_MIN;
  }

  switch (inlink->format) {
    case AV_SAMPLE_FMT_DBL:
      s->silence_detect = silence_detect_dbl;
      break;
    case AV_SAMPLE_FMT_FLT:
    case AV_SAMPLE_FMT_FLTP:
      s->silence_detect = silence_detect_flt;
      break;
    case AV_SAMPLE_FMT_S32:
      s->noise *= INT32_MAX;
      s->silence_detect = silence_detect_s32;
      break;
    case AV_SAMPLE_FMT_S16:
      s->noise *= INT16_MAX;
      s->silence_detect = silence_detect_s16;
      break;
    default:
      av_log(s, AV_LOG_ERROR, "Unknown sample format: %d\n", inlink->format);
      return AVERROR(ENOMEM);
  }

  av_log(s, AV_LOG_INFO,
         "server address: %s, server port: %d, application id: %s, session id: "
         "%s, context id: "
         "%s, overwrite: %d, width: %d, height: %d, bit depth: %d, black "
         "threshold: "
         "%f, white threshold: %f\n",
         s->server_address, s->server_port, s->application_id, s->session_id,
         s->context_id, s->overwrite_existing_data, s->width, s->height,
         s->bit_depth, s->black_threshold, s->white_threshold);

  return 0;
}

static int config_video_input(AVFilterLink *inlink) {
  AVFilterContext *ctx = inlink->dst;
  AdDetectContext *s = ctx->priv;

  const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

  s->koku_index = 0;
  s->bit_depth = desc->comp[0].depth;
  s->black_threshold *= (1 << s->bit_depth);
  s->white_threshold *= (1 << s->bit_depth);

  s->w = inlink->w;
  s->h = inlink->h;
  s->frame_rate = inlink->frame_rate;
  s->video_format = inlink->format;

  s->linesize = av_image_get_linesize(inlink->format, inlink->w, 0);
  s->width = (s->linesize >> (s->bit_depth > 8)) / (MIP_MAP_SIZE >> 1);
  s->height = inlink->h / (MIP_MAP_SIZE >> 1);

  s->sad = ff_scene_sad_get_fn(8);
  if (!s->sad) return AVERROR(EINVAL);

  s->frame = calloc(s->width * s->height, sizeof(uint8_t));
  if (!s->frame) return AVERROR(EINVAL);

  s->free_frame = calloc(s->width * s->height, sizeof(uint8_t));
  if (!s->free_frame) return AVERROR(EINVAL);

  s->acc_frame = calloc(s->width * s->height, sizeof(uint16_t));
  if (!s->acc_frame) return AVERROR(EINVAL);

  s->prev_frame = NULL;

  s->video_time_base = inlink->time_base;

  if (!s->server_address) return AVERROR(EINVAL);
  if (!s->server_port) return AVERROR(EINVAL);
  if (!s->application_id) return AVERROR(EINVAL);
  if (!s->session_id) return AVERROR(EINVAL);
  if (!s->context_id) return AVERROR(EINVAL);

  s->koku_ctx =
      Koku_create(s->server_address, s->server_port, s->application_id,
                  s->session_id, s->context_id, s->overwrite_existing_data);

  if (!s->koku_ctx) return AVERROR(EINVAL);

  av_log(s, AV_LOG_INFO,
         "server address: %s, server port: %d, application id: %s, session id: "
         "%s, context id: "
         "%s, overwrite: %d, width: %d, height: %d, bit depth: %d, black "
         "threshold: "
         "%f, white threshold: %f\n",
         s->server_address, s->server_port, s->application_id, s->session_id,
         s->context_id, s->overwrite_existing_data, s->width, s->height,
         s->bit_depth, s->black_threshold, s->white_threshold);

  return 0;
}

static double get_scene_score(AdDetectContext *s, uint8_t *frame,
                              uint8_t *prev_frame) {
  uint64_t sad = 0;
  const int linesize = s->width;
  const int count = s->width * s->height * (1 << s->bit_depth);

  emms_c();
  s->sad(prev_frame, linesize, frame, linesize, s->width, s->height, &sad);

  const double score = (1.0f * sad) / count;
  if (score > .5) {
    av_log(s, AV_LOG_INFO,
           "get_scene_score: %f --> width: %d height: %d sad: %" PRIu64
           " count: %d\n",
           score, s->width, s->height, sad, count);
  }
  return score;
}

static int get_black_score(const AdDetectContext *s, const uint8_t pixel) {
  return pixel <= s->black_threshold;
}

static int get_white_score(const AdDetectContext *s, const uint8_t pixel) {
  return pixel >= s->white_threshold;
}

static double get_pixel_score(const AdDetectContext *s, const uint8_t *frame,
                              const ad_detect_pixel_score_fn score_fn) {
  const int h = s->height;
  const int w = s->width;
  uint64_t counter = 0;

  const uint8_t *p = &frame[0];
  for (int y = 0, pixel_index = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x, ++pixel_index) {
      counter += score_fn(s, p[pixel_index]);
    }
  }

  return (counter * 1.0) / (w * h);
}

static int filter_audio_frame(AVFilterLink *inlink, AVFrame *frame) {
  AVFilterContext *ctx = inlink->dst;
  AdDetectContext *s = ctx->priv;

  const int nb_channels = inlink->channels;
  const int srate = inlink->sample_rate;
  const int nb_samples = frame->nb_samples * nb_channels;

  s->frame_end =
      frame->pts + av_rescale_q(frame->nb_samples, (AVRational){1, srate},
                                inlink->time_base);

  s->silence_detect(s, frame, nb_samples, srate, inlink->time_base);

  int64_t silence_duration = INT64_MAX;
  for (int c = 0; c < s->nb_channels; ++c) {
    silence_duration = FFMIN(silence_duration, s->silence_duration[c]);
  }

  if (silence_duration == INT64_MIN) {
    s->current_silence_duration = 0.0;
  } else {
    s->current_silence_duration = silence_duration * av_q2d(inlink->time_base);
  }

  if (ctx->nb_outputs > 0) {
    return ff_filter_frame(ctx->outputs[0], frame);
  } else {
    av_frame_free(&frame);
    return 0;
  }
}

static int filter_video_frame(AVFilterLink *inlink, AVFrame *frame) {
  static double audio_levels[32];
  AVFilterContext *ctx = inlink->dst;
  AdDetectContext *s = ctx->priv;

  {
    const int h = frame->height;
    const int w = frame->width;
    const int count = s->width * s->height;
    uint8_t *src_p = &frame->data[0][0];
    uint16_t *dest_p = &s->acc_frame[0];

    memset(dest_p, 0, count * sizeof(uint16_t));
    for (int y = 0, index = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x, ++src_p, ++index) {
        dest_p[index / MIP_MAP_SIZE] += *src_p;
      }
    }

    for (int i = 0; i < count; ++i) {
      s->frame[i] = dest_p[i] / MIP_MAP_SIZE;
    }
  }

  if (s->prev_frame) {
    int encode_index = 1;
    const double pts = frame->pts * av_q2d(s->video_time_base);
    const double scene_score = simple_encode(
        pts, &encode_index, get_scene_score(s, s->frame, s->prev_frame));
    const double black_score = simple_encode(
        pts, &encode_index, get_pixel_score(s, s->frame, get_black_score));
    const double white_score = simple_encode(
        pts, &encode_index, get_pixel_score(s, s->frame, get_white_score));
    const double silence_duration =
        simple_encode(pts, &encode_index, s->current_silence_duration);

    for (int i = 0; i < s->nb_channels; ++i) {
      audio_levels[i] =
          simple_encode(pts, &encode_index, s->audio_levels[i].mean);
    }

    Koku_add_scene_detection_info(s->koku_ctx, pts, scene_score, black_score,
                                  white_score, silence_duration, audio_levels,
                                  s->nb_channels);

    s->audio_levels_need_reset = 1;

    if ((++s->koku_index % NUM_AD_DETECT_INFO) == 0) {
      Koku_transmit_scene_detection_info(s->koku_ctx);
    }

    s->free_frame = s->prev_frame;
  }

  s->prev_frame = s->frame;
  s->frame = s->free_frame;

  if (ctx->nb_outputs > 0) {
    return ff_filter_frame(ctx->outputs[0], frame);
  } else {
    av_frame_free(&frame);
    return 0;
  }
}

static void addetect_uninit(AVFilterContext *ctx) {}

static const AVFilterPad addetect_inputs[] = {
    {
        .name = "default_audio",
        .type = AVMEDIA_TYPE_AUDIO,
        .config_props = config_audio_input,
        .filter_frame = filter_audio_frame,
    },
    {
        .name = "default_video",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = config_video_input,
        .filter_frame = filter_video_frame,
    },
    {NULL}};

static const AVFilterPad addetect_outputs[] = {{NULL}};

AVFilter ff_vf_addetect = {
    .name = "addetect",
    .description = NULL_IF_CONFIG_SMALL("Detect ad video intervals."),
    .priv_size = sizeof(AdDetectContext),
    .query_formats = query_formats,
    .inputs = addetect_inputs,
    .outputs = addetect_outputs,
    .uninit = addetect_uninit,
    .priv_class = &addetect_class,
};