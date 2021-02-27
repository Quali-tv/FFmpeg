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

#include "vf_addetect.h"

#include <float.h>
#include <math.h>

#include "avfilter.h"
#include "internal.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/timestamp.h"
#include "libavutil/tree.h"
#include "scene_sad.h"

double last_ad_start = 0.0;
double last_ad_duration = 0.0;

typedef struct list_node {
  double pts;
  int likely_was_in_ad;
  struct list_node *next;
} list_node;

typedef struct list {
  list_node *head;
  list_node *tail;
} list;

typedef struct AdDetectContext {
  const AVClass *class;

  double min_scene_threshold;
  double last_scene_threshold;
  double ad_min_duration;
  double ad_max_duration;

  int64_t ad_start;
  int64_t ad_end;
  int64_t last_picref_pts;
  int64_t last_scene_pts;
  int ad_started;
  int64_t ad_index;

  AVRational time_base;

  // scene detection
  list *scene_list;
  int bit_depth;
  int nb_planes;
  ptrdiff_t width[4];
  ptrdiff_t height[4];
  ff_scene_sad_fn sad;
  double prev_mafd;
  AVFrame *prev_picref;
} AdDetectContext;

#define OFFSET(x) offsetof(AdDetectContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption addetect_options[] = {
    {"min_d",
     "set minimum ad duration",
     OFFSET(ad_min_duration),
     AV_OPT_TYPE_DOUBLE,
     {.dbl = 10},
     0,
     1024,
     FLAGS},
    {"max_d",
     "set maximum ad duration",
     OFFSET(ad_max_duration),
     AV_OPT_TYPE_DOUBLE,
     {.dbl = 10},
     0,
     1024,
     FLAGS},
    {"mt",
     "set minimum scene score threshold",
     OFFSET(min_scene_threshold),
     AV_OPT_TYPE_DOUBLE,
     {.dbl = .75},
     0,
     1,
     FLAGS},
    {"ldt",
     "set last scene duration threshold",
     OFFSET(last_scene_threshold),
     AV_OPT_TYPE_DOUBLE,
     {.dbl = 30},
     0,
     100,
     FLAGS},
    {NULL}};

AVFILTER_DEFINE_CLASS(addetect);

static int query_formats(AVFilterContext *ctx) {
  static const enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_YUV420P,
                                                AV_PIX_FMT_NONE};

  AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
  if (!fmts_list) return AVERROR(ENOMEM);
  return ff_set_common_formats(ctx, fmts_list);
}

static int clear_list(AdDetectContext *s, list *list) {
  list_node *p = list->head;
  while (p) {
    list_node *s = p;
    p = p->next;
    av_free(s);
  }

  list->head = NULL;
  list->tail = NULL;
  return 0;
}

static int add_node_to_list(AdDetectContext *s, list *list, const double pts,
                            const int likely_was_in_ad) {
  list_node *node = av_mallocz(sizeof(list_node));
  if (!node) {
    av_log(s, AV_LOG_ERROR, "Failed to alloc node for list!");
    return AVERROR(ENOMEM);
  }

  node->pts = pts;
  node->likely_was_in_ad = likely_was_in_ad;

  if (!list->head) {
    list->head = node;
    list->tail = node;
  } else {
    list->tail->next = node;
    list->tail = node;
  }

  return 0;
}

static int config_input(AVFilterLink *inlink) {
  AVFilterContext *ctx = inlink->dst;
  AdDetectContext *s = ctx->priv;

  const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
  const int is_yuv = !(desc->flags & AV_PIX_FMT_FLAG_RGB) &&
                     (desc->flags & AV_PIX_FMT_FLAG_PLANAR) &&
                     desc->nb_components >= 3;

  s->bit_depth = desc->comp[0].depth;
  s->nb_planes = is_yuv ? 1 : av_pix_fmt_count_planes(inlink->format);

  for (int plane = 0; plane < s->nb_planes; plane++) {
    const ptrdiff_t line_size =
        av_image_get_linesize(inlink->format, inlink->w, plane);
    const int vsub = desc->log2_chroma_h;

    s->width[plane] = line_size >> (s->bit_depth > 8);
    s->height[plane] =
        plane == 1 || plane == 2 ? AV_CEIL_RSHIFT(inlink->h, vsub) : inlink->h;
  }

  s->sad = ff_scene_sad_get_fn(s->bit_depth == 8 ? 8 : 16);
  if (!s->sad) return AVERROR(EINVAL);

  s->scene_list = av_mallocz(sizeof(*s->scene_list));
  if (!s->scene_list) {
    return AVERROR(ENOMEM);
  } else {
    add_node_to_list(s, s->scene_list, 0, 0);
  }

  s->time_base = inlink->time_base;

  av_log(s, AV_LOG_INFO, "ad_min_duration:%f ad_max_duration:%f\n",
         s->ad_min_duration, s->ad_max_duration);

  return 0;
}

static double get_scene_score(AVFilterContext *ctx, AVFrame *frame) {
  double ret = 0;
  AdDetectContext *s = ctx->priv;
  AVFrame *prev_picref = s->prev_picref;

  if (prev_picref && frame->height == prev_picref->height &&
      frame->width == prev_picref->width) {
    uint64_t sad = 0;
    double mafd, diff;
    uint64_t count = 0;

    for (int plane = 0; plane < s->nb_planes; plane++) {
      uint64_t plane_sad;
      s->sad(prev_picref->data[plane], prev_picref->linesize[plane],
             frame->data[plane], frame->linesize[plane], s->width[plane],
             s->height[plane], &plane_sad);
      sad += plane_sad;
      count += s->width[plane] * s->height[plane];
    }

    emms_c();
    mafd = (double)sad / count / (1ULL << (s->bit_depth - 8));
    diff = fabs(mafd - s->prev_mafd);
    ret = av_clipf(FFMIN(mafd, diff) / 100., 0, 1);
    s->prev_mafd = mafd;
    av_frame_free(&prev_picref);
  }
  s->prev_picref = av_frame_clone(frame);
  return ret;
}

static void check_ad_end(AVFilterContext *ctx) {
  AdDetectContext *s = ctx->priv;

  const double ad_duration = (s->ad_end - s->ad_start) * av_q2d(s->time_base);
  if (ad_duration >= s->ad_min_duration && ad_duration <= s->ad_max_duration) {
    last_ad_start = s->ad_start * av_q2d(s->time_base);
    last_ad_duration = ad_duration;
    av_log(s, AV_LOG_INFO,
           "index:%lld ad_start:%s ad_end:%s "
           "ad_duration:%f\n",
           s->ad_index++, av_ts2timestr(s->ad_start, &s->time_base),
           av_ts2timestr(s->ad_end, &s->time_base), ad_duration);
  } else if (ad_duration > s->ad_max_duration) {
    av_log(s, AV_LOG_ERROR,
           "LARGE ad: ad_start:%s ad_end:%s "
           "ad_duration:%f\n",
           av_ts2timestr(s->ad_start, &s->time_base),
           av_ts2timestr(s->ad_end, &s->time_base), ad_duration);
  }
}

static int filter_frame(AVFilterLink *inlink, AVFrame *picref) {
  AVFilterContext *ctx = inlink->dst;
  AdDetectContext *s = ctx->priv;

  const double scene_score = get_scene_score(ctx, picref);
  if (scene_score >= s->min_scene_threshold) {
    const double difference_in_time =
        (picref->pts - s->last_scene_pts) * av_q2d(s->time_base);
    const int likely_was_in_ad =
        difference_in_time <= s->last_scene_threshold ? 1 : 0;

    if (s->scene_list->tail) {
      s->scene_list->tail->likely_was_in_ad = likely_was_in_ad;
    }

    av_log(s, AV_LOG_VERBOSE, "scene detected: %s score: %f prev_in_ad: %d\n",
           av_ts2timestr(picref->pts, &s->time_base), scene_score,
           likely_was_in_ad);

    if (likely_was_in_ad) {
      if (!s->ad_started) {
        s->ad_started = 1;
        s->ad_start = picref->pts;

        av_dict_set(&picref->metadata, "lavfi.ad_start",
                    av_ts2timestr(s->ad_start, &s->time_base), 0);

        av_log(s, AV_LOG_INFO, "ad started pts:%s\n",
               av_ts2timestr(s->ad_start, &s->time_base));
      }
    } else if (s->ad_started) {
      s->ad_started = 0;
      s->ad_end = s->last_scene_pts;

      check_ad_end(ctx);

      av_dict_set(&picref->metadata, "lavfi.ad_end",
                  av_ts2timestr(s->ad_end, &s->time_base), 0);

      clear_list(s, s->scene_list);
    }

    add_node_to_list(s, s->scene_list, picref->pts, 0);
    s->last_scene_pts = picref->pts;
  }

  s->last_picref_pts = picref->pts;
  return ff_filter_frame(inlink->dst->outputs[0], picref);
}

static av_cold void uninit(AVFilterContext *ctx) {
  AdDetectContext *s = ctx->priv;

  if (s->ad_started) {
    s->ad_end = s->last_picref_pts;
    check_ad_end(ctx);
  }
}

static const AVFilterPad addetect_inputs[] = {{
                                                  .name = "default",
                                                  .type = AVMEDIA_TYPE_VIDEO,
                                                  .config_props = config_input,
                                                  .filter_frame = filter_frame,
                                              },
                                              {NULL}};

static const AVFilterPad addetect_outputs[] = {{
                                                   .name = "default",
                                                   .type = AVMEDIA_TYPE_VIDEO,
                                               },
                                               {NULL}};

AVFilter ff_vf_addetect = {
    .name = "addetect",
    .description = NULL_IF_CONFIG_SMALL("Detect ad video intervals."),
    .priv_size = sizeof(AdDetectContext),
    .query_formats = query_formats,
    .inputs = addetect_inputs,
    .outputs = addetect_outputs,
    .uninit = uninit,
    .priv_class = &addetect_class,
};
