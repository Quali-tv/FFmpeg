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

typedef struct AdDetectContext {
  const AVClass *class;

  // configuration
  const char *server_address;
  int server_port;
  const char *application_id;
  const char *context_id;
  double black_threshold;
  double white_threshold;

  void *koku_ctx;
  int koku_index;

  AVRational time_base;

  // scene detection
  int bit_depth;
  int nb_planes;
  ptrdiff_t width[4];
  ptrdiff_t height[4];
  ff_scene_sad_fn sad;
  AVFrame *prev_frame;
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
     {.str = NULL},
     0,
     0,
     FLAGS},
    {"server_port",
     "set koku application server port",
     OFFSET(server_port),
     AV_OPT_TYPE_INT,
     {.i64 = 0},
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
    {"context_id",
     "set koku context id",
     OFFSET(context_id),
     AV_OPT_TYPE_STRING,
     {.str = NULL},
     0,
     0,
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
    {NULL}};

AVFILTER_DEFINE_CLASS(addetect);

static int query_formats(AVFilterContext *ctx) {
  static const enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_YUV420P,
                                                AV_PIX_FMT_NONE};

  AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
  if (!fmts_list) return AVERROR(ENOMEM);
  return ff_set_common_formats(ctx, fmts_list);
}

static int config_input(AVFilterLink *inlink) {
  AVFilterContext *ctx = inlink->dst;
  AdDetectContext *s = ctx->priv;

  const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
  const int is_yuv = !(desc->flags & AV_PIX_FMT_FLAG_RGB) &&
                     (desc->flags & AV_PIX_FMT_FLAG_PLANAR) &&
                     desc->nb_components >= 3;

  s->koku_index = 0;
  s->bit_depth = desc->comp[0].depth;
  s->nb_planes = is_yuv ? 1 : av_pix_fmt_count_planes(inlink->format);
  s->black_threshold *= (1 << s->bit_depth);
  s->white_threshold *= (1 << s->bit_depth);

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

  s->time_base = inlink->time_base;

  if (!s->server_address) return AVERROR(EINVAL);
  if (!s->server_port) return AVERROR(EINVAL);
  if (!s->application_id) return AVERROR(EINVAL);
  if (!s->context_id) return AVERROR(EINVAL);

  s->koku_ctx = Koku_create(s->server_address, s->server_port,
                            s->application_id, s->context_id);

  if (!s->koku_ctx) return AVERROR(EINVAL);

  av_log(s, AV_LOG_INFO,
         "server address: %s, server port: %d, application id: %s, context id: "
         "%s, bit depth: "
         "%d, nb planes: %d, black threshold: "
         "%f, white threshold: %f\n",
         s->server_address, s->server_port, s->application_id, s->context_id,
         s->bit_depth, s->nb_planes, s->black_threshold, s->white_threshold);

  return 0;
}

static double get_scene_score(AdDetectContext *s, AVFrame *frame,
                              AVFrame *prev_frame) {
  if (prev_frame && frame->height == prev_frame->height &&
      frame->width == prev_frame->width) {
    uint64_t sad = 0;
    uint64_t count = 0;

    for (int plane = 0; plane < s->nb_planes; plane++) {
      uint64_t plane_sad;

      emms_c();
      s->sad(prev_frame->data[plane], prev_frame->linesize[plane],
             frame->data[plane], frame->linesize[plane], s->width[plane],
             s->height[plane], &plane_sad);
      sad += plane_sad;
      count += s->width[plane] * s->height[plane];
    }

    const double mafd = (1.0f * sad) / count;
    const double diff = fabs(mafd - s->prev_mafd);
    return av_clipf(FFMIN(mafd, diff) / 100., 0, 1);
  }

  return 0;
}

static int get_black_score(const AdDetectContext *s, const uint8_t pixel) {
  return pixel <= s->black_threshold;
}

static int get_white_score(const AdDetectContext *s, const uint8_t pixel) {
  return pixel >= s->white_threshold;
}

static double get_pixel_score(const AdDetectContext *s, const AVFrame *frame,
                              const ad_detect_pixel_score_fn score_fn) {
  const int h = frame->height;
  const int w = frame->width;
  uint64_t counter = 0;

  const uint8_t *p = frame->data[0];
  for (int y = 0, pixel_index = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x, ++pixel_index) {
      counter += score_fn(s, p[pixel_index]);
    }
  }

  return (counter * 1.0) / (w * h);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame) {
  AVFilterContext *ctx = inlink->dst;
  AdDetectContext *s = ctx->priv;
  AVFrame *prev_frame = s->prev_frame;

  if (prev_frame) {
    const double pts = frame->pts * av_q2d(s->time_base);
    const double scene_score = get_scene_score(s, frame, prev_frame);
    const double black_score = get_pixel_score(s, frame, get_black_score);
    const double white_score = get_pixel_score(s, frame, get_white_score);

    Koku_add_scene_detection_info(s->koku_ctx, pts, scene_score, black_score,
                                  white_score);

    if ((++s->koku_index % NUM_AD_DETECT_INFO) == 0) {
      Koku_transmit_scene_detection_info(s->koku_ctx);
    }

    av_frame_free(&prev_frame);
  }

  s->prev_frame = av_frame_clone(frame);
  return ff_filter_frame(inlink->dst->outputs[0], frame);
}

static void addetect_uninit(AVFilterContext *ctx) {}

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
    .uninit = addetect_uninit,
    .priv_class = &addetect_class,
};
