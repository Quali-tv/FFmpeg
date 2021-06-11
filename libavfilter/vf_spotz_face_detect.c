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
 * Video face detector
 */

#include <float.h>
#include <kao/kao.h>
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
#include "vf_spotz_license.h"

typedef struct FaceDetectContext {
  const AVClass *class;

  // configuration
  const char *license;
  const char *server_address;
  int server_port;
  const char *application_id;
  const char *session_id;
  const char *context_id;
  int overwrite_existing_data;

  void *kao_ctx;
  int max_kao_wait_time;
  int max_kao_distance;

  AVRational video_time_base;
  int width, height, bit_depth, linesize;
  int video_format;
} FaceDetectContext;

#define OFFSET(x) offsetof(FaceDetectContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption spotz_face_detect_options[] = {
    {"license",
     "set kao license",
     OFFSET(license),
     AV_OPT_TYPE_STRING,
     {.str = ""},
     0,
     0,
     FLAGS},
    {"server_address",
     "set kao application server address",
     OFFSET(server_address),
     AV_OPT_TYPE_STRING,
     {.str = "api.spotz.ai"},
     0,
     0,
     FLAGS},
    {"server_port",
     "set kao application server port",
     OFFSET(server_port),
     AV_OPT_TYPE_INT,
     {.i64 = 443},
     1,
     64000,
     FLAGS},
    {"application_id",
     "set kao application id",
     OFFSET(application_id),
     AV_OPT_TYPE_STRING,
     {.str = NULL},
     0,
     0,
     FLAGS},
    {"session_id",
     "set kao session id",
     OFFSET(session_id),
     AV_OPT_TYPE_STRING,
     {.str = NULL},
     0,
     0,
     FLAGS},
    {"context_id",
     "set kao context id",
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
    {"max_kao_wait_time",
     "max time to wait for a kao frame to be processed in milliseconds",
     OFFSET(max_kao_wait_time),
     AV_OPT_TYPE_UINT64,
     {.i64 = 1},
     0,
     5000,
     FLAGS},
    {NULL}};

AVFILTER_DEFINE_CLASS(spotz_face_detect);

static int query_formats(AVFilterContext *ctx) {
  static const enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_RGB24,
                                                AV_PIX_FMT_NONE};

  AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
  if (!fmts_list) return AVERROR(ENOMEM);
  return ff_set_common_formats(ctx, fmts_list);
}

static int config_video_input(AVFilterLink *inlink) {
  AVFilterContext *ctx = inlink->dst;
  FaceDetectContext *s = ctx->priv;
  int r;

  r = check_license(s, s->license, s->application_id);
  if (r) {
    av_log(s, AV_LOG_ERROR, "Failed to validate license. Reason: %d\n", r);
    return AVERROR(EINVAL);
  }

  const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

  s->bit_depth = desc->comp[0].depth;
  s->width = inlink->w;
  s->height = inlink->h;
  s->video_format = inlink->format;
  s->linesize = av_image_get_linesize(inlink->format, inlink->w, 0);
  s->video_time_base = inlink->time_base;

  if (!s->server_address) return AVERROR(EINVAL);
  if (!s->server_port) return AVERROR(EINVAL);
  if (!s->application_id) return AVERROR(EINVAL);
  if (!s->session_id) return AVERROR(EINVAL);
  if (!s->context_id) return AVERROR(EINVAL);

  s->kao_ctx = Kao_create(s->server_address, s->server_port, s->application_id,
                          s->session_id, s->context_id,
                          s->overwrite_existing_data, s->max_kao_wait_time);
  if (!s->kao_ctx) return AVERROR(EINVAL);

  av_log(s, AV_LOG_INFO,
         "server address: %s, server port: %d, application id: %s, session id: "
         "%s, context id: "
         "%s, overwrite: %d, width: %d, height: %d, bit depth: %d, max wait "
         "time %d\n",
         s->server_address, s->server_port, s->application_id, s->session_id,
         s->context_id, s->overwrite_existing_data, s->width, s->height,
         s->bit_depth, s->max_kao_wait_time);

  return 0;
}

static int filter_video_frame(AVFilterLink *inlink, AVFrame *frame) {
  AVFilterContext *ctx = inlink->dst;
  FaceDetectContext *s = ctx->priv;

  const int h = frame->height;
  const int w = frame->width;
  const double pts = frame->pts * av_q2d(s->video_time_base);

  // Kao_process_faces(s->kao_ctx, pts, NULL, 0, 0);

  return ff_filter_frame(ctx->outputs[0], frame);
}

static void face_detect_uninit(AVFilterContext *ctx) {
  if (ctx) {
    FaceDetectContext *s = ctx->priv;
    if (s && s->kao_ctx) {
      Kao_delete(s->kao_ctx);
    }
  }
}

static const AVFilterPad face_detect_inputs[] = {
    {
        .name = "default_video",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = config_video_input,
        .filter_frame = filter_video_frame,
    },
    {NULL}};

static const AVFilterPad face_detect_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    {NULL}};

AVFilter ff_vf_spotz_face_detect = {
    .name = "spotz_face_detect",
    .description = NULL_IF_CONFIG_SMALL("Detect faces in video."),
    .priv_size = sizeof(FaceDetectContext),
    .query_formats = query_formats,
    .inputs = face_detect_inputs,
    .outputs = face_detect_outputs,
    .uninit = face_detect_uninit,
    .priv_class = &spotz_face_detect_class,
    .flags = AVFILTER_FLAG_SLICE_THREADS,
};