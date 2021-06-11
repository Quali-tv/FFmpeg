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
 * Spotz License
 */

#include "vf_spotz_license.h"

#include <license++/license-c-bindings.h>
#include <time.h>

static const unsigned char license_manager_signature_key[] = {
    0x5B, 0x6A, 0xF5, 0x93, 0xED, 0xAB, 0xB3, 0x10,
    0xF5, 0xBE, 0x00, 0xE6, 0x4F, 0x1B, 0x70, 0xC8};

static const IssuingAuthorityParameters authorities[] = {
    {"sample-license-authority", "Sample License Authority",
     /*key pair*/ NULL,
     /*private key*/ "",
     "LS0tLS1CRUdJTiBQVUJMSUMgS0VZLS0tLS0KTUlJQklEQU5CZ2txaGtpRzl3MEJBUUVGQUFPQ"
     "0FRMEFNSUlCQ0FLQ0FRRUF0eGdKUENWSUhQanhWamcwNWUydQpaNURqNDNIdDF0WFlUK3VkVV"
     "RTL3RrSlgyQzltcWg4aktQdU9mQXV6cWJQK2V6ckF0Q0hDem1ETmxmRTBqZU5TClVUZlFWbFh"
     "xNzd3UGh6ajZWNm1lWTNlcmYxK0pUY0dROTVDRTdBbFFmaW9ObVoxTU45MFI5ejZCWUkwUmlU"
     "eHUKQVFXckZqdm1rMUsrZ1RRN2dPbVV1WEx1MzJ2R2k1UTRwSUpUcEkwTFhCSnlCclU0SzVlN"
     "1ZNWFowdCtvV1Fzdwpjcm05bkJYWVpleVRJcUZ2VmVkbEpxZTArTm9GTzN4T3VUdjFKK2Jxa1"
     "Z4UW5CVzNDZ3JHa2NPRlZFa0RDRE44CkZoZ0N5SEpJRDliZkdsNlBJUEp0TE94UlF2M21KK25"
     "qS01ycXlrcE9panpZc3JSNFJZeURXTDZ2bWEyWlJkaVkKS1FJQkVRPT0KLS0tLS1FTkQgUFVC"
     "TElDIEtFWS0tLS0tCg==",
     87600U, 1}};

static av_always_inline const char *date_to_string(time_t dt) {
  static char buffer[512];
  struct tm *info;

  info = localtime(&dt);
  strftime(buffer, 80, "%x - %I:%M%p", info);
  return buffer;
}

int check_license(void *s, const char *license, const char *application_id) {
  license_key_register_init(license_manager_signature_key, authorities);

  const void *lm = license_manager_create();
  if (!lm) {
    return -10;
  }

  void *l = license_create();
  if (!l) {
    return -20;
  }

  if (!license_load(l, license)) {
    return -30;
  }

  av_log(s, AV_LOG_INFO, "Licensing Authority: %s\n",
         license_get_issuing_authority_id(l));
  av_log(s, AV_LOG_INFO, "Issued To: %s\n", license_get_licensee(l));
  av_log(s, AV_LOG_INFO, "Issued On: %s\n",
         date_to_string(license_get_issue_date(l)));
  av_log(s, AV_LOG_INFO, "Expires On: %s\n",
         date_to_string(license_get_expiry_date(l)));

  if (!license_manager_validate(lm, l, 0, "")) {
    return -40;
  }

  if (strcmp(license_get_licensee(l), application_id)) {
    return -50;
  }

  return 0;
}