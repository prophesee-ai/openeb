/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#ifndef __TREUZELL_COMMAND_DEFINITION_H
#define __TREUZELL_COMMAND_DEFINITION_H

#define TZ_FAILURE_FLAG (0x80000000) /* Sent by device when a command fails */
#define TZ_WRITE_FLAG (0x40000000)   /* Add to a property to request write */

#define TZ_UNKNOWN_CMD (0 | TZ_FAILURE_FLAG) /* Command not implemented */

/* For backward compatibility with the first firmwares,
 * use TZ_LEGACY_READ_REGFPGA_32 and TZ_LEGACY_WRITE_REGFPGA_32
 * instead of TZ_PROP_DEVICE_REG32 | TZ_WRITE_FLAG
 */
#define TZ_LEGACY_READ_REGFPGA_32 (0x55)
#define TZ_LEGACY_WRITE_REGFPGA_32 (0x56)

#define TZ_PROP_FPGA_STATE (0x71)
#define TZ_RESET_FPGA_MAGIC (0xB007F26A)
#define TZ_PROP_SERIAL (0x72)

#define TZ_PROP_RELEASE_VERSION (0x79)
#define TZ_PROP_BUILD_DATE (0x7A)

#define TZ_PROP_DEVICES (0x10000)
#define TZ_PROP_DEVICE_NAME (0x10001)
#define TZ_PROP_DEVICE_IF_FREQ (0x10002)
#define TZ_PROP_DEVICE_COMPATIBLE (0x10003)
#define TZ_PROP_DEVICE_ENABLE (0x10010)
#define TZ_PROP_DEVICE_REG32 (0x10102)
#define TZ_PROP_DEVICE_STREAM (0x10200)
#define TZ_PROP_DEVICE_OUTPUT_FORMAT (0x10201)

#endif
