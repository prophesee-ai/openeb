/* SPDX-License-Identifier: GPL-2.0 */
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
#define TZ_PROP_DEVICE_REG32 (0x10102)
#define TZ_PROP_DEVICE_STREAM (0x10200)

#endif
