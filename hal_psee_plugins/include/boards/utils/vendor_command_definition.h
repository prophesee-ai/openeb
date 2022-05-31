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

#ifndef METAVISION_HAL_VENDOR_COMMAND_DEFINITION_H
#define METAVISION_HAL_VENDOR_COMMAND_DEFINITION_H

// Vendor Command supported
#define CMD_READ (0x50)  /*Read Transaction*/
#define CMD_WRITE (0x51) /*Write Transaction*/
#define CMD_ERASE (0x52) /*Erase Transaction*/

#define CMD_READ_REGFPGA_32 (0x55)  /*Read Transaction*/
#define CMD_WRITE_REGFPGA_32 (0x56) /*Write Transaction*/

#define CMD_WRITE_VEC (0x60)

#define CMD_WRITE_VEC_REGFPGA_32 (0x61)
#define CMD_WRITE_VEC_SLAVE_REGFPGA_32 (0x62)

#define CMD_READ_VERSION_FX3 (0x70)
#define CMD_CHECK_FPGA_BOOT_STATE (0x71)
#define CMD_READ_SYSTEM_ID (0x72)
#define CMD_READ_SYSTEM_VERSION (0x73)

#define CMD_CTRL_WRITE_DONOTHING (0x74) /*Write Transaction*/

#define CMD_READ_FX3_ID (0x78)
#define CMD_READ_FX3_RELEASE_VERSION (0x79)
#define CMD_READ_FX3_BUILD_DATE (0x7A)
#define CMD_READ_FX3_VERSION_CONTROL_ID (0x7B)

#define CMD_RESET_FPGA (0x7E)
#define CMD_RESET_FX3 (0x7F)

/*IMU's vendor commands*/

/*USB vendor request to read an IMU register */
#define VCMD_IMU_READ (0x80)

/* USB vendor request to write on IMU register, read mask applied in the command  */
#define VCMD_IMU_WRITE (0x81)

/* USB vendor request to get data regarding imu state, used to debug  */
#define VCMD_IMU_MESSAGE (0x82)

/* USB vendor request to indicate a reading operation in ProntoIMU using EVK-Mode on Pronto  */
#define CMD_READ_PRONTO_IMU (0x83)

#define CMD_CX3_GET_TEMPERATURE (0x84) /* Read Cx3 I2C temperature sensor */
/* USB vendor request to write data to SPI flash connected. The flash page size is
 * fixed to 256 bytes. The memory address to start the write is provided in the
 * index field of the request. The maximum allowed request length is 4KB. */
#define CY_FX_RQT_SPI_FLASH_WRITE (0xC2)

/* USB vendor request to read data from SPI flash connected. The flash page size is
 * fixed to 256 bytes. The memory address to start the read is provided in the index
 * field of the request. The maximum allowed request length is 4KB. */
#define CY_FX_RQT_SPI_FLASH_READ (0xC3)

/* USB vendor request to erase a sector on SPI flash connected. The flash sector
 * size is fixed to 64KB. The sector address is provided in the index field of
 * the request. The erase is carried out if the value field is non-zero. If this
 * is zero, then the request returns the write in progress (WIP) bit. WIP should
 * be 0 before issuing any further transactions. */
#define CY_FX_RQT_SPI_FLASH_ERASE_POLL (0xC4)

/* USB vendor request to write data to SPI flash connected. The flash subsector size is
 * fixed to 4KB. The memory address to start the write is provided in the
 * index field of the request. The maximum allowed request length is 4KB.*/
#define CY_FX_RQT_SPI_FLASH_N25Q_WRITE (0xC5)

/* USB vendor request to read data from SPI flash connected. The flash subsector size is
 * fixed to 4KB. The memory address to start the read is provided in the index
 * field of the request. The maximum allowed request length is 4KB. */
#define CY_FX_RQT_SPI_FLASH_N25Q_READ (0xC6)

/* USB vendor request to erase a sector on SPI flash connected. The flash sector
 * size is fixed to 64KB. The sector address is provided in the index field of
 * the request. The erase is carried out if the value field is non-zero. If this
 * is zero, then the request returns the write in progress (WIP) bit. WIP should
 * be 0 before issuing any further transactions. */
#define CY_FX_RQT_SPI_FLASH_N25Q_ERASE_POLL (0xC7)

// Event : New command available
#define VC_NEW_CMD (1 << 0) /* Event flag that the thread waits */

#endif // METAVISION_HAL_VENDOR_COMMAND_DEFINITION_H
