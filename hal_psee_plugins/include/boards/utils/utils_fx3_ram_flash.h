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

#ifndef METAVISION_HAL_UTILS_FX3_RAM_FLASH_H
#define METAVISION_HAL_UTILS_FX3_RAM_FLASH_H

#include <vector>
#include <string>

#ifdef _MSC_VER
#define NOMINMAX // libusb.h includes windows.h which defines min max macros that we don't want
#endif
#include <libusb.h>

namespace LoadApplicativeFirmwareToFx3RAM {

//#define MAX_FWIMG_SIZE  (512 * 1024)        // Maximum size of the firmware binary.
//#define MAX_WRITE_SIZE  (2 * 1024)      // Max. size of data that can be written through one vendor command.
//
//#define VENDORCMD_TIMEOUT   (5000)      // Timeout for each vendor command is set to 5 seconds.
//
//#define GET_LSW(v)  ((unsigned short)((v) & 0xFFFF))
//#define GET_MSW(v)  ((unsigned short)((v) >> 16))
//

int read_firmware_image(const char *filename, unsigned char *buf, int *romsize);
int ram_write(libusb_device_handle *h, unsigned char *buf, unsigned int ramAddress, int len);

int fx3_usbboot_download(libusb_device_handle *h, const char *filename);

} // namespace LoadApplicativeFirmwareToFx3RAM

class FlashCmd {
public:
    uint8_t Write  = 0xC2;
    uint8_t Read   = 0xC3;
    uint8_t Erase  = 0xC4;
    uint8_t Status = 0xC4;

    int step      = 256;
    int erasestep = 65536;

    unsigned int default_test_sector = 700;

private:
    FlashCmd();

public:
    static FlashCmd FlashCmdFx3();
    static FlashCmd FlashCmdFpga();
    int flash_test(libusb_device_handle *dev_handle, int *err_bad_flash, bool erase, bool write, bool read);
    int flash(libusb_device_handle *dev_handle, const char *filename, unsigned long start_sector, long max_sector,
              long file_offset, int *err_bad_flash);
    int dump(libusb_device_handle *dev_handle, const char *filename, int *err_bad_flash);
    int flash_map(libusb_device_handle *dev_handle, long start_sector, const std::string &findexes,
                  const std::string &ftargets, int *err_bad_flash);
    int flash_calib3d(libusb_device_handle *dev_handle, long start_sector, const std::string &calib3d,
                      int *err_bad_flash);
    int dump_calib3d(libusb_device_handle *dev_handle, std::vector<uint8_t> &vdata, long start_sector,
                     int *err_bad_flash);

    int flash_offset_sector(libusb_device_handle *dev_handle, std::vector<uint8_t> &vdata, unsigned long start_sector,
                            long max_sector, int *err_bad_flash);
    int dump_flash(libusb_device_handle *dev_handle, std::vector<uint8_t> &vdata, int *err_bad_flash);
    int dump_flash_offset_sector(libusb_device_handle *dev_handle, std::vector<uint8_t> &vdata, long start_sector,
                                 long end_sector, int *err_bad_flash);

    int flash_serial(libusb_device_handle *dev_handle, int *err_bad_flash, long sector,
                     std::vector<uint8_t> serial_to_write);
    int flash_serial(libusb_device_handle *dev_handle, int *err_bad_flash, long sector, uint16_t serial_to_write);

    int flash_system_id(libusb_device_handle *dev_handle, int *err_bad_flash, long sector, long system_id_to_write);

    void erase_sector(libusb_device_handle *dev_handle, int sector_to_erase, long &num_err);

    bool read_sector(libusb_device_handle *dev_handle, int sector, std::vector<uint8_t> &vread, long &num_err);

    bool write_sector_over_erased_offset(libusb_device_handle *dev_handle, int sector, std::vector<uint8_t> &vdata,
                                         unsigned long offset, long &num_err);
    bool write_sector_over_erased(libusb_device_handle *dev_handle, int sector, std::vector<uint8_t> &vdata,
                                  long &num_err);
    void dump_data(const std::vector<uint8_t> &vdata);

private:
    bool wait_for_status(libusb_device_handle *dev_handle);
    static void coe_2_data(const std::string &ftargets, std::vector<uint8_t> &vdata, long &ntargets);
};

#endif // METAVISION_HAL_UTILS_FX3_RAM_FLASH_H
