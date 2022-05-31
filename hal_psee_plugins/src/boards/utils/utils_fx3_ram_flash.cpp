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

#include <sys/stat.h>
#ifndef _MSC_VER
#include <unistd.h>
#else
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <io.h>
#endif
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>

#include "metavision/hal/utils/hal_log.h"
#include "boards/utils/utils_fx3_ram_flash.h"

namespace LoadApplicativeFirmwareToFx3RAM {

#define MAX_FWIMG_SIZE (512 * 1024) // Maximum size of the firmware binary.
#define MAX_WRITE_SIZE (2 * 1024)   // Max. size of data that can be written through one vendor command.

#define VENDORCMD_TIMEOUT (5000) // Timeout for each vendor command is set to 5 seconds.

#define GET_LSW(v) ((unsigned short)((v)&0xFFFF))
#define GET_MSW(v) ((unsigned short)((v) >> 16))

const int i2c_eeprom_size[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};

int read_firmware_image(const char *filename, unsigned char *buf, int *romsize) {
    int fd;

    struct stat filestat;

    // Verify that the file size does not exceed our limits.
    if (stat(filename, &filestat) != 0) {
        MV_HAL_LOG_ERROR() << "Failed to stat file" << filename;
        return -1;
    }

    int filesize = filestat.st_size;
    if (filesize > MAX_FWIMG_SIZE) {
        MV_HAL_LOG_ERROR() << "File size exceeds maximum firmware image size";
        return -2;
    }

    fd = open(filename, O_RDONLY);
    if (fd < 0) {
        MV_HAL_LOG_ERROR() << "File not found";
        return -3;
    }
    ssize_t r = read(fd, buf, 2); /* Read first 2 bytes, must be equal to 'CY'    */
    if (r >= 2 && strncmp((char *)buf, "CY", 2)) {
        MV_HAL_LOG_ERROR() << "Image does not have 'CY' at start. aborting";
        return -4;
    }
    r = read(fd, buf, 1); /* Read 1 byte. bImageCTL   */
    if (r >= 1 && (buf[0] & 0x01)) {
        MV_HAL_LOG_ERROR() << "Image does not contain executable code";
        return -5;
    }
    if (romsize != 0)
        *romsize = i2c_eeprom_size[(buf[0] >> 1) & 0x07];

    r = read(fd, buf, 1); /* Read 1 byte. bImageType  */
    if (r >= 1 && !(buf[0] == 0xB0)) {
        MV_HAL_LOG_ERROR() << "Not a normal FW binary with checksum";
        return -6;
    }

    // Read the complete firmware binary into a local buffer.
    lseek(fd, 0, SEEK_SET);
    r = read(fd, buf, filesize);

    close(fd);
    return filesize;
}

int ram_write(libusb_device_handle *h, unsigned char *buf, unsigned int ramAddress, int len) {
    int r;
    int index = 0;
    int size;

    while (len > 0) {
        size = (len > MAX_WRITE_SIZE) ? MAX_WRITE_SIZE : len;
        r    = libusb_control_transfer(h, 0x40, 0xA0, GET_LSW(ramAddress), GET_MSW(ramAddress), &buf[index], size,
                                    VENDORCMD_TIMEOUT);
        if (r != size) {
            printf("Vendor write to FX3 RAM failed\n");
            return -1;
        }

        ramAddress += size;
        index += size;
        len -= size;
    }

    return 0;
}

int fx3_usbboot_download(libusb_device_handle *h, const char *filename) {
    unsigned char *fwBuf;
    unsigned int *data_p;
    unsigned int i, checksum;
    unsigned int address, length;
    int r, index;

    fwBuf = (unsigned char *)calloc(1, MAX_FWIMG_SIZE);
    if (fwBuf == 0) {
        printf("Failed to allocate buffer to store firmware binary\n");
        //      sb->showMessage("Error: Failed to get memory for download\n", 5000);
        return -1;
    }

    // Read the firmware image into the local RAM buffer.
    int filesize = read_firmware_image(filename, fwBuf, NULL);
    if (filesize <= 0) {
        printf("Failed to read firmware file %s\n", filename);
        //      sb->showMessage("Error: Failed to read firmware binary\n", 5000);
        free(fwBuf);
        return -2;
    }

    // Run through each section of code, and use vendor commands to download them to RAM.
    index    = 4;
    checksum = 0;
    while (index < filesize) {
        data_p  = (unsigned int *)(fwBuf + index);
        length  = data_p[0];
        address = data_p[1];
        if (length != 0) {
            for (i = 0; i < length; i++)
                checksum += data_p[2 + i];
            r = ram_write(h, fwBuf + index + 8, address, length * 4);
            if (r != 0) {
                printf("Failed to download data to FX3 RAM\n");
                free(fwBuf);
                return -3;
            }
        } else {
            if (checksum != data_p[2]) {
                printf("Checksum error in firmware binary\n");
                free(fwBuf);
                return -4;
            }

            r = libusb_control_transfer(h, 0x40, 0xA0, GET_LSW(address), GET_MSW(address), NULL, 0, VENDORCMD_TIMEOUT);
            if (r != 0)
                printf("Ignored error in control transfer: %d\n", r);
            break;
        }

        index += (8 + length * 4);
    }

    free(fwBuf);
    return 0;
}
} // namespace LoadApplicativeFirmwareToFx3RAM

FlashCmd::FlashCmd() {}

FlashCmd FlashCmd::FlashCmdFx3() {
    FlashCmd ret;
    return ret;
}

FlashCmd FlashCmd::FlashCmdFpga() {
    FlashCmd ret;
    ret.Write  = 0xC5;
    ret.Read   = 0xC6;
    ret.Erase  = 0xC7;
    ret.Status = 0xC7;

    ret.step      = 4096;
    ret.erasestep = 65536;

    ret.default_test_sector = 600;
    return ret;
}

int FlashCmd::flash_test(libusb_device_handle *dev_handle, int *err_bad_flash, bool erase, bool write, bool read) {
    std::vector<uint8_t> vdata;
    for (long i = 0; i != step; ++i) {
        vdata.push_back(i);
    }

    MV_HAL_LOG_TRACE() << "Size to flash" << vdata.size();
    long sectore_erased  = -1;
    long num_err         = 0;
    unsigned long offset = default_test_sector * step;

    int sector          = offset / step;
    int sector_to_erase = offset / erasestep;

    MV_HAL_LOG_TRACE() << "Sector" << sector;

    if (sector_to_erase != sectore_erased) {
        if (erase) {
            erase_sector(dev_handle, sector_to_erase, num_err);
            sectore_erased = sector_to_erase;
        }
    }
    if (write) {
        write_sector_over_erased_offset(dev_handle, sector, vdata, 0, num_err);
    }
    if (read || write) {
        std::vector<uint8_t> vorig = vdata;
        std::vector<uint8_t> vread;
        if (read_sector(dev_handle, sector, vread, num_err)) {
            if (read) {
                dump_data(vread);
            }
        }
        if (write) {
            if (vorig != vread) {
                MV_HAL_LOG_WARNING() << "Bad flash";
                ++(*err_bad_flash);
                ++num_err;
            }
        }
    }
    return 0;
}

int FlashCmd::flash(libusb_device_handle *dev_handle, const char *filename, unsigned long start_sector, long max_sector,
                    long file_offset, int *err_bad_flash) {
    std::ifstream infile(filename, std::ios::in | std::ifstream::binary);
    if (file_offset > 0) {
        infile.ignore(file_offset);
    }
    std::vector<uint8_t> vdata((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());

    if (vdata.empty()) {
        MV_HAL_LOG_ERROR() << "Unable to read" << filename;
        return -1;
    }

    while (vdata.size() % step) {
        vdata.push_back(0);
    }

    MV_HAL_LOG_INFO() << "Size to flash" << vdata.size();
    return flash_offset_sector(dev_handle, vdata, start_sector, max_sector, err_bad_flash);
}

int FlashCmd::dump(libusb_device_handle *dev_handle, const char *filename, int *err_bad_flash) {
    std::ofstream outfile(filename, std::ios::out | std::ifstream::binary);
    std::vector<uint8_t> vdata; //( (std::istreambuf_iterator<char>(infile)),  std::istreambuf_iterator<char>() );

    int ret = dump_flash(dev_handle, vdata, err_bad_flash);
    std::copy(vdata.begin(), vdata.end(), std::ostreambuf_iterator<char>(outfile));
    return ret;
}

int FlashCmd::flash_map(libusb_device_handle *dev_handle, long start_sector, const std::string &findexes,
                        const std::string &ftargets, int *err_bad_flash) {
    std::vector<uint8_t> vdata;
    long nindex = 0;
    coe_2_data(findexes, vdata, nindex);

    MV_HAL_LOG_TRACE() << nindex << "indexes read";

    long ntargets = 0;
    coe_2_data(ftargets, vdata, ntargets);

    MV_HAL_LOG_TRACE() << ntargets << "targets read";
    while (vdata.size() % step) {
        vdata.push_back(0);
    }

    MV_HAL_LOG_TRACE() << "Size to flash" << vdata.size();
    return flash_offset_sector(dev_handle, vdata, start_sector, -1, err_bad_flash);
}

int FlashCmd::flash_calib3d(libusb_device_handle *dev_handle, long start_sector, const std::string &calib3d,
                            int *err_bad_flash) {
    std::ifstream infile(calib3d, std::ios::in | std::ifstream::binary);
    std::vector<uint8_t> vdata((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    while (vdata.size() % step) {
        vdata.push_back(0);
    }

    MV_HAL_LOG_TRACE() << "Size to flash" << vdata.size();
    return flash_offset_sector(dev_handle, vdata, start_sector, -1, err_bad_flash);

    return 1;
}

int FlashCmd::dump_calib3d(libusb_device_handle *dev_handle, std::vector<uint8_t> &vdata, long start_sector,
                           int *err_bad_flash) {
    long num_err = 0;
    for (int sector = start_sector; sector < start_sector + 1000; ++sector) {
        std::vector<uint8_t> vread; //(step,0);
        read_sector(dev_handle, sector, vread, num_err);
        vdata.insert(vdata.end(), vread.begin(), vread.end());
        auto it = vread.rbegin();
        if (it == vread.rend())
            break;
        uint8_t vlast = *it;
        ++it;
        if (it == vread.rend())
            break;
        uint8_t vlast_1 = *it;
        if ((vlast == 0 && vlast_1 == 0) || (vlast == 0xFF && vlast_1 == 0xFF)) {
            break;
        }

        if (num_err > 10)
            break;
    }
    return 0;
}

int FlashCmd::flash_offset_sector(libusb_device_handle *dev_handle, std::vector<uint8_t> &vdata,
                                  unsigned long start_sector, long max_sector, int *err_bad_flash) {
    long offset = start_sector * step;
    if (offset % erasestep) {
        MV_HAL_LOG_ERROR() << "The start sector must be at the beginning of an erase sector";
        return 0;
    }
    long sectore_erased = -1;
    long num_err        = 0;
    long last_index     = vdata.size() + offset - 1;
    if (max_sector >= 0) {
        long max_index = ((max_sector + 1) * step) - 1;
        last_index     = last_index >= max_index ? max_index : last_index;
    }
    for (long index = offset; index <= last_index; index += step) {
        int sector          = index / step;
        int sector_to_erase = index / erasestep;

        if (sector_to_erase != sectore_erased) {
            erase_sector(dev_handle, sector_to_erase, num_err);
            sectore_erased = sector_to_erase;
        }
        write_sector_over_erased_offset(dev_handle, sector, vdata, index - offset, num_err);

        std::vector<uint8_t> vorig(&vdata[index - offset], &vdata[index - offset] + step);
        std::vector<uint8_t> vread; //(step,0);
        read_sector(dev_handle, sector, vread, num_err);
        if (vorig != vread) {
            MV_HAL_LOG_WARNING() << "Bad flash";
            ++(*err_bad_flash);
            ++num_err;
        }
    }
    return 0;
}

int FlashCmd::dump_flash(libusb_device_handle *dev_handle, std::vector<uint8_t> &vdata, int *err_bad_flash) {
    return dump_flash_offset_sector(dev_handle, vdata, 0, 1023, err_bad_flash);
}

int FlashCmd::dump_flash_offset_sector(libusb_device_handle *dev_handle, std::vector<uint8_t> &vdata, long start_sector,
                                       long end_sector, int *err_bad_flash) {
    vdata.clear();
    long num_err = 0;
    for (int sector = start_sector; sector <= end_sector; ++sector) {
        std::vector<uint8_t> vread; //(step,0);
        read_sector(dev_handle, sector, vread, num_err);
        vdata.insert(vdata.end(), vread.begin(), vread.end());

        if (num_err > 10)
            break;
    }
    return 0;
}

int FlashCmd::flash_serial(libusb_device_handle *dev_handle, int *err_bad_flash, long sector,
                           uint16_t serial_to_write) {
    std::vector<uint8_t> vserial_to_write;
    vserial_to_write.push_back((serial_to_write >> 8) & 0xFF);
    vserial_to_write.push_back(serial_to_write & 0xFF);

    return flash_serial(dev_handle, err_bad_flash, sector, vserial_to_write);
}

int FlashCmd::flash_serial(libusb_device_handle *dev_handle, int *err_bad_flash, long sector,
                           std::vector<uint8_t> serial_to_write) {
    std::vector<uint8_t> vdata;

    using SizeType = std::vector<uint8_t>::size_type;

    long sectore_erased = -1;
    long num_err        = 0;
    long sectorstart    = (((sector * step) / erasestep) * erasestep) / step;
    for (long index = 0; index < erasestep / step; index += 1) {
        std::vector<uint8_t> vread; //(step,0);
        read_sector(dev_handle, index + sectorstart, vread, num_err);
        vdata.insert(vdata.end(), vread.begin(), vread.end());
    }

    // TODO:
    // we should think about how to store both serial + length(serial) on memory, to be more generic
    for (SizeType i = 0; i < serial_to_write.size(); i++) {
        vdata[(sector - sectorstart) * step + i] = serial_to_write[i];
    }

    long offset = sectorstart * step;
    for (SizeType index = offset; index < vdata.size() + offset; index += step) {
        int sector          = index / step;
        int sector_to_erase = index / erasestep;

        int r = 0;
        uint8_t status;
        if (sector_to_erase != sectore_erased) {
            erase_sector(dev_handle, sector_to_erase, num_err);

            sectore_erased = sector_to_erase;
        }

        r = libusb_control_transfer(dev_handle, 0x40 /*(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_OUT)*/, Write, 0,
                                    sector, &vdata[index - offset], step, 0);
        if (r <= 0) {
            MV_HAL_LOG_WARNING() << "Error writing :" << libusb_error_name(r);
            ++num_err;
        }

        do {
            int r = libusb_control_transfer(dev_handle, (LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_IN), Status, 0, 0,
                                            &status, 1, 0);
            if (r <= 0) {
                MV_HAL_LOG_WARNING() << "Error reading status :" << libusb_error_name(r);
                ++num_err;
            }
        } while (status != 0);

        std::vector<uint8_t> vorig(&vdata[index - offset], &vdata[index - offset] + step);
        std::vector<uint8_t> vread(step, 0);
        if (read_sector(dev_handle, sector, vread, num_err)) {}

        if (vorig != vread) {
            MV_HAL_LOG_WARNING() << "Bad flash";
            ++(*err_bad_flash);
            ++num_err;
        }
    }
    return 0;
}

int FlashCmd::flash_system_id(libusb_device_handle *dev_handle, int *err_bad_flash, long sector,
                              long system_id_to_write) {
    return flash_serial(dev_handle, err_bad_flash, sector, system_id_to_write);
}

bool FlashCmd::wait_for_status(libusb_device_handle *dev_handle) {
    uint8_t status;

    do {
        int r = libusb_control_transfer(dev_handle, (LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_IN), Status, 0, 0,
                                        &status, 1, 0);
        if (r <= 0) {
            MV_HAL_LOG_ERROR() << "Error reading status :" << libusb_error_name(r);
            return false;
        }
    } while (status != 0);
    return true;
}

void FlashCmd::coe_2_data(const std::string &ftargets, std::vector<uint8_t> &vdata, long &ntargets) {
    std::string stmp;
    std::ifstream infile_target(ftargets);
    if (infile_target)
        std::getline(infile_target, stmp);
    if (infile_target)
        std::getline(infile_target, stmp);
    ntargets = 0;
    while (infile_target) {
#ifdef __ANDROID__
        long i;
        infile_target >> i;
        ++ntargets;
        vdata.push_back(i & 0xFF);
        vdata.push_back((i >> 8) & 0xFF);
        vdata.push_back((i >> 16) & 0xFF);
        vdata.push_back((i >> 24) & 0xFF);
#else
        std::getline(infile_target, stmp);
        try {
            long i = std::stol(stmp);
            ++ntargets;
            vdata.push_back(i & 0xFF);
            vdata.push_back((i >> 8) & 0xFF);
            vdata.push_back((i >> 16) & 0xFF);
            vdata.push_back((i >> 24) & 0xFF);
        } catch (...) {}
#endif
    }
}

void FlashCmd::erase_sector(libusb_device_handle *dev_handle, int sector_to_erase, long &num_err) {
    MV_HAL_LOG_TRACE() << "Erase sector" << sector_to_erase;
    int r = libusb_control_transfer(dev_handle, 0x40 /*(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_OUT)*/, Erase, 1,
                                    sector_to_erase, nullptr, 0, 0);
    if (r < 0) {
        MV_HAL_LOG_WARNING() << "Error erase :" << libusb_error_name(r);
        ++num_err;
    }
    if (!wait_for_status(dev_handle)) {
        ++num_err;
    }
}

bool FlashCmd::read_sector(libusb_device_handle *dev_handle, int sector, std::vector<uint8_t> &vread, long &num_err) {
    MV_HAL_LOG_TRACE() << "Read sector" << sector;
    vread.resize(step, 0);
    std::fill(vread.begin(), vread.end(), 0);
    int r = libusb_control_transfer(dev_handle, (LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_IN), Read, 0, sector,
                                    &vread[0], step, 0);
    if (r <= 0) {
        MV_HAL_LOG_ERROR() << "Error read :" << libusb_error_name(r);
        ++num_err;
        return false;
    }
    return true;
}

bool FlashCmd::write_sector_over_erased_offset(libusb_device_handle *dev_handle, int sector,
                                               std::vector<uint8_t> &vdata, unsigned long offset, long &num_err) {
    if (vdata.size() < offset + step) {
        MV_HAL_LOG_ERROR() << "Error write : not enough datas to fill a sector";
        ++num_err;

        return 0;
    }
    MV_HAL_LOG_INFO() << "Write sector" << sector;
    int r = libusb_control_transfer(dev_handle, 0x40 /*(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_OUT)*/, Write, 0,
                                    sector, &vdata[offset], step, 0);

    if (r <= 0) {
        MV_HAL_LOG_ERROR() << "Error write :" << libusb_error_name(r);
        ++num_err;
        return 0;
    }

    if (!wait_for_status(dev_handle)) {
        ++num_err;
    }
    return 1;
}

bool FlashCmd::write_sector_over_erased(libusb_device_handle *dev_handle, int sector, std::vector<uint8_t> &vdata,
                                        long &num_err) {
    return write_sector_over_erased_offset(dev_handle, sector, vdata, 0, num_err);
}

void FlashCmd::dump_data(const std::vector<uint8_t> &vdata) {
    auto log_op = MV_HAL_LOG_TRACE() << Metavision::Log::no_space;
    int max_i   = std::min(1000, static_cast<int>(vdata.size()));
    for (int i = 0; i < max_i; ++i) {
        if (i % 16 == 0)
            log_op << i << " : ";
        log_op << std::hex << long(vdata[i]) << " " << std::dec;
        if (i % 4)
            log_op << " ";
        if (i % 16 == 15)
            log_op << std::endl;
    }
}
