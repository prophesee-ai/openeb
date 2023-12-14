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

// Application to flash system firmware and FPGA bitstream.

#include <iostream>
#ifndef _MSC_VER
#include <unistd.h>
#else
#include <io.h>
#endif
#include <fstream>
#include <vector>
#include <unordered_map>

#include <libusb.h>
#include <boost/program_options.hpp>
#include <boost/algorithm/hex.hpp>

#include "metavision/sdk/base/utils/log.h"
#include "boards/utils/utils_fx3_ram_flash.h"
#include "boards/utils/vendor_command_definition.h"
#include "devices/utils/device_system_id.h"
#include "devices/others/i2c_eeprom.h"

using namespace std;

namespace {

int fx3_fpga_flash(libusb_device_handle *dev_handle, const char *filename, unsigned long start_sector,
                   unsigned long max_sector, long file_offset, int *err_bad_flash) {
    FlashCmd cmd = FlashCmd::FlashCmdFpga();
    return cmd.flash(dev_handle, filename, start_sector, max_sector, file_offset, err_bad_flash);
}

int fx3_flash(libusb_device_handle *dev_handle, const char *filename, int *err_bad_flash) {
    FlashCmd cmd = FlashCmd::FlashCmdFx3();
    return cmd.flash(dev_handle, filename, 0, -1, 0, err_bad_flash);
}

long convert_usb_speed(long dev_speed) {
    switch (dev_speed) {
    case LIBUSB_SPEED_LOW:
        return 1; // Actual speed would be 1.5Mbit/s but we use integral type.
    case LIBUSB_SPEED_FULL:
        return 12;
    case LIBUSB_SPEED_HIGH:
        return 480;
    case LIBUSB_SPEED_SUPER:
        return 5000;
#if LIBUSB_API_VERSION >= 0x01000106
    // Compiling on 1.0.22 or newer, which starts support for SSP.
    case LIBUSB_SPEED_SUPER_PLUS:
        return 10000;
#endif
    case LIBUSB_SPEED_UNKNOWN:
    default:
        return 0; // Unknown speed is indicated as 0.
    }
}

} // namespace
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    bool fw          = false;
    bool fpga        = false;
    bool recov       = false;
    bool serial_read = false;
    vector<uint8_t> serial;
    std::string serial_string = "";
    std::string ccam5_board   = "";
    std::vector<std::string> ccam5_rev_list{"revA", "revB"};
    std::map<std::string, int> eeprom_addr_list{{"revA", 0x50}, {"revB", 0x5E}};
    int eeprom_dev_addr = 0;
    std::string firmware;

    const std::string program_desc("Flash a CX3 device connected to the machine with the given firmware or bitstream.\n"
                                   "You can only pass one option --usb-fpga or --usb-fw at a time.\n");

    po::options_description options_desc("Options");
    po::options_description hidden_desc("Hidden options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("usb-fpga", po::bool_switch(&fpga)->default_value(false),"Flash fpga.")
        ("usb-fw",   po::bool_switch(&fw)->default_value(false), "Flash firmware.")
        ("usb-recov", po::bool_switch(&recov)->default_value(false), "Load firmware to RAM.")
        ("usb-read-serial",   po::bool_switch(&serial_read)->default_value(false), "Read serial number from firmware.");

    hidden_desc.add_options()
        ("usb-write-serial", po::value<std::string>(&serial_string)->default_value(""), "Serial to write. in hexadecimal, up to 64bits.")
        ("ccam5-revision", po::value<std::string>(&ccam5_board)->default_value("revB"), "CCam5 board revision.")
        ("firmware,f", po::value<std::string>(), "Firmware file.");
    // clang-format on

    po::positional_options_description positional_options;
    positional_options.add("firmware", 1);

    po::options_description desc_composite;
    desc_composite.add(options_desc).add(hidden_desc);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc_composite).positional(positional_options).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help") || argc == 1) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    if (vm.count("firmware")) {
        firmware = vm["firmware"].as<std::string>();
    }

    bool invalid_args = false;

    if ((long(fw) + long(fpga) > 1)) {
        MV_LOG_ERROR() << "Invalid programming arguments : only one of --usb-fpga or --usb-fw parameter can be used at "
                          "the same time.";
        invalid_args = true;
    }

    if (vm.count("usb-write-serial") || vm.count("usb-read-serial")) {
        auto it = std::find(ccam5_rev_list.begin(), ccam5_rev_list.end(), ccam5_board);
        if (it == ccam5_rev_list.end()) {
            // Selected revision does not exist, printing allowed values
            std::string valid_revs = "";
            for (const std::string &board : ccam5_rev_list) {
                valid_revs.append(board);
                valid_revs.append(", ");
            }
            valid_revs.erase(valid_revs.length() - 2);
            MV_LOG_ERROR() << "Invalid board revision for --ccam5-revision.\nExpected values are: " << valid_revs;
            invalid_args = true;
        } else {
            eeprom_dev_addr = eeprom_addr_list[ccam5_board];
        }
    }

    if (invalid_args) {
        return 1;
    }

    libusb_device **devs;             // pointer to pointer of device, used to retrieve a list of devices
    libusb_device_handle *dev_handle; // a device handle
    std::string dev_name;
    std::vector<std::pair<std::string, libusb_device_handle *>> valid_devs;

    //  listing of usb devices for cx3, (name, vid, pid)
    const std::unordered_map<std::string, std::pair<uint16_t, uint16_t>> USB_DEVICES = {
        {"CYBOOT", std::make_pair(0x04b4, 0x00f3)},
        {"CYFLASH", std::make_pair(0x04b4, 0x4720)},
        {"EVK3", std::make_pair(0x04b4, 0x00f4)},
        {"EVK4", std::make_pair(0x04b4, 0x00f5)}};
    libusb_context *ctx = NULL; // a libusb session
    int r;                      // for return values
    ssize_t cnt;                // holding number of devices in list
    r = libusb_init(&ctx);      // initialize the library for the session we just declared
    if (r < 0) {
        MV_LOG_ERROR() << "Init Error" << r; // there was an error
        return 1;
    }

    cnt = libusb_get_device_list(ctx, &devs); // get the list of devices
    if (cnt < 0) {
        MV_LOG_ERROR() << "Get Device Error"; // there was an error
        return 1;
    }
    MV_LOG_INFO() << "Detected" << cnt << "USB devices";

    // Look for valid Psee usb cx3 devices
    for (auto &it : USB_DEVICES) {
        auto name = it.first;
        auto vid  = it.second.first;
        auto pid  = it.second.second;
        auto res  = libusb_open_device_with_vid_pid(ctx, vid, pid);
        if (res != NULL) {
            valid_devs.push_back(std::make_pair(name, res));
        }
    }

    if (valid_devs.empty()) {
        MV_LOG_ERROR() << "No device to flash";
        return 1;
    } else if (valid_devs.size() != 1) {
        MV_LOG_ERROR() << "Found more than one candidate. Ensure only one is connected";

        // Exiting, close what we just opened
        for (auto it = valid_devs.begin(); it != valid_devs.end(); ++it) {
            libusb_close(it->second);
        }

        return 1;
    } else {
        dev_name   = valid_devs[0].first;
        dev_handle = valid_devs[0].second;
        MV_LOG_INFO() << "Found" << dev_name << "device";
    }

    libusb_device *dev_;
    dev_                = libusb_get_device(dev_handle);
    long dev_speed_     = (libusb_speed)libusb_get_device_speed(dev_);
    long dev_data_rate_ = convert_usb_speed(dev_speed_);

    if (dev_speed_ < LIBUSB_SPEED_SUPER && dev_name != "CYBOOT") {
        MV_LOG_ERROR() << dev_name
                       << "device is not enumerated as a USB 3 SuperSpeed device. Found device speed:" << dev_data_rate_
                       << "Mbit/s";
        return 1;
    } else {
        MV_LOG_INFO() << dev_name << "device speed is" << dev_data_rate_ << "Mbit/s";
    }

    uint16_t sys_id   = 0;
    int err_bad_flash = 0;
    int ret_flash     = 0;
    std::vector<uint8_t> vread;
    vread.resize(2);

    if (serial_string != "" || serial_read) {
        r = libusb_control_transfer(dev_handle, 0xC0, CMD_READ_SYSTEM_ID, 0x00, 0, &vread[0], 2, 0);
        if (r <= 0) {
            MV_LOG_WARNING() << "Cannot fetch system ID" << libusb_error_name(r);
        } else {
            sys_id = ((vread[1] & 0xFF) << 8) | (vread[0] & 0xFF);
            MV_LOG_INFO() << Metavision::Log::no_space << "Found System ID 0x" << std::hex << (sys_id & 0xFFFF);
        }
    }

    if (fpga && (firmware != "")) {
        ret_flash = fx3_fpga_flash(dev_handle, firmware.c_str(), 0, -1, 0, &err_bad_flash);

        if (ret_flash) {
            MV_LOG_ERROR() << "Error while flashing";
        } else {
            MV_LOG_INFO() << "Done";
        }

        if (err_bad_flash > 0) {
            MV_LOG_ERROR() << "Flashed" << err_bad_flash << "error(s)";
        }
        MV_LOG_INFO() << "Please, unplug the camera to reset it";
    } else if (fw && firmware != "") {
        ret_flash = fx3_flash(dev_handle, firmware.c_str(), &err_bad_flash);

        if (ret_flash) {
            MV_LOG_ERROR() << "Error while flashing";
        } else {
            MV_LOG_INFO() << "Done";
        }

        if (err_bad_flash > 0) {
            MV_LOG_ERROR() << "Flashed" << err_bad_flash << "error(s)";
        }
        MV_LOG_INFO() << "Please, unplug the camera to reset it";
    } else if (recov && firmware != "") {
        ret_flash = LoadApplicativeFirmwareToFx3RAM::fx3_usbboot_download(dev_handle, firmware.c_str());
        if (ret_flash) {
            MV_LOG_ERROR() << "Could not load firmware to RAM. Error:" << ret_flash;
        } else {
            MV_LOG_INFO() << "Firmware loaded to ram";
        }
    } else if (serial_string != "") {
        serial.clear();

        // clean string if Hexa value contains 0x
        if (serial_string.rfind("0x", 0) == 0) {
            serial_string = serial_string.erase(0, 2);
        }
        // test if length of serial is too long > 64 bits
        // here we count in char
        if (serial_string.length() > 16) {
            MV_LOG_ERROR() << "Error while flashing Serial : Serial must be 64Bits max";
            return 1;
        }
        if (serial_string.length() % 2 != 0 || serial_string.length() < 16) {
            int zeros_to_prepend = 16 - serial_string.length();
            for (int i = 0; i < zeros_to_prepend; i++)
                serial_string.insert(0, 1, '0');
        }
        try {
            boost::algorithm::unhex(serial_string.begin(), serial_string.end(), std::back_inserter(serial));
        } catch (const std::exception &) {
            MV_LOG_ERROR() << "Error while flashing Serial: serial must only contains Hexadecimal characters";
            return 1;
        }
        if (serial.size() > 8) {
            MV_LOG_ERROR() << "Error while flashing Serial : Serial must be 64Bits max";
            return 1;
        }
        MV_LOG_INFO() << "Writing serial" << serial_string;

        if (sys_id == Metavision::SystemId::SYSTEM_EVK3_GEN41 || sys_id == Metavision::SystemId::SYSTEM_EVK3_IMX636) {
            MV_LOG_INFO() << Metavision::Log::no_space << "Using EEPROM address 0x" << std::hex
                          << (eeprom_dev_addr & 0xFF) << " (CCam5 " << ccam5_board << ")";
            I2cEeprom eeprom_handler(eeprom_dev_addr);
            std::reverse(serial.begin(), serial.end());
            ret_flash = eeprom_handler.write(dev_handle, 0x0, serial);

            if (ret_flash == 0) {
                MV_LOG_INFO() << "Done";
            } else {
                MV_LOG_ERROR() << "Error while accesing EEPROM to write serial number.";
                MV_LOG_ERROR() << "Serial number write operation failed.";
            }
        } else if (sys_id == Metavision::SystemId::SYSTEM_CCAM5_GEN31) {
            FlashCmd cmd = FlashCmd::FlashCmdFpga();
            ret_flash    = cmd.flash_serial(dev_handle, &err_bad_flash, 600, serial);
        } else {
            MV_LOG_ERROR() << "Error while flashing Serial : Unknown system";
            return 1;
        }
    } else if (serial_read) {
        if (sys_id == Metavision::SystemId::SYSTEM_EVK3_GEN41 || sys_id == Metavision::SystemId::SYSTEM_EVK3_IMX636) {
            MV_LOG_INFO() << Metavision::Log::no_space << "Using EEPROM address 0x" << std::hex
                          << (eeprom_dev_addr & 0xFF) << " (CCam5 " << ccam5_board << ")";
            I2cEeprom eeprom_handler(eeprom_dev_addr);
            vread.resize(8);
            vread.clear();
            ret_flash = eeprom_handler.read(dev_handle, 0, vread, 8);

            if (ret_flash == 0) {
                uint64_t read_val = 0;
                std::reverse(vread.begin(), vread.end());

                for (int i = 0; i < 8; i++) {
                    read_val = (read_val << 8) + (vread[i] & 0xFF);
                }

                MV_LOG_INFO() << Metavision::Log::no_space << "Serial number read is: 0x" << std::hex << std::setw(16)
                              << std::setfill('0') << read_val;
            } else {
                MV_LOG_ERROR() << "Error while accesing EEPROM to read serial number.";
                MV_LOG_ERROR() << "Serial number read operation failed.";
            }
        }
    }

    // We shouldn't get here with more than one handle, but just in case,
    // close the device we opened
    for (auto it = valid_devs.begin(); it != valid_devs.end(); ++it) {
        libusb_close(it->second);
    }
    libusb_exit(ctx); // needs to be called to end the

    return 0;
}
