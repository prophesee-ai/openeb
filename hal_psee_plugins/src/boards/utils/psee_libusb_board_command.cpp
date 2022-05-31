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

#include "metavision/hal/utils/hal_log.h"
#include "boards/utils/psee_libusb_board_command.h"
#include "boards/utils/utils_fx3_ram_flash.h"
#include "boards/utils/vendor_command_definition.h"
#include "boards/utils/config_registers_map.h"

namespace Metavision {

std::mutex PseeLibUSBBoardCommand::protect_libusb_submit_;

PseeLibUSBBoardCommand::PseeLibUSBBoardCommand() = default;

PseeLibUSBBoardCommand::PseeLibUSBBoardCommand(libusb_device_handle *dev_handle) {
    dev_handle_ = dev_handle;
    dev_speed_  = LIBUSB_SPEED_UNKNOWN;
}

PseeLibUSBBoardCommand::~PseeLibUSBBoardCommand() {
    if (dev_handle_) {
        int r = libusb_release_interface(dev_handle_, 0); // release the claimed interface
        if (r != 0) {
            MV_HAL_LOG_WARNING() << "Cannot release interface";
        } else {
            MV_HAL_LOG_TRACE() << "Released interface";
        }
        libusb_close(dev_handle_); // close the device we opened
    }
}

bool PseeLibUSBBoardCommand::open(const std::string &serial) {
    if (!init_libusb()) {
        return false;
    }
    get_ccam2_with_serial(nullptr, serial);
    return dev_handle_ != nullptr;
}

long PseeLibUSBBoardCommand::get_board_version() {
    uint16_t val = control_read_register_16bits(CMD_READ_VERSION_FX3, 0x00);
    return val;
}

long PseeLibUSBBoardCommand::get_board_id() {
    uint32_t val = control_read_register_32bits(CMD_READ_FX3_ID, 0x00, false);
    return val;
}

long PseeLibUSBBoardCommand::get_board_release_version() {
    uint32_t val = control_read_register_32bits(CMD_READ_FX3_RELEASE_VERSION, 0x00, false);
    return val;
}

long PseeLibUSBBoardCommand::get_board_build_date() {
    uint32_t val = control_read_register_32bits(CMD_READ_FX3_BUILD_DATE, 0x00, false);
    return val;
}

long PseeLibUSBBoardCommand::get_board_speed() {
    if (!dev_handle_) {
        return -1;
    }

    switch (dev_speed_) {
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

long PseeLibUSBBoardCommand::get_board_version_control_id() {
    uint32_t val = control_read_register_32bits(CMD_READ_FX3_VERSION_CONTROL_ID, 0x00, false);
    return val;
}

uint32_t PseeLibUSBBoardCommand::control_read_register_32bits(uint8_t usbvendorcmd, uint32_t address, bool big_endian) {
    uint32_t val = -1;
    if (!dev_handle_) {
        MV_HAL_LOG_ERROR() << "ERR no dev_handle";
        return val;
    }
    unsigned char data[8];
    int r = libusb_control_transfer(dev_handle_,
                                    0xC0
                                    //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_IN)
                                    ,
                                    usbvendorcmd, uint16_t(address & 0xFFFF), uint16_t((address >> 16) & 0xFFFF), data,
                                    8, 0);
    if (r <= 0) {
        MV_HAL_LOG_ERROR() << "control_read_register_32bits" << r << "err" << libusb_error_name(r);
    }

    auto log_op = MV_HAL_LOG_DEBUG() << "control_read_32bits " << r << " ";
    if (r <= 0)
        log_op << "err" << libusb_error_name(r);
    else if (data[0] != 0x40) {
        log_op << std::hex << long(usbvendorcmd) << long(address) << long(data[0]) << long(data[1]) << long(data[2])
               << long(data[3]) << long(data[4]) << long(data[5]) << long(data[6]) << long(data[7]) << big_endian;
        log_op << "transaction failed - returned error code is:" << Metavision::Log::no_space << long(data[0])
               << long(data[1]) << std::endl;
    } else {
        log_op << std::hex << long(usbvendorcmd) << long(address) << long(data[0]) << long(data[1]) << long(data[2])
               << long(data[3]) << long(data[4]) << long(data[5]) << long(data[6]) << long(data[7]) << big_endian;
        log_op << std::endl;
    }

    if (big_endian) {
        val = data[7];
        val |= data[6] << 8;
        val |= data[5] << 16;
        val |= data[4] << 24;
    } else {
        val = data[0];
        val |= data[1] << 8;
        val |= data[2] << 16;
        val |= data[3] << 24;
    }
    log_op << "value" << std::hex << val << std::dec;
    return val;
}

uint16_t PseeLibUSBBoardCommand::control_read_register_16bits(uint8_t usbvendorcmd, uint32_t address) {
    uint16_t val = -1;
    if (!dev_handle_) {
        MV_HAL_LOG_ERROR() << "ERR no dev_handle";
        return val;
    }
    unsigned char data[4];
    int r = libusb_control_transfer(dev_handle_,
                                    0xC0
                                    //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_IN)
                                    ,
                                    usbvendorcmd, address, 0, data, 4, 0);
    if (r <= 0) {
        MV_HAL_LOG_ERROR() << "control_read_register_16bits" << r << "err" << libusb_error_name(r);
    }

    auto log_op = MV_HAL_LOG_DEBUG() << "control_read_register_16bits" << r;
    if (r <= 0)
        log_op << "err" << libusb_error_name(r);
    else
        log_op << std::hex << long(usbvendorcmd) << long(address) << long(data[0]) << long(data[1]) << long(data[2])
               << long(data[3]);
    log_op << std::endl;
    val = data[2];
    val |= static_cast<uint16_t>(data[3]) << 8;

    log_op << "value " << std::hex << val << std::dec;
    return val;
}

void PseeLibUSBBoardCommand::control_write_register_32bits(uint8_t usbvendorcmd, uint32_t address, uint32_t val) {
    if (!dev_handle_) {
        MV_HAL_LOG_ERROR() << "ERR no dev_handle";
        return;
    }
    unsigned char data[8];

    data[0] = (val >> 24) & 0xFF;
    data[1] = (val >> 16) & 0xFF;
    data[2] = (val >> 8) & 0xFF;
    data[3] = (val)&0xFF;
    data[4] = 1;
    data[5] = 2;
    data[6] = 3;
    data[7] = 4;

    int r = libusb_control_transfer(dev_handle_,
                                    0x40
                                    //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_OUT)
                                    ,
                                    usbvendorcmd, uint16_t(address & 0xFFFF), uint16_t((address >> 16) & 0xFFFF), data,
                                    4, 0);
    if (r <= 0) {
        MV_HAL_LOG_ERROR() << "control_write_register_32bits" << r << "err" << libusb_error_name(r);
    }
    auto log_op = MV_HAL_LOG_DEBUG() << "control_write_register_32bits " << r;
    if (r <= 0)
        log_op << std::hex << long(usbvendorcmd) << long(address) << val << std::dec
               << "transaction failed - returned error code is :" << long(data[0]) << long(data[1]) << long(data[2])
               << long(data[3]) << long(data[4]) << long(data[5]) << long(data[6]) << long(data[7]);
    else
        log_op << std::hex << long(usbvendorcmd) << long(address) << val << std::dec << "status" << r;
    if (r <= 0)
        log_op << "err" << libusb_error_name(r);
    log_op << std::endl;
}

void PseeLibUSBBoardCommand::control_write_vector_32bits(uint32_t base_address, const std::vector<uint32_t> &val) {
    if (!dev_handle_) {
        MV_HAL_LOG_ERROR() << "ERR no dev_handle";
        return;
    }
    std::vector<unsigned char> data;
    for (uint32_t v : val) {
        data.push_back((v >> 24) & 0xFF);
        data.push_back((v >> 16) & 0xFF);
        data.push_back((v >> 8) & 0xFF);
        data.push_back((v)&0xFF);
    }
    int r = libusb_control_transfer(
        dev_handle_,
        0x40
        //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_OUT)
        ,
        ((base_address == CCAM2IF_LEFT_BASE_ADDRESS) ? CMD_WRITE_VEC_REGFPGA_32 : CMD_WRITE_VEC_SLAVE_REGFPGA_32), 0, 0,
        data.data(), data.size(), 0);

    if (r <= 0) {
        MV_HAL_LOG_ERROR() << "control_write_vector_32bits" << r << "err" << libusb_error_name(r);
    }

    auto log_op = MV_HAL_LOG_DEBUG() << "control_write_vector_32bits" << r;
    MV_HAL_LOG_DEBUG() << std::hex << "size" << data.size() << std::dec;
    for (auto i : val) {
        log_op << std::hex << long(i) << ",";
    }
    log_op << std::endl;
    if (r <= 0)
        log_op << "err" << libusb_error_name(r);
}

void PseeLibUSBBoardCommand::control_write_register_8bits(uint8_t usbvendorcmd, uint8_t address, uint8_t val) {
    int r = libusb_control_transfer(dev_handle_,
                                    0x40
                                    //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_OUT)
                                    ,
                                    usbvendorcmd, address, 0, &val, 1, 0);
    if (r <= 0) {
        MV_HAL_LOG_ERROR() << "control_write_register_8bits" << r << "err" << libusb_error_name(r);
    }
    {
        auto log_op = MV_HAL_LOG_DEBUG() << "control_write_register_8bits" << r;
        log_op << std::hex << long(usbvendorcmd) << long(address) << val << std::dec;
        if (r <= 0)
            log_op << "err" << libusb_error_name(r);
        log_op << std::endl;
    }
}

uint8_t PseeLibUSBBoardCommand::control_read_register_8bits(uint8_t usbvendorcmd, uint8_t address) {
    unsigned char data[4];
    uint8_t val;
    int r = libusb_control_transfer(dev_handle_,
                                    0xC0
                                    //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_IN)
                                    ,
                                    usbvendorcmd, address, 0, data, 4, 0);
    if (r <= 0) {
        MV_HAL_LOG_ERROR() << "control_read_register_8bits" << r << "err" << libusb_error_name(r);
        return 0;
    }

    {
        auto log_op = MV_HAL_LOG_DEBUG() << "control_read_register_8bits" << r;
        if (r <= 0)
            log_op << "err" << libusb_error_name(r);
        else
            log_op << std::hex << long(usbvendorcmd) << long(address) << long(data[0]) << long(data[1]) << long(data[2])
                   << long(data[3]);
    }

    val = data[2];
    MV_HAL_LOG_DEBUG() << "value" << std::hex << int(val) << std::dec;
    return val;
}

libusb_transfer *PseeLibUSBBoardCommand::contruct_async_bulk_transfer(unsigned char *buf, int packet_size,
                                                                      libusb_transfer_cb_fn async_bulk_cb,
                                                                      void *user_data, unsigned int timeout) {
    if (!dev_handle_) {
        return nullptr;
    }
    libusb_transfer *transfer = nullptr;
    transfer                  = libusb_alloc_transfer(0);
    if (!transfer) {
        MV_HAL_LOG_ERROR() << "libusb_alloc_transfer Failed";
        return transfer;
    }
    prepare_async_bulk_transfer(transfer, buf, packet_size, async_bulk_cb, user_data, timeout);
    return transfer;
}

void PseeLibUSBBoardCommand::prepare_async_bulk_transfer(libusb_transfer *transfer, unsigned char *buf, int packet_size,
                                                         libusb_transfer_cb_fn async_bulk_cb, void *user_data,
                                                         unsigned int timeout) {
    libusb_fill_bulk_transfer(transfer, dev_handle_, (1 | LIBUSB_ENDPOINT_IN), buf, packet_size, async_bulk_cb,
                              user_data, timeout);
    transfer->flags &= ~LIBUSB_TRANSFER_FREE_BUFFER;
    transfer->flags &= ~LIBUSB_TRANSFER_FREE_TRANSFER;
}

void PseeLibUSBBoardCommand::free_async_bulk_transfer(libusb_transfer *transfer) {
    libusb_free_transfer(transfer);
}

int PseeLibUSBBoardCommand::submit_transfer(libusb_transfer *transfer) {
    std::lock_guard<std::mutex> guard(protect_libusb_submit_);
    int r = 0;
    r     = libusb_submit_transfer(transfer);
    if (r < 0) {
        MV_HAL_LOG_ERROR() << "USB Submit Error";
    }
    return r;
}

int PseeLibUSBBoardCommand::bulk_transfer(unsigned char *buf, int packet_size, unsigned int timeout, int &actual_size) {
    if (dev_handle_) {
        return libusb_bulk_transfer(dev_handle_, (1 | LIBUSB_ENDPOINT_IN), buf, packet_size, &actual_size, 100); // 2ms
    } else {
        return LIBUSB_ERROR_NO_DEVICE;
    }
}

void PseeLibUSBBoardCommand::get_ccam2_with_serial(libusb_context *ctx, const std::string &serial) {
    if (dev_handle_) {
        libusb_close(dev_handle_);
        dev_handle_ = nullptr;
        dev_speed_  = LIBUSB_SPEED_UNKNOWN;
    }

    libusb_device **devs;
    int cnt = libusb_get_device_list(ctx, &devs); // get the list of devices
    if (cnt <= 0) {
        MV_HAL_LOG_TRACE() << "Device list empty";
        return;
    }
    libusb_device_handle *dev_handle = nullptr;

    for (int i = 0; i < cnt; i++) {
        libusb_device_descriptor desc;
        int r = libusb_get_device_descriptor(devs[i], &desc);
        if (r < 0) {
            MV_HAL_LOG_TRACE() << "Failed to get device descriptor";
            return;
        }
        if ((desc.idVendor == 0x04b4) &&
            ((desc.idProduct == 0x00f1) || (desc.idProduct == 0x00f4) || (desc.idProduct == 0x00bc))) {
            r = libusb_open(devs[i], &dev_handle);
            if (r != 0) {
                MV_HAL_LOG_TRACE() << "Unable to open device";
                continue;
            }
            if (libusb_kernel_driver_active(dev_handle, 0) == 1) { // find out if kernel driver is attached
                MV_HAL_LOG_TRACE() << "Kernel driver active";
                if (libusb_detach_kernel_driver(dev_handle, 0) == 0) // detach it
                    MV_HAL_LOG_TRACE() << "Kernel driver detached!";
            }
            r = libusb_claim_interface(dev_handle, 0); // claim interface 0 (the first) of device
            if (r < 0) {
                MV_HAL_LOG_TRACE() << "Cannot claim interface";
                libusb_close(dev_handle);
                dev_handle = nullptr;
                continue;
            }
            dev_speed_  = (libusb_speed)libusb_get_device_speed(devs[i]);
            dev_handle_ = dev_handle;
            std::string cur_serial =
                get_serial(); // Done even if serial is "" because get_serial check if FPGA has boot
            if (cur_serial == serial || serial == "") {
                break;
            }
            dev_handle_ = nullptr;
            dev_speed_  = LIBUSB_SPEED_UNKNOWN;
            libusb_release_interface(dev_handle, 0);
            libusb_close(dev_handle);
            dev_handle = nullptr;
        }
    }
    libusb_free_device_list(devs, 1); // free the list, unref the devices in it

    // Not reseting the usb device leads to not flushing the usb data in the driver
    // It leads to time discrepancies at camera start if camera is not unplugged
    // Reseting the device is equivalent to an unplug but should not be necessary.
    // Reseting the device on windows is not necessary as the usb driver behaves correctly
#ifndef _WIN32
    MV_HAL_LOG_TRACE() << "Reset device";
    if (dev_handle_) {
        libusb_reset_device(dev_handle_);
    }
#endif
}

void PseeLibUSBBoardCommand::write_register(Register_Addr register_addr, uint32_t value) {
    init_register(register_addr, value);
    send_register(register_addr);
}

uint32_t PseeLibUSBBoardCommand::read_register(Register_Addr regist) {
    auto it = mregister_state.find(regist);
    if (it == mregister_state.end()) {
        return 0;
    }

    return it->second;
}

void PseeLibUSBBoardCommand::load_register(Register_Addr regist) {
    uint32_t value = control_read_register_32bits(CMD_READ_REGFPGA_32, regist);
    init_register(regist, value);
}

void PseeLibUSBBoardCommand::set_register_bit(Register_Addr register_addr, int idx, bool state) {
    auto it = mregister_state.find(register_addr);
    if (it == mregister_state.end()) {
        it = mregister_state.insert(std::make_pair(register_addr, static_cast<uint32_t>(0))).first;
    }
    if (state) {
        it->second |= (1 << idx);
    } else {
        it->second &= ~(1 << idx);
    }
}

void PseeLibUSBBoardCommand::send_register(Register_Addr register_addr) {
    uint32_t val = 0;
    if (has_register(register_addr)) {
        val = read_register(register_addr);
    }
    control_write_register_32bits(CMD_WRITE_REGFPGA_32, register_addr, val);
}

void PseeLibUSBBoardCommand::send_register_bit(Register_Addr register_addr, int idx, bool state) {
    set_register_bit(register_addr, idx, state);
    send_register(register_addr);
}

uint32_t PseeLibUSBBoardCommand::read_register_bit(Register_Addr register_addr, int idx) {
    MV_HAL_LOG_DEBUG() << __PRETTY_FUNCTION__ << register_addr;
    auto it = mregister_state.find(register_addr);
    if (it == mregister_state.end()) {
        return 0;
    }

    return (it->second >> idx) & 1;
}

void PseeLibUSBBoardCommand::init_register(Register_Addr regist, uint32_t value) {
    mregister_state[regist] = value;
}

bool PseeLibUSBBoardCommand::has_register(Register_Addr regist) {
    auto it = mregister_state.find(regist);
    return it != mregister_state.end();
}

bool PseeLibUSBBoardCommand::wait_fpga_boot_state() {
    long FPGA_boot_state                                           = check_fpga_boot_state();
    std::chrono::time_point<std::chrono::system_clock> const start = std::chrono::system_clock::now();
    std::chrono::duration<double> time_elapsed_s{};

    while (!FPGA_boot_state && time_elapsed_s.count() < 10) {
        FPGA_boot_state = check_fpga_boot_state();
        time_elapsed_s  = std::chrono::system_clock::now() - start;
    }
    return FPGA_boot_state;
}

void PseeLibUSBBoardCommand::reset_fpga() {
    control_write_register_32bits(CMD_RESET_FPGA, 0x34, 0);
}

long PseeLibUSBBoardCommand::check_fpga_boot_state() {
    uint16_t val = control_read_register_16bits(CMD_CHECK_FPGA_BOOT_STATE, 0x00);
    return val;
}

long PseeLibUSBBoardCommand::get_system_id() {
    long FPGA_boot_state = wait_fpga_boot_state();
    if (!FPGA_boot_state) {
        return SYSTEM_INVALID_NO_FPGA;
    }
    uint16_t val = control_read_register_16bits(CMD_READ_SYSTEM_ID, 0x00);
    return val;
}

long PseeLibUSBBoardCommand::get_system_version() {
    uint32_t val_32 = control_read_register_32bits(CMD_READ_REGFPGA_32, STEREO_SYSTEM_CONFIG_VERSION_ADDR);
    return val_32;
}

unsigned int PseeLibUSBBoardCommand::get_system_build_date() {
    uint32_t val_32 = control_read_register_32bits(CMD_READ_REGFPGA_32, STEREO_SYSTEM_CONFIG_BUILD_DATE_ADDR);
    return val_32;
}

unsigned int PseeLibUSBBoardCommand::get_system_version_control_id() {
    uint32_t val_32 = control_read_register_32bits(CMD_READ_REGFPGA_32, STEREO_SYSTEM_CONFIG_VERSION_CONTROL_ID_ADDR);
    return val_32;
}

long PseeLibUSBBoardCommand::get_temperature() {
    return -1;
}

long PseeLibUSBBoardCommand::get_illumination() {
    return -1;
}

// ------------------------------
// Static methods

bool PseeLibUSBBoardCommand::init_libusb() {
    int r;                    // for return values
    r = libusb_init(nullptr); // initialize the library for the session we just declared
    if (r < 0) {
        MV_HAL_LOG_ERROR() << "An error occured while initializing libusb:" << libusb_error_name(r);
    }
    return r >= 0;
}

} // namespace Metavision
