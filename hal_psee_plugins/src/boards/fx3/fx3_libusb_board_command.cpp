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

#include <sstream>
#include <iomanip>

#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/boards/fx3/fx3_libusb_board_command.h"
#include "metavision/psee_hw_layer/boards/utils/psee_libusb.h"
#include "metavision/psee_hw_layer/boards/utils/psee_libusb_data_transfer.h"
#include "boards/utils/utils_fx3_ram_flash.h"
#include "boards/utils/vendor_command_definition.h"
#include "boards/utils/config_registers_map.h"
#include "devices/utils/device_system_id.h"

namespace Metavision {

Fx3LibUSBBoardCommand::Fx3LibUSBBoardCommand() = default;

std::string Fx3LibUSBBoardCommand::get_serial() {
    long version = get_board_version();
    uint32_t val = -1;

    long FPGA_boot_state = wait_fpga_boot_state();

    if (FPGA_boot_state) {
        MV_HAL_LOG_TRACE() << "FPGA is properly configured";
    } else {
        MV_HAL_LOG_ERROR() << "FPGA is not properly configured";
    }
    if (version == 3) {
        // CCam5 Gen41
        val = control_read_register_32bits(CMD_READ_SYSTEM_ID, 0x00);
    } else if (version == 2) {
        //  CCam3 Evk1
        val = control_read_register_32bits(CMD_READ_REGFPGA_32, 0);
        val = control_read_register_32bits(CMD_READ_REGFPGA_32, TEP_SERIAL_MSB_ADDR);
        // At this stage, only the MSB register value is returned as serial number is 16 bits limited.
    } else {
        // CCam2 Legacy
        val = control_read_register_16bits(CMD_READ, TEP_SERIAL_MSB_ADDR);
    }
    std::ostringstream ostr;
    ostr                 //<< std::showbase // show the 0x prefix
        << std::internal // fill between the prefix and the number
        << std::setfill('0') << std::setw(8) << std::hex;
    ostr << val;
    ostr << std::dec;
    return ostr.str();
}

long Fx3LibUSBBoardCommand::try_to_flush() {
    int actual         = 0;
    long total_flush   = 0;
    long last_flush    = 0;
    int r              = 0;
    int num_iterations = 0;
    int max_iterations = 8;
    int max_data       = 300000;

    MV_HAL_LOG_TRACE() << "Start flushing";
    MV_HAL_LOG_TRACE() << "Hard flush";

    if (dev_) {
        dev_->clear_halt((1 | LIBUSB_ENDPOINT_IN));
    }

    do {
        if (num_iterations != 0 || total_flush != 0) {
            MV_HAL_LOG_TRACE() << "Flushing" << total_flush;
        }

        last_flush = total_flush;
        // send_ccam2ctrl_command(TEP_CCAM2_CONTROL_ENABLE_FX3_HOST_IF_BIT_IDX,  0); // Enables the FX3 host I/F and
        // also serves to add a delay. Keep this line, unless the FX3 will not have time to change buffers and reply.
        write_register(TEP_TRIGGERS_ADDR, 0);

        do {
            uint8_t buf[1024];
            actual = 0;
            r      = bulk_transfer(buf, 1024, 100, actual); // 2ms
            total_flush += actual;
            if (total_flush > max_data) {
                break;
            }
        } while (r == 0 && actual > 0);

        num_iterations++;

        if (num_iterations > max_iterations) {
            MV_HAL_LOG_WARNING() << "Aborting flush: maximum number of iterations reached!";
            break;
        }

        if (total_flush > max_data) {
            MV_HAL_LOG_WARNING() << "Aborting flush: maximum data amount reached!";
            break;
        }
    } while (last_flush != total_flush);

    MV_HAL_LOG_TRACE() << "Flushed" << total_flush;
    return total_flush;
}

Fx3LibUSBBoardCommand::Fx3LibUSBBoardCommand(std::shared_ptr<LibUSBDevice> dev) :
    dev_(dev), dev_speed_(LIBUSB_SPEED_UNKNOWN) {}

Fx3LibUSBBoardCommand::~Fx3LibUSBBoardCommand() {
    if (dev_) {
        int r = dev_->release_interface(0); // release the claimed interface
        if (r != 0) {
            MV_HAL_LOG_WARNING() << "Cannot release interface";
        } else {
            MV_HAL_LOG_TRACE() << "Released interface";
        }
    }
}

bool Fx3LibUSBBoardCommand::open(const std::string &serial) {
    std::shared_ptr<LibUSBContext> libusb_ctx;

    try {
        libusb_ctx = std::make_shared<LibUSBContext>();
    } catch (const std::system_error &e) {
        MV_HAL_LOG_ERROR() << "An error occurred while initializing libusb:" << e.what();
        return false;
    }
    get_ccam2_with_serial(libusb_ctx, serial);
    return dev_ != nullptr;
}

long Fx3LibUSBBoardCommand::get_board_version() {
    uint16_t val = control_read_register_16bits(CMD_READ_VERSION_FX3, 0x00);
    return val;
}

long Fx3LibUSBBoardCommand::get_board_id() {
    uint32_t val = control_read_register_32bits(CMD_READ_FX3_ID, 0x00, false);
    return val;
}

long Fx3LibUSBBoardCommand::get_board_release_version() {
    uint32_t val = control_read_register_32bits(CMD_READ_FX3_RELEASE_VERSION, 0x00, false);
    return val;
}

long Fx3LibUSBBoardCommand::get_board_build_date() {
    uint32_t val = control_read_register_32bits(CMD_READ_FX3_BUILD_DATE, 0x00, false);
    return val;
}

long Fx3LibUSBBoardCommand::get_board_speed() {
    if (!dev_) {
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

long Fx3LibUSBBoardCommand::get_board_version_control_id() {
    uint32_t val = control_read_register_32bits(CMD_READ_FX3_VERSION_CONTROL_ID, 0x00, false);
    return val;
}

uint32_t Fx3LibUSBBoardCommand::control_read_register_32bits(uint8_t usbvendorcmd, uint32_t address, bool big_endian) {
    uint32_t val = -1;
    if (!dev_) {
        MV_HAL_LOG_ERROR() << "ERR no dev_handle";
        return val;
    }
    unsigned char data[8];
    int r = 0;
    try {
        dev_->control_transfer(0xC0, //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_IN)
                               usbvendorcmd, uint16_t(address & 0xFFFF), uint16_t((address >> 16) & 0xFFFF), data, 8,
                               0);
    } catch (const std::system_error &e) {
        r = e.code().value();
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

uint16_t Fx3LibUSBBoardCommand::control_read_register_16bits(uint8_t usbvendorcmd, uint32_t address) {
    uint16_t val = -1;
    if (!dev_) {
        MV_HAL_LOG_ERROR() << "ERR no dev_handle";
        return val;
    }
    unsigned char data[4];
    int r = 0;
    try {
        dev_->control_transfer(0xC0, //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_IN)
                               usbvendorcmd, address, 0, data, 4, 0);
    } catch (const std::system_error &e) {
        r = e.code().value();
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

void Fx3LibUSBBoardCommand::control_write_register_32bits(uint8_t usbvendorcmd, uint32_t address, uint32_t val) {
    if (!dev_) {
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

    int r = 0;
    try {
        dev_->control_transfer(0x40, //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_OUT)
                               usbvendorcmd, uint16_t(address & 0xFFFF), uint16_t((address >> 16) & 0xFFFF), data, 4,
                               0);
    } catch (const std::system_error &e) {
        r = e.code().value();
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

void Fx3LibUSBBoardCommand::control_write_vector_32bits(uint32_t base_address, const std::vector<uint32_t> &val) {
    if (!dev_) {
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

    int r = 0;
    try {
        dev_->control_transfer(
            0x40, //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_OUT)
            ((base_address == CCAM2IF_LEFT_BASE_ADDRESS) ? CMD_WRITE_VEC_REGFPGA_32 : CMD_WRITE_VEC_SLAVE_REGFPGA_32),
            0, 0, data.data(), data.size(), 0);
    } catch (const std::system_error &e) {
        r = e.code().value();
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

void Fx3LibUSBBoardCommand::control_write_register_8bits(uint8_t usbvendorcmd, uint8_t address, uint8_t val) {
    int r = 0;
    try {
        dev_->control_transfer(0x40, //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_OUT)
                               usbvendorcmd, address, 0, &val, 1, 0);
    } catch (const std::system_error &e) {
        r = e.code().value();
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

uint8_t Fx3LibUSBBoardCommand::control_read_register_8bits(uint8_t usbvendorcmd, uint8_t address) {
    unsigned char data[4];
    uint8_t val;
    int r = 0;
    try {
        dev_->control_transfer(0xC0, //(LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_IN)
                               usbvendorcmd, address, 0, data, 4, 0);
    } catch (const std::system_error &e) {
        r = e.code().value();
        MV_HAL_LOG_ERROR() << "control_read_register_8bits" << r << "err" << libusb_error_name(r);
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

int Fx3LibUSBBoardCommand::bulk_transfer(unsigned char *buf, int packet_size, unsigned int timeout, int &actual_size) {
    if (dev_) {
        try {
            dev_->bulk_transfer((1 | LIBUSB_ENDPOINT_IN), buf, packet_size, &actual_size, 100); // 2ms
            return 0;
        } catch (const std::system_error &e) { return e.code().value(); }
    } else {
        return LIBUSB_ERROR_NO_DEVICE;
    }
}

void Fx3LibUSBBoardCommand::get_ccam2_with_serial(std::shared_ptr<LibUSBContext> libusb_ctx,
                                                  const std::string &serial) {
    dev_       = nullptr;
    dev_speed_ = LIBUSB_SPEED_UNKNOWN;

    libusb_device **devs;
    int cnt = libusb_get_device_list(libusb_ctx->ctx(), &devs); // get the list of devices
    if (cnt <= 0) {
        MV_HAL_LOG_TRACE() << "Device list empty";
        return;
    }
    std::shared_ptr<LibUSBDevice> dev_handle;

    for (int i = 0; i < cnt; i++) {
        libusb_device_descriptor desc;
        int r = libusb_get_device_descriptor(devs[i], &desc);
        if (r < 0) {
            MV_HAL_LOG_TRACE() << "Failed to get device descriptor";
            return;
        }
        if ((desc.idVendor == 0x04b4) &&
            ((desc.idProduct == 0x00f1) || (desc.idProduct == 0x00f4) || (desc.idProduct == 0x00bc))) {
            try {
                dev_handle = std::make_shared<LibUSBDevice>(libusb_ctx, devs[i]);
            } catch (const std::system_error &e) {
                MV_HAL_LOG_TRACE() << "Unable to open device:" << e.what();
                continue;
            }
            if (dev_handle->kernel_driver_active(0) == 1) { // find out if kernel driver is attached
                MV_HAL_LOG_TRACE() << "Kernel driver active";
                if (dev_handle->detach_kernel_driver(0) == 0) // detach it
                    MV_HAL_LOG_TRACE() << "Kernel driver detached!";
            }
            r = dev_handle->claim_interface(0); // claim interface 0 (the first) of device
            if (r < 0) {
                MV_HAL_LOG_TRACE() << "Cannot claim interface";
                dev_handle = nullptr;
                continue;
            }
            dev_speed_ = (libusb_speed)libusb_get_device_speed(devs[i]);
            dev_       = dev_handle;
            std::string cur_serial =
                get_serial(); // Done even if serial is "" because get_serial check if FPGA has boot
            if (cur_serial == serial || serial == "") {
                break;
            }
            dev_->release_interface(0);
            dev_       = nullptr;
            dev_speed_ = LIBUSB_SPEED_UNKNOWN;
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
    if (dev_) {
        dev_->reset_device();
    }
#endif
}

void Fx3LibUSBBoardCommand::write_register(Register_Addr register_addr, uint32_t value) {
    init_register(register_addr, value);
    send_register(register_addr);
}

uint32_t Fx3LibUSBBoardCommand::read_register(Register_Addr regist) {
    auto it = mregister_state.find(regist);
    if (it == mregister_state.end()) {
        return 0;
    }

    return it->second;
}

void Fx3LibUSBBoardCommand::load_register(Register_Addr regist) {
    uint32_t value = control_read_register_32bits(CMD_READ_REGFPGA_32, regist);
    init_register(regist, value);
}

void Fx3LibUSBBoardCommand::set_register_bit(Register_Addr register_addr, int idx, bool state) {
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

void Fx3LibUSBBoardCommand::send_register(Register_Addr register_addr) {
    uint32_t val = 0;
    if (has_register(register_addr)) {
        val = read_register(register_addr);
    }
    control_write_register_32bits(CMD_WRITE_REGFPGA_32, register_addr, val);
}

void Fx3LibUSBBoardCommand::send_register_bit(Register_Addr register_addr, int idx, bool state) {
    set_register_bit(register_addr, idx, state);
    send_register(register_addr);
}

uint32_t Fx3LibUSBBoardCommand::read_register_bit(Register_Addr register_addr, int idx) {
    MV_HAL_LOG_DEBUG() << __PRETTY_FUNCTION__ << register_addr;
    auto it = mregister_state.find(register_addr);
    if (it == mregister_state.end()) {
        return 0;
    }

    return (it->second >> idx) & 1;
}

void Fx3LibUSBBoardCommand::init_register(Register_Addr regist, uint32_t value) {
    mregister_state[regist] = value;
}

bool Fx3LibUSBBoardCommand::has_register(Register_Addr regist) {
    auto it = mregister_state.find(regist);
    return it != mregister_state.end();
}

bool Fx3LibUSBBoardCommand::wait_fpga_boot_state() {
    long FPGA_boot_state                                           = check_fpga_boot_state();
    std::chrono::time_point<std::chrono::system_clock> const start = std::chrono::system_clock::now();
    std::chrono::duration<double> time_elapsed_s{};

    while (!FPGA_boot_state && time_elapsed_s.count() < 10) {
        FPGA_boot_state = check_fpga_boot_state();
        time_elapsed_s  = std::chrono::system_clock::now() - start;
    }
    return FPGA_boot_state;
}

void Fx3LibUSBBoardCommand::reset_fpga() {
    control_write_register_32bits(CMD_RESET_FPGA, 0x34, 0);
}

long Fx3LibUSBBoardCommand::check_fpga_boot_state() {
    uint16_t val = control_read_register_16bits(CMD_CHECK_FPGA_BOOT_STATE, 0x00);
    return val;
}

long Fx3LibUSBBoardCommand::get_system_id() {
    long FPGA_boot_state = wait_fpga_boot_state();
    if (!FPGA_boot_state) {
        return SYSTEM_INVALID_NO_FPGA;
    }
    uint16_t val = control_read_register_16bits(CMD_READ_SYSTEM_ID, 0x00);
    return val;
}

long Fx3LibUSBBoardCommand::get_system_version() {
    uint32_t val_32 = control_read_register_32bits(CMD_READ_REGFPGA_32, STEREO_SYSTEM_CONFIG_VERSION_ADDR);
    return val_32;
}

unsigned int Fx3LibUSBBoardCommand::get_system_build_date() {
    uint32_t val_32 = control_read_register_32bits(CMD_READ_REGFPGA_32, STEREO_SYSTEM_CONFIG_BUILD_DATE_ADDR);
    return val_32;
}

unsigned int Fx3LibUSBBoardCommand::get_system_version_control_id() {
    uint32_t val_32 = control_read_register_32bits(CMD_READ_REGFPGA_32, STEREO_SYSTEM_CONFIG_VERSION_CONTROL_ID_ADDR);
    return val_32;
}

long Fx3LibUSBBoardCommand::get_temperature() {
    return -1;
}

long Fx3LibUSBBoardCommand::get_illumination() {
    return -1;
}

std::unique_ptr<PseeLibUSBDataTransfer> Fx3LibUSBBoardCommand::build_data_transfer(uint32_t raw_event_size_bytes) {
    return std::make_unique<PseeLibUSBDataTransfer>(dev_, (1 | LIBUSB_ENDPOINT_IN), raw_event_size_bytes);
}

// ------------------------------
// Static methods

Fx3LibUSBBoardCommand::ListSerial Fx3LibUSBBoardCommand::get_list_serial() {
    ListSerial lserial;
    std::shared_ptr<LibUSBContext> libusb_ctx;

    try {
        libusb_ctx = std::make_shared<LibUSBContext>();
    } catch (const std::system_error &e) {
        MV_HAL_LOG_ERROR() << "An error occurred while initializing libusb:" << e.what();
        return lserial;
    }
    get_all_serial(libusb_ctx, lserial);
    return lserial;
}

void Fx3LibUSBBoardCommand::get_all_serial(std::shared_ptr<LibUSBContext> libusb_ctx, ListSerial &lserial) {
    libusb_device **devs;
    int cnt = libusb_get_device_list(libusb_ctx->ctx(), &devs); // get the list of devices
    if (cnt <= 0) {
        MV_HAL_LOG_TRACE() << "EVK1 libusb BC: USB Device list empty cnt=" << cnt;
        return;
    }

    MV_HAL_LOG_TRACE() << "EVK1 libusb BC: libusb_get_device_list found" << cnt << "devices";

    for (int i = 0; i < cnt; i++) {
        libusb_device_descriptor desc;
        int r = libusb_get_device_descriptor(devs[i], &desc);
        if (r < 0) {
            MV_HAL_LOG_ERROR() << "Failed to get device descriptor r=" << r;
            return;
        }
        if ((desc.idVendor == 0x04b4) &&
            ((desc.idProduct == 0x00f1) || (desc.idProduct == 0x00f4) || (desc.idProduct == 0x00bc))) {
            std::shared_ptr<LibUSBDevice> dev_handle;
            try {
                dev_handle = std::make_shared<LibUSBDevice>(libusb_ctx, devs[i]);
            } catch (const std::system_error &e) {
                MV_HAL_LOG_TRACE() << "Unable to open device:" << e.what();
                continue;
            }
            MV_HAL_LOG_TRACE() << "EVK1 libusb BC: PSEE device found";
            if (dev_handle->kernel_driver_active(0) == 1) { // find out if kernel driver is attached
                MV_HAL_LOG_TRACE() << "Kernel driver active";
                if (dev_handle->detach_kernel_driver(0) == 0) // detach it
                    MV_HAL_LOG_TRACE() << "Kernel driver detached!";
            }
            r = dev_handle->claim_interface(0); // claim interface 0 (the first) of device
            if (r < 0) {
                MV_HAL_LOG_ERROR() << Log::no_space << "Camera is busy (r=" << r << ")";
                dev_handle = nullptr;
                continue;
            }

            // Assert LIBUSB_SPEED is at minimum LIBUSB_SPEED_SUPER
            Fx3LibUSBBoardCommand cmd(dev_handle);
            int speed             = libusb_get_device_speed(devs[i]);
            const auto cur_serial = cmd.get_serial();
            if (speed < LIBUSB_SPEED_SUPER) {
                MV_HAL_LOG_WARNING()
                    << "Your EVK camera" << cur_serial
                    << "isn't connected in USB3. Please verify your connection or some malfunction may occur.";
            }

            // Add device even if possibly USB 2
            lserial.push_back(cur_serial);
        }
    }
    libusb_free_device_list(devs, 1); // free the list, unref the devices in it
}
} // namespace Metavision
