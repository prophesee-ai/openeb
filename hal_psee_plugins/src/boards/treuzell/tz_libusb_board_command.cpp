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

#include <assert.h>
#include <iomanip>
#ifndef _MSC_VER
#else
#include <io.h>
#include <stdio.h>
#endif
#include <stdlib.h>
#include <sstream>

#include "metavision/hal/utils/hal_log.h"
#include "boards/utils/config_registers_map.h"
#include "boards/treuzell/tz_libusb_board_command.h"
#include "boards/treuzell/treuzell_command_definition.h"
#include "boards/treuzell/tz_control_frame.h"
#include "devices/utils/device_system_id.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

#ifdef USE_JAVA_BINDINGS
#include "is_usb_java.h"
#endif

#define PSEE_EVK_PROTOCOL 0

#define TZ_MAX_ANSWER_SIZE 1024

namespace Metavision {

// By default, nothing is supported, because we want boards to be ignored by the plugins that can manage it, so that
// only one open a given board
std::vector<UsbInterfaceId> TzLibUSBBoardCommand::known_usb_ids;

TzLibUSBBoardCommand::TzLibUSBBoardCommand(std::shared_ptr<LibUSBContext> ctx, libusb_device *dev,
                                           libusb_device_descriptor &desc) :
    libusb_ctx(ctx) {
    // Check only the first device configuration
    struct libusb_config_descriptor *config;
    int r = libusb_get_config_descriptor(dev, 0, &config);
    if (r != LIBUSB_SUCCESS) {
        throw(HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "config descriptor not readable."));
    }

    // Look for a treuzell interface
    bInterfaceNumber = -1;
    for (int ifc = 0; ifc < config->bNumInterfaces; ifc++) {
        const struct libusb_interface *interface = &config->interface[ifc];
        // Check each setting
        for (int altsetting = 0; altsetting < interface->num_altsetting; altsetting++) {
            const struct libusb_interface_descriptor *ifc_desc = &interface->altsetting[altsetting];
            // Check if USB class is the expected one
            bool supported = false;
            for (const auto &id : TzLibUSBBoardCommand::known_usb_ids) {
                if ((id.vid && (desc.idVendor == id.vid)) && (id.pid && (desc.idProduct == id.pid)) &&
                    (ifc_desc->bInterfaceClass == id.usb_class) && (ifc_desc->bInterfaceSubClass == (id.subclass))) {
                    supported = true;
                }
            }

            if (!supported) {
                continue;
            }

            // Check if the interface has the expected structure
            if ((ifc_desc->bInterfaceProtocol == PSEE_EVK_PROTOCOL) && (ifc_desc->bNumEndpoints == 3) &&
                ((ifc_desc->endpoint[0]).bmAttributes == LIBUSB_TRANSFER_TYPE_BULK) &&
                (((ifc_desc->endpoint[0]).bEndpointAddress & 0x80) == LIBUSB_ENDPOINT_IN) &&
                ((ifc_desc->endpoint[1]).bmAttributes == LIBUSB_TRANSFER_TYPE_BULK) &&
                (((ifc_desc->endpoint[1]).bEndpointAddress & 0x80) == LIBUSB_ENDPOINT_OUT) &&
                ((ifc_desc->endpoint[2]).bmAttributes == LIBUSB_TRANSFER_TYPE_BULK) &&
                (((ifc_desc->endpoint[2]).bEndpointAddress & 0x80) == LIBUSB_ENDPOINT_IN)) {
                bInterfaceNumber = ifc_desc->bInterfaceNumber;
                bEpControlIn     = (ifc_desc->endpoint[0]).bEndpointAddress;
                bEpControlOut    = (ifc_desc->endpoint[1]).bEndpointAddress;
                bEpCommAddress   = (ifc_desc->endpoint[2]).bEndpointAddress;
                break;
            }
        }
        if (bInterfaceNumber >= 0)
            break;
    }
    libusb_free_config_descriptor(config);
    if (bInterfaceNumber < 0) {
        throw(HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "no treuzell interface found."));
    }

    // Open device.
    r = libusb_open(dev, &dev_handle_);
    if (r != 0)
        throw(HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Unable to open device"));

    if (libusb_kernel_driver_active(dev_handle_, bInterfaceNumber) == 1) { // find out if kernel driver is attached
        MV_HAL_LOG_TRACE() << "Kernel Driver Active";
        if (libusb_detach_kernel_driver(dev_handle_, bInterfaceNumber) == 0) // detach it
            MV_HAL_LOG_TRACE() << "Kernel Driver Detached!";
    }
    r = libusb_claim_interface(dev_handle_, bInterfaceNumber);
    if (r < 0) {
        libusb_close(dev_handle_);
        throw(HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Camera is busy"));
    }
    MV_HAL_LOG_TRACE() << "Claimed interface";
    dev_speed_ = (libusb_speed)libusb_get_device_speed(dev);

    if (desc.iManufacturer) {
        char buf[128]; // 256 bytes UTF-16LE, shrinked down to pure ASCII
        if (libusb_get_string_descriptor_ascii(dev_handle_, desc.iManufacturer, (unsigned char *)buf, 128) > 0)
            manufacturer = buf;
    }

    if (desc.iProduct) {
        char buf[128]; // 256 bytes UTF-16LE, shrinked down to pure ASCII
        if (libusb_get_string_descriptor_ascii(dev_handle_, desc.iProduct, (unsigned char *)buf, 128) > 0)
            product = buf;
    }

    try {
        TzGenericCtrlFrame req(TZ_PROP_RELEASE_VERSION);
        transfer_tz_frame(req);
        version = req.get32(0);
    } catch (const std::system_error &e) { MV_HAL_LOG_TRACE() << "Got no board version:" << e.what(); }

    try {
        TzGenericCtrlFrame req(TZ_PROP_BUILD_DATE);
        transfer_tz_frame(req);
        build_date = req.get64(0);
    } catch (const std::system_error &e) { MV_HAL_LOG_TRACE() << "Got no build date:" << e.what(); }
}

TzLibUSBBoardCommand::~TzLibUSBBoardCommand() {
    int r = libusb_release_interface(dev_handle_, bInterfaceNumber); // release the claimed interface
    if (r != 0) {
        MV_HAL_LOG_WARNING() << "Cannot release interface";
    } else {
        MV_HAL_LOG_TRACE() << "Released interface";
    }
    // Old Evk2s used the global device reset to reset their Tz interface
    if ((product == "EVKv2") && (version < 0x010600))
        libusb_reset_device(dev_handle_);
    libusb_close(dev_handle_);
}

std::string TzLibUSBBoardCommand::get_name() {
    return product;
}

std::string TzLibUSBBoardCommand::get_manufacturer() {
    return manufacturer;
}

time_t TzLibUSBBoardCommand::get_build_date() {
    return build_date;
}

uint32_t TzLibUSBBoardCommand::get_version() {
    return version;
}

long TzLibUSBBoardCommand::get_board_speed() {
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

std::string TzLibUSBBoardCommand::get_serial() {
    TzGenericCtrlFrame req(TZ_PROP_SERIAL);
    transfer_tz_frame(req);

    std::ostringstream ostr;
    ostr                 //<< std::showbase // show the 0x prefix
        << std::internal // fill between the prefix and the number
        << std::setfill('0') << std::setw(8) << std::hex;
    try {
        ostr << req.get64(0);
    } catch (const std::system_error &e) {
        // Some implementations only have 32-bits serials
        if (e.code().value() == TZ_TOO_SHORT)
            ostr << req.get32(0);
        else
            throw e;
    }
    ostr << std::dec;
    return ostr.str();
}

void TzLibUSBBoardCommand::write_register(Register_Addr register_addr, uint32_t value) {
    init_register(register_addr, value);
    send_register(register_addr);
}

uint32_t TzLibUSBBoardCommand::read_register(Register_Addr regist) {
    auto it = mregister_state.find(regist);
    if (it == mregister_state.end()) {
        return 0;
    }

    return it->second;
}

void TzLibUSBBoardCommand::load_register(Register_Addr regist) {
    init_register(regist, read_device_register(0, regist)[0]);
}

void TzLibUSBBoardCommand::set_register_bit(Register_Addr register_addr, int idx, bool state) {
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

void TzLibUSBBoardCommand::send_register(Register_Addr register_addr) {
    uint32_t val = 0;
    if (has_register(register_addr)) {
        val = read_register(register_addr);
    }
    write_device_register(0, register_addr, std::vector<uint32_t>(1, val));
}

void TzLibUSBBoardCommand::send_register_bit(Register_Addr register_addr, int idx, bool state) {
    set_register_bit(register_addr, idx, state);
    send_register(register_addr);
}

uint32_t TzLibUSBBoardCommand::read_register_bit(Register_Addr register_addr, int idx) {
    MV_HAL_LOG_DEBUG() << __PRETTY_FUNCTION__ << register_addr;
    auto it = mregister_state.find(register_addr);
    if (it == mregister_state.end()) {
        return 0;
    }

    return (it->second >> idx) & 1;
}

void TzLibUSBBoardCommand::init_register(Register_Addr regist, uint32_t value) {
    mregister_state[regist] = value;
}

bool TzLibUSBBoardCommand::has_register(Register_Addr regist) {
    auto it = mregister_state.find(regist);
    return it != mregister_state.end();
}

bool TzLibUSBBoardCommand::reset_device() {
#ifndef _WINDOWS
    if (dev_handle_) {
        int r = libusb_reset_device(dev_handle_);
        if (r == 0) {
            MV_HAL_LOG_TRACE() << "libusb BC: USB Reset";
            return true;
        } else {
            MV_HAL_LOG_ERROR() << libusb_error_name(r);
            return false;
        }
    } else {
        return false;
    }
#else
    return true;
#endif
}

void TzLibUSBBoardCommand::transfer_tz_frame(TzCtrlFrame &req) {
    int res, sent;
    std::vector<uint8_t> answer(TZ_MAX_ANSWER_SIZE);

    if (!dev_handle_)
        throw std::runtime_error("no libusb dev_handle");

    /* send the command */
    res = libusb_bulk_transfer(dev_handle_, bEpControlOut, req.frame(), req.frame_size(), &sent, 1000);
    if (res < 0)
        throw std::system_error(res, LibUsbError());

    /* get the result */
    res = libusb_bulk_transfer(dev_handle_, bEpControlIn, answer.data(), answer.size(), &sent, 1000);
    if (res < 0)
        throw std::system_error(res, LibUsbError());
    answer.resize(sent);
    req.swap_and_check_answer(answer);
}

unsigned int TzLibUSBBoardCommand::get_device_count() {
    TzGenericCtrlFrame req(TZ_PROP_DEVICES);
    transfer_tz_frame(req);
    return req.get32(0);
}

std::vector<uint32_t> TzLibUSBBoardCommand::read_device_register(uint32_t device, uint32_t address, int nval) {
    TzGenericCtrlFrame req(TZ_PROP_DEVICE_REG32);

    req.push_back32(device);
    req.push_back32(address);
    req.push_back32(nval);

    try {
        transfer_tz_frame(req);
    } catch (std::system_error &e) {
        if (e.code() == std::error_code(TZ_COMMAND_FAILED, TzError())) {
            // in this case, req has been set with the device answer
            int err = req.get32(2);
            throw std::system_error(err, std::generic_category());
        } else {
            throw e;
        }
    }
    if (req.get32(0) != device)
        throw std::system_error(TZ_PROPERTY_MISMATCH, TzError(), "device id mismatch");
    if (req.get32(1) != address)
        throw std::system_error(TZ_PROPERTY_MISMATCH, TzError(), "address mismatch");
    if (req.payload_size() < ((nval + 2) * sizeof(uint32_t)))
        throw std::system_error(TZ_TOO_SHORT, TzError());

    std::vector<uint32_t> res(nval);
    memcpy(res.data(), req.payload() + (2 * sizeof(uint32_t)), nval * sizeof(uint32_t));

    MV_HAL_LOG_DEBUG() << "read_device_register dev " << device << " addr " << address << " val:";
    for (auto const &val : res)
        MV_HAL_LOG_DEBUG() << val;

    return res;
}

void TzLibUSBBoardCommand::write_device_register(uint32_t device, uint32_t address, const std::vector<uint32_t> &val) {
    TzGenericCtrlFrame req(TZ_PROP_DEVICE_REG32 | TZ_WRITE_FLAG);

    req.push_back32(device);
    req.push_back32(address);
    req.push_back32(val);

    MV_HAL_LOG_DEBUG() << "write_device_register dev " << device << " addr " << address << " val:";
    for (auto const &value : val)
        MV_HAL_LOG_DEBUG() << val;

    try {
        transfer_tz_frame(req);
    } catch (std::system_error &e) {
        if (e.code() == std::error_code(TZ_COMMAND_FAILED, TzError())) {
            // in this case, req has been set with the device answer
            int err = req.get32(2);
            throw std::system_error(err, std::generic_category());
        } else {
            throw e;
        }
    }
    if (req.get32(0) != device)
        throw std::system_error(TZ_PROPERTY_MISMATCH, TzError(), "device id mismatch");
    if (req.get32(1) != address)
        throw std::system_error(TZ_PROPERTY_MISMATCH, TzError(), "address mismatch");
}

} // namespace Metavision
