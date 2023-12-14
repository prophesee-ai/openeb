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
#include <unordered_set>

#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/boards/utils/psee_libusb.h"
#include "metavision/psee_hw_layer/boards/utils/psee_libusb_data_transfer.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "boards/treuzell/treuzell_command_definition.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_control_frame.h"
#include "devices/utils/device_system_id.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

#ifdef USE_JAVA_BINDINGS
#include "is_usb_java.h"
#endif

#define PSEE_EVK_PROTOCOL 0

#define TZ_MAX_ANSWER_SIZE 1024

namespace Metavision {

TzLibUSBBoardCommand::TzLibUSBBoardCommand(std::shared_ptr<LibUSBContext> ctx, libusb_device *dev,
                                           libusb_device_descriptor &desc, const std::vector<UsbInterfaceId> &usb_ids) :
    libusb_ctx(ctx), quirks({0}) {
    // Check only the first device configuration
    struct libusb_config_descriptor *config;
    int r = libusb_get_config_descriptor(dev, 0, &config);
    if (r != LIBUSB_SUCCESS) {
        throw(
            Metavision::HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "config descriptor not readable."));
    }

    // Select quirks using only device descriptor
    select_early_quirks(desc);

    // Look for a treuzell interface
    bInterfaceNumber = -1;
    for (int ifc = 0; ifc < config->bNumInterfaces; ifc++) {
        const struct libusb_interface *interface = &config->interface[ifc];
        // Check each setting
        for (int altsetting = 0; altsetting < interface->num_altsetting; altsetting++) {
            const struct libusb_interface_descriptor *ifc_desc = &interface->altsetting[altsetting];
            // Check if USB class is the expected one
            bool supported = false;
            for (const auto &id : usb_ids) {
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
        throw(Metavision::HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "no treuzell interface found."));
    }

    // Open device.
    try {
        dev_ = std::make_shared<LibUSBDevice>(ctx, dev);
    } catch (std::system_error &e) {
        MV_HAL_LOG_WARNING() << "Unable to open device:" << e.what();
        throw(Metavision::HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Unable to open device"));
    }

    if (desc.iManufacturer) {
        char buf[128]; // 256 bytes UTF-16LE, shrinked down to pure ASCII
        if (dev_->get_string_descriptor_ascii(desc.iManufacturer, (unsigned char *)buf, 128) > 0)
            manufacturer = buf;
    }

    if (desc.iProduct) {
        char buf[128]; // 256 bytes UTF-16LE, shrinked down to pure ASCII
        if (dev_->get_string_descriptor_ascii(desc.iProduct, (unsigned char *)buf, 128) > 0)
            product = buf;
    }

    if (dev_->kernel_driver_active(bInterfaceNumber) == 1) { // find out if kernel driver is attached
        MV_HAL_LOG_TRACE() << "Kernel Driver Active on interface" << bInterfaceNumber << "of" << product;
        if (dev_->detach_kernel_driver(bInterfaceNumber) == 0) // detach it
            MV_HAL_LOG_TRACE() << "Kernel Driver Detached from interface" << bInterfaceNumber << "of" << product;
    }
    r = dev_->claim_interface(bInterfaceNumber);
    if (r < 0) {
        throw(Metavision::HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Camera is busy"));
    }
    MV_HAL_LOG_TRACE() << "Claimed interface" << bInterfaceNumber << "of" << product;
    dev_speed_ = (libusb_speed)libusb_get_device_speed(dev);

    if (!quirks.do_not_set_config)
        r = dev_->set_interface_alt_setting(bInterfaceNumber, 0);
    if (r < 0) {
        throw(Metavision::HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Could not set AltSetting"));
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

    select_board_quirks(desc);

    // Add a warning if using an Evk3/4 with a too old firmware
    static std::unordered_set<std::string> outdated_fw_warning_map;
    if ((desc.idVendor == 0x04b4) && ((desc.idProduct == 0x00f4) || (desc.idProduct == 0x00f5))) {
        if (version < 0x30800) {
            const std::string &serial = get_serial();
            if (outdated_fw_warning_map.count(serial) == 0) {
                MV_HAL_LOG_ERROR() << "The EVK camera with serial" << serial
                                   << "is using an old firmware version. Please upgrade to latest version."
                                   << "Check https://support.prophesee.ai for more information.";
                outdated_fw_warning_map.insert(serial);
            }
            throw Metavision::HalException(PseeHalPluginErrorCode::UnsupportedFirmware,
                                           "Firmware of camera " + serial + " is no longer supported");
        }
    }
}

TzLibUSBBoardCommand::~TzLibUSBBoardCommand() {
    int r = dev_->release_interface(bInterfaceNumber); // release the claimed interface
    if (r != 0) {
        MV_HAL_LOG_WARNING() << "Cannot release interface";
    } else {
        MV_HAL_LOG_TRACE() << "Released interface" << bInterfaceNumber << "on" << product;
    }
    if (quirks.reset_on_destroy)
        dev_->reset_device();
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

bool TzLibUSBBoardCommand::reset_device() {
#ifndef _WINDOWS
    int r = dev_->reset_device();
    if (r == 0) {
        MV_HAL_LOG_TRACE() << "libusb BC: USB Reset";
        return true;
    } else {
        MV_HAL_LOG_ERROR() << libusb_error_name(r);
        return false;
    }
#else
    return true;
#endif
}

void TzLibUSBBoardCommand::transfer_tz_frame(TzCtrlFrame &req) {
    int sent;
    std::vector<uint8_t> answer(TZ_MAX_ANSWER_SIZE);
    {
        std::lock_guard<std::mutex> guard(tz_control_mutex_);

        /* send the command */
        dev_->bulk_transfer(bEpControlOut, req.frame(), req.frame_size(), &sent, 1000);

        /* get the result */
        dev_->bulk_transfer(bEpControlIn, answer.data(), answer.size(), &sent, 10000);
    }
    answer.resize(sent);
    req.swap_and_check_answer(answer);
}

unsigned int TzLibUSBBoardCommand::get_device_count() {
    TzGenericCtrlFrame req(TZ_PROP_DEVICES);
    try {
        transfer_tz_frame(req);
    } catch (std::system_error &e) {
        if (!quirks.ignore_size_on_device_prop_answer || (e.code().value() != TZ_SIZE_MISMATCH)) {
            // if quirk is enabled and error is SIZE_MISMATCH, ignore it
            throw e;
        }
    }
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

    if (std::getenv("TZ_LOG_REGISTERS")) {
        MV_HAL_LOG_TRACE() << "read_device_register dev" << device << "addr" << address << "val" << res;
    }

    return res;
}

void TzLibUSBBoardCommand::write_device_register(uint32_t device, uint32_t address, const std::vector<uint32_t> &val) {
    TzGenericCtrlFrame req(TZ_PROP_DEVICE_REG32 | TZ_WRITE_FLAG);

    req.push_back32(device);
    req.push_back32(address);
    req.push_back32(val);

    if (std::getenv("TZ_LOG_REGISTERS")) {
        MV_HAL_LOG_TRACE() << "write_device_register dev" << device << "addr" << address << "val" << val;
    }

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

std::unique_ptr<DataTransfer> TzLibUSBBoardCommand::build_data_transfer(uint32_t raw_event_size_bytes) {
    return std::make_unique<PseeLibUSBDataTransfer>(dev_, bEpCommAddress, raw_event_size_bytes);
}

void TzLibUSBBoardCommand::select_board_quirks(libusb_device_descriptor &desc) {
    // Old Evk2s used the global device reset to reset their Tz interface
    if ((desc.idVendor == 0x03fd) && (desc.idProduct == 0x5832) && (product == "EVKv2")) {
        if (version < 0x010600)
            quirks.reset_on_destroy = true;
        if (version < 0x010800)
            quirks.ignore_size_on_device_prop_answer = true;
    }
    if ((desc.idVendor == 0x03fd) && (desc.idProduct == 0x5832) && (product == "Testboard")) {
        if (version < 0x010600)
            quirks.reset_on_destroy = true;
        if (version < 0x010700)
            quirks.ignore_size_on_device_prop_answer = true;
    }
}

void TzLibUSBBoardCommand::select_early_quirks(libusb_device_descriptor &desc) {
    // Evk3/4
    if ((desc.idVendor == 0x04b4) && ((desc.idProduct == 0x00f4) || (desc.idProduct == 0x00f5))) {
        if (desc.bcdDevice < 0x0307)
            quirks.do_not_set_config = true;
    }
}

} // namespace Metavision
