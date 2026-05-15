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

#ifndef METAVISION_HAL_SAMPLE_USB_CONNECTION_H
#define METAVISION_HAL_SAMPLE_USB_CONNECTION_H

#include <exception>
#include <string>
#ifdef _MSC_VER
#define NOMINMAX // libusb.h includes windows.h which defines min max macros that we don't want
#endif
#include <libusb.h>

class SampleUSBConnection {
public:
    SampleUSBConnection(uint16_t vid, uint16_t pid, int device_interface_num) {
        if (libusb_init(&usb_ctx_) != 0) {
            throw std::runtime_error("Failed to initialize libusb");
        }

        // Open EVK4 via USB
        device_handle_ = libusb_open_device_with_vid_pid(usb_ctx_, vid, pid);
        if (!device_handle_) {
            libusb_exit(usb_ctx_);
            usb_ctx_ = nullptr;
            throw std::runtime_error("Failed to open the USB device");
        }

        // Claim the interface
        if (libusb_claim_interface(device_handle_, device_interface_num) != 0) {
            libusb_close(device_handle_);
            device_handle_ = nullptr;
            libusb_exit(usb_ctx_);
            usb_ctx_ = nullptr;
            throw std::runtime_error(std::string("Failed to claim the interface ") +
                                     std::to_string(device_interface_num));
        }

        // Set the interface alternate setting (to reset the device)
        if (libusb_set_interface_alt_setting(device_handle_, 0, 0) != 0) {
            libusb_close(device_handle_);
            device_handle_ = nullptr;
            libusb_exit(usb_ctx_);
            usb_ctx_ = nullptr;
            throw std::runtime_error("Error setting interface alternate setting.");
        }
    }

    ~SampleUSBConnection() {
        if (device_handle_) {
            libusb_close(device_handle_);
        }
        if (usb_ctx_) {
            libusb_exit(usb_ctx_);
        }
    }

    libusb_device_handle *get_device_handle() const {
        return device_handle_;
    }

private:
    libusb_context *usb_ctx_ = nullptr;
    libusb_device_handle *device_handle_ = nullptr;
};

#endif // METAVISION_HAL_SAMPLE_USB_CONNECTION_H
