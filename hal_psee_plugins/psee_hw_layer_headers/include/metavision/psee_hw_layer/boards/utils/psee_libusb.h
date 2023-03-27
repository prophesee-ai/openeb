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

#ifndef METAVISION_HAL_PSEE_LIBUSB_H
#define METAVISION_HAL_PSEE_LIBUSB_H

#include <system_error>
#include <memory>

#ifdef _MSC_VER
#define NOMINMAX // libusb.h includes windows.h which defines min max macros that we don't want
#endif
#include <libusb.h>

namespace Metavision {

/// LibUSB error category
class LibUsbError : public std::error_category {
public:
    virtual const char *name() const noexcept {
        return "LibUSB";
    }
    virtual std::string message(int err) const {
        return libusb_error_name(err);
    }
};

/* Hereafter follows a partial encapsulation of libusb in C++
 * The goal is to manage with smartpointers the objects that live as long as the camera, and keep the bare libusb
 * implementation for objects managed by a single class
 */

class LibUSBContext {
public:
    LibUSBContext();
    ~LibUSBContext();
    libusb_context *ctx();

private:
    libusb_context *ctx_;
};

class LibUSBDevice {
public:
    LibUSBDevice(std::shared_ptr<LibUSBContext> libusb_ctx, libusb_device *dev);
    LibUSBDevice(std::shared_ptr<LibUSBContext> libusb_ctx, uint16_t vendor_id, uint16_t product_id);
    ~LibUSBDevice();

    // The returned context is managed by a LibUSBContext
    libusb_context *ctx();

    libusb_device *get_device();
    int get_configuration(int *config);
    int set_configuration(int configuration);
    int claim_interface(int interface_number);
    int release_interface(int interface_number);
    int set_interface_alt_setting(int interface_number, int alternate_setting);
    int clear_halt(unsigned char endpoint);
    int reset_device();
    int kernel_driver_active(int interface_number);
    int detach_kernel_driver(int interface_number);
    int attach_kernel_driver(int interface_number);
    int set_auto_detach_kernel_driver(int enable);

    int get_string_descriptor_ascii(uint8_t desc_index, unsigned char *data, int length);

    void control_transfer(uint8_t bmRequestType, uint8_t bRequest, uint16_t wValue, uint16_t wIndex,
                          unsigned char *data, uint16_t wLength, unsigned int timeout);
    void bulk_transfer(unsigned char endpoint, unsigned char *data, int length, int *transferred, unsigned int timeout);
    void interrupt_transfer(unsigned char endpoint, unsigned char *data, int length, int *transferred,
                            unsigned int timeout);

    friend void libusb_fill_control_transfer(struct libusb_transfer *transfer, std::shared_ptr<LibUSBDevice> dev,
                                             unsigned char *buffer, libusb_transfer_cb_fn callback, void *user_data,
                                             unsigned int timeout) {
        libusb_fill_control_transfer(transfer, dev->dev_handle_, buffer, callback, user_data, timeout);
    }

    friend void libusb_fill_bulk_transfer(struct libusb_transfer *transfer, std::shared_ptr<LibUSBDevice> dev,
                                          unsigned char endpoint, unsigned char *buffer, int length,
                                          libusb_transfer_cb_fn callback, void *user_data, unsigned int timeout) {
        libusb_fill_bulk_transfer(transfer, dev->dev_handle_, endpoint, buffer, length, callback, user_data, timeout);
    }

    friend void libusb_fill_bulk_stream_transfer(struct libusb_transfer *transfer, std::shared_ptr<LibUSBDevice> dev,
                                                 unsigned char endpoint, uint32_t stream_id, unsigned char *buffer,
                                                 int length, libusb_transfer_cb_fn callback, void *user_data,
                                                 unsigned int timeout) {
        libusb_fill_bulk_stream_transfer(transfer, dev->dev_handle_, endpoint, stream_id, buffer, length, callback,
                                         user_data, timeout);
    }

    friend void libusb_fill_interrupt_transfer(struct libusb_transfer *transfer, std::shared_ptr<LibUSBDevice> dev,
                                               unsigned char endpoint, unsigned char *buffer, int length,
                                               libusb_transfer_cb_fn callback, void *user_data, unsigned int timeout) {
        libusb_fill_interrupt_transfer(transfer, dev->dev_handle_, endpoint, buffer, length, callback, user_data,
                                       timeout);
    }

    friend void libusb_fill_iso_transfer(struct libusb_transfer *transfer, std::shared_ptr<LibUSBDevice> dev,
                                         unsigned char endpoint, unsigned char *buffer, int length, int num_iso_packets,
                                         libusb_transfer_cb_fn callback, void *user_data, unsigned int timeout) {
        libusb_fill_iso_transfer(transfer, dev->dev_handle_, endpoint, buffer, length, num_iso_packets, callback,
                                 user_data, timeout);
    }

private:
    std::shared_ptr<LibUSBContext> libusb_ctx_;
    libusb_device_handle *dev_handle_;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_LIBUSB_H
