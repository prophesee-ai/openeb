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

#include "metavision/hal/utils/hal_connection_exception.h"
#include "metavision/psee_hw_layer/boards/utils/psee_libusb.h"

namespace Metavision {

const std::error_category &libusb_error_category() {
    // The category singleton
    static LibUsbError instance;
    return instance;
}

LibUSBContext::LibUSBContext() {
    int err;
    err = libusb_init(&ctx_);
    if (err) {
        throw HalConnectionException(err, libusb_error_category());
    }
}

LibUSBContext::~LibUSBContext() {
    libusb_exit(ctx_);
}

libusb_context *LibUSBContext::ctx() {
    return ctx_;
}

LibUSBDevice::LibUSBDevice(std::shared_ptr<LibUSBContext> libusb_ctx, libusb_device *dev) : libusb_ctx_(libusb_ctx) {
    int err;
    err = libusb_open(dev, &dev_handle_);
    if (err) {
        throw HalConnectionException(err, libusb_error_category());
    }
}

LibUSBDevice::LibUSBDevice(std::shared_ptr<LibUSBContext> libusb_ctx, uint16_t vendor_id, uint16_t product_id) :
    libusb_ctx_(libusb_ctx) {
    libusb_context *ctx = libusb_ctx ? libusb_ctx->ctx() : NULL;
    dev_handle_         = libusb_open_device_with_vid_pid(ctx, vendor_id, product_id);
    if (!dev_handle_) {
        throw HalConnectionException(LIBUSB_ERROR_NO_DEVICE, libusb_error_category());
    }
}

LibUSBDevice::~LibUSBDevice() {
    libusb_close(dev_handle_);
}

libusb_context *LibUSBDevice::ctx() {
    return libusb_ctx_->ctx();
}

libusb_device *LibUSBDevice::get_device() {
    return libusb_get_device(dev_handle_);
}

int LibUSBDevice::get_configuration(int *config) {
    return libusb_get_configuration(dev_handle_, config);
}

int LibUSBDevice::set_configuration(int configuration) {
    return libusb_set_configuration(dev_handle_, configuration);
}

int LibUSBDevice::claim_interface(int interface_number) {
    return libusb_claim_interface(dev_handle_, interface_number);
}

int LibUSBDevice::release_interface(int interface_number) {
    return libusb_release_interface(dev_handle_, interface_number);
}

int LibUSBDevice::set_interface_alt_setting(int interface_number, int alternate_setting) {
    return libusb_set_interface_alt_setting(dev_handle_, interface_number, alternate_setting);
}

int LibUSBDevice::clear_halt(unsigned char endpoint) {
    return libusb_clear_halt(dev_handle_, endpoint);
}

int LibUSBDevice::reset_device() {
    return libusb_reset_device(dev_handle_);
}

int LibUSBDevice::kernel_driver_active(int interface_number) {
    return libusb_kernel_driver_active(dev_handle_, interface_number);
}

int LibUSBDevice::detach_kernel_driver(int interface_number) {
    return libusb_detach_kernel_driver(dev_handle_, interface_number);
}

int LibUSBDevice::attach_kernel_driver(int interface_number) {
    return libusb_attach_kernel_driver(dev_handle_, interface_number);
}

int LibUSBDevice::set_auto_detach_kernel_driver(int enable) {
    return libusb_set_auto_detach_kernel_driver(dev_handle_, enable);
}

int LibUSBDevice::get_string_descriptor_ascii(uint8_t desc_index, unsigned char *data, int length) {
    return libusb_get_string_descriptor_ascii(dev_handle_, desc_index, data, length);
}

void LibUSBDevice::control_transfer(uint8_t bmRequestType, uint8_t bRequest, uint16_t wValue, uint16_t wIndex,
                                    unsigned char *data, uint16_t wLength, unsigned int timeout) {
    int res;
    res = libusb_control_transfer(dev_handle_, bmRequestType, bRequest, wValue, wIndex, data, wLength, timeout);
    if (res < 0) {
        throw HalConnectionException(res, libusb_error_category());
    }
}

void LibUSBDevice::bulk_transfer(unsigned char endpoint, unsigned char *data, int length, int *transferred,
                                 unsigned int timeout) {
    int res;
    res = libusb_bulk_transfer(dev_handle_, endpoint, data, length, transferred, timeout);
    if (res < 0) {
        throw HalConnectionException(res, libusb_error_category());
    }
}

void LibUSBDevice::interrupt_transfer(unsigned char endpoint, unsigned char *data, int length, int *transferred,
                                      unsigned int timeout) {
    int res;
    res = libusb_interrupt_transfer(dev_handle_, endpoint, data, length, transferred, timeout);
    if (res < 0) {
        throw HalConnectionException(res, libusb_error_category());
    }
}

} // namespace Metavision
