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
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "metavision/hal/utils/hal_log.h"
#include "boards/v4l2/v4l2_device.h"
#include "boards/v4l2/v4l2_data_transfer.h"
#include "metavision/psee_hw_layer/boards/v4l2/v4l2_board_command.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

namespace Metavision {

V4L2BoardCommand::V4L2BoardCommand(std::string device_path) {
    struct stat st;
    device_ = std::make_shared<V4L2DeviceControl>(device_path);
    sensor_fd_ = device_->get_sensor_entity()->fd;
}

V4L2BoardCommand::~V4L2BoardCommand() {}

std::string V4L2BoardCommand::get_name() {
    return product;
}

std::string V4L2BoardCommand::get_manufacturer() {
    return manufacturer;
}

time_t V4L2BoardCommand::get_build_date() {
    return build_date;
}

uint32_t V4L2BoardCommand::get_version() {
    // TODO: get serial number from media_device,
    // see https://www.kernel.org/doc/html/v4.9/media/uapi/mediactl/media-ioc-device-info.html
    // Not available yet with Thor96 setup
    return version;
}

// TODO: make no sense for V4L2 devices
long V4L2BoardCommand::get_board_speed() {
    return 0;
}

// TODO: does it make sense ? not sure
std::string V4L2BoardCommand::get_serial() {
    // TODO: get serial number from media_device,
    // see https://www.kernel.org/doc/html/v4.9/media/uapi/mediactl/media-ioc-device-info.html
    // Not available yet with Thor96 setup
    return "v4l2_device";
}

// TODO: keep it internal in tz (libusb) board command, and expose generic
// functions allowing to cover all use cases (v4l2, usb, ethernet, whatever).
void V4L2BoardCommand::transfer_tz_frame(TzCtrlFrame &req) {}

unsigned int V4L2BoardCommand::get_device_count() {
    // TODO: extract media device topology and count entities ?
    return 1;
}

std::vector<uint32_t> V4L2BoardCommand::read_device_register(uint32_t device, uint32_t address, int nval) {
    std::vector<uint32_t> res;
    struct v4l2_dbg_match match      = {0};
    struct v4l2_dbg_register get_reg = {0};
    int i, retval;

    match.type    = V4L2_CHIP_MATCH_BRIDGE;
    match.addr    = 0;
    get_reg.match = match;

    for (i = 0; i < nval; i += 4) {
        get_reg.reg = address + i;
        if ((retval = ioctl(sensor_fd_, VIDIOC_DBG_G_REGISTER, &get_reg)) < 0)
            throw std::runtime_error("ioctl: VIDIOC_DBG_G_REGISTER failed to read register");

        res.push_back(get_reg.val);
    }

    return res;
}

void V4L2BoardCommand::write_device_register(uint32_t device, uint32_t address, const std::vector<uint32_t> &val) {
    struct v4l2_dbg_match match      = {0};
    struct v4l2_dbg_register set_reg = {0};
    int i, retval;

    match.type    = V4L2_CHIP_MATCH_BRIDGE;
    match.addr    = 0;
    set_reg.match = match;

    for (auto value : val) {
        set_reg.reg = address;
        set_reg.val = value;
        if ((retval = ioctl(sensor_fd_, VIDIOC_DBG_S_REGISTER, &set_reg)) < 0)
            throw std::runtime_error("ioctl: VIDIOC_DBG_S_REGISTER failed to write register");
        address += 4;
    }
}

std::shared_ptr<V4L2DeviceControl> V4L2BoardCommand::get_device_control() {
    return device_;
}

std::unique_ptr<DataTransfer::RawDataProducer>
    V4L2BoardCommand::build_raw_data_producer(uint32_t raw_event_size_bytes) {
    // TODO: based on the /dev/mediaX device (not available with thor96 psee eb driver), extract the pipeline topology,
    // extract the /dev/videoX associated entity, and populate the DataTransfer with it.
    // Right now, we'll just hard code it to /dev/video0 ¯\_(ツ)_/¯
    // more details in: https://github.com/gjasny/v4l-utils/blob/master/utils/media-ctl/media-ctl.c#L526

    // If the environment set a heap, us it, otherwise, use the driver's allocator
    if (std::getenv("V4L2_HEAP"))
        return std::make_unique<V4l2DataTransfer>(device_->get_video_entity()->fd, raw_event_size_bytes, "/dev/dma_heap",
                                                  std::getenv("V4L2_HEAP"));
    else
        return std::make_unique<V4l2DataTransfer>(device_->get_video_entity()->fd, raw_event_size_bytes);
}

} // namespace Metavision
