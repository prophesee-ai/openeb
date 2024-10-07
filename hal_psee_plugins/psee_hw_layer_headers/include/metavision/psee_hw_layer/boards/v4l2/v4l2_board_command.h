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

#ifndef METAVISION_HAL_V4L2_BOARD_COMMAND_H
#define METAVISION_HAL_V4L2_BOARD_COMMAND_H

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

#include "metavision/psee_hw_layer/boards/treuzell/board_command.h"

namespace Metavision {

class TzCtrlFrame;
class BoardCommand;
class DataTransfer;

// FIXME: should not be a DeviceControl, just way to interact with v4l2.
class V4L2DeviceControl;

class V4L2BoardCommand : public virtual BoardCommand {
public:
    V4L2BoardCommand() = delete;
    V4L2BoardCommand(std::string device_path);
    ~V4L2BoardCommand();

    long get_board_speed() override;

    std::string get_serial() override;
    std::string get_name() override;
    std::string get_manufacturer() override;
    time_t get_build_date() override;
    uint32_t get_version() override;

    uint32_t get_device_count() override;
    std::vector<uint32_t> read_device_register(uint32_t device, uint32_t address, int nval = 1) override;
    void write_device_register(uint32_t device, uint32_t address, const std::vector<uint32_t> &val) override;

    // @brief Create a new RawDataProducer object to stream the currently opened device
    std::unique_ptr<DataTransfer::RawDataProducer> build_raw_data_producer(uint32_t raw_event_size_bytes) override;

    void transfer_tz_frame(TzCtrlFrame &req) override;
    std::shared_ptr<V4L2DeviceControl> get_device_control();

private:
    std::shared_ptr<V4L2DeviceControl> device_;
    std::mutex tz_control_mutex_;
    std::string manufacturer;
    std::string product;
    time_t build_date;
    uint32_t version;
    int sensor_fd_;
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_LIBUSB_BOARD_COMMAND_H
