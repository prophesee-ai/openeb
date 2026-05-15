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
#include "metavision/hal/utils/detail/hal_log_impl.h"
#include "metavision/hal/facilities/i_events_stream.h"

#include "metavision/psee_hw_layer/boards/treuzell/board_command.h"
#include "metavision/psee_hw_layer/boards/v4l2/v4l2_board_command.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/hal/utils/device_builder.h"

#include "metavision/psee_hw_layer/devices/v4l2/v4l2_ll_biases.h"
#include "metavision/psee_hw_layer/devices/v4l2/v4l2_erc.h"
#include "metavision/psee_hw_layer/devices/v4l2/v4l2_roi_interface.h"
#include "metavision/psee_hw_layer/devices/v4l2/v4l2_crop.h"

#include "boards/v4l2/v4l2_device.h"
#include "boards/v4l2/v4l2_hardware_identification.h"

#include "devices/v4l2/v4l2_device_builder.h"
#include "utils/make_decoder.h"

#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <linux/v4l2-controls.h>
#include <fcntl.h>

namespace Metavision {

bool V4L2DeviceBuilder::build_device(std::shared_ptr<BoardCommand> cmd, DeviceBuilder &device_builder,
                                     const DeviceConfig &config) {
    auto v4l2cmd = std::dynamic_pointer_cast<V4L2BoardCommand>(cmd);

    auto ctrl              = v4l2cmd->get_device_control();
    auto software_info     = device_builder.get_plugin_software_info();
    auto hw_identification = device_builder.add_facility(std::make_unique<V4l2HwIdentification>(ctrl, software_info));

    try {
        size_t raw_size_bytes = 0;
        auto format           = StreamFormat(hw_identification->get_current_data_encoding_format());
        auto decoder          = make_decoder(device_builder, format, raw_size_bytes, false);
        device_builder.add_facility(std::make_unique<I_EventsStream>(v4l2cmd->build_raw_data_producer(raw_size_bytes),
                                                                     hw_identification, decoder, ctrl));
    } catch (std::exception &e) { MV_HAL_LOG_WARNING() << "System can't stream:" << e.what(); }

    auto controls = ctrl->get_controls();

    if (controls->has("bias")) {
        MV_HAL_LOG_TRACE() << "Found BIAS controls\n";
        auto sensor_info = hw_identification->get_sensor_info();
        bool relative    = false;
        if (sensor_info.name_ == "IMX636") {
            relative = true;
        }
        device_builder.add_facility(std::make_unique<V4L2LLBiases>(config, controls, relative));
    }

    if (controls->has("erc")) {
        MV_HAL_LOG_TRACE() << "Found ERC controls";
        device_builder.add_facility(std::make_unique<V4L2Erc>(controls));
    }

    // check if has roi v4l2 control OR if crop ioctl is supported
    if (controls->has("roi")) {
        MV_HAL_LOG_TRACE() << "Found ROI controls";
        device_builder.add_facility(std::make_unique<V4L2RoiInterface>(ctrl));
    } else if (ctrl->can_crop(ctrl->get_sensor_entity()->fd)) {
        device_builder.add_facility(std::make_unique<V4L2Crop>(ctrl));
    }

    return true;
}

} // namespace Metavision
