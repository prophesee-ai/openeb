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

#include "metavision/psee_hw_layer/devices/psee-video/tz_psee_video.h"
#include "metavision/psee_hw_layer/boards/treuzell/board_command.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "metavision/hal/utils/hal_log.h"
#include "devices/treuzell/tz_device_builder.h"

namespace Metavision {

std::shared_ptr<TzDevice> TzPseeVideo::build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                             std::shared_ptr<TzDevice> parent) {
    auto variants = {
        // clang-format off
        "psee,video_gen3.1",
        "psee,video_gen4",
        "psee,video_gen4.1",
        "psee,video_imx636",
        "psee,video_saphir",
        "psee,video_rdk2_imx636",
        // clang-format on
    };

    for (auto &variant : variants) {
        try {
            auto method = TzRegisterBuildMethod::recall(variant);
            if (!method.first) {
                // The implementation for this variant is not included in the current plugin
                continue;
            }
            if (method.second && !method.second(cmd, dev_id)) {
                // If there is a Check_Fun returning false, it's not the right implementation
                continue;
            }
            MV_HAL_LOG_TRACE() << "Building PseeVideo" << variant << "variant";
            return method.first(cmd, dev_id, parent);
        } catch (std::system_error &e) { MV_HAL_LOG_TRACE() << "PseeVideo" << variant << "threw" << e.what(); }
    }
    MV_HAL_LOG_TRACE() << "Building generic PseeVideo";
    return std::make_shared<TzPseeVideo>(cmd, dev_id, parent);
}
static TzRegisterBuildMethod method("psee,video", TzPseeVideo::build);

} // namespace Metavision
