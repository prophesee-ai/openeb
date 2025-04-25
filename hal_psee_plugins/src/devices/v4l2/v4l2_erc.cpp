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

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdint>

#include "metavision/psee_hw_layer/devices/v4l2/v4l2_erc.h"
#include "metavision/psee_hw_layer/boards/v4l2/v4l2_controls.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"

namespace Metavision {

V4L2Erc::V4L2Erc(std::shared_ptr<V4L2Controls> controls) : controls_(controls) {
    // reset all erc controls to default values.
    controls_->foreach ([&](V4L2Controls::V4L2Control &ctrl) {
        auto name = std::string(ctrl.query_.name);
        // skip non erc controls
        if (name.find("erc_") != 0) {
            return 0;
        }
        ctrl.reset();
        return 0;
    });
}

bool V4L2Erc::enable(bool en) {
    auto ctrl = controls_->get("erc_enable");
    int ret;

    ret = ctrl.set_bool(en);
    if (ret != 0) {
        MV_HAL_LOG_ERROR() << "Failed to set erc_enable Control value to" << en;
        return false;
    }
    MV_HAL_LOG_INFO() << "Set erc_enable Control value to" << en;

    return true;
}

bool V4L2Erc::is_enabled() const {
    auto ctrl = controls_->get("erc_enable");
    return *ctrl.get_bool() == 1;
}

void V4L2Erc::erc_from_file(const std::string &file_path) {
    throw std::runtime_error("ERC configuration from file not implemented");
}

uint32_t V4L2Erc::get_count_period() const {
    return 1000000;
}

bool V4L2Erc::set_cd_event_count(uint32_t count) {
    auto ctrl = controls_->get("erc_rate");
    MV_HAL_LOG_INFO() << "Set erc_rate Control value to" << count;
    return ctrl.set_int64(count) == 0;
}

uint32_t V4L2Erc::get_min_supported_cd_event_count() const {
    auto ctrl = controls_->get("erc_rate");
    return (uint32_t)*ctrl.get_min<int64_t>();
}

uint32_t V4L2Erc::get_max_supported_cd_event_count() const {
    auto ctrl = controls_->get("erc_rate");
    return (uint32_t)*ctrl.get_max<int64_t>();
}

uint32_t V4L2Erc::get_cd_event_count() const {
    auto ctrl = controls_->get("erc_rate");
    return *ctrl.get_int64();
}

} // namespace Metavision
