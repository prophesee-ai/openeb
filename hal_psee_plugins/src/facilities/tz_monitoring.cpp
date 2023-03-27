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

#include "metavision/hal/utils/hal_exception.h"
#include "metavision/psee_hw_layer/facilities/tz_monitoring.h"

namespace Metavision {

TzMonitoring::TzMonitoring(const std::shared_ptr<TemperatureProvider> &temp,
                           const std::shared_ptr<IlluminationProvider> &illu,
                           const std::shared_ptr<PixelDeadTimeProvider> &pixel_dead_time_provider) :
    temp_(temp), illu_(illu), pixel_dead_time_provider_(pixel_dead_time_provider) {}

int TzMonitoring::get_temperature() {
    if (!temp_) {
        throw HalException(HalErrorCode::OperationNotImplemented);
    }
    return temp_->get_temperature();
}

int TzMonitoring::get_illumination() {
    if (!illu_) {
        throw HalException(HalErrorCode::OperationNotImplemented);
    }
    return illu_->get_illumination();
}

int TzMonitoring::get_pixel_dead_time() {
    if (!pixel_dead_time_provider_) {
        throw HalException(HalErrorCode::OperationNotImplemented);
    }
    return pixel_dead_time_provider_->get_pixel_dead_time();
}

} // namespace Metavision
