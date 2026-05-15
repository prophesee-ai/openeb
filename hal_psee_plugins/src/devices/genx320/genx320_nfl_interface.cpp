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

#include <cmath>
#include "metavision/psee_hw_layer/devices/genx320/genx320_nfl_interface.h"

namespace Metavision {

GenX320NflInterface::GenX320NflInterface(const std::shared_ptr<GenX320NflDriver> &driver) : driver_(driver) {}

bool GenX320NflInterface::enable(bool enable_filter) {
    return driver_->enable(enable_filter);
}

bool GenX320NflInterface::is_enabled() const {
    return driver_->is_enabled();
}

I_EventRateActivityFilterModule::thresholds GenX320NflInterface::is_thresholds_supported() const {
    return driver_->is_thresholds_supported();
}

bool GenX320NflInterface::set_thresholds(const I_EventRateActivityFilterModule::thresholds &thresholds_ev_s) {
    return driver_->set_thresholds(thresholds_ev_s);
}

I_EventRateActivityFilterModule::thresholds GenX320NflInterface::get_thresholds() const {
    return driver_->get_thresholds();
}

I_EventRateActivityFilterModule::thresholds GenX320NflInterface::get_min_supported_thresholds() const {
    return driver_->get_min_supported_thresholds();
}

I_EventRateActivityFilterModule::thresholds GenX320NflInterface::get_max_supported_thresholds() const {
    return driver_->get_max_supported_thresholds();
}

} // namespace Metavision
