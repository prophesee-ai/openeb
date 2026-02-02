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

#ifndef METAVISION_HAL_GENX320_NFL_INTERFACE_H
#define METAVISION_HAL_GENX320_NFL_INTERFACE_H

#include "metavision/hal/facilities/i_event_rate_activity_filter_module.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_nfl_driver.h"

namespace Metavision {

class GenX320NflInterface : public I_EventRateActivityFilterModule {
public:
    GenX320NflInterface(const std::shared_ptr<GenX320NflDriver> &driver);

    /// @brief Enables/disables the noise filter
    /// @param enable_filter Whether to enable the noise filtering
    bool enable(bool enable_filter);

    /// @brief Returns the noise filter state
    /// @return the noise filter state
    bool is_enabled() const;

    I_EventRateActivityFilterModule::thresholds is_thresholds_supported() const;
    bool set_thresholds(const I_EventRateActivityFilterModule::thresholds &thresholds_ev_s);
    I_EventRateActivityFilterModule::thresholds get_thresholds() const;

    I_EventRateActivityFilterModule::thresholds get_min_supported_thresholds() const;
    I_EventRateActivityFilterModule::thresholds get_max_supported_thresholds() const;

private:
    std::shared_ptr<GenX320NflDriver> driver_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_NFL_INTERFACE_H
