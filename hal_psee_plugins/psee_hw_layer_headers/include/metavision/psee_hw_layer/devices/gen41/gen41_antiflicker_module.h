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

#ifndef METAVISION_HAL_GEN41_ANTIFLICKER_MODULE_H
#define METAVISION_HAL_GEN41_ANTIFLICKER_MODULE_H

#include <string>
#include <map>

#include "metavision/hal/facilities/i_antiflicker_module.h"

namespace Metavision {

class RegisterMap;

class Gen41AntiFlickerModule : public I_AntiFlickerModule {
public:
    Gen41AntiFlickerModule(const std::shared_ptr<RegisterMap> &regmap, const std::string &sensor_prefix);

    virtual bool enable(bool b) override;
    virtual bool is_enabled() override;

    virtual bool set_frequency_band(uint32_t low_freq, uint32_t high_freq) override;
    virtual uint32_t get_band_low_frequency() const override;
    virtual uint32_t get_band_high_frequency() const override;
    virtual uint32_t get_min_supported_frequency() const override;
    virtual uint32_t get_max_supported_frequency() const override;
    virtual bool set_filtering_mode(I_AntiFlickerModule::AntiFlickerMode mode) override;
    virtual AntiFlickerMode get_filtering_mode() const override;

    virtual bool set_duty_cycle(float duty_cycle) override;
    virtual float get_duty_cycle() const override;
    virtual float get_min_supported_duty_cycle() const override;
    virtual float get_max_supported_duty_cycle() const override;

    virtual bool set_start_threshold(uint32_t threshold) override;
    virtual bool set_stop_threshold(uint32_t threshold) override;
    virtual uint32_t get_start_threshold() const override;
    virtual uint32_t get_stop_threshold() const override;
    virtual uint32_t get_min_supported_start_threshold() const override;
    virtual uint32_t get_max_supported_start_threshold() const override;
    virtual uint32_t get_min_supported_stop_threshold() const override;
    virtual uint32_t get_max_supported_stop_threshold() const override;

private:
    bool reset();

    std::shared_ptr<RegisterMap> register_map_;
    std::string sensor_prefix_;

    uint32_t low_freq_{50};
    uint32_t high_freq_{520};
    AntiFlickerMode mode_{BAND_STOP};

    uint32_t inverted_duty_cycle_{0x8};

    uint32_t start_threshold_{6};
    uint32_t stop_threshold_{4};
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN41_ANTIFLICKER_MODULE_H
