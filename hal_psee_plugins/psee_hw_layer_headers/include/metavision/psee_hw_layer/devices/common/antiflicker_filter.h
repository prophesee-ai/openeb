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

#ifndef METAVISION_HAL_ANTIFLICKER_FILTER_H
#define METAVISION_HAL_ANTIFLICKER_FILTER_H

#include <string>
#include <map>

#include "metavision/hal/facilities/i_antiflicker_module.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_regmap_device.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

class AntiFlickerFilter : public I_AntiFlickerModule {
public:
    AntiFlickerFilter(const std::shared_ptr<TzDeviceWithRegmap> &dev,
                      const I_HW_Identification::SensorInfo &sensor_info, const std::string &sensor_prefix);

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
    uint32_t freq_to_period(const uint32_t &freq);
    std::pair<uint32_t, uint32_t> compute_invalidation(const uint32_t &max_cutoff_period, const uint32_t &clk_freq);

    std::shared_ptr<TzDeviceWithRegmap> dev_;
    std::string sensor_prefix_;
    bool is_sensor_saphir;
    std::string flag_done_;
    std::string afk_param_;

    uint32_t low_freq_{50};
    uint32_t high_freq_{520};

    uint32_t df_wait_time = 1630;
    AntiFlickerMode mode_{BAND_STOP};

    uint32_t inverted_duty_cycle_{0x8};

    uint32_t start_threshold_{6};
    uint32_t stop_threshold_{4};

    RegisterMap &regmap();
};

} // namespace Metavision

#endif // METAVISION_HAL_ANTIFLICKER_FILTER_H
