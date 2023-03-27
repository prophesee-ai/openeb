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

#ifndef METAVISION_HAL_GEN41_ERC_H
#define METAVISION_HAL_GEN41_ERC_H

#include <string>
#include <map>

#include "metavision/psee_hw_layer/facilities/psee_erc.h"

namespace Metavision {

class RegisterMap;
class PseeDeviceControl;
class TzDevice;

class Gen41Erc : public PseeErc {
public:
    Gen41Erc(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix,
             std::shared_ptr<TzDevice> tzDev = nullptr);

    virtual bool enable(bool en) override;
    virtual bool is_enabled() override;
    virtual void initialize() override;
    virtual void erc_from_file(const std::string &) override;
    virtual uint32_t get_count_period() const override;
    virtual bool set_cd_event_count(uint32_t count) override;
    virtual uint32_t get_min_supported_cd_event_count() const override;
    virtual uint32_t get_max_supported_cd_event_count() const override;
    virtual uint32_t get_cd_event_count() override;

    void set_device_control(const std::shared_ptr<PseeDeviceControl> &device_control);

private:
    static constexpr uint32_t CD_EVENT_COUNT_DEFAULT = 10000;
    static constexpr uint32_t CD_EVENT_COUNT_MAX     = 640000;

    std::shared_ptr<RegisterMap> register_map_;
    std::shared_ptr<PseeDeviceControl> dev_ctrl_;
    std::shared_ptr<TzDevice> tzDev_;
    uint32_t cd_event_count_shadow_{CD_EVENT_COUNT_DEFAULT};
    std::map<std::string, std::map<uint32_t, std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>> lut_configs;
    std::string prefix_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN41_ERC_H
