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

#ifndef METAVISION_HAL_GENX320_ERC_H
#define METAVISION_HAL_GENX320_ERC_H

#include <string>
#include <map>

#include "metavision/hal/facilities/i_erc_module.h"

namespace Metavision {

class RegisterMap;

class GenX320Erc : public I_ErcModule {
public:
    GenX320Erc(const std::shared_ptr<RegisterMap> &regmap);

    virtual bool enable(bool en) override;
    virtual bool is_enabled() override;
    virtual uint32_t get_count_period() const override;
    virtual bool set_cd_event_count(uint32_t count) override;
    virtual uint32_t get_min_supported_cd_event_count() const override;
    virtual uint32_t get_max_supported_cd_event_count() const override;
    virtual uint32_t get_cd_event_count() override;
    virtual void erc_from_file(const std::string &) override;

private:
    static constexpr uint32_t CD_EVENT_COUNT_DEFAULT = 1000;
    static constexpr uint32_t CD_EVENT_COUNT_MAX     = 20000;
    static constexpr uint32_t REF_PERIOD             = 100;

    std::shared_ptr<RegisterMap> register_map_;
    uint32_t cd_event_count_shadow_{CD_EVENT_COUNT_DEFAULT};

    bool wait_status();
    bool dfifo_disable_bypass_dyn();
    std::map<std::string, uint32_t> is_powered_up_dyn();
    bool activate_dyn(const uint32_t &td_target_vx_cnt);
    bool set_evt_rate_dyn(uint32_t ref_period, uint32_t td_target_vx_cnt, uint32_t adr_delayed,
                          uint32_t dfifo_non_td_area);
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_ERC_H
