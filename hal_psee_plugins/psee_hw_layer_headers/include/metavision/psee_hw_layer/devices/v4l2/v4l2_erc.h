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

#ifndef METAVISION_HAL_V4L2_ERC_H
#define METAVISION_HAL_V4L2_ERC_H

#include <string>
#include <map>

#include "metavision/hal/facilities/i_erc_module.h"
#include "metavision/psee_hw_layer/boards/v4l2/v4l2_controls.h"

namespace Metavision {

class RegisterMap;

class V4L2Erc : public I_ErcModule {
public:
    V4L2Erc(std::shared_ptr<V4L2Controls> controls);

    virtual bool enable(bool en) override;
    virtual bool is_enabled() const override;
    virtual uint32_t get_count_period() const override;
    virtual bool set_cd_event_count(uint32_t count) override;
    virtual uint32_t get_min_supported_cd_event_count() const override;
    virtual uint32_t get_max_supported_cd_event_count() const override;
    virtual uint32_t get_cd_event_count() const override;
    virtual void erc_from_file(const std::string &) override;

private:

    std::shared_ptr<V4L2Controls> controls_;

    bool wait_status();
    bool dfifo_disable_bypass_dyn();
    std::map<std::string, uint32_t> is_powered_up_dyn();
    bool activate_dyn(const uint32_t &td_target_vx_cnt);
    bool set_evt_rate_dyn(uint32_t ref_period, uint32_t td_target_vx_cnt, uint32_t adr_delayed,
                          uint32_t dfifo_non_td_area);
};

} // namespace Metavision

#endif // METAVISION_HAL_V4L2_ERC_H
