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

#ifndef METAVISION_HAL_SAMPLE_ERC_H
#define METAVISION_HAL_SAMPLE_ERC_H

#include <memory>
#include <string>

#include <metavision/hal/facilities/i_erc_module.h>

class SampleUSBConnection;

/// @brief Interface for Event Rate Controller (ERC) commands
///
/// This class is the implementation of HAL's facility @ref Metavision::I_ErcModule
class SampleErc : public Metavision::I_ErcModule {
public:
    SampleErc(std::shared_ptr<SampleUSBConnection> usb_connection);
    virtual bool enable(bool en) override;
    virtual bool is_enabled() const override;
    virtual void erc_from_file(const std::string &) override;
    virtual uint32_t get_count_period() const override;
    virtual bool set_cd_event_count(uint32_t count) override;
    virtual uint32_t get_min_supported_cd_event_count() const override;
    virtual uint32_t get_max_supported_cd_event_count() const override;
    virtual uint32_t get_cd_event_count() const override;

private:
    static constexpr uint32_t CD_EVENT_COUNT_DEFAULT = 10000;
    static constexpr uint32_t CD_EVENT_COUNT_MAX     = 640000;

    std::shared_ptr<SampleUSBConnection> usb_connection_;
};


#endif // METAVISION_HAL_SAMPLE_ERC_H
