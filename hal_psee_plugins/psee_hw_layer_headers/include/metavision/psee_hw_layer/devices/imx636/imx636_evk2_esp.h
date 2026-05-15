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

#ifndef METAVISION_HAL_IMX636_EVK2_ESP_H
#define METAVISION_HAL_IMX636_EVK2_ESP_H

#include <string>
#include <map>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

class RegisterMap;
class PseeDeviceControl;
class TzDevice;

class Imx636Evk2Esp : public I_RegistrableFacility<Imx636Evk2Esp> {
public:
    Imx636Evk2Esp(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix);

    virtual bool enable(bool en);
    virtual bool is_enabled() const;
    virtual bool enable_out_th_recovery(bool en);
    virtual bool is_out_th_recovery_enabled() const;
    virtual bool enable_out_gen_last(bool en);
    virtual bool is_out_gen_last_enabled() const;
    virtual bool initialize();

private:
    std::shared_ptr<RegisterMap> register_map_;
    std::string prefix_;
};

} // namespace Metavision

#endif // METAVISION_HAL_IMX636_EVK2_ESP_H
