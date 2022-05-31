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

#ifndef METAVISION_HAL_GEN31_EVK2_INTERFACE_CONTROL_H
#define METAVISION_HAL_GEN31_EVK2_INTERFACE_CONTROL_H

#include <memory>

namespace Metavision {

class RegisterMap;

class Gen31Evk2InterfaceControl {
public:
    Gen31Evk2InterfaceControl(const std::shared_ptr<RegisterMap> &regmap);

    void enable(bool enable);
    void bypass(bool bypass);
    void gen_last(bool enable);

private:
    std::shared_ptr<RegisterMap> register_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN31_EVK2_INTERFACE_CONTROL_H
