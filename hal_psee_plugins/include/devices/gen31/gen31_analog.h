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

#ifndef METAVISION_HAL_GEN31_ANALOG_H
#define METAVISION_HAL_GEN31_ANALOG_H

#include <string>
#include <memory>

namespace Metavision {

class RegisterMap;

class Gen31Analog {
public:
    Gen31Analog(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix, bool is_em);

    void init();
    void start();
    void stop();
    void destroy();
    void analog_td_rstn(bool rstn);
    void analog_em_rstn(bool rstn);
    void enable_lifo_measurement();
    uint32_t lifo_counter();

private:
    std::string prefix_;
    std::shared_ptr<RegisterMap> register_map_;
    bool is_em_ = false;
};

} // namespace Metavision
#endif // METAVISION_HAL_GEN31_ANALOG_H
