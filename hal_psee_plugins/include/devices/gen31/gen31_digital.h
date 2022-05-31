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

#ifndef METAVISION_HAL_GEN31_DIGITAL_H
#define METAVISION_HAL_GEN31_DIGITAL_H

#include <string>
#include <memory>

namespace Metavision {

class RegisterMap;

class Gen31Digital {
public:
    Gen31Digital(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix);

    long long get_chip_id();
    void init();
    void start();
    void stop();
    void destroy();

private:
    std::string prefix_;
    std::shared_ptr<RegisterMap> register_map_;
};

} // namespace Metavision
#endif // METAVISION_HAL_GEN31_DIGITAL_H
