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

#ifndef METAVISION_HAL_IMX636_BIASES_H
#define METAVISION_HAL_IMX636_BIASES_H

#include <string>
#include <map>

#include "metavision/hal/facilities/i_ll_biases.h"

namespace Metavision {

static constexpr uint32_t BIAS_CONF = 0x11A10000;

class Imx636LLBias {
public:
    Imx636LLBias(bool modifiable, std::string register_name, int sensor_offset, int current_value, int factory_default,
                 int min_offset, int max_offset);
    ~Imx636LLBias();
    const std::string &get_register_name() const;
    bool is_modifiable() const;
    int get_min_offset();
    int get_max_offset();
    int get_current_offset();
    void set_current_offset(const int val);
    int get_current_value();
    void set_current_value(const int val);
    int get_factory_default_value();
    void set_factory_default_value(const int val);
    void display_bias();

private:
    std::string register_name_;
    bool modifiable_;
    int current_value_;
    int current_offset_;
    int factory_default_;
    int min_offset_;
    int max_offset_;
};

} // namespace Metavision

#endif // METAVISION_HAL_IMX636_BIASES_H