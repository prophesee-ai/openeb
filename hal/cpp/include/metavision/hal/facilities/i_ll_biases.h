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

#ifndef METAVISION_HAL_I_LL_BIASES_H
#define METAVISION_HAL_I_LL_BIASES_H

#include <string>
#include <map>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Interface facility for Lower Level Biases
class I_LL_Biases : public I_RegistrableFacility<I_LL_Biases> {
public:
    /// @brief Sets bias value
    /// @param bias_name Bias to set
    /// @param bias_value Value to set the bias to
    /// @return true on success
    virtual bool set(const std::string &bias_name, int bias_value) = 0;

    /// @brief Gets bias value
    /// @param bias_name Name of the bias whose value to get
    /// @return The bias value
    virtual int get(const std::string &bias_name) = 0;

    /// @brief Gets all biases values
    /// @return A map containing the biases values
    virtual std::map<std::string, int> get_all_biases() = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_LL_BIASES_H
