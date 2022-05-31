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

#ifndef METAVISION_SDK_DRIVER_BIASES_H
#define METAVISION_SDK_DRIVER_BIASES_H

#include <cstdint>
#include <string>

#include "metavision/hal/facilities/i_ll_biases.h"

namespace Metavision {

/// @brief Facility class to handle biases
class Biases {
public:
    /// @brief Constructor
    Biases(I_LL_Biases *i_ll_biases);

    /// @brief Destructor
    ~Biases();

    /// @brief Sets camera biases from a bias file at biases_filename
    /// @throw CameraException in case of failure. This could happen for example if given path does not exists.
    /// @param biases_filename Path to the bias file used to configure the camera
    void set_from_file(const std::string &biases_filename);

    /// @brief Save the current biases into a file
    /// @param dest_file the destination file
    void save_to_file(const std::string &dest_file) const;

    /// @brief Get corresponding facility in HAL library
    I_LL_Biases *get_facility() const;

private:
    I_LL_Biases *pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_BIASES_H
