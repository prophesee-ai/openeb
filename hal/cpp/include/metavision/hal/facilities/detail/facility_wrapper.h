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

#ifndef METAVISION_HAL_DETAIL_FACILITY_WRAPPER_H
#define METAVISION_HAL_DETAIL_FACILITY_WRAPPER_H

#include <memory>

namespace Metavision {

class I_Facility;

/// @brief Utility class that wraps a @ref I_Facility
class FacilityWrapper {
public:
    /// @brief Constructor
    /// @note The constructor will call the @ref I_Facility::setup function
    /// @param facility
    FacilityWrapper(const std::shared_ptr<I_Facility> &facility);

    /// @brief Destructor
    /// @note The destructor will call the @ref I_Facility::teardown function
    /// @param facility
    ~FacilityWrapper();

    /// @brief Returns the wrapped facility pointer
    /// @return The wrapped facility pointer
    const std::shared_ptr<I_Facility> &facility() const;

private:
    FacilityWrapper(const FacilityWrapper &) = delete;
    FacilityWrapper &operator=(const FacilityWrapper &) = delete;

    std::shared_ptr<I_Facility> facility_;
};

} // namespace Metavision

#endif // METAVISION_HAL_DETAIL_FACILITY_WRAPPER_H
