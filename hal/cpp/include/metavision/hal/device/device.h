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

#ifndef METAVISION_HAL_DEVICE_H
#define METAVISION_HAL_DEVICE_H

#include <string>
#include <memory>
#include <unordered_map>
#include <typeinfo>
#include <functional>

#include "metavision/hal/facilities/i_registrable_facility.h"
#include "metavision/hal/facilities/detail/facility_wrapper.h"

namespace Metavision {

/// @brief Device abstraction
class Device {
public:
    /// @brief Returns facility
    template<typename FacilityType>
    FacilityType *get_facility() {
        static_assert(std::is_base_of<I_RegistrableFacility<FacilityType>, FacilityType>::value,
                      "Unable to get facility of unregistrable facility type.");
        auto it = facilities_.find(I_RegistrableFacility<FacilityType>::class_registration_info());
        if (it != facilities_.end()) {
            return dynamic_cast<FacilityType *>(it->second->facility().get());
        }
        return nullptr;
    }

    /// @brief Returns facility
    template<typename FacilityType>
    const FacilityType *get_facility() const {
        static_assert(std::is_base_of<I_RegistrableFacility<FacilityType>, FacilityType>::value,
                      "Unable to get facility of unregistrable facility type.");
        auto it = facilities_.find(I_RegistrableFacility<FacilityType>::class_registration_info());
        if (it != facilities_.end()) {
            return dynamic_cast<const FacilityType *>(it->second->facility().get());
        }
        return nullptr;
    }

    /// @brief Move constructor
    /// @param dev Device from which this instance will be moved from
    Device(Device &&dev) = default;

    /// @brief Move operator
    /// @param dev Device from which a new instance will be moved from
    /// @return A device moved from @p dev
    Device &operator=(Device &&dev) = default;

private:
    // A device can't be copied
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    /// @brief Registers facility to device
    void register_facility(std::unique_ptr<FacilityWrapper> p);

    // Can be called from DeviceBuilder
    template<typename FacilityInputIterator>
    Device(FacilityInputIterator begin, FacilityInputIterator end) {
        for (auto it = begin; it != end; ++it) {
            register_facility(std::move(*it));
        }
    }

    std::unordered_map<size_t, std::unique_ptr<FacilityWrapper>> facilities_;

    friend class DeviceBuilder;
};

} // namespace Metavision

#endif // METAVISION_HAL_DEVICE_H
