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

#ifndef METAVISION_HAL_I_TRIGGER_OUT_H
#define METAVISION_HAL_I_TRIGGER_OUT_H

#include <cstdint>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Internal interface for trigger out signal configuration
/// @note The trigger out signal is a binary signal
class I_TriggerOut : public I_RegistrableFacility<I_TriggerOut> {
public:
    /// @brief Gets the trigger out signal period (in us)
    /// @return the period set (in us)
    virtual uint32_t get_period() const = 0;

    /// @brief Sets the trigger out signal period (in us)
    /// @param period_us the period to set (in us)
    /// @return true on success
    virtual bool set_period(uint32_t period_us) = 0;

    /// @brief Gets the duty cycle of the trigger out signal
    /// @return the ratio representing pulse_width_us/period_us
    virtual double get_duty_cycle() const = 0;

    /// @brief Sets the duty cycle of the trigger out signal i.e. the pulse duration
    ///
    /// The duty cycle represents the part of the signal, during which its value is 1 (0 otherwise).
    /// Duty cycle represents the quantity pulse_width_us/period_us and thus must be in the range [0, 1].
    /// The period is set with @ref set_period.
    ///
    /// Setting a duty cycle of 0.5 (50%) means that the value of the signal is 1 during the first half of each period,
    /// and 0 during the second half.
    ///
    /// @param period_ratio the ratio representing pulse_width_us/period_us which must be in the range [0,1]
    /// (value is clamped in this range otherwise)
    /// @return true on success
    virtual bool set_duty_cycle(double period_ratio) = 0;

    /// @brief Enables the trigger out
    /// @return true if trigger was successfully enabled, false otherwise
    virtual bool enable() = 0;

    /// @brief Disables the trigger out
    virtual bool disable() = 0;

    /// @brief Checks if trigger out is enabled
    /// @return true if trigger out is enabled, False otherwise
    virtual bool is_enabled() = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_TRIGGER_OUT_H
