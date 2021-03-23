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

#ifndef METAVISION_SDK_DRIVER_TRIGGER_OUT_H
#define METAVISION_SDK_DRIVER_TRIGGER_OUT_H

#include <cstdint>

#include "metavision/hal/facilities/i_trigger_out.h"

namespace Metavision {

/// @brief Trigger out signal handler class
///
/// The trigger out is a signal generator.
/// Its purpose is to synchronize two (or more) devices.
///
/// The configurable signal through this interface is a periodic 1 microsecond pulse.
class TriggerOut {
public:
    /// @brief Constructor
    TriggerOut(I_TriggerOut *i_trigger_out);

    /// @brief Destructor
    virtual ~TriggerOut();

    /// @brief Sets the trigger out signal pulse period
    ///
    /// By default, the system has a period of 100 us meaning that, when enabled, a pulse of 1 microsecond will occur
    /// every 100us.
    ///
    /// @param period_us the signal period in microseconds.
    void set_pulse_period(uint32_t period_us);

    /// @brief Sets the duty cycle of the trigger out signal i.e. the pulse duration
    ///
    /// The duty cycle represents the part of the signal during which its value is 1 (0 otherwise).
    /// Duty cycle represents the quantity pulse_width_us/period_us and thus must be in the range [0, 1].
    /// The period is set with @ref set_pulse_period.
    ///
    /// Setting a duty cycle of 0.5 (50%) means that the value of the signal is 1 during the first half of each period,
    /// and 0 during the second half.
    ///
    /// @param period_ratio Ratio representing pulse_width_us/period_us which must be in the range [0,
    /// 1] (value is clamped in this range otherwise)
    void set_duty_cycle(double period_ratio);

    /// @brief Enables the trigger out signal
    void enable();

    /// @brief Disables the trigger out signal
    void disable();

    /// @brief Get corresponding facility in HAL library
    I_TriggerOut *get_facility() const;

private:
    I_TriggerOut *pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_TRIGGER_OUT_H
