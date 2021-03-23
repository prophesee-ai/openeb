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

#include "metavision/sdk/driver/trigger_out.h"

/*
 * By default in the FPGA:
 *  -> the pulse width is set to 1 us which is the default behavior (decided by PO).
 *  -> the period is set to 100 us by default which is the default behavior (decided by PO).
 *
 *  In this interface, the pulse width remains constant. In term of code logic,
 *  there is nothing to do as this value is set by default and never changes.
 */

namespace Metavision {

TriggerOut::TriggerOut(I_TriggerOut *i_trigger_out) : pimpl_(i_trigger_out) {}

TriggerOut::~TriggerOut() = default;

void TriggerOut::set_pulse_period(uint32_t period_us) {
    pimpl_->set_period(period_us);
}

void TriggerOut::set_duty_cycle(double period_ratio) {
    pimpl_->set_duty_cycle(period_ratio);
}

void TriggerOut::enable() {
    pimpl_->enable();
}

void TriggerOut::disable() {
    pimpl_->disable();
}

I_TriggerOut *TriggerOut::get_facility() const {
    return pimpl_;
}

} // namespace Metavision
