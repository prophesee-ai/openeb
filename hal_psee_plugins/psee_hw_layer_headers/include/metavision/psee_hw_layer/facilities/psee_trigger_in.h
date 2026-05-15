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

#ifndef METAVISION_HAL_PSEE_TRIGGER_IN_H
#define METAVISION_HAL_PSEE_TRIGGER_IN_H

#include <memory>

#include "metavision/hal/facilities/i_trigger_in.h"

namespace Metavision {

class PseeDeviceControl;

class PseeTriggerIn : public I_TriggerIn {
public:
    PseeTriggerIn(const std::shared_ptr<PseeDeviceControl> &device_control);

protected:
    const std::shared_ptr<PseeDeviceControl> &get_device_control();

private:
    void setup() override final;

    std::shared_ptr<PseeDeviceControl> device_control_;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_TRIGGER_IN_H
