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

#ifndef METAVISION_HAL_TZ_ISSD_DEVICE_H
#define METAVISION_HAL_TZ_ISSD_DEVICE_H

#include "metavision/psee_hw_layer/devices/treuzell/tz_regmap_device.h"

namespace Metavision {

class RegisterOperation;
struct Issd;

class TzIssdDevice : public virtual TzDeviceWithRegmap {
public:
    TzIssdDevice(const Issd &issd);
    virtual ~TzIssdDevice();

    virtual void start() override;
    virtual void stop() override;

protected:
    virtual void initialize() override;
    virtual void destroy() override;

private:
    void ApplyRegisterOperationSequence(const std::vector<RegisterOperation> sequence);
    void ApplyRegisterOperation(const RegisterOperation operation);
    const Issd &issd;
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_ISSD_DEVICE_H
