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

#ifndef METAVISION_HAL_PSEE_MONITORING_H
#define METAVISION_HAL_PSEE_MONITORING_H

#include <memory>

#include "metavision/hal/facilities/i_monitoring.h"
#include "metavision/hal/utils/hal_exception.h"

namespace Metavision {

class I_HW_Register;
class PseeMonitoring : public I_Monitoring {
public:
    PseeMonitoring(const std::shared_ptr<I_HW_Register> &i_hw_register);

    virtual int get_pixel_dead_time() override {
        throw HalException(HalErrorCode::OperationNotImplemented);
    }

protected:
    const std::shared_ptr<I_HW_Register> &get_hw_register() const;

private:
    std::shared_ptr<I_HW_Register> i_hw_register_;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_MONITORING_H
