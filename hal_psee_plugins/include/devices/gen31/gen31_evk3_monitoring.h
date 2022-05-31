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

#ifndef METAVISION_HAL_GEN31_EVK3_MONITORING_H
#define METAVISION_HAL_GEN31_EVK3_MONITORING_H

#include "facilities/psee_monitoring.h"

namespace Metavision {

class I_HW_Register;
class TzLibUSBBoardCommand;

class Gen31Evk3Monitoring : public PseeMonitoring {
public:
    Gen31Evk3Monitoring(const std::shared_ptr<I_HW_Register> &i_hw_register,
                        const std::shared_ptr<TzLibUSBBoardCommand> &board_cmd);

    virtual int get_temperature() override;
    virtual int get_illumination() override;

private:
    std::shared_ptr<TzLibUSBBoardCommand> icmd_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN31_EVK3_MONITORING_H
