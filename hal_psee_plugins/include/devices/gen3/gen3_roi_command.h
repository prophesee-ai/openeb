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

#ifndef METAVISION_HAL_GEN3_ROI_COMMAND_H
#define METAVISION_HAL_GEN3_ROI_COMMAND_H

#include <vector>
#include <cstdint>

#include "facilities/psee_roi.h"

namespace Metavision {

class PseeLibUSBBoardCommand;

class Gen3ROICommand : public PseeROI {
public:
    Gen3ROICommand(int width, int height, const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd);

    virtual void enable(bool state) override;
    virtual void write_ROI(const std::vector<unsigned int> &vroiparams) override;

private:
    void reset_to_full_roi();

private:
    std::shared_ptr<PseeLibUSBBoardCommand> icmd_;
    uint32_t base_sensor_address_;
    std::vector<uint32_t> roi_save_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN3_ROI_COMMAND_H
