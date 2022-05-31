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

#ifndef METAVISION_HAL_GEN3_LL_BIASES_H
#define METAVISION_HAL_GEN3_LL_BIASES_H

#include "metavision/hal/facilities/i_ll_biases.h"

namespace Metavision {

class PseeLibUSBBoardCommand;

class Gen3_LL_Biases : public I_LL_Biases {
public:
    Gen3_LL_Biases(const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd);
    ~Gen3_LL_Biases() override;

    virtual bool set(const std::string &bias_name, int bias_value) override;
    virtual int get(const std::string &bias_name) override;
    virtual std::map<std::string, int> get_all_biases() override;

private:
    struct Private;
    std::unique_ptr<Private> pimpl_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN3_LL_BIASES_H
