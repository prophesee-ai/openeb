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

#ifndef METAVISION_HAL_BIAS_SETTINGS_H
#define METAVISION_HAL_BIAS_SETTINGS_H

#include <string>

namespace Metavision {

struct bias_settings {
    std::string name;
    int min_allowed_offset;
    int max_allowed_offset;
    int min_recommended_offset;
    int max_recommended_offset;
    bool modifiable;
};

} // namespace Metavision

#endif // METAVISION_HAL_BIAS_SETTINGS_H