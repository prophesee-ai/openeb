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

#ifndef METAVISION_SDK_CORE_CONTRAST_MAP_GENERATION_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_CONTRAST_MAP_GENERATION_ALGORITHM_IMPL_H

namespace Metavision {

template<typename InputIt>
void ContrastMapGenerationAlgorithm::process_events(InputIt it_begin, InputIt it_end) {
    for (auto it = it_begin; it != it_end; ++it) {
        process_event(*it);
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_CONTRAST_MAP_GENERATION_ALGORITHM_IMPL_H
