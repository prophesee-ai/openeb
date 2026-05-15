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

#ifndef METAVISION_SDK_CORE_HARDWARE_DIFF_PROCESSOR_IMPL_H
#define METAVISION_SDK_CORE_HARDWARE_DIFF_PROCESSOR_IMPL_H

#include "metavision/sdk/core/preprocessors/hardware_diff_processor.h"

namespace Metavision {

template<typename InputIt>
HardwareDiffProcessor<InputIt>::HardwareDiffProcessor(int width, int height, int8_t min_val, int8_t max_val,
                                                      bool allow_rollover) :
    EventPreprocessor<InputIt>(TensorShape({{"H", height}, {"W", width}, {"C", 1}}), BaseType::INT8),
    allow_rollover_(allow_rollover),
    min_val_(min_val),
    max_val_(max_val),
    width_(width) {}

template<typename InputIt>
void HardwareDiffProcessor<InputIt>::process_events(InputIt begin, InputIt end, RawEventFrameDiff &diff) const {
    Tensor wrapper(this->output_tensor_shape_, this->output_tensor_type_,
                   reinterpret_cast<std::byte *>(diff.get_data().data()), false);
    process_events(begin, end, wrapper);
}

template<typename InputIt>
void HardwareDiffProcessor<InputIt>::compute(const timestamp, InputIt it_begin, InputIt it_end, Tensor &tensor) const {
    auto diff = tensor.data<int8_t>();
    for (auto it = it_begin; it != it_end; ++it) {
        const unsigned int idx = it->x + it->y * width_;
        int8_t &sum_polarities = diff[idx];
        const bool should_rollover =
            (sum_polarities == min_val_ && it->p == 0) || (sum_polarities == max_val_ && it->p == 1);
        if (!should_rollover) {
            sum_polarities += (it->p == 0 ? -1 : 1);
        } else if (allow_rollover_) {
            sum_polarities = (it->p == 0 ? max_val_ : min_val_);
        }
        // else sum_polarities is saturated
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_HARDWARE_DIFF_PROCESSOR_IMPL_H
