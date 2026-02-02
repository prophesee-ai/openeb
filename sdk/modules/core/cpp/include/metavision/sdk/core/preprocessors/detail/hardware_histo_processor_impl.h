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

#ifndef METAVISION_SDK_CORE_HARDWARE_HISTO_PROCESSOR_IMPL_H
#define METAVISION_SDK_CORE_HARDWARE_HISTO_PROCESSOR_IMPL_H

#include <assert.h>

#include "metavision/sdk/core/preprocessors/hardware_histo_processor.h"

namespace Metavision {

template<typename InputIt>
HardwareHistoProcessor<InputIt>::HardwareHistoProcessor(int width, int height, uint8_t neg_saturation,
                                                        uint8_t pos_saturation) :
    EventPreprocessor<InputIt>(TensorShape({{"H", height}, {"W", width}, {"C", 2}}), BaseType::UINT8),
    sum_max_neg_(neg_saturation),
    sum_max_pos_(pos_saturation),
    width_(width) {}

template<typename InputIt>
void HardwareHistoProcessor<InputIt>::process_events(InputIt begin, InputIt end, RawEventFrameHisto &histo) const {
    Tensor wrapper(this->output_tensor_shape_, this->output_tensor_type_,
                   reinterpret_cast<std::byte *>(histo.get_data().data()), false);
    process_events(0, begin, end, wrapper);
}

template<typename InputIt>
void HardwareHistoProcessor<InputIt>::compute(const timestamp, InputIt it_begin, InputIt it_end, Tensor &tensor) const {
    auto histo = tensor.data<uint8_t>();
    for (auto it = it_begin; it != it_end; ++it) {
        const unsigned int idx = it->p + (it->x + it->y * width_) * 2;
        const uint8_t sum_max  = (it->p == 0 ? sum_max_neg_ : sum_max_pos_);
        uint8_t &sum_events    = histo[idx];
        if (sum_events <= sum_max - 1)
            ++sum_events;
        // else sum_events is saturated
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_HARDWARE_HISTO_PROCESSOR_IMPL_H
