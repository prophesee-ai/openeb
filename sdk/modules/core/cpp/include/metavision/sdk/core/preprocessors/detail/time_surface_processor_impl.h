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

#ifndef METAVISION_SDK_CORE_DETAIL_TIME_SURFACE_PROCESSOR_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_TIME_SURFACE_PROCESSOR_IMPL_H

#include "metavision/sdk/core/preprocessors/time_surface_processor.h"

namespace Metavision {

template<typename InputIt, int CHANNELS>
TimeSurfaceProcessor<InputIt, CHANNELS>::TimeSurfaceProcessor(int width, int height) :
    EventPreprocessor<InputIt>(TensorShape({{"H", height}, {"W", width}, {"C", CHANNELS}}), BaseType::INT64),
    width_(width) {}

template<typename InputIt, int CHANNELS>
void TimeSurfaceProcessor<InputIt, CHANNELS>::process_events(InputIt begin, InputIt end,
                                                             MostRecentTimestampBuffer &time_surface) const {
    Tensor wrapper(this->output_tensor_shape_, this->output_tensor_type_,
                   reinterpret_cast<std::byte *>(time_surface.ptr()), false);
    process_events(0, begin, end, wrapper);
}

template<typename InputIt, int CHANNELS>
void TimeSurfaceProcessor<InputIt, CHANNELS>::compute(const timestamp, InputIt it_begin, InputIt it_end,
                                                      Tensor &tensor) const {
    auto buffer = tensor.data<timestamp>();
    for (auto it = it_begin; it != it_end; ++it) {
        assert(it->p == 0 || it->p == 1);
        const auto c  = (CHANNELS == 1) ? 0 : it->p;
        const int idx = CHANNELS * (width_ * it->y + it->x) + c;
        buffer[idx]   = it->t;
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_TIME_SURFACE_PROCESSOR_IMPL_H
