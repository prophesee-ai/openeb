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

#ifndef METAVISION_SDK_CORE_TIME_SURFACE_PRODUCER_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_TIME_SURFACE_PRODUCER_ALGORITHM_IMPL_H

namespace Metavision {

template<int CHANNELS>
TimeSurfaceProducerAlgorithm<CHANNELS>::TimeSurfaceProducerAlgorithm(int width, int height) :
    time_surface_(height, width, CHANNELS) {
    time_surface_.set_to(0);

    output_cb_ = [](timestamp, const MostRecentTimestampBuffer &) {};
}

template<int CHANNELS>
void TimeSurfaceProducerAlgorithm<CHANNELS>::set_output_callback(const OutputCb &cb) {
    output_cb_ = cb;
}

template<int CHANNELS>
template<typename InputIt>
inline void TimeSurfaceProducerAlgorithm<CHANNELS>::process_online(InputIt it_begin, InputIt it_end) {
    for (auto it = it_begin; it != it_end; ++it) {
        assert(it->p == 0 || it->p == 1);
        const auto c                      = (CHANNELS == 1) ? 0 : it->p;
        time_surface_.at(it->y, it->x, c) = it->t;
    }
}

template<int CHANNELS>
void TimeSurfaceProducerAlgorithm<CHANNELS>::process_async(const timestamp processing_ts,
                                                           const size_t n_processed_events) {
    output_cb_(processing_ts, time_surface_);
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_TIME_SURFACE_PRODUCER_ALGORITHM_IMPL_H
