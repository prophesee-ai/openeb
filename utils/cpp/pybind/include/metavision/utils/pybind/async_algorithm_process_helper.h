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

#ifndef METAVISION_UTILS_PYBIND_ASYNC_ALGORITHM_PROCESS_HELPER_H
#define METAVISION_UTILS_PYBIND_ASYNC_ALGORITHM_PROCESS_HELPER_H

#include "metavision/utils/pybind/pod_event_buffer.h"

namespace Metavision {

static const char doc_process_events_array_async_str[] = "Processes a buffer of events for later frame generation\n"
                                                         "\n"
                                                         "   :events_np: numpy structured array of events whose fields "
                                                         "are ('x', 'y', 'p', 't'). Note that this order is mandatory";

template<typename Algo, typename InputEvent>
void process_events_array_async(Algo &algo, const py::array_t<InputEvent> &in) {
    auto info = in.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Bad input numpy array");
    }
    auto nelem   = static_cast<size_t>(info.shape[0]);
    auto *in_ptr = static_cast<InputEvent *>(info.ptr);

    algo.process_events(in_ptr, in_ptr + nelem);
}
} // namespace Metavision

#endif // METAVISION_UTILS_PYBIND_ASYNC_ALGORITHM_PROCESS_HELPER_H
