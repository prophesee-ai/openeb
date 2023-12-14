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

#ifndef METAVISION_UTILS_PYBIND_SYNC_ALGORITHM_PROCESS_HELPER_H
#define METAVISION_UTILS_PYBIND_SYNC_ALGORITHM_PROCESS_HELPER_H

#include "metavision/sdk/core/utils/rolling_event_buffer.h"
#include "metavision/utils/pybind/pod_event_buffer.h"

namespace Metavision {

static const char doc_process_events_array_sync_str[] =
    "This method is used to apply the current algorithm on a chunk of events. It takes a numpy array as input and "
    "writes the results into the specified output event buffer\n"
    "   :input_np: input chunk of events (numpy structured array whose fields are ('x', 'y', 'p', 't'). Note that this "
    "order is mandatory)\n"
    "   :output_buf: output buffer of events. It can be converted to a numpy structured array using .numpy()";
static const char doc_process_events_buffer_sync_str[] =
    "This method is used to apply the current algorithm on a chunk of events. It takes an event buffer as input and "
    "writes the results into a distinct output event buffer\n"
    "   :input_buf: input chunk of events (event buffer)\n"
    "   :output_buf: output buffer of events. It can be converted to a numpy structured array using .numpy()";

static const char doc_process_events_array_sync_inplace_str[] =
    "This method is used to apply the current algorithm on a chunk of events. It takes a numpy array as input/output.\n"
    "This method should only be used when the number of output events is the same as the number of input events\n"
    "   :events_np: numpy structured array of events whose fields are ('x', 'y', 'p', 't') used as input/output. "
    "Its content will be overwritten";

static const char doc_process_events_buffer_sync_inplace_str[] =
    "This method is used to apply the current algorithm on a chunk of events. It takes an event buffer as "
    "input/output.\n"
    "This should only be used when the number of output events is the same as the number of input events\n"
    "   :events_buf: Buffer of events used as input/output. Its content will be overwritten. "
    "It can be converted to a numpy structured array using .numpy()";

template<typename Algo, typename InputEvent, typename OutputEvent = InputEvent>
void process_events_array_sync(Algo &algo, const py::array_t<InputEvent> &in, PODEventBuffer<OutputEvent> &out) {
    auto info = in.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Bad input numpy array");
    }
    auto nelem   = static_cast<size_t>(info.shape[0]);
    auto *in_ptr = static_cast<InputEvent *>(info.ptr);

    if (static_cast<void *>(in_ptr) == static_cast<void *>(out.buffer_.data())) {
        std::string error_msg =
            "Error: attempting to call process_events() in place. Consider process_events_() instead";
        throw std::invalid_argument(error_msg);
    }

    out.buffer_.clear();

    algo.process_events(in_ptr, in_ptr + nelem, std::back_inserter(out.buffer_));
}

template<typename Algo, typename InputEvent, typename OutputEvent = InputEvent>
void process_events_buffer_sync(Algo &algo, const PODEventBuffer<InputEvent> &in, PODEventBuffer<OutputEvent> &out) {
    if (static_cast<const void *>(&in) == static_cast<void *>(&out)) {
        std::string error_msg =
            "Error: attempting to call process_events() in place. Consider process_events_() instead";
        throw std::invalid_argument(error_msg);
    }
    out.buffer_.clear();

    algo.process_events(in.buffer_.cbegin(), in.buffer_.cend(), std::back_inserter(out.buffer_));
}

template<typename Algo, typename InputEvent, typename OutputEvent = InputEvent>
void process_events_rolling_buffer_sync(Algo &algo, const RollingEventBuffer<InputEvent> &in,
                                        PODEventBuffer<OutputEvent> &out) {
    out.buffer_.clear();

    algo.process_events(in.cbegin(), in.cend(), std::back_inserter(out.buffer_));
}

// This should only be used when the number of output events is the same as the number of input events
template<typename Algo, typename InputEvent>
void process_events_array_sync_inplace(Algo &algo, py::array_t<InputEvent> &events_np) {
    auto info = events_np.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Bad input numpy array");
    }
    auto nelem    = static_cast<size_t>(info.shape[0]);
    auto *buf_ptr = static_cast<InputEvent *>(info.ptr);

    algo.process_events(buf_ptr, buf_ptr + nelem, buf_ptr);
}

// This should only be used when the number of output events is equal or smaller as the number of input events
template<typename Algo, typename InputEvent>
void process_events_buffer_sync_inplace(Algo &algo, PODEventBuffer<InputEvent> &buf) {
    auto it_end = algo.process_events(buf.buffer_.cbegin(), buf.buffer_.cend(), buf.buffer_.begin());
    buf.buffer_.resize(std::distance(buf.buffer_.begin(), it_end));
}

} // namespace Metavision

#endif // METAVISION_UTILS_PYBIND_SYNC_ALGORITHM_PROCESS_HELPER_H
