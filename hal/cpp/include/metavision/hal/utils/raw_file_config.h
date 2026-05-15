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

#ifndef METAVISION_HAL_RAW_FILE_CONFIG_H
#define METAVISION_HAL_RAW_FILE_CONFIG_H

#include <cstddef>
#include <cstdint>

namespace Metavision {

/// @brief RAW files configuration's options
class RawFileConfig {
public:
    /// Number of events_byte_size blocks to read.
    /// At each read, n_events_to_read_*sizeof(RAW_event) bytes are read.
    /// @warning sizeof(RAW_event) is defined by the events format contained in the RAW file read.
    uint32_t n_events_to_read_ = 1000000;

    /// The maximum number of buffers to allocate and use for reading. Each buffer contains at most @ref
    /// n_events_to_read_ RAW events. The maximum memory allocated to read the RAW file will be read_buffers_count_ *
    /// n_events_to_read_ * sizeof(RAW_event). One can use this parameters to have a finer control on offline memory
    /// usage
    uint32_t n_read_buffers_ = 3;

    /// Take the first timer high of the file as origin of time
    bool do_time_shifting_ = true;

    /// True if indexing should be performed when opening the file
    /// Alternatively, indexing can still be requested by calling I_EventsStream::index directly
    bool build_index_ = true;
};

} // namespace Metavision

#endif // METAVISION_HAL_RAW_FILE_CONFIG_H
