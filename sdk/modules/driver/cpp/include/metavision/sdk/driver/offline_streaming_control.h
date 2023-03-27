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

#ifndef METAVISION_SDK_DRIVER_OFFLINE_STREAMING_CONTROL_H
#define METAVISION_SDK_DRIVER_OFFLINE_STREAMING_CONTROL_H

#include <string>
#include <memory>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Facility class to control offline streaming features
class OfflineStreamingControl {
public:
    ~OfflineStreamingControl();

    /// @brief Checks if all features of the offline streaming control are ready
    /// @return true if all features are ready, false otherwise
    /// @note You should make sure this function returns true before making any other calls
    bool is_ready() const;

    /// @brief Seek to a specified timestamp
    /// @param ts Timestamp to seek to
    /// @return True if seeking succeeded, false otherwise
    bool seek(Metavision::timestamp ts);

    /// @brief Gets the first reachable timestamp that can be seeked to
    /// @return Metavision::timestamp First timestamp to seek to
    Metavision::timestamp get_seek_start_time() const;

    /// @brief Gets the last reachable timestamp that can be seeked to
    /// @return Metavision::timestamp Last timestamp to seek to
    Metavision::timestamp get_seek_end_time() const;

    /// @brief Gets the duration of the recording
    /// @return Metavision::timestamp Duration of the recording
    Metavision::timestamp get_duration() const;

    /// @brief For internal use
    class Private;
    /// @brief For internal use
    Private &get_pimpl();

private:
    OfflineStreamingControl(Private *pimpl);
    std::unique_ptr<Private> pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_OFFLINE_STREAMING_CONTROL_H
