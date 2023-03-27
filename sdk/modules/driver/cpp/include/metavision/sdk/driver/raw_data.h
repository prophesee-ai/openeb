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

#ifndef METAVISION_SDK_DRIVER_RAW_DATA_H
#define METAVISION_SDK_DRIVER_RAW_DATA_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

// Definition of CallbackId
#include "metavision/sdk/base/utils/callback_id.h"

namespace Metavision {

/// @brief Type alias for a callback on a buffer of raw data
using RawDataCallback = std::function<void(const uint8_t *data, size_t size)>;

/// @brief Facility class to handle RAW data
class RawData {
public:
    /// @brief Destructor
    ///
    /// Deletes a RawData class instance.
    virtual ~RawData();

    /// @brief Subscribes to RAW data callback
    ///
    /// Registers a callback that will be called each time a buffer of RAW data has been received.
    ///
    /// @param cb Callback to call each time a buffer of RAW data has been received
    /// @sa @ref RawDataCallback
    /// @return ID of the added callback
    /// @note This callback is always called after all events callbacks have been called, with the
    /// raw data buffer used to decode the events that are passed to the events callback.
    CallbackId add_callback(const RawDataCallback &cb);

    /// @brief Removes a previously registered callback
    /// @param callback_id Callback ID
    /// @return true if the callback has been unregistered correctly, false otherwise.
    /// @sa @ref add_callback
    bool remove_callback(CallbackId callback_id);

    /// @brief For internal use
    class Private;

    /// @brief For internal use
    Private &get_pimpl();

private:
    RawData(Private *);
    std::unique_ptr<Private> pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_RAW_DATA_H
