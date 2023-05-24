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

#ifndef METAVISION_HAL_I_DECODER_H
#define METAVISION_HAL_I_DECODER_H

#include <cstdint>
#include <functional>

#include "metavision/hal/utils/decoder_protocol_violation.h"

namespace Metavision {

class I_Decoder {
public:
    /// @brief Alias for raw data type
    using RawData = uint8_t;

    /// @brief Decodes raw data.
    /// @param raw_data_begin Pointer on first event
    /// @param raw_data_end Pointer after the last event
    virtual void decode(const RawData *const raw_data_begin, const RawData *const raw_data_end) = 0;

    /// @brief Alias for callback on protocol violation
    using ProtocolViolationCallback_t = std::function<void(DecoderProtocolViolation)>;

    /// @brief Adds a function to be called when decoder protocol is breached
    /// @param cb Callback to add
    /// @return ID of the added callback
    /// @note This method is not thread safe. You should add/remove the various callback before starting the streaming
    /// @note It's not allowed to add/remove a callback from the callback itself
    virtual size_t add_protocol_violation_callback(const ProtocolViolationCallback_t &cb);

    /// @brief Removes a previously registered protocol violation callback
    /// @param callback_id Callback ID
    /// @return true if the callback has been unregistered correctly, false otherwise.
    /// @note This method is not thread safe. You should add/remove the various callback before starting the streaming
    virtual bool remove_protocol_violation_callback(size_t callback_id);

    /// @brief Gets size of a raw event element in bytes
    virtual uint8_t get_raw_event_size_bytes() const = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_DECODER_H
