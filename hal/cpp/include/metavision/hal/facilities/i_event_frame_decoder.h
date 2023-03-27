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

#ifndef METAVISION_HAL_I_EVENT_FRAME_DECODER_H
#define METAVISION_HAL_I_EVENT_FRAME_DECODER_H

#include <functional>
#include <map>

#include "metavision/hal/facilities/i_decoder.h"
#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Interface for decoding events
///
/// This class is meant to receive raw data from the camera, and dispatch parts of the buffer to instances
/// of @ref I_EventDecoder for specific event types.
template<class FrameType>
class I_EventFrameDecoder : public I_Decoder, public I_RegistrableFacility<I_EventFrameDecoder<FrameType>> {
public:
    /// @brief Alias for raw data type
    using EventFrameCallback_t = std::function<void(const FrameType &)>;

    I_EventFrameDecoder(int height, int width);

    /// @brief Sets the functions to call to each decoded frame of events
    /// @param cb Callback to add
    /// @return ID of the added callback
    /// @note This method is not thread safe. You should add/remove the various callback before starting the streaming
    /// @note It's not allowed to add/remove a callback from the callback itself
    size_t add_event_frame_callback(const EventFrameCallback_t &cb);

    /// @brief Removes a previously registered callback
    /// @param callback_id Callback ID
    /// @return true if the callback has been unregistered correctly, false otherwise.
    /// @sa @ref add_event_frame_callback
    bool remove_callback(size_t callback_id);

    /// @brief Decodes raw event frame data.
    /// @warning It is mandatory to pass strictly consecutive buffers from the same source to this method
    /// @param raw_data_begin Pointer to start of frame data
    /// @param raw_data_end Pointer after the last byte of frame data
    virtual void decode(const RawData *const raw_data_begin, const RawData *const raw_data_end) = 0;

    /// @brief Gets size of a raw event element in bytes
    virtual uint8_t get_raw_event_size_bytes() const = 0;

    unsigned get_height() const {
        return height_;
    }
    unsigned get_width() const {
        return width_;
    }

protected:
    /// @cond DEV
    void add_event_frame(const FrameType &frame);
    /// @endcond

    const unsigned height_;
    const unsigned width_;

private:
    std::map<size_t, EventFrameCallback_t> cbs_map_;
    size_t next_cb_idx_{0};
};

template<class FrameType>
I_EventFrameDecoder<FrameType>::I_EventFrameDecoder(int height, int width) : height_(height), width_(width) {}

template<class FrameType>
size_t I_EventFrameDecoder<FrameType>::add_event_frame_callback(const EventFrameCallback_t &cb) {
    cbs_map_[next_cb_idx_] = cb;
    return next_cb_idx_++;
}

template<class FrameType>
bool I_EventFrameDecoder<FrameType>::remove_callback(size_t callback_id) {
    return cbs_map_.erase(callback_id);
}

/// @cond DEV
template<class FrameType>
void I_EventFrameDecoder<FrameType>::add_event_frame(const FrameType &frame) {
    for (auto &it : cbs_map_) {
        it.second(frame);
    }
}
/// @endcond

} // namespace Metavision

#endif // METAVISION_HAL_I_EVENT_FRAME_DECODER_H
