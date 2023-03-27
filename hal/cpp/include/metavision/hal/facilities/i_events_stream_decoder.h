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

#ifndef METAVISION_HAL_I_EVENTS_STREAM_DECODER_H
#define METAVISION_HAL_I_EVENTS_STREAM_DECODER_H

#include <functional>
#include <vector>
#include <memory>
#include <map>
#include <array>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/hal/facilities/i_decoder.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_registrable_facility.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_erc_counter.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"

namespace Metavision {

/// @brief Interface for decoding a stream of events
///
/// This class is meant to receive raw data from the camera, and dispatch parts of the buffer to instances
/// of @ref I_EventDecoder for specific event types.
class I_EventsStreamDecoder : public I_Decoder, public I_RegistrableFacility<I_EventsStreamDecoder> {
public:
    /// @brief Alias for callback on timestamp
    using TimeCallback_t = std::function<void(timestamp)>;

    /// @brief Constructor
    /// @param time_shifting_enabled If true, the timestamp of the decoded events will be shifted by the value of first
    /// event
    /// @param event_cd_decoder Optional decoder of CD events
    /// @param event_ext_trigger_decoder Optional decoder of trigger events
    /// @param erc_count_event_decoder Optional decoder of ERC counter events
    I_EventsStreamDecoder(
        bool time_shifting_enabled,
        const std::shared_ptr<I_EventDecoder<EventCD>> &event_cd_decoder = std::shared_ptr<I_EventDecoder<EventCD>>(),
        const std::shared_ptr<I_EventDecoder<EventExtTrigger>> &event_ext_trigger_decoder =
            std::shared_ptr<I_EventDecoder<EventExtTrigger>>(),
        const std::shared_ptr<I_EventDecoder<EventERCCounter>> &erc_count_event_decoder =
            std::shared_ptr<I_EventDecoder<EventERCCounter>>());

    /// @brief Decodes raw data. Identifies the events in the buffer and dispatches it to the instance of @ref
    /// I_EventDecoder corresponding to each event type.
    /// @warning It is mandatory to pass strictly consecutive buffers from the same source to this method
    /// @param raw_data_begin Pointer on first event
    /// @param raw_data_end Pointer after the last event
    void decode(const RawData *const raw_data_begin, const RawData *const raw_data_end) override;

    /// @brief Adds a function that will be called from time to time, giving current timestamp
    /// @param cb Callback to add
    /// @return ID of the added callback
    /// @note This method is not thread safe. You should add/remove the various callback before starting the streaming
    /// @note It's not allowed to add/remove a callback from the callback itself
    size_t add_time_callback(const TimeCallback_t &cb);

    /// @brief Removes a previously registered time callback
    /// @param callback_id Callback ID
    /// @return true if the callback has been unregistered correctly, false otherwise.
    /// @note This method is not thread safe. You should add/remove the various callback before starting the streaming
    bool remove_time_callback(size_t callback_id);

    /// @brief Gets the timestamp of the last event
    /// @return Timestamp of the last event
    virtual timestamp get_last_timestamp() const = 0;

    /// @brief Finds the timestamp shift
    ///
    /// If the timestamp shift (timestamp of the first event in the stream) is already known,
    /// the function returns true and the parameter @p timestamp_shift will be set to its value.
    /// Otherwise, the function returns false and does nothing.
    ///
    /// @return true if the timestamp shift is already known, false otherwise
    virtual bool get_timestamp_shift(Metavision::timestamp &timestamp_shift) const = 0;

    /// @brief Returns true if doing timestamp shift, false otherwise
    bool is_time_shifting_enabled() const;

    /// @brief Resets the decoder last timestamp
    /// @param timestamp Timestamp to reset the decoder to
    /// @return true if the reset operation could complete, false otherwise.
    /// @note After this call has succeeded, that @ref get_last_timestamp returns @p timestamp
    /// @warning If time shifting is enabled, the @p timestamp must be in the shifted time reference
    /// @warning Additional care may be required regarding the expected content of the data to be decoded
    ///          after this function has been called. Refer to the constraints and limitations of a specific
    ///          decoder implementation (e.g EVT2Decoder::reset_timestamp_impl, EVT21Decoder::reset_timestamp_impl and
    ///          EVT3Decoder::reset_timestamp_impl)
    bool reset_timestamp(const Metavision::timestamp &timestamp);

    /// @brief Resets the decoder timestamp shift
    /// @param shift Timestamp shift to reset the decoder to
    /// @return True if the reset operation could complete, false otherwise.
    /// @note If time shifting is disabled, this function does nothing
    bool reset_timestamp_shift(const Metavision::timestamp &shift);

    /// @brief Returns true if the decoded events stream can be indexed
    virtual bool is_decoded_event_stream_indexable() const;

protected:
    /// @cond DEV

    /// @brief Helper class that can be used to simplify buffering and forwarding events to the @ref I_EventDecoder
    ///
    /// For performance reasons, it is not recommended to call @ref I_EventDecoder::add_event_buffer event by event.
    /// The decoder implementation is free to use this helper class or not, but some buffering should be put in place
    /// for better performance.
    template<typename Event, int BUFFER_SIZE = 320>
    struct DecodedEventForwarder {
        /// @brief Constructor
        DecodedEventForwarder(I_EventDecoder<Event> *i_event_decoder);

        /// @brief Forwards events
        /// Forwards the event to I_EventDecoder<Event>, with a sanity check on the internal buffer that stores the
        /// events
        /// @param args Input argument to the constructor of a Event
        template<typename... Args>
        void forward(Args &&...args);

        /// @brief Forwards events
        /// Forwards the event to I_EventDecoder<Event>, without a sanity check on the internal buffer that stores the
        /// events. You can call this methods after reserve(), as many times as the reserved size
        /// @param args Input argument to the constructor of a Event
        template<typename... Args>
        void forward_unsafe(Args &&...args);

        /// @brief Flushes stored events, forwarding them all to I_EventDecoder<Event>
        void flush();

        /// @brief Reserves space in array
        /// Checks if the space asked is available, if not it flushes the events and reset the buffer
        /// After calling this method, you can use forward_unsafe(), instead of operator()
        /// @param size Size to reserve. It has to be <= BUFFER_SIZE
        void reserve(int size);

    private:
        void add_events();
        I_EventDecoder<Event> *i_event_decoder_;
        std::array<Event, BUFFER_SIZE> ev_buf_;
        typename std::array<Event, BUFFER_SIZE>::iterator ev_it_;
    };

    /// @brief Gets the reference to the forwarder of CD events
    DecodedEventForwarder<EventCD> &cd_event_forwarder();

    /// @brief Gets the reference to the forwarder of trigger events
    DecodedEventForwarder<EventExtTrigger, 1> &trigger_event_forwarder();

    /// @brief Gets the reference to the forwarder of CD count events
    DecodedEventForwarder<EventERCCounter, 1> &erc_count_event_forwarder();

    /// @endcond

private:
    /// @brief The implementation of the raw data decoding. Identifies the events in the buffer
    /// and dispatches it to the instance of @ref I_EventDecoder corresponding
    /// to each event type.
    ///
    /// The size of the input buffer is guaranteed to be a multiple of the result of @ref get_raw_event_size_bytes().
    ///
    /// @warning It is mandatory to pass strictly consecutive buffers from the same source to this method
    /// @param raw_data_begin A reference to a pointer on first event.
    /// @param raw_data_end Pointer after the last event
    virtual void decode_impl(const RawData *const raw_data_begin, const RawData *const raw_data_end) = 0;

    /// @brief Implementation of "reset the decoder last timestamp" operation
    /// @param timestamp Timestamp to reset the decoder to
    ///        If >= 0, reset the decoder last timestamp to the actual value @p timestamp
    ///        If < 0, reset the decoder internal state so that the last timestamp will be found from the
    ///        next buffer of events to decoder (the timestamp shift and overflow loop counter is not reset)
    /// @return True if the reset operation could complete, false otherwise.
    /// @note It is expected after this call has succeeded, that @ref get_last_timestamp returns @p timestamp
    /// @warning If time shifting is enabled, the @p timestamp must be in the shifted time reference
    virtual bool reset_timestamp_impl(const Metavision::timestamp &timestamp) = 0;

    /// @brief Implementation of "reset the decoder timestamp shift" operation
    /// @param shift Timestamp shift to reset the decoder to
    /// @return True if the reset operation could complete, false otherwise.
    /// @note If time shifting is disabled, this function does nothing
    virtual bool reset_timestamp_shift_impl(const Metavision::timestamp &shift) = 0;

    const bool is_time_shifting_enabled_;
    std::vector<RawData> incomplete_raw_data_;

    std::map<size_t, TimeCallback_t> time_cbs_map_;
    size_t next_cb_idx_{0};

    std::shared_ptr<I_EventDecoder<EventCD>> cd_event_decoder_;
    std::unique_ptr<DecodedEventForwarder<EventCD>> cd_event_forwarder_;

    std::shared_ptr<I_EventDecoder<EventExtTrigger>> ext_trigger_event_decoder_;
    std::unique_ptr<DecodedEventForwarder<EventExtTrigger, 1>> trigger_event_forwarder_;

    std::shared_ptr<I_EventDecoder<EventERCCounter>> erc_count_event_decoder_;
    std::unique_ptr<DecodedEventForwarder<EventERCCounter, 1>> erc_count_event_forwarder_;
};

} // namespace Metavision

#include "detail/i_events_stream_decoder_impl.h"

#endif // METAVISION_HAL_I_EVENTS_STREAM_DECODER_H
