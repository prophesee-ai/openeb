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

#ifndef METAVISION_HAL_I_EVENTS_STREAM_DECODER_IMPL_H
#define METAVISION_HAL_I_EVENTS_STREAM_DECODER_IMPL_H

namespace Metavision {

template<typename Event, int BUFFER_SIZE>
I_EventsStreamDecoder::DecodedEventForwarder<Event, BUFFER_SIZE>::DecodedEventForwarder(
    I_EventDecoder<Event> *i_event_decoder) :
    i_event_decoder_(i_event_decoder) {
    ev_it_ = ev_buf_.begin();
}

template<typename Event, int BUFFER_SIZE>
template<typename... Args>
void I_EventsStreamDecoder::DecodedEventForwarder<Event, BUFFER_SIZE>::forward(Args &&...args) {
    *ev_it_ = Event(std::forward<Args>(args)...);
    if (++ev_it_ == ev_buf_.end()) {
        add_events();
    }
}

template<typename Event, int BUFFER_SIZE>
template<typename... Args>
void I_EventsStreamDecoder::DecodedEventForwarder<Event, BUFFER_SIZE>::forward_unsafe(Args &&...args) {
    *ev_it_ = Event(std::forward<Args>(args)...);
    ++ev_it_;
}

template<typename Event, int BUFFER_SIZE>
void I_EventsStreamDecoder::DecodedEventForwarder<Event, BUFFER_SIZE>::flush() {
    if (ev_it_ != ev_buf_.begin()) {
        add_events();
    }
}

template<typename Event, int BUFFER_SIZE>
void I_EventsStreamDecoder::DecodedEventForwarder<Event, BUFFER_SIZE>::reserve(int size) {
    // We check that we have room for at least (size+1) events : this is because at most size events can be safely
    // added with forward_unsafe, then, when called, forward() will also add an additional event before checking that
    // the buffer is full.
    if (std::distance(ev_it_, ev_buf_.end()) < size + 1) {
        add_events();
    }
}

template<typename Event, int BUFFER_SIZE>
void I_EventsStreamDecoder::DecodedEventForwarder<Event, BUFFER_SIZE>::add_events() {
    i_event_decoder_->add_event_buffer(ev_buf_.data(), ev_buf_.data() + std::distance(ev_buf_.begin(), ev_it_));
    ev_it_ = ev_buf_.begin();
}

inline I_EventsStreamDecoder::DecodedEventForwarder<EventCD> &I_EventsStreamDecoder::cd_event_forwarder() {
    return *cd_event_forwarder_;
}

inline I_EventsStreamDecoder::DecodedEventForwarder<EventExtTrigger, 1> &
    I_EventsStreamDecoder::trigger_event_forwarder() {
    return *trigger_event_forwarder_;
}

inline I_EventsStreamDecoder::DecodedEventForwarder<EventERCCounter, 1> &
    I_EventsStreamDecoder::erc_count_event_forwarder() {
    return *erc_count_event_forwarder_;
}

} // namespace Metavision

#endif // METAVISION_HAL_I_EVENTS_STREAM_DECODER_IMPL_H
