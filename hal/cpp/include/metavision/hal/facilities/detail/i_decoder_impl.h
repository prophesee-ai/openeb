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

#ifndef METAVISION_HAL_I_DECODER_IMPL_H
#define METAVISION_HAL_I_DECODER_IMPL_H

namespace Metavision {

template<typename Event, int BUFFER_SIZE>
I_Decoder::DecodedEventForwarder<Event, BUFFER_SIZE>::DecodedEventForwarder(I_EventDecoder<Event> *i_event_decoder) :
    i_event_decoder_(i_event_decoder) {
    current_ev_ = &ev_buf_[0];
    ev_end_     = current_ev_ + BUFFER_SIZE;
}

template<typename Event, int BUFFER_SIZE>
template<typename... Args>
void I_Decoder::DecodedEventForwarder<Event, BUFFER_SIZE>::forward(Args &&...args) {
    *current_ev_ = Event(std::forward<Args>(args)...);
    if (++current_ev_ >= ev_end_) {
        add_events();
    }
}

template<typename Event, int BUFFER_SIZE>
template<typename... Args>
void I_Decoder::DecodedEventForwarder<Event, BUFFER_SIZE>::forward_unsafe(Args &&...args) {
    *current_ev_ = Event(std::forward<Args>(args)...);
    ++current_ev_;
}

template<typename Event, int BUFFER_SIZE>
void I_Decoder::DecodedEventForwarder<Event, BUFFER_SIZE>::flush() {
    if (current_ev_ > ev_buf_) {
        add_events();
    }
}

template<typename Event, int BUFFER_SIZE>
void I_Decoder::DecodedEventForwarder<Event, BUFFER_SIZE>::reserve(int size) {
    // We check that we have room for at least (size+1) events : this is because at most size events can be safely
    // added with forward_unsafe, then, when called, forward() will also add an additional event before checking that
    // the buffer is full.
    if (ev_end_ - current_ev_ < size + 1) {
        add_events();
    }
}

template<typename Event, int BUFFER_SIZE>
void I_Decoder::DecodedEventForwarder<Event, BUFFER_SIZE>::add_events() {
    i_event_decoder_->add_event_buffer(ev_buf_, current_ev_);
    current_ev_ = &ev_buf_[0];
}

inline I_Decoder::DecodedEventForwarder<EventCD> &I_Decoder::cd_event_forwarder() {
    return *cd_event_forwarder_;
}

inline I_Decoder::DecodedEventForwarder<EventExtTrigger, 1> &I_Decoder::trigger_event_forwarder() {
    return *trigger_event_forwarder_;
}

} // namespace Metavision

#endif // METAVISION_HAL_I_DECODER_IMPL_H
