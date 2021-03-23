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

#ifndef METAVISION_SDK_CORE_DETAIL_FILE_PRODUCER_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_FILE_PRODUCER_ALGORITHM_IMPL_H

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/base/utils/generic_header.h"

namespace Metavision {

template<class Event>
template<class OutputIt>
inline void FileProducerAlgorithmT<Event>::process_events(OutputIt d_first, timestamp ts) {
    // using Event = typename std::iterator_traits<OutputIt>::value_type;
    // decltype(*d_first) Event;
    if (version_ >= 2) {
        process_output(
            ts, d_first, Event::read_event,
            [](void *data, timestamp delta_ts) { return static_cast<typename Event::RawEvent *>(data)->ts + delta_ts; },
            [](void *data, int increment) { return static_cast<typename Event::RawEvent *>(data) + increment; });
    } else {
        process_output(
            ts, d_first, Event::read_event_v1,
            [](void *data, timestamp delta_ts) {
                return static_cast<typename Event::RawEventV1 *>(data)->ts + delta_ts;
            },
            [](void *data, int increment) { return static_cast<typename Event::RawEventV1 *>(data) + increment; });
    }
}

template<class Event>
inline timestamp FileProducerAlgorithmT<Event>::get_time_at(timestamp time_window, bool backward) {
    if (time_last_event_ < 0) {
        // Need to parse all the file to get the time of the last event (can't just jump
        // directly to the end because we would miss the overflows
        if (version_ >= 2) {
            loop_through_file(Event::read_event, [](void *data, uint64_t increment) {
                return static_cast<typename Event::RawEvent *>(data) + increment;
            });
        } else {
            loop_through_file(Event::read_event_v1, [](void *data, uint64_t increment) {
                return static_cast<typename Event::RawEventV1 *>(data) + increment;
            });
        }
    }
    return (backward) ? std::max(time_first_event_, time_last_event_ - time_window) :
                        std::min(time_last_event_, time_first_event_ + time_window);
}

template<class Event>
inline void FileProducerAlgorithmT<Event>::initialize() {
    if (current_event_ < n_tot_events_) {
        file_->read(&last_buffer_[0], ev_size_);
        if (version_ >= 2) {
            time_first_event_ = static_cast<typename Event::RawEvent *>(last_data_)->ts + delta_ts_overflow_;
        } else {
            time_first_event_ = static_cast<typename Event::RawEventV1 *>(last_data_)->ts + delta_ts_overflow_;
        }
    }

    time_start_event_     = time_first_event_;
    time_last_event_read_ = time_first_event_;
}

template<class Event>
inline void FileProducerAlgorithmT<Event>::start_at_time(timestamp start_time) {
    std::function<timestamp(void *, timestamp)> get_time;
    std::function<Event(void *, timestamp)> read_event;
    std::function<void *(void *, uint64_t)> increment_data;

    if (version_ >= 2) {
        get_time = [](void *data, timestamp delta_ts) {
            return static_cast<typename Event::RawEvent *>(data)->ts + delta_ts;
        };
        read_event     = Event::read_event;
        increment_data = [](void *data, uint64_t increment) {
            return static_cast<typename Event::RawEvent *>(data) + increment;
        };
    } else {
        get_time = [](void *data, timestamp delta_ts) {
            return static_cast<typename Event::RawEventV1 *>(data)->ts + delta_ts;
        };
        read_event     = Event::read_event_v1;
        increment_data = [](void *data, uint64_t increment) {
            return static_cast<typename Event::RawEventV1 *>(data) + increment;
        };
    }

    if (time_last_event_ < 0) {
        // Need to parse all the file to get the time of the last event (can't just jump
        // directly to the end because we would miss the overflows
        loop_through_file(read_event, increment_data);
    }

    if (start_time <= time_first_event_) {
        position_start_event = position_first_event_;
        start_event_         = 0;
        origin_              = start_time;
    } else if (start_time > time_last_event_) {
        position_start_event = (uint64_t)position_first_event_ + ev_size_ * n_tot_events_;
        start_event_         = n_tot_events_;
        origin_              = 0; // start_time;
    } else {
        // Binary search

        uint64_t start_ev = 0;
        uint64_t end_ev   = n_tot_events_ - 1;

        if (events_overflows.size() > 0) {
            n_times_overflows_ = start_time / MAX_TIMESTAMP_32;

            if (n_times_overflows_ > 0) {
                start_ev = events_overflows[n_times_overflows_ - 1];
            }

            delta_ts_overflow_ = n_times_overflows_ * OVERFLOW_LENGTH;

            if (n_times_overflows_ < events_overflows.size()) {
                end_ev = events_overflows[n_times_overflows_];
            }
        }
        uint64_t curr_ev;
        if (events_from_ram_) {
            while (start_ev < end_ev - 1) {
                curr_ev    = (start_ev + end_ev) / 2;
                last_data_ = (&vrawevents_[times_event_in_vrawevents_ * curr_ev]);
                if (get_time(last_data_, delta_ts_overflow_) >= start_time) {
                    end_ev = curr_ev;
                } else {
                    start_ev = curr_ev;
                }
            }
        } else {
            while (start_ev < end_ev - 1) {
                curr_ev = (start_ev + end_ev) / 2;
                file_->seekg((uint64_t)position_first_event_ + ev_size_ * curr_ev);
                file_->read(&last_buffer_[0], ev_size_);
                if (get_time(last_data_, delta_ts_overflow_) >= start_time) {
                    end_ev = curr_ev;
                } else {
                    start_ev = curr_ev;
                }
            }
        }
        start_event_         = end_ev;
        position_start_event = (uint64_t)position_first_event_ + ev_size_ * start_event_;
        origin_              = start_time;
    }

    current_event_ = start_event_;

    if (events_from_ram_) {
        if (current_event_ < n_tot_events_) {
            last_data_ = &vrawevents_[start_event_ * times_event_in_vrawevents_];
        }
    } else {
        file_->seekg(position_start_event);
        if (current_event_ < n_tot_events_) {
            file_->read(&last_buffer_[0], ev_size_);
        }
    }
    time_start_event_     = get_time(last_data_, delta_ts_overflow_);
    time_last_event_read_ = time_first_event_;
}

template<class Event>
template<class OutputIt, class FunctionRead, class FunctionTime, class FunctionIncrement>
inline void FileProducerAlgorithmT<Event>::process_output(timestamp ts, OutputIt d_first,
                                                          const FunctionRead &read_event, const FunctionTime &get_time,
                                                          const FunctionIncrement &increment_data) {
    // Fill the buffer with current events
    while (current_event_ < n_tot_events_ && get_time(last_data_, delta_ts_overflow_) + delta_ts_loop_ - origin_ < ts) {
        // Read the current event
        Event ev = read_event(last_data_, delta_ts_overflow_ + delta_ts_loop_ - origin_);

        if (time_last_event_read_ - get_time(last_data_, delta_ts_overflow_) >= MAX_TIMESTAMP_32 - threshold_) {
            // Looped!
            if (time_last_event_ < 0) {
                events_overflows.push_back(current_event_);
            }
            ++n_times_overflows_;
            delta_ts_overflow_ = n_times_overflows_ * OVERFLOW_LENGTH;

            ev = read_event(last_data_, delta_ts_overflow_ + delta_ts_loop_ - origin_);
        }

        *d_first = ev;
        ++d_first;

        time_last_event_read_ = get_time(last_data_, delta_ts_overflow_);

        ++current_event_;

        // If we have to loop and we arrived at the last element, restart from the beginning
        if (current_event_ == n_tot_events_) { // End of file
            if (time_last_event_ < 0) {
                time_last_event_ = time_last_event_read_;
            }

            if (time_last_event_ + loop_delay_ - origin_ > get_max_loop_length()) {
                set_max_loop_length(time_last_event_ + loop_delay_ - origin_);
            }

            if (!loop_) {
                return;
            }
            ++n_loop;
            current_event_        = start_event_;
            n_times_overflows_    = time_start_event_ / MAX_TIMESTAMP_32;
            delta_ts_overflow_    = n_times_overflows_ * OVERFLOW_LENGTH;
            time_last_event_read_ = time_start_event_;

            delta_ts_loop_ = n_loop * get_max_loop_length();

            if (events_from_ram_) {
                int n      = sizeof(ev_size_) / sizeof(typename decltype(vrawevents_)::value_type);
                last_data_ = &vrawevents_[start_event_ * n]; // TODO put start_event_ * size instead
            } else {
                file_->seekg(position_start_event);
            }
        }
        // no need to check if cur_ev_ < ev_num (if ev_num == 0 we do not enter the while,
        // and if cur_ev_ was == ev_num_ then in the previous if we set it to 0)
        if (events_from_ram_) {
            last_data_ = increment_data(last_data_, 1);
        } else {
            file_->read(&last_buffer_[0], ev_size_);
        }
    }
}

template<class Event>
template<class FunctionRead, class FunctionIncrement>
inline void FileProducerAlgorithmT<Event>::loop_through_file(const FunctionRead &read_event,
                                                             const FunctionIncrement &increment_data) {
    MV_SDK_LOG_INFO() << Log::no_space << "Looping through file " << filename_ << ": this may take some time ...";

    if (events_from_ram_) {
        void *data          = (&vrawevents_[0]); // going to first event
        timestamp last_time = time_first_event_;

        events_overflows.clear();
        for (uint64_t curr_ev = 1; curr_ev < n_tot_events_; ++curr_ev) {
            data = increment_data(data, 1);

            Event ev = read_event(data, 0);
            if (last_time - ev.t >= MAX_TIMESTAMP_32 - threshold_) {
                events_overflows.push_back(curr_ev);
            }
            last_time = ev.t;
        }

        time_last_event_ = last_time + events_overflows.size() * OVERFLOW_LENGTH;

    } else {
        std::streampos current_position = file_->tellg(); // Get the current position in the file

        uint64_t curr_ev = current_event_;

        std::vector<char> buffer(ev_size_);
        void *data = static_cast<void *>(&buffer[0]);

        file_->seekg(position_first_event_); // Reset the position at the first event
        file_->read(&buffer[0], ev_size_);   // Read the first event

        timestamp last_time = time_first_event_;

        events_overflows.clear();
        for (curr_ev = 1; curr_ev < n_tot_events_; ++curr_ev) {
            file_->read(&buffer[0], ev_size_); // read data from file

            Event ev = read_event(data, 0);
            if (last_time - ev.t >= MAX_TIMESTAMP_32 - threshold_) {
                events_overflows.push_back(curr_ev);
            }

            last_time = ev.t;
        }

        time_last_event_ = last_time + events_overflows.size() * OVERFLOW_LENGTH;

        file_->seekg(current_position); // Reset the position
    }

    MV_SDK_LOG_INFO() << "Total length of file is" << Log::no_space << time_last_event_ << "us";
}

template<class Event>
inline FileProducerAlgorithmT<Event>::FileProducerAlgorithmT(std::string filename, bool loop, timestamp loop_delay) :
    file_(new std::ifstream(filename.c_str(), std::ios::binary)),
    filename_(filename),
    loop_(loop),
    loop_delay_(loop_delay) {
    if (!file_->is_open()) {
        throw std::runtime_error("Could not open file " + filename_);
    }

    // Get size and type of the events
    unsigned char ev_type, ev_size;
    // Parse the header, if present
    GenericHeader header_parser(*file_);

    // If no header was found, consider the file to be an old file without
    // event information at the beginning and thus only containing Event2ds
    if (header_parser.empty()) {
        ev_type = 0;
        ev_size = 8;
    } else {
        ev_type = 0;
        ev_size = 8;
        // Read file infos: the first two characters are the info on event type and size
        if (file_)
            file_->read((char *)&ev_type, 1);
        if (file_)
            file_->read((char *)&ev_size, 1);
    }
    ev_size_ = ev_size; // Size of the raw event (in bytes --> ex if the RawEvent is a structure of 64 bit, ev_size_
                        // would be 8)

    // allocate reading buffer
    last_buffer_.reserve(ev_size_);

    // Look for the version, if any
    auto value = header_parser.get_field("Version");
    if (!value.empty())
        version_ = std::stoi(value);

    value  = header_parser.get_field("Width");
    width_ = value.empty() ? 304 : std::stoi(value);

    value   = header_parser.get_field("Height");
    height_ = value.empty() ? 240 : std::stoi(value);

    date_ = header_parser.get_date();

    last_data_ = static_cast<void *>(&last_buffer_[0]);

    // Determine the number of events in file
    position_first_event_ = file_->tellg(); // Get the current position in the file
    file_->seekg(0, std::ios::end);         // Go to the end of the file
    n_tot_events_ = (file_->tellg() - position_first_event_) / (long long)ev_size_; // Compute the number of events
    file_->seekg(position_first_event_); // Reset the position at the first event

    if (n_tot_events_ == 0) {
        MV_SDK_LOG_WARNING() << "No events found in file" << filename_;
    }

    start_event_         = 0;
    current_event_       = start_event_;
    position_start_event = position_first_event_;

    times_event_in_vrawevents_ = sizeof(ev_size_) / sizeof(typename decltype(vrawevents_)::value_type);
    initialize();
}

template<class Event>
inline FileProducerAlgorithmT<Event>::~FileProducerAlgorithmT() {
    file_->close();
}

template<class Event>
inline bool FileProducerAlgorithmT<Event>::is_done() {
    return current_event_ >= n_tot_events_;
}

template<class Event>
inline uint64_t FileProducerAlgorithmT<Event>::get_n_tot_ev() const {
    return n_tot_events_;
}

template<class Event>
inline void FileProducerAlgorithmT<Event>::reset_max_loop_length() {
    set_max_loop_length(0);
}

template<class Event>
inline void FileProducerAlgorithmT<Event>::load_to_ram() {
    events_from_ram_ = true;
    file_->seekg(position_start_event);
    file_->read(&last_buffer_[0], ev_size_);
    vrawevents_.clear();
    while (*file_) {
        uint8_t *p = static_cast<uint8_t *>(last_data_);
        vrawevents_.insert(vrawevents_.end(), p, p + times_event_in_vrawevents_);

        file_->read(&last_buffer_[0], ev_size_);
    }
    last_data_ = (&vrawevents_[0]);
}

template<class Event>
inline int FileProducerAlgorithmT<Event>::get_width() const {
    return width_;
}

template<class Event>
inline int FileProducerAlgorithmT<Event>::get_height() const {
    return height_;
}

template<class Event>
inline std::string FileProducerAlgorithmT<Event>::get_date() const {
    return date_;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_FILE_PRODUCER_ALGORITHM_IMPL_H
