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

#ifndef METAVISION_SDK_CORE_DETAIL_RING_H
#define METAVISION_SDK_CORE_DETAIL_RING_H

#include <mutex>
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>
#include <condition_variable>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/utils/detail/iterator_traits.h"

namespace Metavision {
namespace detail {

template<typename Event>
inline timestamp get_time(const Event &ev) {
    return ev.t;
}

// accumulate events in the given range as long as num_events + diff(start, end) < max_events
// returns true when (updated) num_events == max_events
template<class Iterator>
inline bool keep_upto_max_events(Iterator start_it, unsigned int &start_offset_incl, unsigned int end_offset_excl,
                                 int &num_events, int max_events) {
    int cur_num_events = static_cast<int>(std::distance(start_it + start_offset_incl, start_it + end_offset_excl));
    bool ret           = (num_events + cur_num_events > max_events);
    if (ret) {
        unsigned int rem_events = max_events - num_events;
        start_offset_incl += cur_num_events - rem_events;
        cur_num_events = rem_events;
    }
    num_events += cur_num_events;
    return ret;
}

template<typename Event>
class Ring {
public:
    typedef Event type_data;
    typedef std::vector<Event> type_eventsadd;

    Ring(unsigned int min_ring_capacity = 8, bool auto_shrink = true) :
        min_ring_capacity_(min_ring_capacity), auto_shrink_(auto_shrink) {
        ring_buffer_ = std::vector<std::vector<Event> *>(min_ring_capacity_);
        for (unsigned int i = 0; i < min_ring_capacity_; i++) {
            ring_buffer_[i] = new std::vector<Event>(1024);
        }
        ring_size_       = ring_buffer_.capacity();
        ring_head_       = 0;
        ring_next_       = 0;
        next_buffer_ind_ = 0;
        max_ring_size_   = 0;
    }
    ~Ring() {
        std::unique_lock<std::mutex> lock(ring_mut_);
        for (unsigned int i = 0; i < ring_size_; i++) {
            delete ring_buffer_[i];
        }
    }
    void set_max_ring_size(unsigned int s) {
        max_ring_size_ = s;
    }

    template<typename OutputIt>
    void fill_buffer_to(OutputIt d_first, timestamp ts) {
        static_assert(std::is_same<Event, typename iterator_traits<OutputIt>::value_type>::value,
                      "fill_buffer_to called with invalid type of events.");

        std::unique_lock<std::mutex> lock(ring_mut_);
        if (!data_available_internal()) {
            return;
        }

        if (get_time(ring_buffer_[ring_head_]->back()) < ts) {
            std::copy(ring_buffer_[ring_head_]->cbegin() + next_buffer_ind_, ring_buffer_[ring_head_]->cend(), d_first);
            ring_head_       = (ring_head_ + 1) % ring_size_;
            next_buffer_ind_ = 0;

            while (data_available_internal() && get_time(ring_buffer_[ring_head_]->back()) < ts) {
                std::copy(ring_buffer_[ring_head_]->cbegin(), ring_buffer_[ring_head_]->cend(), d_first);
                ring_head_ = (ring_head_ + 1) % ring_size_;
                if (ring_buffer_[ring_head_]->empty())
                    return;
            }
        }
        if (!data_available_internal())
            return;
        Event ev_ts;
        ev_ts.t    = ts;
        auto first = ring_buffer_[ring_head_]->cbegin() + next_buffer_ind_;
        auto it_ts = std::lower_bound(first, ring_buffer_[ring_head_]->cend(), ev_ts,
                                      [](const Event &ev1, const Event &ev2) { return get_time(ev1) < get_time(ev2); });
        std::copy(first, it_ts, d_first);
        next_buffer_ind_ = static_cast<unsigned int>(std::distance(ring_buffer_[ring_head_]->cbegin(), it_ts));
    }

    // Fill the buffer with max_events events strictly before timestamp ts
    template<typename OutputIt>
    void fill_buffer_to_drop_max_events(OutputIt d_first, timestamp ts, int max_events) {
        static_assert(std::is_same<Event, typename iterator_traits<OutputIt>::value_type>::value,
                      "fill_buffer_to_drop_max_events called with invalid type of events.");

        std::unique_lock<std::mutex> lock(ring_mut_);
        if (!data_available_internal()) {
            return;
        }

        // ----------------- FIND BOUNDS ---------------------
        Event ev_ts;
        ev_ts.t = ts;
        // all the events to keep are in the range :
        // (ring_buffer[p_first]->begin() + first_start_incl, ring_buffer[p_last]->begin() + last_end_excl]
        unsigned int p_first, p_last, p_cur;
        unsigned int p_first_start_incl = 0, p_last_end_excl = 0;
        int num_events = 0;

        // find the last vector which contains events before requested ts using binary search
        unsigned int count = buffer_delay(), cur_index, step;
        p_last             = ring_head_;
        while (count > 0) {
            step      = count / 2;
            cur_index = (p_last + step) % ring_size_;
            if (get_time(ring_buffer_[cur_index]->front()) < ts) {
                p_last = (cur_index + 1) % ring_size_;
                count -= step + 1;
            } else {
                count = step;
            }
        }
        if (p_last != ring_head_)
            p_last = (p_last + ring_size_ - 1) % ring_size_;

        // find last event with t >= ts
        auto it_beg     = ring_buffer_[p_last]->begin() + (p_last == ring_head_ ? next_buffer_ind_ : 0);
        auto it_end     = ring_buffer_[p_last]->end();
        auto it_ts      = std::lower_bound(it_beg, it_end, ev_ts,
                                      [](const Event &ev1, const Event &ev2) { return get_time(ev1) < get_time(ev2); });
        p_last_end_excl = std::distance(ring_buffer_[p_last]->begin(), it_ts);

        // from there, keep at most max_events events
        p_first = p_last;
        it_end  = ring_buffer_[p_last]->begin() + p_last_end_excl;
        if (p_first == ring_head_) {
            p_first_start_incl = next_buffer_ind_;
            keep_upto_max_events(ring_buffer_[p_first]->begin(), p_first_start_incl, p_last_end_excl, num_events,
                                 max_events);
        } else {
            p_first_start_incl = 0;
            if (!keep_upto_max_events(ring_buffer_[p_first]->begin(), p_first_start_incl, p_last_end_excl, num_events,
                                      max_events)) {
                do {
                    p_first = (p_first + ring_size_ - 1) % ring_size_;
                } while (!keep_upto_max_events(ring_buffer_[p_first]->begin(), p_first_start_incl,
                                               ring_buffer_[p_first]->size(), num_events, max_events) &&
                         p_first != ring_head_);
                if (num_events < max_events) {
                    // p_first == ring_head_
                    p_first_start_incl = next_buffer_ind_;
                    keep_upto_max_events(ring_buffer_[p_first]->begin(), p_first_start_incl,
                                         ring_buffer_[p_first]->size(), num_events, max_events);
                }
            }
        }

        // ----------------- FILL BUFFER ---------------------
        p_cur = p_first;
        if (p_cur != p_last) {
            it_beg = ring_buffer_[p_first]->begin() + p_first_start_incl;
            it_end = ring_buffer_[p_first]->end();
            std::copy(it_beg, it_end, d_first);
            p_cur = (p_cur + 1) % ring_size_;

            while (p_cur != p_last) {
                it_beg = ring_buffer_[p_cur]->begin();
                it_end = ring_buffer_[p_cur]->end();
                std::copy(it_beg, it_end, d_first);
                p_cur = (p_cur + 1) % ring_size_;
            }
        }

        // last vector to be added
        it_beg = ring_buffer_[p_last]->begin() + (p_last == p_first ? p_first_start_incl : 0);
        it_end = ring_buffer_[p_last]->begin() + p_last_end_excl;
        std::copy(it_beg, it_end, d_first);

        // update first vector of ring
        next_buffer_ind_ = p_last_end_excl;
        ring_head_       = p_last;
        if (p_last_end_excl == ring_buffer_[p_last]->size()) {
            next_buffer_ind_ = 0;
            ring_head_       = (ring_head_ + 1) % ring_size_;
        }
    }

    template<typename OutputIt>
    void fill_buffer_remaining(OutputIt d_first) {
        static_assert(std::is_same<Event, typename iterator_traits<OutputIt>::value_type>::value,
                      "fill_buffer_remaining called with invalid type of events.");

        std::unique_lock<std::mutex> lock(ring_mut_);
        if (ring_head_ != ring_next_) {
            std::copy(ring_buffer_[ring_head_]->cbegin() + next_buffer_ind_, ring_buffer_[ring_head_]->cend(), d_first);
            ring_head_       = (ring_head_ + 1) % ring_size_;
            next_buffer_ind_ = 0;
        }
        while (ring_head_ != ring_next_) {
            std::copy(ring_buffer_[ring_head_]->cbegin(), ring_buffer_[ring_head_]->cend(), d_first);
            ring_head_ = (ring_head_ + 1) % ring_size_;
            if (ring_buffer_[ring_head_]->empty())
                break;
        }
    }

    void drop() {
        std::unique_lock<std::mutex> lock(ring_mut_);
        ring_head_       = ring_next_;
        next_buffer_ind_ = 0;
    }

    void drop_max_events(int max_events) {
        if (!data_available()) {
            return;
        }

        std::unique_lock<std::mutex> lock(ring_mut_);
        unsigned int start_buffer = ring_next_;

        int num_events = 0;
        while (start_buffer != ring_head_ && num_events < max_events) {
            start_buffer = start_buffer > 0 ? start_buffer - 1 : ring_size_ - 1;
            num_events += ring_buffer_[start_buffer]->size();
        }
        int num_events_pre = num_events - ring_buffer_[start_buffer]->size();
        if (start_buffer == ring_head_)
            num_events -= next_buffer_ind_;
        else
            next_buffer_ind_ = 0;

        ring_head_ = start_buffer;

        if (num_events > max_events) {
            next_buffer_ind_ = ring_buffer_[ring_head_]->size() - (max_events - num_events_pre);
        }
    }

    void drop_to(timestamp ts) {
        if (!data_available(ts))
            return;
        std::unique_lock<std::mutex> lock(ring_mut_);

        if (get_time(ring_buffer_[ring_head_]->back()) < ts) {
            ring_head_       = (ring_head_ + 1) % ring_size_;
            next_buffer_ind_ = 0;

            while (get_time(ring_buffer_[ring_head_]->back()) < ts) {
                ring_head_ = (ring_head_ + 1) % ring_size_;
                if (ring_buffer_[ring_head_]->empty())
                    break;
            }
        }
        Event ev_ts;
        ev_ts.t          = ts;
        auto cbegin      = ring_buffer_[ring_head_]->cbegin();
        auto it_ts       = std::lower_bound(cbegin + next_buffer_ind_, ring_buffer_[ring_head_]->cend(), ev_ts,
                                      [](const Event &ev1, const Event &ev2) { return get_time(ev1) < get_time(ev2); });
        next_buffer_ind_ = static_cast<unsigned int>(std::distance(cbegin, it_ts));
    }

    void add(std::function<void(std::vector<Event> *)> &adder) {
        std::unique_lock<std::mutex> lock(ring_mut_);
        bool ovflow = false;
        if (ring_head_ == (ring_next_ + 1) % ring_size_) {
            if (max_ring_size_ && ring_size_ >= max_ring_size_) {
                ovflow = true;
            } else {
                double_ring();
            }
        } else {
            if (auto_shrink_ && buffer_delay() * 3 < ring_buffer_.capacity()) {
                shrink_ring();
            }
        }
        std::vector<Event> *buffer_to_write = ring_buffer_[ring_next_];
        if (!ovflow) {
            if (auto_shrink_ && buffer_to_write->size() * 3 < buffer_to_write->capacity()) {
                std::vector<Event>().swap(*buffer_to_write);
            } else {
                buffer_to_write->clear();
            }
        }
        try {
            adder(buffer_to_write);
        } catch (std::bad_alloc &) {}

        ring_next_ = (ring_next_ + 1) % ring_size_;
    }

    template<class IteratorEv>
    void add(IteratorEv start, IteratorEv end) {
        std::function<void(std::vector<Event> *)> adder = [start, end](std::vector<Event> *buf) {
            buf->insert(buf->end(), start, end);
        };
        add(adder);
    }

    void add(type_eventsadd &events) {
        std::function<void(std::vector<Event> *)> adder = [&events](std::vector<Event> *buf) {
            if (buf->size() != 0) {
                buf->insert(buf->end(), events.begin(), events.end());
            } else {
                std::swap(*buf, events);
            }
        };
        add(adder);
    }

    bool data_available() const {
        std::unique_lock<std::mutex> lock(ring_mut_);
        return data_available_internal();
    }

    bool data_available(timestamp ts) const {
        std::unique_lock<std::mutex> lock(ring_mut_);
        if (ring_head_ == ring_next_) {
            return false;
        }
        unsigned int last_buf_filled = (ring_next_ + ring_size_ - 1) % ring_size_;

        if (ring_buffer_[last_buf_filled]->empty()) {
            return false;
        }

        return get_time(ring_buffer_[last_buf_filled]->back()) >= ts;
    }

    size_t size() const {
        std::unique_lock<std::mutex> lock(ring_mut_);
        size_t s       = 0;
        unsigned int p = ring_head_;
        if (p != ring_next_) {
            s += std::distance(ring_buffer_[p]->begin() + next_buffer_ind_, ring_buffer_[p]->end());
            p = (p + 1) % ring_size_;
        }
        while (p != ring_next_) {
            s += std::distance(ring_buffer_[p]->begin(), ring_buffer_[p]->end());
            p = (p + 1) % ring_size_;
        }
        return s;
    }

    timestamp get_first_time() const {
        std::unique_lock<std::mutex> lock(ring_mut_);
        if (ring_head_ == ring_next_)
            return -1;

        if (ring_buffer_[ring_head_]->empty())
            return -1;

        return get_time(*(ring_buffer_[ring_head_]->begin() + next_buffer_ind_));
    }

    timestamp get_last_time() const {
        std::unique_lock<std::mutex> lock(ring_mut_);
        if (ring_head_ == ring_next_)
            return -1;

        unsigned int last_buf_filled = (ring_next_ + ring_size_ - 1) % ring_size_;

        if (ring_buffer_[last_buf_filled]->empty())
            return -1;

        return get_time(ring_buffer_[last_buf_filled]->back());
    }

    void clear() {
        std::unique_lock<std::mutex> lock(ring_mut_);
        for (auto v : ring_buffer_) {
            v->clear();
            ring_head_       = 0;
            ring_next_       = 0;
            next_buffer_ind_ = 0;
        }
    }
    void stat_ring(std::ostream &os) {
        os << std::dec << "RD : " << ring_buffer_.size() << " " << ring_buffer_.capacity() << " ";
        long t_cap = 0;
        long t_siz = 0;
        for (auto v : ring_buffer_) {
            if (v) {
                t_cap += v->capacity();
                t_siz += v->size();
            }
        }
        os << "IU : " << buffer_delay() << " ";
        os << " RH " << ring_head_ << " RN " << ring_next_ << " IDX " << next_buffer_ind_ << " ";
        if (ring_head_ == ring_next_)
            os << " E1 ";
        else {
            unsigned int last_buf_filled = (ring_next_ + ring_size_ - 1) % ring_size_;

            if (ring_buffer_[last_buf_filled]->empty())
                os << " E2 ";

            os << get_time(ring_buffer_[last_buf_filled]->back()) << " ";
        }
        os << "EV : " << t_cap << " " << t_siz << " " << std::endl;
    }

private:
    unsigned int ring_size_       = 0;
    unsigned int ring_head_       = 0;
    unsigned int ring_next_       = 0;
    unsigned int next_buffer_ind_ = 0;
    std::vector<std::vector<Event> *> ring_buffer_;
    unsigned int max_ring_size_ = 0;

    bool data_available_internal() const {
        return ring_head_ != ring_next_;
    }

    unsigned int buffer_delay() {
        if (ring_next_ > ring_head_) {
            return ring_next_ - ring_head_;
        } else {
            return ring_next_ + ring_size_ - ring_head_;
        }
    }

    void double_ring() {
        //        std::unique_lock<std::mutex> lock(ring_mut_);

        // Save current size of the buffer
        size_t old_size = ring_buffer_.capacity();

        // Increase ring buffer size
        ring_buffer_.resize(2 * old_size, nullptr);
        // Regroup data if the space was inserted in the middle of it
        if (ring_next_ < ring_head_) {
            for (unsigned int i = 0; i < ring_next_; i++) {
                ring_buffer_[old_size + i] = ring_buffer_[i];
                ring_buffer_[i]            = nullptr;
            }
            ring_next_ += old_size;
        }

        // Update size
        ring_size_ = ring_buffer_.capacity();

        // Create buffers in empty spots
        for (unsigned int i = 0; i < ring_size_; i++) {
            if (ring_buffer_[i] == nullptr)
                ring_buffer_[i] = new std::vector<Event>();
        }
    }

    void shrink_ring() {
        //        std::unique_lock<std::mutex> lock(ring_next_mut_);

        // Save current size of the buffer
        unsigned int old_size = ring_buffer_.capacity();
        unsigned int new_size = (old_size + 1) / 2;
        if (new_size < min_ring_capacity_ || new_size == ring_buffer_.capacity())
            return;
        std::vector<std::vector<Event> *> new_ring_buffer(new_size, nullptr);

        // Increase ring buffer size
        unsigned int idx           = ring_head_;
        unsigned int new_ring_next = 0;
        for (unsigned int i = 0; i < new_size; ++i) {
            new_ring_buffer[i] = ring_buffer_[idx];
            ring_buffer_[idx]  = 0;
            if (idx == ring_next_)
                new_ring_next = i;

            idx = (idx + 1) % old_size;
        }

        for (unsigned int i = 0; i < ring_size_; i++) {
            if (ring_buffer_[i])
                delete ring_buffer_[i];
        }

        ring_buffer_.swap(new_ring_buffer);
        ring_head_ = 0;
        ring_next_ = new_ring_next;
        ring_size_ = new_size;

        // Create buffers in empty spots
        for (unsigned int i = 0; i < ring_size_; i++) {
            if (ring_buffer_[i] == nullptr)
                ring_buffer_[i] = new std::vector<Event>();
        }
    }

    mutable std::mutex ring_mut_;
    unsigned int min_ring_capacity_;
    bool auto_shrink_;
    //    mutable std::mutex ring_next_mut_;
};

} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_RING_H
