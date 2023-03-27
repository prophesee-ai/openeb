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

#ifndef METAVISION_SDK_CORE_SHARED_BUFFER_QUEUE_IMPL_H
#define METAVISION_SDK_CORE_SHARED_BUFFER_QUEUE_IMPL_H

#include <cassert>

namespace Metavision {

template<typename T>
SharedBufferQueue<T>::Range::Range(const std::vector<T> &v) {
    assert(!v.empty());

    first = &v.front();
    last  = &v.back();
}

///////////////////// Iterator

template<typename T>
SharedBufferQueue<T>::const_iterator::const_iterator() {
    crt_    = nullptr;
    ranges_ = nullptr;
}

template<typename T>
SharedBufferQueue<T>::const_iterator::const_iterator(std::deque<Range> &r, bool make_end) : const_iterator() {
    if (!r.empty()) {
        ranges_ = &r;

        if (make_end) {
            range_it_ = std::prev(ranges_->end());
            crt_      = range_it_->last + 1;
        } else {
            range_it_ = ranges_->begin();
            crt_      = range_it_->first;
        }
    }
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator
    SharedBufferQueue<T>::const_iterator::operator+(const difference_type &rhs) const {
    auto it = *this;
    it += rhs;

    return it;
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator
    SharedBufferQueue<T>::const_iterator::operator-(const difference_type &rhs) const {
    auto it = *this;
    it -= rhs;

    return it;
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator &
    SharedBufferQueue<T>::const_iterator::operator+=(const difference_type &rhs) {
    if (rhs > 0) {
        advance_forward(rhs);
    } else {
        advance_backward(-rhs);
    }

    return *this;
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator &
    SharedBufferQueue<T>::const_iterator::operator-=(const difference_type &rhs) {
    if (rhs > 0) {
        advance_backward(rhs);
    } else {
        advance_forward(-rhs);
    }

    return *this;
}

template<typename T>
void SharedBufferQueue<T>::const_iterator::advance_forward(const difference_type &rhs) {
    assert(!ranges_->empty());

    auto dist                  = rhs;
    auto dist_to_last_in_range = std::distance(crt_, range_it_->last);
    auto last_range_it         = std::prev(ranges_->end());

    while (dist > dist_to_last_in_range && range_it_ < last_range_it) {
        ++range_it_;
        crt_ = range_it_->first;
        dist -= (dist_to_last_in_range + 1);
        dist_to_last_in_range = std::distance(range_it_->first, range_it_->last);
    }

    crt_ += dist;
}

template<typename T>
void SharedBufferQueue<T>::const_iterator::advance_backward(const difference_type &rhs) {
    assert(!ranges_->empty());

    auto dist                   = rhs;
    auto dist_to_first_in_range = std::distance(range_it_->first, crt_);

    while (dist > dist_to_first_in_range && range_it_ > ranges_->begin()) {
        --range_it_;
        crt_ = range_it_->last;
        dist -= (dist_to_first_in_range + 1);
        dist_to_first_in_range = std::distance(range_it_->first, range_it_->last);
    }

    crt_ -= dist;
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator::difference_type
    SharedBufferQueue<T>::const_iterator::operator-(const const_iterator &rhs) const {
    difference_type diff = 0;

    const auto is_less_than = *this < rhs;
    const auto &lower_it    = is_less_than ? *this : rhs;
    const auto &upper_it    = is_less_than ? rhs : *this;

    auto lower_range_it   = lower_it.range_it_;
    const auto *lower_ptr = lower_it.crt_;
    const auto *upper_ptr = upper_it.crt_;

    while (lower_range_it != upper_it.range_it_) {
        const auto *last_in_range = lower_range_it->last;
        diff += std::distance(lower_ptr, last_in_range) + 1;
        ++lower_range_it;
        lower_ptr = lower_range_it->first;
    }

    diff += std::distance(lower_ptr, upper_ptr);

    return diff;
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator::reference SharedBufferQueue<T>::const_iterator::operator*() const {
    return *crt_;
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator::pointer SharedBufferQueue<T>::const_iterator::operator->() const {
    return crt_;
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator::reference
    SharedBufferQueue<T>::const_iterator::operator[](const difference_type &rhs) const {
    auto it = *this + rhs;
    return *it;
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator &SharedBufferQueue<T>::const_iterator::operator++() {
    ++crt_;

    if (crt_ > range_it_->last && range_it_ < std::prev(ranges_->end())) {
        ++range_it_;
        crt_ = range_it_->first;
    }

    return *this;
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator SharedBufferQueue<T>::const_iterator::operator++(int) {
    auto it = *this;
    ++it;
    return it;
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator &SharedBufferQueue<T>::const_iterator::operator--() {
    --crt_;

    if (crt_ < range_it_->first && range_it_ > ranges_->begin()) {
        --range_it_;
        crt_ = range_it_->last;
    }

    return *this;
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator SharedBufferQueue<T>::const_iterator::operator--(int) {
    auto it = *this;
    --it;
    return it;
}

template<typename T>
bool SharedBufferQueue<T>::const_iterator::operator==(const const_iterator &rhs) const {
    return crt_ == rhs.crt_;
}

template<typename T>
bool SharedBufferQueue<T>::const_iterator::operator!=(const const_iterator &rhs) const {
    return crt_ != rhs.crt_;
}

template<typename T>
bool SharedBufferQueue<T>::const_iterator::operator>(const const_iterator &rhs) const {
    return (range_it_ > rhs.range_it_) || ((range_it_ == rhs.range_it_) && (crt_ > rhs.crt_));
}

template<typename T>
bool SharedBufferQueue<T>::const_iterator::operator<(const const_iterator &rhs) const {
    return (range_it_ < rhs.range_it_) || ((range_it_ == rhs.range_it_) && (crt_ < rhs.crt_));
}

template<typename T>
bool SharedBufferQueue<T>::const_iterator::operator>=(const const_iterator &rhs) const {
    return (range_it_ > rhs.range_it_) || ((range_it_ == rhs.range_it_) && (crt_ >= rhs.crt_));
}

template<typename T>
bool SharedBufferQueue<T>::const_iterator::operator<=(const const_iterator &rhs) const {
    return (range_it_ < rhs.range_it_) || ((range_it_ == rhs.range_it_) && (crt_ <= rhs.crt_));
}

///////////////////// SharedBufferQueue

template<typename T>
void SharedBufferQueue<T>::insert(typename SharedBufferQueue<T>::SharedBuffer buffer) {
    if (!buffer->empty()) {
        ranges_.emplace_back(Range(*buffer));
        buffer_queue_.emplace_back(std::move(buffer));
    }
}

template<typename T>
void SharedBufferQueue<T>::erase_up_to(const_iterator it) {
    if (it == cend()) {
        clear();
        return;
    }

    const auto distance = std::distance(ranges_.begin(), it.range_it_);
    ranges_.erase(ranges_.cbegin(), it.range_it_);
    buffer_queue_.erase(buffer_queue_.cbegin(), buffer_queue_.cbegin() + distance);

    ranges_.begin()->first = it.crt_;
}

template<typename T>
void SharedBufferQueue<T>::clear() {
    ranges_.clear();
    buffer_queue_ = {};
}

template<typename T>
bool SharedBufferQueue<T>::empty() const {
    return buffer_queue_.empty();
}

template<typename T>
size_t SharedBufferQueue<T>::size() const {
    return static_cast<size_t>(end() - begin());
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator SharedBufferQueue<T>::cbegin() const {
    // it is ok to const_cast here because we are returning a const_iterator anyway
    return const_iterator(const_cast<std::deque<Range> &>(ranges_));
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator SharedBufferQueue<T>::cend() const {
    // it is ok to const_cast here because we are returning a const_iterator anyway
    return const_iterator(const_cast<std::deque<Range> &>(ranges_), true);
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator SharedBufferQueue<T>::begin() const {
    // it is ok to const_cast here because we are returning a const_iterator anyway
    return const_iterator(const_cast<std::deque<Range> &>(ranges_));
}

template<typename T>
typename SharedBufferQueue<T>::const_iterator SharedBufferQueue<T>::end() const {
    // it is ok to const_cast here because we are returning a const_iterator anyway
    return const_iterator(const_cast<std::deque<Range> &>(ranges_), true);
}
} // namespace Metavision

#endif // METAVISION_SDK_CORE_SHARED_BUFFER_QUEUE_IMPL_H
