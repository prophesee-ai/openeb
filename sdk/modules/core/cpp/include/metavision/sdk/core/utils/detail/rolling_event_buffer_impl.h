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

#ifndef METAVISION_SDK_CORE_DETAIL_ROLLING_EVENT_BUFFER_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_ROLLING_EVENT_BUFFER_IMPL_H

namespace Metavision {
namespace detail {

template<typename T>
RollingEventBufferIterator<T>::RollingEventBufferIterator() {
    rolling_buffer_ = nullptr;
    virtual_idx_    = 0;
}

template<typename T>
RollingEventBufferIterator<T>::RollingEventBufferIterator(RollingEventBuffer<value_type> &buffer, bool make_end) {
    rolling_buffer_ = &buffer;
    virtual_idx_    = rolling_buffer_->empty() ? 0 :
                      make_end                 ? to_virtual_index(static_cast<size_t>(rolling_buffer_->last_idx_)) + 1 :
                                                 to_virtual_index(static_cast<size_t>(rolling_buffer_->start_idx_));
}

template<typename T>
RollingEventBufferIterator<T>::RollingEventBufferIterator(const RollingEventBufferIterator<T> &other) :
    rolling_buffer_(other.rolling_buffer_), virtual_idx_(other.virtual_idx_) {}

template<typename T>
RollingEventBufferIterator<T> &RollingEventBufferIterator<T>::operator=(const RollingEventBufferIterator<T> &other) {
    rolling_buffer_ = other.rolling_buffer_;
    virtual_idx_    = other.virtual_idx_;

    return *this;
}

template<typename T>
RollingEventBufferIterator<T> &RollingEventBufferIterator<T>::operator++() {
    ++virtual_idx_;
    return *this;
}

template<typename T>
RollingEventBufferIterator<T> RollingEventBufferIterator<T>::operator++(int) {
    RollingEventBufferIterator<T> tmp = *this;
    ++(*this);
    return tmp;
}

template<typename T>
RollingEventBufferIterator<T> &RollingEventBufferIterator<T>::operator--() {
    --virtual_idx_;
    return *this;
}

template<typename T>
RollingEventBufferIterator<T> RollingEventBufferIterator<T>::operator--(int) {
    RollingEventBufferIterator<T> tmp = *this;
    --(*this);
    return tmp;
}

template<typename T>
RollingEventBufferIterator<T> &RollingEventBufferIterator<T>::operator+=(difference_type n) {
    virtual_idx_ += n;
    return *this;
}

template<typename T>
RollingEventBufferIterator<T> &RollingEventBufferIterator<T>::operator-=(difference_type n) {
    virtual_idx_ -= n;
    return *this;
}

template<typename T>
RollingEventBufferIterator<T> RollingEventBufferIterator<T>::operator+(difference_type n) const {
    RollingEventBufferIterator<T> tmp = *this;
    tmp += n;
    return tmp;
}

template<typename T>
RollingEventBufferIterator<T> RollingEventBufferIterator<T>::operator-(difference_type n) const {
    RollingEventBufferIterator<T> tmp = *this;
    tmp -= n;
    return tmp;
}

template<typename T>
typename RollingEventBufferIterator<T>::difference_type
    RollingEventBufferIterator<T>::operator-(const RollingEventBufferIterator &other) const {
    return static_cast<difference_type>(virtual_idx_) - other.virtual_idx_;
}

template<typename T>
typename RollingEventBufferIterator<T>::reference RollingEventBufferIterator<T>::operator*() const {
    return rolling_buffer_->data_[to_real_index()];
}

template<typename T>
typename RollingEventBufferIterator<T>::pointer RollingEventBufferIterator<T>::operator->() const {
    return &rolling_buffer_->data_[to_real_index()];
}

template<typename T>
bool RollingEventBufferIterator<T>::operator==(const RollingEventBufferIterator &other) const {
    return rolling_buffer_ == other.rolling_buffer_ && virtual_idx_ == other.virtual_idx_;
}

template<typename T>
bool RollingEventBufferIterator<T>::operator!=(const RollingEventBufferIterator &other) const {
    return (rolling_buffer_ != other.rolling_buffer_) || (virtual_idx_ != other.virtual_idx_);
}

template<typename T>
bool RollingEventBufferIterator<T>::operator<(const RollingEventBufferIterator &other) const {
    return virtual_idx_ < other.virtual_idx_;
}

template<typename T>
bool RollingEventBufferIterator<T>::operator<=(const RollingEventBufferIterator &other) const {
    return virtual_idx_ <= other.virtual_idx_;
}

template<typename T>
bool RollingEventBufferIterator<T>::operator>(const RollingEventBufferIterator &other) const {
    return virtual_idx_ > other.virtual_idx_;
}

template<typename T>
bool RollingEventBufferIterator<T>::operator>=(const RollingEventBufferIterator &other) const {
    return virtual_idx_ >= other.virtual_idx_;
}

template<typename T>
size_t RollingEventBufferIterator<T>::to_virtual_index(size_t idx) const {
    // this method is never called when the indices are -1 (i.e. when the buffer is empty), so this is safe
    const auto start_idx = static_cast<size_t>(rolling_buffer_->start_idx_);

    return idx >= start_idx ? idx - start_idx : rolling_buffer_->data_.size() - start_idx + idx;
}

template<typename T>
size_t RollingEventBufferIterator<T>::to_real_index() const {
    return (rolling_buffer_->start_idx_ + virtual_idx_) % rolling_buffer_->data_.size();
}
} // namespace detail

RollingEventBufferConfig RollingEventBufferConfig::make_n_events(std::size_t n_events) {
    return {RollingEventBufferMode::N_EVENTS, 0, n_events};
}

RollingEventBufferConfig RollingEventBufferConfig::make_n_us(Metavision::timestamp n_us) {
    return {RollingEventBufferMode::N_US, n_us, 0};
}

template<typename T>
RollingEventBuffer<T>::RollingEventBuffer(const RollingEventBufferConfig &config) : config_(config) {
    clear();
}

template<typename T>
RollingEventBuffer<T>::RollingEventBuffer(const RollingEventBuffer<T> &other) :
    config_(other.config_),
    data_(other.data_),
    virtual_size_(other.virtual_size_),
    start_idx_(other.start_idx_),
    last_idx_(other.last_idx_) {}

template<typename T>
RollingEventBuffer<T>::RollingEventBuffer(RollingEventBuffer<T> &&other) :
    config_(std::move(other.config_)),
    data_(std::move(other.data_)),
    virtual_size_(other.virtual_size_),
    start_idx_(other.start_idx_),
    last_idx_(other.last_idx_) {}

template<typename T>
RollingEventBuffer<T> &RollingEventBuffer<T>::operator=(const RollingEventBuffer<T> &other) {
    config_       = other.config_;
    data_         = other.data_;
    virtual_size_ = other.virtual_size_;
    start_idx_    = other.start_idx_;
    last_idx_     = other.last_idx_;

    return *this;
}

template<typename T>
RollingEventBuffer<T> &RollingEventBuffer<T>::operator=(RollingEventBuffer<T> &&other) {
    config_       = std::move(other.config_);
    data_         = std::move(other.data_);
    virtual_size_ = other.virtual_size_;
    start_idx_    = other.start_idx_;
    last_idx_     = other.last_idx_;

    return *this;
}

template<typename T>
template<typename InputIt>
void RollingEventBuffer<T>::insert_events(InputIt begin, InputIt end) {
    if (begin == end)
        return;

    if (config_.mode == RollingEventBufferMode::N_EVENTS) {
        insert_n_events_slice(begin, end);
    } else {
        insert_n_us_slice(begin, end);
    }
}

template<typename T>
template<typename InputIt>
void RollingEventBuffer<T>::insert_n_events_slice(InputIt begin, InputIt end) {
    virtual_size_ += std::distance(begin, end);
    virtual_size_       = std::min(virtual_size_, config_.delta_n_events);
    const auto max_size = static_cast<std::int64_t>(config_.delta_n_events);

    for (auto it = begin; it != end; ++it) {
        last_idx_        = (last_idx_ + 1 == max_size) ? 0 : last_idx_ + 1;
        data_[last_idx_] = *it;
    }

    const auto is_full = (virtual_size_ == config_.delta_n_events);
    start_idx_         = is_full ? (last_idx_ + 1) % max_size : 0;
}

template<typename T>
template<typename InputIt>
void RollingEventBuffer<T>::insert_n_us_slice(InputIt begin, InputIt end) {
    // compute the timestamps of the new rolling window
    const auto new_end_ts   = std::prev(end)->t;
    const auto new_start_ts = (new_end_ts < config_.delta_ts) ? 0 : new_end_ts - config_.delta_ts;

    // the beginning of the rolling window can either be located in the current slice or in the given one, so
    // let's try to find the beginning of the rolling window in both of them
    const auto crt_slice_begin =
        std::lower_bound(cbegin(), cend(), new_start_ts, [](const auto &ev, timestamp t) { return ev.t < t; });
    const auto new_slice_begin =
        std::lower_bound(begin, end, new_start_ts, [](const auto &ev, timestamp t) { return ev.t < t; });

    // compute the total number of events in the rolling window (number of valid events in the current slice + number of
    // valid events in the given one)
    const auto crt_size = static_cast<std::int64_t>(data_.size());
    const auto new_size = std::distance(crt_slice_begin, cend()) + std::distance(new_slice_begin, end);

    if (new_size > crt_size) {
        // if the current buffer is too small to contain the rolling window we allocate a new one and copy all the
        // valid events
        std::vector<T> tmp;
        tmp.reserve(new_size);
        tmp.insert(tmp.end(), crt_slice_begin, cend());
        tmp.insert(tmp.end(), new_slice_begin, end);
        std::swap(tmp, data_);
        start_idx_ = 0;
        last_idx_  = new_size - 1;
    } else {
        // otherwise we copy the new valid events and recompute the index of the first valid event based on the size of
        // the new rolling window
        for (auto it = new_slice_begin; it != end; ++it) {
            last_idx_        = (last_idx_ + 1 == crt_size) ? 0 : last_idx_ + 1;
            data_[last_idx_] = *it;
        }

        start_idx_ = (last_idx_ < new_size - 1) ? crt_size - new_size + last_idx_ + 1 : last_idx_ - (new_size - 1);
    }

    virtual_size_ = new_size;
}

template<typename T>
size_t RollingEventBuffer<T>::size() const {
    return virtual_size_;
}

template<typename T>
size_t RollingEventBuffer<T>::capacity() const {
    return data_.size();
}

template<typename T>
bool RollingEventBuffer<T>::empty() const {
    return (virtual_size_ == 0);
}

template<typename T>
void RollingEventBuffer<T>::clear() {
    virtual_size_ = 0;
    start_idx_    = -1;
    last_idx_     = -1;

    data_.clear();

    if (config_.mode == RollingEventBufferMode::N_EVENTS) {
        data_.resize(config_.delta_n_events);
    }
}

template<typename T>
const T &RollingEventBuffer<T>::operator[](size_t idx) const {
    const size_t real_idx = (static_cast<size_t>(start_idx_) + idx) % data_.size();
    return data_[real_idx];
}

template<typename T>
T &RollingEventBuffer<T>::operator[](size_t idx) {
    const size_t real_idx = (static_cast<size_t>(start_idx_) + idx) % data_.size();
    return data_[real_idx];
}

template<typename T>
typename RollingEventBuffer<T>::iterator RollingEventBuffer<T>::begin() {
    return iterator(*this);
}

template<typename T>
typename RollingEventBuffer<T>::iterator RollingEventBuffer<T>::end() {
    return iterator(*this, true);
}

template<typename T>
typename RollingEventBuffer<T>::const_iterator RollingEventBuffer<T>::begin() const {
    return const_iterator(const_cast<RollingEventBuffer<T> &>(*this));
}

template<typename T>
typename RollingEventBuffer<T>::const_iterator RollingEventBuffer<T>::end() const {
    return const_iterator(const_cast<RollingEventBuffer<T> &>(*this), true);
}

template<typename T>
typename RollingEventBuffer<T>::const_iterator RollingEventBuffer<T>::cbegin() const {
    return const_iterator(const_cast<RollingEventBuffer<T> &>(*this));
}

template<typename T>
typename RollingEventBuffer<T>::const_iterator RollingEventBuffer<T>::cend() const {
    return const_iterator(const_cast<RollingEventBuffer<T> &>(*this), true);
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_ROLLING_EVENT_BUFFER_IMPL_H
