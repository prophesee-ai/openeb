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

#ifndef METAVISION_SDK_CORE_ROLLING_EVENT_BUFFER_H
#define METAVISION_SDK_CORE_ROLLING_EVENT_BUFFER_H

#include <cstdint>
#include <cstddef>
#include <vector>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

// Forward declaration of the RollingEventBuffer class
template<typename T>
class RollingEventBuffer;

namespace detail {

/// @brief Iterator to an event in a rolling event buffer
/// @tparam T Type of the events stored in the rolling event buffer
template<typename T>
class RollingEventBufferIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = std::remove_const_t<T>;
    using difference_type   = std::ptrdiff_t;
    using pointer           = T *;
    using reference         = T &;

    /// @brief Default constructor
    RollingEventBufferIterator();

    /// @brief Copy constructor
    /// @param other The iterator to copy
    RollingEventBufferIterator(const RollingEventBufferIterator &other);

    /// @brief Copy assignment operator
    /// @param other The iterator to copy
    /// @return *this
    RollingEventBufferIterator &operator=(const RollingEventBufferIterator &other);

    /// @brief Pre-increment operator
    /// @return *this
    RollingEventBufferIterator &operator++();

    /// @brief Post-increment operator
    /// @return The incremented iterator
    RollingEventBufferIterator operator++(int);

    /// @brief Pre-decrement operator
    /// @return *this
    RollingEventBufferIterator &operator--();

    /// @brief Post-decrement operator
    /// @return The decremented operator
    RollingEventBufferIterator operator--(int);

    /// @brief Increments the current instance
    /// @param n The increments number
    /// @return *this
    /// @note The complexity is O(1)
    RollingEventBufferIterator &operator+=(difference_type n);

    /// @brief Decrements the current instance
    /// @param n The decrements number
    /// @return *this
    /// @note The complexity is O(1)
    RollingEventBufferIterator &operator-=(difference_type n);

    /// @brief Increments the iterator
    /// @param n The increments number
    /// @return The incremented iterator
    /// @note The complexity is O(1)
    RollingEventBufferIterator operator+(difference_type n) const;

    /// @brief Decrements the iterator
    /// @param n The decrements number
    /// @return The decremented iterator
    /// @note The complexity is O(1)
    RollingEventBufferIterator operator-(difference_type n) const;

    /// @brief Returns the increments number between this instance and another one
    /// @param other The other iterator
    /// @return The increments number between the two iterators
    /// @note The complexity is O(1)
    difference_type operator-(const RollingEventBufferIterator &other) const;

    /// @brief Dereferences the iterator
    /// @return A reference to the underlying event
    reference operator*() const;

    /// @brief Dereferences the iterator
    /// @return A pointer to the underlying event
    pointer operator->() const;

    /// @brief Checks if two iterators are equal (i.e. they point to the same underlying event)
    /// @param other The other iterator
    /// @return True if the two iterators are equal
    bool operator==(const RollingEventBufferIterator &other) const;

    /// @brief Checks if two iterators are different (i.e. they don't point to the same underlying event)
    /// @param other The other iterator
    /// @return True if the two iterators are different
    bool operator!=(const RollingEventBufferIterator &other) const;

    /// @brief Checks if the current iterator is strictly less than another one (i.e. the underlying event pointed by
    /// this iterator is ordered before the event pointed by the other iterator)
    /// @param other The other iterator
    /// @return True if the current iterator is strictly less than the other one
    bool operator<(const RollingEventBufferIterator &other) const;

    /// @brief Checks if the current iterator is less or equal to another one (i.e. the underlying event pointed by
    /// this iterator is either the same or is ordered before the event pointed by the other iterator)
    /// @param other The other iterator
    /// @return True if the current iterator is less or equal to the other one
    bool operator<=(const RollingEventBufferIterator &other) const;

    /// @brief Checks if the current iterator is strictly greater than another one (i.e. the underlying event pointed by
    /// this iterator is ordered after the event pointed by the other iterator)
    /// @param other The other iterator
    /// @return True if the current iterator is strictly greater than the other one
    bool operator>(const RollingEventBufferIterator &other) const;

    /// @brief Checks if the current iterator is greater or equal to another one (i.e. the underlying event pointed by
    /// this iterator is either the same or is ordered after the event pointed by the other iterator)
    /// @param other The other iterator
    /// @return True if the current iterator is greater or equal to the other one
    bool operator>=(const RollingEventBufferIterator &other) const;

private:
    friend class RollingEventBuffer<value_type>;

    RollingEventBufferIterator(RollingEventBuffer<value_type> &buffer, bool make_end = false);

    size_t to_virtual_index(size_t idx) const;
    size_t to_real_index() const;

    RollingEventBuffer<value_type> *rolling_buffer_;
    size_t virtual_idx_;
};
} // namespace detail

/// @brief Enumeration defining the rolling buffer mode
enum class RollingEventBufferMode {
    N_US,    ///< The buffer stores a fixed duration of events in microseconds
    N_EVENTS ///< The buffer stores a fixed number of events
};

/// @brief Configuration structure for the @ref RollingEventBuffer class
struct RollingEventBufferConfig {
    RollingEventBufferMode mode; ///< Mode of the rolling buffer
    timestamp delta_ts;          ///< Time slice duration in microseconds (for N_US mode)
    std::size_t delta_n_events;  ///< Number of events to store (for N_EVENTS mode)

    /// @brief Creates a @ref RollingEventBufferConfig for the N_US mode
    /// @param n_us Time slice duration in microseconds
    /// @return a @ref RollingEventBufferConfig for the N_US mode
    static inline RollingEventBufferConfig make_n_us(timestamp n_us);

    /// @brief Creates a @ref RollingEventBufferConfig for the N_EVENTS mode
    /// @param n_events Number of events to store
    /// @return a @ref RollingEventBufferConfig for the N_EVENTS mode
    static inline RollingEventBufferConfig make_n_events(std::size_t n_events);
};

/// @brief A rolling buffer that can store events based on a fixed duration or a fixed number of events
///
/// This class provides a rolling buffer capable of storing events. It can operate in
/// two modes: fixed duration (N_US) or fixed number of events (N_EVENTS). In the N_US mode, the buffer automatically
/// adjusts its size to store events corresponding to a fixed-duration time slice. In the N_EVENTS mode, it stores a
/// fixed number of events, replacing the oldest events when new ones arrive.
/// @tparam T The type of events to store in the buffer
template<typename T>
class RollingEventBuffer {
public:
    using iterator       = detail::RollingEventBufferIterator<T>;
    using const_iterator = detail::RollingEventBufferIterator<const T>;

    /// @brief Constructs a RollingEventBuffer with the specified configuration
    /// @param config The configuration for the rolling buffer
    RollingEventBuffer(const RollingEventBufferConfig &config = RollingEventBufferConfig::make_n_events(5000));

    /// @brief Copy constructor
    /// @param other The instance to copy
    RollingEventBuffer(const RollingEventBuffer &other);

    /// @brief Move constructor
    /// @param other The instance to move
    RollingEventBuffer(RollingEventBuffer &&other);

    /// @brief Copy assignment operator
    /// @param other The instance to copy
    /// @return *this
    RollingEventBuffer &operator=(const RollingEventBuffer &other);

    /// @brief Move assignment operator
    /// @param other The instance to move
    /// @return *this
    RollingEventBuffer &operator=(RollingEventBuffer &&other);

    /// @brief Inserts events into the buffer
    ///
    /// This function inserts events into the buffer based on the current mode (N_US or N_EVENTS)
    /// @tparam InputIt Iterator type for the events
    /// @param begin Iterator pointing to the beginning of the events range
    /// @param end Iterator pointing to the end of the events range
    template<typename InputIt>
    void insert_events(InputIt begin, InputIt end);

    /// @brief Returns the current number of events stored in the buffer
    /// @return The number of events stored
    size_t size() const;

    /// @brief Returns the maximum capacity of the buffer
    /// @return The maximum capacity of the buffer
    size_t capacity() const;

    /// @brief Checks if the buffer is empty
    /// @return `true` if the buffer is empty, `false` otherwise
    bool empty() const;

    /// @brief Clears the buffer, removing all stored events
    void clear();

    /// @brief Accesses events in the buffer using the [] operator
    /// @param idx idx The index of the event to access
    /// @return A reference to the event at the specified index
    const T &operator[](size_t idx) const;

    /// @brief Accesses events in the buffer using the [] operator
    /// @param idx idx The index of the event to access
    /// @return A reference to the event at the specified index
    T &operator[](size_t idx);

    /// @brief Returns an iterator pointing to the beginning of the buffer
    /// @return An iterator pointing to the beginning of the buffer
    iterator begin();

    /// @brief Returns an iterator pointing to the end of the buffer
    /// @return An iterator pointing to the end of the buffer
    iterator end();

    /// @brief Returns a const iterator pointing to the beginning of the buffer
    /// @return A const iterator pointing to the beginning of the buffer
    const_iterator begin() const;

    /// @brief Returns a const iterator pointing to the end of the buffer
    /// @return A const iterator pointing to the end of the buffer
    const_iterator end() const;

    /// @brief Returns a const iterator pointing to the beginning of the buffer
    /// @return A const iterator pointing to the beginning of the buffer
    const_iterator cbegin() const;

    /// @brief Returns a const iterator pointing to the end of the buffer
    /// @return A const iterator pointing to the end of the buffer
    const_iterator cend() const;

private:
    friend class detail::RollingEventBufferIterator<T>;
    friend class detail::RollingEventBufferIterator<const T>;

    template<typename InputIt>
    void insert_n_us_slice(InputIt begin, InputIt end);

    template<typename InputIt>
    void insert_n_events_slice(InputIt begin, InputIt end);

    RollingEventBufferConfig config_;
    std::vector<T> data_;
    size_t virtual_size_;
    std::int64_t start_idx_;
    std::int64_t last_idx_;
};

} // namespace Metavision

#include "metavision/sdk/core/utils/detail/rolling_event_buffer_impl.h"

#endif // METAVISION_SDK_CORE_ROLLING_EVENT_BUFFER_H
