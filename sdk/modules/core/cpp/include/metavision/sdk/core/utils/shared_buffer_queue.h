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

#ifndef METAVISION_SDK_CORE_SHARED_BUFFER_QUEUE_H
#define METAVISION_SDK_CORE_SHARED_BUFFER_QUEUE_H

#include <vector>
#include <memory>
#include <deque>

namespace Metavision {

/// @brief Class that implements a read-only FIFO container by wrapping read-only shared buffers
/// @tparam T The type of the elements contained in the shared buffers
/// @note This class aims to facilitate multi-threaded implementations by sharing buffers between components without
/// copies and explicit locking mechanisms and allowing those components to manipulate the shared buffers as a global
/// one. As the internal buffers are shared among several components they cannot by modified. As a consequence this
/// class implements a read-only FIFO and only exposes const iterators.
template<typename T>
class SharedBufferQueue {
private:
    /// @brief Structure pointing to the first and last elements of a shared buffer
    struct Range {
        explicit Range(const std::vector<T> &v);

        const T *first;
        const T *last;
    };

    using RangeIt = typename std::deque<Range>::iterator;

public:
    using Buffer       = std::vector<T>;
    using SharedBuffer = std::shared_ptr<const Buffer>;

    /// @brief A read-only random access iterator to elements in the shared queue
    class const_iterator {
    public:
        using difference_type   = std::ptrdiff_t;
        using value_type        = T;
        using reference         = const T &;
        using pointer           = const T *;
        using iterator_category = std::random_access_iterator_tag;

        friend class SharedBufferQueue;

        /// @brief Default constructor
        const_iterator();

        /// @brief Default copy constructor
        /// @param rhs The iterator to copy
        const_iterator(const const_iterator &rhs) = default;

        /// @brief Default move constructor
        /// @param rhs The iterator to move
        const_iterator(const_iterator &&rhs) noexcept = default;

        /// @brief Default copy assignment operator
        /// @param rhs The iterator to be copied
        /// @return *this
        const_iterator &operator=(const const_iterator &rhs) = default;

        /// @brief Default move assignment operator
        /// @param rhs The iterator to be moved
        /// @return *this
        const_iterator &operator=(const_iterator &&rhs) noexcept = default;

        /// @brief Increments the iterator
        /// @param rhs The increments number
        /// @return The incremented iterator
        /// @note The complexity is O(M) where M is the number of internal shared buffers
        inline const_iterator operator+(const difference_type &rhs) const;

        /// @brief Decrements the iterator
        /// @param rhs The decrements number
        /// @return The decremented iterator
        /// @note The complexity is O(M) where M is the number of internal shared buffers
        inline const_iterator operator-(const difference_type &rhs) const;

        /// @brief Increments the current instance
        /// @param rhs The increments number
        /// @return *this
        /// @note The complexity is O(M) where M is the number of internal shared buffers
        inline const_iterator &operator+=(const difference_type &rhs);

        /// @brief Decrements the current instance
        /// @param rhs The decrements number
        /// @return *this
        /// @note The complexity is O(M) where M is the number of internal shared buffers
        inline const_iterator &operator-=(const difference_type &rhs);

        /// @brief Returns the increments number between this instance and another one
        /// @param rhs The other iterator
        /// @return The increments number between the two iterators
        /// @note The complexity is O(M) where M is the number of internal shared buffers
        inline difference_type operator-(const const_iterator &rhs) const;

        /// @brief Dereferences the iterator
        /// @return A reference to the underlying element
        inline reference operator*() const;

        /// @brief Dereferences the iterator
        /// @return A pointer to the underlying element
        inline pointer operator->() const;

        /// @brief Array index operator
        /// @param rhs The increments number
        /// @return A reference to the underlying element of the incremented iterator
        /// @note The complexity is O(M) where M is the number of internal shared buffers
        inline reference operator[](const difference_type &rhs) const;

        /// @brief Pre-increment operator
        /// @return *this
        inline const_iterator &operator++();

        /// @brief Post-increment operator
        /// @return The incremented iterator
        inline const_iterator operator++(int);

        /// @brief Pre-decrement operator
        /// @return *this
        inline const_iterator &operator--();

        /// @brief Post-decrement operator
        /// @return The decremented operator
        inline const_iterator operator--(int);

        /// @brief Checks if two iterators are equal (i.e. they point to the same underlying element)
        /// @param rhs The other iterator
        /// @return True if the two iterator are equal
        inline bool operator==(const const_iterator &rhs) const;

        /// @brief Checks if two iterators are different (i.e. they don't point to the same underlying element)
        /// @param rhs The other iterator
        /// @return True if the two iterators are different
        inline bool operator!=(const const_iterator &rhs) const;

        /// @brief Checks if the current iterator is strictly greater than another one (i.e. the underlying element it
        /// points to is after the other iterator's in the shared queue)
        /// @param rhs The other iterator
        /// @return True if the current iterator is strictly greater than the other one
        inline bool operator>(const const_iterator &rhs) const;

        /// @brief Checks if the current iterator is strictly less than another one (i.e. the underlying element it
        /// points to is before the other iterator's in the shared queue)
        /// @param rhs The other iterator
        /// @return True if the current iterator is strictly less than the other one
        inline bool operator<(const const_iterator &rhs) const;

        /// @brief Checks if the current iterator is greater or equal to another one (i.e. the underlying element it
        /// points to is after or equal to the other iterator's in the shared queue)
        /// @param rhs The other iterator
        /// @return True if the current iterator is greater or equal to the other one
        inline bool operator>=(const const_iterator &rhs) const;

        /// @brief Checks if the current iterator is less or equal to another one (i.e. the underlying element it
        /// points to is before or equal to the other iterator's in the shared queue)
        /// @param rhs The other iterator
        /// @return True if the current iterator is before or equal to the other one
        inline bool operator<=(const const_iterator &rhs) const;

    private:
        inline void advance_forward(const difference_type &rhs);
        inline void advance_backward(const difference_type &rhs);

        explicit const_iterator(std::deque<Range> &r, bool make_end = false);

        pointer crt_;
        RangeIt range_it_;
        std::deque<Range> *ranges_;
    };

    /// @brief Returns an iterator to the beginning
    /// @return An iterator to the first element
    const_iterator cbegin() const;

    /// @brief Returns an iterator to the end
    /// @return An iterator to the element following the last element
    const_iterator cend() const;

    /// @brief Returns an iterator to the beginning
    /// @return An iterator to the first element
    const_iterator begin() const;

    /// @brief Returns an iterator to the end
    /// @return An iterator to the element following the last element
    const_iterator end() const;

    /// @brief Inserts a new shared buffer in the queue
    /// @param buffer The shared buffer to add to the queue
    /// @note Only the shared pointer is copied, not the buffer itself
    void insert(SharedBuffer buffer);

    /// @brief Erases all the elements before the provided one
    /// @param it Iterator to the first element to keep
    /// @note If some internal shared buffers are no longer used, they are released
    void erase_up_to(const_iterator it);

    /// @brief Clears the queue, all the internal shared buffers are released
    void clear();

    /// @brief Checks if the queue is empty
    /// @return True if the queue is empty
    bool empty() const;

    /// @brief Returns the size of the queue
    /// @return The size of the queue
    size_t size() const;

private:
    std::deque<SharedBuffer> buffer_queue_;
    std::deque<Range> ranges_;
};
} // namespace Metavision

#include "metavision/sdk/core/utils/detail/shared_buffer_queue_impl.h"

#endif // METAVISION_SDK_CORE_SHARED_BUFFER_QUEUE_H
