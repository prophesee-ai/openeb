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

#ifndef METAVISION_SDK_STREAM_SLICE_ITERATOR_H
#define METAVISION_SDK_STREAM_SLICE_ITERATOR_H

#include <memory>
#include "metavision/sdk/core/utils/concurrent_queue.h"

namespace Metavision {

/// @brief Iterator over slices
/// @tparam SliceT Type of the slice
template<typename SliceT>
class SliceIteratorT {
public:
    using value_type        = SliceT;
    using difference_type   = std::ptrdiff_t;
    using pointer           = SliceT *;
    using reference         = SliceT &;
    using iterator_category = std::input_iterator_tag;

    using QueuePtr = std::shared_ptr<ConcurrentQueue<SliceT>>;

    /// @brief Default constructor
    /// @param q A queue to retrieve slices from, if nullptr, the iterator will be invalid (i.e. end())
    explicit SliceIteratorT(QueuePtr q = nullptr);

    /// @brief Dereference operator
    /// @return A reference to the current slice
    reference operator*();

    /// @brief Dereference operator
    /// @return A pointer to the current slice
    pointer operator->();

    /// @brief Pre-increment operator
    /// @return A reference to this instance
    SliceIteratorT<SliceT> &operator++();

    /// @brief Post-increment operator
    /// @return A copy of this instance after the increment
    SliceIteratorT<SliceT> operator++(int);

    /// @brief Equality comparison operator
    /// @param other The other iterator to compare with
    /// @return True if the two iterators are equal, false otherwise
    bool operator==(const SliceIteratorT<SliceT> &other) const;

    /// @brief Inequality comparison operator
    /// @param other The other iterator to compare with
    /// @return True if the two iterators are different, false otherwise
    bool operator!=(const SliceIteratorT<SliceT> &other) const;

private:
    QueuePtr queue_;
    SliceT slice_;
};

} // namespace Metavision

#include "metavision/sdk/stream/detail/slice_iterator_impl.h"

#endif // METAVISION_SDK_STREAM_SLICE_ITERATOR_H
