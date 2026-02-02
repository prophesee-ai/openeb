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

#ifndef METAVISION_SDK_STREAM_SLICE_ITERATOR_IMPL_H
#define METAVISION_SDK_STREAM_SLICE_ITERATOR_IMPL_H

namespace Metavision {

template<typename SliceT>
SliceIteratorT<SliceT>::SliceIteratorT(std::shared_ptr<ConcurrentQueue<SliceT>> q) : queue_(std::move(q)) {
    ++(*this);
}

template<typename SliceT>
typename SliceIteratorT<SliceT>::reference SliceIteratorT<SliceT>::operator*() {
    return slice_;
}

template<typename SliceT>
typename SliceIteratorT<SliceT>::pointer SliceIteratorT<SliceT>::operator->() {
    return &slice_;
}

template<typename SliceT>
SliceIteratorT<SliceT> &SliceIteratorT<SliceT>::operator++() {
    if (queue_) {
        if (auto opt_slice = queue_->pop_front(); opt_slice) {
            slice_ = std::move(*opt_slice);
        } else {
            queue_ = nullptr;
            slice_ = SliceT{};
        }
    }
    return *this;
}

template<typename SliceT>
SliceIteratorT<SliceT> SliceIteratorT<SliceT>::operator++(int) {
    auto it = *this;
    ++it;
    return it;
}

template<typename SliceT>
bool SliceIteratorT<SliceT>::operator==(const SliceIteratorT<SliceT> &other) const {
    return queue_ == other.queue_ && slice_ == other.slice_;
}

template<typename SliceT>
bool SliceIteratorT<SliceT>::operator!=(const SliceIteratorT<SliceT> &other) const {
    return !(*this == other);
}

} // namespace Metavision

#endif // METAVISION_SDK_STREAM_SLICE_ITERATOR_IMPL_H
