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

#ifndef METAVISION_SDK_CORE_DETAIL_INTERNAL_ALGORITHMS_H
#define METAVISION_SDK_CORE_DETAIL_INTERNAL_ALGORITHMS_H

#include <algorithm>
#include <string>
#include <type_traits>
#include <boost/preprocessor/cat.hpp>

#include "metavision/sdk/core/utils/detail/platform_utility_functions.h"

namespace Metavision {
namespace detail {

template<class InputIterator, class OutputIterator, class ConditionalPredicate, class TransformerPredicate>
OutputIterator transform_if(InputIterator first, InputIterator last, OutputIterator out,
                            ConditionalPredicate conditional, TransformerPredicate transformer) {
    for (; first != last; ++first) {
        if (conditional(*first)) {
            auto copy = *first;
            transformer(copy);
            *out = copy;
            ++out;
        }
    }
    return out;
}

template<class InputIterator, class OutputIterator, class ConditionalTransformerPredicate>
OutputIterator transform_if(InputIterator first, InputIterator last, OutputIterator out,
                            ConditionalTransformerPredicate conditional) {
    return transform_if(first, last, out, conditional, conditional);
}

template<class InputIterator, class OutputIterator, class Predicate>
OutputIterator transform(InputIterator first, InputIterator last, OutputIterator out, Predicate transformer) {
    return std::transform(first, last, out, [&](const auto &element) {
        auto copy = element;
        transformer(copy);
        return copy;
    });
}

struct Prefetch {
    Prefetch(const char *ptr, long stride) : ptr_(ptr), stride_(stride) {}

    template<class InputIterator, class OutputIterator, class ConditionalPredicate>
    OutputIterator operator()(InputIterator &first, InputIterator last, OutputIterator out,
                              ConditionalPredicate conditional) {
        const int prefetch_distance = 4;
        if (first + prefetch_distance < last) {
            for (; first != last - prefetch_distance; ++first) {
                auto future        = first + prefetch_distance;
                unsigned int index = ((future->y) * stride_) + (future->x);

                cross_platform_prefetch(reinterpret_cast<const void *>(ptr_ + index));

                if (conditional(*first)) {
                    *out = *first;
                    ++out;
                }
            }
        }
        return out;
    }
    const char *ptr_;
    long stride_;
};

template<class InputIterator, class OutputIterator, class ConditionalPredicate>
OutputIterator insert_if(InputIterator first, InputIterator last, OutputIterator out,
                         ConditionalPredicate conditional) {
    return std::copy_if(first, last, out, conditional);
}

template<class InputIterator, class OutputIterator, class ConditionalPredicate>
OutputIterator insert_if(InputIterator first, InputIterator last, OutputIterator out, ConditionalPredicate conditional,
                         Prefetch prefetch) {
    OutputIterator updated_out = prefetch(first, last, out, conditional);
    // Process remaining elements
    return insert_if(first, last, updated_out, conditional);
}
} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_INTERNAL_ALGORITHMS_H
