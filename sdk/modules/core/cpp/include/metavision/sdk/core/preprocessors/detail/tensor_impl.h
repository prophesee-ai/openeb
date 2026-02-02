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

#ifndef METAVISION_SDK_CORE_PREPROCESSORS_TENSOR_IMPL_H
#define METAVISION_SDK_CORE_PREPROCESSORS_TENSOR_IMPL_H

#include <stdexcept>

#include "metavision/sdk/core/preprocessors/tensor.h"

namespace Metavision {

template<typename T>
const T *Tensor::data() const {
    return reinterpret_cast<const T *>(ptr_);
}

template<typename T>
T *Tensor::data() {
    return reinterpret_cast<T *>(ptr_);
}

template<typename T>
void Tensor::set_to_impl(T typed_val, size_t n) {
    auto ptr = reinterpret_cast<T *>(ptr_);
    std::fill(ptr, ptr + n, typed_val);
}

template<typename T>
void Tensor::set_to(T val) {
    const size_t n = shape_.get_nb_values();
    switch (type_) {
    case BaseType::BOOL:
        set_to_impl(static_cast<bool>(val), n);
        break;
    case BaseType::UINT8:
        set_to_impl(static_cast<uint8_t>(val), n);
        break;
    case BaseType::UINT16:
        set_to_impl(static_cast<uint16_t>(val), n);
        break;
    case BaseType::UINT32:
        set_to_impl(static_cast<uint32_t>(val), n);
        break;
    case BaseType::UINT64:
        set_to_impl(static_cast<uint64_t>(val), n);
        break;
    case BaseType::INT8:
        set_to_impl(static_cast<int8_t>(val), n);
        break;
    case BaseType::INT16:
        set_to_impl(static_cast<int16_t>(val), n);
        break;
    case BaseType::INT32:
        set_to_impl(static_cast<int32_t>(val), n);
        break;
    case BaseType::INT64:
        set_to_impl(static_cast<int64_t>(val), n);
        break;
    case BaseType::FLOAT32:
        set_to_impl(static_cast<float>(val), n);
        break;
    case BaseType::FLOAT64:
        set_to_impl(static_cast<double>(val), n);
        break;
    case BaseType::FLOAT16:
        throw std::runtime_error("No implementation yet for 2-byte float");
    default:
        throw std::runtime_error("Provided data type not managed.");
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_PREPROCESSORS_TENSOR_IMPL_H
