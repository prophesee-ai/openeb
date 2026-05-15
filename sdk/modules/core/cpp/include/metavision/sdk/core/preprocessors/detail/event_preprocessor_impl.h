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

#ifndef METAVISION_SDK_CORE_DETAIL_EVENT_PREPROCESSOR_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_EVENT_PREPROCESSOR_IMPL_H

#include "metavision/sdk/core/preprocessors/json_parser.h"

#include "metavision/sdk/core/preprocessors/event_preprocessor.h"

namespace Metavision {

template<typename InputIt>
EventPreprocessor<InputIt>::EventPreprocessor(const TensorShape &shape, const BaseType &type) :
    output_tensor_shape_(shape), output_tensor_type_(type) {
    const auto output_width  = get_dim(shape, "W");
    const auto output_height = get_dim(shape, "H");
    if ((output_width < 1) || (output_height < 1)) {
        std::ostringstream oss;
        oss << "EventPreprocessor : invalid value for provided output frame shape (width and height must be >= 1): ";
        oss << output_width << "x" << output_height << std::endl;
        throw std::invalid_argument(oss.str());
    }
}

template<typename InputIt>
const TensorShape &EventPreprocessor<InputIt>::get_output_shape() const {
    return this->output_tensor_shape_;
}

template<typename InputIt>
BaseType EventPreprocessor<InputIt>::get_output_type() const {
    return this->output_tensor_type_;
}

template<typename InputIt>
bool EventPreprocessor<InputIt>::has_expected_shape(const Tensor &t) const {
    const auto &actual_dimensions   = t.shape().dimensions;
    const auto &expected_dimensions = this->output_tensor_shape_.dimensions;
    const size_t n_actual           = actual_dimensions.size();
    const size_t n_expected         = expected_dimensions.size();
    unsigned int i = 0, j = 0;
    while (i < n_actual) {
        if (actual_dimensions[i].dim > 1) {
            const auto &name = actual_dimensions[i].name;
            while (j < n_expected && expected_dimensions[j].dim <= 1)
                ++j;
            if (j == n_expected || name != expected_dimensions[j].name ||
                actual_dimensions[i].dim != expected_dimensions[j].dim)
                return false;
            ++i;
            ++j;
        } else {
            ++i;
        }
    }
    return true;
}

template<typename InputIt>
void EventPreprocessor<InputIt>::process_events(const timestamp cur_frame_start_ts, InputIt begin, InputIt end,
                                                Tensor &tensor) const {
    if (begin == end) {
        return;
    }

    if (!has_expected_shape(tensor)) {
        std::stringstream msg;
        msg << "Incompatible tensor provided : expected shape " << this->output_tensor_shape_ << " but got  shape "
            << tensor.shape() << std::endl;
        throw std::runtime_error(msg.str());
    }

    compute(cur_frame_start_ts, begin, end, tensor);
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_EVENT_PREPROCESSOR_IMPL_H
