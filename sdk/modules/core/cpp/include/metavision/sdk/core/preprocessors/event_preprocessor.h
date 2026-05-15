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

#ifndef METAVISION_SDK_CORE_EVENT_PREPROCESSOR_H
#define METAVISION_SDK_CORE_EVENT_PREPROCESSOR_H

#include <memory>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/core/preprocessors/tensor.h"

namespace Metavision {

/// @brief Processes events to update data from a tensor.
///
/// This is the base class. It handles the rescaling of the events if necessary.
/// It also provides accessors to get the shape of the output tensor.
/// Derived class implement the computation.
/// Calling process_events() on this base class triggers the computation to update the provided tensor.
/// This tensor can typically be used as input of a neural network.
/// @tparam InputIt The type of the input iterator for the range of events to process
template<typename InputIt>
class EventPreprocessor {
public:
    /// @brief Retrieves the shape of the processor's output tensor.
    /// This shape can be used to initialize the output tensor provided to the @ref process_events method.
    /// @returns The output tensor shape
    const TensorShape &get_output_shape() const;

    /// @brief Retrieves the type of the processor's output tensor
    /// This type can be used to initialize the output tensor provided to the @ref process_events method.
    /// @returns The output tensor type
    BaseType get_output_type() const;

    /// @brief Updates the output tensor depending on the input events
    /// @param[in] cur_frame_start_ts starting timestamp of the current frame
    /// @param[in] begin Begin iterator
    /// @param[in] end End iterator
    /// @param[out] tensor Updated output tensor
    /// @warning The tensor needs to have its memory already allocated, which can be done thanks to the class @ref
    /// get_output_shape and
    /// @ref get_output_type methods and the Tensor method @ref Tensor::create.
    void process_events(const timestamp cur_frame_start_ts, InputIt begin, InputIt end, Tensor &tensor) const;

protected:
    /// @brief Constructor
    /// @param shape Shape of the output tensor to update with events
    /// @param type Type of the data contained in the output tensor to update
    EventPreprocessor(const TensorShape &shape, const BaseType &type);

    TensorShape output_tensor_shape_;
    const BaseType output_tensor_type_;

private:
    virtual void compute(const timestamp cur_frame_start_ts, InputIt begin, InputIt end, Tensor &tensor) const = 0;

    /// @brief Returns true if the provided tensor has the expected shape
    bool has_expected_shape(const Tensor &t) const;
};

} // namespace Metavision

#include "metavision/sdk/core/preprocessors/detail/event_preprocessor_impl.h"

#endif // METAVISION_SDK_CORE_EVENT_PREPROCESSOR_H
