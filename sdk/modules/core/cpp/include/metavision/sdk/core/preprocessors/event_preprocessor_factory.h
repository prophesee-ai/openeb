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

#ifndef METAVISION_SDK_CORE_EVENT_PREPROCESSOR_FACTORY_H
#define METAVISION_SDK_CORE_EVENT_PREPROCESSOR_FACTORY_H

#include <memory>

#include "metavision/sdk/core/preprocessors/event_preprocessor.h"

namespace Metavision {

namespace EventPreprocessorFactory {

/// @brief Creates the event processor in adequation with the parameters map describing the event processing to apply
/// @details The map should contain a 'type' key, describing the type of preprocessor to create, as well as the
/// necessary parameters for the selected processor. See the different types and required dictionnary parameters below.
///
/// A 'DIFF' instance requires:
///     - max_incr_per_pixel (float)
///     - clip_value_after_normalization (float)
///     - scale_width (float)
///     - scale_height (float)
///
/// A 'HISTO' instance requires:
///     - max_incr_per_pixel (float)
///     - clip_value_after_normalization (float)
///     - use_CHW (bool)
///     - scale_width (float)
///     - scale_height (float)
///
/// An 'EVENT_CUBE' instance requires:
///     - delta_t (timestamp)
///     - max_incr_per_pixel (float)
///     - clip_value_after_normalization (float)
///     - num_utbins (int)
///     - split_polarity (bool)
///     - scale_width (float)
///     - scale_height (float)
///
/// A 'HARDWARE_DIFF' instance requires:
///     - min_val (int8_t)
///     - max_val (int8_t)
///     - allow_rollover (bool)
///
/// A 'HARDWARE_HISTO' instance requires:
///     - neg_saturation (uint8_t)
///     - pos_saturation (uint8_t)
///     - allow_rollover (bool)
///
/// A 'TIME_SURFACE' instance requires:
///     - nb_channels (uint8_t)
///
/// @tparam InputIt The type of the input iterator for the range of events to process
/// @param proc_params A dictionnary containing parameters describing the processor to instantiate
/// @param tensor_shape Shape of the tensor to fill with preprocessed data (it must match the dimensions of the events
/// which will be passed to the processor). It is expected to provide at least "H" and "W" dimensions.
template<typename InputIt>
std::unique_ptr<EventPreprocessor<InputIt>>
    create(const std::unordered_map<std::string, PreprocessingParameters> &proc_params,
           const TensorShape &tensor_shape);

} // namespace EventPreprocessorFactory

} // namespace Metavision

#include "metavision/sdk/core/preprocessors/detail/event_preprocessor_factory_impl.h"

#endif // METAVISION_SDK_CORE_EVENT_PREPROCESSOR_FACTORY_H
