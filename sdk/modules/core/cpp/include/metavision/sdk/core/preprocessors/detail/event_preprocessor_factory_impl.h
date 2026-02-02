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

#ifndef METAVISION_SDK_CORE_DETAIL_EVENT_PREPROCESSOR_FACTORY_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_EVENT_PREPROCESSOR_FACTORY_IMPL_H

#include <boost/property_tree/ptree.hpp>

#include "metavision/sdk/core/preprocessors/event_preprocessor_type.h"
#include "metavision/sdk/core/preprocessors/json_parser.h"

#include "metavision/sdk/core/preprocessors/diff_processor.h"
#include "metavision/sdk/core/preprocessors/event_cube_processor.h"
#include "metavision/sdk/core/preprocessors/hardware_diff_processor.h"
#include "metavision/sdk/core/preprocessors/hardware_histo_processor.h"
#include "metavision/sdk/core/preprocessors/histo_processor.h"
#include "metavision/sdk/core/preprocessors/time_surface_processor.h"

#include "metavision/sdk/core/preprocessors/event_preprocessor_factory.h"

namespace Metavision {

namespace EventPreprocessorFactory {

template<typename InputIt>
std::unique_ptr<EventPreprocessor<InputIt>>
    create(const std::unordered_map<std::string, PreprocessingParameters> &proc_params,
           const TensorShape &tensor_shape) {
    const int evt_height                        = get_dim(tensor_shape, "H");
    const int evt_width                         = get_dim(tensor_shape, "W");
    const EventPreprocessorType processing_type = std::get<EventPreprocessorType>(proc_params.at("type"));
    switch (processing_type) {
    case EventPreprocessorType::DIFF: {
        const float max_incr_per_pixel = std::get<float>(proc_params.at("max_incr_per_pixel"));
        if (max_incr_per_pixel <= 0.f)
            throw std::runtime_error("max_incr_per_pixel must be strictly greater than 0");
        const float clip_value_after_normalization = std::get<float>(proc_params.at("clip_value_after_normalization"));
        if (clip_value_after_normalization < 0.f)
            throw std::runtime_error("clip_value_after_normalization must be equal or greater than zero");
        const float scale_width =
            proc_params.count("scale_width") ? std::get<float>(proc_params.at("scale_width")) : 1.f;
        const float scale_height =
            proc_params.count("scale_height") ? std::get<float>(proc_params.at("scale_height")) : 1.f;
        return std::make_unique<DiffProcessor<InputIt>>(evt_width, evt_height, max_incr_per_pixel,
                                                        clip_value_after_normalization, scale_width, scale_height);
    }
    case EventPreprocessorType::HISTO: {
        const float max_incr_per_pixel = std::get<float>(proc_params.at("max_incr_per_pixel"));
        if (max_incr_per_pixel <= 0.f)
            throw std::runtime_error("max_incr_per_pixel must be strictly greater than 0");
        const float clip_value_after_normalization = std::get<float>(proc_params.at("clip_value_after_normalization"));
        if (clip_value_after_normalization < 0.f)
            throw std::runtime_error("clip_value_after_normalization must be equal or greater than zero");
        const bool use_CHW = std::get<bool>(proc_params.at("use_CHW"));
        const float scale_width =
            proc_params.count("scale_width") ? std::get<float>(proc_params.at("scale_width")) : 1.f;
        const float scale_height =
            proc_params.count("scale_height") ? std::get<float>(proc_params.at("scale_height")) : 1.f;
        return std::make_unique<HistoProcessor<InputIt>>(evt_width, evt_height, max_incr_per_pixel,
                                                         clip_value_after_normalization, use_CHW, scale_width,
                                                         scale_height);
    }
    case EventPreprocessorType::EVENT_CUBE: {
        const timestamp accumulation_time = std::get<timestamp>(proc_params.at("delta_t"));
        const float max_incr_per_pixel    = std::get<float>(proc_params.at("max_incr_per_pixel"));
        if (max_incr_per_pixel <= 0.f)
            throw std::runtime_error("max_incr_per_pixel must be strictly greater than 0");
        const float clip_value_after_normalization = std::get<float>(proc_params.at("clip_value_after_normalization"));
        if (clip_value_after_normalization < 0.f)
            throw std::runtime_error("clip_value_after_normalization must be equal or greater than zero");
        const int num_utbins      = std::get<int>(proc_params.at("num_utbins"));
        const bool split_polarity = std::get<bool>(proc_params.at("split_polarity"));
        const float scale_width =
            proc_params.count("scale_width") ? std::get<float>(proc_params.at("scale_width")) : 1.f;
        const float scale_height =
            proc_params.count("scale_height") ? std::get<float>(proc_params.at("scale_height")) : 1.f;
        return std::make_unique<EventCubeProcessor<InputIt>>(accumulation_time, evt_width, evt_height, num_utbins,
                                                             split_polarity, max_incr_per_pixel,
                                                             clip_value_after_normalization, scale_width, scale_height);
    }
    case EventPreprocessorType::HARDWARE_DIFF: {
        const int8_t min_val      = std::get<int8_t>(proc_params.at("min_val"));
        const int8_t max_val      = std::get<int8_t>(proc_params.at("max_val"));
        const bool allow_rollover = std::get<bool>(proc_params.at("allow_rollover"));
        return std::make_unique<HardwareDiffProcessor<InputIt>>(evt_width, evt_height, min_val, max_val,
                                                                allow_rollover);
    }
    case EventPreprocessorType::HARDWARE_HISTO: {
        const uint8_t neg_saturation = std::get<uint8_t>(proc_params.at("neg_saturation"));
        const uint8_t pos_saturation = std::get<uint8_t>(proc_params.at("pos_saturation"));
        return std::make_unique<HardwareHistoProcessor<InputIt>>(evt_width, evt_height, neg_saturation, pos_saturation);
    }
    case EventPreprocessorType::TIME_SURFACE: {
        const uint8_t nb_channels = std::get<uint8_t>(proc_params.at("nb_channels"));
        if (nb_channels == 1)
            return std::make_unique<TimeSurfaceProcessor<InputIt, 1>>(evt_width, evt_height);
        else if (nb_channels == 2)
            return std::make_unique<TimeSurfaceProcessor<InputIt, 2>>(evt_width, evt_height);
        else
            throw std::runtime_error("For TimeSurface preprocessing, number of channels should be 1 or 2.");
    }
    default:
        throw std::runtime_error("Unknown EventPreprocessor type: " +
                                 eventPreprocessorTypeToStringMap.at(processing_type));
    }
}

} // namespace EventPreprocessorFactory

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_EVENT_PREPROCESSOR_FACTORY_IMPL_H
