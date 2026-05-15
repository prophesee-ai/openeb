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

#include <stdexcept>
#include <string>

#include "metavision/sdk/core/utils/raw_event_frame_converter.h"

namespace Metavision {

RawEventFrameConverter::RawEventFrameConverter(unsigned height, unsigned width, unsigned num_channels,
                                               HistogramFormat output_format) :
    height_(height), width_(width), num_channels_(num_channels) {
    if (num_channels_ != 1 && num_channels_ != 2) {
        throw std::invalid_argument(
            "Unsupported number of channels for event histogram: " + std::to_string(num_channels_) + " channels");
    }

    set_format(output_format);
}

void RawEventFrameConverter::set_format(HistogramFormat output_format) {
    format_ = output_format;

    channel_stride_ = format_ == HistogramFormat::HWC ? 1 : height_ * width_;
    column_stride_  = format_ == HistogramFormat::HWC ? num_channels_ : 1;
}

}; // namespace Metavision
