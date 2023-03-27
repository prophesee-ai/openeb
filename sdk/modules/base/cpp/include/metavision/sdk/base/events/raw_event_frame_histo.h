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

#ifndef METAVISION_SDK_RAW_EVENT_FRAME_HISTO_H
#define METAVISION_SDK_RAW_EVENT_FRAME_HISTO_H

#include <memory>
#include <vector>
#include <cassert>

#include "metavision/sdk/base/events/detail/event_traits.h"

namespace Metavision {

enum HistogramChannel { NEGATIVE = 0, POSITIVE = 1 };

struct RawEventFrameHistoConfig {
    unsigned width;
    unsigned height;
    std::vector<unsigned> channel_bit_size;
    bool packed;
};

/// @brief Class representing a histogram of CD events:
/// Event data is presented as a "frame" of accumulated events, with two channels per pixel. The first channel being the
/// number of negative events and the second channel being the number of positive events.
class RawEventFrameHisto {
public:
    RawEventFrameHisto(const unsigned height, const unsigned width, const unsigned channel_bit_neg = 4,
                       const unsigned channel_bit_pos = 4, bool packed = false) {
        cfg_.height           = height;
        cfg_.width            = width;
        cfg_.channel_bit_size = {channel_bit_neg, channel_bit_pos};
        cfg_.packed           = packed;
        histogram_.reset(new std::vector<uint8_t>(2 * height * width));
        assert(histogram_);
    }

    /// @brief Default constructor
    RawEventFrameHisto(const RawEventFrameHistoConfig &cfg, std::unique_ptr<const std::vector<uint8_t>> data) :
        cfg_(cfg), histogram_(std::move(data)) {
        assert(histogram_);
    }

    /// @brief Copy constructor
    RawEventFrameHisto(const RawEventFrameHisto &h) :
        cfg_(h.cfg_), histogram_(std::make_unique<const std::vector<uint8_t>>(h.get_data())) {}

    const RawEventFrameHistoConfig &get_config() const {
        return cfg_;
    }

    const std::vector<uint8_t> &get_data() const {
        return *histogram_;
    }

    std::size_t buffer_size() const {
        return histogram_->size();
    }

private:
    std::unique_ptr<const std::vector<uint8_t>> histogram_;
    RawEventFrameHistoConfig cfg_;
};

} // namespace Metavision

#endif // METAVISION_SDK_RAW_EVENT_FRAME_HISTO_H
