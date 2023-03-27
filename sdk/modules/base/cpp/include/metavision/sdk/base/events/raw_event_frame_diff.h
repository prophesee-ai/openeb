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

#ifndef METAVISION_SDK_RAW_EVENT_FRAME_DIFF_H
#define METAVISION_SDK_RAW_EVENT_FRAME_DIFF_H

#include <memory>
#include <vector>
#include <cassert>

#include "metavision/sdk/base/events/detail/event_traits.h"

namespace Metavision {

struct RawEventFrameDiffConfig {
    unsigned width;
    unsigned height;
    unsigned bit_size;
};

/// @brief Class representing a cumulative difference histogram of CD events:
/// Event data is presented as a "frame" of accumulated events, with two channels per pixel. The first channel being the
/// number of negative events and the second channel being the number of positive events.
class RawEventFrameDiff {
public:
    /// @brief Constructor
    RawEventFrameDiff(const unsigned height, const unsigned width) : cfg_{width, height} {
        diff_.reset(new std::vector<int8_t>(height * width));
        assert(diff_);
    }

    /// @brief Default constructor
    RawEventFrameDiff(const RawEventFrameDiffConfig &cfg, std::unique_ptr<const std::vector<int8_t>> data) :
        cfg_(cfg), diff_(std::move(data)) {
        assert(diff_);
    }

    /// @brief Copy constructor
    RawEventFrameDiff(const RawEventFrameDiff &d) :
        cfg_(d.cfg_), diff_(std::make_unique<const std::vector<int8_t>>(d.get_data())) {}

    const RawEventFrameDiffConfig &get_config() const {
        return cfg_;
    }

    const std::vector<int8_t> &get_data() const {
        return *diff_;
    }

    std::size_t buffer_size() const {
        return diff_->size();
    }

private:
    std::unique_ptr<const std::vector<int8_t>> diff_;
    const RawEventFrameDiffConfig cfg_;
};

} // namespace Metavision

#endif // METAVISION_SDK_RAW_EVENT_FRAME_DIFF_H
