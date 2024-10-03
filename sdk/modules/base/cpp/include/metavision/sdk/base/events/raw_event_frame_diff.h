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
#include <stdexcept>
#include <vector>

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
    /// @brief Default constructor
    RawEventFrameDiff() = default;

    /// @brief Constructor
    /// @throws invalid_argument if bit_size is outside the supported range of [2;8].
    RawEventFrameDiff(unsigned height, unsigned width, unsigned bit_size = 8) :
        cfg_{width, height, bit_size}, diff_(height * width, 0) {
        if (bit_size < 2 || bit_size > 8) {
            throw std::invalid_argument("bit_size is outside the supported range of [2;8]!");
        }
    }

    /// @brief Constructor
    /// @throws invalid_argument if cfg.bit_size is outside the supported range of [2;8].
    RawEventFrameDiff(const RawEventFrameDiffConfig &cfg, const std::vector<int8_t> &data) : cfg_(cfg), diff_(data) {
        if (cfg.bit_size < 2 || cfg.bit_size > 8) {
            throw std::invalid_argument("bit_size is outside the supported range of [2;8]!");
        }
    }

    /// @brief Copy constructor
    RawEventFrameDiff(const RawEventFrameDiff &d) : cfg_(d.cfg_), diff_(d.diff_) {}

    /// @brief Resets the event frame configuration and sets all values to 0.
    /// @throws invalid_argument if bit_size is outside the supported range of [2;8].
    void reset(unsigned height, unsigned width, unsigned bit_size = 8) {
        if (bit_size < 2 || bit_size > 8) {
            throw std::invalid_argument("bit_size is outside the supported range of [2;8]!");
        }
        cfg_.width    = width;
        cfg_.height   = height;
        cfg_.bit_size = bit_size;
        reset();
    }

    /// @brief Reset all values in the event frame to 0.
    void reset() {
        diff_.clear();
        diff_.resize(cfg_.height * cfg_.width, 0);
    }

    const RawEventFrameDiffConfig &get_config() const {
        return cfg_;
    }

    const std::vector<int8_t> &get_data() const {
        return diff_;
    }

    std::vector<int8_t> &get_data() {
        return diff_;
    }

    std::size_t buffer_size() const {
        return diff_.size();
    }

    void swap(RawEventFrameDiff &d) {
        diff_.swap(d.diff_);
        std::swap(cfg_, d.cfg_);
    }

private:
    RawEventFrameDiffConfig cfg_;
    std::vector<int8_t> diff_;
};

} // namespace Metavision

#endif // METAVISION_SDK_RAW_EVENT_FRAME_DIFF_H
