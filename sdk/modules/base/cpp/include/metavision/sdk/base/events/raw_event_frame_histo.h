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
#include <stdexcept>
#include <vector>

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
    /// @brief Default constructor
    RawEventFrameHisto() = default;

    /// @brief Constructor
    /// @throws invalid_argument if either bit size is zero or if their sum is more than 8
    RawEventFrameHisto(unsigned height, unsigned width, unsigned channel_bit_neg = 4, unsigned channel_bit_pos = 4,
                       bool packed = false) :
        cfg_{width, height, {channel_bit_neg, channel_bit_pos}, packed}, histo_((packed ? 1 : 2) * height * width, 0) {
        if (channel_bit_neg == 0 || channel_bit_pos == 0 || channel_bit_neg + channel_bit_pos > 8) {
            throw std::invalid_argument("Invalid channel bit sizes!");
        }
    }

    /// @brief Constructor
    /// @throws invalid_argument if either bit size is zero or if their sum is more than 8
    RawEventFrameHisto(const RawEventFrameHistoConfig &cfg, const std::vector<uint8_t> &data) :
        cfg_(cfg), histo_(data) {
        if (cfg.channel_bit_size[0] == 0 || cfg.channel_bit_size[1] == 0 ||
            cfg.channel_bit_size[0] + cfg.channel_bit_size[1] > 8) {
            throw std::invalid_argument("Invalid channel bit sizes!");
        }
    }

    /// @brief Copy constructor
    RawEventFrameHisto(const RawEventFrameHisto &h) : cfg_(h.cfg_), histo_(h.histo_) {}

    /// @brief Resets the event frame configuration and sets all values to 0.
    /// @throws invalid_argument if either bit size is zero or if their sum is more than 8
    void reset(unsigned height, unsigned width, unsigned channel_bit_neg = 4, unsigned channel_bit_pos = 4,
               bool packed = false) {
        if (channel_bit_neg == 0 || channel_bit_pos == 0 || channel_bit_neg + channel_bit_pos > 8) {
            throw std::invalid_argument("Invalid channel bit sizes!");
        }
        cfg_.width            = width;
        cfg_.height           = height;
        cfg_.channel_bit_size = {channel_bit_neg, channel_bit_pos};
        cfg_.packed           = packed;
        reset();
    }

    /// @brief Reset all values in the event frame to 0.
    void reset() {
        histo_.clear();
        histo_.resize((cfg_.packed ? 1 : 2) * cfg_.height * cfg_.width, 0);
    }

    const RawEventFrameHistoConfig &get_config() const {
        return cfg_;
    }

    const std::vector<uint8_t> &get_data() const {
        return histo_;
    }

    std::vector<uint8_t> &get_data() {
        return histo_;
    }

    std::size_t buffer_size() const {
        return histo_.size();
    }

    void swap(RawEventFrameHisto &h) {
        histo_.swap(h.histo_);
        std::swap(cfg_, h.cfg_);
    }

private:
    RawEventFrameHistoConfig cfg_;
    std::vector<uint8_t> histo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_RAW_EVENT_FRAME_HISTO_H
