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

#ifndef METAVISION_SDK_CORE_RAW_EVENT_FRAME_CONVERTER_H
#define METAVISION_SDK_CORE_RAW_EVENT_FRAME_CONVERTER_H

#include <memory>
#include <vector>
#include <cassert>

#include <metavision/sdk/base/events/raw_event_frame_diff.h>
#include <metavision/sdk/base/events/raw_event_frame_histo.h>

namespace Metavision {

/// @brief Describes the layout of dimensions in a histogram
/// CHW : Channel, Height, Width
/// HWC : Height, Width, Channel
enum HistogramFormat { CHW, HWC };

template<typename T>
class EventFrameHisto {
public:
    EventFrameHisto(int height, int width, unsigned num_channels, HistogramFormat format, std::vector<T> &&data) :
        height_(height), width_(width), num_channels_(num_channels), format_(format), histogram_data_(std::move(data)) {
        channel_stride_ = format_ == HistogramFormat::HWC ? 1 : height_ * width_;
        col_stride_     = format_ == HistogramFormat::HWC ? num_channels_ : 1;
    }

    HistogramFormat get_format() const {
        return format_;
    }

    const std::vector<T> &get_data() const {
        return histogram_data_;
    }

    std::size_t get_size() const {
        return histogram_data_.size();
    }

    T operator()(int x, int y, HistogramChannel chan) const {
        return histogram_data_[x * col_stride_ + y * width_ * col_stride_ + chan * channel_stride_];
    }

    /// @brief Gets the channel values for given pixel
    /// @returns a pair (negative, positive) representing channel values for given pixel
    std::pair<T, T> operator()(int x, int y) const {
        return std::make_pair(*this(x, y, HistogramChannel::NEGATIVE), this(x, y, HistogramChannel::POSITIVE));
    }

private:
    unsigned height_;
    unsigned width_;
    unsigned num_channels_;
    HistogramFormat format_;
    std::vector<T> histogram_data_;

    unsigned col_stride_;
    unsigned channel_stride_;
};

template<typename T>
class EventFrameDiff {
public:
    EventFrameDiff(int height, int width, std::vector<T> &&data) :
        height_(height), width_(width), diff_data_(std::move(data)) {}

    const std::vector<T> &get_data() const {
        return diff_data_;
    }

    T operator()(int x, int y) const {
        return diff_data_[x + y * width_];
    }

    std::size_t get_size() const {
        return diff_data_.size();
    }

private:
    unsigned height_;
    unsigned width_;
    std::vector<T> diff_data_;
};

class RawEventFrameConverter {
public:
    RawEventFrameConverter(unsigned height, unsigned width, unsigned num_channels,
                           HistogramFormat output_format = HistogramFormat::HWC);

    void set_format(HistogramFormat output_format);

    template<typename T>
    std::unique_ptr<EventFrameHisto<T>> convert(const RawEventFrameHisto &h) const;

    template<typename T>
    std::unique_ptr<EventFrameDiff<T>> convert(const RawEventFrameDiff &d) const;

    unsigned get_height() const {
        return height_;
    }
    unsigned get_width() const {
        return width_;
    }
    HistogramFormat get_format() const {
        return format_;
    }

private:
    HistogramFormat format_;

    unsigned height_;
    unsigned width_;
    unsigned num_channels_;
    unsigned channel_stride_;
    unsigned column_stride_;
};

template<typename T>
std::unique_ptr<EventFrameHisto<T>> RawEventFrameConverter::convert(const RawEventFrameHisto &h) const {
    assert(num_channels_ == 2); /// histo has 2 channels (by definition)
    auto histo_cfg = h.get_config();
    assert(height_ == histo_cfg.height);
    assert(width_ == histo_cfg.width);

    if (histo_cfg.channel_bit_size.size() != num_channels_) {
        throw std::invalid_argument("Invalid number of channels in histogram. Expected " +
                                    std::to_string(num_channels_) + " channels");
    }

    std::vector<T> output(histo_cfg.height * histo_cfg.width * histo_cfg.channel_bit_size.size());

    unsigned idx = 0;
    if (num_channels_ == 2) {
        const unsigned neg_bits = histo_cfg.channel_bit_size[HistogramChannel::NEGATIVE];
        const unsigned pos_bits = histo_cfg.channel_bit_size[HistogramChannel::POSITIVE];

        for (auto data : h.get_data()) {
            if (histo_cfg.packed) {
                const T neg_val = data & ((1 << neg_bits) - 1);
                const T pos_val = (data >> neg_bits) & ((1 << pos_bits) - 1);

                output[idx * column_stride_]                   = neg_val;
                output[idx * column_stride_ + channel_stride_] = pos_val;
            } else {
                if (idx % 2 == 0) {
                    const T val                        = data & ((1 << neg_bits) - 1);
                    output[(idx / 2) * column_stride_] = val;
                } else {
                    const T val                                          = data & ((1 << pos_bits) - 1);
                    output[(idx / 2) * column_stride_ + channel_stride_] = val;
                }
            }
            ++idx;
        }
    } else {
        for (auto data : h.get_data()) {
            output[idx] = data & ((1 << histo_cfg.channel_bit_size[0]) - 1);
            ++idx;
        }
    }
    return std::make_unique<EventFrameHisto<T>>(height_, width_, num_channels_, format_, std::move(output));
}

template<typename T>
std::unique_ptr<EventFrameDiff<T>> RawEventFrameConverter::convert(const RawEventFrameDiff &d) const {
    assert(height_ == d.get_config().height);
    assert(width_ == d.get_config().width);
    std::vector<T> output(height_ * width_);

    unsigned idx = 0;
    for (auto data : d.get_data()) {
        output[idx] = data;
        ++idx;
    }
    return std::make_unique<EventFrameDiff<T>>(height_, width_, std::move(output));
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_RAW_EVENT_FRAME_CONVERTER_H
