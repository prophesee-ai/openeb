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

#ifndef METAVISION_SDK_CORE_DETAIL_MOSTRECENT_TIMESTAMP_BUFFER_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_MOSTRECENT_TIMESTAMP_BUFFER_IMPL_H

#include <boost/assert.hpp>

namespace Metavision {

/// @brief Default constructor
template<typename timestamp_type>
MostRecentTimestampBufferT<timestamp_type>::MostRecentTimestampBufferT() :
    rows_(0), cols_(0), channels_(0), cols_channels_(0) {}

/// @brief Initialization constructor
template<typename timestamp_type>
inline MostRecentTimestampBufferT<timestamp_type>::MostRecentTimestampBufferT(int rows, int cols, int channels) :
    rows_(0), cols_(0), channels_(0), cols_channels_(cols_ * channels_) {
    create(rows, cols, channels);
}

/// @brief Copy constructor
template<typename timestamp_type>
inline MostRecentTimestampBufferT<timestamp_type>::MostRecentTimestampBufferT(const MostRecentTimestampBufferT &other) :
    rows_(other.rows_),
    cols_(other.cols_),
    channels_(other.channels_),
    cols_channels_(cols_ * channels_),
    tsbuffer_(other.tsbuffer_) {}

/// @brief Destructor
template<typename timestamp_type>
inline MostRecentTimestampBufferT<timestamp_type>::~MostRecentTimestampBufferT() {}

/// @brief Allocates the buffer
template<typename timestamp_type>
inline void MostRecentTimestampBufferT<timestamp_type>::create(int rows, int cols, int channels) {
    tsbuffer_.clear();
    tsbuffer_.resize(rows * cols * channels, 0);
    rows_          = rows;
    cols_          = cols;
    channels_      = channels;
    cols_channels_ = cols * channels;
}

template<typename timestamp_type>
inline void MostRecentTimestampBufferT<timestamp_type>::release() {
    std::vector<timestamp_type>().swap(tsbuffer_);
    rows_          = 0;
    cols_          = 0;
    channels_      = 0;
    cols_channels_ = 0;
}

/// @brief accesses the number of rows of the buffer
template<typename timestamp_type>
inline int MostRecentTimestampBufferT<timestamp_type>::rows() const {
    return rows_;
}

/// @brief Accesses the number of columns of the buffer
template<typename timestamp_type>
inline int MostRecentTimestampBufferT<timestamp_type>::cols() const {
    return cols_;
}

/// @brief Accesses the size of the buffer
template<typename timestamp_type>
inline cv::Size MostRecentTimestampBufferT<timestamp_type>::size() const {
    return cv::Size(cols_, rows_);
}

/// @brief Accesses the number of channels of the buffer
template<typename timestamp_type>
inline int MostRecentTimestampBufferT<timestamp_type>::channels() const {
    return channels_;
}

/// @brief Checks whether the buffer is empty
template<typename timestamp_type>
inline bool MostRecentTimestampBufferT<timestamp_type>::empty() const {
    return (rows_ * cols_ == 0);
}

/// @brief Sets all elements of the timestamp buffer to a constant
template<typename timestamp_type>
inline void MostRecentTimestampBufferT<timestamp_type>::set_to(timestamp_type ts) {
    std::fill(tsbuffer_.begin(), tsbuffer_.end(), ts);
}

template<typename timestamp_type>
inline void
    MostRecentTimestampBufferT<timestamp_type>::copy_to(MostRecentTimestampBufferT<timestamp_type> &other) const {
    other.rows_          = rows_;
    other.cols_          = cols_;
    other.channels_      = channels_;
    other.tsbuffer_      = tsbuffer_;
    other.cols_channels_ = cols_channels_;
}

/// @brief Swaps the timestamp buffer with the specified one
template<typename timestamp_type>
inline void MostRecentTimestampBufferT<timestamp_type>::swap(MostRecentTimestampBufferT &other) {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(channels_, other.channels_);
    std::swap(tsbuffer_, other.tsbuffer_);
    std::swap(cols_channels_, other.cols_channels_);
}

/// @brief Retrieves the reference of the timestamp at the specified pixel
template<typename timestamp_type>
inline const timestamp_type &MostRecentTimestampBufferT<timestamp_type>::at(int y, int x, int c) const {
    BOOST_ASSERT_MSG(x >= 0 && x < cols_ && y >= 0 && y < rows_ && c >= 0 && c < channels_,
                     "Input coordinates are outside the bounds of the buffer!");
    return tsbuffer_[y * cols_channels_ + x * channels_ + c];
}

/// @brief Retrieves the reference of the timestamp at the specified pixel
template<typename timestamp_type>
inline timestamp_type &MostRecentTimestampBufferT<timestamp_type>::at(int y, int x, int c) {
    BOOST_ASSERT_MSG(x >= 0 && x < cols_ && y >= 0 && y < rows_ && c >= 0 && c < channels_,
                     "Input coordinates are outside the bounds of the buffer!");
    return tsbuffer_[y * cols_channels_ + x * channels_ + c];
}

/// @brief Retrieves the pointer to the timestamp at the specified pixel
template<typename timestamp_type>
inline const timestamp_type *MostRecentTimestampBufferT<timestamp_type>::ptr(int y, int x, int c) const {
    BOOST_ASSERT_MSG(x >= 0 && x < cols_ && y >= 0 && y < rows_ && c >= 0 && c < channels_,
                     "Input coordinates are outside the bounds of the buffer!");
    return &tsbuffer_[y * cols_channels_ + x * channels_ + c];
}

/// @brief Retrieves the pointer to the timestamp at the specified pixel
template<typename timestamp_type>
inline timestamp_type *MostRecentTimestampBufferT<timestamp_type>::ptr(int y, int x, int c) {
    BOOST_ASSERT_MSG(x >= 0 && x < cols_ && y >= 0 && y < rows_ && c >= 0 && c < channels_,
                     "Input coordinates are outside the bounds of the buffer!");
    return &tsbuffer_[y * cols_channels_ + x * channels_ + c];
}

/// @brief Retrieves the maximum timestamp across channels at the specified pixel
template<typename timestamp_type>
inline timestamp_type MostRecentTimestampBufferT<timestamp_type>::max_across_channels_at(int y, int x) const {
    BOOST_ASSERT_MSG(x >= 0 && x < cols_ && y >= 0 && y < rows_,
                     "Input coordinates are outside the bounds of the buffer!");
    const timestamp_type *pbuff_ts = &tsbuffer_[(y * cols_ + x) * channels_];
    return *std::max_element(pbuff_ts, pbuff_ts + channels_);
}

// @brief Generates a CV_8UC1 image of the time surface for the 2 channels
// Side-by-side: negative polarity time surface, positive polarity time surface
// The time surface is normalized between last_ts (255) and last_ts - delta_t (0)
template<typename timestamp_type>
inline void MostRecentTimestampBufferT<timestamp_type>::generate_img_time_surface(timestamp_type last_ts,
                                                                                  timestamp_type delta_t,
                                                                                  cv::Mat &out) const {
    out.create(this->rows(), this->channels() * this->cols(), CV_8UC1);
    out.setTo(cv::Scalar::all(0));

    const double ratio    = 255. / delta_t;
    const int nb_channels = this->channels();

    for (int row = 0; row < this->rows(); ++row) {
        for (int p = 0; p < nb_channels; ++p) {
            auto img_ptr  = out.ptr<uint8_t>(row, this->cols() * p);
            auto last_ptr = img_ptr + this->cols();
            auto ts_ptr   = this->ptr(row, 0, p);
            for (; img_ptr != last_ptr; ++img_ptr) {
                auto diff = last_ts - *ts_ptr;
                if (diff <= delta_t) {
                    *img_ptr = static_cast<uint8_t>((delta_t - diff) * ratio);
                }
                ts_ptr += nb_channels /* channels are interleaved */;
            }
        }
    }
}

// @brief Generates a CV_8UC1 image of the time surface, merging the 2 channels
// The time surface is normalized between last_ts (255) and last_ts - delta_t (0)
template<typename timestamp_type>
inline void MostRecentTimestampBufferT<timestamp_type>::generate_img_time_surface_collapsing_channels(
    timestamp_type last_ts, timestamp_type delta_t, cv::Mat &out) const {
    out.create(this->rows(), this->cols(), CV_8UC1);
    out.setTo(cv::Scalar::all(0));
    const double ratio = 255. / delta_t;

    for (int row = 0; row < this->rows(); ++row) {
        auto img_ptr  = out.ptr<uint8_t>(row, 0);
        auto ts_ptr   = this->ptr(row, 0, 0);
        auto last_ptr = img_ptr + this->cols();
        while (img_ptr != last_ptr) {
            auto delta = last_ts - *std::max_element(ts_ptr, ts_ptr + this->channels());

            if (delta <= delta_t) {
                *img_ptr = static_cast<uint8_t>((delta_t - delta) * ratio);
            }
            ++img_ptr;
            ts_ptr += this->channels() /* channels are interleaved */;
        }
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_MOSTRECENT_TIMESTAMP_BUFFER_IMPL_H
