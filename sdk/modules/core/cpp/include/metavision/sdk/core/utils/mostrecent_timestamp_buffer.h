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

#ifndef METAVISION_SDK_CORE_MOSTRECENT_TIMESTAMP_BUFFER_H
#define METAVISION_SDK_CORE_MOSTRECENT_TIMESTAMP_BUFFER_H

#include <opencv2/opencv.hpp>
#if CV_MAJOR_VERSION >= 4
#include <opencv2/imgproc/types_c.h>
#endif

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Class representing a buffer of the most recent timestamps observed at each pixel of the camera
///
/// A most recent timestamp buffer is also called time surface.
///
/// @note The interface follows the one of cv::Mat
template<typename timestamp_type>
class MostRecentTimestampBufferT {
public:
    /// @brief Default constructor
    MostRecentTimestampBufferT();

    /// @brief Initialization constructor
    /// @param rows Sensor's height
    /// @param cols Sensor's width
    /// @param channels Number of channels
    inline MostRecentTimestampBufferT(int rows, int cols, int channels = 1);

    /// @brief Copy constructor
    /// @param other The timestamp buffer to construct from
    inline MostRecentTimestampBufferT(const MostRecentTimestampBufferT<timestamp_type> &other);

    /// @brief Destructor
    virtual inline ~MostRecentTimestampBufferT();

    /// @brief Allocates the buffer
    /// @param rows Sensor's height
    /// @param cols Sensor's width
    /// @param channels Number of channels
    inline void create(int rows, int cols, int channels = 1);

    /// @brief Deallocates the buffer
    inline void release();

    /// @brief Gets the number of rows of the buffer
    inline int rows() const;

    /// @brief Gets the number of columns of the buffer
    inline int cols() const;

    /// @brief Gets the size of the buffer (i.e. Sensor's size as well)
    inline cv::Size size() const;

    /// @brief Gets the number of channels of the buffer
    inline int channels() const;

    /// @brief Checks whether the buffer is empty
    inline bool empty() const;

    /// @brief Sets all elements of the timestamp buffer to a constant
    /// @param ts The constant timestamp value
    inline void set_to(timestamp_type ts);

    /// @brief Copies this timestamp buffer into another timestamp buffer
    /// @param other The timestamp buffer to copy to
    inline void copy_to(MostRecentTimestampBufferT<timestamp_type> &other) const;

    /// @brief Swaps the timestamp buffer with another one
    /// @param other The timestamp buffer to swap with
    inline void swap(MostRecentTimestampBufferT<timestamp_type> &other);

    /// @brief Retrieves a const reference of the timestamp at the specified pixel
    /// @param y The pixel's ordinate
    /// @param x The pixel's abscissa
    /// @param c The channel to retrieve the timestamp from
    /// @return The timestamp at the given pixel
    inline const timestamp_type &at(int y, int x, int c = 0) const;

    /// @brief Retrieves a reference of the timestamp at the specified pixel
    /// @param y The pixel's ordinate
    /// @param x The pixel's abscissa
    /// @param c The channel to retrieve the timestamp from
    /// @return The timestamp at the given pixel
    inline timestamp_type &at(int y, int x, int c = 0);

    /// @brief Retrieves a const pointer to the timestamp at the specified pixel
    /// @param y The pixel's ordinate
    /// @param x The pixel's abscissa
    /// @param c The channel to retrieve the timestamp from
    /// @return The timestamp at the given pixel
    inline const timestamp_type *ptr(int y = 0, int x = 0, int c = 0) const;

    /// @brief Retrieves a pointer to the timestamp at the specified pixel
    /// @param y The pixel's ordinate
    /// @param x The pixel's abscissa
    /// @param c The channel to retrieve the timestamp from
    /// @return The timestamp at the given pixel
    inline timestamp_type *ptr(int y = 0, int x = 0, int c = 0);

    /// @brief Retrieves the maximum timestamp across channels at the specified pixel
    /// @param y The pixel's ordinate
    /// @param x The pixel's abscissa
    /// @return The maximum timestamp at that pixel across all the channels in the buffer
    inline timestamp_type max_across_channels_at(int y, int x) const;

    /// @brief Generates a CV_8UC1 image of the time surface for the 2 channels
    ///
    /// Side-by-side: negative polarity time surface, positive polarity time surface
    /// The time surface is normalized between last_ts (255) and last_ts - delta_t (0)
    ///
    /// @param last_ts Last timestamp value stored in the buffer
    /// @param delta_t Delta time, with respect to @p last_t, above which timestamps are not considered for the image
    /// generation
    /// @param out The produced image
    inline void generate_img_time_surface(timestamp_type last_ts, timestamp_type delta_t, cv::Mat &out) const;

    /// @brief Generates a CV_8UC1 image of the time surface, merging the 2 channels
    ///
    /// The time surface is normalized between last_ts (255) and last_ts - delta_t (0)
    ///
    /// @param last_ts Last timestamp value stored in the buffer
    /// @param delta_t Delta time, with respect to @p last_t, above which timestamps are not considered for the image
    /// generation
    /// @param out The produced image
    inline void generate_img_time_surface_collapsing_channels(timestamp_type last_ts, timestamp_type delta_t,
                                                              cv::Mat &out) const;

private:
    int rows_, cols_, channels_;           ///< Dimensions of the buffer
    int cols_channels_;                    ///< Total number of cells per row (columns x channels)
    std::vector<timestamp_type> tsbuffer_; ///< Buffer of the most recent timestamps
};

/// @brief Class representing a buffer of the most recent timestamps observed at each pixel of the camera
/// @note The interface follows the one of cv::Mat
/// @note This class is a template specialization of @ref MostRecentTimestampBufferT for @ref timestamp
using MostRecentTimestampBuffer = MostRecentTimestampBufferT<Metavision::timestamp>;

} // namespace Metavision

// Function definitions
#include "detail/mostrecent_timestamp_buffer_impl.h"

#endif // METAVISION_SDK_CORE_MOSTRECENT_TIMESTAMP_BUFFER_H
