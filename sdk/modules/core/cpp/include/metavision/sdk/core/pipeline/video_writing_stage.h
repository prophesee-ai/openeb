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

#ifndef METAVISION_SDK_CORE_VIDEO_WRITING_STAGE_H
#define METAVISION_SDK_CORE_VIDEO_WRITING_STAGE_H

#include <boost/any.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <thread>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/pipeline/base_stage.h"
#include "metavision/sdk/core/algorithms/stream_logger_algorithm.h"

namespace Metavision {

/// @brief Stage that writes the input frames to a video file.
class VideoWritingStage : public BaseStage {
public:
    using FramePool = SharedObjectPool<cv::Mat>;
    using FramePtr  = FramePool::ptr_type;
    using FrameData = std::pair<timestamp, FramePtr>;

    /// @brief Constructor
    /// @param filename Name of the output file.
    /// @param width Width of the frame.
    /// @param height Height of the frame.
    /// @param fps Frames per second of the output video.
    /// @param codec Codec used by OpenCV to encode the video.
    /// @param colored If true the incoming frames are expected to be color frames, otherwise grayscale.
    VideoWritingStage(const std::string &filename, int width, int height, int fps, const std::string &codec = "MJPG",
                      bool colored = true) {
        if (codec.size() != 4) {
            throw std::runtime_error("VideoWritingStage : codec must be a 4 letter word.");
        }

#if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION >= 3) || CV_MAJOR_VERSION >= 4
        video_writer_.open(filename, cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]), fps,
                           cv::Size(width, height), colored);
#else
        video_writer_.open(filename, CV_FOURCC(codec[0], codec[1], codec[2], codec[3]), fps, cv::Size(width, height),
                           colored);
#endif

        set_consuming_callback([this](const boost::any &data) {
            try {
                auto res = boost::any_cast<FrameData>(data);
                video_writer_ << *(res.second);
            } catch (boost::bad_any_cast &c) { MV_SDK_LOG_ERROR() << c.what(); }
        });
    }

    /// @brief Constructor
    /// @param prev_stage Previous stage.
    /// @param filename Name of the output file.
    /// @param width Width of the frame.
    /// @param height Height of the frame.
    /// @param fps Frames per second of the output video.
    /// @param codec Codec used by OpenCV to encode the video.
    /// @param colored If true the incoming frames are expected to be color frames, otherwise grayscale.
    VideoWritingStage(BaseStage &prev_stage, const std::string &filename, int width, int height, int fps,
                      const std::string &codec = "MJPG", bool colored = true) :
        VideoWritingStage(filename, width, height, fps, codec, colored) {
        set_previous_stage(prev_stage);
    }

    /// @brief Destructor
    ~VideoWritingStage() {
        video_writer_.release();
    }

private:
    cv::VideoWriter video_writer_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_VIDEO_WRITING_STAGE_H
