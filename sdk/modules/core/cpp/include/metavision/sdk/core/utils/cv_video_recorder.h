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

#ifndef METAVISION_SDK_CORE_CV_VIDEO_RECORDER_H
#define METAVISION_SDK_CORE_CV_VIDEO_RECORDER_H

#include <opencv2/videoio.hpp>

#include "metavision/sdk/core/utils/threaded_process.h"
#include "metavision/sdk/core/utils/video_writer.h"
#include "metavision/sdk/base/utils/object_pool.h"

namespace Metavision {

/// @brief A simple threaded video recorder using OpenCV routines
class CvVideoRecorder {
public:
    CvVideoRecorder(const std::string &output_video_file, const int fourcc, const uint32_t fps, const cv::Size &size,
                    bool colored);

    /// @brief Records all remaining frames then destroys the object
    ~CvVideoRecorder();

    /// @brief Starts the recording thread
    bool start();

    /// @brief Stops adding data to the recording queue through the @ref write methods.
    ///
    /// The recorder thread remains active until all data added in the queue have been dumped.
    ///
    /// @throw cv::Exception when an error occurs (e.g output file too big)
    void stop();

    /// @brief Pushes the input frame for writing.
    ///
    /// This method does nothing if the recorder thread is not active
    void write(cv::Mat &data);

    /// @brief Returns if the recording thread is ongoing.
    bool is_recording();

private:
    VideoWriter writer_;

    using DataPool = SharedObjectPool<cv::Mat>;
    DataPool data_to_write_pool_;
    ThreadedProcess recorder_thread_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_CV_VIDEO_RECORDER_H
