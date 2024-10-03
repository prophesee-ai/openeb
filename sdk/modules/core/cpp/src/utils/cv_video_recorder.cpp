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

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>

#include "metavision/sdk/core/utils/cv_video_recorder.h"

namespace Metavision {

CvVideoRecorder::CvVideoRecorder(const std::filesystem::path &output_video_file, const int fourcc, const uint32_t fps,
                                 const cv::Size &size, bool colored) :
    writer_(output_video_file.string(), fourcc, fps, size, colored), data_to_write_pool_(DataPool::make_bounded()) {
    if (!writer_.isOpened()) {
        std::stringstream message;
        message << "'" << output_video_file << "' is not writable. ";
        if (output_video_file.has_parent_path() && !std::filesystem::exists(output_video_file)) {
            message << "The parent directory '" << output_video_file << "' does not exist.";
        } else {
            message << "Check the output directory write permission.";
        }

        throw std::runtime_error(message.str());
    }
}

CvVideoRecorder::~CvVideoRecorder() {
    try {
        // VideoWriter can throw when released during stop()
        stop();
    } catch (...) {}
}

bool CvVideoRecorder::start() {
    return recorder_thread_.start();
}

void CvVideoRecorder::stop() {
    recorder_thread_.stop();
    writer_.release();
}

void CvVideoRecorder::write(cv::Mat &data) {
    if (!is_recording()) {
        return;
    }

    auto to_write = data_to_write_pool_.acquire();
    data.copyTo(*to_write);
    recorder_thread_.add_task([this, to_write]() { writer_.write(*to_write); });
}

bool CvVideoRecorder::is_recording() {
    return recorder_thread_.is_active();
}

} // namespace Metavision
