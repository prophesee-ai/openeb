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

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <opencv2/videoio.hpp>
#include <opencv2/core/types_c.h>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/utils/video_writer.h"

// defining the stuff required for OpenCV's implementation from v4.5.0
#ifndef CV_NOEXCEPT
#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900 /*MSVS 2015*/)
#define CV_NOEXCEPT noexcept
#endif
#endif
#ifndef CV_NOEXCEPT
#define CV_NOEXCEPT
#endif
#ifndef CV_OVERRIDE
#define CV_OVERRIDE override
#endif
#ifndef CV_LOG_ONCE_WARNING
#define CV_LOG_ONCE_WARNING(...)
#endif

namespace {
template<class T>
inline T castParameterTo(int paramValue) {
    return static_cast<T>(paramValue);
}
} // namespace

#define CV_FOURCC(c1, c2, c3, c4) (((c1)&255) + (((c2)&255) << 8) + (((c3)&255) << 16) + (((c4)&255) << 24))

namespace cv45 {

using String    = ::cv::String;
using Exception = ::cv::Exception;
template<typename T>
using Ptr              = ::cv::Ptr<T>;
using Size             = ::cv::Size;
using InputArray       = ::cv::InputArray;
using Mat              = ::cv::Mat;
using ParallelLoopBody = ::cv::ParallelLoopBody;
using Range            = ::cv::Range;

namespace Error {
enum Code {
    StsAssert       = ::cv::Error::StsAssert,
    StsOutOfRange   = ::cv::Error::StsOutOfRange,
    StsVecLengthErr = ::cv::Error::StsVecLengthErr
};
}

using ::cv::error;
using ::cv::format;
using ::cv::makePtr;
using ::cv::parallel_for_;

class IVideoWriter {
public:
    virtual ~IVideoWriter() {}
    virtual double getProperty(int) const {
        return 0;
    }
    virtual bool setProperty(int, double) {
        return false;
    }
    virtual bool isOpened() const  = 0;
    virtual void close()           = 0;
    virtual void write(InputArray) = 0;
    virtual int getCaptureDomain() const {
        return cv::CAP_ANY;
    } // Return the type of the capture object: CAP_FFMPEG, etc...
};

class VideoWriterParameters {
public:
    struct VideoWriterParameter {
        VideoWriterParameter() = default;

        VideoWriterParameter(int key_, int value_) : key(key_), value(value_) {}

        int key{-1};
        int value{-1};
        mutable bool isConsumed{false};
    };

    VideoWriterParameters() = default;

    explicit VideoWriterParameters(const std::vector<int> &params) {
        const auto count = params.size();
        if (count % 2 != 0) {
            CV_Error_(Error::StsVecLengthErr, ("Vector of VideoWriter parameters should have even length"));
        }
        params_.reserve(count / 2);
        for (std::size_t i = 0; i < count; i += 2) {
            add(params[i], params[i + 1]);
        }
    }

    void add(int key, int value) {
        params_.emplace_back(key, value);
    }

    template<class ValueType>
    ValueType get(int key, ValueType defaultValue) const CV_NOEXCEPT {
        auto it = std::find_if(params_.begin(), params_.end(),
                               [key](const VideoWriterParameter &param) { return param.key == key; });
        if (it != params_.end()) {
            it->isConsumed = true;
            return castParameterTo<ValueType>(it->value);
        } else {
            return defaultValue;
        }
    }

    std::vector<int> getUnused() const CV_NOEXCEPT {
        std::vector<int> unusedParams;
        for (const auto &param : params_) {
            if (!param.isConsumed) {
                unusedParams.push_back(param.key);
            }
        }
        return unusedParams;
    }

private:
    std::vector<VideoWriterParameter> params_;
};

#if CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2
enum VideoCaptureAPIs {
    CAP_OPENCV_MJPEG = 2200,
};
enum VideoCaptureProperties {
    CAP_PROP_BACKEND = 42,
};
#else
enum VideoCaptureAPIs {
    CAP_OPENCV_MJPEG = ::cv::CAP_OPENCV_MJPEG,
};
enum VideoCaptureProperties {
    CAP_PROP_BACKEND = ::cv::CAP_PROP_BACKEND,
};
#endif

enum VideoWriterProperties {
    VIDEOWRITER_PROP_QUALITY    = ::cv::VIDEOWRITER_PROP_QUALITY,
    VIDEOWRITER_PROP_FRAMEBYTES = ::cv::VIDEOWRITER_PROP_FRAMEBYTES,
    VIDEOWRITER_PROP_NSTRIPES   = ::cv::VIDEOWRITER_PROP_NSTRIPES,
#if CV_MAJOR_VERSION >= 4 && CV_MINOR_VERSION >= 4
    VIDEOWRITER_PROP_IS_COLOR = ::cv::VIDEOWRITER_PROP_IS_COLOR,
#else
    VIDEOWRITER_PROP_IS_COLOR = 4,
#endif
};

} // namespace cv45

// including OpenCV's implementation from v4.5.0
#define cv cv45 // not super clean, but it's the easiest way to get the job done
#include "3rdparty/container_avi.cpp"
#include "3rdparty/cap_mjpeg_encoder.cpp"
#undef cv

namespace Metavision {

VideoWriter::VideoWriter() {}

VideoWriter::VideoWriter(const cv::String &filename, int _fourcc, double fps, cv::Size frameSize, bool isColor) {
    open(filename, _fourcc, fps, frameSize, isColor);
}

VideoWriter::VideoWriter(const cv::String &filename, int apiPreference, int _fourcc, double fps, cv::Size frameSize,
                         bool isColor) {
    open(filename, apiPreference, _fourcc, fps, frameSize, isColor);
}

VideoWriter::VideoWriter(const cv::String &filename, int fourcc, double fps, const cv::Size &frameSize,
                         const std::vector<int> &params) {
    open(filename, fourcc, fps, frameSize, params);
}

VideoWriter::VideoWriter(const cv::String &filename, int apiPreference, int fourcc, double fps,
                         const cv::Size &frameSize, const std::vector<int> &params) {
    open(filename, apiPreference, fourcc, fps, frameSize, params);
}

void VideoWriter::release() {
    if (writer_) {
        writer_->close();
    }
    cv::VideoWriter::release();
}

VideoWriter::~VideoWriter() {
    try {
        release();
    } catch (...) {}
}

bool VideoWriter::open(const cv::String &filename, int _fourcc, double fps, cv::Size frameSize, bool isColor) {
    return open(filename, cv::CAP_ANY, _fourcc, fps, frameSize,
                std::vector<int>{cv45::VIDEOWRITER_PROP_IS_COLOR, static_cast<int>(isColor)});
}

bool VideoWriter::open(const cv::String &filename, int apiPreference, int _fourcc, double fps, cv::Size frameSize,
                       bool isColor) {
    return open(filename, apiPreference, _fourcc, fps, frameSize,
                std::vector<int>{cv45::VIDEOWRITER_PROP_IS_COLOR, static_cast<int>(isColor)});
}

bool VideoWriter::open(const cv::String &filename, int fourcc, double fps, const cv::Size &frameSize,
                       const std::vector<int> &params) {
    return open(filename, cv::CAP_ANY, fourcc, fps, frameSize, params);
}

bool VideoWriter::open(const cv::String &filename, int apiPreference, int fourcc, double fps, const cv::Size &frameSize,
                       const std::vector<int> &params) {
    if (isOpened()) {
        release();
    }

    if (fourcc == CV_FOURCC('M', 'J', 'P', 'G')) {
        const cv45::VideoWriterParameters parameters(params);
        writer_ = cv45::createMotionJpegWriter(filename, CV_FOURCC('M', 'J', 'P', 'G'), fps, frameSize, parameters);
        try {
            if (!writer_.empty()) {
                if (writer_->isOpened()) {
                    return true;
                }
                writer_->close();
            }
        } catch (const cv::Exception &e) {
            MV_SDK_LOG_ERROR() << cv::format("VideoWriter raised OpenCV exception:\n\n%s\n", e.what());
        } catch (const std::exception &e) {
            MV_SDK_LOG_ERROR() << cv::format("VideoWriter raised C++ exception:\n\n%s\n", e.what());
        } catch (...) { MV_SDK_LOG_ERROR() << cv::format("VideoWriter raised unknown C++ exception!\n\n"); }
        return false;
    }

#if CV_MAJOR_VERSION == 4 && CV_MINOR_VERSION >= 4
    return cv::VideoWriter::open(filename, apiPreference, fourcc, fps, frameSize, params);
#else
    bool isColor            = false;
    const size_t num_params = params.size();
    for (size_t i = 0; i < num_params; i += 2) {
        if (params[i] == cv45::VIDEOWRITER_PROP_IS_COLOR)
            isColor = static_cast<bool>(params[i + 1]);
    }
#if CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2
    return cv::VideoWriter::open(filename, fourcc, fps, frameSize, isColor);
#else
    return cv::VideoWriter::open(filename, apiPreference, fourcc, fps, frameSize, isColor);
#endif
#endif
}

bool VideoWriter::isOpened() const {
    if (writer_) {
        return !writer_.empty();
    }
    return cv::VideoWriter::isOpened();
}

bool VideoWriter::set(int propId, double value) {
    if (propId != static_cast<int>(cv45::CAP_PROP_BACKEND)) {
        if (writer_) {
            return writer_->setProperty(propId, value);
        }
    }
    return cv::VideoWriter::set(propId, value);
}

double VideoWriter::get(int propId) const {
    if (propId != static_cast<int>(cv45::CAP_PROP_BACKEND)) {
        if (writer_) {
            return writer_->getProperty(propId);
        }
    }
    return cv::VideoWriter::get(propId);
}

cv::String VideoWriter::getBackendName() const {
    if (writer_) {
        return "VideoWriter";
    }
#if CV_MAJOR_VERSION >= 3 && CV_MINOR_VERSION >= 2
    MV_SDK_LOG_ERROR() << "VIDEOIO: VideoWriter does not implement :" << Metavision::Log::function;
    return "";
#else
    return cv::VideoWriter::getBackendName();
#endif
}

void VideoWriter::write(cv::InputArray image) {
    if (writer_) {
        writer_->write(image);
        return;
    }
#if CV_MAJOR_VERSION >= 3 && CV_MINOR_VERSION >= 2
    cv::VideoWriter::write(image.getMat(-1));
#else
    cv::VideoWriter::write(image);
#endif
}

VideoWriter &VideoWriter::operator<<(const cv::Mat &image) {
    if (writer_) {
        write(image);
        return *this;
    }
    return static_cast<VideoWriter &>(cv::VideoWriter::operator<<(image));
}

VideoWriter &VideoWriter::operator<<(const cv::UMat &image) {
    if (writer_) {
        write(image);
        return *this;
    }

#if CV_MAJOR_VERSION >= 3 && CV_MINOR_VERSION >= 2
    MV_SDK_LOG_ERROR() << "VideoWriter :" << Metavision::Log::function << "not implemented";
    return *this;
#else
    return static_cast<VideoWriter &>(cv::VideoWriter::operator<<(image));
#endif
}

} // namespace Metavision