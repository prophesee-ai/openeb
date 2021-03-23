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
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#ifndef METAVISION_SDK_CORE_VIDEO_WRITER_H
#define METAVISION_SDK_CORE_VIDEO_WRITER_H

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace cv45 {
class IVideoWriter;
}

namespace Metavision {

/** @brief Video writer class.

The class provides C++ API for writing video files or image sequences.
@note This class fixes a bug in the OpenCV MJPEG encoder and it is recommended to use it over the original
<a href="https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html">cv::VideoWriter</a>.
*/
class VideoWriter : public cv::VideoWriter {
public:
    /** @brief Default constructors
     */
    VideoWriter();

    /** @overload
    @param filename Name of the output video file.
    @param fourcc 4-character code of codec used to compress the frames. For example,
    VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec, VideoWriter::fourcc('M','J','P','G') is a
    motion-jpeg codec etc. List of codes can be obtained at [Video Codecs by
    FOURCC](http://www.fourcc.org/codecs.php) page. FFMPEG backend with MP4 container natively uses
    other values as fourcc code: see [ObjectType](http://mp4ra.org/#/codecs),
    so you may receive a warning message from OpenCV about fourcc code conversion.
    @param fps Framerate of the created video stream.
    @param frameSize Size of the video frames.
    @param isColor If it is not zero, the encoder will expect and encode color frames, otherwise it
    will work with grayscale frames.

    @b Tips:
    - With some backends `fourcc=-1` pops up the codec selection dialog from the system.
    - To save image sequence use a proper filename (eg. `img_%02d.jpg`) and `fourcc=0`
      OR `fps=0`. Use uncompressed image format (eg. `img_%02d.BMP`) to save raw frames.
    - Most codecs are lossy. If you want lossless video file you need to use a lossless codecs
      (eg. FFMPEG FFV1, Huffman HFYU, Lagarith LAGS, etc...)
    - If FFMPEG is enabled, using `codec=0; fps=0;` you can create an uncompressed (raw) video file.
    */
    VideoWriter(const cv::String &filename, int fourcc, double fps, cv::Size frameSize, bool isColor = true);

    /** @overload
    The `apiPreference` parameter allows to specify API backends to use. Can be used to enforce a specific reader
    implementation if multiple are available: e.g. cv::CAP_FFMPEG or cv::CAP_GSTREAMER.
     */
    VideoWriter(const cv::String &filename, int apiPreference, int fourcc, double fps, cv::Size frameSize,
                bool isColor = true);

    /** @overload
     * The `params` parameter allows to specify extra encoder parameters encoded as pairs (paramId_1, paramValue_1,
     * paramId_2, paramValue_2, ... .) see cv::VideoWriterProperties
     */
    VideoWriter(const cv::String &filename, int fourcc, double fps, const cv::Size &frameSize,
                const std::vector<int> &params);

    /** @overload
     */
    VideoWriter(const cv::String &filename, int apiPreference, int fourcc, double fps, const cv::Size &frameSize,
                const std::vector<int> &params);

    /** @brief Default destructor

    The method first calls VideoWriter::release to close the already opened file.
    */
    virtual ~VideoWriter();

    /** @brief Initializes or reinitializes video writer.

    The method opens video writer. Parameters are the same as in the constructor
    VideoWriter::VideoWriter.
    @return `true` if video writer has been successfully initialized

    The method first calls VideoWriter::release to close the already opened file.
     */
    virtual bool open(const cv::String &filename, int fourcc, double fps, cv::Size frameSize, bool isColor = true);

    /** @overload
     */
    bool open(const cv::String &filename, int apiPreference, int fourcc, double fps, cv::Size frameSize,
              bool isColor = true);

    /** @overload
     */
    bool open(const cv::String &filename, int fourcc, double fps, const cv::Size &frameSize,
              const std::vector<int> &params);

    /** @overload
     */
    bool open(const cv::String &filename, int apiPreference, int fourcc, double fps, const cv::Size &frameSize,
              const std::vector<int> &params);

    /** @brief Returns true if video writer has been successfully initialized.
     */
    virtual bool isOpened() const;

    /** @brief Closes the video writer.

    The method is automatically called by subsequent VideoWriter::open and by the VideoWriter
    destructor.
     */
    virtual void release();

    /** @brief Stream operator to write the next video frame.
    @sa write
    */
    virtual VideoWriter &operator<<(const cv::Mat &image);

    /** @overload
    @sa write
    */
    virtual VideoWriter &operator<<(const cv::UMat &image);

    /** @brief Writes the next video frame

    @param image The written frame. In general, color images are expected in BGR format.

    The function/method writes the specified image to video file. It must have the same size as has
    been specified when opening the video writer.
     */
    virtual void write(cv::InputArray image);

    /** @brief Sets a property in the VideoWriter.

     @param propId Property identifier from cv::VideoWriterProperties (eg. cv::VIDEOWRITER_PROP_QUALITY)
     or one of videoio_flags_others

     @param value Value of the property.
     @return  `true` if the property is supported by the backend used by the VideoWriter instance.
     */
    virtual bool set(int propId, double value);

    /** @brief Returns the specified VideoWriter property

     @param propId Property identifier from cv::VideoWriterProperties (eg. cv::VIDEOWRITER_PROP_QUALITY)
     or one of videoio_flags_others

     @return Value for the specified property. Value 0 is returned when querying a property that is
     not supported by the backend used by the VideoWriter instance.
     */
    virtual double get(int propId) const;

    /** @brief Returns used backend API name

     @note Stream should be opened.
     */
    cv::String getBackendName() const;

private:
    cv::Ptr<cv45::IVideoWriter> writer_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_VIDEO_WRITER_H