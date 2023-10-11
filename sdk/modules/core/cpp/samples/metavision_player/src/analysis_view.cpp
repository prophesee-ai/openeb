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

#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <metavision/sdk/core/utils/colors.h>
#include <metavision/sdk/base/utils/log.h>

#include "utils.h"
#include "analysis_view.h"
#include "analysis_utils.h"

namespace {
const std::string FpsLabel("Fps");
const std::string AccumulationRatioLabel("Acc (%)");
const std::string SequenceStartRatioLabel("Start (%)");
const std::string SequenceDurationRatioLabel("Length (%)");
const std::string FrameIdLabel("Frame #");
const size_t NumTrackBars = 5;
} // namespace

AnalysisView::AnalysisView(Metavision::Camera &camera, Viewer::EventBuffer &event_buffer, const Parameters &parameters,
                           const std::string &window_name) :
    View(camera, event_buffer, parameters, cv::Size(0, NumTrackBars * TRACKBAR_HEIGHT), window_name),
    first_time_us_(event_buffer.front().t),
    last_time_us_(event_buffer.back().t) {}

AnalysisView::AnalysisView(const View &view) :
    View(cv::Size(0, NumTrackBars * TRACKBAR_HEIGHT), view),
    first_time_us_(eventBuffer().front().t),
    last_time_us_(eventBuffer().back().t) {}

void AnalysisView::setup() {
    setup_                  = true;
    const auto &window_name = windowName();
    addTrackBar(FpsLabel, window_name, MinFps(), MaxFps());
    setFps(DefaultFps());
    addTrackBar(AccumulationRatioLabel, window_name, MinAccumulationRatio(), MaxAccumulationRatio());
    setAccumulationRatio(DefaultAccumulationRatio());
    addTrackBar(SequenceStartRatioLabel, window_name, 0, 100);
    setSequenceStartRatio(10);
    addTrackBar(SequenceDurationRatioLabel, window_name, 1, 100);
    setSequenceDurationRatio(100);
    const int frame_period_us = framePeriodUs();
    int time_us               = first_time_us_ + frame_period_us;
    int frame_id              = 0;
    while (time_us + frame_period_us < last_time_us_) {
        time_us += frame_period_us;
        ++frame_id;
    }
    addTrackBar(FrameIdLabel, window_name, 0, frame_id);
    setFrameId(frame_id);
}

int AnalysisView::fps() const {
    return (setup_ ? cv::getTrackbarPos(FpsLabel, windowName()) : DefaultFps());
}

void AnalysisView::setFps(int fps) {
    cv::setTrackbarPos(FpsLabel, windowName(), fps);
}

void AnalysisView::setMinFps(int min_fps) {
    cv::setTrackbarMin(FpsLabel, windowName(), min_fps);
}

void AnalysisView::setMaxFps(int max_fps) {
    cv::setTrackbarMax(FpsLabel, windowName(), max_fps);
}

int AnalysisView::accumulationRatio() const {
    return (setup_ ? cv::getTrackbarPos(AccumulationRatioLabel, windowName()) : DefaultAccumulationRatio());
}

void AnalysisView::setAccumulationRatio(int accumulation_ratio) {
    cv::setTrackbarPos(AccumulationRatioLabel, windowName(), accumulation_ratio);
}

void AnalysisView::setMinAccumulationRatio(int min_accumulation_ratio) {
    cv::setTrackbarMin(AccumulationRatioLabel, windowName(), min_accumulation_ratio);
}

void AnalysisView::setMaxAccumulationRatio(int max_accumulation_ratio) {
    cv::setTrackbarMax(AccumulationRatioLabel, windowName(), max_accumulation_ratio);
}

int AnalysisView::sequenceStartTimeUs() const {
    return compute_sequence_start_time(first_time_us_, last_time_us_, framePeriodUs(), sequenceStartRatio());
}

int AnalysisView::sequenceStartRatio() const {
    return (setup_ ? cv::getTrackbarPos(SequenceStartRatioLabel, windowName()) : 0);
}

void AnalysisView::setSequenceStartRatio(int sequence_start_ratio) {
    cv::setTrackbarPos(SequenceStartRatioLabel, windowName(), sequence_start_ratio);
}

void AnalysisView::setMinSequenceStartRatio(int min_sequence_start_ratio) {
    cv::setTrackbarMin(SequenceStartRatioLabel, windowName(), min_sequence_start_ratio);
}

void AnalysisView::setMaxSequenceStartRatio(int max_sequence_start_ratio) {
    cv::setTrackbarMax(SequenceStartRatioLabel, windowName(), max_sequence_start_ratio);
}

int AnalysisView::sequenceDurationUs() const {
    return compute_sequence_duration(first_time_us_, last_time_us_, framePeriodUs(), sequenceStartRatio(),
                                     sequenceDurationRatio());
}

int AnalysisView::sequenceDurationRatio() const {
    return (setup_ ? cv::getTrackbarPos(SequenceDurationRatioLabel, windowName()) : 100);
}

void AnalysisView::setSequenceDurationRatio(int sequence_duration_us) {
    cv::setTrackbarPos(SequenceDurationRatioLabel, windowName(), sequence_duration_us);
}

void AnalysisView::setMinSequenceDurationRatio(int min_sequence_duration_us) {
    cv::setTrackbarMin(SequenceDurationRatioLabel, windowName(), min_sequence_duration_us);
}

void AnalysisView::setMaxSequenceDurationRatio(int max_sequence_duration_us) {
    cv::setTrackbarMax(SequenceDurationRatioLabel, windowName(), max_sequence_duration_us);
}

int AnalysisView::frameId() const {
    return (setup_ ? cv::getTrackbarPos(FrameIdLabel, windowName()) : 0);
}

void AnalysisView::setFrameId(int frame_id) {
    if (setup_) {
        cv::setTrackbarPos(FrameIdLabel, windowName(), frame_id);
    }
}

void AnalysisView::setMinFrameId(int min_frame_id) {
    cv::setTrackbarMin(FrameIdLabel, windowName(), min_frame_id);
}

void AnalysisView::setMaxFrameId(int max_frame_id) {
    cv::setTrackbarMax(FrameIdLabel, windowName(), max_frame_id);
}

Metavision::timestamp AnalysisView::currentTimeUs() const {
    return compute_current_time(sequenceStartTimeUs(), frameId(), framePeriodUs());
}

void AnalysisView::setCurrentTimeUs(Metavision::timestamp) {}

std::vector<std::string> AnalysisView::getHelpMessages() const {
    // clang-format off
    std::vector<std::string> msgs = {
        "Keyboard/mouse actions:", 
        "  \"h\"           show/hide the help menu",
        "  \"c\"           cycle color theme",
        "  \"a\"           toggle analysis mode",
        "  \"s\"           save a snapshot image of current frame",
        "  \"v\"           save a video of current buffer",
    };
    msgs.push_back("  \"q\" or ESC   exit the application");
    // clang-format on
    return msgs;
}

void AnalysisView::exportVideo() {
    const auto &event_buffer = eventBuffer();
    const auto &palette      = colorPalette();
    const auto &params       = parameters();
    if (event_buffer.empty()) {
        return;
    }

    auto log = MV_LOG_INFO() << Metavision::Log::no_endline << Metavision::Log::no_space << "Exporting to "
                             << params.out_avi_file << "@" << params.out_avi_fps << " fps.";

    auto begin = event_buffer.begin(), end = event_buffer.end();
    const int sequence_start_time_us = sequenceStartTimeUs();
    const int sequence_end_time_us   = sequence_start_time_us + sequenceDurationUs();
    const int frame_period_us        = framePeriodUs();
    const int accumulation_time_us   = accumulationTimeUs();
    const auto &sensor_size          = getCameraSize(camera());
    cv::Mat frame(sensor_size, CV_8UC3);
    cv::VideoWriter video_writer(params.out_avi_file, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), params.out_avi_fps,
                                 sensor_size);
    int n_frames = 0;
    for (Metavision::timestamp ts_us = sequence_start_time_us; ts_us <= sequence_end_time_us;
         ts_us += frame_period_us, ++n_frames) {}

    auto last    = std::chrono::time_point<std::chrono::high_resolution_clock>::max();
    int frame_id = 0;
    for (Metavision::timestamp ts_us = sequence_start_time_us; ts_us <= sequence_end_time_us;
         ts_us += frame_period_us, ++frame_id) {
        auto now   = std::chrono::high_resolution_clock::now();
        long delay = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        if (frame_id == 0 || delay > 500) {
            log << ".";
            tmp_frame_ = frame_.clone();

            std::ostringstream oss;
            const auto &color = getCVColor(palette, Metavision::ColorType::Auxiliary);
            cv::Rect rect(0, frame_.rows - 20, frame_.cols, 20);
            cv::Mat box = tmp_frame_(rect);
            box -= cv::Scalar::all(255 * 0.8);
            oss << "Exporting frame " << frame_id << " of " << n_frames << ", please wait.";
            addText(box, oss.str(), cv::Point(5, 14), color);
            imshow(windowName(), tmp_frame_);
            cv::waitKey(1);
            last = now;
        }
        makeSliceImage(frame, begin, end, ts_us, accumulation_time_us, frame_period_us, params.out_avi_fps, palette);
        video_writer.write(frame);
    }
    log << std::endl;
    MV_LOG_INFO() << "Done writing video, wrote" << n_frames << "frames";
}

void AnalysisView::update(cv::Mat &frame, int key_pressed) {
    const auto &event_buffer = eventBuffer();
    const auto &params       = parameters();
    const auto &palette      = colorPalette();

    switch (key_pressed) {
    case 's': {
        cv::imwrite(params.out_png_file, frame);
        setStatusMessage("Saved frame at " + params.out_png_file);
        MV_LOG_INFO() << "Saved frame at" << params.out_png_file;
        break;
    }
    case 'v':
        frame_ = frame;
        exportVideo();
        setStatusMessage("Saved video at " + params.out_avi_file);
        MV_LOG_INFO() << "Saved video at" << params.out_avi_file;
        break;
    }

    const auto &window_name = windowName();
    AnalysisData data       = compute_analysis_data(first_time_us_, last_time_us_, fps(), accumulationRatio(),
                                              sequenceStartRatio(), sequenceDurationRatio(), currentTimeUs());

    // Update fps
    setMaxFps(data.max_fps);
    setMinFps(data.min_fps);
    setFps(data.fps);

    // Update start ratio
    setMaxSequenceStartRatio(data.max_sequence_start_ratio);
    setMinSequenceStartRatio(data.min_sequence_start_ratio);

    // Update duration ratio
    setMaxSequenceDurationRatio(data.max_sequence_duration_ratio);
    setMinSequenceDurationRatio(data.min_sequence_duration_ratio);

    // Update accumulation ratio
    setMaxAccumulationRatio(data.max_accumulation_ratio);
    setMinAccumulationRatio(data.min_accumulation_ratio);
    setAccumulationRatio(data.accumulation_ratio);

    // Update frame id
    setMaxFrameId(data.max_frame_id);
    setMinFrameId(data.min_frame_id);
    setFrameId(data.frame_id);
}
