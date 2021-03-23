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

#ifndef METAVISION_PLAYER_VIEW_H
#define METAVISION_PLAYER_VIEW_H

#include <vector>
#include <string>
#include <chrono>
#include <opencv2/core.hpp>
#include <metavision/sdk/core/utils/colors.h>
#include <metavision/sdk/driver/camera.h>

#include "viewer.h"
#include "params.h"

class View {
public:
    static constexpr int TRACKBAR_HEIGHT = 60;

    View(Metavision::Camera &camera, Viewer::EventBuffer &event_buffer, const Parameters &parameters,
         const cv::Size &ui_size, const std::string &window_name);
    View(const cv::Size &ui_size, const View &view);

    virtual ~View();

    const std::string &windowName() const;

    void toggleHelpVisibility();
    bool helpVisible() const;

    void cycleColorPalette();
    Metavision::ColorPalette colorPalette() const;

    void setStatusMessage(const std::string &msg, int delay_msecs = 5'000);

    const Metavision::Camera &camera() const;
    Metavision::Camera &camera();

    const Viewer::EventBuffer &eventBuffer() const;

    const Parameters &parameters() const;

    virtual void setAccumulationRatio(int accumulation_ratio) = 0;
    virtual int accumulationRatio() const                     = 0;
    int accumulationTimeUs() const;

    virtual void setFps(int fps) = 0;
    virtual int fps() const      = 0;
    int framePeriodUs() const;

    virtual void setCurrentTimeUs(Metavision::timestamp time) = 0;
    virtual Metavision::timestamp currentTimeUs() const       = 0;

    int update();

protected:
    virtual void setup()                                     = 0;
    virtual void update(cv::Mat &frame, int key_pressed)     = 0;
    virtual std::vector<std::string> getHelpMessages() const = 0;

private:
    void showHelp(cv::Mat &frame);
    void addTextBox(const std::string &text, const cv::Scalar &color, const cv::Rect &rect, const cv::Point &pos);

    Metavision::Camera &camera_;
    Viewer::EventBuffer &event_buffer_;
    Parameters parameters_;
    std::string window_name_;
    cv::Size window_size_;
    cv::Mat frame_;
    bool setup_;
    bool show_help_;
    Metavision::ColorPalette palette_;
    std::string status_msg_;
    std::chrono::high_resolution_clock::time_point status_msg_time_;
    int status_msg_delay_ms_;
};

#endif // METAVISION_PLAYER_VIEW_H
