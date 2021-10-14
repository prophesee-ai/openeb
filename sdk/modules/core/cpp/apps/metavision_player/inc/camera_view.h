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

#ifndef METAVISION_PLAYER_CAMERA_VIEW_H
#define METAVISION_PLAYER_CAMERA_VIEW_H

#include <vector>
#include <string>
#include <chrono>
#include <opencv2/core.hpp>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>

#include "view.h"
#include "params.h"

class CameraView : public View {
public:
    struct RoiControl {
        static constexpr int MIN_ROI_SIZE = 5;

        RoiControl(Metavision::Camera &c) : camera(c) {}

        enum State {
            NONE,
            INIT,
            CREATED,
            REMOVE,
        };

        Metavision::Camera &camera;
        int x, x_end, y, y_end;
        State state{NONE};
    };

    CameraView(Metavision::Camera &camera, Viewer::EventBuffer &event_buffer, const Parameters &parameters, bool live,
               const std::string &window_name = std::string());
    CameraView(bool live, const View &view);
    ~CameraView();

    virtual void setAccumulationRatio(int accumulation_ratio) override;
    virtual int accumulationRatio() const override;

    virtual void setFps(int fps) override;
    virtual int fps() const override;

    virtual void setCurrentTimeUs(Metavision::timestamp time) override;
    virtual Metavision::timestamp currentTimeUs() const override;

    bool is_ready() const;

protected:
    virtual void setup() override;
    virtual void update(cv::Mat &frame, int key_pressed) override;
    virtual std::vector<std::string> getHelpMessages() const override;

private:
    bool live_      = false;
    bool recording_ = false;
    bool ready_     = false;
    std::string raw_filename_;
    RoiControl roi_control_;
    Metavision::timestamp time_us_ = 0;
    int accumulation_ratio_        = 100;
    int fps_                       = 25;
};

#endif // METAVISION_PLAYER_CAMERA_VIEW_H
