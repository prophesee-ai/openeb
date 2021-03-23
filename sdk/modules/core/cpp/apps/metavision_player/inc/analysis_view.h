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

#ifndef METAVISION_PLAYER_ANALYSIS_VIEW_H
#define METAVISION_PLAYER_ANALYSIS_VIEW_H

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>

#include "viewer.h"
#include "view.h"
#include "params.h"

class AnalysisView : public View {
public:
    AnalysisView(Metavision::Camera &camera, Viewer::EventBuffer &event_buffer, const Parameters &parameters,
                 const std::string &window_name = std::string());
    AnalysisView(const View &view);

    virtual void setAccumulationRatio(int accumulation_ratio) override;
    virtual int accumulationRatio() const override;

    virtual void setFps(int fps) override;
    virtual int fps() const override;

    virtual void setCurrentTimeUs(Metavision::timestamp time) override;
    virtual Metavision::timestamp currentTimeUs() const override;

    static constexpr int MinFps() {
        return 1;
    };

    static constexpr int MaxFps() {
        return 2000;
    }

    static constexpr int DefaultFps() {
        return 25;
    }

    static constexpr int MinAccumulationRatio() {
        return 25;
    }

    static constexpr int MaxAccumulationRatio() {
        return 400;
    }

    static constexpr int DefaultAccumulationRatio() {
        return 100;
    }

protected:
    virtual void setup() override;
    virtual void update(cv::Mat &frame, int key_pressed) override;
    virtual std::vector<std::string> getHelpMessages() const override;

private:
    void setMinFps(int min_fps);
    void setMaxFps(int max_fps);

    void setMinAccumulationRatio(int min_accumulation_ratio);
    void setMaxAccumulationRatio(int max_accumulation_ratio);

    int sequenceStartTimeUs() const;
    int sequenceStartRatio() const;
    void setSequenceStartRatio(int sequence_start_ratio);
    void setMinSequenceStartRatio(int min_sequence_start_time_ratio);
    void setMaxSequenceStartRatio(int max_sequence_start_time_ratio);

    int sequenceDurationUs() const;
    int sequenceDurationRatio() const;
    void setSequenceDurationRatio(int sequence_duration_ratio);
    void setMinSequenceDurationRatio(int min_sequence_duration_ratio);
    void setMaxSequenceDurationRatio(int max_sequence_duration_ratio);

    int frameId() const;
    void setFrameId(int frame_id);
    void setMinFrameId(int min_frame_id);
    void setMaxFrameId(int max_frame_id);

    void exportVideo();

    Metavision::timestamp first_time_us_, last_time_us_;
    bool setup_ = false;
    cv::Mat frame_, tmp_frame_;
};

#endif // METAVISION_PLAYER_ANALYSIS_VIEW_H
