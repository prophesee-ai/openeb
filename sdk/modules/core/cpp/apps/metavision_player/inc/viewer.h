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

#ifndef METAVISION_PLAYER_VIEWER_H
#define METAVISION_PLAYER_VIEWER_H

#include <memory>
#include <boost/circular_buffer.hpp>
#include <opencv2/core.hpp>
#include <metavision/sdk/base/events/event2d.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/core/algorithms/generic_producer_algorithm.h>

#include "params.h"

class View;

class Viewer {
public:
    using EventBuffer               = boost::circular_buffer<Metavision::Event2d>;
    static constexpr int FRAME_RATE = 25;

    Viewer(const Parameters &params);
    ~Viewer();
    void run();

private:
    void setup_camera();

    std::unique_ptr<Metavision::GenericProducerAlgorithm<Metavision::Event2d>> prod_;
    Parameters parameters_;
    Metavision::Camera camera_;
    EventBuffer event_buffer_;
    cv::Size sensor_size_;
    cv::Mat frame_;
    std::unique_ptr<View> view_;
};

#endif // METAVISION_PLAYER_VIEWER_H
