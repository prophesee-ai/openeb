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

#include <metavision/sdk/base/utils/log.h>

#include "viewer.h"
#include "view.h"
#include "camera_view.h"
#include "analysis_view.h"

Viewer::Viewer(const Parameters &parameters) :
    parameters_(parameters), event_buffer_(parameters.buffer_size_mev * 1e6) {}

Viewer::~Viewer() {}

void Viewer::setup_camera() {
    // Release camera.
    camera_ = Metavision::Camera();

    // Live camera.
    if (parameters_.in_raw_file.empty()) {
        camera_ = Metavision::Camera::from_first_available();

        // Set biases if file specified.
        if (!parameters_.in_bias_file.empty()) {
            camera_.biases().set_from_file(parameters_.in_bias_file);
        }
    }
    // RAW file.
    else {
        camera_ = Metavision::Camera::from_file(parameters_.in_raw_file, true);
    }

    // Get sensor size
    auto &geometry      = camera_.geometry();
    sensor_size_.width  = geometry.width();
    sensor_size_.height = geometry.height();

    // Add runtime error callback.
    camera_.add_runtime_error_callback([](const Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        throw;
    });

    // Add CD callback
    prod_.reset(new Metavision::GenericProducerAlgorithm<Metavision::Event2d>(500'000));
    camera_.cd().add_callback([this](const Metavision::Event2d *ev_begin, const Metavision::Event2d *ev_end) {
        prod_->register_new_event_buffer(ev_begin, ev_end);
    });
}

void Viewer::run() {
    setup_camera();
    camera_.start();

    bool live = parameters_.in_raw_file.empty();
    view_.reset(new CameraView(camera_, event_buffer_, parameters_, live));

    Metavision::timestamp ts = 0;
    std::vector<Metavision::Event2d> input_evt_buffer;
    bool paused = false;
    while (paused || camera_.is_running()) {
        if (!paused) {
            input_evt_buffer.clear();
            ts += view_->framePeriodUs();
            view_->setCurrentTimeUs(ts);
            prod_->process_events(ts, std::back_inserter(input_evt_buffer));

            // Insert data into the buffer.
            event_buffer_.insert(event_buffer_.end(), input_evt_buffer.cbegin(), input_evt_buffer.cend());
        }

        int key_pressed = view_->update();
        if (key_pressed == 27 || key_pressed == 'q') {
            // Quit
            break;
        } else if (key_pressed == 'a') {
            // Enter/exit pause mode.
            paused = !paused;
            if (paused) {
                if (event_buffer_.back().t - event_buffer_.front().t < 5'000) {
                    // Ignore the key until we have enough events in the buffer ...
                    paused = false;
                } else {
                    // When pausing, stop the camera
                    camera_.stop();
                    view_.reset(new AnalysisView(*view_));
                }
            } else {
                if (live) {
                    // When using a live camera, recreate it from scratch at start
                    // and reset the timestamp
                    setup_camera();
                    ts = 0;
                }

                // When restarting after pause, start the camera and clear the buffer
                camera_.start();
                event_buffer_.clear();
                view_.reset(new CameraView(live, *view_));
            }
        }
    }

    prod_->set_source_as_done();
}
