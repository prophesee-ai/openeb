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
#include <metavision/hal/facilities/i_hw_identification.h>

#include "utils.h"
#include "viewer.h"
#include "view.h"
#include "camera_view.h"
#include "analysis_view.h"

Viewer::Viewer(const Parameters &parameters) :
    parameters_(parameters),
    event_buffer_(parameters.buffer_size_mev * 1e6),
    paused_(false),
    ts_(0),
    cd_events_cb_id_(-1) {}

Viewer::~Viewer() {
    stop();
}

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
        camera_ = Metavision::Camera::from_file(parameters_.in_raw_file);
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
    prod_.reset(new Metavision::GenericProducerAlgorithm<Metavision::Event2d>(40'000));
    cd_events_cb_id_ =
        camera_.cd().add_callback([this](const Metavision::Event2d *ev_begin, const Metavision::Event2d *ev_end) {
            prod_->register_new_event_buffer(ev_begin, ev_end);
        });
}

bool Viewer::is_running() {
    return paused_ || camera_.is_running();
}

void Viewer::start() {
    setup_camera();
    camera_.start();

    bool live = parameters_.in_raw_file.empty();
    view_.reset(new CameraView(camera_, event_buffer_, parameters_, live));
}

bool Viewer::update() {
    if (parameters_.in_raw_file.empty() && parameters_.show_biases) {
        // Handle specific case of trying to set biases when using an IMX636 or a GenX320 camera
        if ((camera_.generation().version_major() == 4 && camera_.generation().version_minor() == 2) ||
            (camera_.generation().version_major() == 320 && camera_.generation().version_minor() == 0)) {
            MV_LOG_ERROR() << "Metavision Player can not be used to set biases for this camera. Please use "
                              "Metavision Studio instead.";
            return false;
        }
    }

    if (!paused_) {
        input_evt_buffer_.clear();
        ts_ += view_->framePeriodUs();
        view_->setCurrentTimeUs(ts_);
        prod_->process_events(ts_, std::back_inserter(input_evt_buffer_));

        // Insert data into the buffer.
        event_buffer_.insert(event_buffer_.end(), input_evt_buffer_.cbegin(), input_evt_buffer_.cend());
    }

    int key_pressed = view_->update();
    if (key_pressed == 27 || key_pressed == 'q') {
        // Quit
        return false;
    } else if (key_pressed == 'a') {
        // Enter/exit pause mode.
        paused_ = !paused_;
        if (paused_) {
            if (event_buffer_.back().t - event_buffer_.front().t < 5'000) {
                // Ignore the key until we have enough events in the buffer ...
                paused_ = false;
            } else {
                // When pausing, stop the camera
                camera_.stop();
                view_.reset(new AnalysisView(*view_));
            }
        } else {
            bool live = parameters_.in_raw_file.empty();
            if (live) {
                // When using a live camera, recreate it from scratch at start
                // and reset the timestamp
                setup_camera();
                ts_ = 0;
            }

            // When restarting after pause, start the camera and clear the buffer
            camera_.start();
            event_buffer_.clear();
            view_.reset(new CameraView(live, *view_));
        }
    }
    return true;
}

void Viewer::stop() {
    // unregister callbacks to make sure they are not called anymore
    if (cd_events_cb_id_ >= 0) {
        camera_.cd().remove_callback(cd_events_cb_id_);
        cd_events_cb_id_ = -1;
    }
    if (camera_.is_running()) {
        MV_LOG_TRACE() << "Closing camera." << std::endl;
        camera_.stop();
    }
    prod_->set_source_as_done();
}
