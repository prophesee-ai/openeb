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
#include <unordered_map>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/core/algorithms/generic_producer_algorithm.h>
#include <metavision/sdk/core/utils/colors.h>
#include <metavision/sdk/base/utils/log.h>

#include "utils.h"
#include "camera_view.h"

namespace {
const std::string DiffOnLabel("diff_on");
const std::string DiffOffLabel("diff_off");
const std::string HpfLabel("hpf");
const std::string FoLabel("fo");
const std::string PrLabel("pr");
const std::string RefrLabel("refr");

std::unordered_map<std::string, std::string> biasLabelToName{
    {DiffOnLabel, "bias_diff_on"}, {DiffOffLabel, "bias_diff_off"}, {HpfLabel, "bias_hpf"}, {FoLabel, "bias_fo"},
    {PrLabel, "bias_pr"},          {RefrLabel, "bias_refr"}};

const size_t NumTrackBars = biasLabelToName.size();

std::unordered_map<std::string, int> orig_biases;
std::unordered_map<std::string, int> current_biases;

int current_roi_x, current_roi_x_end, current_roi_y, current_roi_y_end;
CameraView::RoiControl::State current_roi_state = CameraView::RoiControl::State::NONE;

void diff_off_trackbar_handler(int v, void *ptr) {
    auto *view = reinterpret_cast<CameraView *>(ptr);
    if (view->is_ready()) {
        view->camera().biases().get_facility()->set(biasLabelToName[DiffOffLabel], v);
    }
}

void diff_on_trackbar_handler(int v, void *ptr) {
    auto *view = reinterpret_cast<CameraView *>(ptr);
    if (view->is_ready()) {
        view->camera().biases().get_facility()->set(biasLabelToName[DiffOnLabel], v);
    }
}

void hpf_trackbar_handler(int v, void *ptr) {
    auto *view = reinterpret_cast<CameraView *>(ptr);
    if (view->is_ready()) {
        view->camera().biases().get_facility()->set(biasLabelToName[HpfLabel], v);
    }
}

void fo_trackbar_handler(int v, void *ptr) {
    auto *view = reinterpret_cast<CameraView *>(ptr);
    if (view->is_ready()) {
        view->camera().biases().get_facility()->set(biasLabelToName[FoLabel], v);
    }
}

void pr_trackbar_handler(int v, void *ptr) {
    auto *view = reinterpret_cast<CameraView *>(ptr);
    if (view->is_ready()) {
        view->camera().biases().get_facility()->set(biasLabelToName[PrLabel], v);
    }
}

void refr_trackbar_handler(int v, void *ptr) {
    auto *view = reinterpret_cast<CameraView *>(ptr);
    if (view->is_ready()) {
        view->camera().biases().get_facility()->set(biasLabelToName[RefrLabel], v);
    }
}

void mouse_handler(int event, int x, int y, int, void *ptr) {
    CameraView::RoiControl *roi_control = static_cast<CameraView::RoiControl *>(ptr);

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (roi_control->state != CameraView::RoiControl::INIT) {
            roi_control->state = CameraView::RoiControl::INIT;
            roi_control->x     = x;
            roi_control->y     = y;
            roi_control->x_end = x;
            roi_control->y_end = y;
        }
    } else if (event == cv::EVENT_MOUSEMOVE && roi_control->state == CameraView::RoiControl::INIT) {
        roi_control->x_end = x;
        roi_control->y_end = y;
    } else if (event == cv::EVENT_LBUTTONUP && roi_control->state == CameraView::RoiControl::INIT) {
        roi_control->state = CameraView::RoiControl::CREATED;
        roi_control->x_end = x;
        roi_control->y_end = y;
    } else if (event == cv::EVENT_RBUTTONDOWN) {
        roi_control->state = CameraView::RoiControl::REMOVE;
    }
}

} // namespace

CameraView::CameraView(Metavision::Camera &cam, Viewer::EventBuffer &event_buffer, const Parameters &parameters,
                       bool live, const std::string &window_name) :
    View(cam, event_buffer, parameters,
         cv::Size(0, live && parameters.show_biases ? NumTrackBars * TRACKBAR_HEIGHT : 0), window_name),
    live_(live),
    roi_control_(cam) {}

CameraView::CameraView(bool live, const View &view) :
    View(cv::Size(0, live && view.parameters().show_biases ? NumTrackBars * TRACKBAR_HEIGHT : 0), view),
    live_(live),
    roi_control_(camera()) {}

CameraView::~CameraView() {
    if (recording_) {
        camera().stop_recording();
        MV_LOG_INFO() << "Saved RAW file at" << raw_filename_;
    }

    if (live_) {
        // Update values of current biases
        auto &cam          = camera();
        const auto &params = parameters();
        if (params.show_biases) {
            try {
                auto *bias = cam.biases().get_facility();
                for (auto p : biasLabelToName) {
                    current_biases[p.first] = bias->get(p.second);
                }
            } catch (...) {}
        }
    }
}

void CameraView::setup() {
    if (live_) {
        const auto &window_name = windowName();
        cv::setMouseCallback(window_name, mouse_handler, &roi_control_);

        const auto &params = parameters();
        if (params.show_biases) {
            auto &cam = camera();
            try {
                const auto &gen = cam.generation();
                auto *bias      = cam.biases().get_facility();

                addTrackBar(DiffOffLabel, window_name, 0, 1, diff_off_trackbar_handler, this);
                addTrackBar(DiffOnLabel, window_name, 0, 1, diff_on_trackbar_handler, this);
                addTrackBar(HpfLabel, window_name, 0, 1, hpf_trackbar_handler, this);
                addTrackBar(FoLabel, window_name, 0, 1, fo_trackbar_handler, this);
                if (!(gen.version_major() == 4 && gen.version_minor() == 1)) {
                    addTrackBar(PrLabel, window_name, 0, 1, pr_trackbar_handler, this);
                }
                addTrackBar(RefrLabel, window_name, 0, 1, refr_trackbar_handler, this);

                if (gen.version_major() == 4) {
                    if (gen.version_minor() == 0) {
                        biasLabelToName[FoLabel] = "bias_fo_n";
                    } else {
                        biasLabelToName.erase(PrLabel);
                    }
                }

                if (current_biases.size() == 0) {
                    // In this case it's the first time this view has been build :
                    // we take the original biases
                    for (auto p : biasLabelToName) {
                        orig_biases[p.first]    = bias->get(p.second);
                        current_biases[p.first] = orig_biases[p.first];
                    }
                } else {
                    // In this case we set the biases to their last value
                    for (auto p : biasLabelToName) {
                        bias->set(p.second, current_biases[p.first]);
                    }
                }

                int bias_diff = bias->get("bias_diff");
                if (gen.version_major() == 3 && gen.version_minor() == 0) {
                    cv::setTrackbarMax(DiffOffLabel, window_name, bias_diff - 1);
                    cv::setTrackbarMin(DiffOffLabel, window_name, 0);
                    cv::setTrackbarMax(DiffOnLabel, window_name, 1800);
                    cv::setTrackbarMin(DiffOnLabel, window_name, bias_diff + 1);
                    cv::setTrackbarMax(HpfLabel, window_name, 1800);
                    cv::setTrackbarMin(HpfLabel, window_name, 0);
                    cv::setTrackbarMax(FoLabel, window_name, 1800);
                    cv::setTrackbarMin(FoLabel, window_name, 1650);
                    cv::setTrackbarMax(PrLabel, window_name, 1800);
                    cv::setTrackbarMin(PrLabel, window_name, 1200);
                    cv::setTrackbarMax(RefrLabel, window_name, 1800);
                    cv::setTrackbarMin(RefrLabel, window_name, 1300);
                } else if (gen.version_major() == 3 && gen.version_minor() == 1) {
                    cv::setTrackbarMax(DiffOffLabel, window_name, bias_diff - 1);
                    cv::setTrackbarMin(DiffOffLabel, window_name, 0);
                    cv::setTrackbarMax(DiffOnLabel, window_name, 1800);
                    cv::setTrackbarMin(DiffOnLabel, window_name, bias_diff + 1);
                    cv::setTrackbarMax(HpfLabel, window_name, 1800);
                    cv::setTrackbarMin(HpfLabel, window_name, 0);
                    cv::setTrackbarMax(FoLabel, window_name, 1800);
                    cv::setTrackbarMin(FoLabel, window_name, 0);
                    cv::setTrackbarMax(PrLabel, window_name, 1800);
                    cv::setTrackbarMin(PrLabel, window_name, 0);
                    cv::setTrackbarMax(RefrLabel, window_name, 1800);
                    cv::setTrackbarMin(RefrLabel, window_name, 0);
                } else if (gen.version_major() == 4) {
                    cv::setTrackbarMax(DiffOffLabel, window_name, bias_diff - 1);
                    cv::setTrackbarMin(DiffOffLabel, window_name, 0);
                    cv::setTrackbarMax(DiffOnLabel, window_name, 255);
                    cv::setTrackbarMin(DiffOnLabel, window_name, bias_diff + 1);
                    cv::setTrackbarMax(HpfLabel, window_name, 255);
                    cv::setTrackbarMin(HpfLabel, window_name, 0);
                    cv::setTrackbarMax(FoLabel, window_name, 255);
                    cv::setTrackbarMin(FoLabel, window_name, 0);
                    if (gen.version_minor() != 1) {
                        cv::setTrackbarMax(PrLabel, window_name, 255);
                        cv::setTrackbarMin(PrLabel, window_name, 0);
                    }
                    cv::setTrackbarMax(RefrLabel, window_name, 255);
                    cv::setTrackbarMin(RefrLabel, window_name, 0);
                } else {
                    MV_LOG_ERROR() << "Unknown camera generation";
                    return;
                }
                for (auto p : biasLabelToName) {
                    cv::setTrackbarPos(p.first, window_name, current_biases[p.first]);
                }
            } catch (...) {}
        }

        // Update Roi if necessary :
        if (current_roi_state == RoiControl::CREATED) {
            roi_control_.state = current_roi_state;
            roi_control_.x     = current_roi_x;
            roi_control_.x_end = current_roi_x_end;
            roi_control_.y     = current_roi_y;
            roi_control_.y_end = current_roi_y_end;
        }
        ready_ = true;
    }
}

std::vector<std::string> CameraView::getHelpMessages() const {
    // clang-format off
    std::vector<std::string> msgs = {
        "Keyboard/mouse actions:",
        "  \"h\"           show/hide the help menu",
        "  \"c\"           cycle color theme",
        "  \"a\"           toggle analysis mode",
    };
    // clang-format on
    const auto &params = parameters();
    if (params.show_biases) {
        msgs.push_back("  \"b\"           save current biases in a bias file");
        if (!params.in_bias_file.empty())
            msgs.push_back("  \"r\"           reset biases to values loaded from bias file");
        else
            msgs.push_back("  \"r\"           reset biases to default values");
    }
    msgs.push_back("  SPACE        start/stop recording RAW file");
    msgs.push_back("  \"q\" or ESC   exit the application");
    if (live_) {
        msgs.push_back("  Click and drag to create ROI");
        msgs.push_back("  Right-click to cancel ROI");
    }
    return msgs;
}

int CameraView::accumulationRatio() const {
    return accumulation_ratio_;
}

void CameraView::setAccumulationRatio(int accumulation_ratio) {
    accumulation_ratio_ = accumulation_ratio;
}

int CameraView::fps() const {
    return fps_;
}

void CameraView::setFps(int fps) {
    fps_ = fps;
}

void CameraView::setCurrentTimeUs(Metavision::timestamp time_us) {
    time_us_ = time_us;
}

Metavision::timestamp CameraView::currentTimeUs() const {
    return time_us_;
}

bool CameraView::is_ready() const {
    return ready_;
}

void CameraView::update(cv::Mat &frame, int key_pressed) {
    const auto &window_name = windowName();
    auto &cam               = camera();
    const auto &params      = parameters();

    switch (key_pressed) {
    case 'r': {
        if (params.show_biases) {
            try {
                for (auto p : biasLabelToName) {
                    cam.biases().get_facility()->set(p.second, orig_biases[p.first]);
                    current_biases[p.first] = orig_biases[p.first];
                    cv::setTrackbarPos(p.first, window_name, orig_biases[p.first]);
                }
                if (!params.in_bias_file.empty()) {
                    setStatusMessage("Reset biases to values loaded from bias file");
                    MV_LOG_INFO() << "Reset biases to values loaded from bias file";
                } else {
                    setStatusMessage("Reset biases to default values");
                    MV_LOG_INFO() << "Reset biases to default values";
                }
            } catch (...) {}
        }
        break;
    }
    case 'b': {
        if (params.show_biases) {
            try {
                cam.biases().save_to_file(params.out_bias_file);
                setStatusMessage("Saved bias file at " + params.out_bias_file);
                MV_LOG_INFO() << "Saved bias file at" << params.out_bias_file;
            } catch (...) {}
        }
        break;
    }
    case ' ': {
        recording_ = !recording_;
        if (recording_) {
            raw_filename_ = makeRawFilename(params.out_raw_basename);
            cam.start_recording(raw_filename_);
            setStatusMessage("Saving RAW file at " + raw_filename_);
            MV_LOG_INFO() << "Saving RAW file at" << raw_filename_;
        } else {
            cam.stop_recording();
            setStatusMessage("Saved RAW file at " + raw_filename_);
            MV_LOG_INFO() << "Saved RAW file at" << raw_filename_;
        }
        break;
    }
    }

    const auto &palette = colorPalette();
    switch (roi_control_.state) {
    case RoiControl::NONE:
        break;
    case RoiControl::REMOVE:
        roi_control_.state = RoiControl::NONE;
        roi_control_.camera.roi().unset();
        current_roi_state = RoiControl::NONE;
        break;
    case RoiControl::INIT: {
        int x     = cv::max(0, cv::min(roi_control_.x, roi_control_.x_end));
        int y     = cv::max(0, cv::min(roi_control_.y, roi_control_.y_end));
        int x_end = cv::min(roi_control_.camera.geometry().width(), cv::max(roi_control_.x, roi_control_.x_end));
        int y_end = cv::min(roi_control_.camera.geometry().height(), cv::max(roi_control_.y, roi_control_.y_end));

        cv::Rect rect(x, y, x_end - x, y_end - y);
        const cv::Scalar c = getCVColor(palette, Metavision::ColorType::Auxiliary);
        cv::rectangle(frame, rect, c);
        break;
    }
    case RoiControl::CREATED: {
        int x     = cv::max(0, cv::min(roi_control_.x, roi_control_.x_end));
        int y     = cv::max(0, cv::min(roi_control_.y, roi_control_.y_end));
        int x_end = cv::min(roi_control_.camera.geometry().width(), cv::max(roi_control_.x, roi_control_.x_end));
        int y_end = cv::min(roi_control_.camera.geometry().height(), cv::max(roi_control_.y, roi_control_.y_end));

        Metavision::Roi::Window rect_roi;
        rect_roi.x      = x;
        rect_roi.y      = y;
        rect_roi.width  = x_end - x;
        rect_roi.height = y_end - y;
        if (rect_roi.width >= RoiControl::MIN_ROI_SIZE && rect_roi.height >= RoiControl::MIN_ROI_SIZE) {
            roi_control_.camera.roi().set(rect_roi);
        }

        // Update value of current roi:
        current_roi_state = RoiControl::CREATED;
        current_roi_x     = roi_control_.x;
        current_roi_x_end = roi_control_.x_end;
        current_roi_y     = roi_control_.y;
        current_roi_y_end = roi_control_.y_end;

        roi_control_.state = RoiControl::NONE;
        break;
    }
    }
}
