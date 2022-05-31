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

// Example of using Metavision SDK Driver API for visualizing events stream.

#include <boost/program_options.hpp>
#include <metavision/sdk/driver/camera.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <opencv2/highgui/highgui.hpp>
#if CV_MAJOR_VERSION >= 4
#include <opencv2/highgui/highgui_c.h>
#endif
#include <opencv2/imgproc.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>

static const int ESCAPE = 27;
static const int SPACE  = 32;

namespace po = boost::program_options;

std::string human_readable_time(Metavision::timestamp t) {
    std::ostringstream oss;
    std::array<std::string, 3> ls{":", ".", ""};
    std::array<std::string, 3> vs;
    vs[2] = cv::format("%03d", int(t % 1000));
    t /= 1000; // ms
    vs[1] = cv::format("%03d", int(t % 1000));
    t /= 1000; // s
    vs[0] = cv::format("%02d", int(t % 60));

    size_t i = 0;
    for (; i < 3; ++i) {
        oss << vs[i] << ls[i];
    }
    return oss.str();
}

int process_ui_for(int delay_ms) {
    auto then = std::chrono::high_resolution_clock::now();
    int key   = cv::waitKey(delay_ms);
    auto now  = std::chrono::high_resolution_clock::now();
    // cv::waitKey will not wait if no window is opened, so we wait for it, if needed
    std::this_thread::sleep_for(std::chrono::milliseconds(
        delay_ms - std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count()));

    return key;
}

bool window_was_closed(const std::string &window_name) {
// Small hack: if the window has been closed, it is not visible anymore or property changes from the one we set
#if CV_MAJOR_VERSION >= 3 && CV_MINOR_VERSION >= 2
    if (cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE) == 0) {
#else
    if (cv::getWindowProperty(window_name, cv::WND_PROP_AUTOSIZE) != 0) {
#endif
        return true;
    }
    return false;
}

int setup_cd_callback_and_window(Metavision::Camera &camera, cv::Mat &cd_frame, Metavision::timestamp &cd_frame_ts,
                                 Metavision::CDFrameGenerator &cd_frame_generator, const std::string &window_name) {
    auto &geometry = camera.geometry();
    auto id        = camera.cd().add_callback(
        [&cd_frame_generator](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
            cd_frame_generator.add_events(ev_begin, ev_end);
        });
    cd_frame_generator.start(30, [&cd_frame, &cd_frame_ts](const Metavision::timestamp &ts, const cv::Mat &frame) {
        cd_frame_ts = ts;
        frame.copyTo(cd_frame);
    });
    cv::namedWindow(window_name, CV_GUI_EXPANDED);
    cv::resizeWindow(window_name, geometry.width(), geometry.height());
    cv::moveWindow(window_name, 0, 0);
#if (CV_MAJOR_VERSION == 3 && (CV_MINOR_VERSION * 100 + CV_SUBMINOR_VERSION) >= 408) || \
    (CV_MAJOR_VERSION == 4 && (CV_MINOR_VERSION * 100 + CV_SUBMINOR_VERSION) >= 102)
    cv::setWindowProperty(window_name, cv::WND_PROP_TOPMOST, 1);
#endif
    return id;
}

namespace {
std::atomic<bool> signal_caught{false};

[[maybe_unused]] void sig_handler(int s) {
    MV_LOG_TRACE() << "Interrupt signal received." << std::endl;
    signal_caught = true;
}
} // anonymous namespace

int main(int argc, char *argv[]) {
    std::string serial;
    std::string biases_file;
    std::string in_raw_file_path;
    std::string out_raw_file_path;
    std::vector<uint16_t> roi;

    bool do_retry = false;

    const std::string short_program_desc(
        "Simple viewer to stream events from a RAW file or device, using the SDK driver API.\n");
    std::string long_program_desc(short_program_desc +
                                  "Press SPACE key while running to record or stop recording raw data\n"
                                  "Press 'q' or Escape key to leave the program.\n"
                                  "Press 'r' to toggle the hardware ROI given as input.\n"
                                  "Press 'h' to print this help.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("serial,s",          po::value<std::string>(&serial),"Serial ID of the camera. This flag is incompatible with flag '--input-raw-file'.")
        ("input-raw-file,i",  po::value<std::string>(&in_raw_file_path), "Path to input RAW file. If not specified, the camera live stream is used.")
        ("biases,b",          po::value<std::string>(&biases_file), "Path to a biases file. If not specified, the camera will be configured with the default biases.")
        ("output-raw-file,o", po::value<std::string>(&out_raw_file_path)->default_value("data.raw"), "Path to an output RAW file used for data recording. Default value is 'data.raw'. It also works when reading data from a RAW file.")
        ("roi,r",             po::value<std::vector<uint16_t>>(&roi)->multitoken(), "Hardware ROI to set on the sensor in the format [x y width height].")
    ;
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << short_program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << short_program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    if (!in_raw_file_path.empty()) {
        long_program_desc += "\nIf available, you can also:\n"
                             "Press 'b' key to seek backward 1s\n"
                             "Press 'f' key to seek forward 1s\n"
                             "Press 'o' to toggle the on screen display\n";
    }

    MV_LOG_INFO() << long_program_desc;

    if (vm.count("roi")) {
        if (!in_raw_file_path.empty()) {
            MV_LOG_ERROR() << "Options --roi and --input-raw-file are not compatible.";
            return 1;
        }
        if (roi.size() != 4) {
            MV_LOG_WARNING() << "ROI as argument must be in the format 'x y width height '. ROI has not been set.";
            roi.clear();
        }
    }

    do {
        Metavision::Camera camera;
        bool camera_is_opened = false;

        // If the filename is set, then read from the file
        if (!in_raw_file_path.empty()) {
            if (!serial.empty()) {
                MV_LOG_ERROR() << "Options --serial and --input-raw-file are not compatible.";
                return 1;
            }

            try {
                camera = Metavision::Camera::from_file(in_raw_file_path, true, Metavision::Future::RawFileConfig());
                camera_is_opened = true;
            } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
            // Otherwise, set the input source to the first available camera
        } else {
            try {
                if (!serial.empty()) {
                    camera = Metavision::Camera::from_serial(serial);
                } else {
                    camera = Metavision::Camera::from_first_available();
                }

                if (biases_file != "") {
                    camera.biases().set_from_file(biases_file);
                }

                if (!roi.empty()) {
                    camera.roi().set({roi[0], roi[1], roi[2], roi[3]});
                }
                camera_is_opened = true;
            } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
        }

        if (!camera_is_opened) {
            if (do_retry) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                MV_LOG_INFO() << "Trying to reopen camera...";
                continue;
            } else {
                return -1;
            }
        } else {
            MV_LOG_INFO() << "Camera has been opened successfully.";
        }

        // Add runtime error callback
        camera.add_runtime_error_callback([&do_retry](const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            do_retry = true;
        });

        // Get the geometry of the camera
        auto &geometry = camera.geometry(); // Get the geometry of the camera

        // All cameras have CDs events
        std::string cd_window_name("CD Events");
        cv::Mat cd_frame;
        Metavision::timestamp cd_frame_ts{0};
        Metavision::CDFrameGenerator cd_frame_generator(geometry.width(), geometry.height());
        cd_frame_generator.set_display_accumulation_time_us(10000);
        int cd_events_cb_id =
            setup_cd_callback_and_window(camera, cd_frame, cd_frame_ts, cd_frame_generator, cd_window_name);

        // Start the camera streaming
        camera.start();

        bool recording     = false;
        bool is_roi_set    = true;
        bool osc_available = false;
        bool osc_ready     = false;
        bool osd           = false;

        if (!in_raw_file_path.empty()) {
            try {
                camera.offline_streaming_control().is_ready();
                osc_available = true;
            } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
        }

        while (!signal_caught && camera.is_running()) {
            if (!in_raw_file_path.empty() && osc_available) {
                try {
                    bool was_osc_ready = osc_ready;
                    osc_ready          = camera.offline_streaming_control().is_ready();
                    if (!was_osc_ready && osc_ready) {
                        osd = true;
                    }
                } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
            }

            if (!cd_frame.empty()) {
                if (osd) {
                    const std::string text = human_readable_time(cd_frame_ts) + " / " +
                                             human_readable_time(camera.offline_streaming_control().get_duration());
                    cv::putText(cd_frame, text, cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(108, 143, 255),
                                1, cv::LINE_AA);
                }
                cv::imshow(cd_window_name, cd_frame);
            }

            // Wait for a pressed key for 33ms, that means that the display is refreshed at 30 FPS
            int key = process_ui_for(33);
            switch (key) {
            case 'q':
            case ESCAPE:
                camera.stop();
                do_retry = false;
                break;
            case SPACE:
                if (!recording) {
                    camera.start_recording(out_raw_file_path);
                } else {
                    camera.stop_recording();
                }
                recording = !recording;
                break;
            case 'b':
                if (osc_ready) {
                    Metavision::timestamp pos = cd_frame_ts - 1000 * 1000;
                    if (camera.offline_streaming_control().seek(pos)) {
                        cd_frame_generator.reset();
                    }
                }
                break;
            case 'f':
                if (osc_ready) {
                    Metavision::timestamp pos = cd_frame_ts + 1000 * 1000;
                    if (camera.offline_streaming_control().seek(pos)) {
                        cd_frame_generator.reset();
                    }
                }
                break;
            case 'o': {
                if (osc_ready) {
                    osd = !osd;
                }
                break;
            }
            case 'r': {
                if (roi.size() == 0) {
                    break;
                }
                if (!is_roi_set) {
                    camera.roi().set({roi[0], roi[1], roi[2], roi[3]});
                } else {
                    camera.roi().unset();
                }
                is_roi_set = !is_roi_set;
                break;
            }
            case 'h':
                MV_LOG_INFO() << long_program_desc;
                break;
            default:
                break;
            }
        }

        // unregister callbacks to make sure they are not called anymore
        if (cd_events_cb_id >= 0) {
            camera.cd().remove_callback(cd_events_cb_id);
        }

        // Stop the camera streaming, optional, the destructor will automatically do it
        camera.stop();
    } while (!signal_caught && do_retry);

    return signal_caught;
}
