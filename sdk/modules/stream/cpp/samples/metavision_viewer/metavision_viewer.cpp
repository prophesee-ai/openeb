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

// Example of using Metavision SDK Stream API for visualizing events stream, setting ROI and ERC as well as
// saving and loading the camera settings to and from camera settings files

#include <atomic>
#include <chrono>
#include <iomanip>
#include <signal.h>
#include <thread>
#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>
#if CV_MAJOR_VERSION >= 4
#include <opencv2/highgui/highgui_c.h>
#endif
#include <opencv2/imgproc.hpp>
#include <metavision/hal/facilities/i_erc_module.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_roi.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/core/utils/misc.h>
#include <metavision/sdk/core/utils/rate_estimator.h>
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/stream/camera_error_code.h>
#include <metavision/hal/facilities/i_plugin_software_info.h>

static const int ESCAPE = 27;
static const int SPACE  = 32;

namespace po = boost::program_options;

int processUI(int delay_ms) {
    auto then = std::chrono::high_resolution_clock::now();
    int key   = cv::waitKey(delay_ms);
    auto now  = std::chrono::high_resolution_clock::now();
    // cv::waitKey will not wait if no window is opened, so we wait for it, if needed
    std::this_thread::sleep_for(std::chrono::milliseconds(
        delay_ms - std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count()));

    return key;
}

struct RoiControl {
    bool use_windows; // Whether to set ROIs through windows or lines
    std::vector<bool> cols;
    std::vector<bool> rows;
    std::vector<Metavision::I_ROI::Window> windows;
    Metavision::I_ROI::Mode mode;
    cv::Point mouse_down_coord; // Coordinates of initial pixel while left mouse button is held down
    bool need_refresh; // Whether ROIs need to be updated on the device
    const std::size_t max_windows;

    RoiControl(int width, int height, std::size_t max_supported_windows) :
        cols(width, 0), rows(height, 0), max_windows(max_supported_windows), mode(Metavision::I_ROI::Mode::ROI) {
        reset();
    }

    void reset() {
        std::fill(cols.begin(), cols.end(), false);
        std::fill(rows.begin(), rows.end(), false);
        windows.clear();
        mouse_down_coord.x = -1;
        need_refresh = false;
    }
};

void receiveMouseEvent(int event, int x, int y, int flags, void *userdata) {
    RoiControl *roi_ctrl = reinterpret_cast<RoiControl *>(userdata);

    switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        if (roi_ctrl->mouse_down_coord.x < 0) {
            roi_ctrl->mouse_down_coord.x = x;
            roi_ctrl->mouse_down_coord.y = y;
        }
        break;
    case cv::EVENT_LBUTTONUP:
        if (roi_ctrl->mouse_down_coord.x < 0) {
            break;
        }

        // Just a click from the user, ignore it
        if (roi_ctrl->mouse_down_coord.x == x && roi_ctrl->mouse_down_coord.y == y) {
            roi_ctrl->mouse_down_coord.x = -1;
            break;
        }

        {
            const int start_x = std::min(roi_ctrl->mouse_down_coord.x, x);
            const int end_x = std::max(roi_ctrl->mouse_down_coord.x, x);
            const int start_y = std::min(roi_ctrl->mouse_down_coord.y, y);
            const int end_y = std::max(roi_ctrl->mouse_down_coord.y, y);

            if (roi_ctrl->use_windows) {
                if (roi_ctrl->windows.size() >= roi_ctrl->max_windows) {
                    roi_ctrl->windows.clear();
                }
                roi_ctrl->windows.push_back({start_x, start_y, end_x - start_x + 1, end_y - start_y + 1});
            } else {
                for (int i = start_x; i <= end_x; ++i) {
                    roi_ctrl->cols[i] = true;
                }

                for (int i = start_y; i <= end_y; ++i) {
                    roi_ctrl->rows[i] = true;
                }
            }
            roi_ctrl->need_refresh = true;
        }

        roi_ctrl->mouse_down_coord.x = -1;
        break;
    default:
        break;
    }
}

namespace {
std::atomic<bool> signal_caught{false};

[[maybe_unused]] void signalHandler(int s) {
    MV_LOG_TRACE() << "Interrupt signal received." << std::endl;
    signal_caught = true;
}
} // anonymous namespace

int main(int argc, char *argv[]) {
    signal(SIGINT, signalHandler);

    std::string serial;
    std::string biases_file;
    std::string event_file_path;
    std::string in_cam_config_path;
    std::string out_file_path;
    std::string out_cam_config_path;
    std::vector<uint16_t> roi;
    std::vector<uint16_t> ssf;

    std::atomic<bool> do_retry = false;

    const std::string short_program_desc(
        "Simple viewer to stream events from an event file or a device, using the SDK Stream API.\n");
    std::string long_program_desc(short_program_desc +
                                  "Define a region using the cursor (click and drag) to set a Region of Interest (ROI)\n"
                                  "Press SPACE key while running to record or stop recording raw data\n"
                                  "Press 'q' or Escape key to leave the program.\n"
                                  "Press 'o' to toggle the on screen display.\n"
                                  "Press 'l' to load the camera settings from the input camera config file.\n"
                                  "Press 's' to save the camera settings to the output camera config file.\n"
                                  "Press 'r' to toggle the hardware ROI mode (window mode or lines mode, default: window mode).\n"
                                  "Press 'R' to toggle the ROI/RONI mode.\n"
                                  "Press 'S' to toggle the subsampling.\n"
                                  "Press 'e' to toggle the ERC module (if available).\n"
                                  "Press '+' to increase the ERC threshold (if available).\n"
                                  "Press '-' to decrease the ERC threshold (if available).\n"
                                  "Press 'h' to print this help.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("serial,s",              po::value<std::string>(&serial),"Serial ID of the camera. This flag is incompatible with flag '--input-event-file'.")
        ("input-event-file,i",    po::value<std::string>(&event_file_path), "Path to input event file (RAW, DAT or HDF5). If not specified, the camera live stream is used.")
        ("input-camera-config,j", po::value<std::string>(&in_cam_config_path), "Path to a JSON file containing camera config settings to restore a camera state. Only works for live cameras.")
        ("biases,b",              po::value<std::string>(&biases_file), "Path to a biases file. If not specified, the camera will be configured with the default biases.")
        ("output-file,o",         po::value<std::string>(&out_file_path)->default_value("data.raw"), "Path to an output file used for data recording. Default value is 'data.raw'. It also works when reading data from a file.")
        ("output-camera-config",  po::value<std::string>(&out_cam_config_path)->default_value("settings.json"), "Path to a JSON file where to save the camera config settings. Default value is 'settings.json'. Only works for live camera.")
        ("roi,r",                 po::value<std::vector<uint16_t>>(&roi)->multitoken(), "Hardware ROI to set on the sensor in the format [x y width height].")
        ("subsampling,d",         po::value<std::vector<uint16_t> >(&ssf)->multitoken(), "subsampling factor (ssf) in the format [ssf_row ssf_col]. For example [2 4] keep 1 row over 2 and 1 column over 4.")
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

    if (!event_file_path.empty()) {
        long_program_desc += "\nIf available, you can also:\n"
                             "Press 'b' key to seek backward 1s\n"
                             "Press 'f' key to seek forward 1s\n";
    }

    MV_LOG_INFO() << long_program_desc;
    
    if (vm.count("roi")) {
        if (!event_file_path.empty()) {
            MV_LOG_ERROR() << "Options --roi and --input-event-file are not compatible.";
            return 1;
        }
        if (roi.size() != 4) {
            MV_LOG_WARNING() << "ROI as argument must be in the format 'x y width height '. ROI has not been set.";
            roi.clear();
        }
    }

    if (vm.count("subsampling")) {
        if (!event_file_path.empty()) {
            MV_LOG_ERROR() << "Options --ssf and --input-event-file are not compatible.";
            return 1;
        }
        if (ssf.size() != 2)  {
            MV_LOG_WARNING() << "ssf as argument must be in the format [ssf_row ssf_col]. subsampling has not been enabled.";
            ssf.clear();
        }
        if (ssf[0]==0 || ssf[1]==0 ){
            MV_LOG_ERROR() << "The subsampling parameters were incorrectly defined. The ssf parameters must be strictly positive integers.";
            ssf.clear();
        }

    }    

    do {
        Metavision::Camera camera;
        bool camera_is_opened = false;

        // If the filename is set, then read from the file
        if (!event_file_path.empty()) {
            if (!serial.empty()) {
                MV_LOG_ERROR() << "Options --serial and --input-event-file are not compatible.";
                return 1;
            }

            try {
                Metavision::FileConfigHints hints;
                camera           = Metavision::Camera::from_file(event_file_path, hints);
                camera_is_opened = true;
            } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
            // Otherwise, set the input source to a camera
        } else {
            try {
                if (!serial.empty()) {
                    camera = Metavision::Camera::from_serial(serial);
                } else {
                    camera = Metavision::Camera::from_first_available();
                }

                if (!in_cam_config_path.empty()) {
                    try {
                        camera.load(in_cam_config_path);
                    } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
                }

                if (biases_file != "") {
                    camera.get_facility<Metavision::I_LL_Biases>().load_from_file(biases_file);
                }

                camera_is_opened = true;
            } catch (Metavision::CameraException &e) {
                MV_LOG_ERROR() << e.what();
                if (e.code().value() == Metavision::CameraErrorCode::ConnectionError) {
                    do_retry = true;
                    MV_LOG_INFO() << "Trying to reopen camera...";
                    continue;
                }
            }
        }

        // With the HAL device corresponding to the camera object (file or live camera), we can try to get a facility
        // This gives us access to extra HAL features not covered by the SDK Stream camera API
        try {
            auto *plugin_sw_info = camera.get_device().get_facility<Metavision::I_PluginSoftwareInfo>();
            if (plugin_sw_info) {
                const std::string &plugin_name = plugin_sw_info->get_plugin_name();
                MV_LOG_INFO() << "Plugin used to open the device:" << plugin_name;
            }
        } catch (Metavision::CameraException &) {
            // we ignore the exception as some devices will not provide this facility (e.g. HDF5 files)
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
            if (e.code().value() == Metavision::CameraErrorCode::ConnectionError) {
                MV_LOG_ERROR() << "Lost connection with the device. Please try replugging the device";
            }
            MV_LOG_ERROR() << e.what();
            do_retry = true;
        });

        // Get the geometry of the camera
        const auto &geometry = camera.geometry();

        // Setup CD event rate estimator
        double avg_rate, peak_rate;
        Metavision::RateEstimator cd_rate_estimator(
            [&avg_rate, &peak_rate](Metavision::timestamp ts, double arate, double prate) {
                avg_rate  = arate;
                peak_rate = prate;
            },
            100000, 1000000, true);

        // Setup CD frame generator
        std::mutex cd_frame_generator_mutex;
        Metavision::CDFrameGenerator cd_frame_generator(geometry.get_width(), geometry.get_height());
        cd_frame_generator.set_display_accumulation_time_us(10000);

        std::mutex cd_frame_mutex;
        cv::Mat cd_frame;
        Metavision::timestamp cd_frame_ts{0};
        cd_frame_generator.start(
            30, [&cd_frame_mutex, &cd_frame, &cd_frame_ts](const Metavision::timestamp &ts, const cv::Mat &frame) {
                std::unique_lock<std::mutex> lock(cd_frame_mutex);
                cd_frame_ts = ts;
                frame.copyTo(cd_frame);
            });

        unsigned int max_roi_windows = 0;
        try {
            max_roi_windows = camera.get_facility<Metavision::I_ROI>().get_max_supported_windows_count();
        } catch (...) {}
        RoiControl roi_ctrl(geometry.get_width(), geometry.get_height(), max_roi_windows);
        roi_ctrl.use_windows = true;
        if (roi.size() != 0) {
            roi_ctrl.need_refresh = true;
            roi_ctrl.windows.push_back({roi[0], roi[1], roi[2], roi[3]});
        } else {
            roi_ctrl.need_refresh = false;
        }

        // Setup CD frame display
        std::string cd_window_name("CD Events");
        cv::namedWindow(cd_window_name, CV_GUI_NORMAL);
        cv::resizeWindow(cd_window_name, geometry.get_width(), geometry.get_height());
        cv::moveWindow(cd_window_name, 0, 0);
#if (CV_MAJOR_VERSION == 3 && (CV_MINOR_VERSION * 100 + CV_SUBMINOR_VERSION) >= 408) || \
    (CV_MAJOR_VERSION == 4 && (CV_MINOR_VERSION * 100 + CV_SUBMINOR_VERSION) >= 102)
        cv::setWindowProperty(cd_window_name, cv::WND_PROP_TOPMOST, 1);
#endif
        cv::setMouseCallback(cd_window_name, receiveMouseEvent, &roi_ctrl);

        // Setup camera CD callback to update the frame generator and event rate estimator
        int cd_events_cb_id =
            camera.cd().add_callback([&cd_frame_generator_mutex, &cd_frame_generator, &cd_rate_estimator](
                                         const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
                std::unique_lock<std::mutex> lock(cd_frame_generator_mutex);
                cd_frame_generator.add_events(ev_begin, ev_end);
                cd_rate_estimator.add_data(std::prev(ev_end)->t, std::distance(ev_begin, ev_end));
            });

        // Start the camera streaming
        try {
            camera.start();
        } catch (const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            if (e.code().value() == Metavision::CameraErrorCode::ConnectionError) {
                do_retry = true;
                MV_LOG_INFO() << "Trying to reopen camera...";
                continue;
            }
        }

        bool recording     = false;
        bool osc_available = false;
        bool osc_ready     = false;
        bool osd           = true;
        bool ssf_enabled   = false;

        if (!event_file_path.empty()) {
            try {
                camera.offline_streaming_control().is_ready();
                osc_available = true;
            } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
        }

        while (!signal_caught && camera.is_running()) {
            if (!event_file_path.empty() && osc_available) {
                try {
                    osc_ready = camera.offline_streaming_control().is_ready();
                } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
            }

            {
                std::unique_lock<std::mutex> lock(cd_frame_mutex);
                if (!cd_frame.empty()) {
                    if (osd) {
                        std::string text;
                        if (osc_ready) {
                            text = Metavision::getHumanReadableTime(cd_frame_ts) + " / " +
                                   Metavision::getHumanReadableTime(camera.offline_streaming_control().get_duration());
                        } else {
                            text = Metavision::getHumanReadableTime(cd_frame_ts);
                        }
                        text += "     ";
                        text += Metavision::getHumanReadableRate(avg_rate);
                        cv::putText(cd_frame, text, cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1,
                                    cv::Scalar(108, 143, 255), 1, cv::LINE_AA);
                    }
                    cv::imshow(cd_window_name, cd_frame);
                }
            }

            if (roi_ctrl.need_refresh) {
                try {
                    auto &roi_facility = camera.get_facility<Metavision::I_ROI>();
                    if (roi_ctrl.use_windows) {
                        roi_facility.set_windows(roi_ctrl.windows);
                    } else {
                        roi_facility.set_lines(roi_ctrl.cols, roi_ctrl.rows);
                    }
                    roi_facility.enable(true);
                } catch (...) {}
                roi_ctrl.need_refresh = false;
                ssf_enabled = false;
            }

            // Wait for a pressed key for 33ms, that means that the display is refreshed at 30 FPS
            int key = processUI(33);
            switch (key) {
            case 'q':
            case ESCAPE:
                camera.stop();
                do_retry = false;
                break;
            case SPACE:
                if (!recording) {
                    MV_LOG_INFO() << "Started recording in" << out_file_path;
                    camera.start_recording(out_file_path);
                } else {
                    MV_LOG_INFO() << "Stopped recording in" << out_file_path;
                    camera.stop_recording(out_file_path);
                }
                recording = !recording;
                break;
            case 'b':
                if (osc_ready) {
                    std::unique_lock<std::mutex> lock(cd_frame_mutex);
                    Metavision::timestamp pos = cd_frame_ts - 1000 * 1000;
                    if (camera.offline_streaming_control().seek(pos)) {
                        std::unique_lock<std::mutex> lock2(cd_frame_generator_mutex);
                        cd_frame_generator.reset();
                        MV_LOG_INFO() << "Seeking backward to" << (pos / 1.e6) << "s";
                    }
                }
                break;
            case 'f':
                if (osc_ready) {
                    std::unique_lock<std::mutex> lock(cd_frame_mutex);
                    Metavision::timestamp pos = cd_frame_ts + 1000 * 1000;
                    if (camera.offline_streaming_control().seek(pos)) {
                        std::unique_lock<std::mutex> lock2(cd_frame_generator_mutex);
                        cd_frame_generator.reset();
                        MV_LOG_INFO() << "Seeking forward to" << (pos / 1.e6) << "s";
                    }
                }
                break;
            case 'o': {
                osd = !osd;
                break;
            }
            case 'l': {
                if (in_cam_config_path.empty()) {
                    break;
                }
                try {
                    camera.load(in_cam_config_path);
                    MV_LOG_INFO() << "Settings restored from" << in_cam_config_path;
                } catch (Metavision::CameraException &) {}
                break;
            }
            case 's': {
                if (out_cam_config_path.empty()) {
                    break;
                }
                try {
                    camera.save(out_cam_config_path);
                    MV_LOG_INFO() << "Settings saved to" << out_cam_config_path;
                } catch (Metavision::CameraException &) {}
                break;
            }
            case 'S': {   
                ssf_enabled = !ssf_enabled; 
                std::vector<bool> rows(geometry.get_width(), false);
                std::vector<bool> cols (geometry.get_height(), false);
                
                roi_ctrl.need_refresh = false;       

                if (ssf_enabled && ssf.size() == 2) {                  
                    for (int i = 0; i < (1 + round(rows.size()/ssf[0])); i++) {
                        rows[ssf[0]*i] = true;
                    }                    
                    for (int i = 0; i < (1 + round(cols.size()/ssf[1])); i++) {
                        cols[ssf[1]*i] = true;
                    }
                    try {
                        camera.get_facility<Metavision::I_ROI>().set_lines(rows, cols);
                        camera.get_facility<Metavision::I_ROI>().enable(true);
                        MV_LOG_INFO() << "Subsampling enabled";
                    } catch (...) {} 
                } else {
                    if (camera.get_facility<Metavision::I_ROI>().is_enabled()) {
                        camera.get_facility<Metavision::I_ROI>().enable(false);
                        MV_LOG_INFO() << "Subsampling disabled";
                    }                  
                }       
                break;
            }
            case 'r': {
                roi_ctrl.use_windows = !roi_ctrl.use_windows;
                if (roi_ctrl.use_windows) {
                    MV_LOG_INFO() << "ROI: window mode";
                } else {
                    MV_LOG_INFO() << "ROI: lines mode";
                }
                roi_ctrl.reset();
                try {
                    if (camera.get_facility<Metavision::I_ROI>().is_enabled()) {
                        camera.get_facility<Metavision::I_ROI>().enable(false);
                    }
                } catch (...) {
                    MV_LOG_INFO() << "No ROI facility available";
                }
                break;
            }
            case 'R': {
                if (roi_ctrl.mode == Metavision::I_ROI::Mode::ROI) {
                    MV_LOG_INFO() << "Switching to RONI mode";
                    roi_ctrl.mode = Metavision::I_ROI::Mode::RONI;
                } else {
                    MV_LOG_INFO() << "Switching to ROI mode";
                    roi_ctrl.mode = Metavision::I_ROI::Mode::ROI;
                }

                try {
                    auto &roi_facility = camera.get_facility<Metavision::I_ROI>();
                    if (roi_facility.is_enabled()) {
                        roi_facility.enable(false);
                    }
                    roi_facility.set_mode(roi_ctrl.mode);
                    roi_ctrl.need_refresh = true;
                } catch (...) {
                    MV_LOG_INFO() << "No ROI facility available";
                }
                break;
            }

            case 'e': {
                try {
                    auto &erc_module = camera.get_facility<Metavision::I_ErcModule>();
                    erc_module.enable(!erc_module.is_enabled());
                    MV_LOG_INFO() << "ERC:" << (erc_module.is_enabled() ? "enabled" : "disabled");
                } catch (Metavision::CameraException &) {}
                break;
            }
            case '+': {
                try {
                    auto &erc_module = camera.get_facility<Metavision::I_ErcModule>();
                    erc_module.set_cd_event_rate(erc_module.get_cd_event_rate() + 10000000);
                    MV_LOG_INFO() << "ERC:" << (erc_module.get_cd_event_rate() / 1000000) << "Mev/s";
                } catch (Metavision::CameraException &) {}
                break;
            }
            case '-': {
                try {
                    auto &erc_module = camera.get_facility<Metavision::I_ErcModule>();
                    uint32_t current_rate = erc_module.get_cd_event_rate();
                    uint32_t target_rate = current_rate < 10000000 ? 0 : current_rate - 10000000;

                    erc_module.set_cd_event_rate(target_rate);
                    MV_LOG_INFO() << "ERC:" << (erc_module.get_cd_event_rate() / 1000000) << "Mev/s";
                } catch (Metavision::CameraException &) {}
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
        try {
            camera.stop();
        } catch (const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
        }
        cd_frame_generator.stop();
    } while (!signal_caught && do_retry);

    return signal_caught;
}
