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

#include <cctype>
#include <chrono>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <opencv2/opencv.hpp>

#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/facilities/i_camera_synchronization.h>
#include <metavision/hal/facilities/i_digital_event_mask.h>
#include <metavision/hal/facilities/i_digital_crop.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/hal/facilities/i_event_rate_noise_filter_module.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/hal/facilities/i_geometry.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_monitoring.h>
#include <metavision/hal/facilities/i_plugin_software_info.h>
#include <metavision/hal/facilities/i_trigger_in.h>
#include <metavision/hal/facilities/i_trigger_out.h>
#include <metavision/hal/utils/hal_exception.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/events/event_ext_trigger.h>

class EventAnalyzer {
public:
    cv::Mat img, img_swap;

    std::mutex m;

    // Display colors
    cv::Vec3b color_bg  = cv::Vec3b(52, 37, 30);
    cv::Vec3b color_on  = cv::Vec3b(236, 223, 216);
    cv::Vec3b color_off = cv::Vec3b(201, 126, 64);

    void setup_display(const int width, const int height) {
        img      = cv::Mat(height, width, CV_8UC3);
        img_swap = cv::Mat(height, width, CV_8UC3);
        img.setTo(color_bg);
    }

    // Called from main Thread
    void get_display_frame(cv::Mat &display) {
        // Swap images
        {
            std::unique_lock<std::mutex> lock(m);
            std::swap(img, img_swap);
            img.setTo(color_bg);
        }
        img_swap.copyTo(display);
    }

    // Called from decoding Thread
    void process_events(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        // acquire lock
        {
            std::unique_lock<std::mutex> lock(m);
            for (auto it = begin; it != end; ++it) {
                img.at<cv::Vec3b>(it->y, it->x) = (it->p) ? color_on : color_off;
            }
        }
    }
};

namespace po = boost::program_options;
int main(int argc, char *argv[]) {
    std::string in_raw_file_path;
    std::string out_raw_file_path;
    std::string serial;
    std::string plugin_name;
    std::string pixel_mask_coord;
    std::string crop_region_coord;
    long system_id = -1;

    std::string temperature, illumination, pixel_dead_time;

    const std::string program_desc("Sample code that showcases various Metavision HAL facilities\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
            ("help,h", "Produce help message.")
            ("input-raw-file,i",  po::value<std::string>(&in_raw_file_path), "Path to input RAW file. If not specified, the camera live stream is used.")
            ("output-raw-file,o", po::value<std::string>(&out_raw_file_path), "Path to output RAW file.")
            ("serial,s",          po::value<std::string>(&serial), "Serial ID of the camera.")
            ("pixel-mask-coord,m",po::value<std::string>(&pixel_mask_coord), "Pixel mask coordinates (eg. '-m 12,34')")
            ("crop-region-coord,c",po::value<std::string>(&crop_region_coord), "Crop Region coordinates <startX, startY, endX, endY>[,<reset_orig>] (eg. '-c 12,23,34,45,1')")

            ;
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        std::cerr << program_desc << std::endl;
        std::cerr << options_desc << std::endl;
        std::cerr << "Parsing error: " << e.what() << std::endl;
        return 1;
    }

    if (vm.count("help")) {
        std::cout << program_desc << std::endl;
        std::cout << options_desc << std::endl;
        return 0;
    }

    // Open the device
    std::cout << "Opening camera..." << std::endl;
    std::unique_ptr<Metavision::Device> device;
    try {
        if (in_raw_file_path.empty()) {
            device = Metavision::DeviceDiscovery::open(serial);
        } else {
            device = Metavision::DeviceDiscovery::open_raw_file(in_raw_file_path);
        }
    } catch (Metavision::HalException &e) { std::cout << "Error exception: " << e.what() << std::endl; }

    if (!device) {
        std::cerr << "Camera opening failed." << std::endl;
        return 1;
    }
    std::cout << "Camera open." << std::endl;

    Metavision::I_PluginSoftwareInfo *i_pluginsoftwareinfo = device->get_facility<Metavision::I_PluginSoftwareInfo>();
    if (i_pluginsoftwareinfo) {
        plugin_name = i_pluginsoftwareinfo->get_plugin_name();
        std::cout << "Plugin used: " << plugin_name << std::endl;
    }

    Metavision::I_HW_Identification *i_hw_identification = device->get_facility<Metavision::I_HW_Identification>();
    if (i_hw_identification) {
        system_id = i_hw_identification->get_system_id();
        std::cout << "System ID: " << system_id << std::endl;
    }

    Metavision::I_CameraSynchronization *i_camera_synchronization =
        device->get_facility<Metavision::I_CameraSynchronization>();
    if (in_raw_file_path.empty() && !i_camera_synchronization) {
        std::cerr << "Could not get Camera Synchronization facility." << std::endl;
        return 1;
    }

    /// [RawfileCreation]
    Metavision::I_EventsStream *i_eventsstream = device->get_facility<Metavision::I_EventsStream>();
    if (i_eventsstream) {
        if (out_raw_file_path != "") {
            i_eventsstream->log_raw_data(out_raw_file_path);
        }
    } else {
        std::cerr << "Could not initialize events stream." << std::endl;
        return 3;
    }
    /// [RawfileCreation]

    /// [triggers]
    // On camera providing Trigger Out, we enable it and duplicate the signal on Trigger In using the loopback channel
    // On the other cameras, we enable Trigger In, but we will need to plug a signal generator to create trigger events
    // and we also set the camera as Master so that we can test the Sync Out signal if needed.
    Metavision::I_TriggerOut *i_trigger_out = device->get_facility<Metavision::I_TriggerOut>();
    Metavision::I_TriggerIn *i_trigger_in   = device->get_facility<Metavision::I_TriggerIn>();
    if (i_trigger_in) {
        auto channels = i_trigger_in->get_available_channels();
        if (channels.find(Metavision::I_TriggerIn::Channel::Loopback) != channels.end() && i_trigger_out) {
            std::cout << "Trigger loopback enabled" << std::endl;
            i_trigger_in->enable(Metavision::I_TriggerIn::Channel::Loopback);
            i_trigger_out->set_period(100000);
            i_trigger_out->set_duty_cycle(0.5);
            i_trigger_out->enable();
        } else if (i_camera_synchronization) {
            std::cout << "Could not enable trigger loopback" << std::endl;
            i_trigger_in->enable(Metavision::I_TriggerIn::Channel::Main);
            i_camera_synchronization->set_mode_master();
        }
    }
    /// [triggers]

    /// [geometry]
    Metavision::I_Geometry *i_geometry = device->get_facility<Metavision::I_Geometry>();
    if (!i_geometry) {
        std::cerr << "Could not retrieve geometry." << std::endl;
        return 4;
    }
    std::cout << "Device geometry : " << i_geometry->get_width() << "x" << i_geometry->get_height() << std::endl;
    /// [geometry]

    // Instantiate framer object
    EventAnalyzer event_analyzer;
    event_analyzer.setup_display(i_geometry->get_width(), i_geometry->get_height());

    /// [sign-up cd callback]
    // Get the handler of CD events
    Metavision::I_EventDecoder<Metavision::EventCD> *i_cddecoder =
        device->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();

    if (i_cddecoder) {
        // Register a lambda function to be called on every CD events
        i_cddecoder->add_event_buffer_callback(
            [&event_analyzer](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
                event_analyzer.process_events(begin, end);
            });
    }
    /// [sign-up cd callback]

    Metavision::I_EventDecoder<Metavision::EventExtTrigger> *i_triggerdecoder =
        device->get_facility<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>();
    if (i_triggerdecoder) {
        i_triggerdecoder->add_event_buffer_callback(
            [](const Metavision::EventExtTrigger *begin, const Metavision::EventExtTrigger *end) {
                for (auto ev = begin; ev != end; ++ev) {
                    std::cout << "Trigger "
                              << " " << ev->t << " " << ev->id << " " << ev->p << std::endl;
                }
            });
    } else {
        std::cout << "No trigger decoder." << std::endl;
    }

    Metavision::I_Monitoring *i_monitoring = device->get_facility<Metavision::I_Monitoring>();

    /// [biases]
    // Reading biases from the @ref Metavision::I_LL_Biases facility
    Metavision::I_LL_Biases *i_ll_biases = device->get_facility<Metavision::I_LL_Biases>();
    // check biases
    if (i_ll_biases) {
        auto biases_to_check = i_ll_biases->get_all_biases();
        for (auto &b : biases_to_check) {
            auto v = i_ll_biases->get(b.first);
            std::cout << "Initial value: " << b.first << " " << v << std::endl;
        }
    }
    /// [biases]

    Metavision::I_EventRateNoiseFilterModule *i_event_rate_noise_filter_module =
        device->get_facility<Metavision::I_EventRateNoiseFilterModule>();
    if (i_event_rate_noise_filter_module) {
        std::cout << "Event rate noise filter: streaming from "
                  << i_event_rate_noise_filter_module->get_event_rate_threshold() << "Kev/s" << std::endl;
    }

    auto tokenize = [](std::string str, std::string separator) {
        using Separator = boost::char_separator<char>;
        boost::tokenizer<Separator> tokenizer(str, Separator(separator.c_str()));

        std::vector<size_t> tokens;
        std::transform(tokenizer.begin(), tokenizer.end(), std::back_inserter(tokens),
                       [](const auto &token) { return std::stoul(token); });

        return tokens;
    };

    Metavision::I_DigitalEventMask *digital_event_mask = device->get_facility<Metavision::I_DigitalEventMask>();
    if (pixel_mask_coord.size() && digital_event_mask) {
        try {
            auto tokens = tokenize(pixel_mask_coord, ",");
            if (tokens.size() < 2) {
                throw std::runtime_error{"Failed to parse pixel coordinates"};
            }

            size_t x = tokens[0];
            size_t y = tokens[1];
            std::cout << "Masking pixel {" << x << "," << y << "} - no event will be generated by this pixel \n";

            auto masks = digital_event_mask->get_pixel_masks();
            if (masks.size() >= 1) {
                masks[0]->set_mask(x, y, true);
            }
        } catch (...) {
            std::cerr << "Failed to parse Pixel mask coordinates : " << pixel_mask_coord
                      << " - please see helper description \n";
        }
    }

    auto digital_crop = device->get_facility<Metavision::I_DigitalCrop>();
    if (crop_region_coord.size()) {
        if (!digital_crop) {
            std::cout << "Could not get Digital Crop facility.\n";
        } else {
            try {
                auto crop = tokenize(crop_region_coord, ",");
                if (crop.size() < 4) {
                    throw std::runtime_error{
                        "Failed to parse Crop Region coords (<startX>,<startY>,<endX>,<endY>[,<reset_orig>])"};
                }
                bool reset_orig = crop.size() == 5 ? crop[4] : false;
                digital_crop->set_window_region({crop[0], crop[1], crop[2], crop[3]}, reset_orig);
                digital_crop->enable(true);
                std::cout << "Crop from window region {" << crop[0] << "," << crop[1] << "," << crop[2] << ","
                          << crop[3] << "} (reset origin : " << reset_orig << ")\n";
            } catch (std::exception &e) {
                std::cerr << "Failed to activate Digital Crop :";
                std::cerr << e.what() << "\n";
            }
        }
    }

    // Get the decoder of events & start decoding thread
    Metavision::I_EventsStreamDecoder *i_eventsstreamdecoder =
        device->get_facility<Metavision::I_EventsStreamDecoder>();
    bool stop_decoding    = false;
    bool stop_application = false;
    i_eventsstream->start();
    if (in_raw_file_path.empty()) {
        std::cout << "Camera started." << std::endl;
    }
    std::thread decoding_loop([&]() {
        using namespace std::chrono;
        milliseconds last_update_monitoring = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
        while (!stop_decoding) {
            short ret = i_eventsstream->poll_buffer();
            if (ret < 0) {
                std::cout << "End of file" << std::endl;
                i_eventsstream->stop();
                i_eventsstream->stop_log_raw_data();
                if (in_raw_file_path.empty()) {
                    std::cout << "Camera stopped." << std::endl;
                }
                stop_decoding    = true;
                stop_application = true;
            } else if (ret == 0) {
                if (i_monitoring) {
                    milliseconds current_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
                    if ((current_time - last_update_monitoring).count() > 1000) {
                        last_update_monitoring = current_time;

                        auto call_monitoring_func = [&i_monitoring](auto func) {
                            try {
                                int res = (i_monitoring->*func)();
                                return std::to_string(res);
                            } catch (...) { return std::string("N/A"); }
                        };
                        temperature     = call_monitoring_func(&Metavision::I_Monitoring::get_temperature);
                        illumination    = call_monitoring_func(&Metavision::I_Monitoring::get_illumination);
                        pixel_dead_time = call_monitoring_func(&Metavision::I_Monitoring::get_pixel_dead_time);
                    }
                }
                continue;
            }

            /// [buffer]
            // Here we polled data, so we can launch decoding
            long n_bytes;
            uint8_t *raw_data = i_eventsstream->get_latest_raw_data(n_bytes);

            // This will trigger callbacks set on decoders: in our case EventAnalyzer.process_events
            i_eventsstreamdecoder->decode(raw_data, raw_data + n_bytes);
            /// [buffer]
        }
    });

    // Prepare OpenCV window
    const int fps       = 25; // event-based cameras do not have a frame rate, but we need one for visualization
    const int wait_time = static_cast<int>(std::round(1.f / fps * 1000)); // how much we should wait between two frames
    cv::Mat display;                                                      // frame where events will be accumulated
    const std::string window_name = "Metavision HAL Showcase";
    cv::namedWindow(window_name, cv::WINDOW_GUI_EXPANDED);
    cv::resizeWindow(window_name, i_geometry->get_width(), i_geometry->get_height());

    // Now let's create the loop of main thread
    unsigned long frame = 0;
    while (!stop_application) {
        event_analyzer.get_display_frame(display);

        if (!display.empty()) {
            cv::imshow(window_name, display);
        }

        if (i_monitoring && frame % 25 == 0) {
            std::cout << "Temp : " << temperature << "\t| Illu : " << illumination
                      << "\t| Pixel Dead Time : " << pixel_dead_time << std::endl;
        }

        // if user presses `q` key, quit the loop
        int key = cv::waitKey(wait_time);
        if ((key & 0xff) == 'q') {
            stop_application = true;
            stop_decoding    = true;
            std::cout << "q pressed, exiting." << std::endl;
        }
        frame++;
    }

    // Wait end of decoding loop
    decoding_loop.join();

    return 0;
}
