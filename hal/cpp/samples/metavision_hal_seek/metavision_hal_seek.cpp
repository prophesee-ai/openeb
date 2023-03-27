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

#include <iostream>
#include <thread>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>

#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/facilities/i_geometry.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/base/utils/timestamp.h>

namespace po = boost::program_options;

int main(int argc, char **argv) {
    std::string in_raw_file_path;

    const std::string short_program_desc(
        "This code sample demonstrates how to use Metavision HAL to seek in a RAW file.\n"
        "Note that this sample uses HAL facilities that may not be available for the version of your camera plugin.\n");

    const std::string long_program_desc(short_program_desc +
                                        "Press 'q' key to leave the program.\n"
                                        "Move the slider to seek to different positions in the recording.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-raw-file,i",  po::value<std::string>(&in_raw_file_path)->required(), "Path to input RAW file.")
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

    MV_LOG_INFO() << long_program_desc;

    // Opens the RAW file with overloaded open_raw_file function
    Metavision::RawFileConfig config;
    auto device = Metavision::DeviceDiscovery::open_raw_file(in_raw_file_path, config);
    if (!device) {
        MV_LOG_ERROR() << "Failed to open device from file" << in_raw_file_path;
        return 1;
    }

    // Loads the index
    auto i_eventsstream = device->get_facility<Metavision::I_EventsStream>();
    if (!i_eventsstream) {
        MV_LOG_ERROR() << "A required facility for this sample is not available";
        return 1;
    }

    Metavision::timestamp start_ts, end_ts;
    Metavision::I_EventsStream::IndexStatus seek_status;
    bool index_built = false;
    while (!index_built) {
        seek_status = i_eventsstream->get_seek_range(start_ts, end_ts);
        switch (seek_status) {
        case Metavision::I_EventsStream::IndexStatus::Good:
            index_built = true;
            break;
        case Metavision::I_EventsStream::IndexStatus::Bad:
            MV_LOG_ERROR() << "Index for file" << in_raw_file_path << "could not be built";
            return -1;
        case Metavision::I_EventsStream::IndexStatus::Building: {
            static bool built_message_display = false;
            if (!built_message_display) {
                MV_LOG_INFO() << "Index for file" << in_raw_file_path << "is being built";
                built_message_display = true;
            }
            break;
        }
        default:
            break;
        }
        std::this_thread::yield();
    };
    MV_LOG_INFO() << "Index for file" << in_raw_file_path << "has been successfully loaded.";

    // Gets the facilities
    auto i_eventsstreamdecoder = device->get_facility<Metavision::I_EventsStreamDecoder>();
    if (!i_eventsstreamdecoder) {
        MV_LOG_ERROR() << "A required facility for this sample is not available";
        return 1;
    }
    auto i_cd_decoder = device->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();
    auto i_geometry   = device->get_facility<Metavision::I_Geometry>();

    // OpenCV window
    cv::Vec3b color_bg  = cv::Vec3b(52, 37, 30);
    cv::Vec3b color_on  = cv::Vec3b(236, 223, 216);
    cv::Vec3b color_off = cv::Vec3b(201, 126, 64);
    Metavision::timestamp seek_ts{0};
    Metavision::timestamp accumulation_time = 5000;
    cv::Mat display(i_geometry->get_height(), i_geometry->get_width(), CV_8UC3);
    const std::string window_name = "Metavision HAL Seek";
    cv::namedWindow(window_name, cv::WINDOW_GUI_EXPANDED);
    cv::resizeWindow(window_name, i_geometry->get_width(), i_geometry->get_height());

    // Setup decoding callback
    i_cd_decoder->add_event_buffer_callback([&](auto begin, auto end) {
        for (auto it = begin; it != end; ++it) {
            if (it->t >= seek_ts && it->t < seek_ts + accumulation_time) {
                display.at<cv::Vec3b>(it->y, it->x) = (it->p) ? color_on : color_off;
            }
        }
    });

    // OpenCV controls
    struct SeekBarOperator {
        SeekBarOperator(Metavision::timestamp &seek_ts, Metavision::timestamp start_ts, Metavision::timestamp end_ts) :
            seek_ts(seek_ts), start_ts(start_ts), end_ts(end_ts) {}
        void set(int value) {
            seek_ts = Metavision::timestamp(0.5 + start_ts + (end_ts - start_ts) * value / 100.);
        }
        Metavision::timestamp &seek_ts;
        Metavision::timestamp start_ts, end_ts;
    } seek_op(seek_ts, start_ts, end_ts);

    cv::createTrackbar(
        "Pos. (%)", window_name, nullptr, 100,
        [](int value, void *data) { reinterpret_cast<SeekBarOperator *>(data)->set(value); }, &seek_op);
    cv::setTrackbarMin("Pos. (%)", window_name, 0);
    cv::setTrackbarMax("Pos. (%)", window_name, 100);

    // Start reading the file
    i_eventsstream->start();

    // Now let's create the loop of main thread
    unsigned long frame = 0;
    while (true) {
        if (seek_ts >= 0) {
            MV_LOG_INFO() << "Seeking to position" << seek_ts;
            display.setTo(color_bg);
            long buffer_size_bytes;
            Metavision::timestamp ts_reached;
            const auto status = i_eventsstream->seek(seek_ts, ts_reached);
            if (status == Metavision::I_EventsStream::SeekStatus::Success) {
                i_eventsstreamdecoder->reset_timestamp(ts_reached);
                while (i_eventsstream->wait_next_buffer() > 0) {
                    auto buffer = i_eventsstream->get_latest_raw_data(buffer_size_bytes);
                    i_eventsstreamdecoder->decode(buffer, buffer + buffer_size_bytes);
                    if (i_eventsstreamdecoder->get_last_timestamp() > seek_ts + accumulation_time) {
                        break;
                    }
                }
            }
            cv::imshow(window_name, display);
            seek_ts = -1;
        }

        // if user presses `q` key, quit the loop
        int key = cv::waitKey(1);
        if ((key & 0xff) == 'q') {
            break;
        }
    }

    // Stop reading the file
    i_eventsstream->stop();

    return 0;
}
