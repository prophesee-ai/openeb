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

#include <string>
#include <fstream>
#include <boost/program_options.hpp>

#include <metavision/sdk/base/utils/log.h>
#include <metavision/hal/utils/hal_exception.h>
#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/utils/raw_file_config.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string in_raw_file_path;
    std::string out_raw_file_path;
    double start, end;

    const std::string program_desc(
        "Sample code that demonstrates how to use Metavision HAL API to cut a RAW file.\n"
        "Cuts a RAW file between <start> and <end> seconds where <start> and <end> are "
        "offsets from the beginning of the RAW file and can be expressed as floating point numbers.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-raw-file,i",    po::value<std::string>(&in_raw_file_path)->required(), "Path to input RAW file.")
        ("output-raw-file,o",   po::value<std::string>(&out_raw_file_path)->required(), "Path to output RAW file.")
        ("start,s",   po::value<double>(&start)->required(), "The start of the required sequence in seconds.")
        ("end,e",     po::value<double>(&end)->required(), "The end of the required sequence in seconds.")
        ;
    // clang-format on

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }
    try {
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (end <= start) {
        MV_LOG_ERROR() << "end time" << end << "is less than or equal to start" << start;
        return 1;
    }

    // convert start and end to microseconds
    Metavision::timestamp start_ts = static_cast<Metavision::timestamp>(start * 1000000);
    Metavision::timestamp end_ts   = static_cast<Metavision::timestamp>(end * 1000000);

    // Start processing
    std::unique_ptr<Metavision::Device> device;
    Metavision::RawFileConfig file_config;
    file_config.n_events_to_read_ = 1024; // Small amount of events per read to have a sufficient time precision and
                                          // decode efficiency to match the request

    try {
        device = Metavision::DeviceDiscovery::open_raw_file(in_raw_file_path, file_config);
    } catch (Metavision::HalException &e) {
        MV_LOG_ERROR() << "Error exception:" << e.what();
        return 1;
    }

    // Get the decoder and event stream
    Metavision::I_EventsStreamDecoder *i_eventsstreamdecoder =
        device->get_facility<Metavision::I_EventsStreamDecoder>();
    Metavision::I_EventsStream *i_eventsstream = device->get_facility<Metavision::I_EventsStream>();
    i_eventsstream->start();

    bool recording                = false;
    Metavision::timestamp last_ts = 0;
    while (true) {
        if (!recording) {
            if (last_ts >= start_ts) {
                i_eventsstream->log_raw_data(out_raw_file_path);
                recording = true;
            }
        } else {
            if (last_ts >= end_ts) {
                i_eventsstream->stop_log_raw_data();
                break;
            }
        }

        if (i_eventsstream->wait_next_buffer() < 0) {
            // No more events available (end of file reached)
            break;
        }

        // Retrieves raw buffer
        long n_rawbytes    = 0;
        uint8_t *ev_buffer = i_eventsstream->get_latest_raw_data(n_rawbytes);

        // Decode the raw buffer
        i_eventsstreamdecoder->decode(ev_buffer, ev_buffer + n_rawbytes);

        // Update last timestamp
        last_ts = i_eventsstreamdecoder->get_last_timestamp();
    }

    if (!recording) {
        MV_LOG_WARNING() << "No file saved because the start time provided is after the end of the input file";
    } else {
        MV_LOG_INFO() << "Output saved in file" << out_raw_file_path;
    }

    return 0;
}
