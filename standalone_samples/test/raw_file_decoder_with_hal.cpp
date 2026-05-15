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
#include <iostream>
#include <fstream>
#include <metavision/hal/utils/hal_exception.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/utils/raw_file_config.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/events/event_ext_trigger.h>

int main(int argc, char *argv[]) {
    // Check input arguments validity
    if (argc < 3) {
        std::cerr << "Error : need input filename and output filename for CD events" << std::endl;
        std::cerr << std::endl
                  << "Usage : " << std::string(argv[0]) << " INPUT_FILENAME CD_OUTPUTFILE (TRIGGER_OUTPUTFILE)"
                  << std::endl;
        std::cerr << "Trigger output file will be written only if given as input" << std::endl;
        std::cerr << std::endl << "Example : " << std::string(argv[0]) << " input_file.raw cd_output.csv" << std::endl;
        std::cerr << std::endl;
        std::cerr << "The CD CSV file will have the format : x,y,polarity,timestamp" << std::endl;
        std::cerr << "The Trigger output CSV file will have the format : value,id,timestamp" << std::endl;
        return 1;
    }

    // Open camera
    std::unique_ptr<Metavision::Device> device;
    Metavision::RawFileConfig file_config;
    file_config.do_time_shifting_ = false;

    try {
        device = Metavision::DeviceDiscovery::open_raw_file(argv[1], file_config);
    } catch (Metavision::HalException &e) {
        std::cerr << "Error exception:" << e.what();
        return 1;
    }

    // Open CD csv output file
    std::ofstream cd_output_file(argv[2]);
    if (!cd_output_file.is_open()) {
        std::cerr << "Error : could not open file '" << argv[2] << "' for writing" << std::endl;
        return 1;
    }

    // Get the handler of CD events
    Metavision::I_EventDecoder<Metavision::EventCD> *i_cd_events_decoder =
        device->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();

    i_cd_events_decoder->add_event_buffer_callback(
        [&cd_output_file](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            std::string cd_str = "";

            for (const Metavision::EventCD *ev_cd = begin; ev_cd != end; ++ev_cd) {
                cd_str += std::to_string(ev_cd->x) + "," + std::to_string(ev_cd->y) + "," + std::to_string(ev_cd->p) +
                          "," + std::to_string(ev_cd->t) + "\n";
            }
            cd_output_file << cd_str;
        });

    // Open External Trigger csv output file, if provided
    std::ofstream trigger_output_file;
    if (argc > 3) {
        trigger_output_file.open(argv[3]);
        if (!trigger_output_file.is_open()) {
            std::cerr << "Error : could not open file '" << argv[3] << "' for writing" << std::endl;
            return 1;
        }

        // Get the handler of Trigger events
        Metavision::I_EventDecoder<Metavision::EventExtTrigger> *i_trigger_events_decoder =
            device->get_facility<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>();
        i_trigger_events_decoder->add_event_buffer_callback(
            [&trigger_output_file](const Metavision::EventExtTrigger *begin, const Metavision::EventExtTrigger *end) {
                std::string trigg_str = "";
                for (auto ev = begin; ev != end; ++ev) {
                    trigg_str +=
                        std::to_string(ev->p) + "," + std::to_string(ev->id) + "," + std::to_string(ev->t) + "\n";
                }
                trigger_output_file << trigg_str;
            });
    }

    // Get the decoder and event stream
    Metavision::I_Decoder *i_decoder           = device->get_facility<Metavision::I_EventsStreamDecoder>();
    Metavision::I_EventsStream *i_eventsstream = device->get_facility<Metavision::I_EventsStream>();
    i_eventsstream->start();

    while (true) {
        if (i_eventsstream->wait_next_buffer() < 0) {
            // No more events available (end of file reached)
            break;
        }

        // Retrieves raw buffer
        auto ev_buffer = i_eventsstream->get_latest_raw_data();

        // Decode the raw buffer
        i_decoder->decode(ev_buffer.begin(), ev_buffer.end());
    }

    return 0;
}
