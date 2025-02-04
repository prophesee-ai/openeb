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

#include "utils/make_decoder.h"

#include <utility>
#include <stdexcept>
#include <sstream>

#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_event_frame_decoder.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/decoders/aer/aer_decoder.h"
#include "metavision/hal/decoders/evt2/evt2_decoder.h"
#include "metavision/hal/decoders/evt21/evt21_decoder.h"
#include "metavision/hal/decoders/evt3/evt3_decoder.h"
#include "metavision/hal/decoders/evt4/evt4_decoder.h"
#include "metavision/hal/decoders/ehc/ehc_decoder.h"
#include "metavision/hal/decoders/mtr/mtr_decoder.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"

namespace Metavision {

// Gets pixel layout as a pair of integers representing (negative_bits, positive_bits)
static std::pair<unsigned, unsigned> get_pixel_layout(const std::string &layout_str) {
    try {
        unsigned neg_bits = 0, pos_bits = 0;
        std::string str;
        std::istringstream layout(layout_str);
        // Expected format: 5p/2n
        // This code isn't guaranteed to detect malformed headers
        for (int i = 0; i < 2; ++i) {
            std::getline(layout, str, '/');
            if (str[str.length() - 1] == 'p') {
                pos_bits = std::stoi(str.substr(0, str.length() - 1));
            } else if (str[str.length() - 1] == 'n') {
                neg_bits = std::stoi(str.substr(0, str.length() - 1));
            } else {
                throw std::invalid_argument("");
            }
        }
        if (!pos_bits && !neg_bits) {
            throw std::invalid_argument("");
        }
        return std::pair<unsigned, unsigned>(neg_bits, pos_bits);
    } catch (...) {
        /* Use catch-all handler on purpose, as `stoi("")` may throw exceptions that
         * are of other type than std::exception on some platforms (eg. Android...)
         */
        throw std::invalid_argument("Format is missing a valid pixel layout");
    }
}

std::shared_ptr<I_EventsStreamDecoder> make_decoder(DeviceBuilder &device_builder, const StreamFormat &format,
                                                    size_t &raw_size_bytes, bool do_time_shifting, const Metavision::DeviceConfig &config) {
    std::shared_ptr<I_EventsStreamDecoder> decoder;
    std::shared_ptr<I_Decoder> frame_decoder;
    auto i_geometry = device_builder.add_facility(format.geometry());

    // Set of monitoring Subtypes that have more than one CONTINUED
    // Not supported by current decoders
    std::set<uint16_t> monitoring_id_blacklist = {
        0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x0009,
    };

    raw_size_bytes = 0;
    if (format.name() == "EVT4") {
        auto cd_decoder           = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());
        auto ext_trig_decoder     = device_builder.add_facility(std::make_unique<I_EventDecoder<EventExtTrigger>>());
        auto erc_count_ev_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventERCCounter>>());

        decoder = device_builder.add_facility(make_evt4_decoder(do_time_shifting, i_geometry->get_width(),
                                                                i_geometry->get_height(), cd_decoder, ext_trig_decoder,
                                                                erc_count_ev_decoder));

        raw_size_bytes = decoder->get_raw_event_size_bytes();
    } else if (format.name() == "EVT3") {
        auto cd_decoder           = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());
        auto ext_trig_decoder     = device_builder.add_facility(std::make_unique<I_EventDecoder<EventExtTrigger>>());
        auto erc_count_ev_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventERCCounter>>());
        auto monitoring_ev_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventMonitoring>>());

        decoder = device_builder.add_facility(
            make_evt3_decoder(do_time_shifting, i_geometry->get_height(), i_geometry->get_width(), cd_decoder,
                              ext_trig_decoder, erc_count_ev_decoder, monitoring_ev_decoder, monitoring_id_blacklist));

        raw_size_bytes = decoder->get_raw_event_size_bytes();
    } else if (format.name() == "EVT2") {
        auto cd_decoder       = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());
        auto ext_trig_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventExtTrigger>>());
        auto erc_count_ev_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventERCCounter>>());
        auto monitoring_ev_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventMonitoring>>());

        decoder = device_builder.add_facility(
            std::make_unique<EVT2Decoder>(do_time_shifting, cd_decoder, ext_trig_decoder, erc_count_ev_decoder,
                                          monitoring_ev_decoder, monitoring_id_blacklist));
        raw_size_bytes = decoder->get_raw_event_size_bytes();
    } else if (format.name() == "EVT21") {
            
        auto ext_trig_decoder     = device_builder.add_facility(std::make_unique<I_EventDecoder<EventExtTrigger>>());
        auto erc_count_ev_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventERCCounter>>());
        auto monitoring_ev_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventMonitoring>>());
        auto endianness = format["endianness"];

        auto evt21_keep_vectors = config.get<bool>("evt21_keep_vectors");

        if (evt21_keep_vectors){
            if (endianness != "little"){
                throw std::invalid_argument("Value for option 'endianness` not supported when option `evt21_keep_vectors=true` is set"); 
            }

            auto cd_vector_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCDVector>>());

            decoder = device_builder.add_facility(std::make_unique<EVT21VectorizedDecoder>(
                do_time_shifting, cd_vector_decoder, ext_trig_decoder, erc_count_ev_decoder, monitoring_ev_decoder,
                monitoring_id_blacklist));

        }else{
            auto cd_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());

            if (endianness == "legacy") {
                decoder = device_builder.add_facility(std::make_unique<EVT21LegacyDecoder>(
                    do_time_shifting, cd_decoder, ext_trig_decoder, erc_count_ev_decoder, monitoring_ev_decoder,
                    monitoring_id_blacklist));
            } else {
                decoder = device_builder.add_facility(
                    std::make_unique<EVT21Decoder>(do_time_shifting, cd_decoder, ext_trig_decoder, erc_count_ev_decoder,
                                                   monitoring_ev_decoder, monitoring_id_blacklist));
            }
        }
        raw_size_bytes = decoder->get_raw_event_size_bytes();

    } else if (format.name() == "HISTO3D") {
        auto pixel_layout = get_pixel_layout(format["pixellayout"]);
        int pixel_bytes;
        try {
            pixel_bytes = std::stoi(format["pixelbytes"]);
            if (pixel_bytes <= 0) {
                throw std::invalid_argument("");
            }
        } catch (...) {
            // stoi throws non standard exceptions on Android
            throw std::invalid_argument("Format is missing a valid pixelbytes value");
        }
        frame_decoder = device_builder.add_facility(
            std::make_unique<Histo3dDecoder>(i_geometry->get_height(), i_geometry->get_width(), pixel_layout.first,
                                             pixel_layout.second, pixel_bytes == 2));
        raw_size_bytes = frame_decoder->get_raw_event_size_bytes();
    } else if (format.name() == "DIFF3D") {
        auto pixel_layout = get_pixel_layout(format["pixellayout"]);
        frame_decoder     = device_builder.add_facility(
            std::make_unique<Diff3dDecoder>(i_geometry->get_height(), i_geometry->get_width(), pixel_layout.first));
        raw_size_bytes = frame_decoder->get_raw_event_size_bytes();
    } else if (format.name() == "AER-8b") {
        auto cd_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());
        decoder        = device_builder.add_facility(std::make_unique<AERDecoder<false>>(do_time_shifting, cd_decoder));
        raw_size_bytes = decoder->get_raw_event_size_bytes();
    } else if (format.name() == "AER-4b") {
        auto cd_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());
        decoder         = device_builder.add_facility(std::make_unique<AERDecoder<true>>(do_time_shifting, cd_decoder));
        raw_size_bytes  = decoder->get_raw_event_size_bytes();
    } else if (format.name().rfind("MTR") == 0) {
        auto mtr_decoder = mtr_decoder_from_format(*i_geometry, format.name());
        if (mtr_decoder) {
            frame_decoder  = device_builder.add_facility(std::move(mtr_decoder));
            raw_size_bytes = frame_decoder->get_raw_event_size_bytes();
        }
    }

    if (!decoder && !frame_decoder) {
        throw std::runtime_error("Format " + format.name() + " is not supported");
    }

    return decoder;
}

} // namespace Metavision
