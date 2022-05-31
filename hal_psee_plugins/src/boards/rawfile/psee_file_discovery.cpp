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

#include <memory>
#include <string>

#include "boards/rawfile/psee_file_discovery.h"
#include "decoders/evt2/evt2_decoder.h"
#include "decoders/evt2/future/evt2_decoder.h"
#include "decoders/evt3/evt3_decoder.h"
#include "decoders/evt3/future/evt3_decoder.h"
#include "boards/rawfile/psee_raw_file_header.h"
#include "boards/rawfile/file_hw_identification.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/future/i_events_stream.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/file_data_transfer.h"
#include "metavision/hal/utils/future/file_data_transfer.h"
#include "metavision/hal/utils/raw_file_config.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

bool PseeFileDiscovery::discover(DeviceBuilder &device_builder, std::unique_ptr<std::istream> &stream,
                                 const RawFileHeader &header, const RawFileConfig &file_config) {
    try {
        PseeRawFileHeader psee_header(header);
        std::unique_ptr<I_Geometry> geometry = psee_header.get_geometry();
        std::string format                   = psee_header.get_format();

        // header ill-formed => can't handle this file
        if (!geometry) {
            return false;
        }

        auto file_hw_id = device_builder.add_facility(
            std::make_unique<FileHWIdentification>(device_builder.get_plugin_software_info(), psee_header));

        auto i_geometry       = device_builder.add_facility(std::move(geometry));
        auto cd_decoder       = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());
        auto ext_trig_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventExtTrigger>>());

        // TODO MV-166: remove block creating Future facilities
        {
            std::shared_ptr<Future::I_Decoder> decoder;
            if (format == "EVT3" && i_geometry) {
                decoder = device_builder.add_facility(std::make_unique<Future::EVT3Decoder>(
                    file_config.do_time_shifting_, i_geometry->get_height(), cd_decoder, ext_trig_decoder));
            } else if (format == "EVT2") {
                decoder = device_builder.add_facility(
                    std::make_unique<Future::EVT2Decoder>(file_config.do_time_shifting_, cd_decoder, ext_trig_decoder));
            } else {
                return false;
            }
            device_builder.add_facility(std::make_unique<Future::I_EventsStream>(
                std::make_unique<Future::FileDataTransfer>(stream.get(), decoder->get_raw_event_size_bytes(),
                                                           file_config),
                file_hw_id, decoder));
        }
        {
            std::shared_ptr<I_Decoder> decoder;
            if (format == "EVT3" && i_geometry) {
                decoder = device_builder.add_facility(std::make_unique<EVT3Decoder>(
                    file_config.do_time_shifting_, i_geometry->get_height(), cd_decoder, ext_trig_decoder));
            } else if (format == "EVT2") {
                decoder = device_builder.add_facility(
                    std::make_unique<EVT2Decoder>(file_config.do_time_shifting_, cd_decoder, ext_trig_decoder));
            } else {
                return false;
            }
            device_builder.add_facility(std::make_unique<I_EventsStream>(
                std::make_unique<FileDataTransfer>(std::move(stream), decoder->get_raw_event_size_bytes(), file_config),
                file_hw_id));
        }
        return true;
    } catch (std::exception &e) {
        MV_HAL_LOG_TRACE() << "Could not read file:" << e.what();
        return false;
    }
}

} // namespace Metavision
