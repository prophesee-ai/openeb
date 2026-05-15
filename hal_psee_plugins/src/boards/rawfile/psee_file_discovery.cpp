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
#include "metavision/psee_hw_layer/boards/rawfile/psee_raw_file_header.h"
#include "metavision/psee_hw_layer/boards/rawfile/file_hw_identification.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/file_raw_data_producer.h"
#include "metavision/hal/utils/raw_file_config.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_erc_counter.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "utils/make_decoder.h"

namespace Metavision {

bool PseeFileDiscovery::discover(DeviceBuilder &device_builder, std::unique_ptr<std::istream> &stream,
                                 const RawFileHeader &header, const RawFileConfig &file_config) {
    try {
        size_t raw_size_bytes = 0;
        PseeRawFileHeader psee_header(header);
        StreamFormat format = psee_header.get_format();

        auto decoder = make_decoder(device_builder, format, raw_size_bytes, file_config.do_time_shifting_);

        auto file_hw_id = device_builder.add_facility(
            std::make_unique<FileHWIdentification>(device_builder.get_plugin_software_info(), psee_header));

        device_builder.add_facility(std::make_unique<I_EventsStream>(
            std::make_unique<FileRawDataProducer>(std::move(stream), raw_size_bytes, file_config), file_hw_id,
            decoder));
        return true;
    } catch (std::exception &e) {
        MV_HAL_LOG_TRACE() << "Could not read file:" << e.what();
        return false;
    }
}

} // namespace Metavision
