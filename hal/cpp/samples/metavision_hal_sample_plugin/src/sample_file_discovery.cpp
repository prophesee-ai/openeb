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
#include <metavision/hal/device/device.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/facilities/future/i_events_stream.h>
#include <metavision/hal/utils/device_builder.h>
#include <metavision/hal/utils/file_data_transfer.h>
#include <metavision/hal/utils/future/file_data_transfer.h>

#include "sample_file_discovery.h"
#include "sample_hw_identification.h"
#include "sample_geometry.h"
#include "sample_decoder.h"
#include "future/sample_decoder.h"

bool SampleFileDiscovery::discover(Metavision::DeviceBuilder &device_builder, std::unique_ptr<std::istream> &stream,
                                   const Metavision::RawFileHeader &header,
                                   const Metavision::RawFileConfig &stream_config) {
    // Add facilities to the device
    auto hw_identification = device_builder.add_facility(
        std::make_unique<SampleHWIdentification>(device_builder.get_plugin_software_info(), "File"));
    device_builder.add_facility(std::make_unique<SampleGeometry>());

    auto cd_event_decoder =
        device_builder.add_facility(std::make_unique<Metavision::I_EventDecoder<Metavision::EventCD>>());

    // The new Future::I_EventsStream and Future::I_Decoder facilities are preview versions of the facilities
    // as they will be available in the next major release of Metavision.
    // Those facilities provide new features for offline recording such as seeking at a random position in the stream.
    // They must be instantiated in the plugin to be useable by an application.
    //
    // However, to keep compatibility with previous version of Metavision software, it is mandatory to also
    // instantiate the current version of the I_EventsStream and I_Decoder.
    // When the future facilities are made available in the next major release, it will not be necessary
    // to instantiate both versions of the decoding and streaming facilities.
    {
        auto decoder = device_builder.add_facility(
            std::make_unique<Future::SampleDecoder>(stream_config.do_time_shifting_, cd_event_decoder));
        // Note that the future streaming facility does not take ownership of the stream instance
        // This is ok as we specifically make sure the stream lifetime outlasts the facility's.
        device_builder.add_facility(std::make_unique<Metavision::Future::I_EventsStream>(
            std::make_unique<Metavision::Future::FileDataTransfer>(stream.get(), decoder->get_raw_event_size_bytes(),
                                                                   stream_config),
            hw_identification, decoder));
    }
    {
        auto decoder = device_builder.add_facility(
            std::make_unique<SampleDecoder>(stream_config.do_time_shifting_, cd_event_decoder));
        // Note that the current facility must take ownership of the stream instance (as it was the case in previous
        // versions).
        device_builder.add_facility(std::make_unique<Metavision::I_EventsStream>(
            std::make_unique<Metavision::FileDataTransfer>(std::move(stream), decoder->get_raw_event_size_bytes(),
                                                           stream_config),
            hw_identification));
    }

    return true;
}
