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
#include <metavision/hal/utils/data_transfer.h>
#include <metavision/hal/utils/device_builder.h>

#include "sample_camera_discovery.h"
#include "sample_camera_synchronization.h"
#include "sample_data_transfer.h"
#include "sample_decoder.h"
#include "sample_device_control.h"
#include "sample_geometry.h"
#include "sample_hw_identification.h"

Metavision::CameraDiscovery::SerialList SampleCameraDiscovery::list() {
    SerialList ret;
    ret.push_back(SampleHWIdentification::SAMPLE_SERIAL);
    return ret;
}

Metavision::CameraDiscovery::SystemList SampleCameraDiscovery::list_available_sources() {
    SystemList systems;

    Metavision::PluginCameraDescription description;
    description.serial_     = SampleHWIdentification::SAMPLE_SERIAL;
    description.system_id_  = SampleHWIdentification::SAMPLE_SYSTEM_ID;
    description.connection_ = Metavision::USB_LINK;

    systems.push_back(description);
    return systems;
}

bool SampleCameraDiscovery::discover(Metavision::DeviceBuilder &device_builder, const std::string &serial,
                                     const Metavision::DeviceConfig &config) {
    if (!(serial.empty() || serial == SampleHWIdentification::SAMPLE_SERIAL)) {
        return false;
    }

    // Add facilities to the device builder
    auto hw_identification = device_builder.add_facility(
        std::make_unique<SampleHWIdentification>(device_builder.get_plugin_software_info(), "USB"));
    device_builder.add_facility(std::make_unique<SampleGeometry>());
    device_builder.add_facility(std::make_unique<SampleCameraSynchronization>());

    auto cd_event_decoder =
        device_builder.add_facility(std::make_unique<Metavision::I_EventDecoder<Metavision::EventCD>>());
    auto decoder = device_builder.add_facility(std::make_unique<SampleDecoder>(false, cd_event_decoder));
    device_builder.add_facility(std::make_unique<Metavision::I_EventsStream>(
        std::make_unique<SampleDataTransfer>(decoder->get_raw_event_size_bytes()), hw_identification, decoder,
        std::make_shared<SampleDeviceControl>()));

    return true;
}

bool SampleCameraDiscovery::is_for_local_camera() const {
    return true;
}
