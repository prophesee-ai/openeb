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

#ifndef METAVISION_HAL_SAMPLE_FILE_DISCOVERY_H
#define METAVISION_HAL_SAMPLE_FILE_DISCOVERY_H

#include <metavision/sdk/base/utils/generic_header.h>
#include <metavision/hal/utils/file_discovery.h>

/// @brief Discovers devices from RAW files
///
/// This class is the implementation of HAL's class @ref Metavision::FileDiscovery
class SampleFileDiscovery : public Metavision::FileDiscovery {
public:
    bool discover(Metavision::DeviceBuilder &device_builder, std::unique_ptr<std::istream> &stream,
                  const Metavision::RawFileHeader &header, const Metavision::RawFileConfig &config) override;
};

#endif // METAVISION_HAL_SAMPLE_FILE_DISCOVERY_H
