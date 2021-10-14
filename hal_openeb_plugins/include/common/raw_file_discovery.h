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

#ifndef RAW_FILE_DISCOVERY_H
#define RAW_FILE_DISCOVERY_H

#include <metavision/hal/utils/file_discovery.h>

namespace Metavision {

class RawGeometry;

/// @brief Discovers devices from RAW files
///
/// This class is the implementation of HAL's class @ref Metavision::FileDiscovery
class RawFileDiscovery : public Metavision::FileDiscovery {
public:
    /// @brief Discovers a device and initializes a corresponding @ref DeviceBuilder
    /// @param device_builder Device builder to configure so that it can build a @ref Device from the parameters
    /// @param stream The stream to read events from
    /// @param header Header of the input stream, containing identification information for the stream's source
    /// @param config For building the camera from a file
    /// @return true if a device builder could be discovered from the parameters
    bool discover(Metavision::DeviceBuilder &device_builder, std::unique_ptr<std::istream> &stream,
                  const Metavision::RawFileHeader &header, const Metavision::RawFileConfig &config) override;
};

} // namespace Metavision

#endif // RAW_FILE_DISCOVERY_H