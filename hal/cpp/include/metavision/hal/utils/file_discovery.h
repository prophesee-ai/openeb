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

#ifndef METAVISION_HAL_FILE_DISCOVERY_H
#define METAVISION_HAL_FILE_DISCOVERY_H

#include <istream>
#include <memory>

#include "metavision/hal/utils/raw_file_config.h"
#include "metavision/hal/utils/raw_file_header.h"

namespace Metavision {

class DeviceBuilder;
class Plugin;

/// @brief Creates device simulating a camera streaming the raw events coming from a stream
class FileDiscovery {
public:
    /// @brief Destructor
    virtual ~FileDiscovery();

    /// @brief Gets the file discovery name
    /// @return Name of the file discovery
    virtual std::string get_name() const;

    /// @brief Discovers a device and initializes a corresponding @ref DeviceBuilder
    ///
    /// The input stream is passed using a reference to a (uniquely) owned pointer, so that an implementation can take
    /// ownership of it upon successful opening. Conversely, if the stream can not be opened by the file discovery, the
    /// implementation must NOT take ownership of the stream pointer.
    ///
    /// @warning If the file discovery fails to open from the stream (i.e if the implementation returns a nullptr), the
    /// stream pointer must NOT have been moved from. An exception will be thrown if the stream pointer is null and a
    /// device could not be created.
    ///
    /// @param device_builder Device builder to configure so that it can build a @ref Device from the parameters
    /// @param stream The stream to read events from
    /// @param header Header of the input stream, containing identification information for the stream's source
    /// @param config For building the camera from a file
    /// @return true if a device builder could be discovered from the parameters
    virtual bool discover(DeviceBuilder &device_builder, std::unique_ptr<std::istream> &stream,
                          const RawFileHeader &header, const RawFileConfig &config) = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_FILE_DISCOVERY_H
