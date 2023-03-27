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

#ifndef METAVISION_HAL_RAW_FILE_HEADER_H
#define METAVISION_HAL_RAW_FILE_HEADER_H

#include "metavision/sdk/base/utils/generic_header.h"

namespace Metavision {

/// @brief A dedicated class to handle RAW file headers and handle easily their mandatory fields
class RawFileHeader : public GenericHeader {
public:
    /// @brief Default constructor
    RawFileHeader();

    /// @brief Builds the header map by parsing the input stream.
    /// @param i Input stream to parse
    /// @ref GenericHeader(std::istream &)
    RawFileHeader(std::istream &i);

    /// @brief Builds the header map using a copy of the input @ref HeaderMap
    /// @param h Header map to copy from
    RawFileHeader(const HeaderMap &h);

    /// @brief Gets the integrator name of the source used to generate the RAW file.
    /// @return Returns the integrator name if any, or an empty string otherwise
    std::string get_camera_integrator_name() const;

    /// @brief Sets the name of the integrator of the source used to generate the RAW file
    /// @param integrator_name Name of the integrator
    void set_camera_integrator_name(const std::string &integrator_name);

    /// @brief Gets the integrator name of the plugin used to generate the RAW file.
    /// @return Returns the integrator name if any, or an empty string otherwise
    std::string get_plugin_integrator_name() const;

    /// @brief Sets the name of the integrator of the plugin used to generate the RAW file
    /// @param integrator_name Name of the integrator
    void set_plugin_integrator_name(const std::string &integrator_name);

    /// @brief Gets the name of the plugin to use to read the RAW file.
    /// @return Returns the name of the plugin if any, or an empty string otherwise
    std::string get_plugin_name() const;

    /// @brief Sets the name of the plugin to use to read the RAW file
    /// @param plugin_name Name of the plugin
    void set_plugin_name(const std::string &plugin_name);

    /// @brief Remove the name of the plugin to use to read the RAW file
    void remove_plugin_name();
};
} // namespace Metavision

#endif // METAVISION_HAL_RAW_FILE_HEADER_H
