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

#ifndef METAVISION_SDK_BASE_DAT_HELPER_H
#define METAVISION_SDK_BASE_DAT_HELPER_H

#include <map>
#include <ctime>
#include <string>
#include <sstream>
#include <iostream>
#include "metavision/sdk/base/utils/generic_header.h"
#include "metavision/sdk/base/events/detail/event_traits.h"

namespace Metavision {

/// @brief Writes the DAT header of a file
/// @tparam EventType The type of events stored in the DAT file
/// @param os The stream used to store the DAT data
/// @param header_map An optional map of data to be stored in the header
template<typename EventType>
void write_DAT_header(std::ostream &os, const GenericHeader::HeaderMap &header_map) {
    GenericHeader header_to_write(header_map);
    header_to_write.set_field("Version", "2");
    header_to_write.add_date();
    os << header_to_write;

    // Write type and size (first line after the header)
    unsigned char data[] = {get_event_id<EventType>(), get_event_size<EventType>()};
    os.write((const char *)data, 2);
}

/// @brief Convenience function to get the DAT header as string
/// @tparam EventType The type of events stored in the DAT file
/// @param header_map Additional information to be stored in the header
/// @return The string representing the DAT header
template<typename EventType>
std::string get_DAT_header_as_string(const GenericHeader::HeaderMap &header_map = GenericHeader::HeaderMap()) {
    std::ostringstream oss;
    write_DAT_header<EventType>(oss, header_map);
    return oss.str();
}

/// @brief Convenience function to create a DAT header map to be added in the header using the camera geometry
/// @param width Width of the camera sensor
/// @param height Height of the camera sensor
/// @return The DAT header map
inline GenericHeader make_DAT_header_map_with_geometry(int width, int height) {
    GenericHeader header;
    header.set_field("Width", std::to_string(width));
    header.set_field("Height", std::to_string(height));
    return header;
}

} // namespace Metavision

#endif // METAVISION_SDK_BASE_DAT_HELPER_H
