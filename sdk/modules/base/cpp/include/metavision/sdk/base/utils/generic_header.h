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

#ifndef METAVISION_SDK_BASE_GENERIC_HEADER_H
#define METAVISION_SDK_BASE_GENERIC_HEADER_H

#include <istream>
#include <string>
#include <map>

namespace Metavision {

/// @brief A utility class to hold and format headers information
///
/// A header is composed of fields composed of key (string) / value (string) pairs
class GenericHeader {
public:
    /// Alias for the internal map holding the header information
    using HeaderMap = std::map<std::string, std::string>;

    /// @brief Default constructor
    GenericHeader();

    /// @brief Builds the header map by parsing the input stream.
    ///
    /// The stream internal is expected to point to the start of the header (if any)
    /// This method effectively places the stream cursor at the end of header.
    /// The position of the cursor remains unchanged if no header is actually present.
    ///
    /// @param stream The input stream to parse a header from
    GenericHeader(std::istream &stream);

    /// @brief Builds the header map using a copy of the input @ref HeaderMap
    /// @param header_map The header map to use for initialization
    GenericHeader(const HeaderMap &header_map);

    /// @brief Returns if the header is empty
    /// @return True if the header is empty, false otherwise
    bool empty() const;

    /// @brief Adds the current date and time to the header (in the format Y-m-d H:M:S).
    /// @note The date is updated at each call to this method
    void add_date();

    /// @brief Removes the date if there was any in the header
    void remove_date();

    /// @brief Gets the date and time at which the associated file was recorded
    /// @return The current date and time in string format if it is found, or an empty string
    /// otherwise.
    std::string get_date() const;

    /// @brief Adds a new field in the header
    /// @param key The key of the field in the header
    /// @param value The value of the field
    void set_field(const std::string &key, const std::string &value);

    /// @brief Remove the input field (if exists)
    void remove_field(const std::string &key);

    /// @brief Gets the value associated to the input key.
    /// @return The value associated to the input key if it exists, or an empty string otherwise
    std::string get_field(const std::string &key) const;

    /// @brief Gets the @ref HeaderMap holding the header information
    /// @return The header map
    const HeaderMap &get_header_map() const;

    /// @brief Serializes the header map
    /// @return The header in a string format
    std::string to_string() const;

private:
    /// @brief Checks the prefix of the line and reads it if it belongs to the header
    ///
    /// If the line belongs to the header, the stream is modified
    ///
    /// @return true if the next line in the stream is part of the header
    bool check_prefix_and_read_header_line(std::istream &stream);

    /// @brief Parse the header from the input stream
    void parse_header(std::istream &stream);

    HeaderMap header_; ///< The header map
};

/// @brief Operator overload to write easily the header
std::ostream &operator<<(std::ostream &output, const GenericHeader &header);
} // namespace Metavision

#endif // METAVISION_SDK_BASE_GENERIC_HEADER_H
