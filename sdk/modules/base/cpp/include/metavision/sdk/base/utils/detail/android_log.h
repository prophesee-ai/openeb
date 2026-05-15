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

#ifndef METAVISION_SDK_BASE_DETAIL_ANDROID_LOG_H
#define METAVISION_SDK_BASE_DETAIL_ANDROID_LOG_H

#include <iostream>
#include <sstream>

namespace Metavision {
namespace detail {

/// @brief Streambuf class implementation for printing logs in Android systems
class android_streambuf : public std::streambuf {
public:
    /// @brief Constructor
    /// @param tag Tag name used by Android print log
    android_streambuf(const std::string tag);

    /// @brief Destructor
    ~android_streambuf() = default;

protected:
    /// @brief Writes characters from the array pointed to by s into the controlled output sequence, until either n
    /// characters have been written or the end of the output sequence is reached.
    /// @param s Pointer to the sequence of characters to be written.
    /// @param n Number of characters to write.
    /// @return The number of characters written.
    std::streamsize xsputn(const char *s, std::streamsize n) override;

    /// @brief Puts a character into the controlled output sequence without changing the current position.
    /// It is called by public member functions such as sputc to write a character when there are no writing positions
    /// available at the put pointer (pptr).
    /// @param ch Character to be put.
    /// @return In case of success, the character put is returned, converted to a value of type int_type using member
    /// traits_type::to_int_type. Otherwise, it returns the end-of-file value (EOF) either if called with this value as
    /// argument c or to signal a failure (some implementations may throw an exception instead).
    int overflow(int ch = EOF) override;

    /// @brief Synchronizes the contents in the buffer with those of the associated character sequence.
    /// @return Returns zero, which indicates success. A value of -1 would indicate failure.
    int sync() override;

private:
    std::ostringstream ostr_;
    const std::string tag_;
};

} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_BASE_DETAIL_ANDROID_LOG_H