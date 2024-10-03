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

#ifndef METAVISION_UTILS_GTEST_TEMPORARY_DIRECTORY_HANDLER_H
#define METAVISION_UTILS_GTEST_TEMPORARY_DIRECTORY_HANDLER_H

#include <filesystem>
#include <string>

namespace Metavision {

class TemporaryDirectoryHandler {
public:
    /// @brief Constructor
    ///
    /// Creates an TemporaryDirectoryHandler class instance.
    ///
    /// Throws std::runtime_error if could not create temporary directory
    TemporaryDirectoryHandler(const std::string &dirbasename = "");

    /// A temporary directory handler can not be copied
    TemporaryDirectoryHandler(const TemporaryDirectoryHandler &) = delete;

    /// A temporary directory handler can not be copied
    TemporaryDirectoryHandler operator=(const TemporaryDirectoryHandler &) = delete;

    /// @brief Destructor
    ///
    /// Deletes an TemporaryDirectoryHandler class instance.
    ///
    /// Throws std::runtime_error if could not remove temporary directory
    ~TemporaryDirectoryHandler();

    /// @brief Returns full path of the (created) temporary directory
    std::string get_tmpdir_path() const;

    /// @brief Returns full path obtained by concatenating the path to the temporary directory and the given basename
    /// @note It does NOT create the file/directory returned
    std::string get_full_path(const std::string &basename) const;

    void disable_remove_on_destruction();

private:
    std::filesystem::path tmpdir_;
    bool remove_on_destruction_ = true;
};

} // namespace Metavision

#include "detail/temporary_directory_handler_impl.h"

#endif // METAVISION_UTILS_GTEST_TEMPORARY_DIRECTORY_HANDLER_H
