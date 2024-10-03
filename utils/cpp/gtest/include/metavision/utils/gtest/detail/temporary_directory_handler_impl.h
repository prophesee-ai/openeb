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

#ifndef METAVISION_UTILS_GTEST_DETAIL_TEMPORARY_DIRECTORY_HANDLER_IMPL_H
#define METAVISION_UTILS_GTEST_DETAIL_TEMPORARY_DIRECTORY_HANDLER_IMPL_H

#include <filesystem>
#include <iostream>

namespace Metavision {

inline TemporaryDirectoryHandler::TemporaryDirectoryHandler(const std::string &dirbasename) {
    // Create the tmp directory
    std::string dir_name = dirbasename.empty() ? "tmp_dir" : dirbasename;

    int counter = 1;
    do {
        tmpdir_ =
            std::filesystem::temp_directory_path() / std::filesystem::path(dir_name + std::to_string(counter++));
    } while (std::filesystem::exists(tmpdir_));

    if (!std::filesystem::create_directory(tmpdir_)) {
        throw std::runtime_error("Could not create temporary directory " + tmpdir_.string());
    }
}

inline TemporaryDirectoryHandler::~TemporaryDirectoryHandler() {
    if (remove_on_destruction_ && std::filesystem::remove_all(tmpdir_) == 0) {
        // one reason can be the directory was deleted manually by the user while program is running
        std::cerr << "Could not delete temporary directory" << tmpdir_ << std::endl;
    }
}

inline std::string TemporaryDirectoryHandler::get_tmpdir_path() const {
    return std::filesystem::canonical(tmpdir_).make_preferred().string();
}

inline std::string TemporaryDirectoryHandler::get_full_path(const std::string &file_basename) const {
    return (tmpdir_ / std::filesystem::path(file_basename)).string();
}

inline void TemporaryDirectoryHandler::disable_remove_on_destruction() {
    remove_on_destruction_ = false;
}

} // namespace Metavision

#endif // METAVISION_UTILS_GTEST_DETAIL_TEMPORARY_DIRECTORY_HANDLER_IMPL_H
