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

#ifndef METAVISION_SDK_BASE_SOFTWARE_INFO_H
#define METAVISION_SDK_BASE_SOFTWARE_INFO_H

#include <string>

namespace Metavision {

/// @brief Stores information about the version of the software
struct SoftwareInfo {
    /// @brief Constructor
    ///
    /// @param version_major Major version number
    /// @param version_minor Minor version number
    /// @param version_patch Patch version number
    /// @param version_dev Dev version number
    /// @param vcs_branch VCS branch name
    /// @param vcs_commit VCS commit's hash
    /// @param vcs_date VCS commit's date
    SoftwareInfo(int version_major, int version_minor, int version_patch, int version_dev,
                 const std::string &vcs_branch, const std::string &vcs_commit, const std::string &vcs_date);

    /// @brief Returns major version number
    int get_version_major() const;

    /// @brief Returns minor version number
    int get_version_minor() const;

    /// @brief Returns patch version number
    int get_version_patch() const;

    /// @brief Returns dev version number
    int get_version_dev() const;

    /// @brief Returns version as a string
    std::string get_version() const;

    /// @brief Returns version control software (vcs) branch
    std::string get_vcs_branch() const;

    /// @brief Returns version control software (vcs) commit
    std::string get_vcs_commit() const;

    /// @brief Returns version control software (vcs) commit's date
    std::string get_vcs_date() const;

private:
    /// Major version number
    int version_major_;

    /// Minor version number
    int version_minor_;

    /// Patch version number
    int version_patch_;

    /// Dev version number
    int version_dev_;

    /// VCS branch
    std::string vcs_branch_;

    /// VCS commit
    std::string vcs_commit_;

    /// VCS commit's date
    std::string vcs_date_;
};

/// @brief return Metavision software information
SoftwareInfo &get_metavision_software_info();

} // namespace Metavision

#endif // METAVISION_SDK_BASE_SOFTWARE_INFO_H
