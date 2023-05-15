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

#include "metavision/sdk/version.h"

namespace Metavision {

/// @brief Stores information about the version of the software
struct SoftwareInfo {
    /// @brief Supported version suffixes
    enum class VersionSuffix { NONE = 0, DEV = 1 };

    /// @brief Constructor
    ///
    /// @param version_major Major version number
    /// @param version_minor Minor version number
    /// @param version_patch Patch version number
    /// @param version_suffix_string Version suffix string
    /// @param vcs_branch VCS branch name
    /// @param vcs_commit VCS commit's hash
    /// @param vcs_date VCS commit's date
    SoftwareInfo(int version_major, int version_minor, int version_patch, const std::string &version_suffix_string,
                 const std::string &vcs_branch, const std::string &vcs_commit, const std::string &vcs_date);

    /// @brief Returns major version number
    int get_version_major() const;

    /// @brief Returns minor version number
    int get_version_minor() const;

    /// @brief Returns patch version number
    int get_version_patch() const;

    /// @brief Returns version suffix string
    std::string get_version_suffix() const;

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

    /// Version suffix
    int version_suffix_type_;

    /// VCS branch
    std::string vcs_branch_;

    /// VCS commit
    std::string vcs_commit_;

    /// VCS commit's date
    std::string vcs_date_;
};

/// @brief Returns software information about the Metavision SDK used at build time
static inline SoftwareInfo &get_build_software_info() {
    static Metavision::SoftwareInfo build_sdk_info(METAVISION_SDK_VERSION_MAJOR, METAVISION_SDK_VERSION_MINOR,
                                                   METAVISION_SDK_VERSION_PATCH, METAVISION_SDK_VERSION_SUFFIX,
                                                   METAVISION_SDK_GIT_BRANCH_RAW, METAVISION_SDK_GIT_HASH_RAW,
                                                   METAVISION_SDK_GIT_COMMIT_DATE);
    return build_sdk_info;
}

/// @brief Returns software information about the Metavision SDK used at run time.
/// @note The information values will differ from what get_build_software_info returns if the user
/// software was built using a different version of Metavision SDK than the one available on the system running
/// the software.
SoftwareInfo &get_metavision_software_info();

} // namespace Metavision

#endif // METAVISION_SDK_BASE_SOFTWARE_INFO_H
