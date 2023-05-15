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

#include <stdexcept>
#include "metavision/sdk/base/utils/software_info.h"

namespace Metavision {

namespace detail {

int version_suffix_string_to_type(const std::string &version_suffix_string) {
    if (version_suffix_string == "")
        return (int)SoftwareInfo::VersionSuffix::NONE;
    else if (version_suffix_string == "dev")
        return (int)SoftwareInfo::VersionSuffix::DEV;
    return (int)SoftwareInfo::VersionSuffix::NONE;
}

std::string version_suffix_type_to_string(int version_suffix_type) {
    switch (version_suffix_type) {
    case (int)SoftwareInfo::VersionSuffix::DEV:
        return std::string("dev");
    default:
    case (int)SoftwareInfo::VersionSuffix::NONE:
        return std::string("");
    }
}

} // namespace detail

SoftwareInfo::SoftwareInfo(int version_major, int version_minor, int version_patch,
                           const std::string &version_suffix_string, const std::string &vcs_branch,
                           const std::string &vcs_commit, const std::string &vcs_date) :
    version_major_(version_major),
    version_minor_(version_minor),
    version_patch_(version_patch),
    version_suffix_type_(detail::version_suffix_string_to_type(version_suffix_string)),
    vcs_branch_(vcs_branch),
    vcs_commit_(vcs_commit),
    vcs_date_(vcs_date) {}

int SoftwareInfo::get_version_major() const {
    return version_major_;
}

int SoftwareInfo::get_version_minor() const {
    return version_minor_;
}

int SoftwareInfo::get_version_patch() const {
    return version_patch_;
}

std::string SoftwareInfo::get_version_suffix() const {
    return detail::version_suffix_type_to_string(version_suffix_type_);
}

std::string SoftwareInfo::get_version() const {
    return std::to_string(get_version_major()) + "." + std::to_string(get_version_minor()) + "." +
           std::to_string(get_version_patch()) +
           (version_suffix_type_ > 0 ? std::string("-") + detail::version_suffix_type_to_string(version_suffix_type_) :
                                       std::string(""));
}

std::string SoftwareInfo::get_vcs_branch() const {
    return vcs_branch_;
}

std::string SoftwareInfo::get_vcs_commit() const {
    return vcs_commit_;
}

std::string SoftwareInfo::get_vcs_date() const {
    return vcs_date_;
}

SoftwareInfo &get_metavision_software_info() {
    return get_build_software_info();
}
} // namespace Metavision
