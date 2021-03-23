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

#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/raw_file_header.h"

namespace Metavision {

namespace {
static const std::string integrator_key  = "integrator_name";
static const std::string plugin_name_key = "plugin_name";
} // namespace

RawFileHeader::RawFileHeader() = default;
RawFileHeader::RawFileHeader(std::istream &stream) : GenericHeader(stream) {}
RawFileHeader::RawFileHeader(const HeaderMap &header) : GenericHeader(header) {}

std::string RawFileHeader::get_integrator_name() const {
    return get_field(integrator_key);
}

void RawFileHeader::set_integrator_name(const std::string &integrator_name) {
    set_field(integrator_key, integrator_name);
}

std::string RawFileHeader::get_plugin_name() const {
    return get_field(plugin_name_key);
}

void RawFileHeader::set_plugin_name(const std::string &plugin_name) {
    return set_field(plugin_name_key, plugin_name);
}

void RawFileHeader::remove_plugin_name() {
    return remove_field(plugin_name_key);
}

} // namespace Metavision
