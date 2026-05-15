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

#include "metavision/hal/plugin/plugin.h"
#include "metavision/hal/utils/camera_discovery.h"
#include "metavision/hal/utils/file_discovery.h"
#include "metavision/hal/utils/hal_error_code.h"
#include "metavision/hal/utils/hal_exception.h"

namespace Metavision {

Plugin::Plugin(const std::string &plugin_name) : plugin_name_(plugin_name), plugin_info_(nullptr), hal_info_(nullptr) {}

Plugin::~Plugin() = default;

const std::string &Plugin::get_plugin_name() const {
    return plugin_name_;
}

const std::string &Plugin::get_integrator_name() const {
    return integrator_name_;
}

void Plugin::set_integrator_name(const std::string &integrator_name) {
    integrator_name_ = integrator_name;
}

Plugin::CameraDiscoveryList::CameraDiscoveryList(Plugin &plugin) :
    camera_discovery_list_(plugin.camera_discovery_list_) {}

detail::iterator<CameraDiscovery> Plugin::CameraDiscoveryList::begin() {
    return detail::iterator<CameraDiscovery>(camera_discovery_list_.begin());
}

detail::iterator<CameraDiscovery> Plugin::CameraDiscoveryList::end() {
    return detail::iterator<CameraDiscovery>(camera_discovery_list_.end());
}

size_t Plugin::CameraDiscoveryList::size() const {
    return camera_discovery_list_.size();
}

bool Plugin::CameraDiscoveryList::empty() const {
    return camera_discovery_list_.empty();
}

Plugin::CameraDiscoveryList Plugin::get_camera_discovery_list() {
    return Plugin::CameraDiscoveryList(*this);
}

Plugin::FileDiscoveryList::FileDiscoveryList(Plugin &plugin) : file_discovery_list_(plugin.file_discovery_list_) {}

detail::iterator<FileDiscovery> Plugin::FileDiscoveryList::begin() {
    return detail::iterator<FileDiscovery>(file_discovery_list_.begin());
}

detail::iterator<FileDiscovery> Plugin::FileDiscoveryList::end() {
    return detail::iterator<FileDiscovery>(file_discovery_list_.end());
}

size_t Plugin::FileDiscoveryList::size() const {
    return file_discovery_list_.size();
}

bool Plugin::FileDiscoveryList::empty() const {
    return file_discovery_list_.empty();
}

Plugin::FileDiscoveryList Plugin::get_file_discovery_list() {
    return Plugin::FileDiscoveryList(*this);
}

FileDiscovery &Plugin::add_file_discovery_priv(std::unique_ptr<FileDiscovery> idb) {
    file_discovery_list_.push_back(std::move(idb));
    return *file_discovery_list_.back();
}

CameraDiscovery &Plugin::add_camera_discovery_priv(std::unique_ptr<CameraDiscovery> idb) {
    camera_discovery_list_.push_back(std::move(idb));
    return *camera_discovery_list_.back();
}

const SoftwareInfo &Plugin::get_plugin_info() const {
    if (!plugin_info_) {
        throw HalException(HalErrorCode::FailedInitialization, "Plugin information for loaded plugin is null.");
    }
    return *plugin_info_;
}

void Plugin::set_plugin_info(const SoftwareInfo &info) {
    plugin_info_ = std::unique_ptr<SoftwareInfo>(new SoftwareInfo(info));
}

const SoftwareInfo &Plugin::get_hal_info() const {
    if (!hal_info_) {
        throw HalException(HalErrorCode::FailedInitialization, "HAL information for loaded plugin is null.");
    }
    return *hal_info_;
}

void Plugin::set_hal_info(const SoftwareInfo &info) {
    hal_info_ = std::unique_ptr<SoftwareInfo>(new SoftwareInfo(info));
}

} // namespace Metavision
