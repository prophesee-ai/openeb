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

#ifndef RAW_CONSTANTS_H
#define RAW_CONSTANTS_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <metavision/hal/facilities/i_hw_identification.h>
#include "utils/device_system_id.h"

namespace Metavision {

enum EncodingEVT { ENCODING_EVT2, ENCODING_EVT3 };

struct PluginConfig {
    Metavision::I_HW_Identification::SensorInfo sensor_info_;
    int sensor_width_;
    int sensor_height_;
    std::unordered_set<EncodingEVT> encodings_;
    std::unordered_set<SystemId> compatible_ids_;
};

using PluginsMap  = std::unordered_map<std::string, PluginConfig>;
using SystemIdMap = std::unordered_map<SystemId, std::string>;

// clang-format off
static const PluginsMap plugins_map = {
    {"hal_plugin_gen3_fx3",   {{3, 0}, 640, 480, {ENCODING_EVT2},
                               {SYSTEM_CCAM3_GEN3,
                                SYSTEM_CCAM4_GEN3,
                                SYSTEM_CCAM4_GEN3_REV_B,
                                SYSTEM_CCAM4_GEN3_EVK,
                                SYSTEM_CCAM4_GEN3_REV_B_EVK,
                                SYSTEM_CCAM4_GEN3_REV_B_EVK_BRIDGE,
                                SYSTEM_VISIONCAM_GEN3,
                                SYSTEM_VISIONCAM_GEN3_EVK}}},
    {"hal_plugin_gen31_fx3",  {{3, 1}, 640, 480, {ENCODING_EVT2}, 
                               {SYSTEM_CCAM3_GEN31,
                                SYSTEM_VISIONCAM_GEN31,
                                SYSTEM_VISIONCAM_GEN31_EVK}}},
    {"hal_plugin_gen31_evk2", {{3, 1}, 640, 480, {ENCODING_EVT2},
                               {SYSTEM_EVK2_GEN31}}},
    {"hal_plugin_gen31_evk3", {{3, 1}, 640, 480, {ENCODING_EVT2, ENCODING_EVT3},
                               {SYSTEM_EVK3_GEN31_EVT2, SYSTEM_EVK3_GEN31_EVT3}}},
    {"hal_plugin_gen4_fx3",   {{4, 0}, 1280, 720, {ENCODING_EVT2, ENCODING_EVT3},
                               {SYSTEM_CCAM3_GEN4}}},
    {"hal_plugin_gen4_evk2",  {{4, 0}, 1280, 720, {ENCODING_EVT2, ENCODING_EVT3},
                               {SYSTEM_EVK2_GEN4}}},
    {"hal_plugin_gen41_evk2", {{4, 1}, 1280, 720, {ENCODING_EVT2, ENCODING_EVT3},
                               {SYSTEM_EVK2_GEN41}}},
    {"hal_plugin_gen41_evk3", {{4, 1}, 1280, 720, {ENCODING_EVT2, ENCODING_EVT3},
                               {SYSTEM_EVK3_GEN41}}}
};
// clang-format on

namespace {

inline SystemIdMap reverse_plugins_map(PluginsMap pmap) {
    SystemIdMap res_map;
    for (auto &pmap_key : pmap) {
        for (auto &system_id : pmap_key.second.compatible_ids_) {
            res_map.insert({system_id, pmap_key.first});
        }
    }
    return res_map;
}

} // anonymous namespace

static const SystemIdMap system_id_map = reverse_plugins_map(plugins_map);

static const std::string raw_key_serial_number = "serial_number";
static const std::string raw_key_system_id     = "system_ID";
static const std::string raw_key_width         = "width";
static const std::string raw_key_height        = "height";
static const std::string raw_key_evt           = "evt";
static const std::string raw_evt_version_2     = "2.0";
static const std::string raw_evt_version_3     = "3.0";

static const std::string raw_default_integrator      = "Prophesee";
static const std::string raw_default_connection_type = "File";

} // namespace Metavision

#endif // RAW_CONSTANTS_H
