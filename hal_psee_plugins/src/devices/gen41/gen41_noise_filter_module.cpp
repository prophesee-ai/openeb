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

#include <map>
#include <cmath>

#include "devices/gen41/gen41_noise_filter_module.h"
#include "utils/register_map.h"

namespace Metavision {

namespace {
std::map<const int, std::map<const std::string, const int>> stc_threshold_params = {
    {1, {{"presc", 12}, {"mult", 15}, {"Reserved_D0C0_23_12", 90}}},
    {2, {{"presc", 10}, {"mult", 3}, {"Reserved_D0C0_23_12", 90}}},
    {3, {{"presc", 11}, {"mult", 5}, {"Reserved_D0C0_23_12", 95}}},
    {4, {{"presc", 9}, {"mult", 1}, {"Reserved_D0C0_23_12", 102}}},
    {5, {{"presc", 13}, {"mult", 15}, {"Reserved_D0C0_23_12", 90}}},
    {6, {{"presc", 11}, {"mult", 3}, {"Reserved_D0C0_23_12", 114}}},
    {7, {{"presc", 11}, {"mult", 3}, {"Reserved_D0C0_23_12", 90}}},
    {8, {{"presc", 12}, {"mult", 5}, {"Reserved_D0C0_23_12", 109}}},
    {9, {{"presc", 13}, {"mult", 9}, {"Reserved_D0C0_23_12", 122}}},
    {10, {{"presc", 13}, {"mult", 9}, {"Reserved_D0C0_23_12", 90}}},
    {11, {{"presc", 10}, {"mult", 1}, {"Reserved_D0C0_23_12", 102}}},
    {12, {{"presc", 14}, {"mult", 15}, {"Reserved_D0C0_23_12", 109}}},
    {13, {{"presc", 13}, {"mult", 7}, {"Reserved_D0C0_23_12", 117}}},
    {14, {{"presc", 14}, {"mult", 13}, {"Reserved_D0C0_23_12", 127}}},
    {15, {{"presc", 12}, {"mult", 3}, {"Reserved_D0C0_23_12", 138}}},
    {16, {{"presc", 12}, {"mult", 3}, {"Reserved_D0C0_23_12", 90}}},
    {17, {{"presc", 12}, {"mult", 3}, {"Reserved_D0C0_23_12", 90}}},
    {18, {{"presc", 14}, {"mult", 11}, {"Reserved_D0C0_23_12", 99}}},
    {19, {{"presc", 13}, {"mult", 5}, {"Reserved_D0C0_23_12", 109}}},
    {20, {{"presc", 13}, {"mult", 5}, {"Reserved_D0C0_23_12", 109}}},
    {21, {{"presc", 14}, {"mult", 9}, {"Reserved_D0C0_23_12", 122}}},
    {22, {{"presc", 14}, {"mult", 9}, {"Reserved_D0C0_23_12", 122}}},
    {23, {{"presc", 11}, {"mult", 1}, {"Reserved_D0C0_23_12", 209}}},
    {24, {{"presc", 11}, {"mult", 1}, {"Reserved_D0C0_23_12", 138}}},
    {25, {{"presc", 11}, {"mult", 1}, {"Reserved_D0C0_23_12", 138}}},
    {26, {{"presc", 15}, {"mult", 15}, {"Reserved_D0C0_23_12", 147}}},
    {27, {{"presc", 15}, {"mult", 15}, {"Reserved_D0C0_23_12", 147}}},
    {28, {{"presc", 14}, {"mult", 7}, {"Reserved_D0C0_23_12", 158}}},
    {29, {{"presc", 14}, {"mult", 7}, {"Reserved_D0C0_23_12", 158}}},
    {30, {{"presc", 15}, {"mult", 13}, {"Reserved_D0C0_23_12", 170}}},
    {31, {{"presc", 15}, {"mult", 13}, {"Reserved_D0C0_23_12", 170}}},
    {32, {{"presc", 13}, {"mult", 3}, {"Reserved_D0C0_23_12", 185}}},
    {33, {{"presc", 13}, {"mult", 3}, {"Reserved_D0C0_23_12", 185}}},
    {34, {{"presc", 13}, {"mult", 3}, {"Reserved_D0C0_23_12", 185}}},
    {35, {{"presc", 13}, {"mult", 3}, {"Reserved_D0C0_23_12", 90}}},
    {36, {{"presc", 13}, {"mult", 3}, {"Reserved_D0C0_23_12", 90}}},
    {37, {{"presc", 15}, {"mult", 11}, {"Reserved_D0C0_23_12", 202}}},
    {38, {{"presc", 15}, {"mult", 11}, {"Reserved_D0C0_23_12", 99}}},
    {39, {{"presc", 15}, {"mult", 11}, {"Reserved_D0C0_23_12", 99}}},
    {40, {{"presc", 15}, {"mult", 11}, {"Reserved_D0C0_23_12", 99}}},
    {41, {{"presc", 14}, {"mult", 5}, {"Reserved_D0C0_23_12", 109}}},
    {42, {{"presc", 14}, {"mult", 5}, {"Reserved_D0C0_23_12", 109}}},
    {43, {{"presc", 14}, {"mult", 5}, {"Reserved_D0C0_23_12", 109}}},
    {44, {{"presc", 14}, {"mult", 5}, {"Reserved_D0C0_23_12", 109}}},
    {45, {{"presc", 15}, {"mult", 9}, {"Reserved_D0C0_23_12", 248}}},
    {46, {{"presc", 15}, {"mult", 9}, {"Reserved_D0C0_23_12", 122}}},
    {47, {{"presc", 15}, {"mult", 9}, {"Reserved_D0C0_23_12", 122}}},
    {48, {{"presc", 15}, {"mult", 9}, {"Reserved_D0C0_23_12", 122}}},
    {49, {{"presc", 15}, {"mult", 9}, {"Reserved_D0C0_23_12", 122}}},
    {50, {{"presc", 12}, {"mult", 1}, {"Reserved_D0C0_23_12", 280}}},
    {51, {{"presc", 12}, {"mult", 1}, {"Reserved_D0C0_23_12", 280}}},
    {52, {{"presc", 12}, {"mult", 1}, {"Reserved_D0C0_23_12", 138}}},
    {53, {{"presc", 12}, {"mult", 1}, {"Reserved_D0C0_23_12", 138}}},
    {54, {{"presc", 12}, {"mult", 1}, {"Reserved_D0C0_23_12", 138}}},
    {55, {{"presc", 12}, {"mult", 1}, {"Reserved_D0C0_23_12", 138}}},
    {56, {{"presc", 16}, {"mult", 15}, {"Reserved_D0C0_23_12", 147}}},
    {57, {{"presc", 16}, {"mult", 15}, {"Reserved_D0C0_23_12", 147}}},
    {58, {{"presc", 16}, {"mult", 15}, {"Reserved_D0C0_23_12", 147}}},
    {59, {{"presc", 15}, {"mult", 7}, {"Reserved_D0C0_23_12", 158}}},
    {60, {{"presc", 15}, {"mult", 7}, {"Reserved_D0C0_23_12", 158}}},
    {61, {{"presc", 15}, {"mult", 7}, {"Reserved_D0C0_23_12", 158}}},
    {62, {{"presc", 15}, {"mult", 7}, {"Reserved_D0C0_23_12", 158}}},
    {63, {{"presc", 15}, {"mult", 7}, {"Reserved_D0C0_23_12", 158}}},
    {64, {{"presc", 16}, {"mult", 13}, {"Reserved_D0C0_23_12", 171}}},
    {65, {{"presc", 16}, {"mult", 13}, {"Reserved_D0C0_23_12", 171}}},
    {66, {{"presc", 16}, {"mult", 13}, {"Reserved_D0C0_23_12", 171}}},
    {67, {{"presc", 16}, {"mult", 13}, {"Reserved_D0C0_23_12", 171}}},
    {68, {{"presc", 16}, {"mult", 13}, {"Reserved_D0C0_23_12", 171}}},
    {69, {{"presc", 14}, {"mult", 3}, {"Reserved_D0C0_23_12", 185}}},
    {70, {{"presc", 14}, {"mult", 3}, {"Reserved_D0C0_23_12", 185}}},
    {71, {{"presc", 14}, {"mult", 3}, {"Reserved_D0C0_23_12", 185}}},
    {72, {{"presc", 14}, {"mult", 3}, {"Reserved_D0C0_23_12", 185}}},
    {73, {{"presc", 14}, {"mult", 3}, {"Reserved_D0C0_23_12", 185}}},
    {74, {{"presc", 16}, {"mult", 11}, {"Reserved_D0C0_23_12", 409}}},
    {75, {{"presc", 16}, {"mult", 11}, {"Reserved_D0C0_23_12", 202}}},
    {76, {{"presc", 16}, {"mult", 11}, {"Reserved_D0C0_23_12", 202}}},
    {77, {{"presc", 16}, {"mult", 11}, {"Reserved_D0C0_23_12", 202}}},
    {78, {{"presc", 16}, {"mult", 11}, {"Reserved_D0C0_23_12", 202}}},
    {79, {{"presc", 16}, {"mult", 11}, {"Reserved_D0C0_23_12", 202}}},
    {80, {{"presc", 16}, {"mult", 11}, {"Reserved_D0C0_23_12", 202}}},
    {81, {{"presc", 15}, {"mult", 5}, {"Reserved_D0C0_23_12", 451}}},
    {82, {{"presc", 15}, {"mult", 5}, {"Reserved_D0C0_23_12", 223}}},
    {83, {{"presc", 15}, {"mult", 5}, {"Reserved_D0C0_23_12", 223}}},
    {84, {{"presc", 15}, {"mult", 5}, {"Reserved_D0C0_23_12", 223}}},
    {85, {{"presc", 15}, {"mult", 5}, {"Reserved_D0C0_23_12", 223}}},
    {86, {{"presc", 15}, {"mult", 5}, {"Reserved_D0C0_23_12", 223}}},
    {87, {{"presc", 15}, {"mult", 5}, {"Reserved_D0C0_23_12", 223}}},
    {88, {{"presc", 15}, {"mult", 5}, {"Reserved_D0C0_23_12", 223}}},
    {89, {{"presc", 16}, {"mult", 9}, {"Reserved_D0C0_23_12", 501}}},
    {90, {{"presc", 16}, {"mult", 9}, {"Reserved_D0C0_23_12", 501}}},
    {91, {{"presc", 16}, {"mult", 9}, {"Reserved_D0C0_23_12", 501}}},
    {92, {{"presc", 16}, {"mult", 9}, {"Reserved_D0C0_23_12", 248}}},
    {93, {{"presc", 16}, {"mult", 9}, {"Reserved_D0C0_23_12", 248}}},
    {94, {{"presc", 16}, {"mult", 9}, {"Reserved_D0C0_23_12", 248}}},
    {95, {{"presc", 16}, {"mult", 9}, {"Reserved_D0C0_23_12", 248}}},
    {96, {{"presc", 16}, {"mult", 9}, {"Reserved_D0C0_23_12", 248}}},
    {97, {{"presc", 16}, {"mult", 9}, {"Reserved_D0C0_23_12", 248}}},
    {98, {{"presc", 16}, {"mult", 9}, {"Reserved_D0C0_23_12", 248}}},
    {99, {{"presc", 13}, {"mult", 1}, {"Reserved_D0C0_23_12", 564}}},
    {100, {{"presc", 13}, {"mult", 1}, {"Reserved_D0C0_23_12", 564}}}};
}

Gen41NoiseFilterModule::Gen41NoiseFilterModule(const std::shared_ptr<RegisterMap> &register_map,
                                               const std::string &sensor_prefix) :
    register_map_(register_map), sensor_prefix_(sensor_prefix) {}

void Gen41NoiseFilterModule::enable(I_NoiseFilterModule::Type type, uint32_t threshold) {
    uint32_t threshold_ms = 0;

    if (threshold < 1000 || threshold > 100000) {
        MV_HAL_LOG_ERROR() << "Bad STC threshold value";
        return;
    } else {
        threshold_ms = std::roundf(threshold / 1000.0);
    }
    // Bypass filter
    disable();

    // Start sram init
    (*register_map_)[sensor_prefix_ + "stc/initialization"]["stc_flag_init_done"].write_value(1);
    (*register_map_)[sensor_prefix_ + "stc/initialization"]["stc_req_init"].write_value(1);

    // Setup new configuration
    if (type == I_NoiseFilterModule::Type::STC) {
        (*register_map_)[sensor_prefix_ + "stc/stc_param"].write_value(
            {{"stc_enable", 1}, {"stc_threshold", threshold_ms * 1000}});
        (*register_map_)[sensor_prefix_ + "stc/trail_param"].write_value({"trail_enable", 0});

    } else if (type == I_NoiseFilterModule::Type::TRAIL) {
        (*register_map_)[sensor_prefix_ + "stc/stc_param"].write_value({"stc_enable", 0});
        (*register_map_)[sensor_prefix_ + "stc/trail_param"].write_value(
            {{"trail_enable", 1}, {"trail_threshold", threshold_ms * 1000}});
    }

    (*register_map_)[sensor_prefix_ + "stc/timestamping"].write_value(
        {{"prescaler", stc_threshold_params[threshold_ms]["presc"]},
         {"multiplier", stc_threshold_params[threshold_ms]["mult"]}});
    (*register_map_)[sensor_prefix_ + "stc/Reserved_D0C0"].write_value(
        {"Reserved_23_12", stc_threshold_params[threshold_ms]["Reserved_D0C0_23_12"]});

    // Check sram init done
    bool init_done = 0;
    for (int i = 0; i < 3; i++) {
        init_done = (*register_map_)[sensor_prefix_ + "stc/initialization"]["stc_flag_init_done"].read_value();
        if (init_done == 1) {
            break;
        }
    }
    if (init_done == 0) {
        MV_HAL_LOG_ERROR() << "Bad STC initialization";
        return;
    }

    // Enable filter
    (*register_map_)[sensor_prefix_ + "stc/pipeline_control"]["stc_trail_bypass"].write_value(0b001);
}

void Gen41NoiseFilterModule::disable() {
    (*register_map_)[sensor_prefix_ + "stc/pipeline_control"]["stc_trail_bypass"].write_value(0b101);
}

} // namespace Metavision
