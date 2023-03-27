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

#include "metavision/hal/utils/hal_exception.h"
#include "metavision/psee_hw_layer/devices/imx636/imx636_event_trail_filter_module.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

namespace {
std::map<const int, std::map<const std::string, const int>> stc_threshold_params = {
    {1, {{"presc", 12}, {"mult", 15}, {"dt_fifo_timeout", 90}}},
    {2, {{"presc", 10}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {3, {{"presc", 11}, {"mult", 5}, {"dt_fifo_timeout", 95}}},
    {4, {{"presc", 9}, {"mult", 1}, {"dt_fifo_timeout", 102}}},
    {5, {{"presc", 13}, {"mult", 15}, {"dt_fifo_timeout", 90}}},
    {6, {{"presc", 11}, {"mult", 3}, {"dt_fifo_timeout", 114}}},
    {7, {{"presc", 11}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {8, {{"presc", 12}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {9, {{"presc", 13}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {10, {{"presc", 13}, {"mult", 9}, {"dt_fifo_timeout", 90}}},
    {11, {{"presc", 10}, {"mult", 1}, {"dt_fifo_timeout", 102}}},
    {12, {{"presc", 14}, {"mult", 15}, {"dt_fifo_timeout", 109}}},
    {13, {{"presc", 13}, {"mult", 7}, {"dt_fifo_timeout", 117}}},
    {14, {{"presc", 14}, {"mult", 13}, {"dt_fifo_timeout", 127}}},
    {15, {{"presc", 12}, {"mult", 3}, {"dt_fifo_timeout", 138}}},
    {16, {{"presc", 12}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {17, {{"presc", 12}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {18, {{"presc", 14}, {"mult", 11}, {"dt_fifo_timeout", 99}}},
    {19, {{"presc", 13}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {20, {{"presc", 13}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {21, {{"presc", 14}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {22, {{"presc", 14}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {23, {{"presc", 11}, {"mult", 1}, {"dt_fifo_timeout", 209}}},
    {24, {{"presc", 11}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {25, {{"presc", 11}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {26, {{"presc", 15}, {"mult", 15}, {"dt_fifo_timeout", 147}}},
    {27, {{"presc", 15}, {"mult", 15}, {"dt_fifo_timeout", 147}}},
    {28, {{"presc", 14}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {29, {{"presc", 14}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {30, {{"presc", 15}, {"mult", 13}, {"dt_fifo_timeout", 170}}},
    {31, {{"presc", 15}, {"mult", 13}, {"dt_fifo_timeout", 170}}},
    {32, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {33, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {34, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {35, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {36, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {37, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {38, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 99}}},
    {39, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 99}}},
    {40, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 99}}},
    {41, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {42, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {43, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {44, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {45, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {46, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {47, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {48, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {49, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {50, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 280}}},
    {51, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 280}}},
    {52, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {53, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {54, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {55, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {56, {{"presc", 16}, {"mult", 15}, {"dt_fifo_timeout", 147}}},
    {57, {{"presc", 16}, {"mult", 15}, {"dt_fifo_timeout", 147}}},
    {58, {{"presc", 16}, {"mult", 15}, {"dt_fifo_timeout", 147}}},
    {59, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {60, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {61, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {62, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {63, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {64, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 171}}},
    {65, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 171}}},
    {66, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 171}}},
    {67, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 171}}},
    {68, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 171}}},
    {69, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {70, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {71, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {72, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {73, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {74, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 409}}},
    {75, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {76, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {77, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {78, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {79, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {80, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {81, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 451}}},
    {82, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {83, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {84, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {85, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {86, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {87, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {88, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {89, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 501}}},
    {90, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 501}}},
    {91, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 501}}},
    {92, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {93, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {94, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {95, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {96, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {97, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {98, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {99, {{"presc", 13}, {"mult", 1}, {"dt_fifo_timeout", 564}}},
    {100, {{"presc", 13}, {"mult", 1}, {"dt_fifo_timeout", 564}}}};
}

Imx636EventTrailFilterModule::Imx636EventTrailFilterModule(const std::shared_ptr<RegisterMap> &register_map,
                                                           const std::string &sensor_prefix) :
    register_map_(register_map), sensor_prefix_(sensor_prefix) {}

std::set<I_EventTrailFilterModule::Type> Imx636EventTrailFilterModule::get_available_types() const {
    return {I_EventTrailFilterModule::Type::STC_CUT_TRAIL, I_EventTrailFilterModule::Type::STC_KEEP_TRAIL,
            I_EventTrailFilterModule::Type::TRAIL};
}

bool Imx636EventTrailFilterModule::set_threshold(uint32_t threshold) {
    if (threshold < get_min_supported_threshold() || threshold > get_max_supported_threshold()) {
        std::stringstream ss;
        ss << "Bad STC threshold value: " << threshold << ". Value should be in range [1000, 100000].";
        throw HalException(HalErrorCode::InvalidArgument, ss.str());
    }

    threshold_ms_ = std::roundf(threshold / 1000.0);

    // Reset if needed
    if (is_enabled()) {
        enable(false);
        enable(true);
    }

    return true;
}

bool Imx636EventTrailFilterModule::set_type(I_EventTrailFilterModule::Type type) {
    if (get_available_types().count(type) <= 0) {
        throw HalException(HalErrorCode::UnsupportedValue);
    }

    filtering_type_ = type;

    // Reset if needed
    if (is_enabled()) {
        enable(false);
        enable(true);
    }

    return true;
}

bool Imx636EventTrailFilterModule::enable(bool state) {
    // Bypass filter
    (*register_map_)[sensor_prefix_ + "stc/pipeline_control"].write_value(0b101);
    enabled_ = false;
    if (!state)
        return true;

    // Start sram init
    (*register_map_)[sensor_prefix_ + "stc/initialization"]["stc_flag_init_done"].write_value(1);
    (*register_map_)[sensor_prefix_ + "stc/initialization"]["stc_req_init"].write_value(1);

    // Setup new configuration
    if (filtering_type_ == I_EventTrailFilterModule::Type::STC_CUT_TRAIL ||
        filtering_type_ == I_EventTrailFilterModule::Type::STC_KEEP_TRAIL) {
        (*register_map_)[sensor_prefix_ + "stc/stc_param"].write_value(
            {{"stc_enable", 1},
             {"stc_threshold", threshold_ms_ * 1000},
             {"disable_stc_cut_trail", filtering_type_ == I_EventTrailFilterModule::Type::STC_KEEP_TRAIL}});
        (*register_map_)[sensor_prefix_ + "stc/trail_param"].write_value({"trail_enable", 0});

    } else if (filtering_type_ == I_EventTrailFilterModule::Type::TRAIL) {
        (*register_map_)[sensor_prefix_ + "stc/stc_param"].write_value({"stc_enable", 0});
        (*register_map_)[sensor_prefix_ + "stc/trail_param"].write_value(
            {{"trail_enable", 1}, {"trail_threshold", threshold_ms_ * 1000}});
    }

    (*register_map_)[sensor_prefix_ + "stc/timestamping"].write_value(
        {{"prescaler", stc_threshold_params[threshold_ms_]["presc"]},
         {"multiplier", stc_threshold_params[threshold_ms_]["mult"]},
         {"enable_last_ts_update_at_every_event", 1}});
    (*register_map_)[sensor_prefix_ + "stc/invalidation"].write_value(
        {"dt_fifo_timeout", stc_threshold_params[threshold_ms_]["dt_fifo_timeout"]});

    // Check sram init done
    bool init_done = 0;
    for (int i = 0; i < 3; i++) {
        init_done = (*register_map_)[sensor_prefix_ + "stc/initialization"]["stc_flag_init_done"].read_value();
        if (init_done == 1) {
            break;
        }
    }
    if (init_done == 0) {
        throw HalException(HalErrorCode::InternalInitializationError, "Bad STC initialization");
    }

    // Enable filter
    (*register_map_)[sensor_prefix_ + "stc/pipeline_control"].write_value(0b001);
    enabled_ = true;

    return true;
}

bool Imx636EventTrailFilterModule::is_enabled() const {
    return enabled_;
}

I_EventTrailFilterModule::Type Imx636EventTrailFilterModule::get_type() const {
    return filtering_type_;
}

uint32_t Imx636EventTrailFilterModule::get_threshold() const {
    return threshold_ms_ * 1000;
}

uint32_t Imx636EventTrailFilterModule::get_max_supported_threshold() const {
    return 100000;
}

uint32_t Imx636EventTrailFilterModule::get_min_supported_threshold() const {
    return 1000;
}

} // namespace Metavision
