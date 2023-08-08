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

#include <fstream>
#include <sstream>
#include <iomanip>

#include "metavision/psee_hw_layer/devices/gen41/gen41_erc.h"
#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

namespace {
constexpr uint32_t CD_EVENT_COUNT_DEFAULT = 4000;

std::string hex_to_string(int number, int n_digits) {
    std::stringstream ss;
    ss << std::uppercase << std::hex << std::setw(n_digits) << std::setfill('0') << number;
    return ss.str();
}

std::string int_to_string(int number, int n_digits) {
    std::stringstream ss;
    ss << std::dec << std::setw(n_digits) << std::setfill('0') << number;
    return ss.str();
}

} // namespace

constexpr uint32_t Gen41Erc::CD_EVENT_COUNT_MAX;

Gen41Erc::Gen41Erc(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix,
                   std::shared_ptr<TzDevice> tzDev) :
    register_map_(register_map), cd_event_count_shadow_(CD_EVENT_COUNT_DEFAULT), prefix_(prefix), tzDev_(tzDev) {
    // Set Default Configuration
    for (auto i = 0; i < 230; ++i) {
        lut_configs["Reserved"][i] = std::make_tuple(0x8, 0x8, 0x8, 0x8);
    }
    for (auto i = 0; i < 256; ++i) {
        lut_configs["t_drop_lut"][i] = std::make_tuple((i * 2) + 0, (i * 2) + 1, 0, 0);
    }
}

bool Gen41Erc::enable(bool en) {
    (*register_map_)[prefix_ + "t_dropping_control"].write_value({"t_dropping_en", en});

    if (en) {
        set_cd_event_count(cd_event_count_shadow_);
    }

    return true;
}

bool Gen41Erc::is_enabled() const {
    bool res           = ((*register_map_)[prefix_ + "Reserved_6000"]["Reserved_1_0"].read_value() == 1) ? true : false;
    bool t_dropping_en = (*register_map_)[prefix_ + "t_dropping_control"]["t_dropping_en"].read_value();
    return t_dropping_en && res;
}

void Gen41Erc::initialize() {
    MV_HAL_LOG_TRACE() << "Gen41 ERC Init";

    (*register_map_)[prefix_ + "Reserved_6000"]["Reserved_1_0"].write_value(0);

    char *config = getenv("ERC_CONFIGURATION_PATH");

    if (config) {
        // A file was given to override default configuration
        erc_from_file(std::string(config));
    }

    (*register_map_)[prefix_ + "in_drop_rate_control"]["cfg_event_delay_fifo_en"].write_value(1);
    (*register_map_)[prefix_ + "reference_period"].write_value({"erc_reference_period", 200});
    (*register_map_)[prefix_ + "td_target_event_rate"].write_value({"target_event_rate", cd_event_count_shadow_});
    (*register_map_)[prefix_ + "erc_enable"].write_value({{"erc_en", 1}, {"Reserved_1", 1}, {"Reserved_2", 0}});

    (*register_map_)[prefix_ + "Reserved_602C"]["Reserved_0"].write_value(1);
    for (auto i = 0; i < 230; ++i) {
        (*register_map_)[prefix_ + "Reserved_" + hex_to_string((26624 + 4 * i), 4)].write_value(
            {{"Reserved_5_0", std::get<0>(lut_configs["Reserved"][i])},
             {"Reserved_13_8", std::get<1>(lut_configs["Reserved"][i])},
             {"Reserved_21_16", std::get<2>(lut_configs["Reserved"][i])},
             {"Reserved_29_24", std::get<3>(lut_configs["Reserved"][i])}});
    }
    (*register_map_)[prefix_ + "Reserved_602C"]["Reserved_0"].write_value(0);

    int j = 0;

    for (auto i = 0; i < 256; ++i) {
        std::string lut_index;
        std::string lut_field_index_0;
        std::string lut_field_index_1;

        lut_index         = int_to_string(i, 2);
        lut_field_index_0 = int_to_string(j, 3);
        lut_field_index_1 = int_to_string(j + 1, 3);

        (*register_map_)[prefix_ + "t_drop_lut_" + lut_index].write_value(
            {{"tlut" + lut_field_index_0, std::get<0>(lut_configs["t_drop_lut"][i])},
             {"tlut" + lut_field_index_1, std::get<1>(lut_configs["t_drop_lut"][i])}});

        j += 2;
    }
    (*register_map_)[prefix_ + "t_dropping_control"].write_value({"t_dropping_en", 0});
    (*register_map_)[prefix_ + "h_dropping_control"].write_value({"h_dropping_en", 0});
    (*register_map_)[prefix_ + "v_dropping_control"].write_value({"v_dropping_en", 0});

    (*register_map_)[prefix_ + "Reserved_6000"]["Reserved_1_0"].write_value(1);
}

void Gen41Erc::erc_from_file(const std::string &file_path) {
    uint32_t num, v0, v1, v2, v3;
    std::string reg_name;
    std::ifstream erc_file(file_path.c_str(), std::ios::in);

    MV_HAL_LOG_TRACE() << "Loading ERC configuration from file" << file_path;

    if (!erc_file.is_open()) {
        MV_HAL_LOG_WARNING() << "Could not open file at" << Metavision::Log::no_space << file_path << ": ERC not set.";
        return;
    }

    while (erc_file >> reg_name >> std::hex >> num >> v0 >> v1 >> v2 >> v3) {
        lut_configs[reg_name][num] = std::make_tuple(v0, v1, v2, v3);
    }
}

uint32_t Gen41Erc::get_count_period() const {
    return (*register_map_)[prefix_ + "reference_period"].read_value();
}

bool Gen41Erc::set_cd_event_count(uint32_t count) {
    if (count > CD_EVENT_COUNT_MAX) {
        std::stringstream ss;
        ss << "Cannot set CD event count to :" << count << ". Value should be in the range [0, " << CD_EVENT_COUNT_MAX
           << "]";
        throw HalException(HalErrorCode::ValueOutOfRange, ss.str());
    }
    (*register_map_)[prefix_ + "td_target_event_rate"].write_value(count);
    cd_event_count_shadow_ = count;

    return true;
}

uint32_t Gen41Erc::get_min_supported_cd_event_count() const {
    return 0;
}

uint32_t Gen41Erc::get_max_supported_cd_event_count() const {
    return CD_EVENT_COUNT_MAX;
}

uint32_t Gen41Erc::get_cd_event_count() const {
    return (*register_map_)[prefix_ + "td_target_event_rate"].read_value();
}

void Gen41Erc::set_device_control(const std::shared_ptr<PseeDeviceControl> &device_control) {
    /* Stores DeviceControl facility pointer.

    The goal is to guarantee that Gen41Erc will always be destroyed/freed before its DeviceControl.
    DeviceControl, when freed, will power down the sensor. Hence, further call to Gen41Erc methods will not
    be sent to its hardware or in case of Evk2 will break USB link.

    Note: Gen41Erc is used by DeviceControl through a weak pointer to avoid cyclic references
    */
    dev_ctrl_ = device_control;
}

} // namespace Metavision
