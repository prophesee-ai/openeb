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

#include <cmath>
#include <map>
#include <sstream>
#include <iostream>

#include "metavision/hal/utils/hal_log.h"
#include "boards/utils/psee_libusb_board_command.h"
#include "devices/gen3/gen3_ll_biases.h"
#include "devices/gen3/legacy_regmap_headers/legacy/stereo_pc_mapping.h"
#include "utils/psee_hal_utils.h"

namespace Metavision {

namespace {

#include "gen3_idac.calib.cc"
#include "vdac18_8m1.calib.cc"

class Gen3LLBias {
public:
    enum class Mode {
        Current = 0,
        Voltage = 1,
    };

    Gen3LLBias(long value, const std::pair<long, long> &range, Mode mode, int index, bool modifiable = true) :
        value(value), range(range), mode(mode), index(index), modifiable(modifiable) {}

    long value;
    std::pair<long, long> range;
    Mode mode = Mode::Voltage;
    int index;
    bool modifiable;
};

std::map<long, long long> vdac18_map_s;

std::map<long, long long> load_vdac18_calibration(std::istream &ifs, bool is_voltage, long long config) {
    std::map<long, long long> calibration_data;
    char buf[256];

    // Read data
    unsigned int mv_value, dec_value, bin_value, pn_value, cas_value, prev_value, next_value;
    while (!ifs.eof()) {
        ifs.getline(buf, sizeof(buf));
        if (buf[0] == '%') {
            continue;
        }
        if (ifs.gcount() != 0) {
            sscanf(buf, "%u , %u , %u , %u , %u , %u , %u", &mv_value, &dec_value, &prev_value, &next_value, &pn_value,
                   &cas_value, &bin_value);
            long long bin_data = (0LL << 31)   // bias enable
                                 + (1LL << 30) // output enable
                                 + ((pn_value == 1 ? 1LL : 0LL) << 29) + ((cas_value == 1 ? 1LL : 0LL) << 28) +
                                 ((is_voltage ? 1LL : 0LL) << 27) + (config << 21) + bin_value;

            calibration_data.insert({(long)mv_value, bin_data});

        } else {
            continue;
        }
    }

    return calibration_data;
}

bool load_vdac18_calibration() {
    if (vdac18_map_s.size() == 0) {
        std::string str(vdac18_8m1);
        std::istringstream mistr(str);
        vdac18_map_s = load_vdac18_calibration(mistr, true, 8);
    }

    if (vdac18_map_s.size() == 0) {
        MV_HAL_LOG_ERROR() << "Unable to open gen3_vdac calibration";
        return false;
    }

    return true;
}

long long get_vdac18_values(long value) {
    if (vdac18_map_s.empty()) {
        load_vdac18_calibration();
    }
    return vdac18_map_s[value];
}

struct CodeSysleyIDAC {
    int code_;
    enum Mos { P, N } mos_;
    bool cas_;
};

std::map<int, CodeSysleyIDAC> s_mvtocode;
bool load_v_to_code(std::istream &ifs) {
    s_mvtocode.clear();
    while (ifs) {
        std::string tmp;
        std::getline(ifs, tmp);
        if (tmp.empty())
            break;
        std::istringstream istr(tmp);
        int v;
        CodeSysleyIDAC codeidac;
        istr >> v;
        istr >> codeidac.code_;
        std::string idac;
        istr >> idac;
        if (idac == "PCas") {
            codeidac.mos_ = CodeSysleyIDAC::P;
            codeidac.cas_ = true;
        } else if (idac == "PwoCas") {
            codeidac.mos_ = CodeSysleyIDAC::P;
            codeidac.cas_ = false;
        } else if (idac == "NCas") {
            codeidac.mos_ = CodeSysleyIDAC::N;
            codeidac.cas_ = true;
        } else if (idac == "NwoCas") {
            codeidac.mos_ = CodeSysleyIDAC::N;
            codeidac.cas_ = false;
        } else {
            break;
        }
        s_mvtocode[v] = codeidac;
    }
    return s_mvtocode.size() == 1801;
}

bool load_v_to_code() {
    std::string str(gen3_idac);
    std::istringstream mistr(str);
    if (!load_v_to_code(mistr)) {
        return false;
    }
    return true;
}

struct CCam3BiasEncoding {
    uint32_t voltage_value_ : 8;
    uint32_t current_value_ : 13;

    uint32_t buffer_value_ : 6;
    uint32_t type_ : 1; // Current Voltage
    uint32_t cas_code_ : 1;
    uint32_t polarity_ : 1;
    uint32_t bias_enable_ : 1;
    uint32_t pad_enable_ : 1;

    CCam3BiasEncoding() {
        pad_enable_    = 0;
        bias_enable_   = 0;
        polarity_      = 1;
        cas_code_      = 1;
        type_          = 0;
        buffer_value_  = 0;
        current_value_ = 0;
        voltage_value_ = 0;
    }

    CCam3BiasEncoding(uint32_t bias_enable) {
        pad_enable_    = bias_enable;
        bias_enable_   = bias_enable;
        polarity_      = 1;
        cas_code_      = 1;
        type_          = 0x1;
        buffer_value_  = bias_enable_ ? 0x08 : 0x01;
        current_value_ = 1;
        voltage_value_ = 0;
    }
};

union BiasEncodingCast {
    CCam3BiasEncoding bias_encoding;
    long raw;
    BiasEncodingCast() {
        raw = 0;
    }
};

long get_ccam3_bias_encoding(const Gen3LLBias &bias, int bias_value, bool bias_disabled = false) {
    if (s_mvtocode.empty()) {
        bool loaded = load_v_to_code();
        if (!loaded) {
            MV_HAL_LOG_ERROR() << "Unable to open gen3_idac calibration";
        }
    }

    BiasEncodingCast encoder;
    encoder.bias_encoding = CCam3BiasEncoding(!bias_disabled);

    if (bias.mode == Gen3LLBias::Mode::Current) {
        int v = bias_value;
        if (!Metavision::is_expert_mode_enabled()) {
            if (v < 0)
                v = 0;
            if (v > 1800)
                v = 1800;
        }
        auto it = s_mvtocode.find(v);
        if (it != s_mvtocode.end()) {
            CodeSysleyIDAC codeidac              = it->second;
            encoder.bias_encoding.current_value_ = codeidac.code_;
            encoder.bias_encoding.voltage_value_ = 0;
            encoder.bias_encoding.type_          = 0;
            encoder.bias_encoding.cas_code_      = !codeidac.cas_;
            encoder.bias_encoding.polarity_      = codeidac.mos_ == CodeSysleyIDAC::N;
            encoder.bias_encoding.buffer_value_  = 0x08;
        } else {
            MV_HAL_LOG_WARNING() << "Err value not found in LUT" << v;
            encoder.bias_encoding.voltage_value_ =
                static_cast<long>(round(static_cast<double>(bias_value * 255) / 1800.)) & 0xFF;
            if (bias_value < 900) {
                encoder.bias_encoding.polarity_ = 1;
            } else {
                encoder.bias_encoding.polarity_ = 0;
            }
        }
    } else {
        long calib_encoding                  = get_vdac18_values(bias_value);
        encoder.bias_encoding.current_value_ = 0x1;
        encoder.bias_encoding.voltage_value_ = (calib_encoding >> 0) & 0xFF;
        encoder.bias_encoding.type_          = (calib_encoding >> 27) & 1;
        encoder.bias_encoding.cas_code_      = (calib_encoding >> 28) & 1;
        encoder.bias_encoding.polarity_      = (calib_encoding >> 29) & 1;
    }
    return encoder.raw;
}

std::map<std::string, Gen3LLBias> &get_full_biases_map() {
    // clang-format off
    static std::map<std::string, Gen3LLBias> biases_map {
        {"bias_latchout_or_pu", Gen3LLBias(100, {0, 1800},     Gen3LLBias::Mode::Voltage, 0)},
        {"bias_reqx_or_pu",     Gen3LLBias(100, {0, 1800},     Gen3LLBias::Mode::Voltage, 1)},
        {"bias_req_pux",        Gen3LLBias(1200, {0, 1800},    Gen3LLBias::Mode::Voltage, 2)},
        {"bias_req_puy",        Gen3LLBias(1200, {0, 1800},    Gen3LLBias::Mode::Voltage, 3)},
        {"bias_del_reqx_or",    Gen3LLBias(1000, {0, 1800},    Gen3LLBias::Mode::Voltage, 4)},
        {"bias_sendreq_pdx",    Gen3LLBias(800, {0, 1800},     Gen3LLBias::Mode::Voltage, 5)},
        {"bias_sendreq_pdy",    Gen3LLBias(1000, {0, 1800},    Gen3LLBias::Mode::Voltage, 6)},
        {"bias_del_ack_array",  Gen3LLBias(1000, {0, 1800},    Gen3LLBias::Mode::Voltage, 7)},
        {"bias_del_timeout",    Gen3LLBias(450, {0, 1800},     Gen3LLBias::Mode::Voltage, 8)},
        {"bias_inv",            Gen3LLBias(500, {0, 1800},     Gen3LLBias::Mode::Voltage, 9)},
        {"bias_refr",           Gen3LLBias(1500, {1300, 1800}, Gen3LLBias::Mode::Voltage, 10)},
        {"bias_clk",            Gen3LLBias(600, {0, 1800},     Gen3LLBias::Mode::Voltage, 11)},
        {"bias_overflow",       Gen3LLBias(0, {0, 1800},       Gen3LLBias::Mode::Voltage, 12)},
        {"bias_tail",           Gen3LLBias(0, {0, 1800},       Gen3LLBias::Mode::Voltage, 13)},
        {"bias_out",            Gen3LLBias(0, {0, 1800},       Gen3LLBias::Mode::Voltage, 14)},
        {"bias_hyst",           Gen3LLBias(0, {0, 1800},       Gen3LLBias::Mode::Voltage, 15)},
        {"bias_vrefl",          Gen3LLBias(0, {0, 1800},       Gen3LLBias::Mode::Voltage, 16)},
        {"bias_vrefh",          Gen3LLBias(0, {0, 1800},       Gen3LLBias::Mode::Voltage, 17)},
        {"bias_cas",            Gen3LLBias(1000, {0, 1800},    Gen3LLBias::Mode::Voltage, 18)},
        {"bias_diff_off",       Gen3LLBias(225, {0, 1800},     Gen3LLBias::Mode::Voltage, 19)},
        {"bias_diff_on",        Gen3LLBias(375, {0, 1800},     Gen3LLBias::Mode::Voltage, 20)},
        {"bias_diff",           Gen3LLBias(300, {0, 1800},     Gen3LLBias::Mode::Voltage, 21)},
        {"bias_fo",             Gen3LLBias(1725, {1650, 1800}, Gen3LLBias::Mode::Voltage, 22)},
        {"bias_pr",             Gen3LLBias(1500, {1200, 1800}, Gen3LLBias::Mode::Voltage, 23)},
        {"bias_bulk",           Gen3LLBias(1500, {0, 1800},    Gen3LLBias::Mode::Voltage, 24)},
        {"bias_hpf",            Gen3LLBias(1500, {0, 1800},    Gen3LLBias::Mode::Voltage, 25)},
        {"bias_buf",            Gen3LLBias(600, {0, 1800},     Gen3LLBias::Mode::Voltage, 26)},
    };
    // clang-format on

    return biases_map;
}

std::map<std::string, Gen3LLBias> &get_biases_map() {
    // clang-format off
    static std::map<std::string, Gen3LLBias> biases_map {
        {"bias_refr",     Gen3LLBias(1500, {1300, 1800}, Gen3LLBias::Mode::Voltage, 10, true)},
        {"bias_diff_off", Gen3LLBias(225, {0, 1800},     Gen3LLBias::Mode::Voltage, 19, true)},
        {"bias_diff_on",  Gen3LLBias(375, {0, 1800},     Gen3LLBias::Mode::Voltage, 20, true)},
        {"bias_diff",     Gen3LLBias(300, {0, 1800},     Gen3LLBias::Mode::Voltage, 21, false)},
        {"bias_fo",       Gen3LLBias(1725, {1650, 1800}, Gen3LLBias::Mode::Voltage, 22, true)},
        {"bias_pr",       Gen3LLBias(1500, {1200, 1800}, Gen3LLBias::Mode::Voltage, 23, true)},
        {"bias_hpf",      Gen3LLBias(1500, {0, 1800},    Gen3LLBias::Mode::Voltage, 25, true)},
    };
    // clang-format on

    return biases_map;
}
} /* namespace */

struct Gen3_LL_Biases::Private {
    Private(const std::shared_ptr<PseeLibUSBBoardCommand> &cmd) : cmd(cmd) {
        base_sensor_address = CCAM3_SENSOR_IF_BASE_ADDR;
        base_bgen_address   = CCAM3_SISLEY_IBGEN_BASE_ADDR;

        // initialize gen3 biases by loading reference biases during construction
        biases_map = get_full_biases_map();
        for (auto &&it = biases_map.begin(); it != biases_map.end(); ++it) {
            set(it->first, it->second.value);
        }

        // set default bias map
        biases_map = get_biases_map();
    }

    bool set(const std::string &bias_name, int bias_value) {
        auto it = biases_map.find(bias_name);
        if (it == biases_map.end()) {
            return false;
        }
        auto &bias = it->second;
        if (!bias.modifiable) {
            return false;
        }

        if (bias_name == "bias_diff_on") {
            auto b = get("bias_diff");
            if (b == -1) {
                MV_HAL_LOG_WARNING() << "Cannot clamp bias";
            } else {
                if (bias_value < b + 1)
                    bias_value = b + 1;
            }
        }
        if (bias_name == "bias_diff_off") {
            auto b = get("bias_diff");
            if (b == -1) {
                MV_HAL_LOG_WARNING() << "Cannot clamp bias";
            } else {
                if (bias_value > b - 1)
                    bias_value = b - 1;
            }
        }

        if (!Metavision::is_expert_mode_enabled()) {
            if (bias_value > bias.range.second) {
                bias_value = bias.range.second;
            }
            if (bias_value < bias.range.first) {
                bias_value = bias.range.first;
            }
        }

        uint32_t encoding = get_ccam3_bias_encoding(bias, bias_value);
        if (encoding > 0) {
            cmd->write_register(base_bgen_address + 4 * bias.index, encoding);
            bias.value = bias_value;
        } else {
            MV_HAL_LOG_WARNING() << "Could not set bias" << bias_name;
        }

        return true;
    }

    int get(const std::string &bias_name) const {
        auto it = biases_map.find(bias_name);
        if (it == biases_map.end()) {
            return -1;
        }

        return it->second.value;
    }

    std::map<std::string, int> get_all_biases() {
        std::map<std::string, int> ret;
        for (auto &b : biases_map) {
            ret[b.first] = b.second.value;
        }
        return ret;
    }

    std::shared_ptr<PseeLibUSBBoardCommand> cmd;
    uint32_t base_bgen_address;
    uint32_t base_sensor_address;
    std::map<std::string, Gen3LLBias> biases_map;
};

Gen3_LL_Biases::Gen3_LL_Biases(const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd) :
    pimpl_(new Private(board_cmd)) {}

Gen3_LL_Biases::~Gen3_LL_Biases() = default;

bool Gen3_LL_Biases::set(const std::string &bias_name, int bias_value) {
    return pimpl_->set(bias_name, bias_value);
}

int Gen3_LL_Biases::get(const std::string &bias_name) {
    return pimpl_->get(bias_name);
}

std::map<std::string, int> Gen3_LL_Biases::get_all_biases() {
    return pimpl_->get_all_biases();
}

} // namespace Metavision
