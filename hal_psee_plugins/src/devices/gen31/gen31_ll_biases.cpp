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

#include <cassert>
#include <math.h>
#include <map>
#include <sstream>
#include <iostream>

#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/devices/gen31/gen31_ll_biases.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "utils/psee_hal_utils.h"

// bias_diff, bias_diff_on, bias_diff_off, bias fo, bias_hpf, bias_refr, bias_pr
namespace {
class Gen31LLBias : public Metavision::LL_Bias_Info {
public:
    enum class BiasType {
        PMOSThick = 0,
        PMOSThin  = 1,
        NMOSThick = 2,
        NMOSThin  = 3,
        RailP     = 4,
    };
    enum class Mode {
        Current = 0,
        Voltage = 1,
    };

    Gen31LLBias(long min_value, long max_value, Mode mode, bool modifiable, std::string register_name, BiasType btype,
                const std::string &description, const std::string &category) :
        LL_Bias_Info(0, 1800, min_value, max_value, description, modifiable, category),
        register_name_(register_name),
        mode_(mode),
        btype_(btype) {}

    ~Gen31LLBias() {}
    Mode mode() const {
        return mode_;
    }
    BiasType get_bias_type() const {
        return btype_;
    }
    const std::string &get_register_name() const {
        return register_name_;
    }

private:
    std::string register_name_;
    Mode mode_ = Mode::Voltage;
    BiasType btype_;
};

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
        bias_enable_   = true;
        polarity_      = 1;
        cas_code_      = 1;
        type_          = 0x1;
        buffer_value_  = 0x8; // bias_enable_ ? 0x08 : 0x01;
        current_value_ = 1;
        voltage_value_ = 0;
    }
    struct Cmp_VDac {
        bool operator()(const CCam3BiasEncoding &c1, const CCam3BiasEncoding &c2) const {
            if (c1.polarity_ < c2.polarity_)
                return true;
            if (c1.polarity_ > c2.polarity_)
                return false;
            if (c1.cas_code_ < c2.cas_code_)
                return true;
            if (c1.cas_code_ > c2.cas_code_)
                return false;
            return c1.voltage_value_ < c2.voltage_value_;
        }
    };
    struct Cmp_IDac {
        bool operator()(const CCam3BiasEncoding &c1, const CCam3BiasEncoding &c2) const {
            if (c1.polarity_ < c2.polarity_)
                return true;
            if (c1.polarity_ > c2.polarity_)
                return false;
            if (c1.cas_code_ < c2.cas_code_)
                return true;
            if (c1.cas_code_ > c2.cas_code_)
                return false;
            return c1.current_value_ < c2.current_value_;
        }
    };
};

union BiasEncodingCast {
    CCam3BiasEncoding bias_encoding;
    long raw;
    BiasEncodingCast() {
        raw = 0;
    }
};

//         {"bias_latchout_or_pu", new Gen31Bias("bias_latchout_or_pu", 1250, {0, 1800},    Bias::Mode::Voltage,
//         Bias::standard, false,  0, Gen31Bias::BiasType::PMOSThin,  false)},

std::map<std::string, Gen31LLBias> &get_gen31_biases_map() {
    // clang-format off
    static std::map<std::string, Gen31LLBias> biases_map{
        {"bias_diff_off", Gen31LLBias(0, 1800, Gen31LLBias::Mode::Voltage, true, "bgen_19", Gen31LLBias::BiasType::NMOSThin, Metavision::get_bias_description("bias_diff_off"), Metavision::get_bias_category("bias_diff_off"))},
        {"bias_diff_on", Gen31LLBias(0, 1800, Gen31LLBias::Mode::Voltage, true, "bgen_20", Gen31LLBias::BiasType::NMOSThin, Metavision::get_bias_description("bias_diff_on"), Metavision::get_bias_category("bias_diff_on"))},
        {"bias_diff", Gen31LLBias(200, 400, Gen31LLBias::Mode::Voltage, true, "bgen_21", Gen31LLBias::BiasType::NMOSThin, Metavision::get_bias_description("bias_diff"), Metavision::get_bias_category("bias_diff"))},
        {"bias_fo", Gen31LLBias(1250, 1800, Gen31LLBias::Mode::Current, true, "bgen_22", Gen31LLBias::BiasType::RailP, Metavision::get_bias_description("bias_fo"), Metavision::get_bias_category("bias_fo"))},
        {"bias_pr", Gen31LLBias(975, 1800, Gen31LLBias::Mode::Current, true, "bgen_23", Gen31LLBias::BiasType::PMOSThick, Metavision::get_bias_description("bias_pr"), Metavision::get_bias_category("bias_pr"))},
        {"bias_refr", Gen31LLBias(1300, 1800, Gen31LLBias::Mode::Current, true, "bgen_24", Gen31LLBias::BiasType::NMOSThin, Metavision::get_bias_description("bias_refr"), Metavision::get_bias_category("bias_refr"))},
        {"bias_hpf", Gen31LLBias(900, 1800, Gen31LLBias::Mode::Current, true, "bgen_25", Gen31LLBias::BiasType::PMOSThick, Metavision::get_bias_description("bias_hpf"), Metavision::get_bias_category("bias_hpf"))},
    };
    // clang-format on
    return biases_map;
}

#include "gen3_idac_n_thick.calib.cc"
#include "gen3_idac_n_thin.calib.cc"
#include "gen3_idac_p_thick.calib.cc"
#include "gen3_idac_p_thin.calib.cc"
#include "gen3_idac_railp.calib.cc"
#include "vdac18_8m1.calib.cc"

std::map<long, long long> vdac18_map_s;
std::map<CCam3BiasEncoding, int, CCam3BiasEncoding::Cmp_VDac> inv_vdac18_map_s;

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

bool load_vdac18_calibration(std::map<long, long long> &vdac18_map_s) {
    if (vdac18_map_s.size() == 0) {
        std::string str(vdac18_8m1);
        std::istringstream mistr(str);
        vdac18_map_s = load_vdac18_calibration(mistr, true, 8);
    }

    if (vdac18_map_s.size() == 0) {
        MV_HAL_LOG_ERROR() << "Failed loading vdac calibration.";
        return false;
    }

    return true;
}

void init_map_vdac18() {
    if (vdac18_map_s.empty()) {
        load_vdac18_calibration(vdac18_map_s);
    }
    if (inv_vdac18_map_s.empty()) {
        for (auto &v : vdac18_map_s) {
            CCam3BiasEncoding c;
            c.current_value_ = 0x0;
            c.voltage_value_ = (v.second >> 0) & 0xFF;
            c.type_          = 1;
            c.cas_code_      = (v.second >> 28) & 1;
            c.polarity_      = 0;
            if (inv_vdac18_map_s.find(c) == inv_vdac18_map_s.end()) {
                inv_vdac18_map_s[c] = v.first;
            }
            c.polarity_ = 1;
            if (inv_vdac18_map_s.find(c) == inv_vdac18_map_s.end()) {
                inv_vdac18_map_s[c] = v.first;
            }
        }
    }
}

long long get_vdac18_values(long value) {
    if (vdac18_map_s.empty()) {
        init_map_vdac18();
    }
    return vdac18_map_s[value];
}
int get_inv_vdac18_values(CCam3BiasEncoding value) {
    if (inv_vdac18_map_s.empty()) {
        init_map_vdac18();
    }
    if (inv_vdac18_map_s.find(value) != inv_vdac18_map_s.end())
        return inv_vdac18_map_s[value];
    return -1;
}

struct CodeGen31IDAC {
    int code_;
    enum Mos { P, N } mos_;
    bool cas_;
};

std::map<int, CodeGen31IDAC> s_p_thin_mvtocode;
std::map<int, CodeGen31IDAC> s_p_thick_mvtocode;
std::map<int, CodeGen31IDAC> s_n_thin_mvtocode;
std::map<int, CodeGen31IDAC> s_n_thick_mvtocode;
std::map<int, CodeGen31IDAC> s_railp_mvtocode;

std::map<CCam3BiasEncoding, int, CCam3BiasEncoding::Cmp_IDac> s_inv_p_thin_mvtocode;
std::map<CCam3BiasEncoding, int, CCam3BiasEncoding::Cmp_IDac> s_inv_p_thick_mvtocode;
std::map<CCam3BiasEncoding, int, CCam3BiasEncoding::Cmp_IDac> s_inv_n_thin_mvtocode;
std::map<CCam3BiasEncoding, int, CCam3BiasEncoding::Cmp_IDac> s_inv_n_thick_mvtocode;
std::map<CCam3BiasEncoding, int, CCam3BiasEncoding::Cmp_IDac> s_inv_railp_mvtocode;

bool load_idac_v_to_code(std::istream &ifs, std::map<int, CodeGen31IDAC> &mvtocode) {
    mvtocode.clear();
    while (ifs) {
        std::string tmp;
        std::getline(ifs, tmp);
        if (tmp.empty()) {
            break;
        }
        std::istringstream istr(tmp);
        int v;
        CodeGen31IDAC codeidac;
        istr >> v;
        istr >> codeidac.code_;
        std::string idac;
        istr >> idac;
        if (idac == "PCas") {
            codeidac.mos_ = CodeGen31IDAC::P;
            codeidac.cas_ = true;
        } else if (idac == "PwoCas") {
            codeidac.mos_ = CodeGen31IDAC::P;
            codeidac.cas_ = false;
        } else if (idac == "NCas") {
            codeidac.mos_ = CodeGen31IDAC::N;
            codeidac.cas_ = true;
        } else if (idac == "NwoCas") {
            codeidac.mos_ = CodeGen31IDAC::N;
            codeidac.cas_ = false;
        } else if (idac == "RailP") {
            codeidac.mos_ = CodeGen31IDAC::N;
            codeidac.cas_ = false;
        } else {
            break;
        }
        mvtocode[v] = codeidac;
    }
    return mvtocode.size() == 1801;
}

bool load_idac_v_to_code(const char *embeded_calib, std::map<int, CodeGen31IDAC> &mvtocode,
                         std::map<CCam3BiasEncoding, int, CCam3BiasEncoding::Cmp_IDac> &inv_map) {
    std::string str(embeded_calib);
    std::istringstream mistr(str);
    if (!load_idac_v_to_code(mistr, mvtocode)) {
        MV_HAL_LOG_ERROR() << "Error loading internal calibration";
        return false;
    }
    for (auto &v : mvtocode) {
        CCam3BiasEncoding c;

        CodeGen31IDAC codeidac = v.second;
        c.current_value_       = codeidac.code_;
        c.voltage_value_       = 0;
        c.type_                = 0;
        c.cas_code_            = !codeidac.cas_;
        c.polarity_            = codeidac.mos_ == CodeGen31IDAC::N;
        c.buffer_value_        = 0x08;

        if (inv_map.find(c) == inv_map.end()) {
            inv_map[c] = v.first;
        }
        c.polarity_ = 1;
        if (inv_map.find(c) == inv_map.end()) {
            inv_map[c] = v.first;
        }
    }

    return true;
}

void init_map_idac() {
    if (s_p_thin_mvtocode.empty()) {
        if (!load_idac_v_to_code(gen3_idac_p_thin, s_p_thin_mvtocode, s_inv_p_thin_mvtocode)) {
            MV_HAL_LOG_ERROR() << "Unable to open gen3_idac_p_thin.calib";
        }
    }
    if (s_n_thin_mvtocode.empty()) {
        if (!load_idac_v_to_code(gen3_idac_n_thin, s_n_thin_mvtocode, s_inv_n_thin_mvtocode)) {
            MV_HAL_LOG_ERROR() << "Unable to open gen3_idac_n_thin.calib";
        }
    }
    if (s_p_thick_mvtocode.empty()) {
        if (!load_idac_v_to_code(gen3_idac_p_thick, s_p_thick_mvtocode, s_inv_p_thick_mvtocode)) {
            MV_HAL_LOG_ERROR() << "Unable to open gen3_idac_p_thick.calib";
        }
    }
    if (s_n_thick_mvtocode.empty()) {
        if (!load_idac_v_to_code(gen3_idac_n_thick, s_n_thick_mvtocode, s_inv_n_thick_mvtocode)) {
            MV_HAL_LOG_ERROR() << "Unable to open gen3_idac_n_thick.calib";
        }
    }
    if (s_railp_mvtocode.empty()) {
        if (!load_idac_v_to_code(gen3_idac_railp, s_railp_mvtocode, s_inv_railp_mvtocode)) {
            MV_HAL_LOG_ERROR() << "Unable to open gen3_idac_railp.calib";
        }
    }
}

int get_inv_idac_values(CCam3BiasEncoding value,
                        const std::map<CCam3BiasEncoding, int, CCam3BiasEncoding::Cmp_IDac> &inv_map) {
    init_map_idac();
    auto it = inv_map.find(value);
    if (it == inv_map.end())
        return -1;
    return it->second;
}

long get_ccam3_gen31_bias_encoding(const Gen31LLBias &bias, int bias_value, bool saturate_value) {
    // if idacsisley an,d current mode we read the encode from a specific file
    init_map_idac();
    std::map<int, CodeGen31IDAC> *mvtocode = NULL;
    switch (bias.get_bias_type()) {
    case Gen31LLBias::BiasType::PMOSThin:
        mvtocode = &s_p_thin_mvtocode;
        break;
    case Gen31LLBias::BiasType::PMOSThick:
        mvtocode = &s_p_thick_mvtocode;
        break;
    case Gen31LLBias::BiasType::NMOSThin:
        mvtocode = &s_n_thin_mvtocode;
        break;
    case Gen31LLBias::BiasType::NMOSThick:
        mvtocode = &s_n_thick_mvtocode;
        break;
    case Gen31LLBias::BiasType::RailP:
        mvtocode = &s_railp_mvtocode;
        break;
    default:
        MV_HAL_LOG_WARNING() << "Unknown bias type";
    }

    BiasEncodingCast encoder;
    encoder.bias_encoding = CCam3BiasEncoding();

    long calib_encoding = 0;
    int v               = bias_value;
    if (saturate_value) {
        if (v < 0)
            v = 0;
        if (v > 1800)
            v = 1800;
    }

    switch (bias.mode()) {
    case Gen31LLBias::Mode::Voltage:
        calib_encoding                       = get_vdac18_values(v);
        encoder.bias_encoding.current_value_ = 0x0;
        encoder.bias_encoding.voltage_value_ = (calib_encoding >> 0) & 0xFF;
        encoder.bias_encoding.type_          = (calib_encoding >> 27) & 1;
        encoder.bias_encoding.cas_code_      = (calib_encoding >> 28) & 1;
        switch (bias.get_bias_type()) {
        case Gen31LLBias::BiasType::PMOSThin:
        case Gen31LLBias::BiasType::PMOSThick:
        case Gen31LLBias::BiasType::RailP:
            encoder.bias_encoding.polarity_ = 0;
            break;
        case Gen31LLBias::BiasType::NMOSThin:
        case Gen31LLBias::BiasType::NMOSThick:
            encoder.bias_encoding.polarity_ = 1;
            break;
        default:
            MV_HAL_LOG_WARNING() << "Unknown bias type";
        }
        break;
    case Gen31LLBias::Mode::Current:
        auto it = mvtocode->find(v);
        if (it != mvtocode->end()) {
            CodeGen31IDAC codeidac               = it->second;
            encoder.bias_encoding.current_value_ = codeidac.code_;
            encoder.bias_encoding.voltage_value_ = 0;
            encoder.bias_encoding.type_          = 0;
            encoder.bias_encoding.cas_code_      = !codeidac.cas_;
            encoder.bias_encoding.polarity_      = codeidac.mos_ == CodeGen31IDAC::N;
            encoder.bias_encoding.buffer_value_  = 0x08;
        } else {
            MV_HAL_LOG_ERROR() << "Err no value for" << v;
            encoder.bias_encoding.voltage_value_ =
                static_cast<long>(round(static_cast<double>(v * 255) / 1800.)) & 0xFF;
            if (v < 900) {
                encoder.bias_encoding.polarity_ = 1;
            } else {
                encoder.bias_encoding.polarity_ = 0;
            }
        }
        break;
    }

    return encoder.raw;
}

} // namespace

namespace Metavision {

Gen31_LL_Biases::Gen31_LL_Biases(const DeviceConfig &device_config, const std::shared_ptr<I_HW_Register> &i_hw_register,
                                 const std::string &prefix) :
    I_LL_Biases(device_config),
    i_hw_register_(i_hw_register),
    base_name_(prefix),
    bypass_range_check_(device_config.biases_range_check_bypass()) {
    if (!i_hw_register_) {
        throw(HalException(PseeHalPluginErrorCode::HWRegisterNotFound, "HW Register facility is null."));
    }
}

bool Gen31_LL_Biases::set_impl(const std::string &bias_name, int bias_value) {
    if (!device_config_.biases_range_check_bypass()) {
        if (bias_name == "bias_diff_on") {
            auto bias_diff = get("bias_diff");
            auto bias_fo   = get("bias_fo");
            auto delta     = 75;
            if (bias_fo < 1350) {
                delta = 95;
            }
            int min_bias_diff_on_value = bias_diff + delta;
            if (bias_value < min_bias_diff_on_value) {
                MV_HAL_LOG_WARNING() << "Current bias_diff_on minimal value is" << min_bias_diff_on_value;
                return false;
            }
        }
        if (bias_name == "bias_diff_off") {
            auto bias_diff = get("bias_diff");
            auto bias_fo   = get("bias_fo");
            auto delta     = 65;
            if (bias_fo < 1350) {
                delta = 85;
            }
            int max_bias_diff_off_value = bias_diff - delta;
            if (bias_value > max_bias_diff_off_value) {
                MV_HAL_LOG_WARNING() << "Current bias_diff_off maximal value is" << max_bias_diff_off_value;
                return false;
            }
        }
        if (bias_name == "bias_refr") {
            auto bias_fo = get("bias_fo");
            if (bias_fo < 1400) {
                constexpr int min_bias_refr_value = 1350;
                if (bias_value < min_bias_refr_value) {
                    MV_HAL_LOG_WARNING() << "Current bias_refr minimal value is" << min_bias_refr_value;
                    return false;
                }
            }
        }
    }

    auto it = get_gen31_biases_map().find(bias_name);
    assert(it != get_gen31_biases_map().end());
    auto &bias_info = it->second;

    long reg = get_ccam3_gen31_bias_encoding(bias_info, bias_value, !bypass_range_check_);
    get_hw_register()->write_register(base_name_ + bias_info.get_register_name(), reg);
    return true;
}

int Gen31_LL_Biases::get_impl(const std::string &bias_name) const {
    auto it = get_gen31_biases_map().find(bias_name);
    assert(it != get_gen31_biases_map().end());
    auto &bias_info = it->second;

    auto r = get_hw_register()->read_register(base_name_ + bias_info.get_register_name());
    if (r == uint32_t(-1))
        return -1;
    BiasEncodingCast encoder;
    encoder.raw = r;
    if (encoder.bias_encoding.type_ == 1) {
        return get_inv_vdac18_values(encoder.bias_encoding);
    } else {
        decltype(s_inv_p_thin_mvtocode) *inv_map = nullptr;
        switch (bias_info.get_bias_type()) {
        case Gen31LLBias::BiasType::PMOSThin:
            inv_map = &s_inv_p_thin_mvtocode;
            break;
        case Gen31LLBias::BiasType::PMOSThick:
            inv_map = &s_inv_p_thick_mvtocode;
            break;
        case Gen31LLBias::BiasType::NMOSThin:
            inv_map = &s_inv_n_thin_mvtocode;
            break;
        case Gen31LLBias::BiasType::NMOSThick:
            inv_map = &s_inv_n_thick_mvtocode;
            break;
        case Gen31LLBias::BiasType::RailP:
            inv_map = &s_inv_railp_mvtocode;
            break;
        default:
            MV_HAL_LOG_ERROR() << "Unknown bias type";
        }

        return get_inv_idac_values(encoder.bias_encoding, *inv_map);
    }
    return -1;
}

bool Gen31_LL_Biases::get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &bias_info) const {
    auto it = get_gen31_biases_map().find(bias_name);
    if (it == get_gen31_biases_map().end()) {
        return false;
    }
    bias_info = it->second;
    return true;
}

std::map<std::string, int> Gen31_LL_Biases::get_all_biases() const {
    std::map<std::string, int> ret;
    for (auto &b : get_gen31_biases_map()) {
        ret[b.first] = get(b.first);
    }
    return ret;
}

const std::shared_ptr<I_HW_Register> &Gen31_LL_Biases::get_hw_register() const {
    return i_hw_register_;
}

} // namespace Metavision
