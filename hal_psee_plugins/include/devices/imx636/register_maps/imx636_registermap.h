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

#ifndef METAVISION_HAL_IMX636_REGISTERMAP_H
#define METAVISION_HAL_IMX636_REGISTERMAP_H

#include "metavision/psee_hw_layer/utils/regmap_data.h"

static RegmapElement Imx636RegisterMap[] = {
    // clang-format off

    {R, {"roi_ctrl", 0x0004}},
    {F, {"roi_td_en", 1, 1, 0x0}},
    {F, {"roi_td_shadow_trigger", 5, 1, 0x0}},
    {F, {"td_roi_roni_n_en", 6, 1, 0x1}},
    {F, {"Reserved_8", 8, 1, 0x0}},
    {F, {"px_td_rstn", 10, 1, 0x0}},
    {F, {"Reserved_17_11", 11, 7, 0xA}},
    {F, {"Reserved_25", 25, 1, 0x0}},
    {F, {"Reserved_29_28", 28, 2, 0x3}},
    {F, {"Reserved_31_30", 30, 2, 0x3}},

    {R, {"lifo_ctrl", 0x000C}},
    {F, {"lifo_en", 0, 1, 0x0}},
    {F, {"lifo_out_en", 1, 1, 0x0}},
    {F, {"lifo_cnt_en", 2, 1, 0x0}},
    {F, {"Reserved_31_3", 3, 29, 0x0}},

    {R, {"lifo_status", 0x0010}},
    {F, {"lifo_ton", 0, 29, 0x0}},
    {F, {"lifo_ton_valid", 29, 1, 0x0}},
    {F, {"Reserved_30", 30, 1, 0x0}},

    {R, {"Reserved_0014", 0x0014}},
    {F, {"Reserved_31_0", 0, 32, 0xA0401806}},

    {R, {"spare0", 0x0018}},
    {F, {"Reserved_19_0", 0, 20, 0x0}},
    {F, {"gcd_rstn", 20, 1, 0x0}},
    {F, {"Reserved_31_21", 21, 11, 0x0}},

    {R, {"refractory_ctrl", 0x0020}},
    {F, {"refr_counter", 0, 28, 0x0}},
    {F, {"refr_valid", 28, 1, 0x0}},
    {F, {"Reserved_29", 29, 1, 0x0}},
    {F, {"refr_cnt_en", 30, 1, 0x0}},
    {F, {"refr_en", 31, 1, 0x0}},

    {R, {"roi_win_ctrl", 0x0034}},
    {F, {"roi_master_en", 0, 1, 0x0}},
    {F, {"roi_win_done", 1, 1, 0x0}},

    {R, {"roi_win_start_addr", 0x0038}},
    {F, {"roi_win_start_x", 0, 11, 0x0}},
    {F, {"roi_win_start_y", 16, 10, 0x0}},

    {R, {"roi_win_end_addr", 0x003C}},
    {F, {"roi_win_end_x", 0, 11, 0x4FF}},
    {F, {"roi_win_end_y", 16, 10, 0x2CF}},

    {R, {"dig_pad2_ctrl", 0x0044}},
    {F, {"Reserved_15_0", 0, 16, 0xFCCF}},
    {F, {"pad_sync", 16, 4, 0xF}},
    {F, {"Reserved_31_20", 20, 12, 0xCCF}},

    {R, {"adc_control", 0x004C}},
    {F, {"adc_en", 0, 1, 0x0}},
    {F, {"adc_clk_en", 1, 1, 0x0}},
    {F, {"adc_start", 2, 1, 0x0}},
    {F, {"Reserved_31_3", 3, 29, 0xEC8}},

    {R, {"adc_status", 0x0050}},
    {F, {"adc_dac_dyn", 0, 10, 0x0}},
    {F, {"Reserved_10", 10, 1, 0x0}},
    {F, {"adc_done_dyn", 11, 1, 0x0}},
    {F, {"Reserved_31_12", 12, 20, 0x0}},

    {R, {"adc_misc_ctrl", 0x0054}},
    {F, {"Reserved_0", 0, 1, 0x0}},
    {F, {"adc_buf_cal_en", 1, 1, 0x0}},
    {F, {"Reserved_9_2", 2, 8, 0x84}},
    {F, {"adc_rng", 10, 2, 0x0}},
    {F, {"adc_temp", 12, 1, 0x0}},
    {F, {"Reserved_14_13", 13, 2, 0x0}},

    {R, {"temp_ctrl", 0x005C}},
    {F, {"temp_buf_cal_en", 0, 1, 0x0}},
    {F, {"temp_buf_en", 1, 1, 0x0}},
    {F, {"Reserved_31_2", 2, 30, 0x20}},

    {R, {"iph_mirr_ctrl", 0x0074}},
    {F, {"iph_mirr_en", 0, 1, 0x0}},
    {F, {"iph_mirr_amp_en", 1, 1, 0x1}},
    {F, {"Reserved_31_2", 2, 30, 0x0}},

    {R, {"gcd_ctrl1", 0x0078}},
    {F, {"gcd_en", 0, 1, 0x0}},
    {F, {"gcd_diffamp_en", 1, 1, 0x0}},
    {F, {"gcd_lpf_en", 2, 1, 0x0}},
    {F, {"Reserved_31_3", 3, 29, 0x8003BE9}},

    {R, {"reqy_qmon_ctrl", 0x0088}},
    {F, {"reqy_qmon_en", 0, 1, 0x0}},
    {F, {"reqy_qmon_rstn", 1, 1, 0x0}},
    {F, {"Reserved_3_2", 2, 2, 0x0}},
    {F, {"reqy_qmon_interrupt_en", 4, 1, 0x0}},
    {F, {"reqy_qmon_trip_ctl", 10, 10, 0x0}},
    {F, {"Reserved_31_16", 20, 12, 0x0}},

    {R, {"reqy_qmon_status", 0x008C}},
    {F, {"Reserved_15_0", 0, 16, 0x0}},
    {F, {"reqy_qmon_sum_irq", 16, 10, 0x0}},
    {F, {"reqy_qmon_trip_irq", 26, 1, 0x0}},

    {R, {"gcd_shadow_ctrl", 0x0090}},
    {F, {"Reserved_0", 0, 1, 0x0}},
    {F, {"gcd_irq_sw_override", 1, 1, 0x0}},
    {F, {"gcd_reset_on_copy", 2, 1, 0x0}},

    {R, {"gcd_shadow_status", 0x0094}},
    {F, {"gcd_shadow_valid", 0, 1, 0x0}},
    {F, {"Reserved_31_1", 1, 31, 0x0}},

    {R, {"gcd_shadow_counter", 0x0098}},
    {F, {"gcd_shadow_cnt_off", 0, 16, 0x0}},
    {F, {"gcd_shadow_cnt_on", 16, 16, 0x0}},

    {R, {"stop_sequence_control", 0x00C8}},
    {F, {"stop_sequence_start", 0, 1, 0x0}},
    {F, {"Reserved_15_8", 8, 8, 0x1}},

    {R, {"bias/bias_fo", 0x1004}},
    {F, {"idac_ctl", 0, 8, 0x0}},
    {F, {"Reserved_27_8", 8, 20, 0x3A1E8}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"bias/bias_hpf", 0x100C}},
    {F, {"idac_ctl", 0, 8, 0x0}},
    {F, {"Reserved_27_8", 8, 20, 0x3A1FF}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"bias/bias_diff_on", 0x1010}},
    {F, {"idac_ctl", 0, 8, 0x0}},
    {F, {"Reserved_27_8", 8, 20, 0x1A163}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"bias/bias_diff", 0x1014}},
    {F, {"idac_ctl", 0, 8, 0x4D}},
    {F, {"Reserved_27_8", 8, 20, 0x1A150}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"bias/bias_diff_off", 0x1018}},
    {F, {"idac_ctl", 0, 8, 0x0}},
    {F, {"Reserved_27_8", 8, 20, 0x1A137}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"bias/bias_refr", 0x1020}},
    {F, {"idac_ctl", 0, 8, 0x14}},
    {F, {"Reserved_27_8", 8, 20, 0x38296}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"bias/bgen_ctrl", 0x1100}},
    {F, {"burst_transfer", 0, 1, 0x0}},
    {F, {"Reserved_2_1", 1, 2, 0x0}},

    {R, {"roi/td_roi_x00", 0x2000}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x01", 0x2004}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x02", 0x2008}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x03", 0x200C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x04", 0x2010}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x05", 0x2014}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x06", 0x2018}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x07", 0x201C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x08", 0x2020}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x09", 0x2024}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x10", 0x2028}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x11", 0x202C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x12", 0x2030}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x13", 0x2034}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x14", 0x2038}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x15", 0x203C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x16", 0x2040}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x17", 0x2044}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x18", 0x2048}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x19", 0x204C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x20", 0x2050}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x21", 0x2054}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x22", 0x2058}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x23", 0x205C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x24", 0x2060}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x25", 0x2064}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x26", 0x2068}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x27", 0x206C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x28", 0x2070}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x29", 0x2074}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x30", 0x2078}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x31", 0x207C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x32", 0x2080}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x33", 0x2084}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x34", 0x2088}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x35", 0x208C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x36", 0x2090}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x37", 0x2094}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x38", 0x2098}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_x39", 0x209C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y00", 0x4000}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y01", 0x4004}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y02", 0x4008}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y03", 0x400C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y04", 0x4010}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y05", 0x4014}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y06", 0x4018}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y07", 0x401C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y08", 0x4020}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y09", 0x4024}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y10", 0x4028}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y11", 0x402C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y12", 0x4030}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y13", 0x4034}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y14", 0x4038}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y15", 0x403C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y16", 0x4040}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y17", 0x4044}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y18", 0x4048}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y19", 0x404C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y20", 0x4050}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y21", 0x4054}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"roi/td_roi_y22", 0x4058}},
    {F, {"effective", 0, 16, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFF}},
    {F, {"Reserved_16", 16, 1, 0x1}},
    {F, {"Reserved_17", 17, 1, 0x1}},
    {F, {"Reserved_19_18", 18, 2, 0x3}},
    {F, {"Reserved_21_20", 20, 2, 0x3}},
    {F, {"Reserved_22", 22, 1, 0x1}},
    {F, {"Reserved_23", 23, 1, 0x1}},

    {R, {"erc/Reserved_6000", 0x6000}},
    {F, {"Reserved_1_0", 0, 2, 0x0}},

    {R, {"erc/in_drop_rate_control", 0x6004}},
    {F, {"cfg_event_delay_fifo_en", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"Reserved_10_2", 2, 9, 0x0}},

    {R, {"erc/reference_period", 0x6008}},
    {F, {"erc_reference_period", 0, 10, 0x80}},

    {R, {"erc/td_target_event_rate", 0x600C}},
    {F, {"target_event_rate", 0, 22, 0x80}},

    {R, {"erc/erc_enable", 0x6028}},
    {F, {"erc_en", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"Reserved_2", 2, 1, 0x0}},

    {R, {"erc/Reserved_602C", 0x602C}},
    {F, {"Reserved_0", 0, 1, 0x0}},

    {R, {"erc/t_dropping_control", 0x6050}},
    {F, {"t_dropping_en", 0, 1, 0x0}},

    {R, {"erc/h_dropping_control", 0x6060}},
    {F, {"h_dropping_en", 0, 1, 0x0}},

    {R, {"erc/v_dropping_control", 0x6070}},
    {F, {"v_dropping_en", 0, 1, 0x0}},

    {R, {"erc/h_drop_lut_00", 0x6080}},
    {F, {"hlut00", 0, 5, 0x0}},
    {F, {"hlut01", 8, 5, 0x0}},
    {F, {"hlut02", 16, 5, 0x0}},
    {F, {"hlut03", 24, 5, 0x0}},

    {R, {"erc/h_drop_lut_01", 0x6084}},
    {F, {"hlut04", 0, 5, 0x0}},
    {F, {"hlut05", 8, 5, 0x0}},
    {F, {"hlut06", 16, 5, 0x0}},
    {F, {"hlut07", 24, 5, 0x0}},

    {R, {"erc/h_drop_lut_02", 0x6088}},
    {F, {"hlut08", 0, 5, 0x0}},
    {F, {"hlut09", 8, 5, 0x0}},
    {F, {"hlut10", 16, 5, 0x0}},
    {F, {"hlut11", 24, 5, 0x0}},

    {R, {"erc/h_drop_lut_03", 0x608C}},
    {F, {"hlut12", 0, 5, 0x0}},
    {F, {"hlut13", 8, 5, 0x0}},
    {F, {"hlut14", 16, 5, 0x0}},
    {F, {"hlut15", 24, 5, 0x0}},

    {R, {"erc/h_drop_lut_04", 0x6090}},
    {F, {"hlut16", 0, 5, 0x0}},
    {F, {"hlut17", 8, 5, 0x0}},
    {F, {"hlut18", 16, 5, 0x0}},
    {F, {"hlut19", 24, 5, 0x0}},

    {R, {"erc/h_drop_lut_05", 0x6094}},
    {F, {"hlut20", 0, 5, 0x0}},
    {F, {"hlut21", 8, 5, 0x0}},
    {F, {"hlut22", 16, 5, 0x0}},
    {F, {"hlut23", 24, 5, 0x0}},

    {R, {"erc/h_drop_lut_06", 0x6098}},
    {F, {"hlut24", 0, 5, 0x0}},
    {F, {"hlut25", 8, 5, 0x0}},
    {F, {"hlut26", 16, 5, 0x0}},
    {F, {"hlut27", 24, 5, 0x0}},

    {R, {"erc/h_drop_lut_07", 0x609C}},
    {F, {"hlut28", 0, 5, 0x0}},
    {F, {"hlut29", 8, 5, 0x0}},
    {F, {"hlut30", 16, 5, 0x0}},
    {F, {"hlut31", 24, 5, 0x0}},

    {R, {"erc/v_drop_lut_00", 0x60C0}},
    {F, {"vlut00", 0, 5, 0x0}},
    {F, {"vlut01", 8, 5, 0x0}},
    {F, {"vlut02", 16, 5, 0x0}},
    {F, {"vlut03", 24, 5, 0x0}},

    {R, {"erc/v_drop_lut_01", 0x60C4}},
    {F, {"vlut04", 0, 5, 0x0}},
    {F, {"vlut05", 8, 5, 0x0}},
    {F, {"vlut06", 16, 5, 0x0}},
    {F, {"vlut07", 24, 5, 0x0}},

    {R, {"erc/v_drop_lut_02", 0x60C8}},
    {F, {"vlut08", 0, 5, 0x0}},
    {F, {"vlut09", 8, 5, 0x0}},
    {F, {"vlut10", 16, 5, 0x0}},
    {F, {"vlut11", 24, 5, 0x0}},

    {R, {"erc/v_drop_lut_03", 0x60CC}},
    {F, {"vlut12", 0, 5, 0x0}},
    {F, {"vlut13", 8, 5, 0x0}},
    {F, {"vlut14", 16, 5, 0x0}},
    {F, {"vlut15", 24, 5, 0x0}},

    {R, {"erc/v_drop_lut_04", 0x60D0}},
    {F, {"vlut16", 0, 5, 0x0}},
    {F, {"vlut17", 8, 5, 0x0}},
    {F, {"vlut18", 16, 5, 0x0}},
    {F, {"vlut19", 24, 5, 0x0}},

    {R, {"erc/v_drop_lut_05", 0x60D4}},
    {F, {"vlut20", 0, 5, 0x0}},
    {F, {"vlut21", 8, 5, 0x0}},
    {F, {"vlut22", 16, 5, 0x0}},
    {F, {"vlut23", 24, 5, 0x0}},

    {R, {"erc/v_drop_lut_06", 0x60D8}},
    {F, {"vlut24", 0, 5, 0x0}},
    {F, {"vlut25", 8, 5, 0x0}},
    {F, {"vlut26", 16, 5, 0x0}},
    {F, {"vlut27", 24, 5, 0x0}},

    {R, {"erc/v_drop_lut_07", 0x60DC}},
    {F, {"vlut28", 0, 5, 0x0}},
    {F, {"vlut29", 8, 5, 0x0}},
    {F, {"vlut30", 16, 5, 0x0}},
    {F, {"vlut31", 24, 5, 0x0}},

    {R, {"erc/t_drop_lut_00", 0x6400}},
    {F, {"tlut000", 0, 9, 0x0}},
    {F, {"tlut001", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_01", 0x6404}},
    {F, {"tlut002", 0, 9, 0x0}},
    {F, {"tlut003", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_02", 0x6408}},
    {F, {"tlut004", 0, 9, 0x0}},
    {F, {"tlut005", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_03", 0x640C}},
    {F, {"tlut006", 0, 9, 0x0}},
    {F, {"tlut007", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_04", 0x6410}},
    {F, {"tlut008", 0, 9, 0x0}},
    {F, {"tlut009", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_05", 0x6414}},
    {F, {"tlut010", 0, 9, 0x0}},
    {F, {"tlut011", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_06", 0x6418}},
    {F, {"tlut012", 0, 9, 0x0}},
    {F, {"tlut013", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_07", 0x641C}},
    {F, {"tlut014", 0, 9, 0x0}},
    {F, {"tlut015", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_08", 0x6420}},
    {F, {"tlut016", 0, 9, 0x0}},
    {F, {"tlut017", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_09", 0x6424}},
    {F, {"tlut018", 0, 9, 0x0}},
    {F, {"tlut019", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_10", 0x6428}},
    {F, {"tlut020", 0, 9, 0x0}},
    {F, {"tlut021", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_11", 0x642C}},
    {F, {"tlut022", 0, 9, 0x0}},
    {F, {"tlut023", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_12", 0x6430}},
    {F, {"tlut024", 0, 9, 0x0}},
    {F, {"tlut025", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_13", 0x6434}},
    {F, {"tlut026", 0, 9, 0x0}},
    {F, {"tlut027", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_14", 0x6438}},
    {F, {"tlut028", 0, 9, 0x0}},
    {F, {"tlut029", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_15", 0x643C}},
    {F, {"tlut030", 0, 9, 0x0}},
    {F, {"tlut031", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_16", 0x6440}},
    {F, {"tlut032", 0, 9, 0x0}},
    {F, {"tlut033", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_17", 0x6444}},
    {F, {"tlut034", 0, 9, 0x0}},
    {F, {"tlut035", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_18", 0x6448}},
    {F, {"tlut036", 0, 9, 0x0}},
    {F, {"tlut037", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_19", 0x644C}},
    {F, {"tlut038", 0, 9, 0x0}},
    {F, {"tlut039", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_20", 0x6450}},
    {F, {"tlut040", 0, 9, 0x0}},
    {F, {"tlut041", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_21", 0x6454}},
    {F, {"tlut042", 0, 9, 0x0}},
    {F, {"tlut043", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_22", 0x6458}},
    {F, {"tlut044", 0, 9, 0x0}},
    {F, {"tlut045", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_23", 0x645C}},
    {F, {"tlut046", 0, 9, 0x0}},
    {F, {"tlut047", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_24", 0x6460}},
    {F, {"tlut048", 0, 9, 0x0}},
    {F, {"tlut049", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_25", 0x6464}},
    {F, {"tlut050", 0, 9, 0x0}},
    {F, {"tlut051", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_26", 0x6468}},
    {F, {"tlut052", 0, 9, 0x0}},
    {F, {"tlut053", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_27", 0x646C}},
    {F, {"tlut054", 0, 9, 0x0}},
    {F, {"tlut055", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_28", 0x6470}},
    {F, {"tlut056", 0, 9, 0x0}},
    {F, {"tlut057", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_29", 0x6474}},
    {F, {"tlut058", 0, 9, 0x0}},
    {F, {"tlut059", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_30", 0x6478}},
    {F, {"tlut060", 0, 9, 0x0}},
    {F, {"tlut061", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_31", 0x647C}},
    {F, {"tlut062", 0, 9, 0x0}},
    {F, {"tlut063", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_32", 0x6480}},
    {F, {"tlut064", 0, 9, 0x0}},
    {F, {"tlut065", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_33", 0x6484}},
    {F, {"tlut066", 0, 9, 0x0}},
    {F, {"tlut067", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_34", 0x6488}},
    {F, {"tlut068", 0, 9, 0x0}},
    {F, {"tlut069", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_35", 0x648C}},
    {F, {"tlut070", 0, 9, 0x0}},
    {F, {"tlut071", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_36", 0x6490}},
    {F, {"tlut072", 0, 9, 0x0}},
    {F, {"tlut073", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_37", 0x6494}},
    {F, {"tlut074", 0, 9, 0x0}},
    {F, {"tlut075", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_38", 0x6498}},
    {F, {"tlut076", 0, 9, 0x0}},
    {F, {"tlut077", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_39", 0x649C}},
    {F, {"tlut078", 0, 9, 0x0}},
    {F, {"tlut079", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_40", 0x64A0}},
    {F, {"tlut080", 0, 9, 0x0}},
    {F, {"tlut081", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_41", 0x64A4}},
    {F, {"tlut082", 0, 9, 0x0}},
    {F, {"tlut083", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_42", 0x64A8}},
    {F, {"tlut084", 0, 9, 0x0}},
    {F, {"tlut085", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_43", 0x64AC}},
    {F, {"tlut086", 0, 9, 0x0}},
    {F, {"tlut087", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_44", 0x64B0}},
    {F, {"tlut088", 0, 9, 0x0}},
    {F, {"tlut089", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_45", 0x64B4}},
    {F, {"tlut090", 0, 9, 0x0}},
    {F, {"tlut091", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_46", 0x64B8}},
    {F, {"tlut092", 0, 9, 0x0}},
    {F, {"tlut093", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_47", 0x64BC}},
    {F, {"tlut094", 0, 9, 0x0}},
    {F, {"tlut095", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_48", 0x64C0}},
    {F, {"tlut096", 0, 9, 0x0}},
    {F, {"tlut097", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_49", 0x64C4}},
    {F, {"tlut098", 0, 9, 0x0}},
    {F, {"tlut099", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_50", 0x64C8}},
    {F, {"tlut100", 0, 9, 0x0}},
    {F, {"tlut101", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_51", 0x64CC}},
    {F, {"tlut102", 0, 9, 0x0}},
    {F, {"tlut103", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_52", 0x64D0}},
    {F, {"tlut104", 0, 9, 0x0}},
    {F, {"tlut105", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_53", 0x64D4}},
    {F, {"tlut106", 0, 9, 0x0}},
    {F, {"tlut107", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_54", 0x64D8}},
    {F, {"tlut108", 0, 9, 0x0}},
    {F, {"tlut109", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_55", 0x64DC}},
    {F, {"tlut110", 0, 9, 0x0}},
    {F, {"tlut111", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_56", 0x64E0}},
    {F, {"tlut112", 0, 9, 0x0}},
    {F, {"tlut113", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_57", 0x64E4}},
    {F, {"tlut114", 0, 9, 0x0}},
    {F, {"tlut115", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_58", 0x64E8}},
    {F, {"tlut116", 0, 9, 0x0}},
    {F, {"tlut117", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_59", 0x64EC}},
    {F, {"tlut118", 0, 9, 0x0}},
    {F, {"tlut119", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_60", 0x64F0}},
    {F, {"tlut120", 0, 9, 0x0}},
    {F, {"tlut121", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_61", 0x64F4}},
    {F, {"tlut122", 0, 9, 0x0}},
    {F, {"tlut123", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_62", 0x64F8}},
    {F, {"tlut124", 0, 9, 0x0}},
    {F, {"tlut125", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_63", 0x64FC}},
    {F, {"tlut126", 0, 9, 0x0}},
    {F, {"tlut127", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_64", 0x6500}},
    {F, {"tlut128", 0, 9, 0x0}},
    {F, {"tlut129", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_65", 0x6504}},
    {F, {"tlut130", 0, 9, 0x0}},
    {F, {"tlut131", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_66", 0x6508}},
    {F, {"tlut132", 0, 9, 0x0}},
    {F, {"tlut133", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_67", 0x650C}},
    {F, {"tlut134", 0, 9, 0x0}},
    {F, {"tlut135", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_68", 0x6510}},
    {F, {"tlut136", 0, 9, 0x0}},
    {F, {"tlut137", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_69", 0x6514}},
    {F, {"tlut138", 0, 9, 0x0}},
    {F, {"tlut139", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_70", 0x6518}},
    {F, {"tlut140", 0, 9, 0x0}},
    {F, {"tlut141", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_71", 0x651C}},
    {F, {"tlut142", 0, 9, 0x0}},
    {F, {"tlut143", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_72", 0x6520}},
    {F, {"tlut144", 0, 9, 0x0}},
    {F, {"tlut145", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_73", 0x6524}},
    {F, {"tlut146", 0, 9, 0x0}},
    {F, {"tlut147", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_74", 0x6528}},
    {F, {"tlut148", 0, 9, 0x0}},
    {F, {"tlut149", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_75", 0x652C}},
    {F, {"tlut150", 0, 9, 0x0}},
    {F, {"tlut151", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_76", 0x6530}},
    {F, {"tlut152", 0, 9, 0x0}},
    {F, {"tlut153", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_77", 0x6534}},
    {F, {"tlut154", 0, 9, 0x0}},
    {F, {"tlut155", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_78", 0x6538}},
    {F, {"tlut156", 0, 9, 0x0}},
    {F, {"tlut157", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_79", 0x653C}},
    {F, {"tlut158", 0, 9, 0x0}},
    {F, {"tlut159", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_80", 0x6540}},
    {F, {"tlut160", 0, 9, 0x0}},
    {F, {"tlut161", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_81", 0x6544}},
    {F, {"tlut162", 0, 9, 0x0}},
    {F, {"tlut163", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_82", 0x6548}},
    {F, {"tlut164", 0, 9, 0x0}},
    {F, {"tlut165", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_83", 0x654C}},
    {F, {"tlut166", 0, 9, 0x0}},
    {F, {"tlut167", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_84", 0x6550}},
    {F, {"tlut168", 0, 9, 0x0}},
    {F, {"tlut169", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_85", 0x6554}},
    {F, {"tlut170", 0, 9, 0x0}},
    {F, {"tlut171", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_86", 0x6558}},
    {F, {"tlut172", 0, 9, 0x0}},
    {F, {"tlut173", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_87", 0x655C}},
    {F, {"tlut174", 0, 9, 0x0}},
    {F, {"tlut175", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_88", 0x6560}},
    {F, {"tlut176", 0, 9, 0x0}},
    {F, {"tlut177", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_89", 0x6564}},
    {F, {"tlut178", 0, 9, 0x0}},
    {F, {"tlut179", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_90", 0x6568}},
    {F, {"tlut180", 0, 9, 0x0}},
    {F, {"tlut181", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_91", 0x656C}},
    {F, {"tlut182", 0, 9, 0x0}},
    {F, {"tlut183", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_92", 0x6570}},
    {F, {"tlut184", 0, 9, 0x0}},
    {F, {"tlut185", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_93", 0x6574}},
    {F, {"tlut186", 0, 9, 0x0}},
    {F, {"tlut187", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_94", 0x6578}},
    {F, {"tlut188", 0, 9, 0x0}},
    {F, {"tlut189", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_95", 0x657C}},
    {F, {"tlut190", 0, 9, 0x0}},
    {F, {"tlut191", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_96", 0x6580}},
    {F, {"tlut192", 0, 9, 0x0}},
    {F, {"tlut193", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_97", 0x6584}},
    {F, {"tlut194", 0, 9, 0x0}},
    {F, {"tlut195", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_98", 0x6588}},
    {F, {"tlut196", 0, 9, 0x0}},
    {F, {"tlut197", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_99", 0x658C}},
    {F, {"tlut198", 0, 9, 0x0}},
    {F, {"tlut199", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_100", 0x6590}},
    {F, {"tlut200", 0, 9, 0x0}},
    {F, {"tlut201", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_101", 0x6594}},
    {F, {"tlut202", 0, 9, 0x0}},
    {F, {"tlut203", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_102", 0x6598}},
    {F, {"tlut204", 0, 9, 0x0}},
    {F, {"tlut205", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_103", 0x659C}},
    {F, {"tlut206", 0, 9, 0x0}},
    {F, {"tlut207", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_104", 0x65A0}},
    {F, {"tlut208", 0, 9, 0x0}},
    {F, {"tlut209", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_105", 0x65A4}},
    {F, {"tlut210", 0, 9, 0x0}},
    {F, {"tlut211", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_106", 0x65A8}},
    {F, {"tlut212", 0, 9, 0x0}},
    {F, {"tlut213", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_107", 0x65AC}},
    {F, {"tlut214", 0, 9, 0x0}},
    {F, {"tlut215", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_108", 0x65B0}},
    {F, {"tlut216", 0, 9, 0x0}},
    {F, {"tlut217", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_109", 0x65B4}},
    {F, {"tlut218", 0, 9, 0x0}},
    {F, {"tlut219", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_110", 0x65B8}},
    {F, {"tlut220", 0, 9, 0x0}},
    {F, {"tlut221", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_111", 0x65BC}},
    {F, {"tlut222", 0, 9, 0x0}},
    {F, {"tlut223", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_112", 0x65C0}},
    {F, {"tlut224", 0, 9, 0x0}},
    {F, {"tlut225", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_113", 0x65C4}},
    {F, {"tlut226", 0, 9, 0x0}},
    {F, {"tlut227", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_114", 0x65C8}},
    {F, {"tlut228", 0, 9, 0x0}},
    {F, {"tlut229", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_115", 0x65CC}},
    {F, {"tlut230", 0, 9, 0x0}},
    {F, {"tlut231", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_116", 0x65D0}},
    {F, {"tlut232", 0, 9, 0x0}},
    {F, {"tlut233", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_117", 0x65D4}},
    {F, {"tlut234", 0, 9, 0x0}},
    {F, {"tlut235", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_118", 0x65D8}},
    {F, {"tlut236", 0, 9, 0x0}},
    {F, {"tlut237", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_119", 0x65DC}},
    {F, {"tlut238", 0, 9, 0x0}},
    {F, {"tlut239", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_120", 0x65E0}},
    {F, {"tlut240", 0, 9, 0x0}},
    {F, {"tlut241", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_121", 0x65E4}},
    {F, {"tlut242", 0, 9, 0x0}},
    {F, {"tlut243", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_122", 0x65E8}},
    {F, {"tlut244", 0, 9, 0x0}},
    {F, {"tlut245", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_123", 0x65EC}},
    {F, {"tlut246", 0, 9, 0x0}},
    {F, {"tlut247", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_124", 0x65F0}},
    {F, {"tlut248", 0, 9, 0x0}},
    {F, {"tlut249", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_125", 0x65F4}},
    {F, {"tlut250", 0, 9, 0x0}},
    {F, {"tlut251", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_126", 0x65F8}},
    {F, {"tlut252", 0, 9, 0x0}},
    {F, {"tlut253", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_127", 0x65FC}},
    {F, {"tlut254", 0, 9, 0x0}},
    {F, {"tlut255", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_128", 0x6600}},
    {F, {"tlut256", 0, 9, 0x0}},
    {F, {"tlut257", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_129", 0x6604}},
    {F, {"tlut258", 0, 9, 0x0}},
    {F, {"tlut259", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_130", 0x6608}},
    {F, {"tlut260", 0, 9, 0x0}},
    {F, {"tlut261", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_131", 0x660C}},
    {F, {"tlut262", 0, 9, 0x0}},
    {F, {"tlut263", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_132", 0x6610}},
    {F, {"tlut264", 0, 9, 0x0}},
    {F, {"tlut265", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_133", 0x6614}},
    {F, {"tlut266", 0, 9, 0x0}},
    {F, {"tlut267", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_134", 0x6618}},
    {F, {"tlut268", 0, 9, 0x0}},
    {F, {"tlut269", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_135", 0x661C}},
    {F, {"tlut270", 0, 9, 0x0}},
    {F, {"tlut271", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_136", 0x6620}},
    {F, {"tlut272", 0, 9, 0x0}},
    {F, {"tlut273", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_137", 0x6624}},
    {F, {"tlut274", 0, 9, 0x0}},
    {F, {"tlut275", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_138", 0x6628}},
    {F, {"tlut276", 0, 9, 0x0}},
    {F, {"tlut277", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_139", 0x662C}},
    {F, {"tlut278", 0, 9, 0x0}},
    {F, {"tlut279", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_140", 0x6630}},
    {F, {"tlut280", 0, 9, 0x0}},
    {F, {"tlut281", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_141", 0x6634}},
    {F, {"tlut282", 0, 9, 0x0}},
    {F, {"tlut283", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_142", 0x6638}},
    {F, {"tlut284", 0, 9, 0x0}},
    {F, {"tlut285", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_143", 0x663C}},
    {F, {"tlut286", 0, 9, 0x0}},
    {F, {"tlut287", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_144", 0x6640}},
    {F, {"tlut288", 0, 9, 0x0}},
    {F, {"tlut289", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_145", 0x6644}},
    {F, {"tlut290", 0, 9, 0x0}},
    {F, {"tlut291", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_146", 0x6648}},
    {F, {"tlut292", 0, 9, 0x0}},
    {F, {"tlut293", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_147", 0x664C}},
    {F, {"tlut294", 0, 9, 0x0}},
    {F, {"tlut295", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_148", 0x6650}},
    {F, {"tlut296", 0, 9, 0x0}},
    {F, {"tlut297", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_149", 0x6654}},
    {F, {"tlut298", 0, 9, 0x0}},
    {F, {"tlut299", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_150", 0x6658}},
    {F, {"tlut300", 0, 9, 0x0}},
    {F, {"tlut301", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_151", 0x665C}},
    {F, {"tlut302", 0, 9, 0x0}},
    {F, {"tlut303", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_152", 0x6660}},
    {F, {"tlut304", 0, 9, 0x0}},
    {F, {"tlut305", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_153", 0x6664}},
    {F, {"tlut306", 0, 9, 0x0}},
    {F, {"tlut307", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_154", 0x6668}},
    {F, {"tlut308", 0, 9, 0x0}},
    {F, {"tlut309", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_155", 0x666C}},
    {F, {"tlut310", 0, 9, 0x0}},
    {F, {"tlut311", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_156", 0x6670}},
    {F, {"tlut312", 0, 9, 0x0}},
    {F, {"tlut313", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_157", 0x6674}},
    {F, {"tlut314", 0, 9, 0x0}},
    {F, {"tlut315", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_158", 0x6678}},
    {F, {"tlut316", 0, 9, 0x0}},
    {F, {"tlut317", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_159", 0x667C}},
    {F, {"tlut318", 0, 9, 0x0}},
    {F, {"tlut319", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_160", 0x6680}},
    {F, {"tlut320", 0, 9, 0x0}},
    {F, {"tlut321", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_161", 0x6684}},
    {F, {"tlut322", 0, 9, 0x0}},
    {F, {"tlut323", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_162", 0x6688}},
    {F, {"tlut324", 0, 9, 0x0}},
    {F, {"tlut325", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_163", 0x668C}},
    {F, {"tlut326", 0, 9, 0x0}},
    {F, {"tlut327", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_164", 0x6690}},
    {F, {"tlut328", 0, 9, 0x0}},
    {F, {"tlut329", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_165", 0x6694}},
    {F, {"tlut330", 0, 9, 0x0}},
    {F, {"tlut331", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_166", 0x6698}},
    {F, {"tlut332", 0, 9, 0x0}},
    {F, {"tlut333", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_167", 0x669C}},
    {F, {"tlut334", 0, 9, 0x0}},
    {F, {"tlut335", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_168", 0x66A0}},
    {F, {"tlut336", 0, 9, 0x0}},
    {F, {"tlut337", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_169", 0x66A4}},
    {F, {"tlut338", 0, 9, 0x0}},
    {F, {"tlut339", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_170", 0x66A8}},
    {F, {"tlut340", 0, 9, 0x0}},
    {F, {"tlut341", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_171", 0x66AC}},
    {F, {"tlut342", 0, 9, 0x0}},
    {F, {"tlut343", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_172", 0x66B0}},
    {F, {"tlut344", 0, 9, 0x0}},
    {F, {"tlut345", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_173", 0x66B4}},
    {F, {"tlut346", 0, 9, 0x0}},
    {F, {"tlut347", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_174", 0x66B8}},
    {F, {"tlut348", 0, 9, 0x0}},
    {F, {"tlut349", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_175", 0x66BC}},
    {F, {"tlut350", 0, 9, 0x0}},
    {F, {"tlut351", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_176", 0x66C0}},
    {F, {"tlut352", 0, 9, 0x0}},
    {F, {"tlut353", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_177", 0x66C4}},
    {F, {"tlut354", 0, 9, 0x0}},
    {F, {"tlut355", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_178", 0x66C8}},
    {F, {"tlut356", 0, 9, 0x0}},
    {F, {"tlut357", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_179", 0x66CC}},
    {F, {"tlut358", 0, 9, 0x0}},
    {F, {"tlut359", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_180", 0x66D0}},
    {F, {"tlut360", 0, 9, 0x0}},
    {F, {"tlut361", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_181", 0x66D4}},
    {F, {"tlut362", 0, 9, 0x0}},
    {F, {"tlut363", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_182", 0x66D8}},
    {F, {"tlut364", 0, 9, 0x0}},
    {F, {"tlut365", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_183", 0x66DC}},
    {F, {"tlut366", 0, 9, 0x0}},
    {F, {"tlut367", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_184", 0x66E0}},
    {F, {"tlut368", 0, 9, 0x0}},
    {F, {"tlut369", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_185", 0x66E4}},
    {F, {"tlut370", 0, 9, 0x0}},
    {F, {"tlut371", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_186", 0x66E8}},
    {F, {"tlut372", 0, 9, 0x0}},
    {F, {"tlut373", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_187", 0x66EC}},
    {F, {"tlut374", 0, 9, 0x0}},
    {F, {"tlut375", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_188", 0x66F0}},
    {F, {"tlut376", 0, 9, 0x0}},
    {F, {"tlut377", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_189", 0x66F4}},
    {F, {"tlut378", 0, 9, 0x0}},
    {F, {"tlut379", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_190", 0x66F8}},
    {F, {"tlut380", 0, 9, 0x0}},
    {F, {"tlut381", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_191", 0x66FC}},
    {F, {"tlut382", 0, 9, 0x0}},
    {F, {"tlut383", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_192", 0x6700}},
    {F, {"tlut384", 0, 9, 0x0}},
    {F, {"tlut385", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_193", 0x6704}},
    {F, {"tlut386", 0, 9, 0x0}},
    {F, {"tlut387", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_194", 0x6708}},
    {F, {"tlut388", 0, 9, 0x0}},
    {F, {"tlut389", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_195", 0x670C}},
    {F, {"tlut390", 0, 9, 0x0}},
    {F, {"tlut391", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_196", 0x6710}},
    {F, {"tlut392", 0, 9, 0x0}},
    {F, {"tlut393", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_197", 0x6714}},
    {F, {"tlut394", 0, 9, 0x0}},
    {F, {"tlut395", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_198", 0x6718}},
    {F, {"tlut396", 0, 9, 0x0}},
    {F, {"tlut397", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_199", 0x671C}},
    {F, {"tlut398", 0, 9, 0x0}},
    {F, {"tlut399", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_200", 0x6720}},
    {F, {"tlut400", 0, 9, 0x0}},
    {F, {"tlut401", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_201", 0x6724}},
    {F, {"tlut402", 0, 9, 0x0}},
    {F, {"tlut403", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_202", 0x6728}},
    {F, {"tlut404", 0, 9, 0x0}},
    {F, {"tlut405", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_203", 0x672C}},
    {F, {"tlut406", 0, 9, 0x0}},
    {F, {"tlut407", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_204", 0x6730}},
    {F, {"tlut408", 0, 9, 0x0}},
    {F, {"tlut409", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_205", 0x6734}},
    {F, {"tlut410", 0, 9, 0x0}},
    {F, {"tlut411", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_206", 0x6738}},
    {F, {"tlut412", 0, 9, 0x0}},
    {F, {"tlut413", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_207", 0x673C}},
    {F, {"tlut414", 0, 9, 0x0}},
    {F, {"tlut415", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_208", 0x6740}},
    {F, {"tlut416", 0, 9, 0x0}},
    {F, {"tlut417", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_209", 0x6744}},
    {F, {"tlut418", 0, 9, 0x0}},
    {F, {"tlut419", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_210", 0x6748}},
    {F, {"tlut420", 0, 9, 0x0}},
    {F, {"tlut421", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_211", 0x674C}},
    {F, {"tlut422", 0, 9, 0x0}},
    {F, {"tlut423", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_212", 0x6750}},
    {F, {"tlut424", 0, 9, 0x0}},
    {F, {"tlut425", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_213", 0x6754}},
    {F, {"tlut426", 0, 9, 0x0}},
    {F, {"tlut427", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_214", 0x6758}},
    {F, {"tlut428", 0, 9, 0x0}},
    {F, {"tlut429", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_215", 0x675C}},
    {F, {"tlut430", 0, 9, 0x0}},
    {F, {"tlut431", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_216", 0x6760}},
    {F, {"tlut432", 0, 9, 0x0}},
    {F, {"tlut433", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_217", 0x6764}},
    {F, {"tlut434", 0, 9, 0x0}},
    {F, {"tlut435", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_218", 0x6768}},
    {F, {"tlut436", 0, 9, 0x0}},
    {F, {"tlut437", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_219", 0x676C}},
    {F, {"tlut438", 0, 9, 0x0}},
    {F, {"tlut439", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_220", 0x6770}},
    {F, {"tlut440", 0, 9, 0x0}},
    {F, {"tlut441", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_221", 0x6774}},
    {F, {"tlut442", 0, 9, 0x0}},
    {F, {"tlut443", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_222", 0x6778}},
    {F, {"tlut444", 0, 9, 0x0}},
    {F, {"tlut445", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_223", 0x677C}},
    {F, {"tlut446", 0, 9, 0x0}},
    {F, {"tlut447", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_224", 0x6780}},
    {F, {"tlut448", 0, 9, 0x0}},
    {F, {"tlut449", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_225", 0x6784}},
    {F, {"tlut450", 0, 9, 0x0}},
    {F, {"tlut451", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_226", 0x6788}},
    {F, {"tlut452", 0, 9, 0x0}},
    {F, {"tlut453", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_227", 0x678C}},
    {F, {"tlut454", 0, 9, 0x0}},
    {F, {"tlut455", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_228", 0x6790}},
    {F, {"tlut456", 0, 9, 0x0}},
    {F, {"tlut457", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_229", 0x6794}},
    {F, {"tlut458", 0, 9, 0x0}},
    {F, {"tlut459", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_230", 0x6798}},
    {F, {"tlut460", 0, 9, 0x0}},
    {F, {"tlut461", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_231", 0x679C}},
    {F, {"tlut462", 0, 9, 0x0}},
    {F, {"tlut463", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_232", 0x67A0}},
    {F, {"tlut464", 0, 9, 0x0}},
    {F, {"tlut465", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_233", 0x67A4}},
    {F, {"tlut466", 0, 9, 0x0}},
    {F, {"tlut467", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_234", 0x67A8}},
    {F, {"tlut468", 0, 9, 0x0}},
    {F, {"tlut469", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_235", 0x67AC}},
    {F, {"tlut470", 0, 9, 0x0}},
    {F, {"tlut471", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_236", 0x67B0}},
    {F, {"tlut472", 0, 9, 0x0}},
    {F, {"tlut473", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_237", 0x67B4}},
    {F, {"tlut474", 0, 9, 0x0}},
    {F, {"tlut475", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_238", 0x67B8}},
    {F, {"tlut476", 0, 9, 0x0}},
    {F, {"tlut477", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_239", 0x67BC}},
    {F, {"tlut478", 0, 9, 0x0}},
    {F, {"tlut479", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_240", 0x67C0}},
    {F, {"tlut480", 0, 9, 0x0}},
    {F, {"tlut481", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_241", 0x67C4}},
    {F, {"tlut482", 0, 9, 0x0}},
    {F, {"tlut483", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_242", 0x67C8}},
    {F, {"tlut484", 0, 9, 0x0}},
    {F, {"tlut485", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_243", 0x67CC}},
    {F, {"tlut486", 0, 9, 0x0}},
    {F, {"tlut487", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_244", 0x67D0}},
    {F, {"tlut488", 0, 9, 0x0}},
    {F, {"tlut489", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_245", 0x67D4}},
    {F, {"tlut490", 0, 9, 0x0}},
    {F, {"tlut491", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_246", 0x67D8}},
    {F, {"tlut492", 0, 9, 0x0}},
    {F, {"tlut493", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_247", 0x67DC}},
    {F, {"tlut494", 0, 9, 0x0}},
    {F, {"tlut495", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_248", 0x67E0}},
    {F, {"tlut496", 0, 9, 0x0}},
    {F, {"tlut497", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_249", 0x67E4}},
    {F, {"tlut498", 0, 9, 0x0}},
    {F, {"tlut499", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_250", 0x67E8}},
    {F, {"tlut500", 0, 9, 0x0}},
    {F, {"tlut501", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_251", 0x67EC}},
    {F, {"tlut502", 0, 9, 0x0}},
    {F, {"tlut503", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_252", 0x67F0}},
    {F, {"tlut504", 0, 9, 0x0}},
    {F, {"tlut505", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_253", 0x67F4}},
    {F, {"tlut506", 0, 9, 0x0}},
    {F, {"tlut507", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_254", 0x67F8}},
    {F, {"tlut508", 0, 9, 0x0}},
    {F, {"tlut509", 16, 9, 0x0}},

    {R, {"erc/t_drop_lut_255", 0x67FC}},
    {F, {"tlut510", 0, 9, 0x0}},
    {F, {"tlut511", 16, 9, 0x0}},

    {R, {"erc/Reserved_6800", 0x6800}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6804", 0x6804}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6808", 0x6808}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_680C", 0x680C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6810", 0x6810}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6814", 0x6814}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6818", 0x6818}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_681C", 0x681C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6820", 0x6820}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6824", 0x6824}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6828", 0x6828}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_682C", 0x682C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6830", 0x6830}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6834", 0x6834}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6838", 0x6838}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_683C", 0x683C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6840", 0x6840}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6844", 0x6844}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6848", 0x6848}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_684C", 0x684C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6850", 0x6850}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6854", 0x6854}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6858", 0x6858}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_685C", 0x685C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6860", 0x6860}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6864", 0x6864}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6868", 0x6868}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_686C", 0x686C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6870", 0x6870}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6874", 0x6874}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6878", 0x6878}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_687C", 0x687C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6880", 0x6880}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6884", 0x6884}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6888", 0x6888}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_688C", 0x688C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6890", 0x6890}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6894", 0x6894}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6898", 0x6898}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_689C", 0x689C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68A0", 0x68A0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68A4", 0x68A4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68A8", 0x68A8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68AC", 0x68AC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68B0", 0x68B0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68B4", 0x68B4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68B8", 0x68B8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68BC", 0x68BC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68C0", 0x68C0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68C4", 0x68C4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68C8", 0x68C8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68CC", 0x68CC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68D0", 0x68D0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68D4", 0x68D4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68D8", 0x68D8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68DC", 0x68DC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68E0", 0x68E0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68E4", 0x68E4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68E8", 0x68E8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68EC", 0x68EC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68F0", 0x68F0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68F4", 0x68F4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68F8", 0x68F8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_68FC", 0x68FC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6900", 0x6900}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6904", 0x6904}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6908", 0x6908}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_690C", 0x690C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6910", 0x6910}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6914", 0x6914}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6918", 0x6918}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_691C", 0x691C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6920", 0x6920}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6924", 0x6924}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6928", 0x6928}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_692C", 0x692C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6930", 0x6930}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6934", 0x6934}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6938", 0x6938}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_693C", 0x693C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6940", 0x6940}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6944", 0x6944}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6948", 0x6948}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_694C", 0x694C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6950", 0x6950}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6954", 0x6954}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6958", 0x6958}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_695C", 0x695C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6960", 0x6960}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6964", 0x6964}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6968", 0x6968}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_696C", 0x696C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6970", 0x6970}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6974", 0x6974}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6978", 0x6978}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_697C", 0x697C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6980", 0x6980}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6984", 0x6984}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6988", 0x6988}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_698C", 0x698C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6990", 0x6990}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6994", 0x6994}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6998", 0x6998}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_699C", 0x699C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69A0", 0x69A0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69A4", 0x69A4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69A8", 0x69A8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69AC", 0x69AC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69B0", 0x69B0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69B4", 0x69B4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69B8", 0x69B8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69BC", 0x69BC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69C0", 0x69C0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69C4", 0x69C4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69C8", 0x69C8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69CC", 0x69CC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69D0", 0x69D0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69D4", 0x69D4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69D8", 0x69D8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69DC", 0x69DC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69E0", 0x69E0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69E4", 0x69E4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69E8", 0x69E8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69EC", 0x69EC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69F0", 0x69F0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69F4", 0x69F4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69F8", 0x69F8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_69FC", 0x69FC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A00", 0x6A00}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A04", 0x6A04}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A08", 0x6A08}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A0C", 0x6A0C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A10", 0x6A10}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A14", 0x6A14}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A18", 0x6A18}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A1C", 0x6A1C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A20", 0x6A20}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A24", 0x6A24}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A28", 0x6A28}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A2C", 0x6A2C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A30", 0x6A30}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A34", 0x6A34}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A38", 0x6A38}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A3C", 0x6A3C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A40", 0x6A40}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A44", 0x6A44}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A48", 0x6A48}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A4C", 0x6A4C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A50", 0x6A50}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A54", 0x6A54}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A58", 0x6A58}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A5C", 0x6A5C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A60", 0x6A60}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A64", 0x6A64}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A68", 0x6A68}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A6C", 0x6A6C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A70", 0x6A70}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A74", 0x6A74}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A78", 0x6A78}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A7C", 0x6A7C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A80", 0x6A80}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A84", 0x6A84}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A88", 0x6A88}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A8C", 0x6A8C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A90", 0x6A90}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A94", 0x6A94}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A98", 0x6A98}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6A9C", 0x6A9C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AA0", 0x6AA0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AA4", 0x6AA4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AA8", 0x6AA8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AAC", 0x6AAC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AB0", 0x6AB0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AB4", 0x6AB4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AB8", 0x6AB8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6ABC", 0x6ABC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AC0", 0x6AC0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AC4", 0x6AC4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AC8", 0x6AC8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6ACC", 0x6ACC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AD0", 0x6AD0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AD4", 0x6AD4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AD8", 0x6AD8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6ADC", 0x6ADC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AE0", 0x6AE0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AE4", 0x6AE4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AE8", 0x6AE8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AEC", 0x6AEC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AF0", 0x6AF0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AF4", 0x6AF4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AF8", 0x6AF8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6AFC", 0x6AFC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B00", 0x6B00}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B04", 0x6B04}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B08", 0x6B08}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B0C", 0x6B0C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B10", 0x6B10}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B14", 0x6B14}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B18", 0x6B18}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B1C", 0x6B1C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B20", 0x6B20}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B24", 0x6B24}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B28", 0x6B28}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B2C", 0x6B2C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B30", 0x6B30}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B34", 0x6B34}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B38", 0x6B38}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B3C", 0x6B3C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B40", 0x6B40}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B44", 0x6B44}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B48", 0x6B48}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B4C", 0x6B4C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B50", 0x6B50}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B54", 0x6B54}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B58", 0x6B58}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B5C", 0x6B5C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B60", 0x6B60}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B64", 0x6B64}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B68", 0x6B68}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B6C", 0x6B6C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B70", 0x6B70}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B74", 0x6B74}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B78", 0x6B78}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B7C", 0x6B7C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B80", 0x6B80}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B84", 0x6B84}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B88", 0x6B88}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B8C", 0x6B8C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B90", 0x6B90}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"erc/Reserved_6B94", 0x6B94}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"edf/pipeline_control", 0x7000}},
    {F, {"Reserved_0", 0, 1, 0x1}},
    {F, {"format", 1, 1, 0x0}},
    {F, {"Reserved_2", 2, 1, 0x0}},
    {F, {"Reserved_3", 3, 1, 0x0}},
    {F, {"Reserved_4", 4, 1, 0x0}},
    {F, {"Reserved_31_16", 16, 16, 0xFFFF}},

    {R, {"edf/Reserved_7004", 0x7004}},
    {F, {"Reserved_10", 10, 1, 0x1}},

    {R, {"eoi/Reserved_8000", 0x8000}},
    {F, {"Reserved_7_6", 6, 2, 0x2}},

    {R, {"ro/readout_ctrl", 0x9000}},
    {F, {"Reserved_0", 0, 1, 0x0}},
    {F, {"ro_td_self_test_en", 1, 1, 0x0}},
    {F, {"Reserved_3", 3, 1, 0x1}},
    {F, {"Reserved_4", 4, 1, 0x0}},
    {F, {"ro_inv_pol_td", 5, 1, 0x0}},
    {F, {"Reserved_7_6", 6, 2, 0x0}},
    {F, {"Reserved_31_8", 8, 24, 0x2}},

    {R, {"ro/ro_fsm_ctrl", 0x9004}},
    {F, {"readout_wait", 0, 16, 0x1E}},
    {F, {"Reserved_31_16", 16, 16, 0x0}},

    {R, {"ro/time_base_ctrl", 0x9008}},
    {F, {"time_base_enable", 0, 1, 0x0}},
    {F, {"time_base_mode", 1, 1, 0x0}},
    {F, {"external_mode", 2, 1, 0x0}},
    {F, {"external_mode_enable", 3, 1, 0x0}},
    {F, {"Reserved_10_4", 4, 7, 0x64}},

    {R, {"ro/dig_ctrl", 0x900C}},
    {F, {"dig_crop_enable", 0, 3, 0x0}},
    {F, {"dig_crop_reset_orig", 4, 1, 0x0}},
    {F, {"Reserved_31_5", 5, 27, 0x0}},

    {R, {"ro/dig_start_pos", 0x9010}},
    {F, {"dig_crop_start_x", 0, 11, 0x0}},
    {F, {"dig_crop_start_y", 16, 10, 0x0}},

    {R, {"ro/dig_end_pos", 0x9014}},
    {F, {"dig_crop_end_x", 0, 11, 0x0}},
    {F, {"dig_crop_end_y", 16, 10, 0x0}},

    {R, {"ro/ro_ctrl", 0x9028}},
    {F, {"area_cnt_en", 0, 1, 0x0}},
    {F, {"output_disable", 1, 1, 0x0}},
    {F, {"keep_th", 2, 1, 0x0}},

    {R, {"ro/area_x0_addr", 0x902C}},
    {F, {"x0_addr", 0, 11, 0x0}},

    {R, {"ro/area_x1_addr", 0x9030}},
    {F, {"x1_addr", 0, 11, 0x140}},

    {R, {"ro/area_x2_addr", 0x9034}},
    {F, {"x2_addr", 0, 11, 0x280}},

    {R, {"ro/area_x3_addr", 0x9038}},
    {F, {"x3_addr", 0, 11, 0x3C0}},

    {R, {"ro/area_x4_addr", 0x903C}},
    {F, {"x4_addr", 0, 11, 0x500}},

    {R, {"ro/area_y0_addr", 0x9040}},
    {F, {"y0_addr", 0, 11, 0x0}},

    {R, {"ro/area_y1_addr", 0x9044}},
    {F, {"y1_addr", 0, 11, 0xB4}},

    {R, {"ro/area_y2_addr", 0x9048}},
    {F, {"y2_addr", 0, 11, 0x168}},

    {R, {"ro/area_y3_addr", 0x904C}},
    {F, {"y3_addr", 0, 11, 0x21C}},

    {R, {"ro/area_y4_addr", 0x9050}},
    {F, {"y4_addr", 0, 11, 0x2D0}},

    {R, {"ro/counter_ctrl", 0x9054}},
    {F, {"count_en", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"Reserved_2", 2, 1, 0x1}},

    {R, {"ro/counter_timer_threshold", 0x9058}},
    {F, {"timer_threshold", 0, 32, 0x3E8}},

    {R, {"ro/digital_mask_pixel_00", 0x9100}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_01", 0x9104}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_02", 0x9108}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_03", 0x910C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_04", 0x9110}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_05", 0x9114}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_06", 0x9118}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_07", 0x911C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_08", 0x9120}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_09", 0x9124}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_10", 0x9128}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_11", 0x912C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_12", 0x9130}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_13", 0x9134}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_14", 0x9138}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_15", 0x913C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_16", 0x9140}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_17", 0x9144}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_18", 0x9148}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_19", 0x914C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_20", 0x9150}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_21", 0x9154}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_22", 0x9158}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_23", 0x915C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_24", 0x9160}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_25", 0x9164}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_26", 0x9168}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_27", 0x916C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_28", 0x9170}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_29", 0x9174}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_30", 0x9178}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_31", 0x917C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_32", 0x9180}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_33", 0x9184}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_34", 0x9188}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_35", 0x918C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_36", 0x9190}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_37", 0x9194}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_38", 0x9198}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_39", 0x919C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_40", 0x91A0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_41", 0x91A4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_42", 0x91A8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_43", 0x91AC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_44", 0x91B0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_45", 0x91B4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_46", 0x91B8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_47", 0x91BC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_48", 0x91C0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_49", 0x91C4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_50", 0x91C8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_51", 0x91CC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_52", 0x91D0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_53", 0x91D4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_54", 0x91D8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_55", 0x91DC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_56", 0x91E0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_57", 0x91E4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_58", 0x91E8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_59", 0x91EC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_60", 0x91F0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_61", 0x91F4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_62", 0x91F8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/digital_mask_pixel_63", 0x91FC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/area_cnt00", 0x9200}},
    {F, {"area_cnt_val_00", 0, 32, 0x0}},

    {R, {"ro/area_cnt01", 0x9204}},
    {F, {"area_cnt_val_01", 0, 32, 0x0}},

    {R, {"ro/area_cnt02", 0x9208}},
    {F, {"area_cnt_val_02", 0, 32, 0x0}},

    {R, {"ro/area_cnt03", 0x920C}},
    {F, {"area_cnt_val_03", 0, 32, 0x0}},

    {R, {"ro/area_cnt04", 0x9210}},
    {F, {"area_cnt_val_04", 0, 32, 0x0}},

    {R, {"ro/area_cnt05", 0x9214}},
    {F, {"area_cnt_val_05", 0, 32, 0x0}},

    {R, {"ro/area_cnt06", 0x9218}},
    {F, {"area_cnt_val_06", 0, 32, 0x0}},

    {R, {"ro/area_cnt07", 0x921C}},
    {F, {"area_cnt_val_07", 0, 32, 0x0}},

    {R, {"ro/area_cnt08", 0x9220}},
    {F, {"area_cnt_val_08", 0, 32, 0x0}},

    {R, {"ro/area_cnt09", 0x9224}},
    {F, {"area_cnt_val_09", 0, 32, 0x0}},

    {R, {"ro/area_cnt10", 0x9228}},
    {F, {"area_cnt_val_10", 0, 32, 0x0}},

    {R, {"ro/area_cnt11", 0x922C}},
    {F, {"area_cnt_val_11", 0, 32, 0x0}},

    {R, {"ro/area_cnt12", 0x9230}},
    {F, {"area_cnt_val_12", 0, 32, 0x0}},

    {R, {"ro/area_cnt13", 0x9234}},
    {F, {"area_cnt_val_13", 0, 32, 0x0}},

    {R, {"ro/area_cnt14", 0x9238}},
    {F, {"area_cnt_val_14", 0, 32, 0x0}},

    {R, {"ro/area_cnt15", 0x923C}},
    {F, {"area_cnt_val_15", 0, 32, 0x0}},

    {R, {"ro/evt_vector_cnt_val", 0x9244}},
    {F, {"evt_vector_cnt_val", 0, 32, 0x0}},

    {R, {"mipi_csi/mipi_control", 0xB000}},
    {F, {"mipi_csi_enable", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"Reserved_2", 2, 1, 0x0}},
    {F, {"mipi_data_lane1", 3, 1, 0x1}},
    {F, {"mipi_data_lane2", 4, 1, 0x1}},
    {F, {"mipi_packet_timeout_enable", 5, 1, 0x0}},
    {F, {"line_blanking_clk_disable", 6, 1, 0x1}},
    {F, {"Reserved_7", 7, 1, 0x0}},
    {F, {"line_blanking_en", 8, 1, 0x1}},
    {F, {"frame_blanking_en", 9, 1, 0x0}},
    {F, {"Reserved_31_10", 10, 22, 0x0}},

    {R, {"mipi_csi/mipi_packet_size", 0xB020}},
    {F, {"mipi_packet_size", 0, 15, 0x2000}},

    {R, {"mipi_csi/mipi_packet_timeout", 0xB024}},
    {F, {"mipi_packet_timeout", 0, 16, 0x40}},

    {R, {"mipi_csi/mipi_frame_period", 0xB028}},
    {F, {"mipi_frame_period", 4, 12, 0x7D}},

    {R, {"mipi_csi/mipi_line_blanking", 0xB02C}},
    {F, {"mipi_line_blanking", 0, 8, 0xA}},

    {R, {"mipi_csi/mipi_frame_blanking", 0xB030}},
    {F, {"mipi_frame_blanking", 0, 16, 0x0}},

    {R, {"afk/pipeline_control", 0xC000}},
    {F, {"Reserved_0", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"afk_bypass", 2, 1, 0x0}},

    {R, {"afk/param", 0xC004}},
    {F, {"counter_low", 0, 3, 0x4}},
    {F, {"counter_high", 3, 3, 0x6}},
    {F, {"invert", 6, 1, 0x0}},
    {F, {"drop_disable", 7, 1, 0x0}},

    {R, {"afk/filter_period", 0xC008}},
    {F, {"min_cutoff_period", 0, 8, 0xF}},
    {F, {"max_cutoff_period", 8, 8, 0x9C}},
    {F, {"inverted_duty_cycle", 16, 4, 0x8}},

    {R, {"afk/invalidation", 0xC0C0}},
    {F, {"dt_fifo_wait_time", 0, 12, 0x5A0}},
    {F, {"Reserved_23_12", 12, 12, 0x5A}},
    {F, {"Reserved_27_24", 24, 4, 0xA}},
    {F, {"Reserved_28", 28, 1, 0x0}},

    {R, {"afk/initialization", 0xC0C4}},
    {F, {"afk_req_init", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"afk_flag_init_done", 2, 1, 0x0}},

    {R, {"afk/shadow_ctrl", 0xC0D4}},
    {F, {"timer_en", 0, 1, 0x0}},
    {F, {"Reserved_31_1", 1, 31, 0x2}},

    {R, {"afk/shadow_timer_threshold", 0xC0D8}},
    {F, {"timer_threshold", 0, 32, 0x3E8}},

    {R, {"afk/shadow_status", 0xC0DC}},
    {F, {"shadow_valid", 0, 1, 0x0}},
    {F, {"shadow_overrun", 1, 1, 0x0}},
    {F, {"Reserved_31_2", 2, 30, 0x0}},

    {R, {"afk/total_evt_count", 0xC0E0}},
    {F, {"total_evt_count", 0, 32, 0x0}},

    {R, {"afk/flicker_evt_count", 0xC0E4}},
    {F, {"flicker_evt_count", 0, 32, 0x0}},

    {R, {"afk/vector_evt_count", 0xC0E8}},
    {F, {"vector_evt_count", 0, 32, 0x0}},

    {R, {"stc/pipeline_control", 0xD000}},
    {F, {"Reserved_0", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"stc_trail_bypass", 2, 1, 0x0}},

    {R, {"stc/stc_param", 0xD004}},
    {F, {"stc_enable", 0, 1, 0x0}},
    {F, {"stc_threshold", 1, 19, 0x2710}},
    {F, {"disable_stc_cut_trail", 24, 1, 0x0}},

    {R, {"stc/trail_param", 0xD008}},
    {F, {"trail_enable", 0, 1, 0x0}},
    {F, {"trail_threshold", 1, 19, 0x186A0}},

    {R, {"stc/timestamping", 0xD00C}},
    {F, {"prescaler", 0, 5, 0xD}},
    {F, {"multiplier", 5, 4, 0x1}},
    {F, {"Reserved_9", 9, 1, 0x1}},
    {F, {"enable_last_ts_update_at_every_event", 16, 1, 0x0}},

    {R, {"stc/invalidation", 0xD0C0}},
    {F, {"dt_fifo_wait_time", 0, 12, 0x4}},
    {F, {"dt_fifo_timeout", 12, 12, 0x118}},
    {F, {"Reserved_27_24", 24, 4, 0xA}},
    {F, {"Reserved_28", 28, 1, 0x0}},

    {R, {"stc/initialization", 0xD0C4}},
    {F, {"stc_req_init", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"stc_flag_init_done", 2, 1, 0x0}},

    {R, {"stc/shadow_ctrl", 0xD0D4}},
    {F, {"timer_en", 0, 1, 0x0}},
    {F, {"Reserved_31_1", 1, 31, 0x2}},

    {R, {"stc/shadow_timer_threshold", 0xD0D8}},
    {F, {"timer_threshold", 0, 32, 0x3E8}},

    {R, {"stc/shadow_status", 0xD0DC}},
    {F, {"shadow_valid", 0, 1, 0x0}},
    {F, {"shadow_overrun", 1, 1, 0x0}},
    {F, {"Reserved_31_2", 2, 30, 0x0}},

    {R, {"stc/total_evt_count", 0xD0E0}},
    {F, {"total_evt_count", 0, 32, 0x0}},

    {R, {"stc/stc_evt_count", 0xD0E4}},
    {F, {"stc_evt_count", 0, 32, 0x0}},

    {R, {"stc/trail_evt_count", 0xD0E8}},
    {F, {"trail_evt_count", 0, 32, 0x0}},

    {R, {"stc/output_vector_count", 0xD0EC}},
    {F, {"output_vector_count", 0, 32, 0x0}},

    {R, {"slvs/slvs_control", 0xE000}},
    {F, {"slvs_llp_enable", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"Reserved_2", 2, 1, 0x1}},
    {F, {"slvs_packet_timeout_enable", 5, 1, 0x0}},
    {F, {"slvs_line_blanking_en", 8, 1, 0x1}},
    {F, {"slvs_frame_blanking_en", 9, 1, 0x0}},

    {R, {"slvs/slvs_packet_size", 0xE020}},
    {F, {"slvs_packet_size", 0, 14, 0x1000}},

    {R, {"slvs/slvs_packet_timeout", 0xE024}},
    {F, {"slvs_packet_timeout", 0, 16, 0x40}},

    {R, {"slvs/slvs_line_blanking", 0xE02C}},
    {F, {"slvs_line_blanking", 0, 8, 0xA}},

    {R, {"slvs/slvs_frame_blanking", 0xE030}},
    {F, {"slvs_frame_blanking", 0, 16, 0x0}},

    {R, {"slvs/slvs_phy_logic_ctrl_00", 0xE150}},
    {F, {"oportsel", 0, 2, 0x0}}

    // clang-format on
};

static uint32_t Imx636RegisterMapSize = sizeof(Imx636RegisterMap) / sizeof(Imx636RegisterMap[0]);

#endif // METAVISION_HAL_IMX636_REGISTERMAP_H
