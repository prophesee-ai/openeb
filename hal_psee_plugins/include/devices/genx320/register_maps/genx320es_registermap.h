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

#ifndef METAVISION_HAL_GENX320ES_REGISTERMAP_H
#define METAVISION_HAL_GENX320ES_REGISTERMAP_H

#include "metavision/psee_hw_layer/utils/regmap_data.h"

static RegmapElement GenX320ESRegisterMap[] = {
    // clang-format off

    {R, {"roi_ctrl", 0x0000}},
    {F, {"unused0", 0, 1, 0x0}},
    {F, {"roi_td_en", 1, 1, 0x0}},
    {F, {"unused42", 2, 3, 0x0}},
    {F, {"roi_td_shadow_trigger", 5, 1, 0x0}},
    {F, {"px_iphoto_en", 6, 1, 0x0}},
    {F, {"px_row_mon_rstn", 7, 1, 0x0}},
    {F, {"unused98", 8, 2, 0x0}},
    {F, {"px_sw_rstn", 10, 1, 0x1}},
    {F, {"px_roi_halt_programming", 11, 1, 0x0}},
    {F, {"spare_for_digital", 12, 20, 0x0}},

    {R, {"test_bus_ctrl", 0x0004}},
    {F, {"unused0", 0, 1, 0x0}},
    {F, {"unused1", 1, 1, 0x0}},
    {F, {"ro_dft_buf_en", 2, 1, 0x0}},
    {F, {"unused2", 3, 1, 0x0}},
    {F, {"tbus_tpa1_ctl", 4, 5, 0x0}},
    {F, {"tbus_tpa2_ctl", 9, 5, 0x0}},
    {F, {"tbus_r2r_tpa1_en", 14, 1, 0x0}},
    {F, {"tbus_r2r_tpa2_en", 15, 1, 0x0}},
    {F, {"tbus_adc_sw", 16, 1, 0x0}},
    {F, {"agpio_lv0_ctl", 17, 4, 0x0}},
    {F, {"agpio_lv1_ctl", 21, 4, 0x0}},

    {R, {"lifo_ctrl", 0x0008}},
    {F, {"lifo_en", 0, 1, 0x0}},
    {F, {"lifo_cont_op_en", 1, 1, 0x0}},
    {F, {"lifo_dft_mode_en", 2, 1, 0x0}},
    {F, {"lifo_timer_en", 3, 1, 0x0}},
    {F, {"lifo_timer_threshold", 4, 14, 0x0}},

    {R, {"lifo_toff_status", 0x000C}},
    {F, {"lifo_toff", 0, 16, 0x0}},
    {F, {"lifo_toff_valid", 16, 1, 0x0}},
    {F, {"lifo_toff_overrun", 17, 1, 0x0}},

    {R, {"lifo_ton_status", 0x0010}},
    {F, {"lifo_ton", 0, 28, 0x0}},
    {F, {"lifo_ton_valid", 28, 1, 0x0}},
    {F, {"lifo_ton_overrun", 29, 1, 0x0}},

    {R, {"chip_id", 0x0014}},
    {F, {"chip_id", 0, 32, 0x30501C01}},

    {R, {"spare_ctrl0", 0x0018}},
    {F, {"spare0", 0, 32, 0x0}},

    {R, {"dig_soft_reset", 0x001C}},
    {F, {"digital_csr_srst", 0, 1, 0x0}},
    {F, {"digital_pipe_srst", 1, 1, 0x0}},
    {F, {"analog_rstn", 2, 1, 0x0}},
    {F, {"pdl_override", 3, 1, 0x0}},

    {R, {"refractory_ctrl", 0x0020}},
    {F, {"refr_counter", 0, 28, 0x0}},
    {F, {"refr_valid", 28, 1, 0x0}},
    {F, {"refr_overrun", 29, 1, 0x0}},
    {F, {"refr_cnt_en", 30, 1, 0x0}},
    {F, {"refr_en", 31, 1, 0x0}},

    {R, {"dig_test_bus_ctrl", 0x0024}},
    {F, {"tbus_sel_tpd1", 0, 6, 0x0}},

    {R, {"analog_measure_event", 0x0028}},
    {F, {"adc_event_en", 0, 1, 0x0}},
    {F, {"refr_event_en", 1, 1, 0x0}},
    {F, {"lifo_on_event_en", 2, 1, 0x0}},
    {F, {"lifo_off_event_en", 3, 1, 0x0}},

    {R, {"ro_td_ctrl", 0x002C}},
    {F, {"ro_td_act_pdy_drive", 0, 3, 0x4}},
    {F, {"ro_td_act_pu_drive", 3, 4, 0x4}},
    {F, {"ro_td_sendreq_y_stat_en", 7, 1, 0x0}},
    {F, {"ro_td_sendreq_y_rstn", 8, 1, 0x0}},
    {F, {"ro_td_int_x_rstn", 9, 1, 0x1}},
    {F, {"ro_td_int_y_rstn", 10, 1, 0x1}},
    {F, {"ro_td_int_x_stat_en", 11, 1, 0x0}},
    {F, {"ro_td_int_y_stat_en", 12, 1, 0x0}},
    {F, {"ro_td_addr_y_stat_en", 13, 1, 0x0}},
    {F, {"ro_td_addr_y_rstn", 14, 1, 0x0}},
    {F, {"ro_td_ack_y_rstn", 15, 1, 0x0}},
    {F, {"unused", 16, 1, 0x0}},
    {F, {"ro_td_arb_y_rstn", 17, 1, 0x0}},
    {F, {"ro_td_ack_y_set", 18, 1, 0x0}},
    {F, {"ro_td_int_x_act_pu", 19, 3, 0x4}},
    {F, {"ro_td_reqx_ctrllast_bypass", 22, 1, 0x0}},

    {R, {"roi_slice_ctrl1", 0x0030}},
    {F, {"roi_slice_list_sel", 0, 18, 0x0}},

    {R, {"roi_master_ctrl", 0x0034}},
    {F, {"roi_master_en", 0, 1, 0x0}},
    {F, {"roi_master_run", 1, 1, 0x0}},
    {F, {"roi_master_mode", 2, 1, 0x0}},
    {F, {"roi_win_nb", 3, 5, 0x0}},
    {F, {"roi_master_busy", 16, 1, 0x0}},
    {F, {"roi_master_done", 17, 1, 0x0}},

    {R, {"roi_driver_ctrl", 0x0038}},
    {F, {"roi_driver_en", 0, 1, 0x0}},
    {F, {"roi_driver_run", 1, 1, 0x0}},
    {F, {"roi_driver_mode", 2, 2, 0x0}},
    {F, {"roi_driver_shadow_trig_en", 4, 1, 0x0}},
    {F, {"roi_driver_busy", 16, 1, 0x0}},
    {F, {"roi_driver_done", 17, 1, 0x0}},

    {R, {"roi_slice_ctrl", 0x003C}},
    {F, {"roi_slice_start", 0, 9, 0x0}},
    {F, {"roi_slice_end_p1", 16, 9, 0x0}},

    {R, {"icn_control", 0x0040}},
    {F, {"icn_legacy_mode", 0, 1, 0x0}},
    {F, {"icn_disable_m0", 1, 1, 0x0}},
    {F, {"icn_disable_m1", 2, 1, 0x0}},

    {R, {"roi_master_chicken_bit", 0x0044}},
    {F, {"roi_driver_register_if_en", 0, 1, 0x0}},
    {F, {"roi_hold_time", 1, 5, 0xA}},

    {R, {"cpu_int_control", 0x0048}},
    {F, {"cpu_soft_it", 0, 16, 0x0}},
    {F, {"irq_lock_counter_cycles", 16, 6, 0x4}},
    {F, {"reserved", 22, 10, 0x0}},

    {R, {"adc_control", 0x004C}},
    {F, {"adc_en", 0, 1, 0x0}},
    {F, {"adc_clk_en", 1, 1, 0x0}},
    {F, {"adc_start", 2, 1, 0x0}},
    {F, {"adc_cont_nsingle", 3, 1, 0x0}},
    {F, {"adc_clk_div", 4, 16, 0x4E2}},
    {F, {"adc_pause", 20, 5, 0x2}},

    {R, {"adc_status1", 0x0050}},
    {F, {"adc_dac_dyn", 0, 10, 0x0}},
    {F, {"adc_on_dyn", 10, 1, 0x0}},
    {F, {"adc_done_dyn", 11, 1, 0x0}},
    {F, {"adc_buf_h_offset_dac_dyn", 12, 6, 0x20}},
    {F, {"adc_buf_l_offset_dac_dyn", 18, 6, 0x20}},
    {F, {"adc_buf_in_offset_dac_dyn", 24, 6, 0x20}},
    {F, {"adc_buf_cal_done_dyn", 30, 1, 0x0}},

    {R, {"adc_misc_ctrl", 0x0054}},
    {F, {"adc_buf_cal_en", 0, 1, 0x0}},
    {F, {"adc_cmp_cal_en", 1, 1, 0x0}},
    {F, {"adc_buf_test", 2, 3, 0x0}},
    {F, {"adc_dac_man_init", 5, 1, 0x0}},
    {F, {"adc_dac_test_en", 6, 1, 0x0}},
    {F, {"adc_test_en", 7, 1, 0x0}},
    {F, {"adc_test_cnt", 8, 4, 0x0}},
    {F, {"adc_ext_bg", 12, 1, 0x0}},
    {F, {"adc_ext_in", 13, 1, 0x0}},
    {F, {"adc_ihalf", 14, 1, 0x0}},
    {F, {"adc_rng", 15, 2, 0x0}},
    {F, {"adc_temp", 17, 1, 0x0}},
    {F, {"adc_clk_adj", 18, 2, 0x2}},
    {F, {"adc_vrefh_cnt", 20, 1, 0x0}},
    {F, {"adc_buf_adj_rng", 21, 1, 0x0}},
    {F, {"adc_cmp_adj_rng", 22, 1, 0x0}},
    {F, {"adc_cmp_offset_manual_dac", 23, 8, 0x80}},

    {R, {"adc_manual_ctrl1", 0x0058}},
    {F, {"adc_buf_h_offset_manual_dac", 0, 6, 0x20}},
    {F, {"adc_buf_l_offset_manual_dac", 6, 6, 0x20}},
    {F, {"adc_buf_in_offset_manual_dac", 12, 6, 0x20}},
    {F, {"adc_manual_dac", 18, 10, 0x200}},

    {R, {"temp_ctrl", 0x005C}},
    {F, {"temp_buf_en", 0, 1, 0x0}},
    {F, {"temp_buf_cal_en", 1, 1, 0x0}},
    {F, {"temp_buf_offset_man", 2, 6, 0x20}},
    {F, {"temp_ihalf", 8, 1, 0x0}},
    {F, {"temp_buf_adj_rng", 9, 1, 0x0}},
    {F, {"temp_test_ctl", 10, 2, 0x0}},
    {F, {"temp_test_tp_en", 12, 1, 0x0}},
    {F, {"temp_buf_offset_dac_dyn", 13, 6, 0x0}},
    {F, {"temp_buf_cal_done_dyn", 19, 1, 0x0}},

    {R, {"spare_from_ana", 0x0060}},
    {F, {"spare_from_ana", 0, 32, 0x0}},

    {R, {"top_chicken", 0x0064}},
    {F, {"i2c_prefetch_dis", 0, 1, 0x0}},
    {F, {"override_mipi_mode_en", 1, 1, 0x0}},
    {F, {"override_mipi_mode", 2, 1, 0x0}},
    {F, {"override_histo_mode_en", 3, 1, 0x0}},
    {F, {"override_histo_mode", 4, 1, 0x0}},
    {F, {"i2c_timeout_en", 5, 1, 0x0}},
    {F, {"standby", 6, 1, 0x0}},
    {F, {"i2c_timeout", 16, 16, 0x3E8}},

    {R, {"spare_ctrl1", 0x0068}},
    {F, {"spare1", 0, 32, 0xFFFFFFFF}},

    {R, {"adc_status2", 0x006C}},
    {F, {"adc_cmp_offset_dac_dyn", 0, 8, 0x0}},
    {F, {"adc_cmp_cal_done_dyn", 8, 1, 0x0}},

    {R, {"cp_ctrl", 0x0070}},
    {F, {"cp_en", 0, 1, 0x0}},
    {F, {"cp_clk_en", 1, 1, 0x0}},
    {F, {"cp_cfly", 2, 2, 0x3}},
    {F, {"cp_mphase", 4, 1, 0x1}},
    {F, {"cp_adj", 5, 6, 0x2A}},
    {F, {"unused", 11, 1, 0x0}},
    {F, {"cp_cmp_ctl", 12, 3, 0x0}},
    {F, {"cp_clk_divider", 15, 5, 0x0}},
    {F, {"cp_hiz", 20, 1, 0x0}},

    {R, {"iph_mirr_ctrl", 0x0074}},
    {F, {"iph_mirr_en", 0, 1, 0x0}},
    {F, {"iph_mirr_tbus_in_en", 1, 1, 0x0}},
    {F, {"iph_mirr_calib_en", 2, 1, 0x0}},
    {F, {"iph_mirr_calib_x10_en", 3, 1, 0x0}},
    {F, {"iph_mirr_dft_en", 4, 1, 0x0}},
    {F, {"iph_mirr_dft_sel", 5, 4, 0x0}},

    {R, {"gcd_ctrl", 0x0078}},
    {F, {"gcd_en", 0, 1, 0x0}},
    {F, {"gcd_ulp_en", 1, 1, 0x0}},
    {F, {"gcd_pr_en", 2, 1, 0x0}},
    {F, {"gcd_pr_pwr_ctl", 3, 2, 0x0}},
    {F, {"gcd_fe_cpd_ctl", 5, 2, 0x0}},
    {F, {"gcd_fe_cpr_ctl", 7, 3, 0x4}},
    {F, {"gcd_lpf_en", 10, 1, 0x0}},
    {F, {"gcd_lpf_pwr_ctl", 11, 1, 0x0}},
    {F, {"gcd_lpf_ccap_ctl", 12, 2, 0x3}},
    {F, {"gcd_diffamp_en", 14, 1, 0x0}},
    {F, {"gcd_diffamp_pwr_ctl", 15, 1, 0x0}},
    {F, {"gcd_gain_ctl", 16, 2, 0x1}},
    {F, {"gcd_cmp_en", 18, 1, 0x0}},
    {F, {"gcd_cmp_pwr_ctl", 19, 1, 0x0}},
    {F, {"gcd_vdac_buf_en", 20, 1, 0x0}},
    {F, {"gcd_vdac_buf_pwr_ctl", 21, 1, 0x0}},
    {F, {"gcd_test_en", 22, 1, 0x0}},
    {F, {"gcd_test_sel", 23, 4, 0x0}},
    {F, {"gcd_startup_rstn_dft_en", 27, 1, 0x0}},
    {F, {"gcd_rstn_dft_in", 28, 1, 0x0}},

    {R, {"gcd_ctrl2", 0x007C}},
    {F, {"gcd_hpf_ctl", 0, 3, 0x4}},
    {F, {"gcd_diff_ref_ctl", 3, 3, 0x3}},
    {F, {"gcd_neg_ref_ctl", 6, 3, 0x1}},
    {F, {"gcd_pos_ref_ctl", 9, 3, 0x1}},

    {R, {"gcd_refr_ctrl", 0x0080}},
    {F, {"gcd_refr_dur", 0, 32, 0x10}},

    {R, {"esp_status", 0x0084}},
    {F, {"ro_empty", 0, 1, 0x0}},
    {F, {"ro_busy", 1, 1, 0x0}},
    {F, {"ro_deep_low_power_seen", 2, 1, 0x0}},
    {F, {"nfl_empty", 3, 1, 0x0}},
    {F, {"nfl_busy", 4, 1, 0x0}},
    {F, {"nfl_deep_low_power_seen", 5, 1, 0x0}},
    {F, {"afk_empty", 6, 1, 0x0}},
    {F, {"afk_busy", 7, 1, 0x0}},
    {F, {"afk_deep_low_power_seen", 8, 1, 0x0}},
    {F, {"ehc_empty", 9, 1, 0x0}},
    {F, {"ehc_busy", 10, 1, 0x0}},
    {F, {"ehc_deep_low_power_seen", 11, 1, 0x0}},
    {F, {"stc_empty", 12, 1, 0x0}},
    {F, {"stc_busy", 13, 1, 0x0}},
    {F, {"stc_deep_low_power_seen", 14, 1, 0x0}},
    {F, {"erc_empty", 15, 1, 0x0}},
    {F, {"erc_busy", 16, 1, 0x0}},
    {F, {"erc_deep_low_power_seen", 17, 1, 0x0}},
    {F, {"edf_empty", 18, 1, 0x0}},
    {F, {"edf_busy", 19, 1, 0x0}},
    {F, {"edf_deep_low_power_seen", 20, 1, 0x0}},
    {F, {"cpi_empty", 21, 1, 0x0}},
    {F, {"cpi_busy", 22, 1, 0x0}},
    {F, {"unused", 23, 1, 0x0}},
    {F, {"mipi_empty", 24, 1, 0x0}},
    {F, {"mipi_busy", 25, 1, 0x0}},

    {R, {"qmon_ctrl", 0x0088}},
    {F, {"qmon_en", 0, 1, 0x0}},
    {F, {"qmon_rstn", 1, 1, 0x0}},
    {F, {"qmon_dft_en", 2, 1, 0x0}},
    {F, {"qmon_latch_input", 3, 1, 0x0}},
    {F, {"qmon_interrupt_en", 4, 1, 0x0}},
    {F, {"qmon_clk_prescale", 5, 4, 0x0}},
    {F, {"qmon_trip_ctl", 10, 9, 0x0}},

    {R, {"qmon_status", 0x008C}},
    {F, {"qmon_sum", 0, 10, 0x0}},
    {F, {"qmon_trip", 10, 1, 0x0}},
    {F, {"qmon_sum_irq", 16, 10, 0x0}},
    {F, {"qmon_trip_irq", 26, 1, 0x0}},

    {R, {"gcd_shadow_ctrl", 0x0090}},
    {F, {"gcd_timer_en", 0, 1, 0x0}},
    {F, {"gcd_irq_sw_override", 1, 1, 0x0}},
    {F, {"gcd_reset_on_copy", 2, 1, 0x0}},

    {R, {"gcd_shadow_status", 0x0094}},
    {F, {"gcd_shadow_valid", 0, 1, 0x0}},
    {F, {"gcd_shadow_overrun", 1, 1, 0x0}},

    {R, {"gcd_shadow_timer_ctrl", 0x0098}},
    {F, {"gcd_shadow_timer_threshold", 0, 20, 0x0}},

    {R, {"watchdog_ctrl", 0x009C}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"reset_enable", 1, 1, 0x0}},
    {F, {"irq_enable", 2, 1, 0x0}},
    {F, {"force_overflow", 3, 1, 0x0}},

    {R, {"watchdog_threshold", 0x00A0}},
    {F, {"value", 0, 32, 0x0}},

    {R, {"watchdog_reload", 0x00A4}},
    {F, {"value", 0, 32, 0x0}},

    {R, {"px_scan_ctrl", 0x00A8}},
    {F, {"px_scan_bot_en", 0, 1, 0x0}},
    {F, {"px_scan_top_en", 1, 1, 0x0}},
    {F, {"px_scan_left_en", 2, 1, 0x0}},
    {F, {"px_scan_right_en", 3, 1, 0x0}},
    {F, {"px_scan_bot_trig_first", 4, 1, 0x0}},
    {F, {"px_scan_top_trig_first", 5, 1, 0x0}},
    {F, {"px_scan_left_trig_first", 6, 1, 0x0}},
    {F, {"px_scan_right_trig_first", 7, 1, 0x0}},
    {F, {"px_scan_pol_sel", 8, 1, 0x0}},
    {F, {"px_scan_bot_trig_last_irq_en", 9, 1, 0x0}},
    {F, {"px_scan_top_trig_last_irq_en", 10, 1, 0x0}},
    {F, {"px_scan_left_trig_last_irq_en", 11, 1, 0x0}},
    {F, {"px_scan_right_trig_last_irq_en", 12, 1, 0x0}},
    {F, {"px_scan_timer_prescale", 16, 3, 0x0}},

    {R, {"px_scan_status", 0x00AC}},
    {F, {"px_scan_bot_trig_last", 0, 1, 0x0}},
    {F, {"px_scan_top_trig_last", 1, 1, 0x0}},
    {F, {"px_scan_left_trig_last", 2, 1, 0x0}},
    {F, {"px_scan_right_trig_last", 3, 1, 0x0}},
    {F, {"px_scan_bot_timer_valid", 4, 1, 0x0}},
    {F, {"px_scan_top_timer_valid", 5, 1, 0x0}},
    {F, {"px_scan_left_timer_valid", 6, 1, 0x0}},
    {F, {"px_scan_right_timer_valid", 7, 1, 0x0}},

    {R, {"px_scan_bot_top_timer", 0x00B0}},
    {F, {"px_scan_bot_timer", 0, 16, 0x0}},
    {F, {"px_scan_top_timer", 16, 16, 0x0}},

    {R, {"px_scan_left_right_timer", 0x00B4}},
    {F, {"px_scan_left_timer", 0, 16, 0x0}},
    {F, {"px_scan_right_timer", 16, 16, 0x0}},

    {R, {"sram_initn", 0x00B8}},
    {F, {"afk_initn", 0, 1, 0x0}},
    {F, {"ehc_stc_initn", 1, 1, 0x0}},
    {F, {"erc_dl_initn", 2, 1, 0x0}},
    {F, {"erc_ilg_initn", 3, 1, 0x0}},
    {F, {"erc_tdrop_initn", 4, 1, 0x0}},
    {F, {"mipi_initn", 5, 1, 0x0}},
    {F, {"cpi_initn", 6, 1, 0x0}},
    {F, {"imem_initn", 7, 1, 0x0}},
    {F, {"dmem_initn", 8, 1, 0x0}},
    {F, {"rom_initn", 9, 1, 0x0}},

    {R, {"sram_pd0", 0x00BC}},
    {F, {"afk_alr_pd", 0, 5, 0x1F}},
    {F, {"afk_str0_pd", 5, 5, 0x1F}},
    {F, {"afk_str1_pd", 10, 5, 0x1F}},
    {F, {"stc0_pd", 15, 5, 0x1F}},
    {F, {"stc1_pd", 20, 5, 0x1F}},
    {F, {"ehc_pd", 25, 5, 0x1F}},

    {R, {"sram_pd1", 0x00C0}},
    {F, {"dmem_pd", 0, 1, 0x0}},
    {F, {"imem_pd", 1, 1, 0x0}},
    {F, {"rom_pd", 2, 1, 0x0}},
    {F, {"erc_dl_pd", 3, 1, 0x1}},
    {F, {"erc_ilg_pd", 4, 1, 0x1}},
    {F, {"erc_tdrop_pd", 5, 1, 0x1}},
    {F, {"mipi_pd", 6, 1, 0x1}},
    {F, {"cpi_pd", 7, 1, 0x1}},

    {R, {"gcd_shadow_cnt0", 0x00C4}},
    {F, {"gcd_pos0_cnt", 0, 8, 0x0}},
    {F, {"gcd_pos1_cnt", 8, 8, 0x0}},
    {F, {"gcd_pos2_cnt", 16, 8, 0x0}},
    {F, {"gcd_pos3_cnt", 24, 8, 0x0}},

    {R, {"gcd_shadow_cnt1", 0x00C8}},
    {F, {"gcd_pos4_cnt", 0, 8, 0x0}},
    {F, {"gcd_pos5_cnt", 8, 8, 0x0}},
    {F, {"gcd_pos6_cnt", 16, 8, 0x0}},
    {F, {"gcd_pos7_cnt", 24, 8, 0x0}},

    {R, {"gcd_shadow_cnt2", 0x00CC}},
    {F, {"gcd_pos8_cnt", 0, 8, 0x0}},
    {F, {"gcd_neg0_cnt", 8, 8, 0x0}},
    {F, {"gcd_neg1_cnt", 16, 8, 0x0}},
    {F, {"gcd_neg2_cnt", 24, 8, 0x0}},

    {R, {"gcd_shadow_cnt3", 0x00D0}},
    {F, {"gcd_neg3_cnt", 0, 8, 0x0}},
    {F, {"gcd_neg4_cnt", 8, 8, 0x0}},
    {F, {"gcd_neg5_cnt", 16, 8, 0x0}},
    {F, {"gcd_neg6_cnt", 24, 8, 0x0}},

    {R, {"gcd_shadow_cnt4", 0x00D4}},
    {F, {"gcd_neg7_cnt", 0, 8, 0x0}},
    {F, {"gcd_neg8_cnt", 8, 8, 0x0}},

    {R, {"bgr_shadow_ctrl", 0x00D8}},
    {F, {"bgr_timer_en", 0, 1, 0x0}},
    {F, {"bgr_irq_sw_override", 1, 1, 0x0}},
    {F, {"bgr_reset_on_copy", 2, 1, 0x0}},
    {F, {"bgr_en", 3, 1, 0x0}},

    {R, {"bgr_shadow_status", 0x00DC}},
    {F, {"bgr_shadow_valid", 0, 1, 0x0}},
    {F, {"bgr_shadow_overrun", 1, 1, 0x0}},

    {R, {"bgr_shadow_timer_ctrl", 0x00E0}},
    {F, {"bgr_shadow_timer_threshold", 0, 20, 0x0}},

    {R, {"bgr_shadow_cnt", 0x00E4}},
    {F, {"bgr_pos_cnt", 0, 16, 0x0}},
    {F, {"bgr_neg_cnt", 16, 16, 0x0}},

    {R, {"ref_clk_ctrl", 0x0200}},
    {F, {"ref_clk_en", 0, 1, 0x0}},
    {F, {"ref_clk_switch", 1, 1, 0x0}},
    {F, {"ref_clk_div", 2, 4, 0x0}},

    {R, {"sys_clk_ctrl", 0x0204}},
    {F, {"sys_clk_en", 0, 1, 0x0}},
    {F, {"sys_clk_switch", 1, 1, 0x0}},
    {F, {"phy_clk_off_count", 2, 6, 0x1B}},
    {F, {"phy_clk_on_count", 8, 6, 0xD}},
    {F, {"phy_clk_div2", 14, 1, 0x1}},
    {F, {"sys_clk_auto_mode", 15, 1, 0x1}},

    {R, {"cpu_ss_clk_ctrl", 0x0208}},
    {F, {"cpu_ss_clk_en", 0, 1, 0x0}},
    {F, {"cpu_ss_clk_switch", 1, 1, 0x0}},
    {F, {"cpu_ss_clk_div", 2, 8, 0x0}},

    {R, {"rtc_clk_ctrl", 0x020C}},
    {F, {"rtc_clk_en", 0, 1, 0x0}},
    {F, {"rtc_clk_div", 1, 8, 0x6}},

    {R, {"evt_icn_clk_ctrl", 0x0210}},
    {F, {"evt_icn_clk_en", 0, 1, 0x0}},
    {F, {"evt_icn_clk_switch", 1, 1, 0x0}},
    {F, {"evt_icn_clk_div", 2, 8, 0x0}},
    {F, {"esp_clk_en", 10, 1, 0x0}},
    {F, {"ro_clk_en", 11, 1, 0x0}},

    {R, {"pll_ctrl", 0x0214}},
    {F, {"pl_enable", 0, 1, 0x0}},
    {F, {"pl_ndiv", 1, 8, 0xFA}},
    {F, {"pl_odf", 9, 4, 0x2}},
    {F, {"pl_strb", 13, 1, 0x0}},
    {F, {"pl_strb_bypass", 14, 1, 0x0}},
    {F, {"pl_lockp", 15, 1, 0x0}},
    {F, {"pl_lockp_delayed", 16, 1, 0x0}},

    {R, {"pll_sscg_ctrl", 0x0218}},
    {F, {"pl_sscg_ctrl", 0, 1, 0x0}},
    {F, {"pl_mod_period", 1, 13, 0x0}},
    {F, {"pl_inc_step", 14, 15, 0x0}},
    {F, {"pl_spread_ctrl", 29, 1, 0x0}},

    {R, {"pll_frac_ctrl", 0x021C}},
    {F, {"pl_frac_ctrl", 0, 1, 0x0}},
    {F, {"pl_frac_input", 1, 16, 0x0}},
    {F, {"pl_dither_disable", 17, 2, 0x0}},

    {R, {"roi_win_x0", 0x0400}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y0", 0x0404}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x1", 0x0408}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y1", 0x040C}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x2", 0x0410}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y2", 0x0414}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x3", 0x0418}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y3", 0x041C}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x4", 0x0420}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y4", 0x0424}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x5", 0x0428}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y5", 0x042C}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x6", 0x0430}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y6", 0x0434}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x7", 0x0438}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y7", 0x043C}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x8", 0x0440}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y8", 0x0444}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x9", 0x0448}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y9", 0x044C}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x10", 0x0450}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y10", 0x0454}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x11", 0x0458}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y11", 0x045C}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x12", 0x0460}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y12", 0x0464}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x13", 0x0468}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y13", 0x046C}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x14", 0x0470}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y14", 0x0474}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x15", 0x0478}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y15", 0x047C}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x16", 0x0480}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y16", 0x0484}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"roi_win_x17", 0x0488}},
    {F, {"roi_win_start_x", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_x", 16, 9, 0x0}},

    {R, {"roi_win_y17", 0x048C}},
    {F, {"roi_win_start_y", 0, 9, 0x0}},
    {F, {"roi_win_end_p1_y", 16, 9, 0x0}},

    {R, {"io_ctrl0", 0x0600}},
    {F, {"mode0_en", 0, 1, 0x1}},
    {F, {"mode0_pun", 1, 1, 0x1}},
    {F, {"mode0_pdn", 2, 1, 0x0}},
    {F, {"mode0_drive", 3, 1, 0x0}},
    {F, {"mode0_enzi", 4, 1, 0x1}},
    {F, {"mode1_en", 5, 1, 0x1}},
    {F, {"mode1_pun", 6, 1, 0x1}},
    {F, {"mode1_pdn", 7, 1, 0x0}},
    {F, {"mode1_drive", 8, 1, 0x0}},
    {F, {"mode1_enzi", 9, 1, 0x1}},
    {F, {"mipimode_en", 10, 1, 0x1}},
    {F, {"mipimode_pun", 11, 1, 0x1}},
    {F, {"mipimode_pdn", 12, 1, 0x0}},
    {F, {"mipimode_drive", 13, 1, 0x0}},
    {F, {"mipimode_enzi", 14, 1, 0x1}},
    {F, {"histomode_en", 15, 1, 0x1}},
    {F, {"histomode_pun", 16, 1, 0x1}},
    {F, {"histomode_pdn", 17, 1, 0x0}},
    {F, {"histomode_drive", 18, 1, 0x0}},
    {F, {"histomode_enzi", 19, 1, 0x1}},
    {F, {"rommode_en", 20, 1, 0x1}},
    {F, {"rommode_pun", 21, 1, 0x1}},
    {F, {"rommode_pdn", 22, 1, 0x0}},
    {F, {"rommode_drive", 23, 1, 0x0}},
    {F, {"rommode_enzi", 24, 1, 0x1}},
    {F, {"clki_en", 25, 1, 0x1}},
    {F, {"clki_pun", 26, 1, 0x1}},
    {F, {"clki_pdn", 27, 1, 0x0}},
    {F, {"clki_drive", 28, 1, 0x0}},
    {F, {"clki_enzi", 29, 1, 0x1}},
    {F, {"mipimode_zi", 30, 1, 0x0}},
    {F, {"histomode_zi", 31, 1, 0x0}},

    {R, {"io_ctrl1", 0x0604}},
    {F, {"rstn_en", 0, 1, 0x1}},
    {F, {"rstn_pun", 1, 1, 0x1}},
    {F, {"rstn_pdn", 2, 1, 0x0}},
    {F, {"rstn_drive", 3, 1, 0x0}},
    {F, {"rstn_enzi", 4, 1, 0x1}},
    {F, {"spiclk_en", 5, 1, 0x1}},
    {F, {"spiclk_pun", 6, 1, 0x0}},
    {F, {"spiclk_pdn", 7, 1, 0x1}},
    {F, {"spiclk_drive", 8, 1, 0x0}},
    {F, {"spiclk_enzi", 9, 1, 0x1}},
    {F, {"spimosi_en", 10, 1, 0x1}},
    {F, {"spimosi_pun", 11, 1, 0x1}},
    {F, {"spimosi_pdn", 12, 1, 0x0}},
    {F, {"spimosi_drive", 13, 1, 0x0}},
    {F, {"spimosi_enzi", 14, 1, 0x1}},
    {F, {"spimiso_en", 15, 1, 0x0}},
    {F, {"spimiso_pun", 16, 1, 0x1}},
    {F, {"spimiso_pdn", 17, 1, 0x0}},
    {F, {"spimiso_drive", 18, 1, 0x0}},
    {F, {"spimiso_enzi", 19, 1, 0x0}},
    {F, {"spicsn_en", 20, 1, 0x1}},
    {F, {"spicsn_pun", 21, 1, 0x0}},
    {F, {"spicsn_pdn", 22, 1, 0x1}},
    {F, {"spicsn_drive", 23, 1, 0x0}},
    {F, {"spicsn_enzi", 24, 1, 0x1}},
    {F, {"i2caddr_en", 25, 1, 0x1}},
    {F, {"i2caddr_pun", 26, 1, 0x1}},
    {F, {"i2caddr_pdn", 27, 1, 0x0}},
    {F, {"i2caddr_drive", 28, 1, 0x0}},
    {F, {"i2caddr_enzi", 29, 1, 0x1}},

    {R, {"io_ctrl2", 0x0608}},
    {F, {"sync_en", 0, 1, 0x0}},
    {F, {"sync_pun", 1, 1, 0x1}},
    {F, {"sync_pdn", 2, 1, 0x0}},
    {F, {"sync_drive", 3, 1, 0x0}},
    {F, {"sync_enzi", 4, 1, 0x0}},
    {F, {"pxrstn_en", 5, 1, 0x1}},
    {F, {"pxrstn_pun", 6, 1, 0x0}},
    {F, {"pxrstn_pdn", 7, 1, 0x1}},
    {F, {"pxrstn_drive", 8, 1, 0x0}},
    {F, {"pxrstn_enzi", 9, 1, 0x1}},
    {F, {"exttrig_en", 10, 1, 0x1}},
    {F, {"exttrig_pun", 11, 1, 0x1}},
    {F, {"exttrig_pdn", 12, 1, 0x0}},
    {F, {"exttrig_drive", 13, 1, 0x0}},
    {F, {"exttrig_enzi", 14, 1, 0x1}},
    {F, {"dgpio_en", 15, 1, 0x0}},
    {F, {"dgpio_pun", 16, 1, 0x1}},
    {F, {"dgpio_pdn", 17, 1, 0x0}},
    {F, {"dgpio_drive", 18, 1, 0x0}},
    {F, {"dgpio_enzi", 19, 1, 0x0}},
    {F, {"pixclk_en", 20, 1, 0x0}},
    {F, {"pixclk_pun", 21, 1, 0x1}},
    {F, {"pixclk_pdn", 22, 1, 0x0}},
    {F, {"pixclk_drive", 23, 1, 0x0}},
    {F, {"pixclk_enzi", 24, 1, 0x0}},
    {F, {"hsync_en", 25, 1, 0x0}},
    {F, {"hsync_pun", 26, 1, 0x1}},
    {F, {"hsync_pdn", 27, 1, 0x0}},
    {F, {"hsync_drive", 28, 1, 0x0}},
    {F, {"hsync_enzi", 29, 1, 0x0}},

    {R, {"io_ctrl3", 0x060C}},
    {F, {"vsync_en", 0, 1, 0x0}},
    {F, {"vsync_pun", 1, 1, 0x1}},
    {F, {"vsync_pdn", 2, 1, 0x0}},
    {F, {"vsync_drive", 3, 1, 0x0}},
    {F, {"vsync_enzi", 4, 1, 0x0}},
    {F, {"d0_en", 5, 1, 0x0}},
    {F, {"d0_pun", 6, 1, 0x1}},
    {F, {"d0_pdn", 7, 1, 0x0}},
    {F, {"d0_drive", 8, 1, 0x0}},
    {F, {"d0_enzi", 9, 1, 0x0}},
    {F, {"d1_en", 10, 1, 0x0}},
    {F, {"d1_pun", 11, 1, 0x1}},
    {F, {"d1_pdn", 12, 1, 0x0}},
    {F, {"d1_drive", 13, 1, 0x0}},
    {F, {"d1_enzi", 14, 1, 0x0}},
    {F, {"d2_en", 15, 1, 0x0}},
    {F, {"d2_pun", 16, 1, 0x1}},
    {F, {"d2_pdn", 17, 1, 0x0}},
    {F, {"d2_drive", 18, 1, 0x0}},
    {F, {"d2_enzi", 19, 1, 0x0}},
    {F, {"d3_en", 20, 1, 0x0}},
    {F, {"d3_pun", 21, 1, 0x1}},
    {F, {"d3_pdn", 22, 1, 0x0}},
    {F, {"d3_drive", 23, 1, 0x0}},
    {F, {"d3_enzi", 24, 1, 0x0}},
    {F, {"d4_en", 25, 1, 0x0}},
    {F, {"d4_pun", 26, 1, 0x1}},
    {F, {"d4_pdn", 27, 1, 0x0}},
    {F, {"d4_drive", 28, 1, 0x0}},
    {F, {"d4_enzi", 29, 1, 0x0}},

    {R, {"io_ctrl4", 0x0610}},
    {F, {"d5_en", 0, 1, 0x0}},
    {F, {"d5_pun", 1, 1, 0x1}},
    {F, {"d5_pdn", 2, 1, 0x0}},
    {F, {"d5_drive", 3, 1, 0x0}},
    {F, {"d5_enzi", 4, 1, 0x0}},
    {F, {"d6_en", 5, 1, 0x0}},
    {F, {"d6_pun", 6, 1, 0x1}},
    {F, {"d6_pdn", 7, 1, 0x0}},
    {F, {"d6_drive", 8, 1, 0x0}},
    {F, {"d6_enzi", 9, 1, 0x0}},
    {F, {"d7_en", 10, 1, 0x0}},
    {F, {"d7_pun", 11, 1, 0x1}},
    {F, {"d7_pdn", 12, 1, 0x0}},
    {F, {"d7_drive", 13, 1, 0x0}},
    {F, {"d7_enzi", 14, 1, 0x0}},
    {F, {"agpio0_en", 15, 1, 0x0}},
    {F, {"agpio0_pun", 16, 1, 0x1}},
    {F, {"agpio0_pdn", 17, 1, 0x0}},
    {F, {"agpio0_drive", 18, 1, 0x0}},
    {F, {"agpio0_enzi", 19, 1, 0x0}},
    {F, {"agpio1_en", 20, 1, 0x0}},
    {F, {"agpio1_pun", 21, 1, 0x1}},
    {F, {"agpio1_pdn", 22, 1, 0x0}},
    {F, {"agpio1_drive", 23, 1, 0x0}},
    {F, {"agpio1_enzi", 24, 1, 0x0}},

    {R, {"io_ctrl5", 0x0614}},
    {F, {"scl_i2coute", 0, 1, 0x1}},
    {F, {"scl_loadsel", 1, 1, 0x1}},
    {F, {"scl_drive", 2, 3, 0x7}},
    {F, {"scl_busmode", 5, 1, 0x0}},
    {F, {"scl_delayrx", 6, 1, 0x0}},
    {F, {"scl_fmfe", 7, 1, 0x1}},
    {F, {"scl_i2cine", 8, 1, 0x1}},
    {F, {"sda_i2coute", 9, 1, 0x1}},
    {F, {"sda_loadsel", 10, 1, 0x1}},
    {F, {"sda_drive", 11, 3, 0x7}},
    {F, {"sda_busmode", 14, 1, 0x0}},
    {F, {"sda_delayrx", 15, 1, 0x0}},
    {F, {"sda_fmfe", 16, 1, 0x1}},
    {F, {"sda_i2cine", 17, 1, 0x1}},
    {F, {"agpio_ta", 18, 1, 0x0}},
    {F, {"agpio_ten", 19, 1, 0x0}},
    {F, {"agpio_tm", 20, 1, 0x0}},
    {F, {"unused", 21, 11, 0x0}},

    {R, {"gpo_px_init_ctrl0", 0x0800}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"period", 1, 20, 0xF4240}},
    {F, {"pol", 21, 1, 0x1}},

    {R, {"gpo_px_init_ctrl1", 0x0804}},
    {F, {"delay1", 0, 20, 0x14}},

    {R, {"gpo_px_init_ctrl2", 0x0808}},
    {F, {"delay2", 0, 20, 0xF422C}},

    {R, {"gpo_px_rst_amp_ctrl0", 0x080C}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"period", 1, 16, 0x64}},
    {F, {"pol", 17, 1, 0x1}},

    {R, {"gpo_px_rst_amp_ctrl1", 0x0810}},
    {F, {"delay1", 0, 16, 0x5}},
    {F, {"delay2", 16, 16, 0x64}},

    {R, {"gpo_px_rst_cmp_ctrl0", 0x0814}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"period", 1, 16, 0x64}},
    {F, {"pol", 17, 1, 0x1}},

    {R, {"gpo_px_rst_cmp_ctrl1", 0x0818}},
    {F, {"delay1", 0, 16, 0xA}},
    {F, {"delay2", 16, 16, 0x64}},

    {R, {"gpo_px_rst_bias_ctrl0", 0x081C}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"period", 1, 16, 0x64}},
    {F, {"pol", 17, 1, 0x1}},

    {R, {"gpo_px_rst_bias_ctrl1", 0x0820}},
    {F, {"delay1", 0, 16, 0xC}},
    {F, {"delay2", 16, 16, 0x64}},

    {R, {"gpo_px_act_ctrl0", 0x0824}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"period", 1, 16, 0x64}},
    {F, {"pol", 17, 1, 0x0}},

    {R, {"gpo_px_act_ctrl1", 0x0828}},
    {F, {"delay1", 0, 16, 0x0}},
    {F, {"delay2", 16, 16, 0x0}},

    {R, {"gpo_px_trg_ctrl0", 0x082C}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"period", 1, 16, 0x64}},
    {F, {"pol", 17, 1, 0x0}},

    {R, {"gpo_px_trg_ctrl1", 0x0830}},
    {F, {"delay1", 0, 16, 0x5D}},
    {F, {"delay2", 16, 16, 0x62}},

    {R, {"gpo_px_mask_b_ctrl0", 0x0834}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"period", 1, 16, 0x64}},
    {F, {"pol", 17, 1, 0x0}},

    {R, {"gpo_px_mask_b_ctrl1", 0x0838}},
    {F, {"delay1", 0, 16, 0xE}},
    {F, {"delay2", 16, 16, 0x58}},

    {R, {"gpo_global_ctrl", 0x083C}},
    {F, {"enable", 0, 1, 0x0}},

    {R, {"mem_bank/bank_select", 0xF800}},
    {F, {"bank", 0, 16, 0x0}},
    {F, {"select", 28, 2, 0x0}},
    
    {R, {"bank_mem0", 0xF900}},

    {R, {"bias/bias_pr_hv0", 0x1000}},
    {F, {"bias_ctl", 0, 7, 0x3D}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_fo_hv0", 0x1004}},
    {F, {"bias_ctl", 0, 7, 0x27}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_fes_hv0", 0x1008}},
    {F, {"bias_ctl", 0, 7, 0x3F}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_pr_hv1", 0x100C}},
    {F, {"bias_ctl", 0, 7, 0x3D}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_fo_hv1", 0x1010}},
    {F, {"bias_ctl", 0, 7, 0x27}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_fes_hv1", 0x1014}},
    {F, {"bias_ctl", 0, 7, 0x3F}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_hpf_lv0", 0x1100}},
    {F, {"bias_ctl", 0, 7, 0x0}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_diff_on_lv0", 0x1104}},
    {F, {"bias_ctl", 0, 7, 0x18}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"ibtype_sel", 19, 1, 0x1}},
    {F, {"unused1", 20, 1, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_diff_lv0", 0x1108}},
    {F, {"bias_ctl", 0, 7, 0x33}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_diff_off_lv0", 0x110C}},
    {F, {"bias_ctl", 0, 7, 0x13}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"ibtype_sel", 19, 1, 0x1}},
    {F, {"unused1", 20, 1, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_inv_lv0", 0x1110}},
    {F, {"bias_ctl", 0, 7, 0x39}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_refr_lv0", 0x1114}},
    {F, {"bias_ctl", 0, 7, 0x52}},
    {F, {"unused", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"range_sel", 19, 1, 0x0}},
    {F, {"dc_trim_bit", 20, 1, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_invp_lv0", 0x1118}},
    {F, {"bias_ctl", 0, 7, 0x42}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_req_pu_lv0", 0x111C}},
    {F, {"bias_ctl", 0, 8, 0x74}},
    {F, {"unused0", 16, 2, 0x0}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_sm_pdy_lv0", 0x1120}},
    {F, {"bias_ctl", 0, 8, 0xA4}},
    {F, {"unused0", 16, 2, 0x0}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_hpf_lv1", 0x1124}},
    {F, {"bias_ctl", 0, 7, 0x0}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_diff_on_lv1", 0x1128}},
    {F, {"bias_ctl", 0, 7, 0x18}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"ibtype_sel", 19, 1, 0x1}},
    {F, {"unused1", 20, 1, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_diff_lv1", 0x112C}},
    {F, {"bias_ctl", 0, 7, 0x33}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_diff_off_lv1", 0x1130}},
    {F, {"bias_ctl", 0, 7, 0x13}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"ibtype_sel", 19, 1, 0x1}},
    {F, {"unused1", 20, 1, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_inv_lv1", 0x1134}},
    {F, {"bias_ctl", 0, 7, 0x39}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_refr_lv1", 0x1138}},
    {F, {"bias_ctl", 0, 7, 0x52}},
    {F, {"unused", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"range_sel", 19, 1, 0x0}},
    {F, {"dc_trim_bit", 20, 1, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_invp_lv1", 0x113C}},
    {F, {"bias_ctl", 0, 7, 0x42}},
    {F, {"unused0", 7, 1, 0x0}},
    {F, {"buf_stg", 16, 2, 0x1}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_req_pu_lv1", 0x1140}},
    {F, {"bias_ctl", 0, 8, 0x74}},
    {F, {"unused0", 16, 2, 0x0}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x1}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bias_sm_pdy_lv1", 0x1144}},
    {F, {"bias_ctl", 0, 8, 0xA4}},
    {F, {"unused0", 16, 2, 0x0}},
    {F, {"unused1", 19, 2, 0x0}},
    {F, {"bias_en", 24, 1, 0x1}},
    {F, {"pull_sel", 25, 1, 0x0}},
    {F, {"single", 28, 1, 0x0}},

    {R, {"bias/bgen_spr", 0x1200}},
    {F, {"bias_spare", 0, 16, 0x0}},

    {R, {"bias/bgen_timer", 0x1204}},
    {F, {"bias_timer", 0, 8, 0xA}},

    {R, {"bias/bgen_ctrl", 0x1208}},
    {F, {"burst_transfer_hv_bank_0", 0, 1, 0x0}},
    {F, {"burst_transfer_hv_bank_1", 1, 1, 0x0}},
    {F, {"burst_transfer_lv_bank_0", 2, 1, 0x0}},
    {F, {"burst_transfer_lv_bank_1", 3, 1, 0x0}},
    {F, {"bias_rstn_hv", 4, 1, 0x0}},
    {F, {"bias_rstn_lv", 5, 1, 0x0}},

    {R, {"bias/bgen_bu_dft", 0x120C}},
    {F, {"bias_dft_ctl", 0, 4, 0x0}},
    {F, {"bias_dft_ctl_valid", 4, 1, 0x0}},

    {R, {"bias/bgen_bu_irpoly_dft", 0x1210}},
    {F, {"bias_irpoly_dft_en", 0, 1, 0x0}},
    {F, {"bias_irpoly_sel", 1, 1, 0x0}},

    {R, {"bias/bgen_ref_dft", 0x1214}},
    {F, {"bias_ref_dft_en", 0, 1, 0x0}},
    {F, {"bias_ref_dft_ctl", 1, 4, 0x0}},

    {R, {"bias/bgen_cgm_sub", 0x1218}},
    {F, {"bias_cgm_sub_en", 0, 1, 0x0}},
    {F, {"bias_cgm_sub_trim", 1, 4, 0x8}},
    {F, {"bias_cgm_sub_slp_ctl", 5, 4, 0x8}},

    {R, {"bias/bgen_cgm_sub_status", 0x121C}},
    {F, {"bias_cgm_sub_rdy_dyn", 0, 1, 0x0}},

    {R, {"bias/bgen_cc_en", 0x1220}},
    {F, {"bias_cc_hv_en", 0, 1, 0x0}},
    {F, {"bias_cc_lv_en", 1, 1, 0x0}},

    {R, {"bias/bgen_nmir_en", 0x1224}},
    {F, {"bias_irpoly_mir_n_hv_en", 0, 1, 0x0}},

    {R, {"bias/bgen_fes_shdio", 0x1228}},
    {F, {"bias_fes_shift_ctl", 0, 4, 0x7}},
    {F, {"bias_fes_diode_ctl", 4, 2, 0x0}},

    {R, {"bias/bgen_diff_on_vdsn", 0x122C}},
    {F, {"bias_vdsn_diff_on_ctl", 0, 3, 0x4}},

    {R, {"bias/bgen_diff_off_vdsn", 0x1230}},
    {F, {"bias_vdsn_diff_off_ctl", 0, 3, 0x4}},

    {R, {"roi/td_roi_x00", 0x2000}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x01", 0x2004}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x02", 0x2008}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x03", 0x200C}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x04", 0x2010}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x05", 0x2014}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x06", 0x2018}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x07", 0x201C}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x08", 0x2020}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x09", 0x2024}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x10", 0x2028}},
    {F, {"scan_px_left", 0, 1, 0x1}},
    {F, {"dummy_p_px_left", 1, 1, 0x1}},
    {F, {"dummy_m_px_left", 2, 2, 0x3}},
    {F, {"dummy_m_px_right", 4, 2, 0x3}},
    {F, {"dummy_p_px_right", 6, 1, 0x1}},
    {F, {"scan_and_test_px_right", 7, 1, 0x1}},

    {R, {"roi/td_roi_x_xor00", 0x2800}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x_xor01", 0x2804}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x_xor02", 0x2808}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x_xor03", 0x280C}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x_xor04", 0x2810}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x_xor05", 0x2814}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x_xor06", 0x2818}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x_xor07", 0x281C}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x_xor08", 0x2820}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x_xor09", 0x2824}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_x_xor10", 0x2828}},
    {F, {"scan_px_left", 0, 1, 0x1}},
    {F, {"dummy_p_px_left", 1, 1, 0x1}},
    {F, {"dummy_m_px_left", 2, 2, 0x3}},
    {F, {"dummy_m_px_right", 4, 2, 0x3}},
    {F, {"dummy_p_px_right", 6, 1, 0x1}},
    {F, {"scan_and_test_px_right", 7, 1, 0x1}},

    {R, {"roi/td_roi_y00", 0x3000}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_y01", 0x3004}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_y02", 0x3008}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_y03", 0x300C}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_y04", 0x3010}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_y05", 0x3014}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_y06", 0x3018}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_y07", 0x301C}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_y08", 0x3020}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_y09", 0x3024}},
    {F, {"effective", 0, 32, 0xFFFFFFFF}},

    {R, {"roi/td_roi_y10", 0x3028}},
    {F, {"test_row0", 0, 1, 0x0}},
    {F, {"test_row1", 1, 1, 0x0}},
    {F, {"test_row2", 2, 1, 0x0}},
    {F, {"test_row3", 3, 1, 0x0}},
    {F, {"test_row4", 4, 1, 0x0}},
    {F, {"test_row5", 5, 1, 0x0}},
    {F, {"test_row6", 6, 1, 0x0}},
    {F, {"test_row7", 7, 1, 0x0}},
    {F, {"test_row8", 8, 1, 0x0}},
    {F, {"test_row9", 9, 1, 0x0}},

    {R, {"ehc/pipeline_control", 0x4000}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"drop_nbackpressure", 1, 1, 0x0}},
    {F, {"bypass", 2, 1, 0x0}},

    {R, {"ehc/pipeline_status", 0x4004}},
    {F, {"empty", 0, 1, 0x1}},
    {F, {"busy", 1, 1, 0x0}},
    {F, {"deep_low_power_ready", 2, 1, 0x0}},

    {R, {"ehc/ehc_control", 0x4040}},
    {F, {"algo_sel", 0, 1, 0x1}},
    {F, {"trig_sel", 1, 1, 0x1}},

    {R, {"ehc/bits_splitting", 0x4044}},
    {F, {"negative_bit_length", 0, 4, 0x4}},
    {F, {"positive_bit_length", 4, 4, 0x4}},
    {F, {"out_16bits_padding_mode", 8, 1, 0x0}},

    {R, {"ehc/integration_period", 0x4048}},
    {F, {"value_us", 4, 13, 0x186A}},

    {R, {"ehc/event_rate_threshold", 0x404C}},
    {F, {"value", 0, 17, 0x186A0}},

    {R, {"ehc/initialization", 0x4100}},
    {F, {"req_init", 0, 1, 0x0}},
    {F, {"flag_init_busy", 1, 1, 0x0}},
    {F, {"flag_init_done", 2, 1, 0x0}},

    {R, {"ehc/icn_sram_control", 0x4104}},
    {F, {"req_trigger", 0, 1, 0x0}},
    {F, {"req_type", 1, 1, 0x0}},
    {F, {"data_sel", 2, 2, 0x0}},

    {R, {"ehc/icn_sram_address", 0x4108}},
    {F, {"x_addr", 0, 11, 0x0}},
    {F, {"y_addr", 16, 11, 0x0}},

    {R, {"ehc/icn_sram_data", 0x410C}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"ehc/flags", 0x4110}},
    {F, {"err_draining_retrig", 0, 1, 0x0}},
    {F, {"err_wrong_config", 1, 1, 0x0}},
    {F, {"fct_wake_up", 2, 1, 0x0}},

    {R, {"ehc/irq_control", 0x4114}},
    {F, {"err_draining_retrig_sel", 0, 1, 0x0}},
    {F, {"err_wrong_config_sel", 1, 1, 0x0}},
    {F, {"fct_wake_up_sel", 2, 1, 0x0}},
    {F, {"irq_hold_cycle_num", 16, 10, 0x32}},

    {R, {"ehc/wake_up_period", 0x4118}},
    {F, {"value_us", 4, 12, 0x0}},

    {R, {"ehc/chicken0_bits", 0x41C0}},
    {F, {"drain_blocking_mode", 0, 1, 0x0}},
    {F, {"drain_trashing_mode", 1, 1, 0x0}},
    {F, {"diff3d_sat", 2, 1, 0x0}},
    {F, {"timebase_rec_td_th_only", 3, 1, 0x0}},
    {F, {"force_drain_req", 4, 1, 0x0}},

    {R, {"erc/pipeline_control", 0x6000}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"drop_nbackpressure", 1, 1, 0x0}},
    {F, {"bypass", 2, 1, 0x0}},

    {R, {"erc/pipeline_status", 0x6004}},
    {F, {"empty", 0, 1, 0x1}},
    {F, {"busy", 1, 1, 0x0}},
    {F, {"deep_low_power_seen", 2, 1, 0x0}},

    {R, {"erc/pipeline_flush", 0x6010}},
    {F, {"in_progress", 0, 2, 0x0}},
    {F, {"flush_cmd_en", 30, 1, 0x1}},
    {F, {"status", 31, 1, 0x1}},

    {R, {"erc/ahvt_dropping_control", 0x6014}},
    {F, {"h_dropping_en", 0, 1, 0x0}},
    {F, {"v_dropping_en", 1, 1, 0x0}},
    {F, {"t_dropping_en", 2, 1, 0x0}},
    {F, {"t_dropping_lut_en", 3, 1, 0x0}},
    {F, {"drop_all_td_when_drop_geq", 4, 10, 0x201}},
    {F, {"status", 31, 1, 0x0}},

    {R, {"erc/ref_period_flavor", 0x6020}},
    {F, {"reference_period", 0, 10, 0x80}},
    {F, {"avg_drop_rate_delayed", 16, 1, 0x0}},

    {R, {"erc/delay_fifo_non_td_rsvd_area", 0x6024}},
    {F, {"val", 0, 13, 0x101}},
    {F, {"auto_raise", 16, 1, 0x0}},

    {R, {"erc/delay_fifo_size", 0x6028}},
    {F, {"val", 0, 15, 0x1408}},

    {R, {"erc/td_target_event_count", 0x602C}},
    {F, {"val", 0, 21, 0x80}},

    {R, {"erc/interest_level", 0x6030}},
    {F, {"select_pong_intlvl_grid", 0, 1, 0x0}},
    {F, {"use_intlvl_grid", 1, 1, 0x0}},
    {F, {"weak_threshold", 2, 4, 0x0}},
    {F, {"status", 31, 1, 0x1}},

    {R, {"erc/monitoring_event_control", 0x6034}},
    {F, {"first_module_tag_en", 0, 1, 0x0}},
    {F, {"avg_drop_rate_en", 1, 1, 0x0}},
    {F, {"in_td_cnt_en", 2, 1, 0x0}},
    {F, {"df_td_vect_drop_cnt_en", 3, 1, 0x0}},
    {F, {"df_non_td_vect_drop_cnt_en", 4, 1, 0x0}},
    {F, {"alldr_evt_drop_cnt_en", 5, 1, 0x0}},
    {F, {"hdr_evt_drop_cnt_en", 6, 1, 0x0}},
    {F, {"vdr_evt_drop_cnt_en", 7, 1, 0x0}},
    {F, {"tdr_evt_drop_cnt_en", 8, 1, 0x0}},
    {F, {"erc_td_evt_cnt_en", 9, 1, 0x0}},
    {F, {"last_module_tag_en", 10, 1, 0x0}},

    {R, {"erc/irq_pending", 0x6038}},
    {F, {"icn_cfg_error", 0, 1, 0x0}},
    {F, {"icn_hdr_error", 1, 1, 0x0}},
    {F, {"icn_vdr_error", 2, 1, 0x0}},
    {F, {"icn_tdr_error", 3, 1, 0x0}},
    {F, {"icn_ilg_error", 4, 1, 0x0}},
    {F, {"df_non_td_drop", 5, 1, 0x0}},
    {F, {"df_td_drop", 6, 1, 0x0}},
    {F, {"interest_level_cmd_done", 7, 1, 0x0}},
    {F, {"stat_snap_update", 8, 1, 0x0}},
    {F, {"user_pipe_flush_done", 9, 1, 0x0}},
    {F, {"ahvt_dropping_cmd_done", 10, 1, 0x0}},
    {F, {"deep_low_power_seen", 11, 1, 0x0}},
    {F, {"sram_power_up_down_done", 28, 1, 0x0}},
    {F, {"first_time_high_error", 29, 1, 0x0}},
    {F, {"exit_deep_low_power_error", 30, 1, 0x0}},
    {F, {"df_sram_bypass_done", 31, 1, 0x0}},

    {R, {"erc/irq_status", 0x603C}},
    {F, {"icn_cfg_error", 0, 1, 0x0}},
    {F, {"icn_hdr_error", 1, 1, 0x0}},
    {F, {"icn_vdr_error", 2, 1, 0x0}},
    {F, {"icn_tdr_error", 3, 1, 0x0}},
    {F, {"icn_ilg_error", 4, 1, 0x0}},
    {F, {"df_non_td_drop", 5, 1, 0x0}},
    {F, {"df_td_drop", 6, 1, 0x0}},
    {F, {"interest_level_cmd_done", 7, 1, 0x0}},
    {F, {"stat_snap_update", 8, 1, 0x0}},
    {F, {"user_pipe_flush_done", 9, 1, 0x0}},
    {F, {"ahvt_dropping_cmd_done", 10, 1, 0x0}},
    {F, {"deep_low_power_seen", 11, 1, 0x0}},
    {F, {"sram_power_up_down_done", 28, 1, 0x0}},
    {F, {"first_time_high_error", 29, 1, 0x0}},
    {F, {"exit_deep_low_power_error", 30, 1, 0x0}},
    {F, {"df_sram_bypass_done", 31, 1, 0x0}},

    {R, {"erc/irq_mask", 0x6040}},
    {F, {"icn_cfg_error", 0, 1, 0x0}},
    {F, {"icn_hdr_error", 1, 1, 0x0}},
    {F, {"icn_vdr_error", 2, 1, 0x0}},
    {F, {"icn_tdr_error", 3, 1, 0x0}},
    {F, {"icn_ilg_error", 4, 1, 0x0}},
    {F, {"df_non_td_drop", 5, 1, 0x0}},
    {F, {"df_td_drop", 6, 1, 0x0}},
    {F, {"interest_level_cmd_done", 7, 1, 0x0}},
    {F, {"stat_snap_update", 8, 1, 0x0}},
    {F, {"user_pipe_flush_done", 9, 1, 0x0}},
    {F, {"ahvt_dropping_cmd_done", 10, 1, 0x0}},
    {F, {"deep_low_power_seen", 11, 1, 0x0}},
    {F, {"sram_power_up_down_done", 28, 1, 0x0}},
    {F, {"first_time_high_error", 29, 1, 0x0}},
    {F, {"exit_deep_low_power_error", 30, 1, 0x0}},
    {F, {"df_sram_bypass_done", 31, 1, 0x0}},

    {R, {"erc/irq_df_non_td_drop_cnt", 0x6044}},
    {F, {"val", 0, 16, 0x0}},

    {R, {"erc/irq_df_td_drop_cnt", 0x6048}},
    {F, {"val", 0, 16, 0x0}},

    {R, {"erc/shadow_ctrl", 0x604C}},
    {F, {"timer_en", 0, 1, 0x0}},
    {F, {"irq_sw_override", 1, 1, 0x0}},
    {F, {"reset_on_copy", 2, 1, 0x1}},

    {R, {"erc/shadow_timer_threshold", 0x6050}},
    {F, {"timer_threshold", 0, 16, 0x1}},

    {R, {"erc/shadow_status", 0x6054}},
    {F, {"shadow_valid", 0, 1, 0x0}},
    {F, {"shadow_overrun", 1, 1, 0x0}},

    {R, {"erc/stat_in_td_event_count", 0x6058}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"erc/stat_erc_td_event_count", 0x605C}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"erc/stat_df_td_event_drop_count", 0x6060}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"erc/stat_df_non_td_event_drop_count", 0x6064}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"erc/stat_alldrop_td_event_drop_count", 0x6068}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"erc/stat_hdrop_td_event_drop_count", 0x606C}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"erc/stat_vdrop_td_event_drop_count", 0x6070}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"erc/stat_tdrop_td_event_drop_count", 0x6074}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"erc/stat_other_drop_15_0", 0x6078}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"erc/stat_other_drop_31_16", 0x607C}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"erc/snap_first_last_module_tag", 0x6080}},
    {F, {"start_seq", 0, 1, 0x0}},
    {F, {"select_pong", 1, 1, 0x0}},
    {F, {"intlvl_en", 2, 1, 0x0}},
    {F, {"weak_thold", 3, 4, 0x0}},
    {F, {"adr_delayed", 7, 1, 0x0}},
    {F, {"all_dr_geq", 8, 10, 0x0}},
    {F, {"hdr_en", 18, 1, 0x0}},
    {F, {"vdr_en", 19, 1, 0x0}},
    {F, {"tdr_en", 20, 1, 0x0}},
    {F, {"tdr_lut_en", 21, 1, 0x0}},
    {F, {"flush_erc_disable", 24, 1, 0x0}},
    {F, {"flush_icn_cmd", 25, 1, 0x0}},
    {F, {"flush_deep_low_power", 26, 1, 0x0}},
    {F, {"last_cycle_timeout", 27, 1, 0x0}},
    {F, {"last_ffwd_timeout", 28, 1, 0x0}},
    {F, {"last_before_bypass", 29, 1, 0x0}},

    {R, {"erc/snap_avg_drop_rate", 0x6084}},
    {F, {"iD", 0, 16, 0x0}},
    {F, {"other_drops", 16, 2, 0x0}},

    {R, {"erc/sram_power_down", 0x6090}},
    {F, {"delay_fifo_pd", 0, 1, 0x0}},
    {F, {"intlvl_grid_pd", 1, 1, 0x0}},
    {F, {"tdrop_mem_pd", 2, 1, 0x0}},
    {F, {"delay_fifo_pd_status", 16, 1, 0x1}},
    {F, {"intlvl_grid_pd_status", 17, 1, 0x1}},
    {F, {"tdrop_mem_pd_status", 18, 1, 0x1}},
    {F, {"pd_status", 31, 1, 0x1}},

    {R, {"erc/sram_read_write_cycle_timing", 0x6094}},
    {F, {"adr_intlvl_grid0_wtsel", 0, 2, 0x1}},
    {F, {"adr_intlvl_grid0_rtsel", 2, 2, 0x1}},
    {F, {"edr_intlvl_grid0_wtsel", 8, 2, 0x1}},
    {F, {"edr_intlvl_grid0_rtsel", 10, 2, 0x1}},
    {F, {"t_drop_wtsel", 16, 2, 0x1}},
    {F, {"t_drop_rtsel", 18, 2, 0x1}},

    {R, {"erc/intlvl_read_bank_sel", 0x6098}},
    {F, {"bank_id", 0, 2, 0x0}},

    {R, {"erc/end_of_period_cycles_per_us", 0x609C}},
    {F, {"nb_cycles", 0, 12, 0x0}},

    {R, {"erc/delay_fifo_flush_and_bypass", 0x60A0}},
    {F, {"en", 0, 1, 0x1}},
    {F, {"status", 31, 1, 0x0}},

    {R, {"erc/timestamp_from_th_td_only", 0x60A4}},
    {F, {"en", 0, 1, 0x0}},

    {R, {"erc/manual_td_avg_drop_rate", 0x60A8}},
    {F, {"iD", 0, 16, 0x0}},
    {F, {"other_drops", 16, 2, 0x0}},
    {F, {"en", 31, 1, 0x0}},

    {R, {"erc/manual_td_evt_drop_rate", 0x60AC}},
    {F, {"en", 0, 1, 0x0}},
    {F, {"Dij", 1, 10, 0x0}},

    {R, {"erc/force_immediate_change", 0x60B0}},
    {F, {"ahvt_dropping_control", 0, 1, 0x0}},
    {F, {"interest_level", 1, 1, 0x0}},

    {R, {"erc/reset_tdrop_counter_on_mtag_first", 0x60B4}},
    {F, {"en", 0, 1, 0x1}},

    {R, {"erc/delay_fifo_bypass_wo_back_pressure", 0x60B8}},
    {F, {"en", 0, 1, 0x0}},

    {R, {"erc/nice_to_have", 0x60BC}},
    {F, {"eos_on_new_ref_period", 0, 1, 0x0}},
    {F, {"bugfree_event_period_cycle_cnt", 1, 1, 0x0}},
    {F, {"add_cycle_timeout_irq", 2, 1, 0x0}},
    {F, {"add_sram_power_state_irq", 3, 1, 0x1}},

    {R, {"erc/h_dropping_lut_00", 0x6100}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/h_dropping_lut_01", 0x6104}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/h_dropping_lut_02", 0x6108}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/h_dropping_lut_03", 0x610C}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/h_dropping_lut_04", 0x6110}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/h_dropping_lut_05", 0x6114}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/h_dropping_lut_06", 0x6118}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/h_dropping_lut_07", 0x611C}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/h_dropping_lut_08", 0x6120}},
    {F, {"v0", 0, 5, 0x0}},

    {R, {"erc/v_dropping_lut_00", 0x6140}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/v_dropping_lut_01", 0x6144}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/v_dropping_lut_02", 0x6148}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/v_dropping_lut_03", 0x614C}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/v_dropping_lut_04", 0x6150}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/v_dropping_lut_05", 0x6154}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/v_dropping_lut_06", 0x6158}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/v_dropping_lut_07", 0x615C}},
    {F, {"v0", 0, 5, 0x0}},
    {F, {"v1", 8, 5, 0x0}},
    {F, {"v2", 16, 5, 0x0}},
    {F, {"v3", 24, 5, 0x0}},

    {R, {"erc/v_dropping_lut_08", 0x6160}},
    {F, {"v0", 0, 5, 0x0}},

    {R, {"erc/t_dropping_lut_000", 0x6400}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_001", 0x6404}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_002", 0x6408}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_003", 0x640C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_004", 0x6410}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_005", 0x6414}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_006", 0x6418}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_007", 0x641C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_008", 0x6420}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_009", 0x6424}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_010", 0x6428}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_011", 0x642C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_012", 0x6430}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_013", 0x6434}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_014", 0x6438}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_015", 0x643C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_016", 0x6440}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_017", 0x6444}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_018", 0x6448}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_019", 0x644C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_020", 0x6450}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_021", 0x6454}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_022", 0x6458}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_023", 0x645C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_024", 0x6460}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_025", 0x6464}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_026", 0x6468}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_027", 0x646C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_028", 0x6470}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_029", 0x6474}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_030", 0x6478}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_031", 0x647C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_032", 0x6480}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_033", 0x6484}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_034", 0x6488}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_035", 0x648C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_036", 0x6490}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_037", 0x6494}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_038", 0x6498}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_039", 0x649C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_040", 0x64A0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_041", 0x64A4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_042", 0x64A8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_043", 0x64AC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_044", 0x64B0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_045", 0x64B4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_046", 0x64B8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_047", 0x64BC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_048", 0x64C0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_049", 0x64C4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_050", 0x64C8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_051", 0x64CC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_052", 0x64D0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_053", 0x64D4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_054", 0x64D8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_055", 0x64DC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_056", 0x64E0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_057", 0x64E4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_058", 0x64E8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_059", 0x64EC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_060", 0x64F0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_061", 0x64F4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_062", 0x64F8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_063", 0x64FC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_064", 0x6500}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_065", 0x6504}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_066", 0x6508}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_067", 0x650C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_068", 0x6510}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_069", 0x6514}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_070", 0x6518}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_071", 0x651C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_072", 0x6520}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_073", 0x6524}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_074", 0x6528}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_075", 0x652C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_076", 0x6530}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_077", 0x6534}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_078", 0x6538}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_079", 0x653C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_080", 0x6540}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_081", 0x6544}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_082", 0x6548}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_083", 0x654C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_084", 0x6550}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_085", 0x6554}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_086", 0x6558}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_087", 0x655C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_088", 0x6560}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_089", 0x6564}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_090", 0x6568}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_091", 0x656C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_092", 0x6570}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_093", 0x6574}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_094", 0x6578}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_095", 0x657C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_096", 0x6580}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_097", 0x6584}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_098", 0x6588}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_099", 0x658C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_100", 0x6590}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_101", 0x6594}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_102", 0x6598}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_103", 0x659C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_104", 0x65A0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_105", 0x65A4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_106", 0x65A8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_107", 0x65AC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_108", 0x65B0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_109", 0x65B4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_110", 0x65B8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_111", 0x65BC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_112", 0x65C0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_113", 0x65C4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_114", 0x65C8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_115", 0x65CC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_116", 0x65D0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_117", 0x65D4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_118", 0x65D8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_119", 0x65DC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_120", 0x65E0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_121", 0x65E4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_122", 0x65E8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_123", 0x65EC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_124", 0x65F0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_125", 0x65F4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_126", 0x65F8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_127", 0x65FC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_128", 0x6600}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_129", 0x6604}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_130", 0x6608}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_131", 0x660C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_132", 0x6610}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_133", 0x6614}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_134", 0x6618}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_135", 0x661C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_136", 0x6620}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_137", 0x6624}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_138", 0x6628}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_139", 0x662C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_140", 0x6630}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_141", 0x6634}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_142", 0x6638}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_143", 0x663C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_144", 0x6640}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_145", 0x6644}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_146", 0x6648}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_147", 0x664C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_148", 0x6650}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_149", 0x6654}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_150", 0x6658}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_151", 0x665C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_152", 0x6660}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_153", 0x6664}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_154", 0x6668}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_155", 0x666C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_156", 0x6670}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_157", 0x6674}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_158", 0x6678}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_159", 0x667C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_160", 0x6680}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_161", 0x6684}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_162", 0x6688}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_163", 0x668C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_164", 0x6690}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_165", 0x6694}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_166", 0x6698}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_167", 0x669C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_168", 0x66A0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_169", 0x66A4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_170", 0x66A8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_171", 0x66AC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_172", 0x66B0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_173", 0x66B4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_174", 0x66B8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_175", 0x66BC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_176", 0x66C0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_177", 0x66C4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_178", 0x66C8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_179", 0x66CC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_180", 0x66D0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_181", 0x66D4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_182", 0x66D8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_183", 0x66DC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_184", 0x66E0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_185", 0x66E4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_186", 0x66E8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_187", 0x66EC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_188", 0x66F0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_189", 0x66F4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_190", 0x66F8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_191", 0x66FC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_192", 0x6700}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_193", 0x6704}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_194", 0x6708}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_195", 0x670C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_196", 0x6710}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_197", 0x6714}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_198", 0x6718}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_199", 0x671C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_200", 0x6720}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_201", 0x6724}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_202", 0x6728}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_203", 0x672C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_204", 0x6730}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_205", 0x6734}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_206", 0x6738}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_207", 0x673C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_208", 0x6740}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_209", 0x6744}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_210", 0x6748}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_211", 0x674C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_212", 0x6750}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_213", 0x6754}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_214", 0x6758}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_215", 0x675C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_216", 0x6760}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_217", 0x6764}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_218", 0x6768}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_219", 0x676C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_220", 0x6770}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_221", 0x6774}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_222", 0x6778}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_223", 0x677C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_224", 0x6780}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_225", 0x6784}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_226", 0x6788}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_227", 0x678C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_228", 0x6790}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_229", 0x6794}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_230", 0x6798}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_231", 0x679C}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_232", 0x67A0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_233", 0x67A4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_234", 0x67A8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_235", 0x67AC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_236", 0x67B0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_237", 0x67B4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_238", 0x67B8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_239", 0x67BC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_240", 0x67C0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_241", 0x67C4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_242", 0x67C8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_243", 0x67CC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_244", 0x67D0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_245", 0x67D4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_246", 0x67D8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_247", 0x67DC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_248", 0x67E0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_249", 0x67E4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_250", 0x67E8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_251", 0x67EC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_252", 0x67F0}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_253", 0x67F4}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_254", 0x67F8}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/t_dropping_lut_255", 0x67FC}},
    {F, {"v0", 0, 9, 0x0}},
    {F, {"v1", 16, 9, 0x0}},

    {R, {"erc/pong_drop_intlevel_000", 0x6800}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_000", 0x6800}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_001", 0x6804}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_001", 0x6804}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_002", 0x6808}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_002", 0x6808}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_003", 0x680C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_003", 0x680C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_004", 0x6810}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_004", 0x6810}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_005", 0x6814}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_005", 0x6814}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_006", 0x6818}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_006", 0x6818}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_007", 0x681C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_007", 0x681C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_008", 0x6820}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_008", 0x6820}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_009", 0x6824}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_009", 0x6824}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_010", 0x6828}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_010", 0x6828}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_011", 0x682C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_011", 0x682C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_012", 0x6830}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_012", 0x6830}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_013", 0x6834}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_013", 0x6834}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_014", 0x6838}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_014", 0x6838}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_015", 0x683C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_015", 0x683C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_016", 0x6840}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_016", 0x6840}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_017", 0x6844}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_017", 0x6844}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_018", 0x6848}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_018", 0x6848}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/pong_drop_intlevel_019", 0x684C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/ping_drop_intlevel_019", 0x684C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_000", 0x6880}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_001", 0x6884}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_002", 0x6888}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_003", 0x688C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_004", 0x6890}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_005", 0x6894}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_006", 0x6898}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_007", 0x689C}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_008", 0x68A0}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_009", 0x68A4}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_010", 0x68A8}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_011", 0x68AC}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_012", 0x68B0}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_013", 0x68B4}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_014", 0x68B8}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_015", 0x68BC}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_016", 0x68C0}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_017", 0x68C4}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_018", 0x68C8}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"erc/intlevel_readonly_bank_019", 0x68CC}},
    {F, {"v0", 0, 4, 0x0}},
    {F, {"v1", 4, 4, 0x0}},
    {F, {"v2", 8, 4, 0x0}},
    {F, {"v3", 12, 4, 0x0}},
    {F, {"v4", 16, 4, 0x0}},

    {R, {"edf/pipeline_control", 0x7000}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"drop_nbackpressure", 1, 1, 0x0}},
    {F, {"bypass", 2, 1, 0x0}},

    {R, {"edf/pipeline_status", 0x7004}},
    {F, {"empty", 0, 1, 0x1}},
    {F, {"busy", 1, 1, 0x0}},
    {F, {"deep_low_power_seen", 2, 1, 0x0}},

    {R, {"edf/event_type_en", 0x7040}},
    {F, {"en_left_td_low", 0, 1, 0x1}},
    {F, {"en_left_td_high", 1, 1, 0x1}},
    {F, {"en_left_aps_low", 2, 1, 0x1}},
    {F, {"en_left_aps_high", 3, 1, 0x1}},
    {F, {"en_right_td_low", 4, 1, 0x1}},
    {F, {"en_right_td_high", 5, 1, 0x1}},
    {F, {"en_right_aps_low", 6, 1, 0x1}},
    {F, {"en_right_aps_high", 7, 1, 0x1}},
    {F, {"en_time_high", 8, 1, 0x1}},
    {F, {"en_stereo_disp", 9, 1, 0x0}},
    {F, {"en_ext_trigger", 10, 1, 0x1}},
    {F, {"en_gray_level", 11, 1, 0x0}},
    {F, {"en_opt_flow", 12, 1, 0x0}},
    {F, {"en_orientation", 13, 1, 0x0}},
    {F, {"en_others_and_linked_continued", 14, 1, 0x1}},
    {F, {"en_continued_alone", 15, 1, 0x1}},

    {R, {"edf/control", 0x7044}},
    {F, {"format", 0, 2, 0x0}},
    {F, {"reserved", 2, 2, 0x0}},
    {F, {"endianness", 4, 1, 0x0}},

    {R, {"edf/event_injection", 0x7048}},
    {F, {"sysmon_end_of_frame_en", 0, 1, 0x0}},

    {R, {"edf/output_interface_control", 0x704C}},
    {F, {"start_of_frame_timeout", 4, 12, 0x7D}},

    {R, {"edf/external_output_adapter", 0x7100}},
    {F, {"qos_timeout", 0, 16, 0xFFFF}},
    {F, {"atomic_qos_mode", 16, 1, 0x0}},

    {R, {"edf/chicken0_bits", 0x71C0}},
    {F, {"legacy_evt21_endianness", 0, 1, 0x0}},

    {R, {"cpi/pipeline_control", 0x8000}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"drop_nbackpressure", 1, 1, 0x0}},
    {F, {"output_fifo_bypass", 3, 1, 0x0}},
    {F, {"output_data_format", 4, 1, 0x0}},
    {F, {"output_if_mode", 5, 1, 0x0}},
    {F, {"output_width", 6, 1, 0x0}},
    {F, {"packet_fixed_size_enable", 7, 1, 0x0}},
    {F, {"packet_fixed_rate_enable", 8, 1, 0x0}},
    {F, {"frame_fixed_size_enable", 9, 1, 0x0}},
    {F, {"clk_out_en", 10, 1, 0x0}},
    {F, {"clk_control_inversion", 11, 1, 0x1}},
    {F, {"clk_out_gating_enable", 12, 1, 0x0}},
    {F, {"clk_timeout", 13, 8, 0xF}},
    {F, {"packet_pad_empty_enable", 21, 1, 0x0}},
    {F, {"hot_disable_enable", 22, 1, 0x0}},

    {R, {"cpi/pipeline_status", 0x8004}},
    {F, {"empty", 0, 1, 0x1}},
    {F, {"busy", 1, 1, 0x0}},
    {F, {"deep_low_power_ready", 2, 1, 0x0}},
    {F, {"no_atomic_packet", 3, 1, 0x0}},

    {R, {"cpi/packet_size_control", 0x8008}},
    {F, {"packet_size", 0, 13, 0x800}},

    {R, {"cpi/packet_time_control", 0x800C}},
    {F, {"packet_period", 0, 16, 0x810}},
    {F, {"packet_blanking", 16, 16, 0x1}},

    {R, {"cpi/frame_size_control", 0x8010}},
    {F, {"frame_size", 0, 16, 0x20}},

    {R, {"cpi/frame_time_control", 0x8014}},
    {F, {"frame_blanking", 0, 16, 0x1}},

    {R, {"cpi/output_if_control", 0x8018}},
    {F, {"output_if_hsync_pol", 0, 1, 0x0}},
    {F, {"output_if_vsync_pol", 1, 1, 0x0}},

    {R, {"cpi/pkt_self_test", 0x801C}},
    {F, {"self_test_ctrl", 0, 3, 0x0}},
    {F, {"self_test_rate", 3, 10, 0x0}},
    {F, {"self_test_repeat", 16, 13, 0x0}},

    {R, {"cpi/pkt_self_test_data", 0x8020}},
    {F, {"self_test_data", 0, 8, 0x0}},

    {R, {"ro/readout_ctrl", 0x9000}},
    {F, {"ro_test_pixel_mux_en", 0, 1, 0x0}},
    {F, {"ro_self_test_en", 1, 1, 0x0}},
    {F, {"cpm_record_mode_en", 2, 1, 0x0}},
    {F, {"ro_analog_pipe_en", 3, 1, 0x0}},
    {F, {"erc_self_test_en", 4, 1, 0x0}},
    {F, {"ro_inv_pol_td", 5, 1, 0x0}},
    {F, {"ro_flip_x", 6, 1, 0x0}},
    {F, {"ro_flip_y", 7, 1, 0x0}},
    {F, {"ro_digital_pipe_en", 9, 1, 0x0}},
    {F, {"ro_avoid_bpress_td", 10, 1, 0x0}},
    {F, {"drop_en", 12, 1, 0x0}},
    {F, {"drop_on_full_en", 13, 1, 0x0}},
    {F, {"delay_ro_td_int_x_act_fal", 14, 4, 0x0}},
    {F, {"delay_ro_td_int_x_act_ris", 18, 4, 0x0}},

    {R, {"ro/ro_fsm_ctrl", 0x9004}},
    {F, {"delay_sample_ro_line", 0, 9, 0x0}},
    {F, {"delay_release_ro_ack", 9, 9, 0x0}},

    {R, {"ro/time_base_ctrl", 0x9008}},
    {F, {"time_base_enable", 0, 1, 0x0}},
    {F, {"time_base_mode", 1, 1, 0x0}},
    {F, {"external_mode", 2, 1, 0x0}},
    {F, {"external_mode_enable", 3, 1, 0x0}},
    {F, {"us_counter_max", 4, 7, 0x32}},
    {F, {"th_every_64us_en", 11, 1, 0x0}},
    {F, {"time_base_srst", 16, 1, 0x0}},

    {R, {"ro/oor_ctrl", 0x900C}},
    {F, {"oor_crop_enable", 0, 1, 0x0}},
    {F, {"oor_detect_enable", 1, 1, 0x0}},
    {F, {"oor_rm_td", 2, 1, 0x0}},
    {F, {"oor_crop_reset_orig", 4, 1, 0x0}},

    {R, {"ro/oor_start_pos", 0x9010}},
    {F, {"oor_crop_start_x", 0, 9, 0x0}},
    {F, {"oor_crop_start_y", 16, 9, 0x0}},

    {R, {"ro/oor_end_pos", 0x9014}},
    {F, {"oor_crop_end_x", 0, 9, 0x0}},
    {F, {"oor_crop_end_y", 16, 9, 0x0}},

    {R, {"ro/oor_td_cnt", 0x9018}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"ro/self_test_data", 0x901C}},
    {F, {"self_test_data", 0, 32, 0x5A5A5A5A}},

    {R, {"ro/oor_td_addr", 0x9020}},
    {F, {"oor_td_x_addr", 0, 11, 0x0}},
    {F, {"oor_td_y_addr", 11, 10, 0x0}},
    {F, {"oor_td_pol", 21, 1, 0x0}},

    {R, {"ro/pipeline_status", 0x9024}},
    {F, {"empty", 0, 1, 0x1}},
    {F, {"busy", 1, 1, 0x0}},
    {F, {"deep_low_power_seen", 2, 1, 0x0}},

    {R, {"ro/ro_lp_ctrl", 0x9028}},
    {F, {"lp_cnt_en", 0, 1, 0x0}},
    {F, {"lp_output_disable", 1, 1, 0x0}},
    {F, {"lp_keep_th", 2, 1, 0x0}},

    {R, {"ro/lp_x0", 0x902C}},
    {F, {"x0_addr", 0, 9, 0x0}},

    {R, {"ro/lp_x1", 0x9030}},
    {F, {"x1_addr", 0, 9, 0x40}},

    {R, {"ro/lp_x2", 0x9034}},
    {F, {"x2_addr", 0, 9, 0xA0}},

    {R, {"ro/lp_x3", 0x9038}},
    {F, {"x3_addr", 0, 9, 0x100}},

    {R, {"ro/lp_x4", 0x903C}},
    {F, {"x4_addr", 0, 9, 0x140}},

    {R, {"ro/lp_y0", 0x9040}},
    {F, {"y0_addr", 0, 9, 0x0}},

    {R, {"ro/lp_y1", 0x9044}},
    {F, {"y1_addr", 0, 9, 0x40}},

    {R, {"ro/lp_y2", 0x9048}},
    {F, {"y2_addr", 0, 9, 0xA0}},

    {R, {"ro/lp_y3", 0x904C}},
    {F, {"y3_addr", 0, 9, 0x100}},

    {R, {"ro/lp_y4", 0x9050}},
    {F, {"y4_addr", 0, 9, 0x140}},

    {R, {"ro/shadow_ctrl", 0x9054}},
    {F, {"timer_en", 0, 1, 0x0}},
    {F, {"irq_sw_override", 1, 1, 0x0}},
    {F, {"reset_on_copy", 2, 1, 0x1}},

    {R, {"ro/shadow_timer_threshold", 0x9058}},
    {F, {"timer_threshold", 0, 32, 0x3E8}},

    {R, {"ro/shadow_status", 0x905C}},
    {F, {"shadow_valid", 0, 1, 0x0}},
    {F, {"shadow_overrun", 1, 1, 0x0}},

    {R, {"ro/sw_evt_bypass_msb", 0x9060}},
    {F, {"sw_evt_bypass_msb_val", 0, 32, 0x0}},

    {R, {"ro/sw_evt_bypass_lsb", 0x9064}},
    {F, {"sw_evt_bypass_lsb_val", 0, 32, 0x0}},

    {R, {"ro/sw_evt_bypass_ctrl", 0x9068}},
    {F, {"sw_evt_bypass_en", 0, 1, 0x0}},
    {F, {"sw_evt_bypass_trig", 1, 1, 0x0}},
    {F, {"sw_evt_bypass_first", 2, 1, 0x0}},
    {F, {"sw_evt_bypass_status", 3, 1, 0x0}},

    {R, {"ro/crazy_pixel_ctrl00", 0x9100}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data00", 0x9104}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl01", 0x9108}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data01", 0x910C}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl02", 0x9110}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data02", 0x9114}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl03", 0x9118}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data03", 0x911C}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl04", 0x9120}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data04", 0x9124}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl05", 0x9128}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data05", 0x912C}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl06", 0x9130}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data06", 0x9134}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl07", 0x9138}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data07", 0x913C}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl08", 0x9140}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data08", 0x9144}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl09", 0x9148}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data09", 0x914C}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl10", 0x9150}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data10", 0x9154}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl11", 0x9158}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data11", 0x915C}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl12", 0x9160}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data12", 0x9164}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl13", 0x9168}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data13", 0x916C}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl14", 0x9170}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data14", 0x9174}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/crazy_pixel_ctrl15", 0x9178}},
    {F, {"x_group", 0, 4, 0x0}},
    {F, {"y", 4, 9, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"ro/crazy_pixel_data15", 0x917C}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"ro/lp_cnt00", 0x9200}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt01", 0x9204}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt02", 0x9208}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt03", 0x920C}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt04", 0x9210}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt05", 0x9214}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt06", 0x9218}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt07", 0x921C}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt08", 0x9220}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt09", 0x9224}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt10", 0x9228}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt11", 0x922C}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt12", 0x9230}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt13", 0x9234}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt14", 0x9238}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/lp_cnt15", 0x923C}},
    {F, {"lp_cnt_val", 0, 32, 0x0}},

    {R, {"ro/evt21_cd_dropped_cnt", 0x9240}},
    {F, {"evt21_cd_dropped_cnt_val", 0, 32, 0x0}},

    {R, {"ro/evt_cd_cnt", 0x9244}},
    {F, {"evt_cd_cnt_val", 0, 32, 0x0}},

    {R, {"ro/evt21_cd_cnt", 0x9248}},
    {F, {"evt21_cd_cnt_val", 0, 32, 0x0}},

    {R, {"ro/line_cd_cnt", 0x924C}},
    {F, {"line_cd_cnt_val", 0, 32, 0x0}},

    {R, {"ldo/bg", 0xA000}},
    {F, {"pmu_ulp_en", 0, 1, 0x0}},
    {F, {"bg_en", 1, 1, 0x0}},
    {F, {"bg_buf_en", 2, 1, 0x0}},
    {F, {"bg_bypass", 3, 2, 0x0}},
    {F, {"bg_adj", 5, 8, 0x20}},
    {F, {"bg_th", 13, 3, 0x0}},
    {F, {"bg_force_start", 16, 1, 0x0}},
    {F, {"bg_chk", 17, 1, 0x0}},
    {F, {"bg_ind_out_dyn", 18, 1, 0x0}},

    {R, {"ldo/ldo_hv", 0xA004}},
    {F, {"ldo_hv_en", 0, 1, 0x0}},
    {F, {"ldo_hv_climit_en", 1, 1, 0x0}},
    {F, {"ldo_hv_climit", 2, 2, 0x1}},
    {F, {"ldo_hv_adj", 4, 4, 0x2}},
    {F, {"ldo_hv_bypass", 8, 1, 0x0}},
    {F, {"ldo_hv_ron", 9, 2, 0x0}},
    {F, {"ldo_hv_comp", 11, 2, 0x0}},
    {F, {"ldo_hv_pwrup", 13, 1, 0x0}},
    {F, {"ldo_hv_gain_dwn", 14, 1, 0x0}},
    {F, {"ldo_hv_ind_en", 15, 1, 0x0}},
    {F, {"ldo_hv_ind_slw", 16, 1, 0x0}},
    {F, {"ldo_hv_ind_vth_ok", 17, 1, 0x0}},
    {F, {"ldo_hv_ind_vth_bo", 18, 3, 0x0}},
    {F, {"ldo_hv_ind_out_dyn", 21, 1, 0x0}},
    {F, {"ldo_hv_en_delay", 22, 1, 0x0}},
    {F, {"ldo_hv_start_pulse", 23, 1, 0x0}},
    {F, {"ldo_hv_start_trig", 24, 1, 0x0}},

    {R, {"ldo/ldo_lv", 0xA008}},
    {F, {"ldo_lv_en", 0, 1, 0x0}},
    {F, {"ldo_lv_climit_en", 1, 1, 0x0}},
    {F, {"ldo_lv_climit", 2, 2, 0x1}},
    {F, {"ldo_lv_adj", 4, 4, 0x8}},
    {F, {"ldo_lv_bypass", 8, 1, 0x0}},
    {F, {"ldo_lv_ron", 9, 2, 0x0}},
    {F, {"ldo_lv_comp", 11, 2, 0x0}},
    {F, {"ldo_lv_pwrup", 13, 1, 0x0}},
    {F, {"ldo_lv_gain_dwn", 14, 1, 0x0}},
    {F, {"ldo_lv_ind_en", 15, 1, 0x0}},
    {F, {"ldo_lv_ind_slw", 16, 1, 0x0}},
    {F, {"ldo_lv_ind_vth_ok", 17, 1, 0x0}},
    {F, {"ldo_lv_ind_vth_bo", 18, 3, 0x0}},
    {F, {"ldo_lv_ind_out_dyn", 21, 1, 0x0}},
    {F, {"ldo_lv_en_delay", 22, 1, 0x0}},
    {F, {"ldo_lv_start_pulse", 23, 1, 0x0}},
    {F, {"ldo_lv_start_trig", 24, 1, 0x0}},

    {R, {"ldo/ldo_hv_en_delay_max", 0xA00C}},
    {F, {"val", 0, 16, 0x12C}},

    {R, {"ldo/ldo_hv_start_pulse_max", 0xA010}},
    {F, {"val", 0, 16, 0x927C}},

    {R, {"ldo/ldo_lv_en_delay_max", 0xA014}},
    {F, {"val", 0, 16, 0x12C}},

    {R, {"ldo/ldo_lv_start_pulse_max", 0xA018}},
    {F, {"val", 0, 16, 0x30D4}},

    {R, {"ldo/pmu", 0xA01C}},
    {F, {"pmu_icgm_en", 0, 1, 0x0}},
    {F, {"pmu_v2i_en", 1, 1, 0x0}},
    {F, {"pmu_v2i_adj", 2, 4, 0x7}},
    {F, {"pmu_iref_dft_cnt", 6, 2, 0x0}},
    {F, {"test_pixel_cur_dac", 8, 2, 0x0}},
    {F, {"mipi_v2i_en", 10, 1, 0x0}},
    {F, {"mipi_adj", 11, 3, 0x4}},
    {F, {"pmu_auto_start", 14, 1, 0x1}},
    {F, {"pmu_v2i_cal_en", 15, 1, 0x0}},
    {F, {"pmu_v2i_adj_dac_dyn", 16, 4, 0x7}},
    {F, {"pmu_v2i_cal_done_dyn", 20, 1, 0x0}},

    {R, {"mipi_csi/csi_ctrl", 0xB000}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"empty", 1, 1, 0x0}},
    {F, {"busy", 2, 1, 0x0}},
    {F, {"frame_sync_en", 3, 1, 0x0}},
    {F, {"line_sync_en", 4, 1, 0x0}},
    {F, {"vchannel", 8, 2, 0x0}},
    {F, {"data_type", 10, 6, 0x30}},
    {F, {"pkt_size", 16, 14, 0x1000}},

    {R, {"mipi_csi/cpu_ctrl", 0xB004}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"empty", 1, 1, 0x0}},
    {F, {"busy", 2, 1, 0x0}},
    {F, {"data_type", 10, 6, 0x0}},
    {F, {"pkt_size", 16, 14, 0x0}},

    {R, {"mipi_csi/lane_ctrl", 0xB008}},
    {F, {"reserved", 0, 32, 0x0}},

    {R, {"mipi_csi/clk_ctrl", 0xB00C}},
    {F, {"txclkesc_divider", 0, 8, 0x6}},
    {F, {"txclkesc_en", 8, 1, 0x0}},
    {F, {"stbus_en", 9, 1, 0x0}},
    {F, {"cl_clkesc_en", 10, 1, 0x0}},
    {F, {"dl_clkesc_en", 11, 1, 0x0}},
    {F, {"txbyteclkhs_en", 12, 1, 0x0}},

    {R, {"mipi_csi/frame_ctrl", 0xB010}},
    {F, {"pkt_timeout_en", 0, 1, 0x0}},
    {F, {"pkt_fix_rate_en", 1, 1, 0x0}},
    {F, {"pkt_fix_size_en", 2, 1, 0x0}},
    {F, {"frame_fix_rate_en", 3, 1, 0x0}},
    {F, {"frame_fix_size_en", 4, 1, 0x0}},
    {F, {"fix_rate_empty_pkt", 5, 1, 0x0}},

    {R, {"mipi_csi/frame_cfg", 0xB014}},
    {F, {"period", 0, 16, 0x2000}},
    {F, {"size", 16, 16, 0x140}},

    {R, {"mipi_csi/frame_sync", 0xB018}},
    {F, {"val", 0, 16, 0x4}},

    {R, {"mipi_csi/mid", 0xB01C}},
    {F, {"mid", 0, 16, 0x41D}},

    {R, {"mipi_csi/bl_line", 0xB020}},
    {F, {"val", 0, 16, 0x0}},
    {F, {"ck_lane_hs", 30, 1, 0x0}},
    {F, {"enable", 31, 1, 0x0}},

    {R, {"mipi_csi/bl_frame", 0xB024}},
    {F, {"val", 0, 24, 0x0}},
    {F, {"ck_lane_hs", 30, 1, 0x0}},
    {F, {"enable", 31, 1, 0x0}},

    {R, {"mipi_csi/bl_frame_start", 0xB028}},
    {F, {"val", 0, 16, 0x0}},
    {F, {"enable", 31, 1, 0x0}},

    {R, {"mipi_csi/bl_frame_end", 0xB02C}},
    {F, {"val", 0, 16, 0x0}},
    {F, {"enable", 31, 1, 0x0}},

    {R, {"mipi_csi/power", 0xB030}},
    {F, {"cl_rstn", 0, 1, 0x0}},
    {F, {"dl_rstn", 1, 1, 0x0}},
    {F, {"cl_enable", 2, 1, 0x0}},
    {F, {"dl_enable", 3, 1, 0x0}},
    {F, {"cur_en", 4, 1, 0x0}},

    {R, {"mipi_csi/cl_ctrl", 0xB034}},
    {F, {"ulps", 0, 1, 0x0}},
    {F, {"ulpsexit", 1, 1, 0x0}},
    {F, {"ulpsactivenot", 2, 1, 0x0}},
    {F, {"stopstate", 3, 1, 0x0}},

    {R, {"mipi_csi/dl_ctrl", 0xB038}},
    {F, {"requestesc", 0, 1, 0x0}},
    {F, {"ulpsesc", 4, 1, 0x0}},
    {F, {"ulpsexit", 8, 1, 0x0}},
    {F, {"ulpsactivenot", 16, 1, 0x0}},
    {F, {"stopstate", 20, 1, 0x0}},

    {R, {"mipi_csi/ulps_ctrl", 0xB03C}},
    {F, {"wakeup", 0, 24, 0xD6D8}},
    {F, {"status", 24, 4, 0x0}},
    {F, {"enable", 31, 1, 0x0}},

    {R, {"mipi_csi/cpu_data", 0xB040}},
    {F, {"data", 0, 32, 0x0}},

    {R, {"mipi_csi/cpu_cmd", 0xB044}},
    {F, {"send", 0, 1, 0x0}},
    {F, {"busy", 31, 1, 0x0}},

    {R, {"mipi_csi/cpu_cfg", 0xB048}},
    {F, {"sram_part", 0, 3, 0x0}},

    {R, {"mipi_csi/irq_mask", 0xB060}},
    {F, {"cpu_data_send", 0, 1, 0x0}},
    {F, {"stats_avail", 1, 1, 0x0}},
    {F, {"frame_start", 2, 1, 0x0}},
    {F, {"frame_end", 3, 1, 0x0}},
    {F, {"cl_stop", 4, 1, 0x0}},
    {F, {"cl_ulpsactivenot", 5, 1, 0x0}},
    {F, {"dl_stop", 6, 1, 0x0}},
    {F, {"dl_ulpsactivenot", 7, 1, 0x0}},

    {R, {"mipi_csi/irq_pending", 0xB064}},
    {F, {"cpu_data_send", 0, 1, 0x0}},
    {F, {"stats_avail", 1, 1, 0x0}},
    {F, {"frame_start", 2, 1, 0x0}},
    {F, {"frame_end", 3, 1, 0x0}},
    {F, {"cl_stop", 4, 1, 0x0}},
    {F, {"cl_ulpsactivenot", 5, 1, 0x0}},
    {F, {"dl_stop", 6, 1, 0x0}},
    {F, {"dl_ulpsactivenot", 7, 1, 0x0}},

    {R, {"mipi_csi/irq_status", 0xB068}},
    {F, {"cpu_data_send", 0, 1, 0x0}},
    {F, {"stats_avail", 1, 1, 0x0}},
    {F, {"frame_start", 2, 1, 0x0}},
    {F, {"frame_end", 3, 1, 0x0}},
    {F, {"cl_stop", 4, 1, 0x0}},
    {F, {"cl_ulpsactivenot", 5, 1, 0x0}},
    {F, {"dl_stop", 6, 1, 0x0}},
    {F, {"dl_ulpsactivenot", 7, 1, 0x0}},

    {R, {"mipi_csi/gpio_ctrl", 0xB06C}},
    {F, {"frame_start", 0, 1, 0x0}},
    {F, {"frame_end", 1, 1, 0x0}},
    {F, {"csi_data", 2, 1, 0x0}},
    {F, {"pad_data", 3, 1, 0x0}},
    {F, {"cpu_data", 4, 1, 0x0}},
    {F, {"cl_stop", 5, 1, 0x0}},
    {F, {"cl_ulpsactivenot", 6, 1, 0x0}},
    {F, {"dl_stop", 7, 1, 0x0}},
    {F, {"dl_ulpsactivenot", 8, 1, 0x0}},
    {F, {"sof", 9, 1, 0x0}},
    {F, {"atomic", 10, 1, 0x0}},

    {R, {"mipi_csi/stat_ctrl", 0xB080}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"trigger", 1, 1, 0x0}},
    {F, {"clear", 2, 1, 0x0}},

    {R, {"mipi_csi/stat_frame_cnt", 0xB084}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"mipi_csi/stat_byte_cnt", 0xB088}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"mipi_csi/stat_pad_cnt", 0xB08C}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"mipi_csi/stat_pkt_cnt", 0xB090}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"mipi_csi/stat_inc_pkt_cnt", 0xB094}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"mipi_csi/stat_frame_period", 0xB098}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"mipi_csi/spg_cfg", 0xB0C0}},
    {F, {"self_test_ctrl", 0, 3, 0x0}},
    {F, {"self_test_rate", 4, 10, 0x0}},
    {F, {"self_test_repeat", 16, 13, 0x0}},

    {R, {"mipi_csi/spg_data", 0xB0C4}},
    {F, {"self_test_data", 0, 32, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_w_tx_clk_zero_hs_prp_reg", 0xB400}},
    {F, {"reg_3_0", 0, 4, 0x0}},
    {F, {"reg_7_4", 4, 4, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_w_tx_clk_bctl_reg", 0xB404}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_1_1", 1, 1, 0x0}},
    {F, {"reg_2_2", 2, 1, 0x0}},
    {F, {"band_ctrl", 3, 5, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_w_tx_clk_ana_ctrl1_reg", 0xB408}},
    {F, {"o_TM_PD_disable", 0, 1, 0x0}},
    {F, {"w_bypass_PDN", 1, 1, 0x0}},
    {F, {"w_sel_ana_PDN", 2, 1, 0x0}},
    {F, {"w_bypass_ULPS_PDN", 3, 1, 0x0}},
    {F, {"w_sel_ULPS_PDN", 4, 1, 0x0}},
    {F, {"inv_LPTX_DN", 5, 1, 0x0}},
    {F, {"inv_LPTX_DP", 6, 1, 0x0}},
    {F, {"bypass_LPTX_DP_DN", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_w_tx_clk_ana_ctrl2_reg", 0xB40C}},
    {F, {"w_bypass_LPTX_PD_RE", 0, 1, 0x0}},
    {F, {"w_sel_ana_LPTX_PD_REF", 1, 1, 0x0}},
    {F, {"o_ana_LOOPBACK_PDNB", 2, 1, 0x0}},
    {F, {"o_ana_LOOPBACK_EN", 3, 1, 0x0}},
    {F, {"tx_clk_ana_ctrl2_reg4", 4, 1, 0x0}},
    {F, {"tx_clk_ana_ctrl2_reg5", 5, 1, 0x0}},
    {F, {"w_bypass_POR", 6, 1, 0x0}},
    {F, {"w_sel_ana_POR", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_o_ana_tx_clk_tbit1", 0xB410}},
    {F, {"LPTX_BIAS_TM", 0, 2, 0x0}},
    {F, {"Reserved0", 2, 2, 0x0}},
    {F, {"LPTX_LDO_PD", 4, 1, 0x0}},
    {F, {"Reserved1", 5, 1, 0x0}},
    {F, {"LPTX_bias_PD", 6, 1, 0x0}},
    {F, {"Reserved2", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_o_ana_tx_clk_tbit2", 0xB414}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_7_1", 1, 7, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_o_ana_tx_clk_tbit3", 0xB418}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_1_1", 1, 1, 0x0}},
    {F, {"reg_2_2", 2, 1, 0x0}},
    {F, {"reg_3_3", 3, 1, 0x0}},
    {F, {"reg_7_4", 4, 4, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_o_ana_tx_clk_tbit4", 0xB41C}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_7_1", 1, 7, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_o_ana_tx_clk_tbit5", 0xB420}},
    {F, {"reg_6_0", 0, 7, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_o_ana_tx_clk_tbit6", 0xB424}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_1_1", 1, 1, 0x0}},
    {F, {"reg_2_2", 2, 1, 0x0}},
    {F, {"reg_3_3", 3, 1, 0x0}},
    {F, {"reg_4_4", 4, 1, 0x0}},
    {F, {"reg_5_5", 5, 1, 0x0}},
    {F, {"reg_6_6", 6, 1, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_o_ana_tx_clk_tbit7", 0xB428}},
    {F, {"reg_1_0", 0, 2, 0x0}},
    {F, {"reg_3_2", 2, 2, 0x0}},
    {F, {"reg_4_4", 4, 1, 0x0}},
    {F, {"reg_5_5", 5, 1, 0x0}},
    {F, {"reg_6_6", 6, 1, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_o_ana_tx_clk_tbit8", 0xB42C}},
    {F, {"reg_6_0", 0, 7, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_o_ana_tx_clk_tbit9", 0xB430}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_o_ana_tx_slew_set", 0xB434}},
    {F, {"reg_6_0", 0, 7, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_w_bist_en", 0xB438}},
    {F, {"BIST_en", 0, 1, 0x0}},
    {F, {"reg_7_1", 1, 7, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_w_tx_clk_prp_reg", 0xB43C}},
    {F, {"reg_3_0", 0, 4, 0x0}},
    {F, {"reg_7_4", 4, 4, 0x0}},

    {R, {"mipi_dphy/DPHY_BLOCK_CLK_W_LDO_PWRUP_CNT", 0xB440}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_clk_w_data_int", 0xB444}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx_clk_status_reg", 0xB448}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_m_clk_module_id", 0xB44C}},
    {F, {"reg_7_0", 0, 8, 0x1}},

    {R, {"mipi_dphy/dphy_block_m_clk_version_id", 0xB450}},
    {F, {"reg_7_0", 0, 8, 0x1}},

    {R, {"mipi_dphy/dphy_block_clk_ulps", 0xB454}},
    {F, {"M_Clk_SwapDpDn_CTX", 0, 1, 0x0}},
    {F, {"TxULPSExitClk_CTX", 1, 1, 0x0}},
    {F, {"TxULPSClk_CTX", 2, 1, 0x0}},
    {F, {"TxEnableClk_CTX", 3, 1, 0x0}},
    {F, {"reg_7_4", 4, 4, 0x0}},

    {R, {"mipi_dphy/DPHY_BLOCK_CLK_ESCAPE", 0xB458}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_1_1", 1, 1, 0x0}},
    {F, {"reg_2_2", 2, 1, 0x0}},
    {F, {"reg_3_3", 3, 1, 0x0}},
    {F, {"reg_4_4", 4, 1, 0x0}},
    {F, {"TxULPSActiveNotClk_CTX", 5, 1, 0x0}},
    {F, {"TxStopStateClk_CTX", 6, 1, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_data_zero_hs_prp_reg", 0xB45C}},
    {F, {"reg_3_0", 0, 4, 0x0}},
    {F, {"reg_7_4", 4, 4, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_t1_data_clk_prp_reg", 0xB460}},
    {F, {"reg_3_0", 0, 4, 0x0}},
    {F, {"reg_7_4", 4, 4, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_tx_data_hstx_bits_reg", 0xB464}},
    {F, {"ForceRxMode0", 0, 1, 0x0}},
    {F, {"w_bypass_PDN", 1, 1, 0x0}},
    {F, {"w_sel_ana_PDN", 2, 1, 0x0}},
    {F, {"w_bypass_ULPS_PDN", 3, 1, 0x0}},
    {F, {"w_sel_ULPS_PDN", 4, 1, 0x0}},
    {F, {"test_ana_HS_LP_DnB", 5, 1, 0x0}},
    {F, {"test_ana_HS_LP_DpB", 6, 1, 0x0}},
    {F, {"test_sel_ana_HS_LP_D", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_tx_data_ana_ctrl1_reg", 0xB468}},
    {F, {"ForceRXMode", 0, 1, 0x0}},
    {F, {"w_bypass_PDN", 1, 1, 0x0}},
    {F, {"reg_2_2", 2, 1, 0x0}},
    {F, {"reg_3_3", 3, 1, 0x0}},
    {F, {"reg_4_4", 4, 1, 0x0}},
    {F, {"reg_5_5", 5, 1, 0x0}},
    {F, {"reg_6_6", 6, 1, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/DPHY_BLOCK_TX1_W_TX_DATA_ANA_CTRL2_REG", 0xB46C}},
    {F, {"w_bypass_LPTX_PD_REF", 0, 1, 0x0}},
    {F, {"w_sel_ana_LPTX_PD_REF", 1, 1, 0x0}},
    {F, {"o_ana_LOOPBACK_PDNB", 2, 1, 0x0}},
    {F, {"o_ana_LOOPBACK_EN", 3, 1, 0x0}},
    {F, {"o_TM_PD_disable", 4, 1, 0x0}},
    {F, {"test_sel_ana_IN", 5, 1, 0x0}},
    {F, {"w_bypass_POR", 6, 1, 0x0}},
    {F, {"w_sel_ana_POR", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_tx_data_band_ctl_reg", 0xB470}},
    {F, {"band_ctrl", 0, 5, 0x0}},
    {F, {"reg_5_5", 5, 1, 0x0}},
    {F, {"reg_6_6", 6, 1, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_o_ana_tx_data_tbit1", 0xB474}},
    {F, {"reg_1_0", 0, 2, 0x0}},
    {F, {"reg_3_2", 2, 2, 0x0}},
    {F, {"reg_4_4", 4, 1, 0x0}},
    {F, {"reg_5_5", 5, 1, 0x0}},
    {F, {"reg_6_6", 6, 1, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/DPHY_BLOCK_TX1_O_ANA_TX_DATA_TBIT2", 0xB478}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_7_1", 1, 7, 0x0}},

    {R, {"mipi_dphy/DPHY_BLOCK_TX1_O_ANA_TX_DATA_TBIT3", 0xB47C}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_1_1", 1, 1, 0x0}},
    {F, {"reg_2_2", 2, 1, 0x0}},
    {F, {"reg_3_3", 3, 1, 0x0}},
    {F, {"reg_7_4", 4, 4, 0x0}},

    {R, {"mipi_dphy/DPHY_BLOCK_TX1_O_ANA_TX_DATA_TBIT4", 0xB480}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_1_1", 1, 1, 0x0}},
    {F, {"reg_2_2", 2, 1, 0x0}},
    {F, {"reg_3_3", 3, 1, 0x0}},
    {F, {"reg_4_4", 4, 1, 0x0}},
    {F, {"reg_7_5", 5, 3, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_o_ana_tx_data_tbit5", 0xB484}},
    {F, {"reg_6_0", 0, 7, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_o_ana_tx_data_tbit6", 0xB488}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_o_ana_tx_data_tbit7", 0xB48C}},
    {F, {"reg_1_0", 0, 2, 0x0}},
    {F, {"reg_3_2", 2, 2, 0x0}},
    {F, {"reg_4_4", 4, 1, 0x0}},
    {F, {"reg_5_5", 5, 1, 0x0}},
    {F, {"reg_6_6", 6, 1, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_o_ana_tx_data_tbit8", 0xB490}},
    {F, {"reg_6_0", 0, 7, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_ldo_pwrup_cnt", 0xB494}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_data_lptx_data", 0xB498}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_tx_data_lp_ctrl", 0xB49C}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_1_1", 1, 1, 0x0}},
    {F, {"reg_2_2", 2, 1, 0x0}},
    {F, {"reg_3_3", 3, 1, 0x0}},
    {F, {"reg_4_4", 4, 1, 0x0}},
    {F, {"reg_5_5", 5, 1, 0x0}},
    {F, {"reg_6_6", 6, 1, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_o_ana_tx_data_slew_set", 0xB4A0}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_burnin", 0xB4A4}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_7_1", 1, 7, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_en", 0xB4A8}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_7_1", 1, 7, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_test_mode", 0xB4AC}},
    {F, {"reg_2_0", 0, 3, 0x0}},
    {F, {"reg_7_3", 3, 5, 0x0}},

    {R, {"mipi_dphy/DPHY_BLOCK_TX1_W_BIST_PRBS_MODE", 0xB4B0}},
    {F, {"reg_1_0", 0, 2, 0x0}},
    {F, {"reg_7_2", 2, 6, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_freeze", 0xB4B4}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_7_1", 1, 7, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_inject_err", 0xB4B8}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_7_1", 1, 7, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_idle_time", 0xB4BC}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_low_pulse_time", 0xB4C0}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_total_pulse_time", 0xB4C4}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_test_pat1", 0xB4C8}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_test_pat2", 0xB4CC}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_pkt_num", 0xB4D0}},
    {F, {"reg_7_0", 0, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_frame_idle_time", 0xB4D4}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_run_len", 0xB4D8}},
    {F, {"reg_11_0", 0, 12, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_bist_err_inject_point", 0xB4DC}},
    {F, {"reg_11_0", 0, 12, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_o_tm_pd_disable", 0xB4E0}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_w_reg29", 0xB4E4}},
    {F, {"reg_6_0", 0, 7, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_data_edge_detect", 0xB4E8}},
    {F, {"reg_7_0", 0, 8, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_lpdet_ulpdet", 0xB4EC}},
    {F, {"Data_SwapDpDn", 0, 1, 0x0}},
    {F, {"ForceTxStopMode", 1, 1, 0x0}},
    {F, {"Enable", 2, 1, 0x0}},
    {F, {"LPDTEsc", 3, 1, 0x0}},
    {F, {"RequestEsc", 4, 1, 0x0}},
    {F, {"ULPSExitEsc", 5, 1, 0x0}},
    {F, {"ULPSEsc", 6, 1, 0x0}},
    {F, {"reserved", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_tx1_escape", 0xB4F0}},
    {F, {"ErrSyncEsc", 0, 1, 0x0}},
    {F, {"ErrEsc", 1, 1, 0x0}},
    {F, {"ErrControl", 2, 1, 0x0}},
    {F, {"ErrContentionLP1", 3, 1, 0x0}},
    {F, {"ErrContentionLP0", 4, 1, 0x0}},
    {F, {"ULPSActiveNot", 5, 1, 0x0}},
    {F, {"StopState", 6, 1, 0x0}},
    {F, {"Direction", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_bist_results_reg", 0xB98C}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_1_1", 1, 1, 0x0}},
    {F, {"reg_2_2", 2, 1, 0x0}},
    {F, {"reg_3_3", 3, 1, 0x0}},
    {F, {"reg_4_4", 4, 1, 0x0}},
    {F, {"reg_5_5", 5, 1, 0x0}},
    {F, {"reg_6_6", 6, 1, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_status_clk_lane_reg", 0xB990}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_1_1", 1, 1, 0x0}},
    {F, {"reg_2_2", 2, 1, 0x0}},
    {F, {"reg_3_3", 3, 1, 0x0}},
    {F, {"reg_4_4", 4, 1, 0x0}},
    {F, {"reg_5_5", 5, 1, 0x0}},
    {F, {"reg_6_6", 6, 1, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"mipi_dphy/dphy_block_status_tx1_lane_reg", 0xB994}},
    {F, {"reg_0_0", 0, 1, 0x0}},
    {F, {"reg_1_1", 1, 1, 0x0}},
    {F, {"reg_2_2", 2, 1, 0x0}},
    {F, {"reg_3_3", 3, 1, 0x0}},
    {F, {"reg_4_4", 4, 1, 0x0}},
    {F, {"reg_5_5", 5, 1, 0x0}},
    {F, {"reg_6_6", 6, 1, 0x0}},
    {F, {"reg_7_7", 7, 1, 0x0}},

    {R, {"afk/pipeline_control", 0xC000}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"drop_nbackpressure", 1, 1, 0x0}},
    {F, {"bypass", 2, 1, 0x0}},

    {R, {"afk/pipeline_status", 0xC004}},
    {F, {"empty", 0, 1, 0x1}},
    {F, {"busy", 1, 1, 0x0}},
    {F, {"deep_low_power_ready", 2, 1, 0x0}},

    {R, {"afk/afk_param", 0xC008}},
    {F, {"counter_low", 0, 3, 0x4}},
    {F, {"counter_high", 3, 3, 0x6}},
    {F, {"invert", 6, 1, 0x0}},
    {F, {"drop_disable", 7, 1, 0x0}},

    {R, {"afk/filter_period", 0xC00C}},
    {F, {"min_cutoff_period", 0, 8, 0xF}},
    {F, {"max_cutoff_period", 8, 8, 0x9C}},
    {F, {"inverted_duty_cycle", 16, 4, 0x8}},

    {R, {"afk/invalidation", 0xC0C0}},
    {F, {"dt_fifo_wait_time", 0, 12, 0xF3C}},
    {F, {"dt_fifo_timeout", 12, 12, 0x5A}},
    {F, {"in_parallel", 24, 3, 0x5}},
    {F, {"flag_inv_busy", 27, 1, 0x0}},

    {R, {"afk/initialization", 0xC0C4}},
    {F, {"req_init", 0, 1, 0x0}},
    {F, {"flag_init_busy", 1, 1, 0x0}},
    {F, {"flag_init_done", 2, 1, 0x0}},

    {R, {"afk/icn_sram_ctrl", 0xC0C8}},
    {F, {"req_trigger", 0, 1, 0x0}},
    {F, {"req_type", 1, 1, 0x0}},
    {F, {"data_sel", 2, 3, 0x0}},

    {R, {"afk/icn_sram_address", 0xC0CC}},
    {F, {"x_addr", 0, 11, 0x0}},
    {F, {"y_addr", 16, 11, 0x0}},

    {R, {"afk/icn_sram_data", 0xC0D0}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"afk/shadow_ctrl", 0xC0D4}},
    {F, {"timer_en", 0, 1, 0x0}},
    {F, {"irq_sw_override", 1, 1, 0x0}},
    {F, {"reset_on_copy", 2, 1, 0x1}},

    {R, {"afk/shadow_timer_threshold", 0xC0D8}},
    {F, {"timer_threshold", 0, 32, 0x3E8}},

    {R, {"afk/shadow_status", 0xC0DC}},
    {F, {"shadow_valid", 0, 1, 0x0}},
    {F, {"shadow_overrun", 1, 1, 0x0}},

    {R, {"afk/total_evt_cnt", 0xC0E0}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"afk/flicker_evt_cnt", 0xC0E4}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"afk/output_vector_cnt", 0xC0E8}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"afk/perf_ctrl", 0xC0EC}},
    {F, {"lite_version_enable", 0, 1, 0x0}},

    {R, {"afk/chicken0_bits", 0xC1C0}},
    {F, {"powerdown_enable", 0, 1, 0x1}},
    {F, {"inv_powerdown_enable", 1, 1, 0x1}},
    {F, {"sram_clk_gating_enable", 3, 1, 0x1}},
    {F, {"enable_inv_abs_timebase", 4, 1, 0x1}},

    {R, {"afk/chicken1_bits", 0xC1C4}},
    {F, {"enable_inv_alr_last_ts", 0, 1, 0x1}},
    {F, {"enable_inv_alr_last_burst", 1, 1, 0x1}},
    {F, {"enable_inv_abs_threshold", 16, 12, 0x80}},

    {R, {"stc/pipeline_control", 0xD000}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"drop_nbackpressure", 1, 1, 0x0}},
    {F, {"bypass", 2, 1, 0x0}},

    {R, {"stc/pipeline_status", 0xD004}},
    {F, {"empty", 0, 1, 0x1}},
    {F, {"busy", 1, 1, 0x0}},
    {F, {"deep_low_power_ready", 2, 1, 0x0}},

    {R, {"stc/stc_param", 0xD008}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"threshold", 1, 19, 0x2710}},
    {F, {"disable_stc_cut_trail", 24, 1, 0x1}},

    {R, {"stc/trail_param", 0xD00C}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"threshold", 1, 19, 0x186A0}},

    {R, {"stc/timestamping", 0xD010}},
    {F, {"prescaler", 0, 5, 0xD}},
    {F, {"multiplier", 5, 4, 0x1}},
    {F, {"enable_rightshift_round", 9, 1, 0x1}},
    {F, {"enable_last_ts_update_at_every_event", 16, 1, 0x1}},

    {R, {"stc/invalidation", 0xD0C0}},
    {F, {"dt_fifo_wait_time", 0, 12, 0x4}},
    {F, {"dt_fifo_timeout", 12, 12, 0x118}},
    {F, {"in_parallel", 24, 3, 0x5}},
    {F, {"flag_inv_busy", 27, 1, 0x0}},

    {R, {"stc/initialization", 0xD0C4}},
    {F, {"req_init", 0, 1, 0x0}},
    {F, {"flag_init_busy", 1, 1, 0x0}},
    {F, {"flag_init_done", 2, 1, 0x0}},

    {R, {"stc/icn_sram_ctrl", 0xD0C8}},
    {F, {"req_trigger", 0, 1, 0x0}},
    {F, {"req_type", 1, 1, 0x0}},
    {F, {"data_sel", 2, 1, 0x0}},

    {R, {"stc/icn_sram_address", 0xD0CC}},
    {F, {"x_addr", 0, 11, 0x0}},
    {F, {"y_addr", 16, 11, 0x0}},

    {R, {"stc/icn_sram_data", 0xD0D0}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"stc/shadow_ctrl", 0xD0D4}},
    {F, {"timer_en", 0, 1, 0x0}},
    {F, {"irq_sw_override", 1, 1, 0x0}},
    {F, {"reset_on_copy", 2, 1, 0x1}},

    {R, {"stc/shadow_timer_threshold", 0xD0D8}},
    {F, {"timer_threshold", 0, 32, 0x3E8}},

    {R, {"stc/shadow_status", 0xD0DC}},
    {F, {"shadow_valid", 0, 1, 0x0}},
    {F, {"shadow_overrun", 1, 1, 0x0}},

    {R, {"stc/total_evt_cnt", 0xD0E0}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"stc/stc_evt_cnt", 0xD0E4}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"stc/trail_evt_cnt", 0xD0E8}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"stc/output_vector_cnt", 0xD0EC}},
    {F, {"val", 0, 32, 0x0}},

    {R, {"stc/chicken0_bits", 0xD1C0}},
    {F, {"powerdown_enable", 0, 1, 0x1}},
    {F, {"inv_powerdown_enable", 1, 1, 0x1}},
    {F, {"sram_clk_gating_enable", 2, 1, 0x1}},
    {F, {"enable_inv_abs_timebase", 3, 1, 0x1}},

    {R, {"stc/chicken1_bits", 0xD1C4}},
    {F, {"enable_inv_alr_last_ts", 0, 1, 0x1}},
    {F, {"unused0", 1, 15, 0x0}},
    {F, {"enable_inv_abs_threshold", 16, 12, 0x10}},

    {R, {"nfl/pipeline_control", 0xE000}},
    {F, {"enable", 0, 1, 0x0}},
    {F, {"drop_nbackpressure", 1, 1, 0x0}},
    {F, {"bypass", 2, 1, 0x0}},

    {R, {"nfl/pipeline_status", 0xE004}},
    {F, {"empty", 0, 1, 0x1}},
    {F, {"busy", 1, 1, 0x0}},
    {F, {"deep_low_power_ready", 2, 1, 0x0}},

    {R, {"nfl/reference_period", 0xE008}},
    {F, {"val", 0, 11, 0x400}},

    {R, {"nfl/min_voxel_threshold_on", 0xE00C}},
    {F, {"val", 0, 21, 0x20}},

    {R, {"nfl/min_voxel_threshold_off", 0xE010}},
    {F, {"val", 0, 21, 0x20}},

    {R, {"nfl/max_voxel_threshold_off", 0xE014}},
    {F, {"val", 0, 21, 0x1FFFFF}},

    {R, {"nfl/max_voxel_threshold_on", 0xE018}},
    {F, {"val", 0, 21, 0x1FFFFF}},

    {R, {"nfl/insert_drop_monitoring", 0xE01C}},
    {F, {"en", 0, 1, 0x0}},

    {R, {"nfl/voxels_in_window", 0xE020}},
    {F, {"val", 0, 21, 0x0}},
    {F, {"first_td_seen", 25, 1, 0x1}},
    {F, {"min_voxel_drop_mode", 26, 1, 0x1}},
    {F, {"max_voxel_drop_mode", 27, 1, 0x0}},

    {R, {"nfl/chicken_bits", 0xE024}},
    {F, {"timestamp_from_th_td_only", 0, 1, 0x0}},
    {F, {"drop_monitor_for_each_dropped_td", 1, 1, 0x0}},

    {R, {"mbx/cpu_start_en", 0xF000}},
    {F, {"cpu_start_en", 0, 1, 0x0}},
    {F, {"unused", 1, 31, 0x0}},

    {R, {"mbx/cpu_soft_reset", 0xF004}},
    {F, {"cpu_soft_reset", 0, 1, 0x0}},
    {F, {"unused", 1, 31, 0x0}},

    {R, {"mbx/cmd_ptr", 0xF008}},
    {F, {"cmd_ptr", 0, 32, 0x0}},

    {R, {"mbx/status_ptr", 0xF00C}},
    {F, {"status_ptr", 0, 32, 0x0}},

    {R, {"mbx/misc", 0xF010}},
    {F, {"misc", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem0", 0xF900}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem1", 0xF904}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem2", 0xF908}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem3", 0xF90C}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem4", 0xF910}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem5", 0xF914}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem6", 0xF918}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem7", 0xF91C}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem8", 0xF920}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem9", 0xF924}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem10", 0xF928}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem11", 0xF92C}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem12", 0xF930}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem13", 0xF934}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem14", 0xF938}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem15", 0xF93C}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem16", 0xF940}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem17", 0xF944}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem18", 0xF948}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem19", 0xF94C}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem20", 0xF950}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem21", 0xF954}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem22", 0xF958}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem23", 0xF95C}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem24", 0xF960}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem25", 0xF964}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem26", 0xF968}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem27", 0xF96C}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem28", 0xF970}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem29", 0xF974}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem30", 0xF978}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem31", 0xF97C}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem32", 0xF980}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem33", 0xF984}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem34", 0xF988}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem35", 0xF98C}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem36", 0xF990}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem37", 0xF994}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem38", 0xF998}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem39", 0xF99C}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem40", 0xF9A0}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem41", 0xF9A4}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem42", 0xF9A8}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem43", 0xF9AC}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem44", 0xF9B0}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem45", 0xF9B4}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem46", 0xF9B8}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem47", 0xF9BC}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem48", 0xF9C0}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem49", 0xF9C4}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem50", 0xF9C8}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem51", 0xF9CC}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem52", 0xF9D0}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem53", 0xF9D4}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem54", 0xF9D8}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem55", 0xF9DC}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem56", 0xF9E0}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem57", 0xF9E4}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem58", 0xF9E8}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem59", 0xF9EC}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem60", 0xF9F0}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem61", 0xF9F4}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem62", 0xF9F8}},
    {F, {"mem_data", 0, 32, 0x0}},

    {R, {"mem_bank/bank_mem63", 0xF9FC}},
    {F, {"mem_data", 0, 32, 0x0}},

    // clang-format on
};

static uint32_t GenX320ESRegisterMapSize = sizeof(GenX320ESRegisterMap) / sizeof(GenX320ESRegisterMap[0]);

#endif // METAVISION_HAL_GENX320ES_REGISTERMAP_H
