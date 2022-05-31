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

#include <iomanip>
#include <iostream>
#include <chrono>
#include <thread>

#include "metavision/hal/utils/hal_log.h"
#include "boards/utils/psee_libusb_board_command.h"
#include "facilities/psee_trigger_in.h"
#include "facilities/psee_trigger_out.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "boards/utils/config_registers_map.h"
#include "devices/gen3/gen3_device_control.h"
#include "devices/gen3/legacy_regmap_headers/legacy/stereo_pc_mapping.h"
#include "devices/gen3/legacy_regmap_headers/tep_register_control_register_map.h"
#include "devices/utils/device_system_id.h"
#include "geometries/vga_geometry.h"
#include "geometries/hvga_geometry.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

namespace Metavision {

uint32_t base_address        = CCAM3_SYS_REG_BASE_ADDR;
uint32_t base_sensor_address = CCAM3_SENSOR_IF_BASE_ADDR;

Gen3DeviceControl::Gen3DeviceControl(const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd) :
    PseeDeviceControl(EvtFormat::EVT2_0),
    icmd_(board_cmd),
    base_address_(CCAM3_SYS_REG_BASE_ADDR),
    base_sensor_address_(CCAM3_SENSOR_IF_BASE_ADDR) {}

void Gen3DeviceControl::destroy_camera() {
    enable_LDO0_VDDA(false);
    enable_SW_CTRL_BGEN_RSTN(false);
    enable_LDO2_VDDD(false);
    enable_LDO1_VDDC(false);
}

void Gen3DeviceControl::initialize_common_0() {
    enable_LDO0_VDDA(false);
    enable_LDO2_VDDD(false);
    enable_LDO1_VDDC(false);
}

void Gen3DeviceControl::initialize_common_1() {
    icmd_->try_to_flush();
    reset_ts_internal();
    set_mode_init();

    // ----------------------------------------------------------------------------
    // INIT

    enable_LDO1_VDDC(true); // VDDC
    enable_LDO2_VDDD(true); // VDDD
    enable_SW_CTRL_BGEN_RSTN(true);
}

void Gen3DeviceControl::start_camera_common_0(bool is_gen3EM) {
    //    reset_ts();
    set_mode_standalone_impl(); // default synchronization mode

    disable_roi_reset(true);
    if (is_gen3EM) {
        enable_hvga_bypass(false);
        set_oob_filter_bounds(HVGAGeometry::width_, HVGAGeometry::height_);
    } else {
        enable_hvga_bypass(true);
        set_oob_filter_bounds(VGAGeometry::width_, VGAGeometry::height_);
    }

    enable_oob_filter(true);
    enable_master_reset(false);
}

void Gen3DeviceControl::start_camera_common_1(bool is_gen3EM) {
    enable_stereo_merge_module(true);
    enable_ctrl_sync();
    enable_readout(true);

    if (is_gen3EM) {
        icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                                SISLEY_SENSOR_READOUT_CTRL_RO_STAT_X_BIT_IDX, true);

        icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                                SISLEY_SENSOR_READOUT_CTRL_RO_ACT_PUX_BIT_IDX, true);
        icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                                SISLEY_SENSOR_READOUT_CTRL_RO_ACT_PUX_BIT_IDX + 1, true);
        icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                                SISLEY_SENSOR_READOUT_CTRL_RO_ACT_PUX_BIT_IDX + 2, false);

        icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                                SISLEY_SENSOR_READOUT_CTRL_RO_ACT_PUY_BIT_IDX, true);

        icmd_->send_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                                 SISLEY_SENSOR_READOUT_CTRL_RO_INV_POL_EM_BIT_IDX, true); // invert EM polarity
    }

    // this needs to be done before roi for em as this function initialize the value for the ROI register
    enable_roi_TD(true);

    if (is_gen3EM) {
        enable_roi_EM(true);
    }

    enable_test_bus();
    enable_spare_control(true);
}

void Gen3DeviceControl::stop_camera_common() {
    enable_stereo_merge_module(false);
    set_mode_init();
    auto trigger_in = get_trigger_in(false);
    if (trigger_in) {
        for (uint32_t i = 0; i < 8; ++i) {
            trigger_in->disable(i);
        }
    }
}

void Gen3DeviceControl::disable_roi_reset(bool state) {
    // -------------- Disable ROI reset
    MV_HAL_LOG_DEBUG() << "-------------- Disable ROI reset";
    MV_HAL_LOG_DEBUG() << Metavision::Log::no_space << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_address_ + SYSTEM_CONTROL_ADDR << std::dec << "\t|\t"
                       << TEP_ATIS_CONTROL_ROI_TD_RSTN_BIT_IDX << std::dec << " " << (state ? 1 : 0);
    icmd_->send_register_bit(base_address_ + SYSTEM_CONTROL_ADDR, TEP_ATIS_CONTROL_ROI_TD_RSTN_BIT_IDX,
                             state); // Disable ROI reset
}

void Gen3DeviceControl::enable_hvga_bypass(bool state) {
    MV_HAL_LOG_DEBUG() << "-------------- Bypass HVGA remap";
    MV_HAL_LOG_DEBUG() << Metavision::Log::no_space << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_address_ + SYSTEM_CONTROL_ADDR << std::dec << "\t|\t"
                       << TEP_ATIS_CONTROL_SISLEY_HVGA_REMAP_BYPASS_BIT_IDX << std::dec << " " << (state ? 1 : 0);
    icmd_->send_register_bit(base_address_ + SYSTEM_CONTROL_ADDR, TEP_ATIS_CONTROL_SISLEY_HVGA_REMAP_BYPASS_BIT_IDX,
                             state); // Bypass HVGA remap
}

void Gen3DeviceControl::set_oob_filter_bounds(int width, int height) {
    MV_HAL_LOG_DEBUG() << "-------------- Set OOB filter bounds";
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << CCAM3_OUT_OF_FOV_FILTER_WIDTH_ADDR << "\t|\t" << width << std::dec;
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << CCAM3_OUT_OF_FOV_FILTER_HEIGHT_ADDR << "\t|\t" << height << std::dec;
    icmd_->write_register(base_address_ + CCAM3_OUT_OF_FOV_FILTER_WIDTH_ADDR, width);
    icmd_->write_register(base_address_ + CCAM3_OUT_OF_FOV_FILTER_HEIGHT_ADDR, height);
}

void Gen3DeviceControl::enable_oob_filter(bool state) {
    MV_HAL_LOG_DEBUG() << "-------------- OOB filter";
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_address_ + CCAM2_CONTROL_ADDR << std::dec << "\t|\t"
                       << TEP_CCAM2_CONTROL_ENABLE_OUT_OF_FOV_BIT_IDX << std::dec << " " << (state ? 1 : 0);
    icmd_->send_register_bit(base_address_ + CCAM2_CONTROL_ADDR, TEP_CCAM2_CONTROL_ENABLE_OUT_OF_FOV_BIT_IDX, state);
}

void Gen3DeviceControl::enable_master_reset(bool state) {
    // -------------- Gen3 deassert master reset
    MV_HAL_LOG_DEBUG() << "-------------- Gen3 deassert master reset";
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_address_ + SYSTEM_CONTROL_ADDR << std::dec << "\t|\t"
                       << TEP_ATIS_CONTROL_SENSOR_SOFT_RESET_BIT_IDX << std::dec << " " << (state ? 1 : 0);
    icmd_->send_register_bit(base_address_ + SYSTEM_CONTROL_ADDR, TEP_ATIS_CONTROL_SENSOR_SOFT_RESET_BIT_IDX,
                             state); // Gen3 deassert master reset
}

void Gen3DeviceControl::enable_ctrl_sync() {
    // CONTROL SYNC
    uint32_t clksync_ctrl = 0x0;
    clksync_ctrl |= (1 << SISLEY_SENSOR_CLKSYNC_CTRL_CLKOUT_EN_BIT_IDX);
    clksync_ctrl |= (1 << SISLEY_SENSOR_CLKSYNC_CTRL_CLK_DIFF_HI_CM_BIT_IDX);
    clksync_ctrl |= (1 << SISLEY_SENSOR_CLKSYNC_CTRL_CLK_DIFF_IOUT_X2_BIT_IDX);
    //    clksync_ctrl |= (1 << SISLEY_SENSOR_CLKSYNC_CTRL_CLK_POLARITY_BIT_IDX);
    clksync_ctrl |= (0x7 << SISLEY_SENSOR_CLKSYNC_CTRL_RO_TD_REQ_LAT_BIT_IDX); /// setting 3 bits

    MV_HAL_LOG_DEBUG() << "----------------------------------------------------------------------------";
    MV_HAL_LOG_DEBUG() << "CONTROL SYNC";
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_sensor_address_ + SISLEY_SENSOR_CLKSYNC_CTRL_ADDR << "\t|\t" << clksync_ctrl << std::dec;
    icmd_->write_register(base_sensor_address_ + SISLEY_SENSOR_CLKSYNC_CTRL_ADDR, clksync_ctrl);
}

void Gen3DeviceControl::enable_readout(bool state) {
    // ----------------------------------------------------------------------------
    // READOUT
    icmd_->init_register(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR, 0x0);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                            SISLEY_SENSOR_READOUT_CTRL_RO_DELAY_RSTN_BIT_IDX, state);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                            SISLEY_SENSOR_READOUT_CTRL_RO_INTERFACE_X_RSTN_BIT_IDX, state);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                            SISLEY_SENSOR_READOUT_CTRL_RO_LATCH_Y_RSTN_BIT_IDX, state);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                            SISLEY_SENSOR_READOUT_CTRL_RO_ACK_ARRAY_RSTN_BIT_IDX, state);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                            SISLEY_SENSOR_READOUT_CTRL_RO_CTRL_RSTN_BIT_IDX, state);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                            SISLEY_SENSOR_READOUT_CTRL_RO_LATCH_DUM_CONNECT_BIT_IDX, state);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                            SISLEY_SENSOR_READOUT_CTRL_RO_LATCH_DUM_RSTN_BIT_IDX, state);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                            SISLEY_SENSOR_READOUT_CTRL_RO_PIXEL_DUM_RSTN_BIT_IDX, state);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                            SISLEY_SENSOR_READOUT_CTRL_RO_INTERFACE_Y_RSTN_BIT_IDX, state);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR,
                            SISLEY_SENSOR_READOUT_CTRL_RO_LATCH_X_RSTN_BIT_IDX, state);

    uint32_t readout_ctrl = icmd_->read_register(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR);
    MV_HAL_LOG_DEBUG() << "----------------------------------------------------------------------------";
    MV_HAL_LOG_DEBUG() << "READOUT";
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR << "\t|\t" << readout_ctrl << std::dec;

    icmd_->send_register(base_sensor_address_ + SISLEY_SENSOR_READOUT_CTRL_ADDR);
}

void Gen3DeviceControl::enable_roi_TD(bool state) {
    // ----------------------------------------------------------------------------
    // ROI
    // -- Enable TD ROi
    icmd_->init_register(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR, SISLEY_SENSOR_ROI_CTRL);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR,
                            SISLEY_SENSOR_ROI_CTRL_ROI_TD_EN_BIT_IDX, state);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR,
                            SISLEY_SENSOR_ROI_CTRL_ROI_TD_SHADOW_TRIG_BIT_IDX, state);
    uint32_t enable_roi = icmd_->read_register(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR);
    MV_HAL_LOG_DEBUG() << "----------------------------------------------------------------------------";
    MV_HAL_LOG_DEBUG() << "ROI";
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR << "\t|\t" << enable_roi << std::dec;
    icmd_->send_register(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR);
}
void Gen3DeviceControl::enable_roi_EM(bool state) {
    // ----------------------------------------------------------------------------
    // ROI EM
    // -- Enable EM ROI
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR,
                            SISLEY_SENSOR_ROI_CTRL_ROI_EM_EN_BIT_IDX, state);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR,
                            SISLEY_SENSOR_ROI_CTRL_ROI_EM_SHADOW_TRIG_BIT_IDX, state);

    uint32_t enable_roi = icmd_->read_register(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR);
    MV_HAL_LOG_DEBUG() << "----------------------------------------------------------------------------";
    MV_HAL_LOG_DEBUG() << "ROI EM";
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR << "\t|\t" << enable_roi << std::dec;
    icmd_->send_register(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR);
}

void Gen3DeviceControl::enable_test_bus() {
    // ----------------------------------------------------------------------------
    // TEST BUS
    icmd_->init_register(base_sensor_address_ + SISLEY_SENSOR_TESTBUS_CTRL_ADDR, SISLEY_SENSOR_TESTBUS_CTRL);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_TESTBUS_CTRL_ADDR,
                            SISLEY_SENSOR_TESTBUS_CTRL_TP_SEL_TEST_PHOTODIODE_BIT_IDX, 1);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_TESTBUS_CTRL_ADDR,
                            SISLEY_SENSOR_TESTBUS_CTRL_TP_PIXEL_EN_BIT_IDX, 1);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_TESTBUS_CTRL_ADDR,
                            SISLEY_SENSOR_TESTBUS_CTRL_TBUS_SEL_TPA1_BIT_IDX, 1);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_TESTBUS_CTRL_ADDR,
                            SISLEY_SENSOR_TESTBUS_CTRL_TBUS_SEL_TPA2_BIT_IDX, 1);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_TESTBUS_CTRL_ADDR,
                            SISLEY_SENSOR_TESTBUS_CTRL_TBUS_SEL_TPA3_BIT_IDX, 1);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_TESTBUS_CTRL_ADDR,
                            SISLEY_SENSOR_TESTBUS_CTRL_TBUS_SEL_TPA4_BIT_IDX, 1);

    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_TESTBUS_CTRL_ADDR,
                            SISLEY_SENSOR_TESTBUS_CTRL_TP_PIXEL_DIODE_EN_BIT_IDX, 0);
    uint32_t testbus_ctrl = icmd_->read_register(base_sensor_address_ + SISLEY_SENSOR_TESTBUS_CTRL_ADDR);
    MV_HAL_LOG_DEBUG() << "----------------------------------------------------------------------------";
    MV_HAL_LOG_DEBUG() << "TEST BUS" << std::endl;
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_sensor_address_ + SISLEY_SENSOR_TESTBUS_CTRL_ADDR << "\t|\t" << testbus_ctrl << std::dec;
    icmd_->send_register(base_sensor_address_ + SISLEY_SENSOR_TESTBUS_CTRL_ADDR);
}

void Gen3DeviceControl::enable_spare_control(bool state) {
    // ----------------------------------------------------------------------------
    // SPARE CTRL
    icmd_->init_register(base_sensor_address_ + SISLEY_SENSOR_SPARE_CTRL_ADDR, SISLEY_SENSOR_SPARE_CTRL);
    icmd_->set_register_bit(base_address_ + SISLEY_SENSOR_SPARE_CTRL_ADDR,
                            SISLEY_SENSOR_SPARE_CTRL_RO_ADDR_Y_STAT_BIT_IDX, state);
    icmd_->set_register_bit(base_address_ + SISLEY_SENSOR_SPARE_CTRL_ADDR,
                            SISLEY_SENSOR_SPARE_CTRL_RO_ADDR_X_STAT_BIT_IDX, state);
    icmd_->set_register_bit(base_address_ + SISLEY_SENSOR_SPARE_CTRL_ADDR, SISLEY_SENSOR_SPARE_CTRL_B_TM_BIT_IDX,
                            state);
    uint32_t spare_ctrl = icmd_->read_register(base_address_ + SISLEY_SENSOR_SPARE_CTRL_ADDR);
    MV_HAL_LOG_DEBUG() << "----------------------------------------------------------------------------";
    MV_HAL_LOG_DEBUG() << "SPARE CTRL";
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_sensor_address_ + SISLEY_SENSOR_SPARE_CTRL_ADDR << "\t|\t" << spare_ctrl << std::dec;
    icmd_->send_register(base_sensor_address_ + SISLEY_SENSOR_SPARE_CTRL_ADDR);
}

void Gen3DeviceControl::low_consumption_startup_post_biases_initialize() {
    enable_LDO0_VDDA(true); // TO do after biases and before start
}

void Gen3DeviceControl::enable_stereo_merge_module(bool state) {
    // -------------- Disable Stereo Merge module
    MV_HAL_LOG_DEBUG() << "-------------- Disable Stereo Merge module";
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_address_ + CCAM2_CONTROL_ADDR << std::dec << "\t|\t"
                       << TEP_CCAM2_CONTROL_STEREO_MERGE_ENABLE_BIT_IDX << std::dec << " " << (state ? 1 : 0);
    icmd_->send_register_bit(base_address_ + CCAM2_CONTROL_ADDR, TEP_CCAM2_CONTROL_STEREO_MERGE_ENABLE_BIT_IDX,
                             state); // Enable Stereo Merge module
}

void Gen3DeviceControl::set_mode_init() {
    icmd_->set_register_bit(base_address_ + TEP_CCAM2_MODE_ADDR, TEP_CCAM2_MODE_MODE_BIT_IDX,
                            TEP_CCAM2_CONTROL_MODE_INIT & 1);
    icmd_->set_register_bit(base_address_ + TEP_CCAM2_MODE_ADDR, TEP_CCAM2_MODE_MODE_BIT_IDX + 1,
                            (TEP_CCAM2_CONTROL_MODE_INIT >> 1) & 1);
    icmd_->send_register(base_address_ + TEP_CCAM2_MODE_ADDR);
}

void Gen3DeviceControl::set_mode_run_slave() {
    icmd_->set_register_bit(base_address_ + TEP_CCAM2_MODE_ADDR, TEP_CCAM2_MODE_MODE_BIT_IDX,
                            TEP_CCAM2_CONTROL_MODE_SLAVE & 1);
    icmd_->set_register_bit(base_address_ + TEP_CCAM2_MODE_ADDR, TEP_CCAM2_MODE_MODE_BIT_IDX + 1,
                            (TEP_CCAM2_CONTROL_MODE_SLAVE >> 1) & 1);
    icmd_->send_register(base_address_ + TEP_CCAM2_MODE_ADDR);
}

void Gen3DeviceControl::enable_LDO0_VDDA(bool enable) {
    // -------------- ENABLE/DISABLE LDO 0
    MV_HAL_LOG_DEBUG() << (enable ? "-------------- ENABLE LDO 0" : "-------------- DISABLE LDO 0");
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_address_ + SYSTEM_CONTROL_ADDR << std::dec << "\t|\t" << TEP_ATIS_CONTROL_EN_VDDA_BIT_IDX
                       << " " << (enable ? 1 : 0);
    icmd_->send_register_bit(base_address_ + SYSTEM_CONTROL_ADDR, TEP_ATIS_CONTROL_EN_VDDA_BIT_IDX,
                             enable ? 1 : 0); // Enable LDO 0
}

void Gen3DeviceControl::enable_LDO1_VDDC(bool enable) {
    // -------------- ENABLE/DISABLE LDO 1
    MV_HAL_LOG_DEBUG() << (enable ? "-------------- ENABLE LDO 1" : "-------------- DISABLE LDO 1");
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_address_ + SYSTEM_CONTROL_ADDR << std::dec << "\t|\t" << TEP_ATIS_CONTROL_EN_VDDC_BIT_IDX
                       << std::dec << " " << (enable ? 1 : 0);
    icmd_->send_register_bit(base_address_ + SYSTEM_CONTROL_ADDR, TEP_ATIS_CONTROL_EN_VDDC_BIT_IDX,
                             enable ? 1 : 0); // Enable LDO 1
}
void Gen3DeviceControl::enable_LDO2_VDDD(bool enable) {
    // -------------- ENABLE/DISABLE LDO 2
    MV_HAL_LOG_DEBUG() << (enable ? "-------------- ENABLE LDO 2" : "-------------- DISABLE LDO 2");
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_address_ + SYSTEM_CONTROL_ADDR << std::dec << "\t|\t" << TEP_ATIS_CONTROL_EN_VDDD_BIT_IDX
                       << std::dec << " " << (enable ? 1 : 0);
    icmd_->send_register_bit(base_address_ + SYSTEM_CONTROL_ADDR, TEP_ATIS_CONTROL_EN_VDDD_BIT_IDX,
                             enable ? 1 : 0); // Enable LDO 2
}

void Gen3DeviceControl::set_CTRL_SW_bit(bool enable) {
    // -------------- Enable analog
    MV_HAL_LOG_DEBUG() << (enable ? "-------------- Enable analog" : "-------------- Disable analog");
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_sensor_address_ + SISLEY_SENSOR_GLOBAL_CTRL_ADDR << "\t|\t"
                       << SISLEY_SENSOR_GLOBAL_CTRL_SW_GLOBAL_EN_BIT_IDX << std::dec << " " << (enable ? 1 : 0);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_GLOBAL_CTRL_ADDR,
                            SISLEY_SENSOR_GLOBAL_CTRL_SW_GLOBAL_EN_BIT_IDX,
                            enable); // Enable analog
}

void Gen3DeviceControl::set_CTRL_BGEN_bit(bool enable) {
    MV_HAL_LOG_DEBUG() << (enable ? "-------------- Enable Ctrl Bgen" : "-------------- Disable Ctrl Bgen");
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_sensor_address_ + SISLEY_SENSOR_GLOBAL_CTRL_ADDR << "\t|\t"
                       << SISLEY_SENSOR_GLOBAL_CTRL_BGEN_EN_BIT_IDX << std::dec << " " << (enable ? 1 : 0);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_GLOBAL_CTRL_ADDR,
                            SISLEY_SENSOR_GLOBAL_CTRL_BGEN_EN_BIT_IDX, enable);

    std::this_thread::sleep_for(std::chrono::microseconds(1000));
}

void Gen3DeviceControl::set_CTRL_BGEN_RSTN_bit(bool enable) {
    MV_HAL_LOG_DEBUG() << (enable ? "-------------- Enable Bgen RSTN" : "-------------- Disable Bgen RSTN");
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_sensor_address_ + SISLEY_SENSOR_GLOBAL_CTRL_ADDR << "\t|\t"
                       << SISLEY_SENSOR_GLOBAL_CTRL_BGEN_RSTN_BIT_IDX << std::dec << " " << (enable ? 1 : 0);
    icmd_->set_register_bit(base_sensor_address_ + SISLEY_SENSOR_GLOBAL_CTRL_ADDR,
                            SISLEY_SENSOR_GLOBAL_CTRL_BGEN_RSTN_BIT_IDX, enable);
}

void Gen3DeviceControl::enable_SW_CTRL_BGEN_RSTN(bool enable) {
    set_CTRL_SW_bit(enable);
    set_CTRL_BGEN_bit(enable);
    set_CTRL_BGEN_RSTN_bit(enable);
    icmd_->send_register(base_sensor_address_ + SISLEY_SENSOR_GLOBAL_CTRL_ADDR);
}

void Gen3DeviceControl::reset() {}

void Gen3DeviceControl::reset_ts_internal() {
    // -------------- SOFT RESET
    int bit_value = 1;
    MV_HAL_LOG_DEBUG() << "-------------- SOFT RESET";
    MV_HAL_LOG_DEBUG() << std::hex << std::showbase << std::internal << std::setfill('0')
                       << base_address_ + TEP_TRIGGERS_ADDR << std::dec << "\t|\t" << TEP_TRIGGER_SOFT_RESET_BIT_IDX
                       << std::dec << " " << bit_value;

    icmd_->send_register_bit(base_address_ + TEP_TRIGGERS_ADDR, TEP_TRIGGER_SOFT_RESET_BIT_IDX,
                             bit_value); // Soft reset the system
}

long long Gen3DeviceControl::get_sensor_id() {
    icmd_->load_register(base_sensor_address_ + SISLEY_SENSOR_CHIP_ID_ADDR);
    return icmd_->read_register(base_sensor_address_ + SISLEY_SENSOR_CHIP_ID_ADDR);
}

long long Gen3DeviceControl::get_sensor_id(PseeLibUSBBoardCommand &board_cmd) {
    board_cmd.load_register(base_sensor_address + SISLEY_SENSOR_CHIP_ID_ADDR);
    return board_cmd.read_register(base_sensor_address + SISLEY_SENSOR_CHIP_ID_ADDR);
}

bool Gen3DeviceControl::is_gen3EM() {
    long sensor_id = get_sensor_id();
    return is_gen3EM(sensor_id);
}

bool Gen3DeviceControl::is_gen3EM(long sensor_id) {
    return device_is_gen3_EM(sensor_id);
}

bool Gen3DeviceControl::is_gen3EM(PseeLibUSBBoardCommand &board_cmd) {
    board_cmd.load_register(base_sensor_address + SISLEY_SENSOR_CHIP_ID_ADDR);
    auto sensor_id = board_cmd.read_register(base_sensor_address + SISLEY_SENSOR_CHIP_ID_ADDR);
    return device_is_gen3_EM(sensor_id);
}

uint32_t Gen3DeviceControl::get_base_address() {
    return base_address_;
}

bool Gen3DeviceControl::set_mode_master_impl() {
    if (get_trigger_out()->is_enabled()) {
        return false;
    }
    icmd_->send_register_bit(get_base_address() + CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_ADDR,
                             CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_MASTER_MODE_BIT_IDX, true);
    icmd_->send_register_bit(get_base_address() + CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_ADDR,
                             CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_USE_EXT_START_BIT_IDX, true);
    return true;
}

bool Gen3DeviceControl::set_mode_slave_impl() {
    if (get_trigger_in()->is_enabled(7)) {
        return false;
    }
    icmd_->send_register_bit(get_base_address() + CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_ADDR,
                             CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_MASTER_MODE_BIT_IDX, false);
    icmd_->send_register_bit(get_base_address() + CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_ADDR,
                             CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_USE_EXT_START_BIT_IDX, true);
    return true;
}

bool Gen3DeviceControl::set_mode_standalone_impl() {
    icmd_->send_register_bit(get_base_address() + CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_ADDR,
                             CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_MASTER_MODE_BIT_IDX, true);
    icmd_->send_register_bit(get_base_address() + CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_ADDR,
                             CCAM2_SYSTEM_CONTROL_ATIS_CONTROL_USE_EXT_START_BIT_IDX, false);
    return true;
}

} // namespace Metavision
