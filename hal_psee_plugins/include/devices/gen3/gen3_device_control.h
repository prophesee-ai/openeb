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

#ifndef METAVISION_HAL_GEN3_DEVICE_CONTROL_H
#define METAVISION_HAL_GEN3_DEVICE_CONTROL_H

#include "facilities/psee_device_control.h"

struct libusb_device_handle;

namespace Metavision {

class PseeLibUSBBoardCommand;

class Gen3DeviceControl : public PseeDeviceControl {
public:
    Gen3DeviceControl(const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd);

    virtual void reset() override;

    /// get the type of sensor
    /// 0x90100402h uniform TD feedback PPD VGA
    /// 0x90100403h uniform EM HVGA
    long long get_sensor_id() override;
    static long long get_sensor_id(PseeLibUSBBoardCommand &board_cmd);

    bool is_gen3EM();
    virtual bool set_mode_standalone_impl() override;
    virtual bool set_mode_master_impl() override;
    virtual bool set_mode_slave_impl() override;

    static bool is_gen3EM(PseeLibUSBBoardCommand &board_cmd);
    static bool is_gen3EM(long sensor_id);

    void low_consumption_startup_post_biases_initialize();

protected:
    uint32_t get_base_address();
    void set_mode_run_slave();

    void start_camera_common_0(bool is_gen3EM);
    void start_camera_common_1(bool is_gen3EM);
    void stop_camera_common();
    void destroy_camera();
    void initialize_common_0();
    void initialize_common_1();
    virtual void reset_ts_internal();
    virtual void set_mode_init();

    std::shared_ptr<PseeLibUSBBoardCommand> icmd_;

private:
    uint32_t base_address_;
    uint32_t base_sensor_address_;

    void enable_LDO0_VDDA(bool enable);
    void enable_LDO1_VDDC(bool enable);
    void enable_LDO2_VDDD(bool enable);

    void set_CTRL_SW_bit(bool enable);
    void set_CTRL_BGEN_bit(bool enable);
    void set_CTRL_BGEN_RSTN_bit(bool enable);
    void enable_SW_CTRL_BGEN_RSTN(bool enable);
    void enable_stereo_merge_module(bool state);
    void enable_fx3_interface(bool state);
    void enable_spare_control(bool state);

    void enable_roi_EM(bool state);
    void enable_roi_TD(bool state);
    void disable_roi_reset(bool state);

    void enable_readout(bool state);
    void enable_master_reset(bool state);
    void enable_hvga_bypass(bool state);
    void enable_oob_filter(bool state);
    void set_oob_filter_bounds(int width, int height);

    void enable_test_bus();
    void enable_ctrl_sync();

    // reset_ts calls SOFT RESET and should flush the fifo after that
    // as it is not done we transform the reset_ts so it does nothing
    // and use reset_ts_internal() instead in initialize (when the camdevice is created and camera is not streaming yet

    virtual void enable_interface(bool state) = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN3_DEVICE_CONTROL_H
