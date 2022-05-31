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

#ifndef METAVISION_HAL_CCAM_GEN31_DEVICE_CONTROL_H
#define METAVISION_HAL_CCAM_GEN31_DEVICE_CONTROL_H

#include "facilities/psee_device_control.h"
#include "devices/utils/evt_format.h"

namespace Metavision {

class RegisterMap;
class Gen31Fpga;
class Gen31Sensor;
class I_TriggerIn;
class I_TriggerOut;

class Gen31DeviceControl : public PseeDeviceControl {
public:
    Gen31DeviceControl(const std::shared_ptr<RegisterMap> &register_map, const std::shared_ptr<Gen31Fpga> &fpga,
                       const std::shared_ptr<Gen31Sensor> &sensor);

    virtual void reset() override;

    /// get the type of sensor
    /// 0x90100402h uniform TD feedback PPD VGA
    /// 0x90100403h uniform EM HVGA
    long long get_sensor_id() override;
    static long long get_sensor_id(RegisterMap &register_map);

    bool is_gen31EM();
    static bool is_gen31EM(long sensor_id);
    static bool is_gen31EM(RegisterMap &register_map);

    void low_consumption_startup_post_biases_initialize();
    virtual bool set_mode_standalone_impl() override;
    virtual bool set_mode_slave_impl() override;
    virtual bool set_mode_master_impl() override;

protected:
    void set_mode_run_slave();

    void start_camera_common(bool is_gen31EM, bool allow_dual_readout = true);
    void stop_camera_common();
    void destroy_camera();
    virtual void reset_ts_internal();
    void fpga_init();
    void sensor_init();
    void enable_double_bandwith(bool en);
    void sensor_enable_clk_out(bool en);
    void soft_reset();

    std::shared_ptr<RegisterMap> register_map_;
    std::shared_ptr<Gen31Fpga> fpga_;
    std::shared_ptr<Gen31Sensor> sensor_;

private:
    virtual void start_impl() override;
    virtual bool set_evt_format_impl(EvtFormat fmt) override;

    void enable_LDO0_VDDA(bool enable);
    void enable_LDO1_VDDC(bool enable);
    void enable_LDO2_VDDD(bool enable);
    void fpga_enable_sensor_clk(bool enable);
    void terminate_camera();

    void enable_SW_CTRL_BGEN_RSTN(bool enable);
    void enable_fx3_interface(bool state);
    void enable_spare_control(bool state);

    void enable_roi_EM(bool state);
    void enable_roi_TD(bool state);

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

#endif // METAVISION_HAL_CCAM_GEN31_DEVICE_CONTROL_H
