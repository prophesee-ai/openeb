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

#ifndef METAVISION_HAL_PSEE_STC_REGISTERS_API_V0_H
#define METAVISION_HAL_PSEE_STC_REGISTERS_API_V0_H

#include <cstdint>
#include <string>

#include "metavision/hal/facilities/i_facility.h"
#include "metavision/hal/facilities/i_noise_filter_module.h"

namespace Metavision {

/**
 * Interface for the Registers of the STC API v0  commands
 */
class PseeSTCRegistersAPIv0 : public I_NoiseFilterModule {
public:
    enum class STCModes { BYPASS, ON, OFF_DROP, OFF_NDROP, OUT };

    virtual void enable(Type type, uint32_t threshold) override final;
    virtual void disable() override final;

    virtual void set_pipeline_control(bool enable, bool drop_nbackpressure, bool bypass)    = 0;
    virtual void get_pipeline_control(bool &enable, bool &drop_nbackpressure, bool &bypass) = 0;

    virtual void set_stc_param(bool enable, uint32_t threshold)   = 0;
    virtual void get_stc_param(bool &enable, uint32_t &threshold) = 0;

    virtual void set_trail_param(bool enable, uint32_t threshold)   = 0;
    virtual void get_trail_param(bool &enable, uint32_t &threshold) = 0;

    virtual void set_timestamping(uint32_t prescaler, uint32_t multiplier, bool enable_rightshift_round)    = 0;
    virtual void get_timestamping(uint32_t &prescaler, uint32_t &multiplier, bool &enable_rightshift_round) = 0;

    virtual void set_invalidation(uint32_t dt_fifo_wait_time, uint32_t dt_fifo_timeout, uint32_t in_parallel,
                                  bool flag_inv_busy)  = 0;
    virtual void get_invalidation(uint32_t &dt_fifo_wait_time, uint32_t &dt_fifo_timeout, uint32_t &in_parallel,
                                  bool &flag_inv_busy) = 0;

    virtual void set_initialization(bool req_init, bool flag_init_busy, bool flag_init_done)    = 0;
    virtual void get_initialization(bool &req_init, bool &flag_init_busy, bool &flag_init_done) = 0;

    virtual void set_shadow_ctrl(bool timer_en, bool irq_sw_override, bool reset_on_copy)    = 0;
    virtual void get_shadow_ctrl(bool &timer_en, bool &irq_sw_override, bool &reset_on_copy) = 0;

    virtual void set_shadow_timer_threshold(uint32_t timer_threshold)  = 0;
    virtual void get_shadow_timer_threshold(uint32_t &timer_threshold) = 0;

    virtual void set_shadow_status(bool shadow_valid, bool shadow_overrun)   = 0;
    virtual void get_shadow_status(bool &shadow_valid, bool &shadow_overrun) = 0;

    virtual void get_total_evt_cnt(uint32_t &val) = 0;

    virtual void get_stc_evt_cnt(uint32_t &val) = 0;

    virtual void get_trail_evt_cnt(uint32_t &val) = 0;

    virtual void get_output_vector_cnt(uint32_t &val) = 0;

    virtual void set_chicken1_bits(bool enable_inv_alr_last_ts, uint32_t enable_inv_abs_threshold)   = 0;
    virtual void get_chicken1_bits(bool &enable_inv_alr_last_ts, uint32_t &enable_inv_abs_threshold) = 0;

    // higher level apis:
    virtual bool is_present()       = 0;
    virtual STCModes get_mode()     = 0;
    virtual bool set_mode(STCModes) = 0;
    virtual void initialize()       = 0;
    virtual void start()            = 0;
    virtual void stop()             = 0;

private:
    virtual bool wait_for_init() = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_STC_REGISTERS_API_V0_H
