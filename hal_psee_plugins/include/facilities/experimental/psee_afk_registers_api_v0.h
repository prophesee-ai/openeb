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

#ifndef METAVISION_HAL_PSEE_AFK_REGISTERS_API_V0_H
#define METAVISION_HAL_PSEE_AFK_REGISTERS_API_V0_H

#include <cstdint>
#include <string>

#include "metavision/hal/facilities/i_antiflicker_module.h"

namespace Metavision {

/**
 * Interface for the Registers of the AFK API v0  commands
 */
class PseeAFKRegistersAPIv0 : public I_AntiFlickerModule {
public:
    enum class AFKModes { BYPASS, ON, OFF_DROP, OFF_NDROP, OUT };

    /// @brief Enables the anti-flicker filter
    virtual void enable() override final;

    /// @brief Disables the anti-flicker filter
    virtual void disable() override final;

    /// @brief Sets anti-flicker parameters.
    ///
    /// Defines the frequency band to be kept or removed :
    /// [frequency_center - bandwidth/2, frequency_center + bandwidth/2]
    /// This frequency range should be in the range [50 - 500] Hz
    ///
    /// @param frequency_center Center of the frequency band (in Hz)
    /// @param bandwidth Range of frequencies around the frequency_center (in Hz)
    /// @param stop If true, band-stop (by default); if false, band-pass
    ///
    /// @note band-stop removes all frequencies between min and max
    ///       band-pass removes all events outside of the band sequence defined
    /// @throw exception if frequency band is not in the range [50 - 500] Hz
    virtual void set_frequency(uint32_t frequency_center, uint32_t bandwidth, bool stop = true) override final;

    /// @brief Sets anti-flicker parameters.
    ///
    /// Defines the frequency band to be kept or removed in the range [50 - 500] Hz
    ///
    /// @param min_freq Lower frequency of the band (in Hz)
    /// @param max_freq Higher frequency of the band (in Hz)
    /// @param stop If true, band-stop; if false, band-pass
    ///
    /// @note band-stop removes all frequencies between min and max
    ///       band-pass removes all events outside of the band sequence defined
    /// @throw exception if frequencies are outside of the range [50 - 500] Hz
    virtual void set_frequency_band(uint32_t min_freq, uint32_t max_freq, bool stop = true) override final;

    /// @brief Changes the behavior of the anti-flicker (AFK) filter
    ///
    /// @param enable If true activates the AFK filter
    /// @param drop_nbackpressure If AFK is disabled and drop_nbackpressure is true, events are dropped to avoid
    /// backpressure
    /// @param bypass If AFK is enabled and bypass is true, the AFK filter is bypassed
    virtual void set_pipeline_control(bool enable, bool drop_nbackpressure, bool bypass)    = 0;
    virtual void get_pipeline_control(bool &enable, bool &drop_nbackpressure, bool &bypass) = 0;

    /// @brief Changes the AFK parameters
    ///
    /// @param counter_low Threshold under which the event is not flickering anymore
    /// @param counter_high Threshold upper which the event is not flickering anymore
    /// @param invert filter Events that are not flickering
    /// @param drop_disable Disable the drop of events
    virtual void set_afk_param(uint32_t counter_low, uint32_t counter_high, bool invert, bool drop_disable)     = 0;
    virtual void get_afk_param(uint32_t &counter_low, uint32_t &counter_high, bool &invert, bool &drop_disable) = 0;

    /// @brief Sets the filter period
    virtual void set_filter_period(uint32_t min_cutoff_period, uint32_t max_cutoff_period,
                                   uint32_t inverted_duty_cycle)  = 0;
    virtual void get_filter_period(uint32_t &min_cutoff_period, uint32_t &max_cutoff_period,
                                   uint32_t &inverted_duty_cycle) = 0;

    /// @brief Invalidation of the memory
    ///
    /// To avoid considering old values the memory should be invalidate with a period lower than a threshold
    /// The setting of theses parameter is a tradeoff between back pressure and consumption
    /// The dt_fifo_timeout should be at least 1 readout full lines time to guarantee the detection of deadtime.
    /// the readout full line can take 90 clock cycles.
    ///
    /// @param dt_fifo_wait_time Dead time; it should be greater than 4
    /// @param dt_fifo_timeout Timeout while waiting for a dead time to do the invalidation recommended to be > 90
    /// @param in_parallel Number of invalidation done in parallel (possible values: 1, 2, 5, 10)
    /// @param flag_inv_busy Read only value?
    ///
    /// @note (dt_fifo_wait_time + dt_fifo_timeout) * (10/in_parallel) * 720 * T clk (us) < 2^LSB (253 -
    /// max_cutoff_period)(us)
    /// Recommendation before the characterization:
    ///     - in_parallel 10
    ///     - dt_fifo_timeout 90
    ///     - dt_fifo_wait_time variable depending on the equation
    virtual void set_invalidation(uint32_t dt_fifo_wait_time, uint32_t dt_fifo_timeout, uint32_t in_parallel,
                                  bool flag_inv_busy)  = 0;
    virtual void get_invalidation(uint32_t &dt_fifo_wait_time, uint32_t &dt_fifo_timeout, uint32_t &in_parallel,
                                  bool &flag_inv_busy) = 0;

    virtual void set_initialization(bool req_init, bool flag_init_busy, bool flag_init_done)    = 0;
    virtual void get_initialization(bool &req_init, bool &flag_init_busy, bool &flag_init_done) = 0;

    virtual uint32_t to_period(uint32_t)           = 0;
    virtual uint32_t to_inverted_duty_cycle(float) = 0;

    // higher level apis:
    virtual bool is_present()       = 0;
    virtual AFKModes get_mode()     = 0;
    virtual bool set_mode(AFKModes) = 0;
    virtual void initialize()       = 0;
    virtual void start()            = 0;
    virtual void stop()             = 0;

private:
    virtual bool wait_for_init() = 0;

    bool initialized_ = false;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_AFK_REGISTERS_API_V0_H
