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

#ifndef METAVISION_HAL_PSEE_DEVICE_CONTROL_H
#define METAVISION_HAL_PSEE_DEVICE_CONTROL_H

#include <cstdint>
#include <ostream>

#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/utils/device_control.h"

namespace Metavision {

class PseeTriggerIn;
class PseeTriggerOut;

/// @brief Device Control facility controls camera mode and allows to start, reset and stop it.
class PseeDeviceControl : public I_CameraSynchronization, public Metavision::DeviceControl {
public:
    PseeDeviceControl(StreamFormat fmt);

    /// @brief Starts the generation of events from the camera side
    /// @warning All triggers will be disabled at stop. User should re-enable required triggers before start.
    virtual void start() override final;

    /// @brief Stops the generation of events from the camera side
    virtual void stop() override final;

    /// @brief Sets the camera streaming EVT format
    bool set_evt_format(const StreamFormat &fmt);

    /// @brief Retrieves EVT format
    /// @return event format
    const StreamFormat &get_evt_format() const;

    /// @brief Sets the camera in standalone mode.
    ///
    /// The camera does not interact with other devices.
    /// @return true on success
    virtual bool set_mode_standalone() override final;

    /// @brief Sets the camera as master
    ///
    /// The camera sends clock signal to another device
    /// @return true on success
    virtual bool set_mode_master() override final;

    /// @brief Sets the camera as slave
    ///
    /// The camera receives the clock from another device
    /// @return true on success
    virtual bool set_mode_slave() override final;

    /// @brief Retrieves Synchronization mode
    /// @return synchronization mode
    virtual SyncMode get_mode() override final;

    virtual long long get_sensor_id();

    /// @brief Gets the trigger in facility
    /// @param checked If true, will throw an exception if the trigger in facility has not been set
    /// @return The trigger in facility
    std::shared_ptr<PseeTriggerIn> get_trigger_in(bool checked = true) const;
    void set_trigger_in(const std::shared_ptr<PseeTriggerIn> &trigger_in);

    /// @brief Gets the trigger out facility
    /// @param checked If true, will throw an exception if the trigger out facility has not been set
    /// @return The trigger out facility
    std::shared_ptr<PseeTriggerOut> get_trigger_out(bool checked = true) const;
    void set_trigger_out(const std::shared_ptr<PseeTriggerOut> &trigger_out);

protected:
    virtual void initialize();
    virtual void destroy();

private:
    virtual void setup() override;
    virtual void teardown() override;

    virtual void start_impl() = 0;
    virtual void stop_impl()  = 0;

    virtual bool set_evt_format_impl(const StreamFormat &fmt) = 0;
    virtual bool set_mode_standalone_impl()                   = 0;
    virtual bool set_mode_slave_impl()                        = 0;
    virtual bool set_mode_master_impl()                       = 0;

    // ----------------------------------------------------------------------
    // Important : store those as weak_ptrs to avoid cyclic ownership issues
    std::weak_ptr<PseeTriggerOut> trigger_out_;
    std::weak_ptr<PseeTriggerIn> trigger_in_;
    // ----------------------------------------------------------------------

    StreamFormat format_;
    SyncMode sync_mode_;
    bool streaming_;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_DEVICE_CONTROL_H
