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

#include <memory>

#include "metavision/psee_hw_layer/facilities/psee_device_control.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

namespace Metavision {

PseeDeviceControl::PseeDeviceControl(StreamFormat fmt) :
    format_(fmt), sync_mode_(SyncMode::STANDALONE), streaming_(false) {}

void PseeDeviceControl::start() {
    start_impl();
    streaming_ = true;
}

void PseeDeviceControl::stop() {
    stop_impl();
    streaming_ = false;
}

bool PseeDeviceControl::set_evt_format(const StreamFormat &fmt) {
    if (streaming_) {
        return false;
    }
    bool valid = set_evt_format_impl(fmt);
    if (valid) {
        format_ = fmt;
    }
    return valid;
}

const StreamFormat &PseeDeviceControl::get_evt_format() const {
    return format_;
}

bool PseeDeviceControl::set_mode_standalone() {
    if (streaming_) {
        return false;
    }
    bool valid = set_mode_standalone_impl();
    if (valid) {
        sync_mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
    }
    return valid;
}

bool PseeDeviceControl::set_mode_slave() {
    if (streaming_) {
        return false;
    }
    bool valid = set_mode_slave_impl();
    if (valid) {
        sync_mode_ = I_CameraSynchronization::SyncMode::SLAVE;
    }
    return valid;
}

bool PseeDeviceControl::set_mode_master() {
    if (streaming_) {
        return false;
    }
    bool valid = set_mode_master_impl();
    if (valid) {
        sync_mode_ = I_CameraSynchronization::SyncMode::MASTER;
    }
    return valid;
}

I_CameraSynchronization::SyncMode PseeDeviceControl::get_mode() const {
    return sync_mode_;
}

long long PseeDeviceControl::get_sensor_id() {
    return -1;
}

std::shared_ptr<PseeTriggerIn> PseeDeviceControl::get_trigger_in(bool checked) const {
    auto trigger_in = trigger_in_.lock();
    if (checked && !trigger_in) {
        throw(HalException(PseeHalPluginErrorCode::TriggerInNotFound, "Trigger in facility not set."));
    }
    return trigger_in;
}

void PseeDeviceControl::set_trigger_in(const std::shared_ptr<PseeTriggerIn> &trigger_in) {
    trigger_in_ = trigger_in;
}

std::shared_ptr<PseeTriggerOut> PseeDeviceControl::get_trigger_out(bool checked) const {
    auto trigger_out = trigger_out_.lock();
    if (checked && !trigger_out) {
        throw(HalException(PseeHalPluginErrorCode::TriggerOutNotFound, "Trigger out facility not set."));
    }
    return trigger_out;
}

void PseeDeviceControl::set_trigger_out(const std::shared_ptr<PseeTriggerOut> &trigger_out) {
    trigger_out_ = trigger_out;
}

void PseeDeviceControl::initialize() {}

void PseeDeviceControl::destroy() {}

void PseeDeviceControl::setup() {
    initialize();
}

void PseeDeviceControl::teardown() {
    stop();
    destroy();
}

} // namespace Metavision
