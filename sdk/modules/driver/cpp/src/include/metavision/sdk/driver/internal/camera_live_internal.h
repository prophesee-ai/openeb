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

#ifndef METAVISION_SDK_DRIVER_CAMERA_LIVE_INTERNAL_H
#define METAVISION_SDK_DRIVER_CAMERA_LIVE_INTERNAL_H

#include "metavision/sdk/driver/internal/camera_internal.h"

namespace Metavision {

class Device;
class I_CameraSynchronization;
class I_EventsStreamDecoder;
class I_EventsStream;

namespace detail {

class LivePrivate : public Camera::Private {
public:
    LivePrivate(DeviceConfig *dev_config_ptr = nullptr);
    LivePrivate(OnlineSourceType input_source_type, uint32_t source_index, DeviceConfig *dev_config_ptr = nullptr);
    LivePrivate(const std::string &serial, DeviceConfig *dev_config_ptr = nullptr);
    ~LivePrivate() override;

private:
    void init();

    Device &device() override;
    OfflineStreamingControl &offline_streaming_control() override;
    TriggerOut &trigger_out() override;
    Biases &biases() override;
    Roi &roi() override;
    AntiFlickerModule &antiflicker_module() override;
    ErcModule &erc_module() override;
    EventTrailFilterModule &event_trail_filter_module() override;

    timestamp get_last_timestamp() const override;
    void start_impl() override;
    void stop_impl() override;
    bool process_impl() override;

    template<typename TimingProfilerType>
    bool process_impl(TimingProfilerType *);

    bool start_recording_impl(const std::string &file_path) override;

    std::unique_ptr<Device> device_                    = nullptr;
    I_EventsStream *i_events_stream_                   = nullptr;
    I_EventsStreamDecoder *i_events_stream_decoder_    = nullptr;
    I_CameraSynchronization *i_camera_synchronization_ = nullptr;

    std::unique_ptr<Roi> roi_;
    std::unique_ptr<TriggerOut> trigger_out_;
    std::unique_ptr<Biases> biases_;
    std::unique_ptr<AntiFlickerModule> afk_;
    std::unique_ptr<ErcModule> ercm_;
    std::unique_ptr<EventTrailFilterModule> event_trail_filter_;
};

} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_CAMERA_LIVE_INTERNAL_H
