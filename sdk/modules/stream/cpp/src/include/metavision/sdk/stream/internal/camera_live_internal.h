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

#ifndef METAVISION_SDK_STREAM_CAMERA_LIVE_INTERNAL_H
#define METAVISION_SDK_STREAM_CAMERA_LIVE_INTERNAL_H

#include <filesystem>

#include "metavision/sdk/stream/internal/camera_internal.h"

namespace Metavision {

class Device;
class I_CameraSynchronization;
class I_Decoder;
class I_EventsStreamDecoder;
class I_EventsStream;

namespace detail {

class LivePrivate : public Camera::Private {
public:
    LivePrivate(const DeviceConfig *dev_config_ptr = nullptr);
    LivePrivate(OnlineSourceType input_source_type, uint32_t source_index,
                const DeviceConfig *dev_config_ptr = nullptr);
    LivePrivate(const std::string &serial, const DeviceConfig *dev_config_ptr = nullptr);
    ~LivePrivate() override;

private:
    void init();

    Device &device() override;
    OfflineStreamingControl &offline_streaming_control() override;

    timestamp get_last_timestamp() const override;
    void start_impl() override;
    void stop_impl() override;
    bool process_impl() override;

    template<typename TimingProfilerType>
    bool process_impl(TimingProfilerType *);

    bool start_recording_impl(const std::filesystem::path &file_path) override;

    void save(std::ostream &) const override;
    void load(std::istream &) override;

    std::unique_ptr<Device> device_                    = nullptr;
    I_EventsStream *i_events_stream_                   = nullptr;
    I_EventsStreamDecoder *i_events_stream_decoder_    = nullptr;
    I_Decoder *i_decoder_                              = nullptr;
};

} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_STREAM_CAMERA_LIVE_INTERNAL_H
