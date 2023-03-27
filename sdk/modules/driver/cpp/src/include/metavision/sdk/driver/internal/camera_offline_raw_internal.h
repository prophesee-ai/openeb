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

#ifndef METAVISION_SDK_DRIVER_CAMERA_OFFLINE_RAW_INTERNAL_H
#define METAVISION_SDK_DRIVER_CAMERA_OFFLINE_RAW_INTERNAL_H

#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/sdk/driver/internal/camera_internal.h"

namespace Metavision {

class Device;
class I_EventsStream;
class RAWEventFileReader;

namespace detail {

class OfflineRawPrivate : public Camera::Private {
public:
    OfflineRawPrivate(const std::string &rawfile, const FileConfigHints &hints);
    ~OfflineRawPrivate() override;

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

    std::unique_ptr<Device> device_  = nullptr;
    I_EventsStream *i_events_stream_ = nullptr;
    I_Decoder *i_decoder_            = nullptr;

    bool realtime_playback_speed_;
    timestamp first_ts_, last_ts_;
    uint64_t first_ts_clock_;
    std::unique_ptr<OfflineStreamingControl> osc_;
    std::unique_ptr<RAWEventFileReader> file_reader_;
};

} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_CAMERA_OFFLINE_RAW_INTERNAL_H
