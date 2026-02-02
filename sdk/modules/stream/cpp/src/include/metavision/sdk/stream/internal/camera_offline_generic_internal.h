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

#ifndef METAVISION_SDK_STREAM_CAMERA_OFFLINE_GENERIC_INTERNAL_H
#define METAVISION_SDK_STREAM_CAMERA_OFFLINE_GENERIC_INTERNAL_H

#include <filesystem>
#include "metavision/sdk/stream/internal/camera_internal.h"

namespace Metavision {
class EventFileReader;
class I_Geometry;

namespace detail {
struct GenericGeometry;

class OfflineGenericPrivate : public Camera::Private {
public:
    OfflineGenericPrivate(const std::filesystem::path &file_path, const FileConfigHints &hints);
    ~OfflineGenericPrivate() override;

private:
    void init();

    Device &device() override;
    OfflineStreamingControl &offline_streaming_control() override;

    timestamp get_last_timestamp() const override;
    I_Geometry &get_geometry() override;
    void start_impl() override;
    void stop_impl() override;
    bool process_impl() override;

    template<typename TimingProfilerType>
    bool process_impl(TimingProfilerType *);

    void save(std::ostream &) const override;
    void load(std::istream &) override;

    bool realtime_playback_speed_;
    timestamp first_ts_, last_ts_;
    uint64_t first_ts_clock_;
    std::unique_ptr<OfflineStreamingControl> osc_;
    std::unique_ptr<EventFileReader> file_reader_;
    std::unique_ptr<GenericGeometry> gen_geom_;
};

} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_STREAM_CAMERA_OFFLINE_GENERIC_INTERNAL_H
