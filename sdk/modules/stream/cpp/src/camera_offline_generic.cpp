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

#include <regex>

#include "metavision/sdk/base/utils/get_time.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/sdk/stream/internal/callback_tag_ids.h"
#include "metavision/sdk/stream/internal/camera_error_code_internal.h"
#include "metavision/sdk/stream/internal/camera_generation_internal.h"
#include "metavision/sdk/stream/internal/camera_offline_generic_internal.h"
#include "metavision/sdk/stream/internal/cd_internal.h"
#include "metavision/sdk/stream/internal/erc_counter_internal.h"
#include "metavision/sdk/stream/internal/ext_trigger_internal.h"
#include "metavision/sdk/stream/internal/offline_streaming_control_internal.h"
#include "metavision/sdk/stream/dat_event_file_reader.h"
#include "metavision/sdk/stream/hdf5_event_file_reader.h"

namespace Metavision {
namespace detail {

struct GenericGeometry : public I_Geometry {
    GenericGeometry(int width, int height) : width_(width), height_(height) {}
    int get_width() const override {
        return width_;
    }
    int get_height() const override {
        return height_;
    }
    int width_, height_;
};

OfflineGenericPrivate::OfflineGenericPrivate(const std::filesystem::path &file_path, const FileConfigHints &hints) :
    Private(detail::Config()) {
    // clang-format off
    try {
        if (file_path.extension().string() == ".hdf5" || file_path.extension().string() == ".h5") {
            file_reader_ = std::make_unique<HDF5EventFileReader>(file_path, hints.time_shift());
        } else {
            file_reader_ = std::make_unique<DATEventFileReader>(file_path);
        }
    } catch (const CameraException &) {
        throw;
    } catch (...) {
        throw CameraException(CameraErrorCode::CouldNotOpenFile);
    }
    // clang-format on

    realtime_playback_speed_ = hints.real_time_playback();

    init();
}

OfflineGenericPrivate::~OfflineGenericPrivate() {
    if (is_init_) {
        stop();
    }
}

Device &OfflineGenericPrivate::device() {
    throw CameraException(UnsupportedFeatureErrors::DeviceUnavailable, "Device unavailable.");
}

OfflineStreamingControl &OfflineGenericPrivate::offline_streaming_control() {
    return *osc_;
}

timestamp OfflineGenericPrivate::get_last_timestamp() const {
    return last_ts_;
}

I_Geometry &OfflineGenericPrivate::get_geometry() {
    return *gen_geom_;
}

void OfflineGenericPrivate::start_impl() {
    first_ts_ = last_ts_ = -1;
    first_ts_clock_      = 0;
}

void OfflineGenericPrivate::stop_impl() {}

bool OfflineGenericPrivate::process_impl() {
    if (config_.print_timings) {
        return process_impl(timing_profiler_tuple_.get_profiler<true>());
    } else {
        return process_impl(timing_profiler_tuple_.get_profiler<false>());
    }
}

template<typename TimingProfilerType>
bool OfflineGenericPrivate::process_impl(TimingProfilerType *profiler) {
    if (!file_reader_->read()) {
        return false;
    }

    const bool has_decode_callbacks = index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID);
    if (has_decode_callbacks) {
        // emulate real time if needed
        if (realtime_playback_speed_) {
            const timestamp cur_ts = last_ts_;

            const uint64_t cur_ts_clock = get_system_time_us();

            // compute the offset first, if never done
            if (first_ts_clock_ == 0 && cur_ts != first_ts_) {
                first_ts_clock_ = cur_ts_clock;
                first_ts_       = cur_ts;
            }

            const uint64_t expected_ts = first_ts_clock_ + (cur_ts - first_ts_);

            if (cur_ts_clock < expected_ts) {
                std::this_thread::sleep_for(std::chrono::microseconds(expected_ts - cur_ts_clock));
            }
        }
    }

    return true;
}

std::string versions_to_name(int version_major, int version_minor) {
    static const std::map<std::pair<int, int>, std::string> m{
        {{3, 0}, "Gen3.0"}, {{3, 1}, "Gen3.1"}, {{4, 0}, "Gen4.0"}, {{4, 1}, "Gen4.1"}, {{4, 2}, "IMX636"},
    };
    auto it = m.find(std::make_pair(version_major, version_minor));
    if (it != m.end()) {
        return it->second;
    }
    return std::string();
}

void OfflineGenericPrivate::save(std::ostream &) const {
    throw CameraException(UnsupportedFeatureErrors::SerializationUnsupported,
                          "Cannot serialize when running from a file.");
}

void OfflineGenericPrivate::load(std::istream &) {
    throw CameraException(UnsupportedFeatureErrors::SerializationUnsupported,
                          "Cannot deserialize when running from a file.");
}

void OfflineGenericPrivate::init() {
    is_init_ = true;

    metadata_map_ = file_reader_->get_metadata_map();

    auto it = metadata_map_.find("serial_number");
    if (it != metadata_map_.end()) {
        camera_configuration_.serial_number = it->second;
    }
    it = metadata_map_.find("integrator_name");
    if (it != metadata_map_.end()) {
        camera_configuration_.integrator = it->second;
    }
    it = metadata_map_.find("firmware_version");
    if (it != metadata_map_.end()) {
        camera_configuration_.firmware_version = it->second;
    }
    camera_configuration_.data_encoding_format = "ECF";

    it = metadata_map_.find("geometry");
    if (it != metadata_map_.end()) {
        const auto &g = it->second;
        std::regex rgx("(\\d+)x(\\d+)");
        std::smatch match;
        if (std::regex_search(g.begin(), g.end(), match, rgx)) {
            gen_geom_.reset(new GenericGeometry(std::stoi(match.str(1)), std::stoi(match.str(2))));
        }
    }

    cd_.reset(CD::Private::build(index_manager_));
    file_reader_->add_read_callback([this](const EventCD *begin, const EventCD *end) {
        for (auto &&cb : cd_->get_pimpl().get_cbs()) {
            cb(begin, end);
        }
        last_ts_ = std::prev(end)->t;
    });

    ext_trigger_.reset(ExtTrigger::Private::build(index_manager_));
    file_reader_->add_read_callback([this](const EventExtTrigger *begin, const EventExtTrigger *end) {
        for (auto &&cb : ext_trigger_->get_pimpl().get_cbs()) {
            cb(begin, end);
        }
    });

    erc_counter_.reset(ERCCounter::Private::build(index_manager_));
    file_reader_->add_read_callback([this](const EventERCCounter *begin, const EventERCCounter *end) {
        for (auto &&cb : erc_counter_->get_pimpl().get_cbs()) {
            cb(begin, end);
        }
    });

    file_reader_->add_seek_callback([this](timestamp t) {
        first_ts_       = t;
        first_ts_clock_ = 0;
    });

    it = metadata_map_.find("generation");
    if (it != metadata_map_.end()) {
        const auto &g = it->second;
        std::regex rgx("(\\d+).(\\d+)");
        std::smatch match;
        if (std::regex_search(g.begin(), g.end(), match, rgx)) {
            auto version_major = std::stoi(match.str(1));
            auto version_minor = std::stoi(match.str(2));
            generation_.reset(CameraGeneration::Private::build(version_major, version_minor,
                                                               versions_to_name(version_major, version_minor)));
        }
    }

    osc_.reset(OfflineStreamingControl::Private::build(*file_reader_));
}

} // namespace detail
} // namespace Metavision
