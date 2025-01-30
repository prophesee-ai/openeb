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

#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_event_frame_decoder.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/sdk/base/utils/get_time.h"
#include "metavision/sdk/stream/internal/callback_tag_ids.h"
#include "metavision/sdk/stream/internal/camera_error_code_internal.h"
#include "metavision/sdk/stream/internal/camera_generation_internal.h"
#include "metavision/sdk/stream/internal/camera_offline_raw_internal.h"
#include "metavision/sdk/stream/internal/cd_internal.h"
#include "metavision/sdk/stream/internal/ext_trigger_internal.h"
#include "metavision/sdk/stream/internal/erc_counter_internal.h"
#include "metavision/sdk/stream/internal/frame_diff_internal.h"
#include "metavision/sdk/stream/internal/frame_histo_internal.h"
#include "metavision/sdk/stream/internal/offline_streaming_control_internal.h"
#include "metavision/sdk/stream/internal/monitoring_internal.h"
#include "metavision/sdk/stream/internal/raw_data_internal.h"
#include "metavision/sdk/stream/raw_event_file_reader.h"

namespace Metavision {
namespace detail {

OfflineRawPrivate::OfflineRawPrivate(const std::filesystem::path &rawfile, const FileConfigHints &hints) :
    Private(detail::Config()) {
    RawFileConfig raw_file_stream_config;
    raw_file_stream_config.n_events_to_read_ = hints.max_read_per_op() / 4;
    raw_file_stream_config.n_read_buffers_   = hints.max_memory() / hints.max_read_per_op();
    raw_file_stream_config.do_time_shifting_ = hints.time_shift();
    raw_file_stream_config.build_index_      = hints.get<bool>("index", raw_file_stream_config.build_index_);

    device_ = DeviceDiscovery::open_raw_file(rawfile, raw_file_stream_config);
    if (!device_) {
        // We should never get here as open_raw_file should throw an exception if the system is unknown
        throw CameraException(CameraErrorCode::InvalidRawfile,
                              "The RAW file at " + rawfile.string() +
                                  " could not be read. Please check that the file has "
                                  "been recorded with an event-based device or contact "
                                  "the support.");
    }

    file_reader_ = std::make_unique<RAWEventFileReader>(*device_, rawfile);

    realtime_playback_speed_ = hints.real_time_playback();

    init();
}

OfflineRawPrivate::~OfflineRawPrivate() {
    if (is_init_) {
        stop();
    }
}

Device &OfflineRawPrivate::device() {
    return *device_;
}

OfflineStreamingControl &OfflineRawPrivate::offline_streaming_control() {
    if (!osc_) {
        throw CameraException(UnsupportedFeatureErrors::OfflineStreamingControlUnavailable,
                              "Offline streaming control unavailable.");
    }

    return *osc_;
}

timestamp OfflineRawPrivate::get_last_timestamp() const {
    return last_ts_;
}

void OfflineRawPrivate::start_impl() {
    first_ts_ = last_ts_ = -1;
    first_ts_clock_      = 0;

    file_reader_->start();
}

void OfflineRawPrivate::stop_impl() {
    file_reader_->stop();
}

bool OfflineRawPrivate::process_impl() {
    if (config_.print_timings) {
        return process_impl(timing_profiler_tuple_.get_profiler<true>());
    } else {
        return process_impl(timing_profiler_tuple_.get_profiler<false>());
    }
}

template<typename TimingProfilerType>
bool OfflineRawPrivate::process_impl(TimingProfilerType *profiler) {
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

void OfflineRawPrivate::save(std::ostream &) const {
    throw CameraException(UnsupportedFeatureErrors::SerializationUnsupported,
                          "Cannot serialize when running from a file.");
}

void OfflineRawPrivate::load(std::istream &) {
    throw CameraException(UnsupportedFeatureErrors::SerializationUnsupported,
                          "Cannot deserialize when running from a file.");
}

void OfflineRawPrivate::init() {
    is_init_ = true;
    if (!device_) {
        throw CameraException(CameraErrorCode::CameraNotFound);
    }
    auto *hw_identification = device_->get_facility<I_HW_Identification>();
    if (!hw_identification) {
        throw(CameraException(InternalInitializationErrors::IBoardIdentificationNotFound));
    }

    camera_configuration_.data_encoding_format = hw_identification->get_current_data_encoding_format();
    camera_configuration_.serial_number        = hw_identification->get_serial();
    camera_configuration_.integrator           = hw_identification->get_integrator();
    camera_configuration_.firmware_version     = hw_identification->get_system_info()["System Version"];

    metadata_map_ = file_reader_->get_metadata_map();

    auto *plugin_info = device_->get_facility<Metavision::I_PluginSoftwareInfo>();
    if (plugin_info) {
        camera_configuration_.plugin_name = plugin_info->get_plugin_name();
    }

    i_events_stream_ = device_->get_facility<I_EventsStream>();
    if (!i_events_stream_) {
        throw CameraException(InternalInitializationErrors::IEventsStreamNotFound);
    }

    i_decoder_ = device_->get_facility<I_EventsStreamDecoder>();

    generation_.reset(CameraGeneration::Private::build(*device_));

    raw_data_.reset(RawData::Private::build(index_manager_));
    file_reader_->add_raw_read_callback([this](const std::uint8_t *begin, const std::uint8_t *end) {
        for (auto &&cb : raw_data_->get_pimpl().get_cbs()) {
            cb(begin, std::distance(begin, end));
        }
    });

    I_EventDecoder<EventCD> *i_cd_events_decoder = device_->get_facility<I_EventDecoder<EventCD>>();
    if (i_cd_events_decoder) {
        cd_.reset(CD::Private::build(index_manager_));
        file_reader_->add_read_callback([this](const EventCD *begin, const EventCD *end) {
            for (auto &&cb : cd_->get_pimpl().get_cbs()) {
                cb(begin, end);
            }
            last_ts_ = std::prev(end)->t;
        });
    }

    I_EventDecoder<EventExtTrigger> *i_ext_trigger_events_decoder =
        device_->get_facility<I_EventDecoder<EventExtTrigger>>();
    if (i_ext_trigger_events_decoder) {
        ext_trigger_.reset(ExtTrigger::Private::build(index_manager_));
        file_reader_->add_read_callback([this](const EventExtTrigger *begin, const EventExtTrigger *end) {
            for (auto &&cb : ext_trigger_->get_pimpl().get_cbs()) {
                cb(begin, end);
            }
        });
    }

    I_EventDecoder<EventERCCounter> *i_erc_count_events_decoder =
        device_->get_facility<I_EventDecoder<EventERCCounter>>();
    if (i_erc_count_events_decoder) {
        erc_counter_.reset(ERCCounter::Private::build(index_manager_));
        file_reader_->add_read_callback([this](const EventERCCounter *begin, const EventERCCounter *end) {
            for (auto &&cb : erc_counter_->get_pimpl().get_cbs()) {
                cb(begin, end);
            }
        });
    }

    I_EventFrameDecoder<RawEventFrameHisto> *i_histogram_decoder =
        device_->get_facility<I_EventFrameDecoder<RawEventFrameHisto>>();
    if (i_histogram_decoder) {
        frame_histo_.reset(FrameHisto::Private::build(index_manager_));
        file_reader_->add_read_callback([this](const RawEventFrameHisto &h) {
            for (auto &&cb : frame_histo_->get_pimpl().get_cbs()) {
                cb(h);
            }
        });

        i_decoder_ = i_histogram_decoder;
    }

    I_EventFrameDecoder<RawEventFrameDiff> *i_diff_decoder =
        device_->get_facility<I_EventFrameDecoder<RawEventFrameDiff>>();
    if (i_diff_decoder) {
        frame_diff_.reset(FrameDiff::Private::build(index_manager_));
        file_reader_->add_read_callback([this](const RawEventFrameDiff &h) {
            for (auto &&cb : frame_diff_->get_pimpl().get_cbs()) {
                cb(h);
            }
        });
        i_decoder_ = i_diff_decoder;
    }

    I_EventFrameDecoder<PointCloud> *i_pointcloud_decoder = device_->get_facility<I_EventFrameDecoder<PointCloud>>();
    if (i_pointcloud_decoder) {
        file_reader_->add_read_callback([this](const PointCloud &pc) {
            // For now PointClouds are not handled by Camera object
        });
        i_decoder_ = i_pointcloud_decoder;
    }

    I_EventDecoder<EventMonitoring> *i_monitoring_events_decoder =
        device_->get_facility<I_EventDecoder<EventMonitoring>>();
    if (i_monitoring_events_decoder) {
        monitoring_.reset(Monitoring::Private::build(index_manager_));
        i_monitoring_events_decoder->add_event_buffer_callback(
            [this](const EventMonitoring *begin, const EventMonitoring *end) {
                for (auto &&cb : monitoring_->get_pimpl().get_cbs()) {
                    cb(begin, end);
                }
            });
    }

    if (!i_decoder_) {
        throw CameraException(InternalInitializationErrors::IDecoderNotFound);
    }

    osc_.reset(OfflineStreamingControl::Private::build(*file_reader_));
    file_reader_->add_seek_callback([this](timestamp t) {
        first_ts_       = t;
        first_ts_clock_ = 0;
    });
}

} // namespace detail
} // namespace Metavision
