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

#include <boost/filesystem.hpp>

#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_event_frame_decoder.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/sdk/driver/internal/callback_tag_ids.h"
#include "metavision/sdk/driver/internal/camera_error_code_internal.h"
#include "metavision/sdk/driver/internal/camera_generation_internal.h"
#include "metavision/sdk/driver/internal/camera_live_internal.h"
#include "metavision/sdk/driver/internal/cd_internal.h"
#include "metavision/sdk/driver/internal/ext_trigger_internal.h"
#include "metavision/sdk/driver/internal/erc_counter_internal.h"
#include "metavision/sdk/driver/internal/frame_diff_internal.h"
#include "metavision/sdk/driver/internal/frame_histo_internal.h"
#include "metavision/sdk/driver/internal/raw_data_internal.h"

namespace Metavision {
namespace detail {

DeviceConfig dummy_config;

LivePrivate::LivePrivate(DeviceConfig *dev_config_ptr) : Private(detail::Config()) {
    AvailableSourcesList available_systems = Camera::list_online_sources();
    AvailableSourcesList::iterator it;
    if ((it = available_systems.find(OnlineSourceType::EMBEDDED)) != available_systems.end()) {
        if (!it->second.empty()) {
            device_ = DeviceDiscovery::open(it->second[0], dev_config_ptr ? *dev_config_ptr : dummy_config);
        }
    } else if ((it = available_systems.find(OnlineSourceType::USB)) != available_systems.end()) {
        if (!it->second.empty()) {
            device_ = DeviceDiscovery::open(it->second[0], dev_config_ptr ? *dev_config_ptr : dummy_config);
        }
    }

    init();
}

LivePrivate::LivePrivate(OnlineSourceType input_source_type, uint32_t source_index, DeviceConfig *dev_config_ptr) :
    Private(detail::Config()) {
    AvailableSourcesList available_systems = Camera::list_online_sources();
    AvailableSourcesList::iterator it;
    if ((it = available_systems.find(input_source_type)) != available_systems.end()) {
        if (it->second.size() > source_index) {
            device_ = DeviceDiscovery::open(it->second[source_index], dev_config_ptr ? *dev_config_ptr : dummy_config);
        } else {
            throw CameraException(CameraErrorCode::CameraNotFound,
                                  "Camera " + std::to_string(source_index) + "not found. Check that at least " +
                                      std::to_string(source_index) + " camera of input type are plugged and retry.");
        }
    }

    init();
}

LivePrivate::LivePrivate(const std::string &serial, DeviceConfig *dev_config_ptr) : Private(detail::Config()) {
    device_ = DeviceDiscovery::open(serial, dev_config_ptr ? *dev_config_ptr : dummy_config);
    if (!device_) {
        throw CameraException(CameraErrorCode::CameraNotFound, "Camera with serial " + serial + " has not been found.");
    }

    init();
}

LivePrivate::~LivePrivate() {
    if (is_init_) {
        stop();
    }
}

Device &LivePrivate::device() {
    return *device_;
}

OfflineStreamingControl &LivePrivate::offline_streaming_control() {
    throw CameraException(UnsupportedFeatureErrors::OfflineStreamingControlUnavailable,
                          "Cannot get offline streaming control from a live camera.");
}

TriggerOut &LivePrivate::trigger_out() {
    if (!trigger_out_) {
        throw CameraException(UnsupportedFeatureErrors::TriggerOutUnavailable);
    }
    return *trigger_out_;
}

Biases &LivePrivate::biases() {
    if (!biases_) {
        throw CameraException(InternalInitializationErrors::ILLBiasesNotFound);
    }

    return *biases_;
}

Roi &LivePrivate::roi() {
    if (!roi_) {
        throw CameraException(InternalInitializationErrors::IRoiNotFound);
    }
    return *roi_;
}

AntiFlickerModule &LivePrivate::antiflicker_module() {
    if (!afk_) {
        throw CameraException(UnsupportedFeatureErrors::AntiFlickerModuleUnavailable);
    }
    return *afk_;
}

ErcModule &LivePrivate::erc_module() {
    if (!ercm_) {
        throw CameraException(UnsupportedFeatureErrors::ErcModuleUnavailable);
    }
    return *ercm_;
}

EventTrailFilterModule &LivePrivate::event_trail_filter_module() {
    if (!event_trail_filter_) {
        throw CameraException(UnsupportedFeatureErrors::EventTrailFilterModuleUnavailable);
    }
    return *event_trail_filter_;
}

timestamp LivePrivate::get_last_timestamp() const {
    if (index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID) > 0) {
        if (i_events_stream_decoder_) {
            return i_events_stream_decoder_->get_last_timestamp();
        }
    }
    return -1;
}

void LivePrivate::start_impl() {
    if (i_events_stream_decoder_) {
        // it could be the first time we start streaming and feeding events to the decoder, but
        // if it's not the case, we need to reset the decoder state so that new events are not
        // decoded using the current state (which is probably wrong : i.e wrong time base, etc.)
        i_events_stream_decoder_->reset_timestamp(-1);
    }
    if (i_events_stream_) {
        i_events_stream_->start();
    }
}

void LivePrivate::stop_impl() {
    if (i_events_stream_) {
        i_events_stream_->stop();
    }
}

bool LivePrivate::process_impl() {
    if (config_.print_timings) {
        return process_impl(timing_profiler_tuple_.get_profiler<true>());
    } else {
        return process_impl(timing_profiler_tuple_.get_profiler<false>());
    }
}

template<typename TimingProfilerType>
bool LivePrivate::process_impl(TimingProfilerType *profiler) {
    int res             = 0;
    long int n_rawbytes = 0;

    {
        typename TimingProfilerType::TimedOperation t("Polling", profiler);
        res = i_events_stream_->wait_next_buffer();
    }

    if (res < 0) {
        return false;
    } else if (res > 0) {
        typename TimingProfilerType::TimedOperation t("Processing", profiler);
        I_EventsStream::RawData *ev_buffer, *ev_buffer_end;
        ev_buffer     = i_events_stream_->get_latest_raw_data(n_rawbytes);
        ev_buffer_end = ev_buffer + n_rawbytes;

        const bool has_decode_callbacks = index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID);
        if (has_decode_callbacks) {
            i_events_stream_decoder_->decode(ev_buffer, ev_buffer_end);
            t.setNumProcessedElements(n_rawbytes / i_events_stream_decoder_->get_raw_event_size_bytes());
        }

        // ... then we call the raw buffer callback so that a user has access to some info (e.g last
        // decoded timestamp) when the raw callback is called
        for (auto &cb : raw_data_->get_pimpl().get_cbs()) {
            cb(ev_buffer, n_rawbytes);
        }
    }

    return true;
}

bool LivePrivate::start_recording_impl(const std::string &file_path) {
    std::string base_path = boost::filesystem::change_extension(file_path, "").string();
    if (biases_) {
        biases_->save_to_file(base_path + ".bias");
    }
    return Camera::Private::start_recording_impl(file_path);
}

void LivePrivate::init() {
    is_init_ = true;
    if (!device_) {
        throw CameraException(CameraErrorCode::CameraNotFound);
    }

    auto *hw_identification = device_->get_facility<Metavision::I_HW_Identification>();
    if (!hw_identification) {
        throw(CameraException(InternalInitializationErrors::IBoardIdentificationNotFound));
    }
    camera_configuration_.data_encoding_format = hw_identification->get_current_data_encoding_format();
    camera_configuration_.system_ID            = std::to_string(hw_identification->get_system_id());
    camera_configuration_.serial_number        = hw_identification->get_serial();
    camera_configuration_.integrator           = hw_identification->get_integrator();
    camera_configuration_.firmware_version     = hw_identification->get_system_info()["System Version"];

    auto *plugin_info = device_->get_facility<Metavision::I_PluginSoftwareInfo>();
    if (plugin_info) {
        camera_configuration_.plugin_name = plugin_info->get_plugin_name();
    }

    i_events_stream_ = device_->get_facility<I_EventsStream>();
    if (!i_events_stream_) {
        throw CameraException(InternalInitializationErrors::IEventsStreamNotFound);
    }

    I_Geometry *i_geometry = device_->get_facility<I_Geometry>();
    if (!i_geometry) {
        throw CameraException(InternalInitializationErrors::IGeometryNotFound);
    }
    geometry_.reset(new Geometry(i_geometry));

    i_events_stream_decoder_ = device_->get_facility<I_EventsStreamDecoder>();

    generation_.reset(CameraGeneration::Private::build(*device_));

    raw_data_.reset(RawData::Private::build(index_manager_));

    cd_.reset(CD::Private::build(index_manager_));
    I_EventDecoder<EventCD> *i_cd_events_decoder = device_->get_facility<I_EventDecoder<EventCD>>();
    if (i_cd_events_decoder) {
        i_cd_events_decoder->add_event_buffer_callback([this](const EventCD *begin, const EventCD *end) {
            for (auto &&cb : cd_->get_pimpl().get_cbs()) {
                cb(begin, end);
            }
        });
    }

    I_EventDecoder<EventExtTrigger> *i_ext_trigger_events_decoder =
        device_->get_facility<I_EventDecoder<EventExtTrigger>>();
    if (i_ext_trigger_events_decoder) {
        ext_trigger_.reset(ExtTrigger::Private::build(index_manager_));
        i_ext_trigger_events_decoder->add_event_buffer_callback(
            [this](const EventExtTrigger *begin, const EventExtTrigger *end) {
                for (auto &&cb : ext_trigger_->get_pimpl().get_cbs()) {
                    cb(begin, end);
                }
            });
    }

    I_EventDecoder<EventERCCounter> *i_erc_count_events_decoder =
        device_->get_facility<I_EventDecoder<EventERCCounter>>();
    if (i_erc_count_events_decoder) {
        erc_counter_.reset(ERCCounter::Private::build(index_manager_));
        i_erc_count_events_decoder->add_event_buffer_callback(
            [this](const EventERCCounter *begin, const EventERCCounter *end) {
                for (auto &&cb : erc_counter_->get_pimpl().get_cbs()) {
                    cb(begin, end);
                }
            });
    }

    I_EventFrameDecoder<RawEventFrameHisto> *i_histogram_decoder =
        device_->get_facility<I_EventFrameDecoder<RawEventFrameHisto>>();
    if (i_histogram_decoder) {
        frame_histo_.reset(FrameHisto::Private::build(index_manager_));
        i_histogram_decoder->add_event_frame_callback([this](const RawEventFrameHisto &h) {
            for (auto &&cb : frame_histo_->get_pimpl().get_cbs()) {
                cb(h);
            }
        });
    }

    I_EventFrameDecoder<RawEventFrameDiff> *i_diff_decoder =
        device_->get_facility<I_EventFrameDecoder<RawEventFrameDiff>>();
    if (i_diff_decoder) {
        frame_diff_.reset(FrameDiff::Private::build(index_manager_));
        i_diff_decoder->add_event_frame_callback([this](const RawEventFrameDiff &h) {
            for (auto &&cb : frame_diff_->get_pimpl().get_cbs()) {
                cb(h);
            }
        });
    }

    if (!i_events_stream_decoder_ && !i_histogram_decoder && !i_diff_decoder) {
        throw CameraException(InternalInitializationErrors::IDecoderNotFound);
    }

    i_camera_synchronization_ = device_->get_facility<I_CameraSynchronization>();
    if (!i_camera_synchronization_) {
        throw CameraException(InternalInitializationErrors::IDeviceControlNotFound);
    }

    I_ROI *i_roi = device_->get_facility<I_ROI>();
    if (i_roi) {
        roi_.reset(new Roi(i_roi));
    }

    I_TriggerOut *i_trigger_out = device_->get_facility<I_TriggerOut>();
    if (i_trigger_out) {
        trigger_out_.reset(new TriggerOut(i_trigger_out));
    }

    // all ext trigger enabled by default

    I_LL_Biases *i_ll_biases = device_->get_facility<I_LL_Biases>();
    if (i_ll_biases) {
        biases_.reset(new Biases(i_ll_biases));
    }

    I_AntiFlickerModule *i_afk = device_->get_facility<I_AntiFlickerModule>();
    if (i_afk) {
        afk_.reset(new AntiFlickerModule(i_afk));
    }

    I_ErcModule *i_ercm = device_->get_facility<I_ErcModule>();
    if (i_ercm) {
        ercm_.reset(new ErcModule(i_ercm));
    }

    I_EventTrailFilterModule *i_event_trail_filter = device_->get_facility<I_EventTrailFilterModule>();
    if (i_event_trail_filter) {
        event_trail_filter_.reset(new EventTrailFilterModule(i_event_trail_filter));
    }
}

} // namespace detail
} // namespace Metavision
