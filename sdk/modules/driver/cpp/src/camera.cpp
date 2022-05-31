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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <boost/filesystem.hpp>

#include "metavision/sdk/driver/internal/camera_internal.h"
#include "metavision/sdk/driver/biases.h"
#include "metavision/sdk/driver/offline_streaming_control.h"
#include "metavision/sdk/driver/internal/offline_streaming_control_internal.h"
#include "metavision/sdk/base/utils/callback_id.h"
#include "metavision/sdk/base/utils/get_time.h"
#include "metavision/sdk/core/utils/callback_manager.h"
#include "metavision/sdk/driver/camera_error_code.h"
#include "metavision/sdk/driver/internal/camera_error_code_internal.h"
#include "metavision/sdk/driver/camera_exception.h"
#include "metavision/sdk/driver/cd.h"
#include "metavision/sdk/driver/internal/cd_internal.h"
#include "metavision/sdk/core/utils/counter_map.h"
#include "metavision/sdk/base/events/event2d.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/sdk/driver/ext_trigger.h"
#include "metavision/sdk/driver/internal/ext_trigger_internal.h"
#include "metavision/sdk/driver/geometry.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/facilities/i_roi.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/sdk/core/utils/index_generator.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_trigger_out.h"
#include "metavision/sdk/driver/raw_data.h"
#include "metavision/sdk/driver/internal/raw_data_internal.h"
#include "metavision/sdk/driver/internal/camera_generation_internal.h"
#include "metavision/sdk/driver/roi.h"
#include "metavision/sdk/driver/trigger_out.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/driver/internal/callback_tag_ids.h"

namespace Metavision {

// ********************
// PIMPL
Camera::Private::Private(bool empty_init) {
    if (empty_init) {
        return;
    }

    is_init_              = true;
    detail::Config config = detail::Config();

    std::string serial                     = "";
    AvailableSourcesList available_systems = Camera::list_online_sources();
    if (available_systems.find(OnlineSourceType::EMBEDDED) != available_systems.end()) {
        if (available_systems[OnlineSourceType::EMBEDDED].size()) {
            serial  = available_systems[OnlineSourceType::EMBEDDED][0];
            device_ = DeviceDiscovery::open(available_systems[OnlineSourceType::EMBEDDED][0]);
        }
    } else if (available_systems.find(OnlineSourceType::USB) != available_systems.end()) {
        if (available_systems[OnlineSourceType::USB].size()) {
            serial  = available_systems[OnlineSourceType::USB][0];
            device_ = DeviceDiscovery::open(available_systems[OnlineSourceType::USB][0]);
        }
    }

    init_online_interfaces(config);
    init_common_interfaces(serial, config);
}

Camera::Private::Private(OnlineSourceType input_source_type, uint32_t source_index) {
    is_init_              = true;
    detail::Config config = detail::Config();

    AvailableSourcesList available_systems = Camera::list_online_sources();

    if (available_systems.find(input_source_type) != available_systems.end()) {
        if (available_systems[input_source_type].size() > source_index) {
            device_ = DeviceDiscovery::open(available_systems[input_source_type][source_index]);
        } else {
            throw CameraException(CameraErrorCode::CameraNotFound,
                                  "Camera " + std::to_string(source_index) + "not found. Check that at least " +
                                      std::to_string(source_index) + " camera of input type are plugged and retry.");
        }
    }

    init_online_interfaces(config);
    init_common_interfaces(available_systems[input_source_type][source_index], config);
}

Camera::Private::Private(const Serial &serial) {
    is_init_              = true;
    detail::Config config = detail::Config();

    device_ = DeviceDiscovery::open(serial.serial_);

    if (!device_) {
        throw CameraException(CameraErrorCode::CameraNotFound,
                              "Camera with serial " + serial.serial_ + " has not been found.");
    }

    init_online_interfaces(config);
    init_common_interfaces(serial.serial_, config);
}

Camera::Private::Private(const std::string &rawfile, const Future::RawFileConfig &file_stream_config,
                         bool realtime_playback_speed) {
    open_raw_file(rawfile, file_stream_config, realtime_playback_speed);
}

Camera::Private::~Private() {
    if (is_init_) {
        stop();
    }
}

void Camera::Private::open_raw_file(const std::string &rawfile, const Future::RawFileConfig &file_stream_config,
                                    bool realtime_playback_speed) {
    if (is_init_ && run_thread_.joinable()) {
        stop();
    }

    is_init_              = true;
    detail::Config config = detail::Config();

    if (!boost::filesystem::exists(rawfile)) {
        throw CameraException(CameraErrorCode::FileDoesNotExist,
                              "Opening RAW file at " + rawfile + ": not an existing file.");
    }

    if (!boost::filesystem::is_regular_file(rawfile)) {
        throw CameraException(CameraErrorCode::NotARegularFile);
    }

    if (boost::filesystem::extension(rawfile) != ".raw") {
        throw CameraException(CameraErrorCode::WrongExtension,
                              "Expected .raw as extension for the provided input file " + rawfile + ".");
    }

    raw_file_stream_config_.n_events_to_read_ = file_stream_config.n_events_to_read_;
    raw_file_stream_config_.n_read_buffers_   = file_stream_config.n_read_buffers_;
    raw_file_stream_config_.do_time_shifting_ = file_stream_config.do_time_shifting_;
    raw_file_stream_config_.build_index_      = file_stream_config.build_index_;
    device_                                   = DeviceDiscovery::open_raw_file(rawfile, raw_file_stream_config_);
    if (!device_) {
        // We should never get here as open_raw_file should throw an exception if the system is unknown
        throw CameraException(CameraErrorCode::InvalidRawfile,
                              "The RAW file at " + rawfile +
                                  " could not be read. Please check that the file has "
                                  "been recorded with an event-based device or contact "
                                  "the support.");
    }

    from_file_         = true;
    emulate_real_time_ = realtime_playback_speed;

    auto *board_id = device_->get_facility<I_HW_Identification>();
    if (!board_id) {
        throw(CameraException(InternalInitializationErrors::IBoardIdentificationNotFound));
    }

    init_common_interfaces(board_id->get_serial(), config);
}

// -- Camera interface mirror functions
CallbackId Camera::Private::add_runtime_error_callback(RuntimeErrorCallback error_callback) {
    check_camera_device_instance();
    CallbackId save_id = index_manager_.index_generator_.get_next_index();
    {
        std::unique_lock<std::mutex> lock(cbs_mutex_);
        runtime_error_callback_map_[save_id] = error_callback;
    }
    return save_id;
}
bool Camera::Private::remove_runtime_error_callback(CallbackId callback_id) {
    std::unique_lock<std::mutex> lock(cbs_mutex_);

    auto error_cb_it = runtime_error_callback_map_.find(callback_id);
    if (error_cb_it != runtime_error_callback_map_.end()) {
        runtime_error_callback_map_.erase(error_cb_it);
        return true;
    }

    return false;
}

CallbackId Camera::Private::add_status_change_callback(StatusChangeCallback status_change_callback) {
    check_camera_device_instance();
    CallbackId save_id = index_manager_.index_generator_.get_next_index();
    {
        std::unique_lock<std::mutex> lock(cbs_mutex_);
        status_change_callback_map_[save_id] = status_change_callback;
    }
    return save_id;
}

bool Camera::Private::remove_status_change_callback(CallbackId callback_id) {
    std::unique_lock<std::mutex> lock(cbs_mutex_);

    auto error_cb_it = status_change_callback_map_.find(callback_id);
    if (error_cb_it != status_change_callback_map_.end()) {
        status_change_callback_map_.erase(error_cb_it);
        return true;
    }

    return false;
}

std::shared_ptr<Camera::Private::EventsStreamUpdateCallback>
    Camera::Private::add_events_stream_update_callback(const std::function<bool(void)> &f) {
    auto cb_ptr = std::make_shared<EventsStreamUpdateCallback>(f);

    std::unique_lock<std::mutex> lock(cbs_mutex_);
    events_stream_update_callbacks_.push_back(cb_ptr);
    return cb_ptr;
}

bool Camera::Private::remove_events_stream_update_callback(const std::shared_ptr<EventsStreamUpdateCallback> &cb) {
    std::unique_lock<std::mutex> lock(cbs_mutex_);

    auto error_cb_it = std::find(events_stream_update_callbacks_.begin(), events_stream_update_callbacks_.end(), cb);
    if (error_cb_it != events_stream_update_callbacks_.end()) {
        events_stream_update_callbacks_.erase(error_cb_it);
        return true;
    }

    return false;
}

bool Camera::Private::call_events_stream_update_callbacks() {
    bool ret = false;
    std::vector<std::shared_ptr<EventsStreamUpdateCallback>> cbs;
    {
        std::unique_lock<std::mutex> lock(cbs_mutex_);
        std::swap(cbs, events_stream_update_callbacks_);
    }
    for (auto &cb : cbs) {
        ret |= cb->call();
    }
    return ret;
}

void Camera::Private::cancel_events_stream_update_callbacks() {
    std::vector<std::shared_ptr<EventsStreamUpdateCallback>> cbs;
    {
        std::unique_lock<std::mutex> lock(cbs_mutex_);
        std::swap(cbs, events_stream_update_callbacks_);
    }
    for (auto &cb : cbs) {
        cb->cancel();
    }
}

bool Camera::Private::start() {
    check_initialization();

    {
        std::unique_lock<std::mutex> lock(run_thread_mutex_);
        if (run_thread_.joinable()) { // Already started
            return false;
        }

        camera_is_started_ = false;
        run_thread_        = std::thread([this] {
            if (print_timings_) {
                run(timing_profiler_tuple_.get_profiler<true>());
            } else {
                run(timing_profiler_tuple_.get_profiler<false>());
            }
        });

        // Be sure the thread has been launched to set is_running to true
        // Thus, checking 'is_running()' right after start is expected to return true
        // unless cases where the thread ends after one iteration (end of file already reached, camera unplugged ...)
        while (!run_thread_.joinable()) {}

        set_is_running(true);
        run_thread_status_ = RunThreadStatus::STARTED;
    }

    // notifies the thread that it can start running
    run_thread_cond_.notify_one();

    while (!camera_is_started_) {}

    return true;
}

bool Camera::Private::stop() {
    check_initialization();

    std::unique_lock<std::mutex> lock(run_thread_mutex_);
    if (!run_thread_.joinable()) {
        return false;
    }

    // makes sure that the thread is running before trying to stop it
    run_thread_cond_.wait(lock, [this]() { return run_thread_status_ == RunThreadStatus::RUNNING; });
    run_thread_status_ = RunThreadStatus::STOPPED;

    set_is_running(false);

    if (i_future_events_stream_) {
        i_future_events_stream_->stop();
    } else {
        i_events_stream_->stop();
    }
    if (i_device_control_) {
        i_device_control_->stop();
    }

    try {
        run_thread_.join();
    } catch (CameraException &e) {}

    // stop recording if needed
    // doing now, after we have stopped the decoding thread and the event stream
    // ensures that we will have logged every events that were available up until
    // we stopped the camera
    stop_recording();

    return true;
}

void Camera::Private::start_recording(const std::string &rawfile_path) {
    check_camera_device_instance();
    check_events_stream_instance();

    stop_recording();
    std::string base_path = boost::filesystem::change_extension(rawfile_path, "").string();

    // Log biases
    if (biases_) {
        biases_->save_to_file(base_path + ".bias");
    }

    if (i_future_events_stream_) {
        if (!i_future_events_stream_->log_raw_data(base_path + ".raw")) {
            throw CameraException(
                CameraErrorCode::CouldNotOpenFile,
                "Could not open file '" + base_path +
                    ".raw' to record. Make sure it is a valid filename and that you have permissions to write it.");
        } else {
            is_recording_ = true;
        }
    } else {
        if (!i_events_stream_->log_raw_data(base_path + ".raw")) {
            throw CameraException(
                CameraErrorCode::CouldNotOpenFile,
                "Could not open file '" + base_path +
                    ".raw' to record. Make sure it is a valid filename and that you have permissions to write it.");
        } else {
            is_recording_ = true;
        }
    }
}

void Camera::Private::stop_recording() {
    check_events_stream_instance();
    if (i_future_events_stream_) {
        i_future_events_stream_->stop_log_raw_data();
    } else {
        i_events_stream_->stop_log_raw_data();
    }

    is_recording_ = false;
}

Biases &Camera::Private::biases() {
    if (from_file_) {
        throw CameraException(UnsupportedFeatureErrors::BiasesUnavailable, "Cannot get biases from a file.");
    }

    check_biases_instance();

    return *biases_;
}

OfflineStreamingControl &Camera::Private::offline_streaming_control() {
    check_camera_device_instance();

    if (!from_file_) {
        throw CameraException(UnsupportedFeatureErrors::OfflineStreamingControlUnavailable,
                              "Cannot get offline streaming control from a live camera.");
    }

    if (!osc_->get_pimpl().is_valid()) {
        throw CameraException(UnsupportedFeatureErrors::OfflineStreamingControlUnavailable,
                              "Offline streaming control unavailable.");
    }

    return *osc_;
}

Roi &Camera::Private::roi() {
    check_camera_device_instance();
    if (from_file_) {
        throw CameraException(UnsupportedFeatureErrors::RoiUnavailable,
                              "Cannot get roi instance when running from a file.");
    }

    if (!roi_) {
        throw CameraException(InternalInitializationErrors::IRoiNotFound);
    }
    return *roi_;
}

const Geometry &Camera::Private::geometry() const {
    check_camera_device_instance();
    if (!geometry_) {
        throw CameraException(InternalInitializationErrors::IGeometryNotFound);
    };
    return *geometry_;
}

const CameraGeneration &Camera::Private::generation() const {
    check_camera_device_instance();
    return *generation_;
}

RawData &Camera::Private::raw_data() {
    check_camera_device_instance();
    return *raw_data_;
}

CD &Camera::Private::cd() {
    check_camera_device_instance();
    return *cd_;
}

ExtTrigger &Camera::Private::ext_trigger() {
    check_camera_device_instance();
    if (!ext_trigger_) {
        throw CameraException(UnsupportedFeatureErrors::ExtTriggerUnavailable);
    }
    return *ext_trigger_;
}

AntiFlickerModule &Camera::Private::antiflicker_module() {
    check_camera_device_instance();
    if (from_file_) {
        throw CameraException(UnsupportedFeatureErrors::AntiFlickerModuleUnavailable,
                              "Cannot get anti-flicker instance when running from a file.");
    }

    if (!afk_) {
        throw CameraException(UnsupportedFeatureErrors::AntiFlickerModuleUnavailable);
    }
    return *afk_;
}

ErcModule &Camera::Private::erc_module() {
    check_camera_device_instance();
    if (from_file_) {
        throw CameraException(UnsupportedFeatureErrors::ErcModuleUnavailable,
                              "Cannot get erc instance when running from a file.");
    }

    if (!ercm_) {
        throw CameraException(UnsupportedFeatureErrors::ErcModuleUnavailable);
    }
    return *ercm_;
}

NoiseFilterModule &Camera::Private::noise_filter_module() {
    check_camera_device_instance();
    if (from_file_) {
        throw CameraException(UnsupportedFeatureErrors::NoiseFilterModuleUnavailable,
                              "Cannot get noise filter instance when running from a file.");
    }

    if (!noise_filter_) {
        throw CameraException(UnsupportedFeatureErrors::NoiseFilterModuleUnavailable);
    }
    return *noise_filter_;
}

TriggerOut &Camera::Private::trigger_out() {
    check_camera_device_instance();
    if (from_file_) {
        throw CameraException(UnsupportedFeatureErrors::TriggerOutUnavailable,
                              "Cannot get trigger out instance when running from a file.");
    }
    if (!trigger_out_) {
        throw CameraException(UnsupportedFeatureErrors::TriggerOutUnavailable);
    }
    return *trigger_out_;
}

// -- Pimpl functions
void Camera::Private::init_online_interfaces(const detail::Config &config) {
    check_camera_device_instance();

    i_device_control_ = device_->get_facility<I_DeviceControl>();
    check_ccam_instance();

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

    I_Erc *i_ercm = device_->get_facility<I_Erc>();
    if (i_ercm) {
        ercm_.reset(new ErcModule(i_ercm));
    }

    I_NoiseFilterModule *i_noise_filter = device_->get_facility<I_NoiseFilterModule>();
    if (i_noise_filter) {
        noise_filter_.reset(new NoiseFilterModule(i_noise_filter));
    }
}

void Camera::Private::init_common_interfaces(const std::string &serial, const detail::Config &config) {
    i_future_events_stream_ = device_->get_facility<Future::I_EventsStream>();
    if (!i_future_events_stream_) {
        i_events_stream_ = device_->get_facility<I_EventsStream>();
    }
    check_events_stream_instance();

    I_Geometry *i_geometry = device_->get_facility<I_Geometry>();
    if (!i_geometry) {
        throw CameraException(InternalInitializationErrors::IGeometryNotFound);
    }
    geometry_.reset(new Geometry(i_geometry));

    i_future_decoder_ = device_->get_facility<Future::I_Decoder>();
    if (!i_future_decoder_) {
        i_decoder_ = device_->get_facility<I_Decoder>();
    }
    check_decoder_device_instance();

    raw_data_.reset(RawData::Private::build(index_manager_));

    cd_.reset(CD::Private::build(index_manager_));

    generation_.reset(CameraGeneration::Private::build(*device_));

    camera_configuration_.serial_number = serial;

    if (from_file_) {
        osc_.reset(new OfflineStreamingControl(*this));
    }

    if (config.print_timings) {
        print_timings_ = true;
    }

    init_callbacks();
}

void Camera::Private::init_callbacks() {
    // CD
    I_EventDecoder<EventCD> *i_cd_events_decoder = device_->get_facility<I_EventDecoder<EventCD>>();
    if (!i_cd_events_decoder) {
        throw CameraException(InternalInitializationErrors::ICDDecoderNotFound);
    }
    i_cd_events_decoder->add_event_buffer_callback([this](const EventCD *begin, const EventCD *end) {
        for (auto &&cb : cd_->get_pimpl().get_cbs()) {
            cb(begin, end);
        }
    });

    // External triggers
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
}

void Camera::Private::init_clocks() {
    if (i_future_decoder_) {
        first_ts_ = i_future_decoder_->get_last_timestamp();
    } else {
        first_ts_ = i_decoder_->get_last_timestamp();
    }
    first_ts_clock_ = 0;
}

template<typename TimingProfilerType>
void Camera::Private::run(TimingProfilerType *profiler) {
    {
        // makes sure that start() has finished and is_running_ is true
        std::unique_lock<std::mutex> lock(run_thread_mutex_);
        run_thread_cond_.wait(lock, [this]() { return run_thread_status_ == RunThreadStatus::STARTED; });
        run_thread_status_ = RunThreadStatus::RUNNING;
    }

    // notifies that this thread can now be stopped if needed
    run_thread_cond_.notify_one();

    check_camera_device_instance();
    check_events_stream_instance();
    check_decoder_device_instance();
    if (from_file_) {
        run_from_file(profiler);
    } else {
        run_from_camera(profiler);
    }
    set_is_running(false);

    // cancel any waiting event stream update callbacks
    // we do this after setting is_running_ to false, because these callbacks are only added when the main loop is
    // running, so those which have been added before, if they were not called, must be cancelled
    cancel_events_stream_update_callbacks();
}

template<typename TimingProfilerType>
int Camera::Private::run_main_loop(TimingProfilerType *profiler) {
    camera_is_started_ = true;

    int res             = 0;
    long int n_rawbytes = 0;

    init_clocks();

    while (is_running_) {
        call_events_stream_update_callbacks();

        {
            typename TimingProfilerType::TimedOperation t("Polling", profiler);
            if (i_future_events_stream_) {
                res = i_future_events_stream_->wait_next_buffer();
            } else {
                res = i_events_stream_->wait_next_buffer();
            }
        }

        if (res < 0) {
            break;
        } else if (res > 0) {
            typename TimingProfilerType::TimedOperation t("Processing", profiler);
            I_EventsStream::RawData *ev_buffer, *ev_buffer_end;
            if (i_future_events_stream_) {
                ev_buffer = i_future_events_stream_->get_latest_raw_data(n_rawbytes);
            } else {
                ev_buffer = i_events_stream_->get_latest_raw_data(n_rawbytes);
            }
            ev_buffer_end = ev_buffer + n_rawbytes;

            // Decode events chunk by chunk to allow early stop and better cadencing when emulating real time
            const bool has_decode_callbacks =
                index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID);
            constexpr uint32_t events_per_buffer_to_decode = 1024;
            const uint32_t bytes_step_to_decode = (i_future_decoder_ ? i_future_decoder_->get_raw_event_size_bytes() :
                                                                       i_decoder_->get_raw_event_size_bytes()) *
                                                  events_per_buffer_to_decode;
            long bytes_to_decode;
            for (; ev_buffer < ev_buffer_end; ev_buffer += bytes_to_decode) {
                const uint32_t remains = std::distance(ev_buffer, ev_buffer_end);
                bytes_to_decode        = std::min(remains, bytes_step_to_decode);

                // we first decode the buffer and call the corresponding events callback ...
                if (has_decode_callbacks) {
                    if (i_future_decoder_) {
                        i_future_decoder_->decode(ev_buffer, ev_buffer + bytes_to_decode);
                        t.setNumProcessedElements(bytes_to_decode / i_future_decoder_->get_raw_event_size_bytes());
                    } else {
                        i_decoder_->decode(ev_buffer, ev_buffer + bytes_to_decode);
                        t.setNumProcessedElements(bytes_to_decode / i_decoder_->get_raw_event_size_bytes());
                    }
                }

                // ... then we call the raw buffer callback so that a user has access to some info (e.g last
                // decoded timestamp) when the raw callback is called
                for (auto &cb : raw_data_->get_pimpl().get_cbs()) {
                    cb(ev_buffer, bytes_to_decode);
                }

                // call the callbacks that could modify the events stream, and early stop the decoding loop if the
                // stream has been modified. this can happen in some cases (e.g when we seek and the remaining data to
                // be decoded is garbage anyway)
                if (call_events_stream_update_callbacks()) {
                    break;
                }

                if (has_decode_callbacks) {
                    // emulate real time if needed
                    if (emulate_real_time_) {
                        const timestamp cur_ts      = i_future_decoder_ ? i_future_decoder_->get_last_timestamp() :
                                                                          i_decoder_->get_last_timestamp();
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
            }
        }
    }

    return res;
}

template<typename TimingProfilerType>
int Camera::Private::run_from_camera(TimingProfilerType *profiler) {
    check_ccam_instance();

    if (i_future_events_stream_) {
        // should never happen for the moment ...
    } else {
        i_events_stream_->start();
    }
    i_device_control_->start();
    i_device_control_->reset();

    if (!run_main_loop(profiler)) {
        return false;
    }

    // don't have to wait for the exposure frame thread to finish, if we got here, it means
    // stop() has been called explicitly

    return true;
}

template<typename TimingProfilerType>
int Camera::Private::run_from_file(TimingProfilerType *profiler) {
    if (i_future_events_stream_) {
        i_future_events_stream_->start();
    } else {
        i_events_stream_->start();
    }

    if (!run_main_loop(profiler)) {
        return false;
    }

    return true;
}

void Camera::Private::set_is_running(bool is_running) {
    if (is_running_ != is_running) {
        is_running_ = is_running;
        std::map<CallbackId, StatusChangeCallback> callbacks_to_call;
        {
            std::unique_lock<std::mutex> lock(cbs_mutex_);
            callbacks_to_call = status_change_callback_map_;
        }
        for (auto &callback : callbacks_to_call) {
            callback.second(is_running ? CameraStatus::STARTED : CameraStatus::STOPPED);
        }
    }
}

void Camera::Private::check_initialization() const {
    if (!is_init_) {
        throw CameraException(CameraErrorCode::CameraNotInitialized);
    }
}

void Camera::Private::check_biases_instance() const {
    check_initialization();
    if (!biases_) {
        throw CameraException(InternalInitializationErrors::ILLBiasesNotFound);
    }
}

void Camera::Private::check_events_stream_instance() const {
    check_initialization();
    if (!i_future_events_stream_ && !i_events_stream_) {
        throw CameraException(InternalInitializationErrors::IEventsStreamNotFound);
    }
}

void Camera::Private::check_ccam_instance() const {
    check_initialization();
    if (!i_device_control_) {
        throw CameraException(InternalInitializationErrors::IDeviceControlNotFound);
    }
}

void Camera::Private::check_camera_device_instance() const {
    check_initialization();
    if (!device_) {
        throw CameraException(CameraErrorCode::CameraNotFound);
    }
}

void Camera::Private::check_decoder_device_instance() const {
    check_initialization();
    if (!i_future_decoder_ && !i_decoder_) {
        throw CameraException(InternalInitializationErrors::IDecoderNotFound);
    }
}

// ********************
// CAMERA API

AvailableSourcesList Camera::list_online_sources() {
    AvailableSourcesList ret;

    // Get Connected (mipi, usb) available sources
    DeviceDiscovery::SystemList available_systems = DeviceDiscovery::list_available_sources_local();

    // Get only remote sources
    DeviceDiscovery::SystemList available_remote_systems = DeviceDiscovery::list_available_sources_remote();

    // First, scan remote sources :
    for (auto system : available_remote_systems) {
        if (ret.find(OnlineSourceType::REMOTE) == ret.end()) {
            ret[OnlineSourceType::REMOTE] = std::vector<std::string>();
        }
        ret[OnlineSourceType::REMOTE].push_back(system.get_full_serial());
    }

    for (auto system : available_systems) {
        switch (system.connection_) {
        case ConnectionType::MIPI_LINK: {
            if (ret.find(OnlineSourceType::EMBEDDED) == ret.end()) {
                ret[OnlineSourceType::EMBEDDED] = std::vector<std::string>();
            }
            ret[OnlineSourceType::EMBEDDED].push_back(system.get_full_serial());
            break;
        }
        case ConnectionType::USB_LINK: {
            if (ret.find(OnlineSourceType::USB) == ret.end()) {
                ret[OnlineSourceType::USB] = std::vector<std::string>();
            }
            ret[OnlineSourceType::USB].push_back(system.get_full_serial());
            break;
        }
        default:
            break;
        }
    }

    // sort to ensure the indexes are always the same in the map
    for (auto systems : ret) {
        std::sort(systems.second.begin(), systems.second.end());
    }

    return ret;
}

Camera::Camera() : pimpl_(new Private(true)) {}

Camera::Camera(Camera &&camera) = default;

Camera &Camera::operator=(Camera &&camera) = default;

Camera::~Camera() {}

Camera::Camera(Private *pimpl) : pimpl_(pimpl) {}

Camera Camera::from_first_available() {
    return Camera(new Private(false));
}

Camera Camera::from_source(OnlineSourceType input_source_type, uint32_t source_index) {
    return Camera(new Private(input_source_type, source_index));
}

Camera Camera::from_serial(const std::string &serial) {
    return Camera(new Private(Private::Serial(serial)));
}

// TODO MV-166: remove this overload
Camera Camera::from_file(const std::string &rawfile, bool realtime_playback_speed, const RawFileConfig &file_config) {
    Future::RawFileConfig raw_file_config;
    raw_file_config.do_time_shifting_ = file_config.do_time_shifting_;
    raw_file_config.n_events_to_read_ = file_config.n_events_to_read_;
    raw_file_config.n_read_buffers_   = file_config.n_read_buffers_;
    // to keep the same behavior as before, do not build index by default
    raw_file_config.build_index_ = false;
    return Camera(new Private(rawfile, raw_file_config, realtime_playback_speed));
}

// TODO MV-166: mention that RAW index files will automatically be constructed starting from next major release
// unless RawFileConfig::build_index_ is set to false
Camera Camera::from_file(const std::string &rawfile, bool realtime_playback_speed,
                         const Future::RawFileConfig &file_config) {
    return Camera(new Private(rawfile, file_config, realtime_playback_speed));
}

RawData &Camera::raw_data() {
    return pimpl_->raw_data();
}

CD &Camera::cd() {
    return pimpl_->cd();
}

ExtTrigger &Camera::ext_trigger() {
    return pimpl_->ext_trigger();
}

AntiFlickerModule &Camera::antiflicker_module() {
    return pimpl_->antiflicker_module();
}

ErcModule &Camera::erc_module() {
    return pimpl_->erc_module();
}

NoiseFilterModule &Camera::noise_filter_module() {
    return pimpl_->noise_filter_module();
}

TriggerOut &Camera::trigger_out() {
    return pimpl_->trigger_out();
}

Roi &Camera::roi() {
    return pimpl_->roi();
}

CallbackId Camera::add_runtime_error_callback(RuntimeErrorCallback error_callback) {
    return pimpl_->add_runtime_error_callback(error_callback);
}

bool Camera::remove_runtime_error_callback(CallbackId callback_id) {
    return pimpl_->remove_runtime_error_callback(callback_id);
}

CallbackId Camera::add_status_change_callback(StatusChangeCallback status_change_callback) {
    return pimpl_->add_status_change_callback(status_change_callback);
}

bool Camera::remove_status_change_callback(CallbackId callback_id) {
    return pimpl_->remove_status_change_callback(callback_id);
}

Biases &Camera::biases() {
    return pimpl_->biases();
}

OfflineStreamingControl &Camera::offline_streaming_control() {
    return pimpl_->offline_streaming_control();
}

const Geometry &Camera::geometry() const {
    return pimpl_->geometry();
}

const CameraGeneration &Camera::generation() const {
    return pimpl_->generation();
}

bool Camera::start() {
    return pimpl_->start();
}

bool Camera::is_running() {
    return pimpl_->is_running_;
}

bool Camera::stop() {
    return pimpl_->stop();
}

void Camera::start_recording(const std::string &rawfile_path) {
    pimpl_->start_recording(rawfile_path);
}

void Camera::stop_recording() {
    pimpl_->stop_recording();
}

const CameraConfiguration &Camera::get_camera_configuration() {
    return pimpl_->camera_configuration_;
}

Metavision::timestamp Camera::get_last_timestamp() const {
    const bool decode_cbs_registered =
        pimpl_->index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID);
    if (!decode_cbs_registered)
        return -1;
    return pimpl_->i_future_decoder_ ? pimpl_->i_future_decoder_->get_last_timestamp() :
                                       pimpl_->i_decoder_->get_last_timestamp();
}

Device &Camera::get_device() {
    return *pimpl_->device_;
}

Camera::Private &Camera::get_pimpl() {
    return *pimpl_;
}

} // namespace Metavision
