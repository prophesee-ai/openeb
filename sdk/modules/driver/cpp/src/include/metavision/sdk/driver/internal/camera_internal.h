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

#ifndef METAVISION_SDK_DRIVER_CAMERA_INTERNAL_H
#define METAVISION_SDK_DRIVER_CAMERA_INTERNAL_H

#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <map>

#include "metavision/hal/facilities/i_device_control.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_decoder.h"
#include "metavision/hal/utils/raw_file_config.h"
#include "metavision/sdk/driver/camera.h"
#include "metavision/sdk/core/utils/index_manager.h"
#include "metavision/sdk/core/utils/timing_profiler.h"

namespace Metavision {

class Device;
class CD;
class ExtTrigger;
class TriggerOut;
class RawData;
class Biases;

namespace detail {
struct Config {
    bool print_timings{false};
};

} // namespace detail

class Camera::Private {
public:
    struct Serial {
        Serial(const std::string &serial) : serial_(serial){};
        std::string serial_ = "";
    };

    Private() = delete;
    Private(bool empty_init);
    Private(OnlineSourceType input_source_type, uint32_t source_index);
    Private(const Serial &serial);
    Private(const std::string &rawfile, const RawFileConfig &file_stream_config, bool reproduce_camera_behavior);

    ~Private();

    void open_raw_file(const std::string &rawfile, const RawFileConfig &file_stream_config,
                       bool reproduce_camera_behavior);

    RawFileConfig raw_file_stream_config_;

    // Camera interface mirror functions
    CallbackId add_runtime_error_callback(RuntimeErrorCallback error_callback);
    bool remove_runtime_error_callback(CallbackId callback_id);

    CallbackId add_status_change_callback(StatusChangeCallback status_change_callback);
    bool remove_status_change_callback(CallbackId callback_id);

    bool start();
    bool stop();

    // Get Event handlers classes :
    CD &cd();
    ExtTrigger &ext_trigger();
    TriggerOut &trigger_out();
    RawData &raw_data();

    // Get biases handler class :
    Biases &biases();

    // Get the roi handler class
    Roi &roi();

    // Get the sensor antiflicker handler class
    AntiFlickerModule &antiflicker_module();

    // Get the noise filter handler class for configuring hardware side module
    NoiseFilterModule &noise_filter_module();

    // Get the geometry
    const Geometry &geometry() const;

    // Get the generation
    const CameraGeneration &generation() const;

    void start_recording(const std::string &rawfile_path);
    void stop_recording();

    // Pimpl functions
    void init_online_interfaces(const detail::Config &cfg = detail::Config());
    void init_common_interfaces(const std::string &serial = std::string(),
                                const detail::Config &cfg = detail::Config());

    void init_callbacks(); // initialize decoded events cbs

    template<typename TimingProfilerType>
    void run(TimingProfilerType *profiler);
    template<typename TimingProfilerType>
    int run_from_camera(TimingProfilerType *profiler);
    template<typename TimingProfilerType>
    int run_from_file(TimingProfilerType *profiler);
    template<typename TimingProfilerType>
    int run_main_loop(TimingProfilerType *profiler);
    void emulate_real_time(I_EventsStream::RawData *ev_buffer, long n_rawbytes);
    void init_clocks();

    void set_up_from_config();
    void end_run(int run_output);
    void set_is_running(bool);

    // initialization check up
    void check_initialization() const;
    void check_biases_instance() const;
    void check_events_stream_instance() const;
    void check_ccam_instance() const;
    void check_camera_device_instance() const;
    void check_decoder_device_instance() const;

    CameraConfiguration camera_configuration_;
    bool emulate_real_time_ = false;
    timestamp first_ts_;
    uint64_t first_ts_clock_;
    bool print_timings_ = false;
    TimingProfilerPair<> timing_profiler_tuple_;

    std::unique_ptr<Device> device_    = nullptr;
    I_DeviceControl *i_device_control_ = nullptr;
    I_EventsStream *i_events_stream_   = nullptr;
    I_Decoder *i_decoder_              = nullptr;

    bool from_file_    = false;
    bool is_init_      = false;
    bool is_recording_ = false;
    std::atomic<bool> is_running_{false}, done_decoding_{true};

    std::thread run_thread_;
    std::mutex run_thread_mutex_, cbs_mutex_;
    enum class RunThreadStatus { STARTED, RUNNING, STOPPED };
    RunThreadStatus run_thread_status_ = RunThreadStatus::STOPPED;
    std::condition_variable run_thread_cond_;
    std::atomic<bool> camera_is_started_;

    // Facilities' wrappers :
    std::unique_ptr<Geometry> geometry_;
    std::unique_ptr<Roi> roi_;
    std::unique_ptr<TriggerOut> trigger_out_;
    std::unique_ptr<Biases> biases_;
    std::unique_ptr<AntiFlickerModule> afk_;
    std::unique_ptr<NoiseFilterModule> noise_filter_;

    // Classes that handle events cbs :
    IndexManager index_manager_;
    std::unique_ptr<CD> cd_;
    std::unique_ptr<ExtTrigger> ext_trigger_;

    std::unique_ptr<RawData> raw_data_;
    std::unique_ptr<CameraGeneration> generation_;

    std::vector<Event2d> cd_events_front_, cd_events_back_;
    std::mutex cd_events_mutex_;
    CallbackId cd_events_cb_id_;

    std::map<CallbackId, RuntimeErrorCallback> runtime_error_callback_map_;
    std::map<CallbackId, StatusChangeCallback> status_change_callback_map_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_CAMERA_INTERNAL_H
