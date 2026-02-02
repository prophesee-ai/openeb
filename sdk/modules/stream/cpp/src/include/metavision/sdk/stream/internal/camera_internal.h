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

#ifndef METAVISION_SDK_STREAM_CAMERA_INTERNAL_H
#define METAVISION_SDK_STREAM_CAMERA_INTERNAL_H

#include <filesystem>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <map>
#include <unordered_map>

#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/core/utils/index_manager.h"
#include "metavision/sdk/core/utils/timing_profiler.h"

namespace Metavision {

namespace detail {
struct Config {
    bool print_timings{false};
};

} // namespace detail

class I_Geometry;

class Camera::Private {
public:
    Private();
    Private(const detail::Config &config);
    virtual ~Private();

    CallbackId add_runtime_error_callback(RuntimeErrorCallback error_callback);
    bool remove_runtime_error_callback(CallbackId callback_id);

    CallbackId add_status_change_callback(StatusChangeCallback status_change_callback);
    bool remove_status_change_callback(CallbackId callback_id);

    bool start();
    bool stop();

    bool start_recording(const std::filesystem::path &file_path);
    bool stop_recording(const std::filesystem::path &file_path = std::filesystem::path());

    CD &cd();
    ExtTrigger &ext_trigger();
    RawData &raw_data();
    ERCCounter &erc_counter();
    FrameHisto &frame_histo();
    FrameDiff &frame_diff();
    Monitoring &monitoring();

    const CameraGeneration &generation() const;

    virtual Device &device();
    virtual OfflineStreamingControl &offline_streaming_control();

    virtual timestamp get_last_timestamp() const;
    virtual void start_impl();
    virtual void stop_impl();
    virtual bool process_impl();
    virtual bool start_recording_impl(const std::filesystem::path &file_path);
    virtual bool stop_recording_impl(const std::filesystem::path &file_path);
    virtual I_Geometry &get_geometry();

    virtual void save(std::ostream &os) const;
    virtual void load(std::istream &is);

    void run();
    void set_is_running(bool);

    void check_initialization() const;

    void propagate_runtime_error(const CameraException &e);

    Camera *pub_ptr_ = nullptr;

    detail::Config config_;
    CameraConfiguration camera_configuration_;
    std::unordered_map<std::string, std::string> metadata_map_;
    TimingProfilerPair<> timing_profiler_tuple_;

    bool is_init_ = false;
    std::atomic<bool> is_running_{false};

    std::thread run_thread_;
    std::mutex run_thread_mutex_, cbs_mutex_;
    enum class RunThreadStatus { STARTED, RUNNING, STOPPED };
    RunThreadStatus run_thread_status_ = RunThreadStatus::STOPPED;
    std::condition_variable run_thread_cond_;
    std::atomic<bool> camera_is_started_;

    IndexManager index_manager_;
    std::unique_ptr<CD> cd_;
    std::unique_ptr<ExtTrigger> ext_trigger_;
    std::unique_ptr<ERCCounter> erc_counter_;
    std::unique_ptr<FrameHisto> frame_histo_;
    std::unique_ptr<FrameDiff> frame_diff_;
    std::unique_ptr<Monitoring> monitoring_;

    std::unique_ptr<RawData> raw_data_;
    std::unique_ptr<CameraGeneration> generation_;

    std::unordered_multimap<std::string, CallbackId> recording_cb_ids_;

    std::map<CallbackId, RuntimeErrorCallback> runtime_error_callback_map_;
    std::map<CallbackId, StatusChangeCallback> status_change_callback_map_;
};

} // namespace Metavision

#endif // METAVISION_SDK_STREAM_CAMERA_INTERNAL_H
