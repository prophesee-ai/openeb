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

#include <future>

#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/utils/hal_connection_exception.h"
#include "metavision/sdk/stream/internal/camera_internal.h"
#include "metavision/sdk/stream/internal/camera_live_internal.h"
#include "metavision/sdk/stream/internal/camera_offline_generic_internal.h"
#include "metavision/sdk/stream/internal/camera_offline_raw_internal.h"
#include "metavision/sdk/stream/internal/camera_error_code_internal.h"
#include "metavision/sdk/stream/camera_exception.h"
#include "metavision/sdk/stream/raw_event_file_logger.h"
#include "metavision/sdk/stream/hdf5_event_file_writer.h"

namespace Metavision {

// ********************
// PIMPL
Camera::Private::Private() {}

Camera::Private::Private(const detail::Config &config) : config_(config) {}

Camera::Private::~Private() {}

// -- Camera interface mirror functions
CallbackId Camera::Private::add_runtime_error_callback(RuntimeErrorCallback error_callback) {
    check_initialization();

    CallbackId save_id = index_manager_.index_generator_.get_next_index();
    {
        std::unique_lock<std::mutex> lock(cbs_mutex_);
        runtime_error_callback_map_[save_id] = error_callback;
    }
    return save_id;
}

bool Camera::Private::remove_runtime_error_callback(CallbackId callback_id) {
    check_initialization();

    std::unique_lock<std::mutex> lock(cbs_mutex_);
    auto error_cb_it = runtime_error_callback_map_.find(callback_id);
    if (error_cb_it != runtime_error_callback_map_.end()) {
        runtime_error_callback_map_.erase(error_cb_it);
        return true;
    }

    return false;
}

CallbackId Camera::Private::add_status_change_callback(StatusChangeCallback status_change_callback) {
    check_initialization();

    CallbackId save_id = index_manager_.index_generator_.get_next_index();
    {
        std::unique_lock<std::mutex> lock(cbs_mutex_);
        status_change_callback_map_[save_id] = status_change_callback;
    }
    return save_id;
}

bool Camera::Private::remove_status_change_callback(CallbackId callback_id) {
    check_initialization();

    std::unique_lock<std::mutex> lock(cbs_mutex_);
    auto error_cb_it = status_change_callback_map_.find(callback_id);
    if (error_cb_it != status_change_callback_map_.end()) {
        status_change_callback_map_.erase(error_cb_it);
        return true;
    }

    return false;
}

bool Camera::Private::start() {
    check_initialization();

    {
        std::unique_lock<std::mutex> lock(run_thread_mutex_);
        if (run_thread_.joinable()) { // Already started
            return false;
        }

        camera_is_started_ = false;
        std::promise<bool> thread_started;
        std::future<bool> has_started = thread_started.get_future();

        run_thread_ = std::thread([&] {
            thread_started.set_value(true);
            run();
        });

        // Be sure the thread has been launched to set is_running to true
        // Thus, checking 'is_running()' right after start is expected to return true
        // unless cases where the thread ends after one iteration (end of file already reached, camera unplugged ...)
        has_started.wait();

        set_is_running(true);
        run_thread_status_ = RunThreadStatus::STARTED;
    }

    // notifies the thread that it can start running
    run_thread_cond_.notify_one();

    while (!camera_is_started_ && is_running_) {}

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

    try {
        stop_impl();
    } catch (const HalConnectionException &) {
        // The implementation (probably in a plugin) reported an error. It is unknown whether the run thread will
        // terminate properly. If we join on it, we may wait forever, but if we don't, the thread remains joinable
        // and a future call to stop() may wait for the thread to be running while it is already stopped.
        // Detaching the thread makes it non-joinable, preserving Metavision execution, while preserving the
        // opportunity for the plugin to cancel its operations.
        MV_HAL_LOG_WARNING() << "Camera implementation did not stop properly, detaching run thread";
        run_thread_.detach();
        // We can't ensure that every event was logged
        stop_recording();
        throw;
    }

    try {
        run_thread_.join();
    } catch (CameraException &) {}

    // stop recording if needed
    // doing now, after we have stopped the decoding thread and the event stream
    // ensures that we will have logged every events that were available up until
    // we stopped the camera
    stop_recording();

    return true;
}

bool Camera::Private::start_recording(const std::filesystem::path &file_path) {
    check_initialization();

    stop_recording(file_path);

    // clang-format off
    bool ret = false;
    try {
        ret = start_recording_impl(file_path);
    } catch (const CameraException &) {
         throw;
    } catch (...) {
        throw CameraException(CameraErrorCode::CouldNotOpenFile,
                              "Could not open file '" + file_path.string() +
                                  "' to record. Make sure it is a valid filename and that you have "
                                  "permissions to write it.");
    }
    // clang-format on

    return ret;
}

bool Camera::Private::stop_recording(const std::filesystem::path &file_path) {
    check_initialization();

    bool ret = true;
    if (file_path.empty()) {
        for (const auto &p : recording_cb_ids_) {
            ret = ret && stop_recording_impl(p.first);
        }
    } else {
        ret = stop_recording_impl(file_path);
    }
    return ret;
}

I_Geometry &Camera::Private::get_geometry() {
    try {
        auto geom = device().get_facility<I_Geometry>();
        if (geom) {
            return *geom;
        }
    } catch (const CameraException &e) {
        if (e.code().value() != UnsupportedFeatureErrors::DeviceUnavailable) {
            throw;
        }
    }
    // Shouldn't reach here
    throw CameraException(CameraErrorCode::UnsupportedFeature,
                          std::string("Unavailable facility ") + typeid(I_Geometry).name());
}

const CameraGeneration &Camera::Private::generation() const {
    check_initialization();
    return *generation_;
}

RawData &Camera::Private::raw_data() {
    check_initialization();
    if (!raw_data_) {
        throw CameraException(UnsupportedFeatureErrors::RawDataUnavailable);
    }
    return *raw_data_;
}

CD &Camera::Private::cd() {
    check_initialization();
    if (!cd_) {
        throw CameraException(UnsupportedFeatureErrors::CDUnavailable);
    }
    return *cd_;
}

ExtTrigger &Camera::Private::ext_trigger() {
    check_initialization();
    if (!ext_trigger_) {
        throw CameraException(UnsupportedFeatureErrors::ExtTriggerUnavailable);
    }
    return *ext_trigger_;
}

Device &Camera::Private::device() {
    throw CameraException(CameraErrorCode::CameraNotInitialized);
}

ERCCounter &Camera::Private::erc_counter() {
    check_initialization();
    if (!erc_counter_) {
        throw CameraException(UnsupportedFeatureErrors::ERCCounterUnavailable);
    }
    return *erc_counter_;
}

FrameHisto &Camera::Private::frame_histo() {
    check_initialization();
    if (!frame_histo_) {
        throw CameraException(UnsupportedFeatureErrors::FrameHistoUnavailable);
    }
    return *frame_histo_;
}

FrameDiff &Camera::Private::frame_diff() {
    check_initialization();
    if (!frame_diff_) {
        throw CameraException(UnsupportedFeatureErrors::FrameDiffUnavailable);
    }
    return *frame_diff_;
}

Monitoring &Camera::Private::monitoring() {
    check_initialization();
    if (!monitoring_) {
        throw CameraException(UnsupportedFeatureErrors::MonitoringUnavailable);
    }
    return *monitoring_;
}

OfflineStreamingControl &Camera::Private::offline_streaming_control() {
    throw CameraException(CameraErrorCode::CameraNotInitialized);
}

timestamp Camera::Private::get_last_timestamp() const {
    throw CameraException(CameraErrorCode::CameraNotInitialized);
}

void Camera::Private::start_impl() {
    throw CameraException(CameraErrorCode::CameraNotInitialized);
}

void Camera::Private::stop_impl() {
    throw CameraException(CameraErrorCode::CameraNotInitialized);
}

bool Camera::Private::process_impl() {
    throw CameraException(CameraErrorCode::CameraNotInitialized);
}

bool Camera::Private::start_recording_impl(const std::filesystem::path &file_path) {
    std::string ext = file_path.extension().string();
    std::shared_ptr<Metavision::EventFileWriter> writer;
    if (ext == ".raw") {
        writer = std::make_shared<Metavision::RAWEventFileLogger>(file_path);
    } else if (ext == ".hdf5") {
        writer = std::make_shared<Metavision::HDF5EventFileWriter>(file_path);
    } else {
        throw CameraException(CameraErrorCode::WrongExtension,
                              "Unsupported extension for the recording destination " + file_path.string() + ".");
    }
    writer->add_metadata_map_from_camera(*pub_ptr_);
    if (ext == ".raw") {
        if (!raw_data_) {
            throw CameraException(UnsupportedFeatureErrors::RawRecordingUnavailable,
                                  "Cannot record to a RAW file from this type of camera.");
        }
        recording_cb_ids_.emplace(file_path.string(),
                                  raw_data_->add_callback([writer](const std::uint8_t *ptr, size_t size) {
                                      auto *raw_writer = static_cast<Metavision::RAWEventFileLogger *>(writer.get());
                                      raw_writer->add_raw_data(ptr, size);
                                  }));
    } else {
        auto timeshift_added = std::make_shared<bool>(false);
        recording_cb_ids_.emplace(
            file_path.string(), cd_->add_callback([timeshift_added, writer](const EventCD *begin, const EventCD *end) {
                if (!*timeshift_added) {
                    writer->add_metadata("time_shift", std::to_string(begin->t));
                    *timeshift_added = true;
                }
                writer->add_events(begin, end);
            }));
        if (ext_trigger_) {
            recording_cb_ids_.emplace(file_path.string(), ext_trigger_->add_callback([writer](auto begin, auto end) {
                writer->add_events(begin, end);
            }));
        }
        if (erc_counter_) {
            // TODO: add support for ERC counter in events writer in MV-905
            /*
            recording_cb_ids_.emplace(file_path,
                                      erc_counter_->add_callback([writer](auto begin, auto end) {
                                          writer->add_events(begin, end);
                                      }));
                                      */
        }
    }
    return true;
}

bool Camera::Private::stop_recording_impl(const std::filesystem::path &file_path) {
    auto range = recording_cb_ids_.equal_range(file_path.string());
    if (range.first != range.second) {
        for (auto it = range.first; it != range.second; ++it) {
            // only one of those calls will succeed, we don't care which one
            if (raw_data_) {
                raw_data_->remove_callback(it->second);
            }
            cd_->remove_callback(it->second);
            if (ext_trigger_) {
                ext_trigger_->remove_callback(it->second);
            }
            if (erc_counter_) {
                erc_counter_->remove_callback(it->second);
            }
            if (monitoring_) {
                monitoring_->remove_callback(it->second);
            }
        }
        return true;
    }
    return false;
}

void Camera::Private::save(std::ostream &os) const {
    throw CameraException(CameraErrorCode::CameraNotInitialized);
}

void Camera::Private::load(std::istream &is) {
    throw CameraException(CameraErrorCode::CameraNotInitialized);
}

void Camera::Private::propagate_runtime_error(const CameraException &e) {
    std::map<CallbackId, RuntimeErrorCallback> callbacks;
    {
        std::unique_lock<std::mutex> lock(cbs_mutex_);
        callbacks = runtime_error_callback_map_;
    }
    for (auto &&p : callbacks) {
        p.second(e);
    }
}

void Camera::Private::run() {
    check_initialization();

    {
        // makes sure that start() has finished and is_running_ is true
        std::unique_lock<std::mutex> lock(run_thread_mutex_);
        run_thread_cond_.wait(lock, [this]() { return run_thread_status_ == RunThreadStatus::STARTED; });
        run_thread_status_ = RunThreadStatus::RUNNING;
    }

    // notifies that this thread can now be stopped if needed
    run_thread_cond_.notify_one();

    try {
        start_impl();
    } catch (const HalConnectionException &e) {
        const CameraException camera_error =
            CameraException(CameraErrorCode::ConnectionError, std::string("Connection error: ") + e.what());
        propagate_runtime_error(camera_error);
        set_is_running(false);
        return;
    }
    camera_is_started_ = true;

    while (is_running_) {
        try {
            if (!process_impl()) {
                break;
            }
        } catch (const HalConnectionException &e) {
            const CameraException camera_error =
                CameraException(CameraErrorCode::ConnectionError, std::string("Connection error: ") + e.what());
            propagate_runtime_error(camera_error);
            break;
        } catch (const std::exception &e) {
            const CameraException camera_error =
                CameraException(CameraErrorCode::RuntimeError, std::string("Unexpected error : ") + e.what());
            propagate_runtime_error(camera_error);
            break;
        }
    }
    set_is_running(false);
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

Camera::Camera() : pimpl_(new Private()) {
    pimpl_->pub_ptr_ = this;
}

Camera::Camera(Camera &&camera) : pimpl_(std::move(camera.pimpl_)) {
    pimpl_->pub_ptr_ = this;
}

Camera &Camera::operator=(Camera &&camera) {
    if (this != &camera) {
        pimpl_           = std::move(camera.pimpl_);
        pimpl_->pub_ptr_ = this;
    }
    return *this;
}

Camera::~Camera() {}

Camera::Camera(Private *pimpl) : pimpl_(pimpl) {
    pimpl_->pub_ptr_ = this;
}

Camera Camera::from_first_available() {
    try {
        return Camera(new detail::LivePrivate());
    } catch (const HalConnectionException &e) {
        throw CameraException(CameraErrorCode::ConnectionError, std::string("Connection error: ") + e.what());
    }
}

Camera Camera::from_first_available(const DeviceConfig &config) {
    try {
        return Camera(new detail::LivePrivate(&config));
    } catch (const HalConnectionException &e) {
        throw CameraException(CameraErrorCode::ConnectionError, std::string("Connection error: ") + e.what());
    }
}

Camera Camera::from_source(OnlineSourceType input_source_type, uint32_t source_index) {
    try {
        return Camera(new detail::LivePrivate(input_source_type, source_index));
    } catch (const HalConnectionException &e) {
        throw CameraException(CameraErrorCode::ConnectionError, std::string("Connection error: ") + e.what());
    }
}

Camera Camera::from_source(OnlineSourceType input_source_type, const DeviceConfig &config, uint32_t source_index) {
    try {
        return Camera(new detail::LivePrivate(input_source_type, source_index, &config));
    } catch (const HalConnectionException &e) {
        throw CameraException(CameraErrorCode::ConnectionError, std::string("Connection error: ") + e.what());
    }
}

Camera Camera::from_serial(const std::string &serial) {
    try {
        return Camera(new detail::LivePrivate(serial));
    } catch (const HalConnectionException &e) {
        throw CameraException(CameraErrorCode::ConnectionError, std::string("Connection error: ") + e.what());
    }
}

Camera Camera::from_serial(const std::string &serial, const DeviceConfig &config) {
    try {
        return Camera(new detail::LivePrivate(serial, &config));
    } catch (const HalConnectionException &e) {
        throw CameraException(CameraErrorCode::ConnectionError, std::string("Connection error: ") + e.what());
    }
}

Camera Camera::from_file(const std::filesystem::path &file_path, const FileConfigHints &hints) {
    if (file_path.has_extension()) {
        if (!std::filesystem::exists(file_path)) {
            throw CameraException(CameraErrorCode::FileDoesNotExist,
                                  "Opening file at " + file_path.string() + ": not an existing file.");
        }

        if (!std::filesystem::is_regular_file(file_path)) {
            throw CameraException(CameraErrorCode::NotARegularFile);
        }
    }

    if (file_path.extension().string() == ".raw") {
        return Camera(new detail::OfflineRawPrivate(file_path, hints));
    } else if (file_path.extension().string() == ".hdf5" || file_path.extension().string() == ".h5") {
#if defined HAS_HDF5
        return Camera(new detail::OfflineGenericPrivate(file_path, hints));
#endif
    } else if (file_path.extension().string() == ".dat" || !file_path.has_extension()) {
        return Camera(new detail::OfflineGenericPrivate(file_path, hints));
    }

    throw CameraException(CameraErrorCode::WrongExtension,
                          "Unsupported extension for the provided input file " + file_path.string() + ".");
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

ERCCounter &Camera::erc_counter() {
    return pimpl_->erc_counter();
}

FrameHisto &Camera::frame_histo() {
    return pimpl_->frame_histo();
}

FrameDiff &Camera::frame_diff() {
    return pimpl_->frame_diff();
}

Monitoring &Camera::monitoring() {
    return pimpl_->monitoring();
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

OfflineStreamingControl &Camera::offline_streaming_control() {
    return pimpl_->offline_streaming_control();
}

template<>
I_Geometry &Camera::get_facility<I_Geometry>() {
    return pimpl_->get_geometry();
}

template<>
const I_Geometry &Camera::get_facility<I_Geometry>() const {
    return pimpl_->get_geometry();
}

const I_Geometry &Camera::geometry() const {
    return get_facility<I_Geometry>();
}

const CameraGeneration &Camera::generation() const {
    return pimpl_->generation();
}

bool Camera::start() {
    try {
        return pimpl_->start();
    } catch (const HalConnectionException &e) {
        throw CameraException(CameraErrorCode::ConnectionError, std::string("Connection error: ") + e.what());
    }
}

bool Camera::is_running() {
    return pimpl_->is_running_;
}

bool Camera::stop() {
    try {
        return pimpl_->stop();
    } catch (const HalConnectionException &e) {
        throw CameraException(CameraErrorCode::ConnectionError, std::string("Connection error: ") + e.what());
    }
}

bool Camera::start_recording(const std::filesystem::path &file_path) {
    return pimpl_->start_recording(file_path);
}

bool Camera::stop_recording(const std::filesystem::path &file_path) {
    return pimpl_->stop_recording(file_path);
}

const CameraConfiguration &Camera::get_camera_configuration() const {
    return pimpl_->camera_configuration_;
}

const std::unordered_map<std::string, std::string> &Camera::get_metadata_map() const {
    return pimpl_->metadata_map_;
}

Metavision::timestamp Camera::get_last_timestamp() const {
    return pimpl_->get_last_timestamp();
}

bool Camera::save(const std::filesystem::path &path) const {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw CameraException(CameraErrorCode::CouldNotOpenFile,
                              "Could not open file '" + path.string() +
                                  "' to save camera settings to. Make sure it is a valid filename and that you have "
                                  "permissions to write it.");
    }

    ofs << *this;
    return ofs.good();
}

bool Camera::load(const std::filesystem::path &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw CameraException(CameraErrorCode::CouldNotOpenFile,
                              "Could not open file '" + path.string() +
                                  "' to load camera settings from. Make sure it is a valid filename and that you have "
                                  "permissions to read it.");
    }

    ifs >> *this;
    return ifs.good();
}

Device &Camera::get_device() {
    return pimpl_->device();
}

const Device &Camera::get_device() const {
    return pimpl_->device();
}

Camera::Private &Camera::get_pimpl() {
    return *pimpl_;
}

const Camera::Private &Camera::get_pimpl() const {
    return *pimpl_;
}

std::ostream &operator<<(std::ostream &os, const Camera &camera) {
    camera.get_pimpl().save(os);
    return os;
}

std::istream &operator>>(std::istream &is, Camera &camera) {
    camera.get_pimpl().load(is);
    return is;
}

} // namespace Metavision
