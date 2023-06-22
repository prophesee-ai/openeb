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

#ifndef METAVISION_SDK_DRIVER_CAMERA_H
#define METAVISION_SDK_DRIVER_CAMERA_H

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <opencv2/core.hpp>

// Metavision HAL Device class
#include "metavision/hal/device/device.h"

// Metavision SDK Driver File Config Hints class
#include "metavision/sdk/driver/file_config_hints.h"

// Metavision SDK Driver CD handler class
#include "metavision/sdk/driver/cd.h"

// Metavision SDK Driver ERCCounter handler class
#include "metavision/sdk/driver/erc_counter.h"

// Metavision SDK Driver External Trigger handler class
#include "metavision/sdk/driver/ext_trigger.h"

// Metavision SDK Driver FrameDiff handler class
#include "metavision/sdk/driver/frame_diff.h"

// Metavision SDK Driver FrameHisto handler class
#include "metavision/sdk/driver/frame_histo.h"

// Metavision SDK Driver Trigger Out handler class
#include "metavision/sdk/driver/trigger_out.h"

// Metavision SDK Driver RAW data handler class
#include "metavision/sdk/driver/raw_data.h"

// Metavision SDK Driver camera ROI handler
#include "metavision/sdk/driver/roi.h"

// Metavision SDK driver Biases class
#include "metavision/sdk/driver/biases.h"

// Metavision SDK driver OfflineStreamingControl class
#include "metavision/sdk/driver/offline_streaming_control.h"

// Metavision SDK Driver Geometry handler class
#include "metavision/sdk/driver/geometry.h"

// Metavision device generation
#include "metavision/sdk/driver/camera_generation.h"

// Metavision SDK Driver camera exceptions
#include "metavision/sdk/driver/camera_exception.h"

// Metavision SDK callback id
#include "metavision/sdk/base/utils/callback_id.h"

// Metavision SDK timestamp
#include "metavision/sdk/base/utils/timestamp.h"

// Metavision SDK Driver AntiFlickerModule class
#include "metavision/sdk/driver/antiflicker_module.h"

// Metavision SDK Driver ErcModule class
#include "metavision/sdk/driver/erc_module.h"

// Metavision SDK Driver EventTrailFilterModule class
#include "metavision/sdk/driver/event_trail_filter_module.h"

namespace Metavision {

/// @brief Online camera type input sources: USB, embedded, remote
enum class OnlineSourceType : short {
    /// Data from an embedded event-based camera
    EMBEDDED = 0,

    /// Data from an USB event-based camera
    USB = 1,

    /// Data from a remote event-based camera
    REMOTE = 2,
};

enum class CameraStatus : short {
    /// Camera is started, see @ref Camera::start and @ref Camera::is_running
    STARTED = 0,

    /// Camera is stopped, see @ref Camera::stop and @ref Camera::is_running
    STOPPED = 1
};

/// @brief Available online sources type alias
using AvailableSourcesList = std::map<OnlineSourceType, std::vector<std::string>>;

/// @brief Callback type alias for @ref CameraException
/// @ref CameraException the camera exception generated.
using RuntimeErrorCallback = std::function<void(const CameraException &)>;

/// @brief Callback type alias to be used with @ref Camera::add_status_change_callback
using StatusChangeCallback = std::function<void(const CameraStatus &)>;

/// @brief Struct with the current camera configuration.
///
/// In case of an online source, the fields apply to the current online source.
/// In case of an offline source, the fields correspond to the online source used to record the data.
struct CameraConfiguration {
    std::string serial_number;
    std::string system_ID;
    std::string plugin_name;
    std::string integrator;
    std::string data_encoding_format;
    std::string firmware_version;
};

/// @brief Main class for the camera interface
class Camera {
public:
    /// @brief Lists available sources for the online mode.
    /// @return @ref AvailableSourcesList structure containing available cameras (plugged on the system) along with
    /// their serial numbers.
    static AvailableSourcesList list_online_sources();

    /// @brief Constructor
    ///
    /// Creates an uninitialized camera instance.
    Camera();

    /// @brief Copy constructor
    ///
    /// A Camera object can not be copy-constructed, but it can be move-constructed.
    ///
    /// @sa @ref Camera(Camera &&camera);
    Camera(Camera &camera) = delete;

    /// @brief Move constructor
    ///
    /// A Camera can not be copied, but it can be move-constructed from another instance.
    Camera(Camera &&camera);

    /// @brief Copy assignment
    ///
    /// A Camera object can not be copy-assigned, but it can be move-assigned.
    ///
    /// @sa @ref Camera & operator=(Camera &&camera);
    Camera &operator=(Camera &camera) = delete;

    /// @brief Move assignment
    ///
    /// A Camera can not be copied, but it can be move-assigned from another instance.
    Camera &operator=(Camera &&camera);

    /// @brief Destructor
    ~Camera();

    /// @brief Initializes a camera instance from the first available camera plugged on the system
    ///
    /// Open the first available camera following at first EMBEDDED and then USB order.\n
    /// Please note that remote cameras will not be opened with this function. To do that,
    /// please specify the @ref OnlineSourceType and use the @ref from_source function,
    /// or else specify the serial number and use the @ref from_serial function.\n
    /// Serial numbers and types of available sources can be found with @ref list_online_sources function.
    ///
    /// @throw CameraException in case of initialization failure.
    static Camera from_first_available();

    /// @brief Initializes a camera instance from the first available camera plugged on the system
    ///
    /// Open the first available camera following at first EMBEDDED and then USB order.\n
    /// Please note that remote cameras will not be opened with this function. To do that,
    /// please specify the @ref OnlineSourceType and use the @ref from_source function,
    /// or else specify the serial number and use the @ref from_serial function.\n
    /// Serial numbers and types of available sources can be found with @ref list_online_sources function.
    ///
    /// @throw CameraException in case of initialization failure.
    /// @param config Configuration used to open the camera
    /// @overload
    static Camera from_first_available(DeviceConfig &config);

    /// @brief Initializes a camera instance from an @ref OnlineSourceType and a source index
    ///
    /// Open the source_index camera of online input_source_type if available from @ref list_online_sources"".\n
    /// By default, it opens the first available camera listed by @ref list_online_sources of type input_source_type.\n
    /// Serial numbers and types of available sources can be found with @ref list_online_sources function.
    /// @throw CameraException if the camera corresponding to the input source type and the source index has not been
    /// found.
    ///
    /// @param input_source_type @ref OnlineSourceType
    /// @param source_index Index of the source in the list of available online sources
    /// @return @ref Camera instance initialized from the source
    static Camera from_source(OnlineSourceType input_source_type, uint32_t source_index = 0);

    /// @brief Initializes a camera instance from an @ref OnlineSourceType and a source index
    ///
    /// Open the source_index camera of online input_source_type if available from @ref list_online_sources"".\n
    /// By default, it opens the first available camera listed by @ref list_online_sources of type input_source_type.\n
    /// Serial numbers and types of available sources can be found with @ref list_online_sources function.
    /// @throw CameraException if the camera corresponding to the input source type and the source index has not been
    /// found.
    ///
    /// @param input_source_type @ref OnlineSourceType
    /// @param config Configuration used to open the camera
    /// @param source_index Index of the source in the list of available online sources
    /// @return @ref Camera instance initialized from the source
    /// @overload
    static Camera from_source(OnlineSourceType input_source_type, DeviceConfig &config, uint32_t source_index = 0);

    /// @brief Initializes a camera instance from a 'serial' number
    ///
    /// Serial numbers of available sources can be found by with @ref list_online_sources function.\n
    /// If 'serial' is an empty string, the function works as the main constructor.
    ///
    /// @throw CameraException if the camera with the input serial number has not been found.
    /// @param serial Serial number of the camera
    /// @return @ref Camera instance initialized from the serial number
    static Camera from_serial(const std::string &serial);

    /// @brief Initializes a camera instance from a 'serial' number
    /// @throw CameraException if the camera with the input serial number has not been found.
    /// @param serial Serial number of the camera
    /// @param config Configuration used to open the camera
    /// @return @ref Camera instance initialized from the serial number
    /// @overload
    static Camera from_serial(const std::string &serial, DeviceConfig &config);

    /// @brief Initializes a camera instance from a file
    /// @throw CameraException in case of initialization failure.
    /// @param file_path Path to the file
    /// @param hints Hints expressing how the file should be read, for more details see @ref FileConfigHints
    /// @return @ref Camera instance initialized from the input file
    static Camera from_file(const std::string &file_path, const FileConfigHints &hints = FileConfigHints());

    /// @brief Gets class to handle RAW data from the camera
    /// @throw CameraException if the camera has not been initialized.
    RawData &raw_data();

    /// @brief Gets class to handle CD events
    /// @throw CameraException if the camera has not been initialized.
    CD &cd();

    /// @brief Gets class to handle External Triggers events
    /// @throw CameraException if the camera has not been initialized.
    ExtTrigger &ext_trigger();

    /// @brief Gets class to handle trigger out signal
    /// @throw CameraException if the camera has not been initialized.
    /// @throw CameraException in case of failure (for example if camera runs from an offline source).
    TriggerOut &trigger_out();

    /// @brief Gets class to handle ERCCounter events
    /// @throw CameraException if the camera has not been initialized.
    ERCCounter &erc_counter();

    /// @brief Gets class to handle RawEventFrameHisto
    /// @throw CameraException if the camera has not been initialized.
    FrameHisto &frame_histo();

    /// @brief Gets class to handle RawEventFrameDiff
    /// @throw CameraException if the camera has not been initialized.
    FrameDiff &frame_diff();

    /// @brief Gets class to handle Roi on the sensor
    /// @throw CameraException in case of failure (for instance if the camera is not initialized or the camera is
    /// running from an offline source).
    Roi &roi();

    /// @brief Gets class to handle AFK on the hardware side
    /// @throw CameraException in case of failure (for instance if the camera is not initialized or the camera is
    /// running from an offline source).
    AntiFlickerModule &antiflicker_module();

    /// @brief Gets class to handle Event Rater Controller on the hardware side
    /// @throw CameraException in case of failure (for instance if the camera is not initialized or the camera is
    /// running from an offline source).
    ErcModule &erc_module();

    /// @brief Gets class to handle STC or TRAIL Event Trail Filter Module on the hardware side
    /// @throw CameraException in case of failure (for instance if the camera is not initialized or the camera is
    /// running from an offline source).
    EventTrailFilterModule &event_trail_filter_module();

    /// @brief Registers a callback that will be called when a runtime error occurs
    ///
    /// When a camera runtime error occurs, the camera thread is left and events are no longer sent.
    /// You are notified by this callback whenever this happens.
    ///
    /// @throw CameraException if the camera has not been initialized.
    /// @param error_callback The error callback to call
    /// @return ID of the added callback
    /// @warning It is forbidden to call the @ref stop from a runtime error callback.
    /// @sa @ref RuntimeErrorCallback
    CallbackId add_runtime_error_callback(RuntimeErrorCallback error_callback);

    /// @brief Removes a previously registered callback
    /// @param callback_id Callback ID
    /// @return true if the callback has been unregistered correctly, false otherwise.
    /// @sa @ref CD::add_callback
    /// @sa @ref ExtTrigger::add_callback
    bool remove_runtime_error_callback(CallbackId callback_id);

    /// @brief Registers a callback that will be called when the camera status changes.
    ///
    /// The callback will be called with the new status of the @ref Camera as a parameter,
    /// when the camera is started or stopped (by a call to @ref stop or
    /// when no more events are available in a recording).
    ///
    /// @param status_change_callback The status change callback to call
    /// @return ID of the added callback
    /// @warning It is forbidden to call the @ref stop from the status callback.
    /// @sa @ref StatusChangeCallback
    CallbackId add_status_change_callback(StatusChangeCallback status_change_callback);

    /// @brief Removes a previously registered callback
    ///
    /// @param callback_id Callback ID
    /// @return true if the callback has been unregistered correctly, false otherwise.
    bool remove_status_change_callback(CallbackId callback_id);

    /// @brief Gets class to handle camera biases
    /// @throw CameraException in case of failure (for example if camera runs from an offline source).
    Biases &biases();

    /// @brief Gets class to control offline streaming
    /// @throw CameraException if the camera has not been initialized or if the feature is not available.
    OfflineStreamingControl &offline_streaming_control();

    /// @brief Gets the device's geometry
    /// @throw CameraException if the camera has not been initialized.
    const Geometry &geometry() const;

    /// @brief Gets the device's generation
    /// @throw CameraException if the camera has not been initialized.
    const CameraGeneration &generation() const;

    /// @brief Starts the camera from the given input source
    ///
    /// It will start polling events from the source and calling specified events callbacks.\n
    /// It has no effect if the start function has been already called and not the @ref stop function.
    ///
    /// @throw CameraException if the camera has not been initialized.
    /// @sa @ref CD::add_callback
    /// @sa @ref ExtTrigger::add_callback
    /// @return true if the camera started successfully, false otherwise. Also returns false, if the camera is already
    /// started.
    bool start();

    /// @brief Checks if the camera is running or there are data remaining from an offline source
    ///
    /// If the source is online, it always returns true unless the @ref stop function has been called.\n
    /// If the input is offline (from a file), it returns false whenever no data are remaining in the input file.
    ///
    /// @return true if the camera is running or there are data remaining from an offline source, false otherwise.
    bool is_running();

    /// @brief Stops polling events from the camera or from the file
    ///
    /// Stops ongoing streaming.
    ///
    /// @throw CameraException if the camera has not been initialized.
    /// @return true if the camera instance has been stopped successfully, false otherwise. If the camera was not
    /// running, this function returns false.
    bool stop();

    /// @brief Records data from camera to a file at the specified path
    ///
    /// The function creates a new file at the given @p file_path or overwrites the already existing file.\n
    /// In case of an offline input source, the function can be used to split the file and record only a portion of it.
    /// @throw CameraException if the camera has not been initialized.
    /// @warning Calling this function will overwrite the file at the path @p file_path if it already exists.\n
    /// @note This functions is the recommended way to save recording with SDK driver.
    /// It uses a separate thread to write the file for efficiency, so it will not slow down the decoding thread
    /// as opposed to \ref I_EventsStream::log_raw_data and \ref I_EventsStream::stop_log_raw_data
    /// It also enables writing to supported formats other than RAW file, altough the writing speed will probably
    /// decrease for those formats
    /// It can also be called several times with different paths to record the stream to multiple files at the same time
    /// @note For more control over the way the data is recorded and to select with precision which events
    /// will be recorded, you may directly use the API provided by \ref EventFileWriter and its inherited classes
    /// For more information, refer to the metavision_file_cutter sample
    /// @param file_path Path to the file containing the recorded data
    /// @return true if recording could be started, false otherwise
    bool start_recording(const std::string &file_path);

    /// @brief Stops recording data from camera to the specified path
    ///
    /// This function stops recording data to the file at the given @p file_path.\n
    /// If the @p file_path is empty, all recordings of the current camera stream are stopped.\n
    /// @param file_path Path to the file containing the recorded data. If empty, all ongoing recordings are stopped.
    /// @throw CameraException if the camera has not been initialized.
    /// @return true if recording could be stopped, false otherwise
    bool stop_recording(const std::string &file_path = std::string());

    /// @brief Returns @ref CameraConfiguration of the camera that holds the available camera properties (serial,
    /// systemID, format of events, etc.)
    ///
    /// Read-only structure.
    ///
    /// @sa @ref CameraConfiguration
    const CameraConfiguration &get_camera_configuration() const;

    /// @brief Returns a dictionary of the camera metadata if available (e.g from a file which contains manually added
    /// comments)
    ///
    /// Read-only structure.
    const std::unordered_map<std::string, std::string> &get_metadata_map() const;

    /// @brief Gets the last decoded timestamp
    /// @return timestamp Last decoded timestamp
    /// @warning If no event decoding callback has been set, this functions returns -1
    timestamp get_last_timestamp() const;

    /// @brief Gets corresponding @ref Device in HAL library
    ///
    /// This Device retrieved can then be used to call the different facilities of the camera.
    /// for example: camera.get_device()->get_facility<Metavision::I_TriggerIn>()->enable(channel_id)
    /// or: camera.get_device()->get_facility<Metavision::I_Monitoring>()->get_temperature())
    ///
    /// @return The @ref Device used internally by the class Camera
    Device &get_device();

    /// @brief Gets corresponding @ref Device in HAL library
    ///
    /// This Device retrieved can then be used to call the different facilities of the camera.
    /// for example: camera.get_device()->get_facility<Metavision::I_TriggerIn>()->enable(channel_id)
    /// or: camera.get_device()->get_facility<Metavision::I_Monitoring>()->get_temperature())
    ///
    /// @return The @ref Device used internally by the class Camera
    const Device &get_device() const;

    /// @brief For internal use
    class Private;

    /// @brief For internal use
    Private &get_pimpl();

    /// @brief For internal use
    Camera(Private *);

private:
    /// @brief For internal use
    std::unique_ptr<Private> pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_CAMERA_H
