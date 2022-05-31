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

#include <opencv2/core.hpp>

// Metavision HAL Device class
#include "metavision/hal/device/device.h"
#include "metavision/hal/utils/raw_file_config.h"
#include "metavision/hal/utils/future/raw_file_config.h"

// Metavision SDK Driver CD handler class
#include "metavision/sdk/driver/cd.h"

// Metavision SDK Driver External Trigger handler class
#include "metavision/sdk/driver/ext_trigger.h"

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

// Metavision SDK Driver NoiseFilterModule class
#include "metavision/sdk/driver/noise_filter_module.h"

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
struct CameraConfiguration {
    /// @brief Serial number of the camera
    ///
    /// In case of an online source, it is the serial number of the current online source.\n
    /// In case of an offline source, it is the serial number of the online source used to record the data.
    std::string serial_number = "";
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
    /// @sa @ref Camera(Camera &&camera);
    Camera(Camera &camera) = delete;

    /// @brief Move constructor
    ///
    /// A Camera can not be copied, but it can be move-constructed from another instance.
    Camera(Camera &&camera);

    /// @brief Copy assignment
    ///
    /// A Camera object can not be copy-assigned, but it can be move-assigned.
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
    /// please specify the @ref OnlineSourceType and use the @ref Camera::from_source function,
    /// or else specify the serial number and use the @ref Camera::from_serial function.\n
    /// Serial numbers and types of available sources can be found with @ref Camera::list_online_sources function.
    /// @throw CameraException in case of initialization failure.
    static Camera from_first_available();

    /// @brief Initializes a camera instance from an @ref OnlineSourceType and a source index
    ///
    /// Open the source_index camera of online input_source_type if available from @ref list_online_sources\.\n
    /// By default, it opens the first available camera listed by @ref list_online_sources of type input_source_type.\n
    /// Serial numbers and types of available sources can be found with @ref Camera::list_online_sources function.
    /// @throw CameraException if the camera corresponding to the input source type and the source index has not been
    /// found.
    /// @param input_source_type @ref OnlineSourceType
    /// @param source_index Index of the source in the list of available online sources
    /// @return @ref Camera instance initialized from the source
    static Camera from_source(OnlineSourceType input_source_type, uint32_t source_index = 0);

    /// @brief Initializes a camera instance from a 'serial' number
    ///
    /// Serial numbers of available sources can be found by with @ref Camera::list_online_sources function.\n
    /// If 'serial' is an empty string, the function works as the main constructor.
    /// @throw CameraException if the camera with the input serial number has not been found.
    /// @param serial Serial number of the camera
    /// @return @ref Camera instance initialized from the serial number
    static Camera from_serial(const std::string &serial);

    /// @brief Initializes a camera instance from a RAW file
    /// @throw CameraException in case of initialization failure.
    /// @param rawfile Path to the RAW file
    /// @param realtime_playback_speed If true, the RAW file will be read at the same speed as was sent by the camera
    ///                                  when the file was recorded, and the events will be available after the same
    ///                                  amount of time it took for them to be received when recording the RAW file. If
    ///                                  false, the file will be read as fast as possible and the events will be
    ///                                  available as soon as possible as well. The max_event_lifespan will only be
    ///                                  taken into account when reproducing the camera behavior.
    /// @param file_config Configuration describing how to read the file (see @ref RawFileConfig)
    /// @note Since 2.1.0, the @p realtime_playback_speed is only taken into account if at least one event callback
    ///       is registered (CD or ExtTrigger), it will have no effect if only a RawData callback is registered.
    /// @return @ref Camera instance initialized from the input RAW file
    /// @return @ref Camera instance initialized from the input RAW file
    static Camera from_file(const std::string &rawfile, bool realtime_playback_speed = true,
                            const RawFileConfig &file_config = RawFileConfig());

    /// @brief Initializes a camera instance from a RAW file
    /// @throw CameraException in case of initialization failure.
    /// @param rawfile Path to the RAW file
    /// @param realtime_playback_speed If true, the RAW file will be read at the same speed as was sent by the camera
    ///                                  when the file was recorded, and the events will be available after the same
    ///                                  amount of time it took for them to be received when recording the RAW file. If
    ///                                  false, the file will be read as fast as possible and the events will be
    ///                                  available as soon as possible as well. The max_event_lifespan will only be
    ///                                  taken into account when reproducing the camera behavior.
    /// @param file_config Configuration describing how to read the file (see @ref Future::RawFileConfig)
    /// @note Since 2.1.0, the @p realtime_playback_speed is only taken into account if at least one event callback
    ///       is registered (CD or ExtTrigger), it will have no effect if only a RawData callback is registered.
    /// @return @ref Camera instance initialized from the input RAW file
    static Camera from_file(const std::string &rawfile, bool realtime_playback_speed,
                            const Future::RawFileConfig &file_config);

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

    /// @brief Gets class to handle STC or TRAIL Noise Filter Module on the hardware side
    /// @throw CameraException in case of failure (for instance if the camera is not initialized or the camera is
    /// running from an offline source).
    NoiseFilterModule &noise_filter_module();

    /// @brief Registers a callback that will be called when a runtime error occurs
    ///
    /// When a camera runtime error occurs, the camera thread is left and events are no longer sent.
    /// You are notified by this callback whenever this happens.
    /// @throw CameraException if the camera has not been initialized.
    /// @param error_callback The error callback to call
    /// @return ID of the added callback
    /// @warning It is forbidden to call the @ref Camera::stop from a runtime error callback.
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
    /// when the camera is started or stopped (by a call to @ref Camera::stop or
    /// when no more events are available in a recording).
    ///
    /// @param status_change_callback The status change callback to call
    /// @return ID of the added callback
    /// @warning It is forbidden to call the @ref Camera::stop from the status callback.
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
    /// @return true if the camera is running or there are data remaining from an offline source, false otherwise.
    bool is_running();

    /// @brief Stops polling events from the camera or from the file
    ///
    /// Stops ongoing streaming.
    /// @throw CameraException if the camera has not been initialized.
    /// @return true if the camera instance has been stopped successfully, false otherwise. If the camera was not
    /// running, this function returns false.
    bool stop();

    /// @brief Records data from camera to a file with .raw extension
    ///
    /// The call to this function stops ongoing recording.\n
    /// The function creates a new file at the given path or overwrites the already existing file.\n
    /// In case of an offline input source, the function can be used to split the RAW file.
    /// In case of not having rights to write at the provided path, the function will not record anything.
    /// @throw CameraException if the camera has not been initialized.
    /// @param rawfile_path Path to the RAW file used for data recording.
    void start_recording(const std::string &rawfile_path);

    /// @brief Stops an ongoing recording
    /// @throw CameraException if the camera has not been initialized.
    void stop_recording();

    /// @brief Returns @ref CameraConfiguration of the camera that holds the camera properties (dimensions, camera
    /// biases, ...)
    ///
    /// Read-only structure.
    /// @sa @ref CameraConfiguration
    const CameraConfiguration &get_camera_configuration();

    /// @brief Gets the last decoded timestamp
    /// @return timestamp Last decoded timestamp
    /// @warning If no event decoding callback has been set, this functions returns -1
    timestamp get_last_timestamp() const;

    /// @brief Gets corresponding @ref Device in HAL library
    /// @return The @ref Device used internally by the class Camera
    Device &get_device();

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
