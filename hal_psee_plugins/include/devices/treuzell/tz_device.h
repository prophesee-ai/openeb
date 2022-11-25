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

#ifndef TZ_DEVICE_H
#define TZ_DEVICE_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <utility>

#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/utils/device_builder.h"

namespace Metavision {

class DeviceBuilder;
class DeviceConfig;
class TzLibUSBBoardCommand;
class TzDeviceBuilder;
class LibUSBContext;

class TzDevice : public std::enable_shared_from_this<TzDevice> {
public:
    struct StreamFormat {
        std::string name;
        std::unique_ptr<I_Geometry> geometry;
    };
    std::string get_name();
    std::vector<std::string> get_compatible();
    virtual void get_device_info(I_HW_Identification::SystemInfo &info, std::string prefix);
    virtual StreamFormat get_output_format() = 0;

    virtual void start() = 0;
    virtual void stop()  = 0;

protected:
    TzDevice(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent);
    /* to be called by the TzDeviceBuilder, once the shared pointer on the TzDevice exists */
    virtual void spawn_facilities(DeviceBuilder &device_builder) = 0;
    virtual ~TzDevice();
    std::shared_ptr<TzDevice> parent;
    std::weak_ptr<TzDevice> child;
    std::shared_ptr<TzLibUSBBoardCommand> cmd;
    uint32_t tzID;
    friend TzDeviceBuilder;
};

class TzRegisterBuildMethod;
class TzDeviceBuilder {
public:
    using Build_Fun = std::function<std::shared_ptr<TzDevice>(std::shared_ptr<TzLibUSBBoardCommand>, uint32_t id,
                                                              std::shared_ptr<TzDevice> parent)>;
    using Check_Fun = std::function<bool(std::shared_ptr<TzLibUSBBoardCommand>, uint32_t id)>;
    using Build_Map = std::unordered_map<std::string, std::pair<Build_Fun, Check_Fun>>;

    TzDeviceBuilder() : map(generic_map()) {}

    void set(std::string key, Build_Fun method, Check_Fun buildable = nullptr) {
        map[key] = {method, buildable};
    }
    bool can_build(std::shared_ptr<TzLibUSBBoardCommand>);
    bool can_build_device(std::shared_ptr<TzLibUSBBoardCommand>, uint32_t dev_id);
    /******************************************************************************************************************
    The builder is meant to evolve to be able to manage future use cases:
    * builder will, for the board having the required serial, build a TzDevice for every device plugged on the board.
    * TzDevice derivated classes will implement, when necessary, the init/start/stop/destroy (ISSD) sequencies for the
      device, and spawn the facilities as defined by Metavision HAL.
    * As the TzDevice need to give a reference of itself to the facilities, and the lifetime of objects is managed
      using shared pointers, the constructor is protected, and the construction is done through a static method.
    * The order of device build may be important as some devices depend on others for power or clock. Today, it is
      assumed that building it treuzell index order is fine, but, if needed, commands may be added in Treuzell to
      describe chaining. The builder shall build device in appropriate order. At some point, some devices may choose
      how to build others, it is not implemented yet, but suggested implementation, if needed, would be to add a
      specific interface to devices requiring this, and detect it through runtine type information (rtti)
    * The builder will also spawn the device control and propagate start/stop as needed. The propagation mechanism will
      be strictly internal, and can be rewritten to manage multiple successors if needed at some point in the future.
    ******************************************************************************************************************/
    bool build_devices(std::shared_ptr<TzLibUSBBoardCommand> cmd, DeviceBuilder &device_builder,
                       const DeviceConfig &config);

private:
    Build_Map map;
    static Build_Map &generic_map();
    friend TzDevice;
    friend TzRegisterBuildMethod;
};

class TzRegisterBuildMethod {
public:
    TzRegisterBuildMethod(std::string key, TzDeviceBuilder::Build_Fun method,
                          TzDeviceBuilder::Check_Fun buildable = nullptr) {
        TzDeviceBuilder::generic_map().insert({key, {method, buildable}});
    }
};

} // namespace Metavision
#endif /* TZ_DEVICE_H */
