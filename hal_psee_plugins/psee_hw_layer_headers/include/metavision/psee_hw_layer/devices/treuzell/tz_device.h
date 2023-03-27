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
#include <list>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <utility>
#include <stdexcept>

#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/device_config.h"

namespace Metavision {

class DeviceBuilder;
class DeviceConfig;
class TzLibUSBBoardCommand;
class StreamFormat;

class TzDevice : public std::enable_shared_from_this<TzDevice> {
public:
    std::string get_name();
    std::vector<std::string> get_compatible();
    virtual void get_device_info(I_HW_Identification::SystemInfo &info, std::string prefix);
    virtual DeviceConfigOptionMap get_device_config_options() const;
    virtual std::list<StreamFormat> get_supported_formats() const;
    virtual StreamFormat get_output_format() const;
    virtual StreamFormat set_output_format(const std::string &format_name);
    void set_child(std::shared_ptr<TzDevice> dev);

    virtual void start();
    virtual void stop();

    /* to be called by the TzDeviceBuilder, once the shared pointer on the TzDevice exists */
    virtual void spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config) = 0;

protected:
    TzDevice(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent);
    virtual ~TzDevice();

    virtual void initialize();
    virtual void destroy();

    std::string name;
    std::shared_ptr<TzDevice> parent;
    std::weak_ptr<TzDevice> child;
    std::shared_ptr<TzLibUSBBoardCommand> cmd;
    uint32_t tzID;
};

} // namespace Metavision
#endif /* TZ_DEVICE_H */
