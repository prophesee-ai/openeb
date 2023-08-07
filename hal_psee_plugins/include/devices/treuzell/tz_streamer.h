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

#ifndef METAVISION_HAL_TZ_STREAMER_H
#define METAVISION_HAL_TZ_STREAMER_H

#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"

namespace Metavision {

class DeviceBuilder;
class DeviceBuilderParameters;
class DeviceConfig;

class TzStreamer : public TzDevice {
public:
    TzStreamer(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent);
    virtual ~TzStreamer();
    static std::shared_ptr<TzDevice> build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                           std::shared_ptr<TzDevice> parent);

    virtual void start();
    virtual void stop();
    virtual std::list<StreamFormat> get_supported_formats() const override;
    StreamFormat set_output_format(const std::string &format_name) override;
    StreamFormat get_output_format() const override;

protected:
    virtual void spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config);
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_STREAMER_H
