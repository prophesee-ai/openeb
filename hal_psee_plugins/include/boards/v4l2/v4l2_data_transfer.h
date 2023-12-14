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

#ifndef METAVISION_HAL_PSEE_PLUGINS_V4L2_DATA_TRANSFER_H
#define METAVISION_HAL_PSEE_PLUGINS_V4L2_DATA_TRANSFER_H

#include "metavision/hal/utils/data_transfer.h"

namespace Metavision {

class V4L2DeviceControl;
class V4l2DeviceUserPtr; // @TODO Replace with a V4l2 Buffer class interface

class V4l2DataTransfer : public DataTransfer {
public:
    V4l2DataTransfer(std::shared_ptr<V4L2DeviceControl> device, uint32_t raw_event_size_bytes);
    ~V4l2DataTransfer();

private:
    std::shared_ptr<V4L2DeviceControl> device_;
    std::unique_ptr<V4l2DeviceUserPtr> buffers;

    void start_impl(BufferPtr buffer) override final;
    void run_impl() override final;
    void stop_impl() override final;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_V4L2_DATA_TRANSFER_H
