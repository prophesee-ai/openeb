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

#include <thread>
#include <chrono>

#include "boards/v4l2/v4l2_device.h"
#include "boards/v4l2/v4l2_data_transfer.h"
#include "boards/v4l2/v4l2_user_ptr_data.h"

#include "metavision/hal/utils/hal_log.h"

using namespace Metavision;

constexpr bool allow_buffer_drop           = true;
constexpr size_t data_stream_buffer_number = 32;
constexpr size_t data_stream_buffer_size   = 1 * 1024;

constexpr size_t device_buffer_size   = 8 * 1024 * 1024;
constexpr size_t device_buffer_number = 32;

V4l2DataTransfer::V4l2DataTransfer(std::shared_ptr<V4l2Device> device, uint32_t raw_event_size_bytes) :
    DataTransfer(raw_event_size_bytes,
                 DataTransfer::BufferPool::make_bounded(data_stream_buffer_number, data_stream_buffer_size),
                 allow_buffer_drop),
    device_(device) {}

V4l2DataTransfer::~V4l2DataTransfer() {}

void V4l2DataTransfer::start_impl(BufferPtr) {
    MV_HAL_LOG_INFO() << "V4l2DataTransfer - start_impl() ";

    buffers = std::make_unique<V4l2DeviceUserPtr>(device_, "/dev/dma_heap", "linux,cma", device_buffer_size,
                                                  device_buffer_number);

    MV_HAL_LOG_TRACE() << " Nb buffers pre allocated: " << buffers->get_nb_buffers() << std::endl;
    for (unsigned int i = 0; i < buffers->get_nb_buffers(); ++i) {
        buffers->release_buffer(i);
    }
}

void V4l2DataTransfer::run_impl() {
    MV_HAL_LOG_INFO() << "V4l2DataTransfer - run_impl() ";

    while (!should_stop()) {
        // Grab a MIPI frame
        int idx                  = buffers->poll_buffer();
        auto [data, data_length] = buffers->get_buffer_desc(idx);

        MV_HAL_LOG_TRACE() << "Grabed buffer " << idx << "from: " << std::hex << data << " of: " << std::dec
                           << data_length << " Bytes.";

        // Get transfer buffer from the pool and transfer the data
        auto local_buff = get_buffer();
        local_buff->resize(data_length);
        std::memcpy(local_buff->data(), data, data_length);
        transfer_data(local_buff);

        // Reset the buffer data
        memset(data, 0, data_length);

        buffers->release_buffer(idx);
    }
}

void V4l2DataTransfer::stop_impl() {
    MV_HAL_LOG_TRACE() << "V4l2DataTransfer - stop_impl() ";
    buffers.reset();
}
