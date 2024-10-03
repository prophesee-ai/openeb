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

#include <metavision/sdk/base/utils/get_time.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/hal/utils/data_transfer.h>
#include <metavision/hal/utils/device_config.h>

#include "sample_camera_discovery.h"
#include "sample_data_transfer.h"
#include "internal/sample_register_access.h"
#include "internal/sample_usb_connection.h"

#include <iostream>

SampleDataTransfer::SampleDataTransfer(uint32_t raw_event_size_bytes,
                                       std::shared_ptr<SampleUSBConnection> usb_connection) :
    buffer_pool_(Metavision::DataTransfer::DefaultBufferPool::make_unbounded()), usb_connection_(usb_connection) {}

SampleDataTransfer::~SampleDataTransfer() = default;

void SampleDataTransfer::start_impl() {}

void SampleDataTransfer::run_impl(const Metavision::DataTransfer &data_transfer) {
    while (!data_transfer.should_stop()) {
        auto buffer = buffer_pool_.acquire();
        buffer->resize(16384); // Accommodate for a full USB packet size

        int transferred = 0;
        auto result     = libusb_bulk_transfer(usb_connection_->get_device_handle(), kEvk4EndpointDataIn, buffer->data(),
                                               buffer->size(), &transferred, 1000);
        if (result != 0) {
            std::cerr << "DATA STREAM endpoint: no data is available with result " << result << std::endl;
        }

        // Resize the buffer to the actual transferred data size
        buffer->resize(transferred);

        // makes the buffer available to the events stream
        data_transfer.transfer_data(buffer);
    }
}
