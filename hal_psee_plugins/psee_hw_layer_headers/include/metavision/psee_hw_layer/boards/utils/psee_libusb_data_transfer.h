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

#ifndef METAVISION_HAL_PSEE_LIBUSB_DATA_TRANSFER_H
#define METAVISION_HAL_PSEE_LIBUSB_DATA_TRANSFER_H

#include <memory>
#include "metavision/psee_hw_layer/boards/utils/psee_libusb.h"

#include "metavision/hal/utils/data_transfer.h"

namespace Metavision {

class PseeLibUSBDataTransfer : public DataTransfer::RawDataProducer {
    using EpId = unsigned char;

    using BufferPtr = DataTransfer::DefaultBufferPtr;

public:
    static DataTransfer::DefaultBufferPool make_buffer_pool(size_t max_pool_byte_size = 0);

    PseeLibUSBDataTransfer(const std::shared_ptr<LibUSBDevice> dev, EpId endpoint, uint32_t raw_event_size_bytes,
                           const DataTransfer::DefaultBufferPool &buffer_pool = make_buffer_pool());
    ~PseeLibUSBDataTransfer() override;

private:
    void start_impl() override final;
    void run_impl(const DataTransfer &data_transfer) override final;
    void stop_impl() override final;

    void run_transfers(const DataTransfer &data_transfer);

    DataTransfer::DefaultBufferPool buffer_pool_;
    std::shared_ptr<LibUSBDevice> dev_;
    EpId bEpCommAddress_;

    class AsyncTransfer;
    std::vector<AsyncTransfer> vtransfer_;
    uint32_t timeout_cnt_;

    static const size_t packet_size_;
    static const size_t async_transfer_num_;
    static const unsigned int timeout_;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_LIBUSB_DATA_TRANSFER_H
