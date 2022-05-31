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

#include <atomic>
#include <memory>
#include <thread>
#include <libusb.h>

#include "metavision/hal/utils/data_transfer.h"

namespace Metavision {

class PseeLibUSBBoardCommand;

class PseeLibUSBDataTransfer : public DataTransfer {
public:
    PseeLibUSBDataTransfer(const std::shared_ptr<PseeLibUSBBoardCommand> &cmd, uint32_t raw_event_size_bytes);
    ~PseeLibUSBDataTransfer() override;

private:
    void start_impl(BufferPtr buffer) override final;
    void run_impl() override final;
    void stop_impl() override final;

    class UserParamForAsyncBulkCallback;

    virtual void preprocess_transfer(libusb_transfer *transfer);
    void initiate_async_transfers();
    void release_async_transfers();

    std::shared_ptr<PseeLibUSBBoardCommand> cmd_;

    std::mutex protect_vtransfert_;
    std::vector<std::unique_ptr<UserParamForAsyncBulkCallback>> vtransfer_;

    std::atomic<uint32_t> active_bulks_transfers_{0};
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_LIBUSB_DATA_TRANSFER_H
