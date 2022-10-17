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

#ifndef METAVISION_HAL_TZ_BOARD_DATA_TRANSFER_H
#define METAVISION_HAL_TZ_BOARD_DATA_TRANSFER_H

#include <atomic>
#include <memory>
#include <thread>
#include <libusb.h>
#include <thread>
#include <mutex>
#include <chrono>
#include <list>

#include "metavision/hal/utils/data_transfer.h"
#include "boards/treuzell/tz_libusb_board_command.h"

namespace Metavision {

class TzBoardDataTransfer : public DataTransfer {
public:
    static DataTransfer::BufferPool make_buffer_pool(size_t max_pool_byte_size = 0);

    TzBoardDataTransfer(const std::shared_ptr<TzLibUSBBoardCommand> &cmd, uint32_t raw_event_size_bytes,
                        const DataTransfer::BufferPool &buffer_pool = make_buffer_pool());
    ~TzBoardDataTransfer() override;

private:
    void start_impl(BufferPtr buffer) override final;
    void run_impl() override final;
    void stop_impl() override final;
    void flush();

    class UserParamForAsyncBulkCallback;

    void preprocess_transfer(libusb_transfer *transfer);
    void initiate_async_transfers();
    void release_async_transfers();
    int bulk_transfer(unsigned char *buf, int packet_size, unsigned int timeout, int &actual_size);

    static uint32_t timeout_cnt_;
    std::shared_ptr<TzLibUSBBoardCommand> cmd_;
    libusb_device_handle *dev_handle_;
    int bEpCommAddress;

    std::mutex protect_vtransfert_;
    std::vector<std::unique_ptr<UserParamForAsyncBulkCallback>> vtransfer_;

    std::atomic<uint32_t> active_bulks_transfers_{0};

    // USB Commands
    libusb_transfer *contruct_async_bulk_transfer(unsigned char *buf, int packet_size,
                                                  libusb_transfer_cb_fn async_bulk_cb, void *user_data,
                                                  unsigned int timeout);
    static void free_async_bulk_transfer(libusb_transfer *transfer);
    static int submit_transfer(libusb_transfer *transfer);
    void prepare_async_bulk_transfer(libusb_transfer *transfer, unsigned char *buf, int packet_size,
                                     libusb_transfer_cb_fn async_bulk_cb, void *user_data, unsigned int timeout);

    static const size_t packet_size_;
    static const size_t async_transfer_num_;
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_BOARD_DATA_TRANSFER_H
