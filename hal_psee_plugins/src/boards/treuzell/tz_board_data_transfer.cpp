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

#include <assert.h>

#ifdef _WIN32
#ifndef _MSC_VER
#define WIN_CALLBACK_DECL __stdcall
#else
#define WIN_CALLBACK_DECL WINAPI
#endif
#else
#define WIN_CALLBACK_DECL
#endif

#include "boards/treuzell/tz_board_data_transfer.h"
#include "boards/treuzell/tz_libusb_board_command.h"
#include "boards/utils/config_registers_map.h"

#include "metavision/hal/utils/hal_log.h"

const static int USB_TIME_OUT                = 100;
static const int N_ASYNC_TRANFERS_PER_DEVICE = 20;
const static int PACKET_SIZE                 = 128 * 1024;

namespace Metavision {

uint32_t TzBoardDataTransfer::timeout_cnt_;
TzBoardDataTransfer::TzBoardDataTransfer(const std::shared_ptr<TzLibUSBBoardCommand> &cmd,
                                         uint32_t raw_event_size_bytes) :
    DataTransfer(raw_event_size_bytes), cmd_(cmd) {
    flush();
}

void TzBoardDataTransfer::preprocess_transfer(libusb_transfer *transfer) {
    if (transfer->status == LIBUSB_TRANSFER_TIMED_OUT) {
        if (transfer->actual_length != 0) {
            timeout_cnt_ = 0;
        } else {
            timeout_cnt_++;
            if (timeout_cnt_ >= 100) {
                MV_HAL_LOG_TRACE() << "\rBulk Transfer NACK " << timeout_cnt_;
            }
        }
    } else if (transfer->status == LIBUSB_TRANSFER_COMPLETED) {
        if (timeout_cnt_ != 0) {
            timeout_cnt_ = 0;
        }
    }
}

class TzBoardDataTransfer::UserParamForAsyncBulkCallback {
public:
    UserParamForAsyncBulkCallback(int id, const std::shared_ptr<TzLibUSBBoardCommand> &cmd,
                                  TzBoardDataTransfer &libusb_data_transfer);
    ~UserParamForAsyncBulkCallback();
    static void WIN_CALLBACK_DECL async_bulk_cb(struct libusb_transfer *transfer);

    void start();
    void stop();

private:
    bool proceed_async_bulk(struct libusb_transfer *transfer);

    const static int timeout_ = USB_TIME_OUT;
    DataTransfer::BufferPtr buf_;
    std::mutex transfer_mutex_;
    libusb_transfer *transfer_{nullptr};
    bool stop_ = true;
    std::atomic<bool> submitted_transfer_{false};
    std::shared_ptr<TzLibUSBBoardCommand> cmd_;
    TzBoardDataTransfer &libusb_data_transfer_;
};

TzBoardDataTransfer::~TzBoardDataTransfer() {
    stop_impl();
}

void TzBoardDataTransfer::start_impl(BufferPtr buffer) {
    initiate_async_transfers();
}

void TzBoardDataTransfer::run_impl() {
    MV_HAL_LOG_TRACE() << "poll thread running";
    while (!should_stop() && active_bulks_transfers_ > 0) {
        struct timeval tv = {0, 1};
        libusb_handle_events_timeout(cmd_->libusb_ctx->ctx(), &tv);
    }
    MV_HAL_LOG_TRACE() << "poll thread shutting down";

    release_async_transfers();
}

void TzBoardDataTransfer::stop_impl() {
    for (auto &transfer : vtransfer_) {
        transfer->stop();
    }
}

void TzBoardDataTransfer::initiate_async_transfers() {
    for (int i = 0; i < N_ASYNC_TRANFERS_PER_DEVICE; ++i) {
        vtransfer_.push_back(std::make_unique<UserParamForAsyncBulkCallback>(i, cmd_, *this));
        vtransfer_.back()->start();
    }
}

void TzBoardDataTransfer::release_async_transfers() {
    vtransfer_.clear();
}

TzBoardDataTransfer::UserParamForAsyncBulkCallback::UserParamForAsyncBulkCallback(
    int id, const std::shared_ptr<TzLibUSBBoardCommand> &cmd, TzBoardDataTransfer &libusb_data_transfer) :
    cmd_(cmd), libusb_data_transfer_(libusb_data_transfer) {
    buf_ = libusb_data_transfer.get_buffer();
    buf_->resize(PACKET_SIZE);
    transfer_ =
        libusb_data_transfer.contruct_async_bulk_transfer(buf_->data(), PACKET_SIZE, async_bulk_cb, this, timeout_);
}

TzBoardDataTransfer::UserParamForAsyncBulkCallback::~UserParamForAsyncBulkCallback() {
    stop();

    // Wait for submitted transfer to be processed before releasing this object.
    while (submitted_transfer_) {
        struct timeval tv = {0, 1};
        libusb_handle_events_timeout(cmd_->libusb_ctx->ctx(), &tv);
    };

    if (transfer_) {
        free_async_bulk_transfer(transfer_);
        transfer_ = nullptr;
    }
}

void WIN_CALLBACK_DECL
    TzBoardDataTransfer::UserParamForAsyncBulkCallback::async_bulk_cb(struct libusb_transfer *transfer) {
    if (!transfer->user_data)
        return;

    UserParamForAsyncBulkCallback *param = static_cast<UserParamForAsyncBulkCallback *>(transfer->user_data);
    const bool ret                       = param->proceed_async_bulk(transfer);
    if (!ret) {
        param->stop();
    }
    param->submitted_transfer_ = ret;
}

bool TzBoardDataTransfer::UserParamForAsyncBulkCallback::proceed_async_bulk(struct libusb_transfer *transfer) {
    std::lock_guard<std::mutex> lock(transfer_mutex_);

    assert(transfer == transfer_);
    assert(transfer->buffer == buf_->data());

    if (stop_) {
        return false;
    }

    libusb_data_transfer_.preprocess_transfer(transfer);

    if (transfer->status != LIBUSB_TRANSFER_COMPLETED && transfer->status != LIBUSB_TRANSFER_TIMED_OUT) {
        {
            MV_HAL_LOG_ERROR() << "ErrTransfert";
            MV_HAL_LOG_ERROR() << libusb_error_name(transfer->status);
            if (transfer->status == LIBUSB_TRANSFER_NO_DEVICE) {
                MV_HAL_LOG_ERROR() << "LIBUSB_TRANSFER_NO_DEVICE";
                return false;
            }
        }
        int r = submit_transfer(transfer);
        if (r != 0) {
            MV_HAL_LOG_ERROR() << "Resubmit Error after Error";
            MV_HAL_LOG_ERROR() << libusb_error_name(r);
            return false;
        }
        return true;
    }

    const auto remainder = transfer->actual_length % libusb_data_transfer_.get_raw_event_size_bytes();
    if (remainder) {
        MV_HAL_LOG_WARNING() << "Buffer is not a multiple of a RAW events byte size ("
                             << libusb_data_transfer_.get_raw_event_size_bytes() << "). A RAW event has been dropped.";
    }

    buf_->resize(transfer->actual_length - remainder);
    auto next_buf = libusb_data_transfer_.transfer_data(buf_);

    next_buf->resize(PACKET_SIZE);
    transfer->buffer = next_buf->data();
    buf_             = next_buf;
    int r            = submit_transfer(transfer);
    if (r != 0) {
        MV_HAL_LOG_ERROR() << "Resubmit error after transfer OK";
        MV_HAL_LOG_ERROR() << libusb_error_name(r);
        return false;
    }

    return true;
}

void TzBoardDataTransfer::UserParamForAsyncBulkCallback::start() {
    std::lock_guard<std::mutex> lock(transfer_mutex_); // avoid concurrent access to stop

    submitted_transfer_ = true;
    int r               = submit_transfer(transfer_);
    if (r != 0) {
        MV_HAL_LOG_ERROR() << "Submit error in start";
        MV_HAL_LOG_ERROR() << libusb_error_name(r);
        return;
    }

    stop_ = false;
    ++libusb_data_transfer_.active_bulks_transfers_;
}

void TzBoardDataTransfer::UserParamForAsyncBulkCallback::stop() {
    std::lock_guard<std::mutex> lock(transfer_mutex_);
    if (stop_) {
        return;
    }
    stop_ = true;
    --libusb_data_transfer_.active_bulks_transfers_;
}

void TzBoardDataTransfer::prepare_async_bulk_transfer(libusb_transfer *transfer, unsigned char *buf, int packet_size,
                                                      libusb_transfer_cb_fn async_bulk_cb, void *user_data,
                                                      unsigned int timeout) {
    libusb_fill_bulk_transfer(transfer, cmd_->dev_handle_, cmd_->bEpCommAddress, buf, packet_size, async_bulk_cb,
                              user_data, timeout);
    transfer->flags &= ~LIBUSB_TRANSFER_FREE_BUFFER;
    transfer->flags &= ~LIBUSB_TRANSFER_FREE_TRANSFER;
}

libusb_transfer *TzBoardDataTransfer::contruct_async_bulk_transfer(unsigned char *buf, int packet_size,
                                                                   libusb_transfer_cb_fn async_bulk_cb, void *user_data,
                                                                   unsigned int timeout) {
    if (!cmd_->dev_handle_) {
        return nullptr;
    }
    libusb_transfer *transfer = nullptr;
    transfer                  = libusb_alloc_transfer(0);
    if (!transfer) {
        MV_HAL_LOG_ERROR() << "libusb_alloc_transfer Failed";
        return transfer;
    }
    prepare_async_bulk_transfer(transfer, buf, packet_size, async_bulk_cb, user_data, timeout);
    return transfer;
}

void TzBoardDataTransfer::free_async_bulk_transfer(libusb_transfer *transfer) {
    libusb_free_transfer(transfer);
}

int TzBoardDataTransfer::bulk_transfer(unsigned char *buf, int packet_size, unsigned int timeout, int &actual_size) {
    if (cmd_->dev_handle_) {
        return libusb_bulk_transfer(cmd_->dev_handle_, cmd_->bEpCommAddress, buf, packet_size, &actual_size, 100);
    } else {
        return LIBUSB_ERROR_NO_DEVICE;
    }
}

int TzBoardDataTransfer::submit_transfer(libusb_transfer *transfer) {
    int r = 0;
    r     = libusb_submit_transfer(transfer);
    if (r < 0) {
        MV_HAL_LOG_ERROR() << "USB Submit Error";
    }
    return r;
}

void TzBoardDataTransfer::flush() {
    int bytes_cnt;
    long total_flush = 0;
    int r;
    int flush_max_data = 512 << 10; // 512kB

    MV_HAL_LOG_TRACE() << "Data Transfer: Try to flush";

    do {
        uint8_t buf[16 << 10];
        r = libusb_bulk_transfer(cmd_->dev_handle_, cmd_->bEpCommAddress, buf, 16 << 10, &bytes_cnt, 100);
        if ((r == 0) || (r == LIBUSB_ERROR_TIMEOUT))
            total_flush += bytes_cnt;
        if (total_flush >= flush_max_data) {
            break;
        }
    } while (r == 0 && bytes_cnt > 0);

    MV_HAL_LOG_TRACE() << "Total of " << total_flush << " bytes flushed";
}

} // namespace Metavision
