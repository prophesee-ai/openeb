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
#include <sstream>

#ifdef _WIN32
#ifndef _MSC_VER
#define WIN_CALLBACK_DECL __stdcall
#else
#define WIN_CALLBACK_DECL WINAPI
#endif
#else
#define WIN_CALLBACK_DECL
#endif

#include "metavision/hal/utils/data_transfer.h"
#include "metavision/hal/utils/hal_connection_exception.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/boards/utils/psee_libusb_data_transfer.h"

namespace Metavision {

namespace {

constexpr size_t USB_TIME_OUT                = 100;
constexpr size_t N_ASYNC_TRANFERS_PER_DEVICE = 20;
constexpr size_t PACKET_SIZE                 = 128 * 1024;
constexpr uint32_t BULK_NAK_REPORT_THRESHOLD = 100;

size_t get_envar_or_default(const std::string &envvar, size_t default_val) {
    size_t val = default_val;
    try {
        auto *envar_value = getenv(envvar.c_str());
        if (envar_value) {
            std::stringstream ss(envar_value);
            ss >> val;
        }
    } catch (...) {}
    return val;
}

size_t get_async_transfer_number() {
    return get_envar_or_default("MV_PSEE_DEBUG_PLUGIN_USB_ASYNC_TRANSFER", N_ASYNC_TRANFERS_PER_DEVICE);
}

size_t get_packet_size() {
    return get_envar_or_default("MV_PSEE_DEBUG_PLUGIN_USB_PACKET_SIZE", PACKET_SIZE);
}

size_t get_time_out() {
    return get_envar_or_default("MV_PSEE_DEBUG_PLUGIN_USB_TIME_OUT", USB_TIME_OUT);
}

} // namespace

class PseeLibUSBDataTransfer::AsyncTransfer {
    using libusb_free_transfer_fn = void (*)(libusb_transfer *);

public:
    explicit AsyncTransfer(int iso_packets = 0) : transfer_(libusb_alloc_transfer(iso_packets), &libusb_free_transfer) {
        if (!transfer_) {
            throw std::system_error(ENOMEM, std::generic_category(), "Could not allocate libusb_transfer");
        }
        // A transfer is completed as long as it is not queued
        completion_ = 1;
    }
    // Delete copy operators: it would be hard to manage with libusb beneath
    AsyncTransfer(AsyncTransfer const &)                 = delete;
    AsyncTransfer(AsyncTransfer &&) noexcept             = delete;
    AsyncTransfer &operator=(const AsyncTransfer &other) = delete;
    AsyncTransfer const &operator=(AsyncTransfer &&)     = delete;
    ~AsyncTransfer() {
        try {
            cancel();
            wait_completion();
        } catch (const std::exception &e) { MV_HAL_LOG_TRACE() << "Exception in ~AsyncTransfer:" << e.what(); }
    }

    BufferPtr &&get_buf() {
        if (!completed()) {
            throw std::runtime_error("Trying to alter an ongoing transfer");
        }
        buf_->resize(static_cast<DataTransfer::DefaultBufferType::size_type>(transfer_->actual_length));
        return std::move(buf_);
    }
    libusb_transfer_status status() const {
        if (!completed()) {
            throw std::runtime_error("Trying to get the result of an ongoing transfer");
        }
        // This status is only valid at completion time
        return transfer_->status;
    }
    void prepare(std::shared_ptr<LibUSBDevice> dev, EpId endpoint, BufferPtr &&data, unsigned int timeout);
    void submit();
    void wait_completion();
    void cancel();

private:
    // a completion token
    int completion_;
    // Same thing cast as bool for static analysis
    bool completed() const {
        return static_cast<bool>(completion_);
    }
    // The USB device for which the transfer is prepared
    std::shared_ptr<LibUSBDevice> dev_;
    // The memory to be transfered from/to
    BufferPtr buf_;
    // The underlying object for libusb
    std::unique_ptr<libusb_transfer, libusb_free_transfer_fn> transfer_;

    // callback for libusb
    static void WIN_CALLBACK_DECL async_bulk_cb(struct libusb_transfer *transfer);
};

const size_t PseeLibUSBDataTransfer::packet_size_        = get_packet_size();
const size_t PseeLibUSBDataTransfer::async_transfer_num_ = get_async_transfer_number();
const unsigned int PseeLibUSBDataTransfer::timeout_      = static_cast<unsigned int>(get_time_out());

DataTransfer::DefaultBufferPool PseeLibUSBDataTransfer::make_buffer_pool(size_t default_pool_byte_size) {
    auto pool =
        DataTransfer::DefaultBufferPool::make_unbounded(PseeLibUSBDataTransfer::async_transfer_num_, get_packet_size());

    auto buffer_pool_byte_size =
        get_envar_or_default("MV_PSEE_PLUGIN_DATA_TRANSFER_BUFFER_POOL_BYTE_SIZE", default_pool_byte_size);
    if (buffer_pool_byte_size) {
        auto num_obj_pool = buffer_pool_byte_size / packet_size_;
        MV_HAL_LOG_INFO() << "Creating Fixed size data pool of : " << num_obj_pool << "x" << packet_size_ << "B";
        pool = DataTransfer::DefaultBufferPool::make_bounded(num_obj_pool, packet_size_);
    }

    return pool;
}

PseeLibUSBDataTransfer::PseeLibUSBDataTransfer(const std::shared_ptr<LibUSBDevice> dev, EpId endpoint,
                                               uint32_t raw_event_size_bytes,
                                               const DataTransfer::DefaultBufferPool &buffer_pool) :
    buffer_pool_(buffer_pool), dev_(dev), bEpCommAddress_(endpoint), vtransfer_(async_transfer_num_) {}

PseeLibUSBDataTransfer::~PseeLibUSBDataTransfer() {
    // Nothing to do, just need the definition of ~AsyncTransfer where this destructor is
    // defined, to be able to delete vtransfer_
}

void PseeLibUSBDataTransfer::start_impl() {
    // Drop existing URB on device side
    if (auto r = dev_->clear_halt(bEpCommAddress_); r < 0) {
        throw HalConnectionException(r, libusb_error_category());
    }
    auto timeout = timeout_;
    auto shift   = timeout_ / async_transfer_num_;
    // Queue transfers on host side
    for (auto &transfer : vtransfer_) {
        auto buffer = buffer_pool_.acquire();
        buffer->resize(packet_size_);
        transfer.prepare(dev_, bEpCommAddress_, std::move(buffer), timeout);
        transfer.submit();
        // Timeout is counted from submit time
        // Desynchronize the first completions so that completions happen steadily
        timeout += shift;
    }
}

void PseeLibUSBDataTransfer::run_impl(const DataTransfer &data_transfer) {
    MV_HAL_LOG_TRACE() << "poll thread running";
    timeout_cnt_ = 0;
    try {
        while (!data_transfer.should_stop()) {
            run_transfers(data_transfer);
        }
    } catch (const HalConnectionException &e) {
        if (e.code().value() == LIBUSB_TRANSFER_CANCELLED) {
            MV_HAL_LOG_TRACE() << "libusb data transfer was cancelled";
        } else {
            MV_HAL_LOG_TRACE() << "PseeLibUSBDataTransfer: " << e.what();
            stop_impl();
            throw;
        }
    }
    MV_HAL_LOG_TRACE() << "poll thread shutting down";
}

void PseeLibUSBDataTransfer::run_transfers(const DataTransfer &data_producer) {
    // start() queued the transfers, wait for their completion, pass the data to DataTransfer,
    // and requeue an other buffer
    for (auto &transfer : vtransfer_) {
        std::pair<BufferPtr, bool> transfer_res;

        // There is an unlimited wait, but if stop() is called, the transfers will be cancelled,
        // generating an event
        transfer.wait_completion();
        // if we are stopping, there is no point forwarding data and requeuing buffers
        if (data_producer.should_stop()) {
            break;
        }

        // There was an event and we are still streaming, check the transfer status
        switch (transfer.status()) {
        case LIBUSB_TRANSFER_TIMED_OUT: {
            timeout_cnt_++;
            if ((timeout_cnt_ % BULK_NAK_REPORT_THRESHOLD) == 0) {
                MV_HAL_LOG_TRACE() << "\rBulk Transfer NACK " << timeout_cnt_;
            }
            // No data, just requeue the same buffer
            transfer.submit();
            break;
        }
        case LIBUSB_TRANSFER_COMPLETED: {
            timeout_cnt_ = 0;
            // Note: unlike raw libusb, partial transfers are reported as completed
            // Pass the data and get a new buffer
            data_producer.transfer_data(transfer.get_buf());
            // Ensure the buffer can receive a transfer
            auto buff = buffer_pool_.acquire();
            buff->resize(packet_size_);
            // Attach it to the transfer
            transfer.prepare(dev_, bEpCommAddress_, std::move(buff), timeout_);
            // Requeue the transfer with the new buffer
            transfer.submit();
            break;
        }
        default:
            throw HalConnectionException(transfer.status(), libusb_error_category());
        }
    }
}

void PseeLibUSBDataTransfer::stop_impl() {
    int error = LIBUSB_SUCCESS;

    constexpr char message[] = "Could not cancel USB transfer in PseeLibUSBDataTransfer::stop_impl";

    for (auto &transfer : vtransfer_) {
        try {
            transfer.cancel();
            // Here the run_impl thread may catch the completion first, we rely on the fact that cancelled transfers
            // are not re-queued and remain in completed state
            transfer.wait_completion();
        } catch (const HalConnectionException &e) {
            // HAL does not like exceptions here, just leave a warning once
            if (error == LIBUSB_SUCCESS) {
                MV_HAL_LOG_WARNING() << message << e.what();
                error = e.code().value();
            } else {
                MV_HAL_LOG_TRACE() << message << e.what();
            }
        }
    }
    if (error != LIBUSB_SUCCESS) {
        throw HalConnectionException(error, libusb_error_category(), message);
    }
}

void PseeLibUSBDataTransfer::AsyncTransfer::prepare(std::shared_ptr<LibUSBDevice> dev, EpId endpoint, BufferPtr &&data,
                                                    unsigned int timeout) {
    dev_ = dev;
    buf_ = std::move(data);
    libusb_fill_bulk_transfer(transfer_.get(), dev_, endpoint, buf_->data(), static_cast<int>(buf_->size()),
                              async_bulk_cb, this, timeout);
}

void PseeLibUSBDataTransfer::AsyncTransfer::submit() {
    completion_ = 0;

    auto r = libusb_submit_transfer(transfer_.get());
    if (r < 0) {
        MV_HAL_LOG_ERROR() << "USB Submit Error";
        if (r != LIBUSB_ERROR_BUSY) {
            completion_ = 1; // avoid wait on non-submitted transfers
        }
        throw HalConnectionException(r, libusb_error_category());
    }
}

void PseeLibUSBDataTransfer::AsyncTransfer::wait_completion() {
    // libusb forever currently has a 60s timeout
    while (!completed()) {
        auto err = libusb_handle_events_completed(dev_->ctx(), &completion_);
        // No recoverable error has been identified yet, just throw
        if (err) {
            throw HalConnectionException(err, libusb_error_category());
        }
    }
}

void PseeLibUSBDataTransfer::AsyncTransfer::cancel() {
    auto err = libusb_cancel_transfer(transfer_.get());
    // By design, we call cancel on complete and cancelled transfers, no need to throw in that case
    if (err && (err != LIBUSB_ERROR_NOT_FOUND)) {
        throw HalConnectionException(err, libusb_error_category());
    }
}

void WIN_CALLBACK_DECL PseeLibUSBDataTransfer::AsyncTransfer::async_bulk_cb(struct libusb_transfer *transfer) {
    auto *xfer = reinterpret_cast<AsyncTransfer *>(transfer->user_data);

    xfer->completion_ = 1;
    // Change a bit the semantic with regard to libusb:
    // If a transfer times out but has data, reports it as completed
    if ((transfer->status == LIBUSB_TRANSFER_TIMED_OUT) && (transfer->actual_length != 0)) {
        transfer->status = LIBUSB_TRANSFER_COMPLETED;
    }
}

} // namespace Metavision
