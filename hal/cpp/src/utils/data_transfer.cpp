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

#include <future>
#include "metavision/hal/utils/hal_connection_exception.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/data_transfer.h"

namespace Metavision {

DataTransfer::DataTransfer(RawDataProducerPtr data_producer_ptr) : data_producer_ptr_(data_producer_ptr) {}

DataTransfer::~DataTransfer() {
    stop();
}

void DataTransfer::start() {
    if (run_transfers_thread_.joinable()) {
        if (stop_) {
            // if the transfer stopped by itself (e.g end of file reached)
            // stop may never have been called so we have to join the thread first
            run_transfers_thread_.join();
        } else {
            return;
        }
    }

    {
        std::lock(suspend_mutex_, running_mutex_);
        std::unique_lock<std::mutex> lock1(suspend_mutex_, std::adopt_lock);
        std::unique_lock<std::mutex> lock2(running_mutex_, std::adopt_lock);
        stop_    = false;
        running_ = false;
    }

    data_producer_ptr_->start_impl();

    std::promise<void> thread_is_started;
    std::future<void> has_started = thread_is_started.get_future();

    run_transfers_thread_ = std::thread([this, &thread_is_started]() {
        thread_is_started.set_value();
        for (auto cb : status_change_cbs_) {
            cb.second(Status::Started);
        }

        try {
            while (!stop_) {
                {
                    std::unique_lock<std::mutex> lock(suspend_mutex_);
                    suspend_ = false;
                }
                {
                    std::unique_lock<std::mutex> lock(running_mutex_);
                    running_ = true;
                }

                data_producer_ptr_->run_impl(*this);

                if (!suspend_) {
                    break;
                } else {
                    {
                        std::unique_lock<std::mutex> lock(running_mutex_);
                        running_ = false;
                    }
                    running_cond_.notify_all();

                    std::unique_lock<std::mutex> lock(suspend_mutex_);
                    suspend_cond_.wait(lock, [this] { return !suspend_ || stop_; });
                }
            }
        } catch (const std::exception &) {
            for (auto cb : transfer_error_cbs_) {
                cb.second(std::current_exception());
            }
        }

        notify_stop();

        for (auto cb : status_change_cbs_) {
            cb.second(Status::Stopped);
        }
    });

    has_started.wait();
}

void DataTransfer::stop() {
    if (!run_transfers_thread_.joinable()) {
        return;
    }

    try {
        data_producer_ptr_->stop_impl();
    } catch (const HalConnectionException &) {
        notify_stop();
        // We can't be sure the thread will terminate
        run_transfers_thread_.detach();
        throw;
    }

    notify_stop();
    run_transfers_thread_.join();
}

void DataTransfer::notify_stop() {
    {
        std::lock(suspend_mutex_, running_mutex_);
        std::unique_lock<std::mutex> lock1(suspend_mutex_, std::adopt_lock);
        std::unique_lock<std::mutex> lock2(running_mutex_, std::adopt_lock);
        stop_ = true;
    }
    suspend_cond_.notify_all();
    running_cond_.notify_all();
}

void DataTransfer::suspend() {
    {
        std::unique_lock<std::mutex> lock(suspend_mutex_);
        suspend_ = true;
    }

    std::unique_lock<std::mutex> lock(running_mutex_);
    running_cond_.wait(lock, [this] { return !running_ || stop_; });
}

void DataTransfer::resume() {
    {
        std::unique_lock<std::mutex> lock(suspend_mutex_);
        suspend_ = false;
    }
    suspend_cond_.notify_all();
}

size_t DataTransfer::add_status_changed_callback(StatusChangeCallback_t cb) {
    status_change_cbs_[cb_index_] = cb;
    auto ret                      = cb_index_;
    ++cb_index_;
    return ret;
}

size_t DataTransfer::add_new_buffer_callback(NewBufferCallback_t cb) {
    new_buffer_cbs_[cb_index_] = cb;
    auto ret                   = cb_index_;
    ++cb_index_;
    return ret;
}

size_t DataTransfer::add_transfer_error_callback(TransferErrorCallback_t cb) {
    transfer_error_cbs_[cb_index_] = cb;
    auto ret                       = cb_index_;
    ++cb_index_;
    return ret;
}

void DataTransfer::remove_callback(size_t cb_id) {
    status_change_cbs_.erase(cb_id);
    new_buffer_cbs_.erase(cb_id);
    transfer_error_cbs_.erase(cb_id);
}

bool DataTransfer::should_stop() const {
    return stop_ || suspend_;
}

bool DataTransfer::stopped() const {
    return stop_;
}

void DataTransfer::fire_callbacks(const BufferPtr &buffer) const {
    for (auto &cb : new_buffer_cbs_) {
        cb.second(buffer);
    }
}

DataTransfer::BufferPtr::BufferPtr(std::any buffer, PtrType data, std::size_t buffer_size) :
    internal_buffer_(buffer), buffer_data_(data), buffer_size_(buffer_size) {}

bool DataTransfer::BufferPtr::operator==(const BufferPtr &other) const {
    if (other.buffer_size_ != buffer_size_) {
        return false;
    }

    return std::equal(begin(), end(), other.begin());
}

DataTransfer::BufferPtr::operator bool() const noexcept {
    return buffer_data_ != nullptr;
}

std::size_t DataTransfer::BufferPtr::size() const noexcept {
    return buffer_size_;
}

DataTransfer::BufferPtr::PtrType DataTransfer::BufferPtr::data() const noexcept {
    return buffer_data_;
}

DataTransfer::BufferPtr::PtrType DataTransfer::BufferPtr::begin() const noexcept {
    return buffer_data_;
}

DataTransfer::BufferPtr::PtrType DataTransfer::BufferPtr::end() const noexcept {
    return buffer_data_ + buffer_size_;
}

const DataTransfer::BufferPtr::PtrType DataTransfer::BufferPtr::cbegin() const noexcept {
    return begin();
}

const DataTransfer::BufferPtr::PtrType DataTransfer::BufferPtr::cend() const noexcept {
    return end();
}

DataTransfer::BufferPtr DataTransfer::BufferPtr::clone() const {
    auto new_buffer = std::make_shared<CloneType>();
    new_buffer->reserve(buffer_size_);

    auto from = buffer_data_;
    auto end  = buffer_data_ + buffer_size_;
    std::copy(from, end, std::back_inserter(*new_buffer));

    return make_buffer_ptr(new_buffer);
}

DataTransfer::BufferPtr::SharedCloneType DataTransfer::BufferPtr::any_clone_cast() const {
    return std::any_cast<SharedCloneType>(internal_buffer_);
}

void DataTransfer::BufferPtr::reset() noexcept {
    internal_buffer_ = {};
    buffer_data_     = nullptr;
    buffer_size_     = 0;
}
} // namespace Metavision
