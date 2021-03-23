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

#include <memory>

#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/utils/hal_error_code.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

I_EventsStream::I_EventsStream(std::unique_ptr<DataTransfer> data_transfer,
                               const std::shared_ptr<I_HW_Identification> &hw_identification) :
    data_transfer_(std::move(data_transfer)), hw_identification_(hw_identification), stop_(true) {
    if (!hw_identification_) {
        throw(HalException(HalErrorCode::FailedInitialization, "HW identification facility is null."));
    }
    data_transfer_->add_new_buffer_callback([this](const DataTransfer::BufferPtr &buffer) {
        std::lock_guard<std::mutex> lock(new_buffer_safety_);
        if (!stop_) {
            available_buffers_.push(buffer);
            new_buffer_cond_.notify_all();
        }
    });

    data_transfer_->add_status_changed_callback([this](DataTransfer::Status status) {
        if (status == DataTransfer::Status::Stopped) {
            std::lock_guard<std::mutex> lock(new_buffer_safety_);
            stop_ = true;
            new_buffer_cond_.notify_all();
        }
    });
}

I_EventsStream::~I_EventsStream() {
    stop();
    data_transfer_.reset(nullptr);
}

void I_EventsStream::start() {
    std::lock_guard<std::mutex> lock(start_stop_safety_);
    {
        std::lock_guard<std::mutex> lock(new_buffer_safety_);
        stop_ = false;
    }
    data_transfer_->start();
}

void I_EventsStream::stop() {
    std::lock_guard<std::mutex> lock(start_stop_safety_);
    {
        std::lock_guard<std::mutex> lock(new_buffer_safety_);
        available_buffers_ = {};
        returned_buffer_.reset();
        stop_ = true;
        new_buffer_cond_.notify_all();
        stop_log_raw_data();
    }
    data_transfer_->stop();
}

short I_EventsStream::poll_buffer() {
    std::lock_guard<std::mutex> lock(new_buffer_safety_);
    if (!available_buffers_.empty()) {
        return 1;
    }

    return stop_ ? -1 : 0;
}

short I_EventsStream::wait_next_buffer() {
    std::unique_lock<std::mutex> lock(new_buffer_safety_);
    new_buffer_cond_.wait(lock, [this]() { return !available_buffers_.empty() || stop_; });

    return available_buffers_.empty() ? -1 : 1;
}

I_EventsStream::RawData *I_EventsStream::get_latest_raw_data(long &size) {
    std::lock_guard<std::mutex> lock(new_buffer_safety_);

    if (available_buffers_.empty()) {
        // If no new buffer available yet
        size = 0;
        return nullptr;
    }

    // Keep a reference to returned buffer to ensure validity until next call to this function
    returned_buffer_ = available_buffers_.front();
    size             = returned_buffer_->size();
    available_buffers_.pop();

    std::lock_guard<std::mutex> log_lock(log_raw_safety_);
    if (log_raw_data_) {
        log_raw_data_->write(reinterpret_cast<char *>(returned_buffer_->data()), size * sizeof(RawData));
    }
    return returned_buffer_->data();
}

void I_EventsStream::stop_log_raw_data() {
    std::lock_guard<std::mutex> guard(log_raw_safety_);
    log_raw_data_.reset(nullptr);
}

bool I_EventsStream::log_raw_data(const std::string &f) {
    if (f == underlying_filename_) {
        return false;
    }

    auto header = hw_identification_->get_header();
    header.add_date();

    std::lock_guard<std::mutex> guard(log_raw_safety_);
    log_raw_data_.reset(new std::ofstream(f, std::ios::binary));
    if (!log_raw_data_->is_open()) {
        log_raw_data_ = nullptr;
        return false;
    }

    (*log_raw_data_) << header;
    return true;
}

void I_EventsStream::set_underlying_filename(const std::string &filename) {
    underlying_filename_ = filename;
}
} // namespace Metavision
