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

#include "sample_data_transfer.h"
#include "internal/sample_data_transfer_pattern_generator.h"

constexpr short SampleDataTransfer::PatternGenerator::SIZE_SQUARE;
constexpr short SampleDataTransfer::PatternGenerator::N_RANDOM;
constexpr Metavision::timestamp SampleDataTransfer::PatternGenerator::STEP_RANDOM;

SampleDataTransfer::SampleDataTransfer(uint32_t raw_event_size_bytes) :
    DataTransfer(raw_event_size_bytes), current_time_(0), gen_(new SampleDataTransfer::PatternGenerator()) {}

SampleDataTransfer::~SampleDataTransfer() = default;

void SampleDataTransfer::start_impl(BufferPtr buffer) {
    buffer_ = buffer;
}

void SampleDataTransfer::run_impl() {
    // This is the method that you'll need to implement when writing your own plugin.
    // In this sample we just generate some fake events, but you'll have to replace
    // the following code with your implementation (for example, getting data from
    // the USB)
    static constexpr size_t SIZE_FAKE_EVENTS = 340 * sizeof(SampleEventsFormat);

    // To generate real time events, we need to time how long it takes to fill the
    // events, and sleep for the necessary time
    Metavision::timestamp time_start = current_time_;
    uint64_t first_ts_clock_         = Metavision::get_system_time_us();
    while (!should_stop()) {
        // Fill fake_events
        buffer_->resize(SIZE_FAKE_EVENTS);
        for (auto it = buffer_->begin(); it < buffer_->end(); it += sizeof(SampleEventsFormat)) {
            (*gen_)(reinterpret_cast<SampleEventsFormat &>(*it), current_time_);
        }

        // Makes the buffer available to the events stream
        auto next_buffer      = transfer_data(buffer_);
        buffer_               = next_buffer.first;
        uint64_t cur_ts_clock = Metavision::get_system_time_us();
        uint64_t expected_ts  = first_ts_clock_ + (current_time_ - time_start);
        if (expected_ts > cur_ts_clock) {
            std::this_thread::sleep_for(std::chrono::microseconds(expected_ts - cur_ts_clock));
        }
    }
}
