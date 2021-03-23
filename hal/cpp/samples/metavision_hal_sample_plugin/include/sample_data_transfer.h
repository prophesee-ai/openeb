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

#ifndef METAVISION_HAL_SAMPLE_DATA_TRANSFER_H
#define METAVISION_HAL_SAMPLE_DATA_TRANSFER_H

#include <memory>

#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/hal/utils/data_transfer.h>

/// @brief Class for getting buffers from cameras or files.
///
/// This class is the implementation of HAL's facility @ref Metavision::DataTransfer
class SampleDataTransfer : public Metavision::DataTransfer {
public:
    /// @brief Constructor
    ///
    /// @param raw_event_size_bytes The size of a RAW event in bytes
    SampleDataTransfer(uint32_t raw_event_size_bytes);

    /// @brief Destructor
    ~SampleDataTransfer() override;

private:
    void start_impl(BufferPtr buffer) override final;
    void run_impl() override final;

    Metavision::timestamp current_time_;
    BufferPtr buffer_;

    struct PatternGenerator;
    std::unique_ptr<PatternGenerator> gen_;
};

#endif // METAVISION_HAL_SAMPLE_DATA_TRANSFER_H
