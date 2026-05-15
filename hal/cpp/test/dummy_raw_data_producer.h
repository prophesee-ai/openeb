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

#ifndef METAVISION_HAL_TEST_DUMMY_RAW_DATA_PRODUCER_H
#define METAVISION_HAL_TEST_DUMMY_RAW_DATA_PRODUCER_H

#include <cstdint>
#include <vector>

#include "metavision/hal/utils/data_transfer.h"
#include "metavision/sdk/base/utils/object_pool.h"

// This DataTransfer will transfer 255 values an incremental counter on buffers doubling length at each iteration
// as follows: {0}, {1, 2}, {3, 4, 5, 6}, {7, 8, 9, 10, 11, 12, 13, 14}, ...
// The DummyAllocator has no specific behavior and is just meant to check DataTransfer::Allocator properties

using custom_vector = std::vector<uint8_t>;

struct DummyRawDataProducer : public Metavision::DataTransfer::RawDataProducer {
    using Pool = Metavision::SharedObjectPool<std::vector<uint8_t>>;
    Pool pool;

    DummyRawDataProducer() : pool(Pool::make_bounded(4)) {}

    virtual void run_impl(const Metavision::DataTransfer &data_transfer) override {
        auto buff   = pool.acquire();
        int size    = 1;
        int counter = 0;

        buff->reserve(128);

        for (int i = 0; (i < 8) && !data_transfer.should_stop(); i++) {
            buff->resize(size);
            for (auto &data : *buff) {
                data = counter++;
            }
            data_transfer.transfer_data(buff);
            size *= 2;
            buff = pool.acquire();
        }
    }
};

#endif // METAVISION_HAL_TEST_DUMMY_RAW_DATA_PRODUCER_H
