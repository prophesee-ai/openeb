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

#ifndef METAVISION_HAL_TEST_DUMMY_DATATRANSFER_H
#define METAVISION_HAL_TEST_DUMMY_DATATRANSFER_H

// This DataTransfer will transfer 255 values an incremental counter on buffers doubling length at each iteration
// as follows: {0}, {1, 2}, {3, 4, 5, 6}, {7, 8, 9, 10, 11, 12, 13, 14}, ...
// The DummyAllocator has no specific behavior and is just meant to check DataTransfer::Allocator properties

struct DummyDataTransfer : public Metavision::DataTransfer {
    using Allocator = Metavision::DataTransfer::Allocator;
    class DummyAllocator : public Allocator::DefaultAllocator {};

    DummyDataTransfer() :
        Metavision::DataTransfer(1, BufferPool::make_bounded(4, Allocator(Allocator::ImplPtr(new DummyAllocator())))) {}

    virtual void run_impl() override {
        auto buff   = get_buffer();
        int size    = 1;
        int counter = 0;
        // On the first buffer we put a known data (42) to be able to check that resize does not erase it.
        buff->reserve(128);
        *(buff->data() + 1) = 42;
        for (int i = 0; (i < 8) && !should_stop(); i++) {
            buff->resize(size);
            for (auto &data : *buff) {
                data = counter++;
            }
            auto [next, dropped] = transfer_data(buff);
            size *= 2;
            buff = next;
        }
    }
};

#endif // METAVISION_HAL_TEST_DUMMY_DATATRANSFER_H
