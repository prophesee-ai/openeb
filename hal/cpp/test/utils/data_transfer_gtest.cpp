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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <numeric>
#include <memory>
#include <sys/types.h>

#include "metavision/hal/utils/data_transfer.h"
#include "metavision/sdk/base/utils/object_pool.h"

using namespace Metavision;
using testing::_;
using testing::AnyNumber;

class MockRawDataProducer : public DataTransfer::RawDataProducer {
public:
    MOCK_METHOD(void, start_impl, (), (override));
    MOCK_METHOD(void, run_impl, (const DataTransfer &), (override));
    MOCK_METHOD(void, stop_impl, (), (override));
};

TEST(DataTransfer, should_have_default_data_transfer_constructor) {
    auto data_prod = std::make_shared<testing::StrictMock<MockRawDataProducer>>();

    DataTransfer data_transfer(data_prod);

    EXPECT_CALL(*data_prod, start_impl()).Times(0);
    EXPECT_CALL(*data_prod, run_impl(_)).Times(0);
    EXPECT_CALL(*data_prod, stop_impl()).Times(0);
}

TEST(DataTransfer, should_stop_without_explicit_call_to_stop) {
    auto data_prod = std::make_shared<MockRawDataProducer>();
    DataTransfer data_transfer(data_prod);

    EXPECT_CALL(*data_prod, start_impl()).Times(1);
    EXPECT_CALL(*data_prod, run_impl(_)).Times(AnyNumber());
    EXPECT_CALL(*data_prod, stop_impl()).Times(1);

    data_transfer.start();
}

TEST(DataTransfer, should_stop_with_explicit_call_to_stop) {
    auto data_prod = std::make_shared<MockRawDataProducer>();

    DataTransfer data_transfer(data_prod);

    EXPECT_CALL(*data_prod, start_impl()).Times(1);
    EXPECT_CALL(*data_prod, run_impl(_)).Times(AnyNumber());
    EXPECT_CALL(*data_prod, stop_impl()).Times(1);

    data_transfer.start();
    data_transfer.stop();
}

TEST(DataTransferBufferPtr, should_default_construct) {
    DataTransfer::BufferPtr bufferptr;
    EXPECT_FALSE(bufferptr);
    EXPECT_EQ(bufferptr.data(), nullptr);
    EXPECT_EQ(bufferptr.size(), 0);
}

TEST(DataTransferBufferPtr, buffer_ptr_should_hold_reference_count_from_shared_ptr) {
    auto vec_buff = std::make_shared<std::vector<uint8_t>>();

    EXPECT_EQ(vec_buff.use_count(), 1);
    {
        DataTransfer::BufferPtr bufferptr = DataTransfer::make_buffer_ptr(vec_buff);
        EXPECT_EQ(vec_buff.use_count(), 2);
    }
    EXPECT_EQ(vec_buff.use_count(), 1);
}

TEST(DataTransferBufferPtr, should_release_reference_count_on_reset) {
    auto vec_buff                     = std::make_shared<std::vector<uint8_t>>();
    DataTransfer::BufferPtr bufferptr = DataTransfer::make_buffer_ptr(vec_buff);

    EXPECT_EQ(vec_buff.use_count(), 2);
    bufferptr.reset();
    EXPECT_EQ(vec_buff.use_count(), 1);
}

TEST(DataTransferBufferPtr, buffer_ptr_should_have_range_semantic) {
    auto vec_buff = std::make_shared<std::vector<uint8_t>>();
    vec_buff->insert(vec_buff->begin(), {1, 2, 3});

    DataTransfer::BufferPtr buff_ptr = DataTransfer::make_buffer_ptr(vec_buff);

    auto sum = std::reduce(buff_ptr.begin(), buff_ptr.end());
    EXPECT_EQ(sum, 6);

    auto const_data_sum = std::reduce(buff_ptr.cbegin(), buff_ptr.cend());
    EXPECT_EQ(const_data_sum, 6);
}

TEST(DataTransferBufferPtr, should_construct_BufferPtr_from_pool) {
    using StringPool   = SharedObjectPool<std::string>;
    using StringBuffer = StringPool::ptr_type;

    StringPool pool          = StringPool::make_unbounded(1, "Hello World");
    StringBuffer string_buff = pool.acquire();

    auto data_prod = std::make_shared<MockRawDataProducer>();
    DataTransfer data_transfer(data_prod);

    DataTransfer::BufferPtr buffer_ptr;
    data_transfer.add_new_buffer_callback([&](auto buff) { buffer_ptr = buff; });

    data_transfer.transfer_data(string_buff);

    EXPECT_EQ(buffer_ptr.size(), buffer_ptr.size());
    EXPECT_EQ(reinterpret_cast<const char *>(buffer_ptr.data()), string_buff->data());
    EXPECT_STREQ(reinterpret_cast<const char *>(buffer_ptr.data()), "Hello World");
}

TEST(DataTransferBufferPtr, should_clone_buffer_ptr) {
    auto vec_buff = std::make_shared<std::vector<uint8_t>>();
    vec_buff->insert(vec_buff->begin(), {1, 2, 3});

    DataTransfer::BufferPtr buff_ptr = DataTransfer::make_buffer_ptr(vec_buff);
    auto cloned_buff_ptr             = buff_ptr.clone();

    EXPECT_NE(cloned_buff_ptr.data(), buff_ptr.data()) << "Underlying data should differ";

    auto cloned_vec = cloned_buff_ptr.any_clone_cast();
    EXPECT_THAT(*cloned_vec, testing::ContainerEq(*vec_buff)) << "Items should have been copied";
}

TEST(DataTransferBufferPtr, should_throw_on_casting_back_non_cloned_buffer) {
    auto vec_of_char = std::make_shared<std::vector<char>>();
    vec_of_char->insert(vec_of_char->begin(), {'a', 'b', 'c'});

    DataTransfer::BufferPtr buff_ptr = DataTransfer::make_buffer_ptr(vec_of_char);
    EXPECT_THROW(buff_ptr.any_clone_cast(), std::bad_cast);
}
