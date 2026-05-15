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

#ifndef METAVISION_HAL_PSEE_PLUGINGS_TEST_I_HW_REGISTER_MOCK
#define METAVISION_HAL_PSEE_PLUGINGS_TEST_I_HW_REGISTER_MOCK

#include <gmock/gmock.h>
#include "metavision/hal/facilities/i_hw_register.h"

namespace Metavision {
namespace Test {

class HW_Register_Mock : public Metavision::I_HW_Register {
public:
    HW_Register_Mock() {}
    HW_Register_Mock(const HW_Register_Mock &lhs) {}

    MOCK_METHOD(void, write_register, (uint32_t address, uint32_t v), (override));
    MOCK_METHOD(void, write_register, (const std::string &address, uint32_t v), (override));
    MOCK_METHOD(void, write_register, (const std::string &address, const std::string &bitfield, uint32_t v),
                (override));
    MOCK_METHOD(uint32_t, read_register, (const std::string &address, const std::string &bitfield), (override));
    MOCK_METHOD(uint32_t, read_register, (uint32_t address), (override));
    MOCK_METHOD(uint32_t, read_register, (const std::string &address), (override));
};

using HW_Register_Nice_Mock = ::testing::NiceMock<HW_Register_Mock>;

} // namespace Test
} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINGS_TEST_I_HW_REGISTER_MOCK
