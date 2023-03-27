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

#ifndef METAVISION_HAL_DECODER_PROTOCOL_VIOLATION_H
#define METAVISION_HAL_DECODER_PROTOCOL_VIOLATION_H

#include <string>
#include <unordered_map>

#include "metavision/hal/utils/hal_error_code.h"

namespace Metavision {

enum DecoderProtocolViolation : HalErrorCodeType {
    NullProtocolViolation = 0,
    NonMonotonicTimeHigh,
    PartialVect_12_12_8,
    PartialContinued_12_12_4,
    NonContinuousTimeHigh,
    MissingYAddr,
    InvalidVectBase,
};

inline std::ostream &operator<<(std::ostream &o, const DecoderProtocolViolation protocol_violation) {
    static const std::unordered_map<DecoderProtocolViolation, std::string> protocol_violation_to_str = {
        {DecoderProtocolViolation::NullProtocolViolation, "NullProtocolViolation"},
        {DecoderProtocolViolation::NonMonotonicTimeHigh, "NonMonotonicTimeHigh"},
        {DecoderProtocolViolation::PartialVect_12_12_8, "PartialVect_12_12_8"},
        {DecoderProtocolViolation::PartialContinued_12_12_4, "PartialContinued_12_12_4"},
        {DecoderProtocolViolation::NonContinuousTimeHigh, "NonContinuousTimeHigh"},
        {DecoderProtocolViolation::MissingYAddr, "MissingYAddr"},
        {DecoderProtocolViolation::InvalidVectBase, "InvalidVectBase"},

    };

    o << protocol_violation_to_str.at(protocol_violation);
    return o;
}

} // namespace Metavision

#endif // METAVISION_HAL_DECODER_PROTOCOL_VIOLATION_H
