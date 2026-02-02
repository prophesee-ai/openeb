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

#ifndef METAVISION_SDK_STREAM_CALLBACK_TAG_IDS_H
#define METAVISION_SDK_STREAM_CALLBACK_TAG_IDS_H

#include <cstdint>

namespace Metavision {
namespace CallbackTagIds {

static constexpr std::uint8_t DECODE_CALLBACK_TAG_ID = 0;
static constexpr std::uint8_t RAW_CALLBACK_TAG_ID    = DECODE_CALLBACK_TAG_ID + 1;
static constexpr std::uint8_t READ_CALLBACK_TAG_ID   = DECODE_CALLBACK_TAG_ID + 2;
static constexpr std::uint8_t SEEK_CALLBACK_TAG_ID   = DECODE_CALLBACK_TAG_ID + 3;

} // namespace CallbackTagIds
} // namespace Metavision

#endif // METAVISION_SDK_STREAM_CALLBACK_TAG_IDS_H
