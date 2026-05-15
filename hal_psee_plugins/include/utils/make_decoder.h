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

#ifndef MAKE_DECODER_H
#define MAKE_DECODER_H

#include <memory>
#include "metavision/hal/utils/device_config.h"

namespace Metavision {

class StreamFormat;
class I_EventsStreamDecoder;
class DeviceBuilder;

/* Based on the stream format, this fonction will instantiate decoders and record them in the device builder.
 * If a I_EventsStreamDecoder was created, the function will return a shared pointer to a registered
 * I_EventsStreamDecoder, and nullptr if only other types of decoders were registered. In both case, raw_size_bytes
 * will be set. If the format has a geometry, the I_Geometry facility will also be created.
 * If the provided fromat is not handled, the function will throw.
 */
std::shared_ptr<I_EventsStreamDecoder> make_decoder(DeviceBuilder &, const StreamFormat &, size_t &raw_size_bytes,
                                                    bool do_time_shifting, const Metavision::DeviceConfig &config = Metavision::DeviceConfig{});

} // namespace Metavision
#endif /* MAKE_DECODER_H */
