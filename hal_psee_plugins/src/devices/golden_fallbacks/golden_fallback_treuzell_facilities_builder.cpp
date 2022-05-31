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

#include "devices/golden_fallbacks/golden_fallback_treuzell_facilities_builder.h"
#include "metavision/hal/utils/device_config.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/device_builder.h"

namespace Metavision {

bool build_golden_fallback_treuzell_device(DeviceBuilder &device_builder,
                                           const DeviceBuilderParameters &device_builder_params,
                                           const DeviceConfig &device_config) {
    throw HalException(
        HalErrorCode::GoldenFallbackBooted,
        "The FPGA seems to be in an invalid state, contact the support for help at support@prophesee.ai.");
}

} // namespace Metavision
