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

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include "devices/treuzell/tz_psee_video.h"
#include "devices/gen31/gen31_evk2_tz_device.h"
#ifdef HAL_GEN4_SUPPORT
#include "devices/gen4/gen4_evk2_tz_device.h"
#endif
#include "devices/gen41/gen41_evk2_tz_device.h"
#include "devices/imx636/imx636_evk2_tz_device.h"
#include "devices/utils/device_system_id.h"
#include "boards/treuzell/tz_libusb_board_command.h"
#include "devices/treuzell/tz_device.h"
#include "boards/treuzell/tz_control_frame.h"
#include "metavision/hal/utils/device_builder.h"

namespace Metavision {

std::shared_ptr<TzDevice> TzPseeVideo::build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id,
                                             std::shared_ptr<TzDevice> parent) {
    switch (cmd->read_device_register(dev_id, 0x800)[0]) {
    case SYSTEM_EVK2_GEN31:
        return std::make_shared<TzEvk2Gen31>(cmd, dev_id, parent);
#ifdef HAL_GEN4_SUPPORT
    case SYSTEM_EVK2_GEN4:
        return std::make_shared<TzEvk2Gen4>(cmd, dev_id, parent);
#endif
    case SYSTEM_EVK2_GEN41:
        return std::make_shared<TzEvk2Gen41>(cmd, dev_id, parent);
    case SYSTEM_EVK2_IMX636:
        return std::make_shared<TzEvk2Imx636>(cmd, dev_id, parent);
    default:
        return std::make_shared<TzPseeVideo>(cmd, dev_id, parent);
    }
}

} // namespace Metavision
