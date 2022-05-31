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

#include "metavision/hal/utils/hal_log.h"
#include "devices/gen31/gen31_fx3_device_control.h"
#include "boards/utils/psee_libusb_board_command.h"
#include "devices/gen31/gen31_evk1_fpga.h"
#include "devices/gen31/gen31_sensor.h"
#include "utils/register_map.h"

namespace Metavision {
namespace {
std::string SENSOR_PREFIX = "SENSOR_IF/GEN31/";
} // namespace

Gen31Fx3DeviceControl::Gen31Fx3DeviceControl(const std::shared_ptr<RegisterMap> &register_map) :
    Gen31DeviceControl(register_map, std::make_shared<Gen31Evk1Fpga>(register_map, is_gen31EM(*register_map)),
                       std::make_shared<Gen31Sensor>(register_map, SENSOR_PREFIX, is_gen31EM(*register_map))) {}

void Gen31Fx3DeviceControl::enable_interface(bool state) {
    (*register_map_)["SYSTEM_CONTROL/CCAM2_CONTROL"]["HOST_IF_EN"] = state;
}

void Gen31Fx3DeviceControl::start_impl() {
    bool gen31EM = is_gen31EM();
    start_camera_common(gen31EM);
}

void Gen31Fx3DeviceControl::stop_impl() {
    stop_camera_common();
}

std::string Gen31Fx3DeviceControl::get_sensor_prefix() const {
    return SENSOR_PREFIX;
}

void Gen31Fx3DeviceControl::initialize() {
    Gen31DeviceControl::initialize();
    destroy_camera();
    fpga_init();
    sensor_init();
}

void Gen31Fx3DeviceControl::destroy() {
    destroy_camera();
    Gen31DeviceControl::destroy();
}

void Gen31Fx3DeviceControl::reset_ts_internal() {}

} // namespace Metavision
