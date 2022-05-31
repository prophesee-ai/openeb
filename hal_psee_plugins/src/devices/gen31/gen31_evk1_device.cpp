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

#include "utils/register_map.h"
#include "utils/regmap_data.h"
#include "devices/utils/register_maps/common/ccam2_system_monitor_trigger_ext_adc.h"
#include "devices/utils/register_maps/common/ccam3_single_gen3_tep_register_control_registermap.h"
#include "devices/utils/register_maps/common/imu_registermap.h"
#include "devices/utils/register_maps/common/system_config_registermap.h"
#include "devices/gen31/register_maps/ccam3_single_gen31_sensorif_registermap.h"
#include "devices/utils/register_maps/common/stereo_fx3_hostif_registermap.h"

namespace Metavision {

void build_gen31_register_map(RegisterMap &regmap) {
    std::vector<std::tuple<RegmapData *, int, std::string, int>> CCAM3SingleGen31RegisterMap_init = {
        std::make_tuple(CCAM2SystemMonitorTriggerExtADC, CCAM2SystemMonitorTriggerExtADCSize, "SYSTEM_MONITOR", 64),
        std::make_tuple(ccam3_single_gen3_TEPRegisterControlRegisterMap,
                        ccam3_single_sisley_TEPRegisterControlRegisterMapSize, "SYSTEM_CONTROL", 0),
        std::make_tuple(IMURegisterMap, IMURegisterMapSize, "IMU", 6400),
        std::make_tuple(SystemConfigRegisterMap, SystemConfigRegisterMapSize, "SYSTEM_CONFIG", 2048),
        std::make_tuple(ccam3_single_gen31_Gen31SensorIFRegisterMap, ccam3_single_gen31_Gen31SensorIFRegisterMapSize,
                        "SENSOR_IF", 512),
        std::make_tuple(stereo_FX3HostIFRegisterMap, stereo_FX3HostIFRegisterMapSize, "FX3_HOST_IF", 5120),

    };

    init_device_regmap(regmap, CCAM3SingleGen31RegisterMap_init);
}

} // namespace Metavision
