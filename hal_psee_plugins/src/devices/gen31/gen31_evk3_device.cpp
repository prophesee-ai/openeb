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

namespace Metavision {
void build_gen31_evk3_register_map(RegisterMap &regmap) {
#include "devices/gen31/register_maps/ccam5_single_gen31_system_control_registermap.h"
#include "devices/utils/register_maps/common/ccam2_system_monitor_trigger_ext_adc.h"
#include "devices/utils/register_maps/common/system_config_registermap.h"
#include "devices/gen31/register_maps/ccam3_single_gen31_sensorif_registermap.h"
#include "devices/utils/register_maps/common/mipitx_registermap.h"
#include "devices/utils/register_maps/common/spi_flash_master_registermap.h"

    std::vector<std::tuple<RegmapData *, int, std::string, int>> CCAM5SingleGen31RegisterMap_init = {
        std::make_tuple(ccam5_single_gen31_SystemControlRegisterMap, ccam5_single_gen31_SystemControlRegisterMapSize,
                        "SYSTEM_CONTROL", 0),
        std::make_tuple(CCAM2SystemMonitorTriggerExtADC, CCAM2SystemMonitorTriggerExtADCSize, "SYSTEM_MONITOR", 0x40),

        std::make_tuple(ccam3_single_gen31_Gen31SensorIFRegisterMap, ccam3_single_gen31_Gen31SensorIFRegisterMapSize,
                        "SENSOR_IF", 0x200),
        std::make_tuple(SystemConfigRegisterMap, SystemConfigRegisterMapSize, "SYSTEM_CONFIG", 0x800),
        std::make_tuple(MIPITXRegisterMap, MIPITXRegisterMapSize, "MIPI_TX", 0x1500),
        std::make_tuple(SPIFlashMasterRegisterMap, SPIFlashMasterRegisterMapSize, "FLASH", 0x1600),

    };

    init_device_regmap(regmap, CCAM5SingleGen31RegisterMap_init);
}
} // namespace Metavision
