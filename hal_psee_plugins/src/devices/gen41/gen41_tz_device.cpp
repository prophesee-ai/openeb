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

#include <math.h>
#include <thread>
#include <chrono>

#include "devices/gen41/gen41_tz_device.h"
#include "devices/utils/device_system_id.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "devices/treuzell/tz_device_builder.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "devices/common/issd.h"
#include "devices/gen41/gen41_evk3_issd.h"
#include "metavision/psee_hw_layer/devices/common/antiflicker_filter.h"
#include "metavision/psee_hw_layer/devices/common/event_trail_filter.h"
#include "metavision/psee_hw_layer/devices/gen41/gen41_digital_event_mask.h"
#include "metavision/psee_hw_layer/devices/gen41/gen41_digital_crop.h"
#include "metavision/psee_hw_layer/devices/gen41/gen41_erc.h"
#include "metavision/psee_hw_layer/devices/gen41/gen41_ll_biases.h"
#include "metavision/psee_hw_layer/devices/gen41/gen41_roi_command.h"
#include "devices/gen41/register_maps/gen41_registermap.h"
#include "metavision/psee_hw_layer/devices/gen41/gen41_tz_trigger_event.h"
#include "metavision/psee_hw_layer/facilities/psee_hw_register.h"
#include "geometries/hd_geometry.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/device_builder.h"
#include "plugin/psee_plugin.h"
#include "utils/psee_hal_utils.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {
namespace {
std::string ROOT_PREFIX   = "PSEE/GEN41/";
std::string SENSOR_PREFIX = "";
} // namespace

TzGen41::TzGen41(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent),
    TzIssdDevice(gen41_evk3_issd),
    TzDeviceWithRegmap(Gen41RegisterMap, Gen41RegisterMapSize, ROOT_PREFIX) {
    sync_mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
    iph_mirror_control(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    lifo_control(true, true, true);
}

std::shared_ptr<TzDevice> TzGen41::build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                         std::shared_ptr<TzDevice> parent) {
    if (can_build(cmd, dev_id)) {
        return std::make_shared<TzGen41>(cmd, dev_id, parent);
    } else {
        return nullptr;
    }
}
static TzRegisterBuildMethod method("psee,ccam5_gen41", TzGen41::build, TzGen41::can_build);

bool TzGen41::can_build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id) {
    auto ret = cmd->read_device_register(dev_id, 0x14)[0];
    return (ret == 0xA0301003 || ret == 0xA0301002);
}

void TzGen41::spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config) {
    // FIXME: make_shared called on a reference
    device_builder.add_facility(std::make_unique<EventTrailFilter>(
        std::make_shared<RegisterMap>(regmap()), get_sensor_info(), SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<AntiFlickerFilter>(
        std::make_shared<RegisterMap>(regmap()), get_sensor_info(), SENSOR_PREFIX));

    auto erc = device_builder.add_facility(
        std::make_unique<Gen41Erc>(register_map, SENSOR_PREFIX + "erc/", shared_from_this()));
    erc->initialize();
    erc->enable(true);

    auto geometry = HDGeometry();

    auto hw_register = std::make_shared<PseeHWRegister>(register_map);
    device_builder.add_facility(std::make_unique<Gen41_LL_Biases>(device_config, hw_register, SENSOR_PREFIX));

    device_builder.add_facility(
        std::make_unique<Gen41ROICommand>(geometry.get_width(), geometry.get_height(), register_map, SENSOR_PREFIX));

    device_builder.add_facility(std::make_unique<Gen41TzTriggerEvent>(register_map, SENSOR_PREFIX, shared_from_this()));

    device_builder.add_facility(
        std::make_unique<Gen41DigitalEventMask>(register_map, SENSOR_PREFIX + "ro/digital_mask_pixel_"));

    device_builder.add_facility(std::make_unique<Gen41DigitalCrop>(register_map, SENSOR_PREFIX));
}

TzGen41::~TzGen41() {}

long long TzGen41::get_sensor_id() {
    return (*register_map)[SENSOR_PREFIX + "Reserved_0014"].read_value();
}

DeviceConfigOptionMap TzGen41::get_device_config_options() const {
    return {{"format", DeviceConfigOption({"EVT21", "EVT3"}, "EVT3")}};
}

std::list<StreamFormat> TzGen41::get_supported_formats() const {
    std::list<StreamFormat> formats;
    formats.push_back(StreamFormat("EVT3;height=720;width=1280"));
    formats.push_back(StreamFormat("EVT21;height=720;width=1280;endianness=legacy"));
    return formats;
}

StreamFormat TzGen41::set_output_format(const std::string &format_name) {
    if (format_name == "EVT21") {
        (*register_map)[SENSOR_PREFIX + "edf/pipeline_control"]["format"].write_value(0x1);
        (*register_map)[SENSOR_PREFIX + "eoi/Reserved_8000"]["Reserved_7_6"].write_value(0x0);
    } else {
        // Default as EVT3
        (*register_map)[SENSOR_PREFIX + "edf/pipeline_control"]["format"].write_value(0x0);
        (*register_map)[SENSOR_PREFIX + "eoi/Reserved_8000"]["Reserved_7_6"].write_value(0x2);
    }
    return get_output_format();
}

StreamFormat TzGen41::get_output_format() const {
    StreamFormat format((*register_map)[SENSOR_PREFIX + "edf/pipeline_control"]["format"].read_value() ? "EVT21" :
                                                                                                         "EVT3");
    format["width"]  = "1280";
    format["height"] = "720";
    if (format.name() == "EVT21") {
        format["endianness"] = "legacy";
    }
    return format;
}

long TzGen41::get_system_id() const {
    return SystemId::SYSTEM_EVK3_GEN41;
}

bool TzGen41::set_mode_standalone() {
    time_base_config(false, true);

    sync_mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
    return true;
}

bool TzGen41::set_mode_master() {
    time_base_config(true, true);

    sync_mode_ = I_CameraSynchronization::SyncMode::MASTER;
    return true;
}

bool TzGen41::set_mode_slave() {
    time_base_config(true, false);

    sync_mode_ = I_CameraSynchronization::SyncMode::SLAVE;
    return true;
}

I_CameraSynchronization::SyncMode TzGen41::get_mode() {
    return sync_mode_;
}

int TzGen41::get_illumination() {
    bool valid       = false;
    uint16_t retries = 0;
    uint32_t counter = 0;

    while (valid == false && retries < 10) {
        uint32_t reg_val = (*register_map)[SENSOR_PREFIX + "lifo_status"].read_value();
        valid            = reg_val & 1 << 29;
        counter          = reg_val & ((1 << 27) - 1);
        retries += 1;
    }

    if (!valid) {
        MV_HAL_LOG_ERROR() << "Failed to get illumination";
        return -1;
    }

    if (counter != decltype(counter)(-1)) {
        float t = float(counter) / 100.;
        return powf(10, 3.5 - logf(t * 0.37) / logf(10));
    }
    return -1;
}

void TzGen41::time_base_config(bool external, bool master) {
    /* Configure sensor time base settings.
       By default, the sensor is in monocular mode
    */
    (*register_map)[SENSOR_PREFIX + "ro/time_base_ctrl"].write_value(vfield{
        {"time_base_mode", external},       // 0 : Internal, 1 : External
        {"external_mode", master},          // 0 : Slave, 1 : Master (valid when in external mode)
        {"external_mode_enable", external}, // 0 : External mode disabled, 1 : External mode enabled
        {"Reserved_10_4", 100}              // default 100 = 1us
    });

    if (external) {
        if (master) {
            // set SYNCHRO IO to output mode
            (*register_map)[SENSOR_PREFIX + "dig_pad2_ctrl"]["pad_sync"].write_value(0b1100);
        } else {
            // set SYNCHRO IO to input mode
            (*register_map)[SENSOR_PREFIX + "dig_pad2_ctrl"]["pad_sync"].write_value(0b1111);
        }
    }
}

void TzGen41::lifo_control(bool enable, bool out_en, bool cnt_en) {
    /* Control the LIFO settings.

    Args:
        enable (bool): Puts the LIFO in ready mode.
        out_en (bool): Turns on the LIFO.
        cnt_en (bool): Turns on the LIFO counter in the digital to start integrating.

    */

    if (enable && out_en) {
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_en"].write_value(enable);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_out_en"].write_value(out_en);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } else if (enable && !out_en) {
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_en"].write_value(enable);
    } else if (!enable && out_en) {
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_out_en"].write_value(out_en);
    } else if (!enable && !out_en) {
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_en"].write_value(enable);
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_out_en"].write_value(out_en);
    }
    (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_cnt_en"].write_value(cnt_en);
}

void TzGen41::iph_mirror_control(bool enable) {
    (*register_map)[SENSOR_PREFIX + "iph_mirr_ctrl"]["iph_mirr_en"].write_value(enable);
    std::this_thread::sleep_for(std::chrono::microseconds(20));
    (*register_map)[SENSOR_PREFIX + "iph_mirr_ctrl"]["iph_mirr_amp_en"].write_value(enable);
    std::this_thread::sleep_for(std::chrono::microseconds(20));
}

} // namespace Metavision
