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

#include "sample_camera_synchronization.h"
#include "internal/sample_register_access.h"

SampleCameraSynchronization::SampleCameraSynchronization(std::shared_ptr<SampleUSBConnection> usb_connection) :
    usb_connection_(usb_connection) {}

bool SampleCameraSynchronization::set_mode_standalone() {
    // We update register 0x00009008 (ro/time_base_ctrl) which has a default value of 0x00000640
    // We write the following bits (0 being the least significant one):
    // - bit 1: time_base_mode set to 0 (internal)
    // - bit 2: external_mode set to 1 (master, but value ignored in internal mode)
    // - bit 3: external_mode_ set to 0 (internal)
    write_register(*usb_connection_, 0x00009008, 0x00000644);

    return true;
}

bool SampleCameraSynchronization::set_mode_master() {
    // First, update register 0x00009008 (ro/time_base_ctrl) which has a default value of 0x00000640
    // We write the following bits (0 being the least significant one):
    // - bit 1: time_base_mode set to 1 (external)
    // - bit 2: external_mode set to 1 (master)
    // - bit 3: external_mode_ set to 1 (external)
    write_register(*usb_connection_, 0x00009008, 0x0000064E);

    // Then, update register 0x00000044 (dig_pad2_ctrl) which has a default value of 0xCCFFFCCF
    // To set camera to master, we update the bits [19:16] to 1100 (C) so we set the value 0xCCFCFCCF
    write_register(*usb_connection_, 0x00000044, 0xCCFCFCCF);

    return true;
}

bool SampleCameraSynchronization::set_mode_slave() {
    // This code is baseline that should be checked and enhanced by reviewing the datasheet and Prophesee plugin code.

    // First, update register 0x00009008 (ro/time_base_ctrl) which has a default value of 0x00000640
    // We write the following bits (0 being the least significant one):
    // - bit 1: time_base_mode set to 1 (external)
    // - bit 2: external_mode set to 0 (slave)
    // - bit 3: external_mode_ set to 1 (external)
    write_register(*usb_connection_, 0x00009008, 0x0000064A);

    // Then, update register 0x00000044 (dig_pad2_ctrl) which has a default value of 0xCCFFFCCF
    // To set camera to master, we leave the bits [19:16] at 1111 (F) so we set the value 0xCCFFFCCF
    write_register(*usb_connection_, 0x00000044, 0xCCFFFCCF);

    return true;
}

SampleCameraSynchronization::SyncMode SampleCameraSynchronization::get_mode() const {
    auto time_base_ctrl_register_value = read_register(*usb_connection_, 0x00009008);
    if (time_base_ctrl_register_value == 0x00000644)
        return SyncMode::SLAVE;
    else if (time_base_ctrl_register_value == 0x0000064E)
        return SyncMode::MASTER;
    else return SyncMode::STANDALONE;
}
