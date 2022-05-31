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

#ifndef METAVISION_HAL_FX3_LIBUSB_BOARD_COMMAND_H
#define METAVISION_HAL_FX3_LIBUSB_BOARD_COMMAND_H

#include "boards/utils/psee_libusb_board_command.h"

namespace Metavision {

class Fx3LibUSBBoardCommand : public PseeLibUSBBoardCommand {
public:
    static ListSerial get_list_serial();

    Fx3LibUSBBoardCommand();

    std::string get_serial() override final;
    long try_to_flush() override final;

private:
    static void get_all_serial(libusb_context *ctx, ListSerial &lserial);

    Fx3LibUSBBoardCommand(libusb_device_handle *dev_handle);
};
} // namespace Metavision
#endif // METAVISION_HAL_FX3_LIBUSB_BOARD_COMMAND_H
