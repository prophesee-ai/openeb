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

#include <iomanip>
#include <sstream>

#include "metavision/hal/utils/hal_log.h"
#include "boards/utils/vendor_command_definition.h"
#include "boards/utils/config_registers_map.h"
#include "boards/fx3/fx3_libusb_board_command.h"
#include "devices/utils/device_system_id.h"

namespace Metavision {

Fx3LibUSBBoardCommand::Fx3LibUSBBoardCommand() = default;

Fx3LibUSBBoardCommand::Fx3LibUSBBoardCommand(libusb_device_handle *dev_handle) : PseeLibUSBBoardCommand(dev_handle) {}

std::string Fx3LibUSBBoardCommand::get_serial() {
    long version = get_board_version();
    uint32_t val = -1;

    long FPGA_boot_state = wait_fpga_boot_state();

    if (FPGA_boot_state) {
        MV_HAL_LOG_TRACE() << "FPGA is properly configured";
    } else {
        MV_HAL_LOG_ERROR() << "FPGA is not properly configured";
    }
    if (version == 3) {
        // CCam5 Gen41
        val = control_read_register_32bits(CMD_READ_SYSTEM_ID, 0x00);
    } else if (version == 2) {
        //  CCam3 Evk1
        val = control_read_register_32bits(CMD_READ_REGFPGA_32, 0);
        val = control_read_register_32bits(CMD_READ_REGFPGA_32, TEP_SERIAL_MSB_ADDR);
        // At this stage, only the MSB register value is returned as serial number is 16 bits limited.
    } else {
        // CCam2 Legacy
        val = control_read_register_16bits(CMD_READ, TEP_SERIAL_MSB_ADDR);
    }
    std::ostringstream ostr;
    ostr                 //<< std::showbase // show the 0x prefix
        << std::internal // fill between the prefix and the number
        << std::setfill('0') << std::setw(8) << std::hex;
    ostr << val;
    ostr << std::dec;
    return ostr.str();
}

long Fx3LibUSBBoardCommand::try_to_flush() {
    int actual         = 0;
    long total_flush   = 0;
    long last_flush    = 0;
    int r              = 0;
    int num_iterations = 0;
    int max_iterations = 8;
    int max_data       = 300000;

    MV_HAL_LOG_TRACE() << "Start flushing";
    MV_HAL_LOG_TRACE() << "Hard flush";

    if (dev_handle_) {
        libusb_clear_halt(dev_handle_, (1 | LIBUSB_ENDPOINT_IN));
    }

    do {
        if (num_iterations != 0 || total_flush != 0) {
            MV_HAL_LOG_TRACE() << "Flushing" << total_flush;
        }

        last_flush = total_flush;
        // send_ccam2ctrl_command(TEP_CCAM2_CONTROL_ENABLE_FX3_HOST_IF_BIT_IDX,  0); // Enables the FX3 host I/F and
        // also serves to add a delay. Keep this line, unless the FX3 will not have time to change buffers and reply.
        write_register(TEP_TRIGGERS_ADDR, 0);

        do {
            uint8_t buf[1024];
            actual = 0;
            r      = bulk_transfer(buf, 1024, 100, actual); // 2ms
            total_flush += actual;
            if (total_flush > max_data) {
                break;
            }
        } while (r == 0 && actual > 0);

        num_iterations++;

        if (num_iterations > max_iterations) {
            MV_HAL_LOG_WARNING() << "Aborting flush: maximum number of iterations reached!";
            break;
        }

        if (total_flush > max_data) {
            MV_HAL_LOG_WARNING() << "Aborting flush: maximum data amount reached!";
            break;
        }
    } while (last_flush != total_flush);

    MV_HAL_LOG_TRACE() << "Flushed" << total_flush;
    return total_flush;
}

// ------------------------------
// Static methods

PseeLibUSBBoardCommand::ListSerial Fx3LibUSBBoardCommand::get_list_serial() {
    ListSerial lserial;
    if (!init_libusb()) {
        return lserial;
    }
    get_all_serial(nullptr, lserial);
    return lserial;
}

void Fx3LibUSBBoardCommand::get_all_serial(libusb_context *ctx, ListSerial &lserial) {
    libusb_device **devs;
    int cnt = libusb_get_device_list(ctx, &devs); // get the list of devices
    if (cnt <= 0) {
        MV_HAL_LOG_TRACE() << "EVK1 libusb BC: USB Device list empty cnt=" << cnt;
        return;
    }

    MV_HAL_LOG_TRACE() << "EVK1 libusb BC: libusb_get_device_list found" << cnt << "devices";

    for (int i = 0; i < cnt; i++) {
        libusb_device_descriptor desc;
        int r = libusb_get_device_descriptor(devs[i], &desc);
        if (r < 0) {
            MV_HAL_LOG_ERROR() << "Failed to get device descriptor r=" << r;
            return;
        }
        if ((desc.idVendor == 0x04b4) &&
            ((desc.idProduct == 0x00f1) || (desc.idProduct == 0x00f4) || (desc.idProduct == 0x00bc))) {
            libusb_device_handle *dev_handle = nullptr;
            r                                = libusb_open(devs[i], &dev_handle);
            if (r != 0) {
                MV_HAL_LOG_ERROR() << "Unable to open device";
                continue;
            } else {
                MV_HAL_LOG_TRACE() << "EVK1 libusb BC: PSEE device found";
            }
            if (libusb_kernel_driver_active(dev_handle, 0) == 1) { // find out if kernel driver is attached
                MV_HAL_LOG_TRACE() << "Kernel Driver Active";
                if (libusb_detach_kernel_driver(dev_handle, 0) == 0) // detach it
                    MV_HAL_LOG_TRACE() << "Kernel Driver Detached!";
            }
            r = libusb_claim_interface(dev_handle, 0); // claim interface 0 (the first) of device
            if (r < 0) {
                MV_HAL_LOG_ERROR() << Log::no_space << "Camera is busy (r=" << r << ")";
                libusb_close(dev_handle);
                continue;
            }

            // Assert LIBUSB_SPEED is at minimum LIBUSB_SPEED_SUPER
            Fx3LibUSBBoardCommand cmd(dev_handle);
            int speed             = libusb_get_device_speed(devs[i]);
            const auto cur_serial = cmd.get_serial();
            if (speed < LIBUSB_SPEED_SUPER) {
                MV_HAL_LOG_WARNING()
                    << "Your EVK camera" << cur_serial
                    << "isn't connected in USB3. Please verify your connection or some malfunction may occur.";
            }

            // Add device even if possibly USB 2
            lserial.push_back(cur_serial);
        }
    }
    libusb_free_device_list(devs, 1); // free the list, unref the devices in it
}
} // namespace Metavision