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

#ifndef METAVISION_HAL_TZ_LIBUSB_BOARD_COMMAND_H
#define METAVISION_HAL_TZ_LIBUSB_BOARD_COMMAND_H

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

#include "boards/treuzell/tz_libusb.h"
#include "devices/utils/device_system_id.h"

namespace Metavision {

class TzCtrlFrame;
class TzBoardDataTransfer;

struct UsbInterfaceId {
    uint16_t vid;
    uint16_t pid;
    uint8_t usb_class;
    uint8_t subclass;
};

class TzLibUSBBoardCommand {
    using Register_Addr = uint32_t;

public:
    TzLibUSBBoardCommand() = delete;
    TzLibUSBBoardCommand(std::shared_ptr<LibUSBContext> ctx, libusb_device *dev, libusb_device_descriptor &desc);
    ~TzLibUSBBoardCommand();

    long get_board_speed();
    std::shared_ptr<LibUSBContext> get_libusb_ctx() {
        return libusb_ctx;
    }

    std::string get_serial();
    std::string get_name();
    std::string get_manufacturer();
    time_t get_build_date();
    uint32_t get_version();

    /// @brief Writes shadow register (value stored on computer side)
    void write_register(Register_Addr regist, uint32_t value);

    /// @brief Reads shadow register (value stored on computer side)
    /// @return The value of the register
    uint32_t read_register(Register_Addr regist);

    /// @brief Loads the register on the board side with the value stored on computer
    /// @return Nothing to express that the method loads the value from the board and stores it
    void load_register(Register_Addr regist);

    void set_register_bit(Register_Addr regist, int idx, bool state);
    void send_register(Register_Addr regist);
    void send_register_bit(Register_Addr regist, int idx, bool state);
    uint32_t read_register_bit(Register_Addr register_addr, int idx);
    void init_register(Register_Addr regist, uint32_t value);

    void transfer_tz_frame(TzCtrlFrame &req);

    unsigned int get_device_count();
    std::vector<uint32_t> read_device_register(uint32_t device, uint32_t address, int nval = 1);
    void write_device_register(uint32_t device, uint32_t address, const std::vector<uint32_t> &val);

    static void add_usb_id(uint16_t vid, uint16_t pid, uint8_t subclass) {
        known_usb_ids.push_back({vid, pid, 0xFF, subclass});
    }

protected:
    std::map<Register_Addr, uint32_t> mregister_state;

private:
    bool clear_endpoint();
    bool reset_device();

    // Board state
    std::shared_ptr<LibUSBContext> libusb_ctx;
    int bInterfaceNumber;
    int bEpControlIn;
    int bEpControlOut;
    int bEpCommAddress;
    libusb_device_handle *dev_handle_;
    libusb_speed dev_speed_ = LIBUSB_SPEED_UNKNOWN;
    std::string manufacturer;
    std::string product;
    time_t build_date;
    uint32_t version;

    bool has_register(Register_Addr regist);
    std::mutex thread_safety_;
    friend TzBoardDataTransfer;

    static std::vector<UsbInterfaceId> known_usb_ids;
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_LIBUSB_BOARD_COMMAND_H
