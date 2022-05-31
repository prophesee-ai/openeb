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

#ifndef METAVISION_HAL_PSEE_LIBUSB_BOARD_COMMAND_H
#define METAVISION_HAL_PSEE_LIBUSB_BOARD_COMMAND_H

#include <libusb.h>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

#include "devices/utils/device_system_id.h"

namespace Metavision {

/// Type of camera to be returned
enum CameraType {
    REMOTE = 1,
    LOCAL  = 2,
    ANY    = 3,
};

class PseeLibUSBBoardCommand {
public:
    using Register_Addr = uint32_t;
    using ListSerial    = std::list<std::string>;

    PseeLibUSBBoardCommand();
    virtual ~PseeLibUSBBoardCommand();

    virtual long try_to_flush()      = 0;
    virtual std::string get_serial() = 0;

    virtual bool open(const std::string &serial);
    virtual long get_board_version();
    virtual long get_board_id();
    virtual long get_board_release_version();
    virtual long get_board_build_date();
    virtual long get_board_speed();
    virtual long get_board_version_control_id();

    virtual long get_system_id();
    long get_system_version();
    unsigned int get_system_build_date();
    unsigned int get_system_version_control_id();

    virtual long get_temperature();
    virtual long get_illumination();

    virtual uint32_t control_read_register_32bits(uint8_t usbvendorcmd, uint32_t address, bool big_endian = true);
    virtual uint16_t control_read_register_16bits(uint8_t usbvendorcmd, uint32_t address);
    virtual void control_write_register_32bits(uint8_t usbvendorcmd, uint32_t address, uint32_t val);
    virtual void control_write_vector_32bits(uint32_t address, const std::vector<uint32_t> &val);
    virtual void control_write_register_8bits(uint8_t usbvendorcmd, uint8_t address, uint8_t val);
    virtual uint8_t control_read_register_8bits(uint8_t usbvendorcmd, uint8_t address);

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

    bool wait_fpga_boot_state();
    void reset_fpga();

    libusb_transfer *contruct_async_bulk_transfer(unsigned char *buf, int packet_size,
                                                  libusb_transfer_cb_fn async_bulk_cb, void *user_data,
                                                  unsigned int timeout);
    virtual void prepare_async_bulk_transfer(libusb_transfer *transfer, unsigned char *buf, int packet_size,
                                             libusb_transfer_cb_fn async_bulk_cb, void *user_data,
                                             unsigned int timeout);
    static void free_async_bulk_transfer(libusb_transfer *transfer);
    static int submit_transfer(libusb_transfer *transfer);

protected:
    PseeLibUSBBoardCommand(libusb_device_handle *dev_handle);

    static bool init_libusb();

    virtual int bulk_transfer(unsigned char *buf, int packet_size, unsigned int timeout, int &actual_size);

    std::map<Register_Addr, uint32_t> mregister_state;
    libusb_device_handle *dev_handle_ = nullptr; // a device handle
    libusb_speed dev_speed_           = LIBUSB_SPEED_UNKNOWN;

private:
    PseeLibUSBBoardCommand(const PseeLibUSBBoardCommand &)  = delete;
    PseeLibUSBBoardCommand(const PseeLibUSBBoardCommand &&) = delete;
    PseeLibUSBBoardCommand &operator=(const PseeLibUSBBoardCommand &) = delete;
    PseeLibUSBBoardCommand &operator=(const PseeLibUSBBoardCommand &&) = delete;

    bool has_register(Register_Addr regist);
    virtual void get_ccam2_with_serial(libusb_context *ctx, const std::string &serial);
    long check_fpga_boot_state();

    std::mutex thread_safety_;

    static std::mutex protect_libusb_submit_; // mutex used to protect libusb_submit as if polling thread is already
                                              // running (2nd cam) the starting of async transferts
};
} // namespace Metavision
#endif // METAVISION_HAL_PSEE_LIBUSB_BOARD_COMMAND_H
