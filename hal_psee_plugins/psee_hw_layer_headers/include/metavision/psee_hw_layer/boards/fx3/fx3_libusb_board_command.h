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

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

#ifdef _MSC_VER
#define NOMINMAX // libusb.h includes windows.h which defines min max macros that we don't want
#endif
#include <libusb.h>

namespace Metavision {

class LibUSBContext;
class LibUSBDevice;
class PseeLibUSBDataTransfer;

/// Type of camera to be returned
enum CameraType {
    REMOTE = 1,
    LOCAL  = 2,
    ANY    = 3,
};

class Fx3LibUSBBoardCommand {
public:
    using Register_Addr = uint32_t;
    using ListSerial    = std::list<std::string>;

    static ListSerial get_list_serial();

    Fx3LibUSBBoardCommand();
    virtual ~Fx3LibUSBBoardCommand();

    long try_to_flush();
    std::string get_serial();

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

    // @brief Create a new DataTransfer object to stream the currently opened device
    std::unique_ptr<PseeLibUSBDataTransfer> build_data_transfer(uint32_t raw_event_size_bytes);

protected:
    virtual int bulk_transfer(unsigned char *buf, int packet_size, unsigned int timeout, int &actual_size);

    std::map<Register_Addr, uint32_t> mregister_state;
    std::shared_ptr<LibUSBDevice> dev_;
    libusb_speed dev_speed_ = LIBUSB_SPEED_UNKNOWN;

private:
    Fx3LibUSBBoardCommand(const Fx3LibUSBBoardCommand &)  = delete;
    Fx3LibUSBBoardCommand(const Fx3LibUSBBoardCommand &&) = delete;
    Fx3LibUSBBoardCommand &operator=(const Fx3LibUSBBoardCommand &) = delete;
    Fx3LibUSBBoardCommand &operator=(const Fx3LibUSBBoardCommand &&) = delete;

    bool has_register(Register_Addr regist);
    virtual void get_ccam2_with_serial(std::shared_ptr<LibUSBContext> libusb_ctx, const std::string &serial);
    long check_fpga_boot_state();

    std::mutex thread_safety_;

    static std::mutex protect_libusb_submit_; // mutex used to protect libusb_submit as if polling thread is already
                                              // running (2nd cam) the starting of async transferts

    static void get_all_serial(std::shared_ptr<LibUSBContext> libusb_ctx, ListSerial &lserial);

    Fx3LibUSBBoardCommand(std::shared_ptr<LibUSBDevice> dev);
};
} // namespace Metavision
#endif // METAVISION_HAL_FX3_LIBUSB_BOARD_COMMAND_H
