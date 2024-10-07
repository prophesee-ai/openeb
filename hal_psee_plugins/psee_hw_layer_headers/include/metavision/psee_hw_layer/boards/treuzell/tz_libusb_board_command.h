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
#include <mutex>
#include <string>
#include <vector>
#include <memory>

#include "metavision/hal/utils/data_transfer.h"
#include "metavision/psee_hw_layer/boards/utils/psee_libusb.h"
#include "metavision/psee_hw_layer/boards/treuzell/board_command.h"

namespace Metavision {

class TzCtrlFrame;
class DataTransfer;

struct UsbInterfaceId {
    uint16_t vid;
    uint16_t pid;
    uint8_t usb_class;
    uint8_t subclass;
};

// A list of hacks to accommodate known misbehaving systems
struct BoardQuirks {
    bool reset_on_destroy;
    bool ignore_size_on_device_prop_answer;
    bool do_not_set_config;
};

class TzLibUSBBoardCommand : public virtual BoardCommand {
public:
    TzLibUSBBoardCommand() = delete;
    TzLibUSBBoardCommand(std::shared_ptr<LibUSBContext> ctx, libusb_device *dev, libusb_device_descriptor &desc,
                         const std::vector<UsbInterfaceId> &usb_ids);
    ~TzLibUSBBoardCommand();

    long get_board_speed() override;
    std::shared_ptr<LibUSBContext> get_libusb_ctx() {
        return libusb_ctx;
    }

    std::string get_serial() override;
    std::string get_name() override;
    std::string get_manufacturer() override;
    time_t get_build_date() override;
    uint32_t get_version() override;

    void transfer_tz_frame(TzCtrlFrame &req) override;

    unsigned int get_device_count() override;
    std::vector<uint32_t> read_device_register(uint32_t device, uint32_t address, int nval = 1) override;
    void write_device_register(uint32_t device, uint32_t address, const std::vector<uint32_t> &val) override;

    // @brief Create a new DataTransfer object to stream the currently opened device
    std::unique_ptr<DataTransfer::RawDataProducer> build_raw_data_producer(uint32_t raw_event_size_bytes) override;

private:
    bool clear_endpoint();
    bool reset_device();

    // Board state
    std::shared_ptr<LibUSBContext> libusb_ctx;
    std::shared_ptr<LibUSBDevice> dev_;
    int bInterfaceNumber;
    int bEpControlIn;
    int bEpControlOut;
    int bEpCommAddress;
    std::mutex tz_control_mutex_;
    libusb_speed dev_speed_ = LIBUSB_SPEED_UNKNOWN;
    std::string manufacturer;
    std::string product;
    time_t build_date;
    uint32_t version;

    // Workaround know bugs
    struct BoardQuirks quirks;
    void select_board_quirks(libusb_device_descriptor &desc);
    // Quirks to be selected before sending TzFrame
    void select_early_quirks(libusb_device_descriptor &desc);
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_LIBUSB_BOARD_COMMAND_H
