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

#include <boost/program_options.hpp>
#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/hal/facilities/i_hw_register.h>
#include <metavision/hal/utils/hal_exception.h>
#include <metavision/sdk/base/utils/log.h>

namespace po = boost::program_options;

uint32_t mbx_read_uint32(Metavision::I_HW_Register *regs) {
    // Wait until a new value is available
    while (regs->read_register("PSEE/GENX320/mbx/misc") == 0) {}

    // Get the value from the RISC V
    uint32_t val = regs->read_register("PSEE/GENX320/mbx/status_ptr");

    // Indicate that the value has been consumed
    regs->write_register("PSEE/GENX320/mbx/misc", 0);

    return val;
}

int main(int argc, char *argv[]) {
    std::string serial;

    const std::string program_desc(
        "This sample shows how to receive log messages from an application embedded in the sensor.\n"
        "To use it, define the environment variable MV_FLAGS_RISCV_FW_PATH containing the PATH of a .hex file."
        "We provide an example (hello_world.hex) that is delivered along with the C++ sample source code.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("serial,s", po::value<std::string>(&serial)->default_value(""), "Serial ID of the camera.")
        ("help,h", "Produce help message.")
    ;
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    // Open the device
    std::cout << "Opening camera..." << std::endl;
    std::unique_ptr<Metavision::Device> device;
    try {
        device = Metavision::DeviceDiscovery::open(serial);
    } catch (const Metavision::HalException &e) { MV_LOG_ERROR() << "Error exception:" << e.what(); }

    if (!device) {
        MV_LOG_ERROR() << "Camera opening failed.";
        return 1;
    }
    MV_LOG_INFO() << "Found a camera";

    Metavision::I_HW_Identification *i_hw_identification = device->get_facility<Metavision::I_HW_Identification>();
    if (!i_hw_identification || ((i_hw_identification->get_sensor_info().name_ != "GenX320") && (i_hw_identification->get_sensor_info().name_ != "GenX320MP"))) {
        MV_LOG_ERROR() << "Failed to get sensor info or sensor is not 'GenX320'.";
        return 1;
    }

    Metavision::I_EventsStream *i_eventsstream = device->get_facility<Metavision::I_EventsStream>();
    if (!i_eventsstream) {
        MV_LOG_ERROR() << "Could not initialize events stream.";
        return 1;
    }

    Metavision::I_HW_Register *regs = device->get_facility<Metavision::I_HW_Register>();
    if (!regs) {
        MV_LOG_ERROR() << "Could not initialize HW register access.";
        return 1;
    }

    // Start the camera
    i_eventsstream->start();

    std::atomic<bool> stop_streaming = false;
    std::thread polling_loop([&]() {
        while (!stop_streaming) {
            short ret = i_eventsstream->poll_buffer();
            if (ret < 0) {
                stop_streaming = true;
            } else if (ret == 0) {
                continue;
            }
            // Grab and drop raw data
            auto raw_data = i_eventsstream->get_latest_raw_data();
        }
    });

    // Keep polling and printing messages while the camera stream is on
    MV_LOG_INFO() << "Received messages:";
    while (!stop_streaming) {
        uint32_t v = mbx_read_uint32(regs);
        for (std::size_t pos = 0; pos < sizeof(uint32_t); ++pos) {
            char c = v & 0xff;
            if (c == 0)
                break;
            std::cout << char(c);
            v >>= 8;
        }
    }

    // Wait end of decoding loop
    i_eventsstream->stop();
    MV_LOG_INFO() << "Camera stopped.";
    polling_loop.join();
}
