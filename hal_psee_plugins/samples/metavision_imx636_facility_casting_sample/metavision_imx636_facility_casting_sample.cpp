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

#include <exception>
#include <iostream>
#include <iomanip>
#include <boost/program_options.hpp>

#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/utils/hal_exception.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/hal/facilities/i_ll_biases.h>

#include <metavision/psee_hw_layer/devices/imx636/imx636_ll_biases.h>

namespace po = boost::program_options;
int main(int argc, char *argv[]) {
    std::string serial;
    const std::string program_desc(
        "This code sample demonstrates how to access device specific implementations of HAL facilities");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("serial,s", po::value<std::string>(&serial), "Serial ID of the camera.")
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
    } catch (Metavision::HalException &e) { std::cout << "Error exception: " << e.what() << std::endl; }

    if (!device) {
        std::cerr << "Camera opening failed." << std::endl;
        return 1;
    }

    auto ll_biases = device->get_facility<Metavision::I_LL_Biases>();
    if (!ll_biases) {
        std::cerr << "Camera does not have LL Bias facility" << std::endl;
        return 1;
    }

    // TODO : MV-551 use I_EventRateActivityFilterModule facility in this sample
    Metavision::Imx636_LL_Biases *imx636_ll_biases = dynamic_cast<Metavision::Imx636_LL_Biases *>(ll_biases);
    if (!imx636_ll_biases) {
        std::cerr << "Camera is not an IMX636 camera" << std::endl;
        return 1;
    }

    std::cout << "Available IMX636 biases: " << std::endl;
    for (auto bias : imx636_ll_biases->get_all_biases()) {
        std::cout << "- " << bias.first << " = " << bias.second << std::endl;
    }

    return 0;
}
