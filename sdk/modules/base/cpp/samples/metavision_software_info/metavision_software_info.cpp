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
#include <metavision/sdk/base/utils/software_info.h>
#include <metavision/sdk/base/utils/log.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    const std::string program_desc("The source code of this application demonstrates how to use Metavision SDK Base "
                                   "API to get information on Metavision software\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("commit,c", "Print commit hash at compile time.")
        ("date,d", "Print commit date at compile time.")
        ("version,v", "Print version.")
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

    if (vm.count("help") || (argc <= 1)) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    auto &metavision_sdk_software_info = Metavision::get_metavision_software_info();
    if (vm.count("commit")) {
        MV_LOG_INFO() << metavision_sdk_software_info.get_vcs_commit();
    }
    if (vm.count("date")) {
        MV_LOG_INFO() << metavision_sdk_software_info.get_vcs_date();
    }
    if (vm.count("version")) {
        MV_LOG_INFO() << metavision_sdk_software_info.get_version();
    }

    return 0;
}
