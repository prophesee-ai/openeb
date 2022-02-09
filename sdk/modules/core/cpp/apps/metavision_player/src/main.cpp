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

#ifdef _WIN32
#include <Shlobj.h>
#include <windows.h>
#endif
#include <iostream>
#include <atomic>
#include <signal.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <metavision/sdk/base/utils/log.h>

#include "params.h"
#include "viewer.h"
#include "utils.h"

namespace po = boost::program_options;

// Parse command line input and return parameters for the application.
// Returns true if the application is supposed to continue running, false otherwise.
bool parse_command_line(int argc, const char *argv[], Parameters &app_params) {
    const std::string program_desc("Metavision Player allows to stream/records events and analyse event-based data.\n");

    boost::filesystem::path docs_path;
#ifdef _WIN32

    PWSTR ppszPath; // variable to receive the path memory block pointer.
    HRESULT hr = SHGetKnownFolderPath(FOLDERID_Documents, 0, NULL, &ppszPath);
    std::wstring myPath;
    if (SUCCEEDED(hr)) {
        myPath = ppszPath; // make a local copy of the path
    }
    CoTaskMemFree(ppszPath); // free up the path memory block
    docs_path = boost::filesystem::path(myPath);

#else
    char *home_path_ptr = getenv("HOME");
    if (home_path_ptr) {
        std::string home_path(home_path_ptr);
        docs_path = boost::filesystem::path(home_path) / "Documents";
    }
#endif

    if (docs_path.empty()) {
        MV_LOG_ERROR() << "Could not determine default Documents directory ";
        return false;
    }

    docs_path /= "Prophesee";
    if (!boost::filesystem::exists(docs_path) && !boost::filesystem::create_directories(docs_path)) {
        MV_LOG_ERROR() << "Could not create directory " << docs_path.string();
        return false;
    }

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("biases,b",               po::value<std::string>(&app_params.in_bias_file), "Path to a bias file. If not specified, default biases will be used.")
        ("show-biases",            po::bool_switch(&app_params.show_biases)->default_value(false), "Show sliders to change biases dynamically.")
        ("buffer-capacity,k",      po::value<int>(&app_params.buffer_size_mev)->default_value(100), "Max number of events to be stored in the buffer, in millions of events.")
        ("output-bias-file",       po::value<std::string>(&app_params.out_bias_file)->default_value((docs_path / "out.bias").string()), "Path to the output bias file for exporting, only available if --show-biases is used.")
        ("output-png-file,p",      po::value<std::string>(&app_params.out_png_file)->default_value((docs_path / "frame.png").string()), "Path to the output PNG file for exporting.")
        ("output-avi-file,v",      po::value<std::string>(&app_params.out_avi_file)->default_value((docs_path / "video.avi").string()), "Path to the output AVI file for exporting.")
        ("output-avi-framerate,f", po::value<int>(&app_params.out_avi_fps)->default_value(25), "Frame rate of the output AVI file.")
        ("output-raw-basename,o",  po::value<std::string>(&app_params.out_raw_basename)->default_value((docs_path / "out").string()),
            "Path and base name of the output RAW file for exporting. Each file will have the name <path>/<basename>_<date>.raw, where <date> represents the day and time the file was recorded.")
        ("input-raw-file,i",       po::value<std::string>(&app_params.in_raw_file), "Path to input RAW file. If not specified, the camera live stream is used.")
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
        return false;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return false;
    }

    // Check extension of provided output files
    if (boost::filesystem::extension(app_params.out_bias_file) != ".bias") {
        MV_LOG_ERROR() << "Wrong extension for provided output bias file: supported extension is '.bias'";
        return false;
    }
    if (boost::filesystem::extension(app_params.out_png_file) != ".png") {
        MV_LOG_ERROR() << "Wrong extension for provided output PNG file: supported extension is '.png'";
        return false;
    }
    if (boost::filesystem::extension(app_params.out_avi_file) != ".avi") {
        MV_LOG_ERROR() << "Wrong extension for provided output AVI file: supported extension is '.avi'";
        return false;
    }
    return true;
}

namespace {
std::atomic<bool> signal_caught{false};

void sig_handler(int s) {
    MV_LOG_TRACE() << "Interrupt signal received." << std::endl;
    signal_caught = true;
}
} // anonymous namespace

int main(int argc, const char *argv[]) {
    signal(SIGINT, sig_handler);

    // Parse command line.
    Parameters params;
    if (!parse_command_line(argc, argv, params)) {
        return 1;
    }

    Viewer viewer(params);
    try {
        viewer.start();
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }

    bool ret = true;
    while (!signal_caught && ret) {
        ret = viewer.update();
    }

    viewer.stop();

    return signal_caught;
}
