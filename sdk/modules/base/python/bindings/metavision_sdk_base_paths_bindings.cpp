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
#include <iostream>
#include <pybind11/pybind11.h>
#define NOMINMAX
#include <windows.h>
#include <string>

#include "metavision/utils/pybind/python_binder_helper.h"

namespace {
bool try_import() {
    try {
        py::module::import("metavision_sdk_base_dummy");
        return true;
    } catch (...) {}
    return false;
}
} // namespace

namespace py = pybind11;

PYBIND11_MODULE(metavision_sdk_base_paths_internal, m) {
    bool import = try_import();
    if (!import) {
        for (auto &fn : {Metavision::setMetavisionDllPathsForBindings, Metavision::setBuildDirDllPathsForBindings,
                         Metavision::setPseeInstallPathsForBindings}) {
            // try to import after updating the environment incrementally using paths
            // deduced by MV_DLL_PATH, the Prophesee reg keys or the build folder
            if (fn()) {
                if (try_import()) {
                    import = true;
                    break;
                }
            }
        }
    }

    if (!import) {
        std::cerr << "Unable to locate the path to necessary DLLs for Metavision Python bindings." << std::endl;
        std::cerr << "If you have installed Metavision using one of our installers, please try to uninstall and "
                     "install again."
                  << std::endl;
        DWORD attrs = GetFileAttributes(BUILD_BIN_DIR);
        if ((attrs != INVALID_FILE_ATTRIBUTES) && (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
            std::cerr << "If compiling the code from source, make sure the PATH environment variable contains the path "
                      << BUILD_BIN_DIR << std::endl;
        }
    }
}