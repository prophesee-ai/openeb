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
#include <windows.h>
#include <shlobj.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <filesystem>

#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/resources_folder.h"
#include "metavision_hal_install_path.h"

namespace {

#ifdef _WIN32
std::string read_registry_subkey_hklm(LPCSTR subkey, LPCSTR registry_value_name) {
    std::string result("");

    HKEY hKeyExt;
    DWORD dwDisposition;
    long ret = RegOpenKeyEx(HKEY_LOCAL_MACHINE, subkey, 0, KEY_QUERY_VALUE, &hKeyExt);

    if (ret == ERROR_SUCCESS) {
        DWORD dwTaille = 1024;
        DWORD dwType;
        BYTE buf[1024];
        ret = RegQueryValueEx(hKeyExt, registry_value_name, 0, &dwType, buf, &dwTaille);
        if (ret != ERROR_SUCCESS) {
            buf[0] = 0;
        } else {
            if (buf[dwTaille - 1] == '.')
                buf[dwTaille - 1] = 0; // windows adds a . at the end of paths!!!
            result = std::string((char *)(&buf[0]));
        }
        RegCloseKey(hKeyExt);
    }
    return result;
}
#else

bool exists(const std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

#endif

} /* anonymous namespace */

namespace Metavision {

#ifndef __ANDROID__
std::filesystem::path ResourcesFolder::get_user_path() {
    std::filesystem::path p;
#ifdef _WIN32
    wchar_t *widePath;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, NULL, &widePath))) {
        p = std::filesystem::path(widePath) / "Metavision";
        CoTaskMemFree(widePath);
    }
#elif __linux__
    const char *homeDir = std::getenv("HOME");
    if (homeDir) {
        p = std::filesystem::path(homeDir) / ".local" / "share" / "metavision";
    }
#elif __APPLE__
    const char *homeDir = std::getenv("HOME");
    if (homeDir) {
        p = std::filesystem::path(homeDir) / "Library" / "Application Support" / "Metavision";
    }
#endif
    if (p.empty()) {
        throw std::runtime_error("Unable to get user path");
    }
    p = p / "hal";

    if (!std::filesystem::exists(p)) {
        std::filesystem::create_directories(p);
    }
    return p;
}
#endif

std::string ResourcesFolder::get_install_path() {
    char *p = getenv("MV_HAL_INSTALL_PATH");
    if (p) {
        return p;
    }

#ifdef _WIN32
    return read_registry_subkey_hklm(METAVISION_SUBKEY, METAVISION_SUBKEY_INSTALL_PATH) + "\\" +
           HAL_INSTALL_SUPPORT_RELATIVE_PATH;
#else

    auto path_candidates = Metavision::get_root_installation_path_candidates();
    for (auto &candidate : path_candidates) {
        if (exists(candidate + "/" + Metavision::METAVISION_HAL_LIB_RELATIVE_PATH)) {
            return candidate + "/" + HAL_INSTALL_SUPPORT_RELATIVE_PATH;
        }
    }
    return "";
#endif
}

std::string ResourcesFolder::get_plugin_install_path() {
#ifdef _WIN32
    return read_registry_subkey_hklm(METAVISION_SUBKEY, METAVISION_SUBKEY_INSTALL_PATH) + "\\" +
           HAL_INSTALL_PLUGIN_RELATIVE_PATH;
#else
    auto path_candidates = Metavision::get_root_installation_path_candidates();
    for (auto &path_candidate : path_candidates) {
        std::string candidate = path_candidate + "/" + Metavision::HAL_INSTALL_PLUGIN_RELATIVE_PATH;
        if (exists(candidate)) {
            // Check if folder is empty or not
            DIR *dir_descriptor;
            dir_descriptor = opendir(candidate.c_str());
            if (dir_descriptor) {
                struct dirent *entries;
                // Loop over the contents of the directory, to make sure that it's not empty and that it doesn't contain
                // only subdirectories
                // Remark : readdir() doesn't guarantee a specific order, so we loop over the all directory
                while ((entries = readdir(dir_descriptor)) != NULL) {
                    // Verify that the entry is not a directory
                    struct stat statbuf;
                    bool is_dir = false;
                    if (stat(entries->d_name, &statbuf) == 0) {
                        is_dir = S_ISDIR(statbuf.st_mode);
                    }
                    if (!is_dir) {
                        closedir(dir_descriptor);
                        return candidate;
                    }
                }
                closedir(dir_descriptor);
            }
        }
    }
    return "";
#endif
}

} // namespace Metavision
