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

#include <memory>
#include <algorithm>
#include <dirent.h>
#ifdef _WIN32
#include <windows.h>
#include <strsafe.h>
#else
#include <dlfcn.h>
#endif

#include "metavision/hal/plugin/detail/plugin_loader.h"
#include "metavision/hal/plugin/plugin.h"
#include "metavision/hal/plugin/plugin_entrypoint.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/sdk/base/utils/string.h"

namespace {

#ifdef _WIN32
static void dlclose(void *library) {
    FreeLibrary((HMODULE)library);
    return;
}

void Error(LPTSTR lpszFunction) {
    // Retrieve the system error message for the last-error code

    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError();

    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, dw,
                  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, NULL);

    // Display the error message and exit the process

    lpDisplayBuf = (LPVOID)LocalAlloc(
        LMEM_ZEROINIT, (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR));
    StringCchPrintf((LPTSTR)lpDisplayBuf, LocalSize(lpDisplayBuf) / sizeof(TCHAR), TEXT("%s failed with error %d: %s"),
                    lpszFunction, dw, lpMsgBuf);
    MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
}
#endif

// we only show error messages if a specific environment variable is set
// this is because we cannot assume that all shared libraries found in the plugin path
// are plugins for HAL, for MSVC, it will happen to be required DLLs that we shouldn't
// try to load, or at least we should not consider it an error when loading those.
inline void showErrorMsg(const std::string &error_msg) {
    char *p = getenv("MV_HAL_DEBUG_PLUGIN");

    if (p) {
        MV_HAL_LOG_WARNING() << error_msg;
#ifdef _WIN32
        Error((LPTSTR)error_msg.c_str());
#endif
    }
}

// Gets a plugin name
//
// Given :
//  - a filename
//  - an extension
//  - (optionally) a prefix, to be removed from filename if present
//  - (optionally) a suffix, to be removed from filename if should_have_suffix = true or simply to make sure it does not
//    exist otherwise
// Returns plugin where filename = [prefix]plugin[suffix if should_have_suffix][.extension]
std::string get_plugin_name(const std::string &filename, const std::string &ext, const std::string &prefix = "lib",
                            const std::string &suffix = "", bool should_have_suffix = false) {
    std::string name;
    if (filename.length() >= prefix.length() + ext.length() + suffix.length() + 1) {
        if (filename.substr(filename.length() - ext.length() - 1) == ("." + ext)) {
            if (prefix.empty() || (filename.substr(0, prefix.length()) == prefix)) {
                if (suffix.empty() ||
                    (should_have_suffix == (filename.substr(filename.length() - ext.length() - suffix.length() - 1) ==
                                            (suffix + "." + ext)))) {
                    name = filename.substr(prefix.length(), filename.length() - ext.length() - prefix.length() -
                                                                (should_have_suffix ? suffix.length() : 0) - 1);
                }
            }
        }
    }

    return name;
}

using PluginEntry = decltype(&initialize_plugin);

struct dlcloser {
    void operator()(void *handle) {
        if (handle) {
            dlclose(handle);
        }
    }
};
} // namespace

namespace Metavision {

struct PluginLoader::PluginInfo {
    PluginInfo(const std::filesystem::path &folder, const std::string &filename) {
#ifdef _WIN32
#ifdef _DEBUG
        name = get_plugin_name(filename, "dll", "", "_d", true);
#else
        name = get_plugin_name(filename, "dll", "", "_d", false);
#endif
#elif defined __APPLE__
        name = get_plugin_name(filename, "dylib");
#else
        name = get_plugin_name(filename, "so");
#endif
        path = folder / filename;
    }

    std::string name;
    std::filesystem::path path;
};

struct PluginLoader::Library {
    Library(const std::string &entrypoint_name, const std::string &name, const std::filesystem::path &path) :
        handle(load_library(path)) {
        if (handle && !entrypoint_name.empty()) {
            auto entrypoint = reinterpret_cast<PluginEntry>(load_entrypoint(handle.get(), entrypoint_name.c_str()));
            if (entrypoint) {
                plugin = PluginLoader::make_plugin(name);
                entrypoint(plugin.get());
            }
        }
    }

#ifdef _WIN32
    void *load_library(const std::filesystem::path &path) {
        const std::filesystem::path dir = std::filesystem::canonical(path).parent_path();

        LPSTR old_dll_dir = nullptr;
        if (!dir.empty()) {
            // Save old dll directory
            DWORD size = GetDllDirectoryA(0, old_dll_dir);
            if (size > 0) {
                old_dll_dir = new char[size + 1];
                GetDllDirectoryA(size + 1, old_dll_dir);
                if (std::string(old_dll_dir) == "") {
                    delete[] old_dll_dir;
                    old_dll_dir = nullptr;
                }
            }
            SetDllDirectoryA(dir.string().c_str());
        }

        void *handler = (void *)LoadLibraryA(std::filesystem::canonical(path).string().c_str());
        if (handler == nullptr) {
            showErrorMsg("LoadLibraryA");
        }

        if (!dir.empty()) {
            // Restore dll directory
            SetDllDirectoryA(old_dll_dir);
        }

        if (old_dll_dir) {
            delete[] old_dll_dir;
        }

        return handler;
    }

    void *load_entrypoint(void *handler, const char *name) const {
        void *entrypoint = (void *)GetProcAddress((HMODULE)handler, name);
        if (entrypoint == nullptr) {
            showErrorMsg("GetProcAddress");
        }
        return entrypoint;
    }
#else
    void *load_library(const std::filesystem::path &path) {
        dlerror();
        void *handler = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);
        if (handler == nullptr) {
            showErrorMsg(std::string("dlopen error: ") + std::string(dlerror()));
        }
        return handler;
    }

    void *load_entrypoint(void *handler, const char *name) const {
        dlerror();
        void *entrypoint = dlsym(handler, name);
        if (entrypoint == nullptr) {
            showErrorMsg(std::string("dlsym error: ") + std::string(dlerror()));
        }
        return entrypoint;
    }
#endif

    // order is important here, we need to delete the plugin before we delete the handle which will
    // close the shared library
    std::unique_ptr<void, dlcloser> handle;
    std::unique_ptr<Metavision::Plugin> plugin;
};

PluginLoader::PluginLoader() = default;

PluginLoader::~PluginLoader() = default;

void PluginLoader::clear_folders() {
    folders_.clear();
}

void PluginLoader::insert_folder(const std::filesystem::path &folder) {
    if (std::find(folders_.begin(), folders_.end(), folder) == folders_.end()) {
        folders_.push_back(folder);
    }
}

void PluginLoader::insert_folders(const std::vector<std::string> &folders) {
    for (const auto &folder : folders) {
        insert_folder(folder);
    }
}

void PluginLoader::insert_folders(const std::vector<std::filesystem::path> &folders) {
    for (const auto &folder : folders) {
        insert_folder(folder);
    }
}

void PluginLoader::load_plugins() {
    for (auto folder : folders_) {
        if (std::filesystem::is_directory(folder)) {
            for (auto const &dir_entry : std::filesystem::directory_iterator(folder)) {
                auto plugin_info = PluginInfo(folder, dir_entry.path().filename().string());
                insert_plugin(plugin_info);
            }
        }
    }
}

void PluginLoader::unload_plugins() {
    folders_.clear();
    libraries_.clear();
}

void PluginLoader::insert_plugin(const std::string &name, const std::filesystem::path &library_path) {
    if (!name.empty() && !library_path.empty()) {
        auto library = std::make_unique<Library>(get_plugin_entry_point(), name, library_path);
        if (library->plugin) {
            libraries_.push_back(std::move(library));
        }
    }
}

void PluginLoader::insert_plugin(const PluginInfo &info) {
    insert_plugin(info.name, info.path);
}

std::unique_ptr<Plugin> PluginLoader::make_plugin(const std::string &plugin_name) {
    return std::unique_ptr<Plugin>(new Plugin(plugin_name));
}

PluginLoader::PluginList::iterator::iterator(const typename container::iterator &it) : it_(it) {}

bool PluginLoader::PluginList::iterator::operator!=(const iterator &it) const {
    return it_ != it.it_;
}

PluginLoader::PluginList::iterator &PluginLoader::PluginList::iterator::operator++() {
    ++it_;
    return *this;
}

PluginLoader::PluginList::iterator::reference PluginLoader::PluginList::iterator::operator*() const {
    return *(*it_)->plugin;
}

PluginLoader::PluginList::iterator::pointer PluginLoader::PluginList::iterator::operator->() const {
    return (*it_)->plugin.get();
}

PluginLoader::PluginList::iterator PluginLoader::PluginList::begin() {
    return iterator(libraries_.begin());
}

PluginLoader::PluginList::iterator PluginLoader::PluginList::end() {
    return iterator(libraries_.end());
}

PluginLoader::PluginList::PluginList(container &libraries) : libraries_(libraries) {}

bool PluginLoader::PluginList::empty() const {
    return libraries_.empty();
}

size_t PluginLoader::PluginList::size() const {
    return libraries_.size();
}

PluginLoader::PluginList PluginLoader::get_plugin_list() {
    return PluginList(libraries_);
}

} // namespace Metavision
