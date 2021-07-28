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

#ifndef METAVISION_UTILS_PYBIND_PYTHON_BINDER_HELPER_H
#define METAVISION_UTILS_PYBIND_PYTHON_BINDER_HELPER_H

#include <functional>
#include <algorithm>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <string>
#endif

namespace py = pybind11;

namespace Metavision {

namespace detail {

/// Returns hash code of type T
template<typename T>
size_t get_type_hash() {
    return typeid(T).hash_code();
}

/// The following struct stores the information about a class binded in python
struct python_class_definition {
    python_class_definition(const std::function<void(py::module &)> &f, size_t dc,
                            const std::vector<size_t> &ndc = std::vector<size_t>());

    std::function<void(py::module &)> f; ///< function that defines the python class
    size_t defined_class;                ///< hash code of the respective c++ class

    /// @brief vector containing the hash codes
    /// of all the c++ classes that need to
    /// be defined in python before the current one (the one whose hash
    /// corresponds to defined_class)
    std::vector<size_t> needed_defined_classes;
};

/// Returns all the functions that define a python class
template<typename tag>
std::vector<python_class_definition> &get_class_definitions_binding_cbs();

/// Returns all the functions that define generic python binding, other than classes (enum for example)
template<typename tag>
std::vector<std::function<void(py::module &)>> &get_generic_binding_cbs();

} // namespace detail

#ifdef _WIN32
static std::wstring stringTowstring(const std::string &s) {
    int len;
    int slength  = (int)s.length() + 1;
    len          = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
    wchar_t *buf = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
    std::wstring r(buf);
    delete[] buf;
    return r;
}

inline bool setPseeInstallPathsForBindings() {
    HKEY key;
    if (RegOpenKey(HKEY_LOCAL_MACHINE, TEXT("SOFTWARE\\Prophesee"), &key) == ERROR_SUCCESS) {
        DWORD value_length = 1024;
        char value[1024];
        DWORD dwType = REG_SZ;
        RegQueryValueEx(key, "INSTALL_PATH", NULL, &dwType, reinterpret_cast<BYTE *>(&value), &value_length);
        std::string installPath = value;

        AddDllDirectory(stringTowstring(installPath + "\\bin").c_str());
        AddDllDirectory(stringTowstring(installPath + "\\third_party\\bin").c_str());
        AddDllDirectory(stringTowstring(installPath + "\\third_party\\debug\bin").c_str());
        return true;
    }
    return false;
}

inline bool setBuildDirDllPathsForBindings() {
    DWORD attrs = GetFileAttributes(BUILD_BIN_DIR);
    if ((attrs != INVALID_FILE_ATTRIBUTES) && (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
        AddDllDirectory(stringTowstring(BUILD_BIN_DIR).c_str());
        return true;
    }
    return false;
}

inline bool setMetavisionDllPathsForBindings() {
    char *mv_dll_directory = getenv("MV_DLL_PATH");
    if (mv_dll_directory == nullptr) {
        return false;
    } else {
        AddDllDirectory(stringTowstring(mv_dll_directory).c_str());
        return true;
    }
}
#endif

/// @brief Class that handles the generic (i.e. not classes) python bindings
template<typename tag>
class GenericPythonBinderHelper {
public:
    virtual ~GenericPythonBinderHelper() {}

    GenericPythonBinderHelper(std::function<void(py::module &)> &&f) {
        detail::get_generic_binding_cbs<tag>().push_back(f);
    }
};

/// @brief Class that handles the generic python bindings for classes
///
/// ClassTypeDefined is the C++ class that is exported in python
/// Args are all the other templates that are passed to pybind11
///
/// This will define a python class in the following way :
///
///       pybind11::class_<ClassTypeDefined, Args...>
///
/// (so it's actually the same signature used normally when using pybind11 directly)
template<typename tag, typename ClassTypeDefined, typename... Args>
class ClassPythonBinderHelper {
public:
    virtual ~ClassPythonBinderHelper() {}

    /// BoostPythonClassCtorArgs are the arguments passed to the constructor of pybind11::class_
    template<typename... BoostPythonClassCtorArgs>
    ClassPythonBinderHelper(const std::function<void(py::module &, py::class_<ClassTypeDefined, Args...> &)> &f,
                            const BoostPythonClassCtorArgs &...args) {
        detail::get_class_definitions_binding_cbs<tag>().emplace_back(
            [=](py::module &m) {
                auto c = py::class_<ClassTypeDefined, Args...>(m, args...);
                f(m, c);
            },
            detail::get_type_hash<ClassTypeDefined>());
    }
};

/// @brief Class that handles the python class creation order
///
/// The python classes should be defined after the base classes
/// This class should be used to ensure the order
/// @warning Only bases that are defined inside the same module should be declared as it
template<class... Classes>
class PythonBases {
public:
    virtual ~PythonBases(){};

    PythonBases(Classes...){};
};

/// Handling class ClassPythonBinderHelper in the special case where the class that we want to export has Bases
///
/// @note While in Python we can use the template PythonBases<> in any position, here it must be the second
/// template, otherwise we would use the implementation above
template<typename tag, typename ClassTypeDefined, typename... Bases, typename... Args>
class ClassPythonBinderHelper<tag, ClassTypeDefined, PythonBases<Bases...>, Args...> {
public:
    virtual ~ClassPythonBinderHelper() {}

    template<typename... BoostPythonClassCtorArgs>
    ClassPythonBinderHelper(
        const std::function<void(py::module &, py::class_<ClassTypeDefined, Bases..., Args...> &)> &f,
        const BoostPythonClassCtorArgs &...args) {
        detail::get_class_definitions_binding_cbs<tag>().emplace_back(
            [=](py::module &m) {
                auto c = py::class_<ClassTypeDefined, Bases..., Args...>(m, args...);
                f(m, c);
            },
            detail::get_type_hash<ClassTypeDefined>(), get_hash<Bases...>());
    }

private:
    template<typename... Tn>
    typename std::enable_if<(sizeof...(Tn) == 0), std::vector<size_t>>::type get_hash() const {
        return {};
    }
    template<typename T, typename... Tn>
    typename std::enable_if<std::is_base_of<T, ClassTypeDefined>::value, std::vector<size_t>>::type get_hash() const {
        std::vector<size_t> vh = get_hash<Tn...>();
        vh.push_back(detail::get_type_hash<T>());
        return vh;
    }
};

namespace detail {

inline python_class_definition::python_class_definition(const std::function<void(py::module &)> &f, size_t dc,
                                                        const std::vector<size_t> &ndc) :
    f(f), defined_class(dc), needed_defined_classes(ndc) {}

template<typename tag>
std::vector<python_class_definition> &get_class_definitions_binding_cbs() {
    static std::vector<python_class_definition> class_definitions_binding_cbs;
    return class_definitions_binding_cbs;
}

template<typename tag>
std::vector<std::function<void(py::module &)>> &get_generic_binding_cbs() {
    static std::vector<std::function<void(py::module &)>> generic_binding_cbs;
    return generic_binding_cbs;
}

} // namespace detail

template<typename tag>
void export_python_bindings(py::module &module) {
    // Start with independent cbs
    const auto &indep_csb = detail::get_generic_binding_cbs<tag>();
    for (auto &f : indep_csb) {
        f(module);
    }

    // Now handle the cbs that require a certain call order
    auto class_defs_csb_origin = detail::get_class_definitions_binding_cbs<tag>();
    std::vector<detail::python_class_definition> class_defs_still_to_call;

    std::vector<size_t> already_defined_classes;

    while (true) {
        for (auto &pcd : class_defs_csb_origin) {
            bool can_export = true;

            // Check that all the needed classes are already been exported
            auto it_begin = already_defined_classes.begin();
            auto it_end   = already_defined_classes.end();
            for (auto hash : pcd.needed_defined_classes) {
                auto it = std::find(it_begin, it_end, hash);
                if (it == it_end) {
                    can_export = false;
                    break;
                }
            }
            if (can_export) {
                pcd.f(module);
                already_defined_classes.push_back(pcd.defined_class);
            } else {
                class_defs_still_to_call.push_back(pcd);
            }
        }
        if (class_defs_still_to_call.empty()) {
            break;
        }
        if (class_defs_still_to_call.size() == class_defs_csb_origin.size()) {
            break;
        }
        std::swap(class_defs_still_to_call, class_defs_csb_origin);
        class_defs_still_to_call.clear();
    }

    // If some base classes are defined in another module, we may have that !still_to_define.empty()
    for (auto &pcd : class_defs_still_to_call) {
        pcd.f(module);
    }
}
} // namespace Metavision

#endif // METAVISION_UTILS_PYBIND_PYTHON_BINDER_HELPER_H
