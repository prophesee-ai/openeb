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

#ifndef METAVISION_SDK_BASE_PYTHON_BINDINGS_DOC_H
#define METAVISION_SDK_BASE_PYTHON_BINDINGS_DOC_H

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <vector>

namespace Metavision {

/// @brief This class is used to provide documentation for python bindings
class PythonBindingsDoc {
public:
    /// @brief Default constructor
    PythonBindingsDoc() = default;

    /// @brief Constructs the map from a sequence of strings
    ///
    /// The sequence must contain an even number of elements
    ///
    /// @param sequence_of_strings should be in the order: key1, val1, key2, val2, ... keyn, valn
    PythonBindingsDoc(const std::vector<std::string> &sequence_of_strings) {
        if (sequence_of_strings.size() % 2 != 0) {
            std::ostringstream oss;
            oss << "Error: expecting a flat vector of key/value pairs. Size should be even ";
            oss << "(size of current vector is: " << sequence_of_strings.size() << ")" << std::endl;
            throw std::logic_error(oss.str());
        }
        auto nb_keys = sequence_of_strings.size() / 2;
        for (uint32_t idx = 0; idx < nb_keys; ++idx) {
            map_doc_[sequence_of_strings[2 * idx]] = sequence_of_strings[2 * idx + 1];
        }
    }

    /// @brief Returns the documentation of the given class or method
    ///
    /// To access documentation for the MethodYXZ of MyClass, access the element :  ["Metavision::MyClass::MethodXYZ"]
    /// If the documentation was not built (or empty vector was provided in the constructor of this class), this
    /// function systematically returns a string to indicate the documentation is missing.
    ///
    /// If this class is queried with an invalid key, an exception is thrown
    ///
    /// If several methods have the same name but different arguments, using the name of the method will result in an
    /// exception being thrown with an error message to indicate the key is ambiguous. In this situation, use the
    /// full name of the method and its arguments to disambiguate.
    /// For example MyClass contains a MethodXYZ(float x) and another MethodXYZ(int x, float y), to access the
    /// documentation of the latter, one should use the key string : 'Metavision::MyClass::MethodXYZ(int x, float y)'
    ///
    /// @param key Name of the class or method
    /// @return Documentation string for that class or method
    const char *operator[](const std::string &key) const {
        if (map_doc_.empty()) {
            return "###########################################\n"
                   "#  PYTHON BINDINGS WITHOUT DOCUMENTATION  #\n"
                   "###########################################\n";
        }

        auto it = map_doc_.find(key);

        if (it == map_doc_.end()) {
            std::ostringstream oss;
            oss << "Error: invalid key for python documentation: " << key << std::endl;
            throw std::logic_error(oss.str());
        }

        const std::string &value = it->second;
        if (value == "... ... ...") {
            std::ostringstream oss;
            oss << "Error: ambiguous python documentation for: \n";
            oss << "\t <key> : '" << key << "'\n";
            oss << "\t <value> : '" << value << "'\n";
            oss << "Your <key> needs to be specified as it matches multiple definitions.\n";
            oss << "Update your cpp python bindings by replacing your <key> with one of the following options:\n";
            const auto start_with = [](const std::string &str, const std::string &prefix) {
                return str.substr(0, prefix.size()) == prefix;
            };
            std::for_each(++it, map_doc_.end(), [&](const auto it) {
                if (start_with(it.first, key + "(")) {
                    oss << "\t- \"" << it.first << "\"\n";
                }
            });
            throw std::logic_error(oss.str());
        }
        return value.c_str();
    }

private:
    std::map<std::string, std::string> map_doc_;
};

} // namespace Metavision

#endif // METAVISION_SDK_BASE_PYTHON_BINDINGS_DOC_H
