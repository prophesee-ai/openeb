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

#ifndef PSEE_FORMAT_H
#define PSEE_FORMAT_H

#include <cstdint>
#include <memory>
#include <string>
#include <map>

namespace Metavision {

class I_Geometry;

class StreamFormat {
public:
    StreamFormat(std::string format);
    std::string name() const;
    std::string to_string() const;
    std::unique_ptr<I_Geometry> geometry() const;

    // accessor to format options, such as width, heigth, ...
    bool contains(const std::string &name) const;
    std::string &operator[](const std::string &name);
    const std::string &operator[](const std::string &name) const;

private:
    std::string format_name;
    std::map<std::string, std::string> options;
};

} // namespace Metavision
#endif /* PSEE_FORMAT_H */
