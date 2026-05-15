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

#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>

#include "metavision/sdk/base/utils/log.h"
#include "metavision/sdk/base/utils/generic_header.h"

namespace Metavision {
namespace {

static const std::string field_prefix           = "%";
static const std::vector<std::string> date_keys = {"date", "Date"};
static const std::string end_key                = "end";
} // namespace

GenericHeader::GenericHeader() = default;

GenericHeader::GenericHeader(std::istream &stream) {
    parse_header(stream);
}

GenericHeader::GenericHeader(const HeaderMap &header) : header_(header) {}

bool GenericHeader::empty() const {
    return header_.empty();
}

void GenericHeader::add_date() {
    static const std::string format = "%Y-%m-%d %H:%M:%S";
    std::time_t tt                  = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct std::tm *ptm             = std::localtime(&tt);
    std::stringstream date_str;
    date_str << std::put_time(ptm, format.c_str());
    remove_date();
    set_field(date_keys[0], date_str.str());
}

void GenericHeader::remove_date() {
    for (const auto &key : date_keys) {
        remove_field(key);
    }
}

std::string GenericHeader::get_date() const {
    std::string date;
    for (const auto &key : date_keys) {
        date = get_field(key);
        if (!date.empty()) {
            return date;
        }
    }
    return date;
}

void GenericHeader::set_field(const std::string &key, const std::string &value) {
    if (key.empty()) {
        MV_LOG_ERROR() << "When setting field in header map: can not set field with empty key.";
        return;
    }

    header_[key] = value;
}

void GenericHeader::remove_field(const std::string &key) {
    header_.erase(key);
}

std::string GenericHeader::get_field(const std::string &key) const {
    auto it = header_.find(key);
    return (it == header_.end()) ? "" : it->second;
}

const GenericHeader::HeaderMap &GenericHeader::get_header_map() const {
    return header_;
}

std::string GenericHeader::to_string() const {
    std::string header_str;

    for (const auto &field : header_) {
        header_str += field_prefix + " " + field.first + " " + field.second + "\n";
    }

    header_str += field_prefix + " " + end_key + "\n";

    return header_str;
}

bool GenericHeader::check_prefix_and_read_header_line(std::istream &stream) {
    if (!stream) {
        return false;
    }

    // no feedling with seekg and tellg, they don't have consistent behavior on Windows
    int c = stream.peek();
    if (c == field_prefix[0]) {
        stream.get();
        c = stream.peek();
        if (c == ' ') {
            stream.get();
            return true;
        } else if (c != EOF) {
            stream.unget();
        }
    }

    return false;
}

void GenericHeader::parse_header(std::istream &stream) {
    while (check_prefix_and_read_header_line(stream)) {
        // In order for the value to be insert in the map, the line of the header has to be:
        // % Key value
        std::string line;
        std::string tmp_key;
        if (std::getline(stream, line)) {
            std::vector<std::string> vec;
            std::istringstream iss(line);

            // After a call to is_next_line_commented, the first two characters have been read already
            // so we expect the key to be the first extracted field
            std::string key, value;
            if (iss >> key) {
                if (key == end_key) {
                    break;
                }
                std::string tot_value;
                iss >> tot_value;
                while (iss >> value) {
                    tot_value += " " + value;
                }
                header_[key] = tot_value;
            }
        }
    }
}

std::ostream &operator<<(std::ostream &output, const GenericHeader &header) {
    output << header.to_string();
    return output;
}

} // namespace Metavision
