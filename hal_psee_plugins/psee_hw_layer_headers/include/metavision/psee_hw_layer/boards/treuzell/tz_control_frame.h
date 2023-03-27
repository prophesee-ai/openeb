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

#ifndef TZ_CONTROL_FRAME_H
#define TZ_CONTROL_FRAME_H

#include <system_error>
#include <cstdint>
#include <vector>

namespace Metavision {

class TzCtrlFrame {
public:
    TzCtrlFrame(uint32_t property);
    virtual ~TzCtrlFrame();

    virtual uint32_t get_property();
    virtual uint8_t *frame();
    virtual std::size_t frame_size();
    virtual void swap_and_check_answer(std::vector<uint8_t> &x);

protected:
    std::vector<uint8_t> vect;
    TzCtrlFrame();

private:
    void update_size(void);
};

class TzGenericCtrlFrame : public TzCtrlFrame {
public:
    TzGenericCtrlFrame(uint32_t property);
    virtual ~TzGenericCtrlFrame();

    virtual uint8_t *payload();
    virtual std::size_t payload_size();

    void push_back32(const uint32_t &val);
    void push_back32(const std::vector<uint32_t> &val);
    uint32_t get32(std::size_t payload_index);
    uint64_t get64(std::size_t payload_index);
};

class TzDeviceStringsCtrlFrame : public TzCtrlFrame {
public:
    TzDeviceStringsCtrlFrame(uint32_t property, uint32_t device);
    std::vector<std::string> get_strings();
    void push_back(const std::string &);
};

/// Treuzell protocol error category
enum {
    TZ_NOT_IMPLEMENTED,
    TZ_COMMAND_FAILED,
    TZ_PROPERTY_MISMATCH,
    TZ_SIZE_MISMATCH,
    TZ_TOO_SHORT,
    TZ_INVALID_ANSWER,
};

class TzError : public std::error_category {
public:
    virtual const char *name() const noexcept {
        return "treuzell";
    }
    virtual std::string message(int err) const {
        switch (err) {
        case TZ_NOT_IMPLEMENTED:
            return "command not implemented by the target";
        case TZ_COMMAND_FAILED:
            return "command failed";
        case TZ_PROPERTY_MISMATCH:
            return "received frame didn't match request";
        case TZ_SIZE_MISMATCH:
            return "received frame doesn't match its advertised size";
        case TZ_TOO_SHORT:
            return "frame too short to be valid";
        case TZ_INVALID_ANSWER:
            return "received answer doesn't match the specified format";
        default:
            return "unknown error";
        }
    }
};
} // namespace Metavision
#endif /* TZ_CONTROL_FRAME_H */
