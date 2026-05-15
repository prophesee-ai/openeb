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

#include "metavision/psee_hw_layer/boards/treuzell/tz_control_frame.h"
#include "boards/treuzell/treuzell_command_definition.h"
#include <stdexcept>
#include <cstring>

namespace Metavision {

struct ctrlFrameHeader {
    uint32_t property;
    uint32_t size;
};

TzCtrlFrame::TzCtrlFrame(uint32_t property) {
    struct ctrlFrameHeader *frame;
    vect.resize(8);
    frame           = reinterpret_cast<ctrlFrameHeader *>(vect.data());
    frame->property = property;
    frame->size     = 0;
}

// do nothing. children classes will manage it
TzCtrlFrame::TzCtrlFrame() {}

TzCtrlFrame::~TzCtrlFrame() {}

uint32_t TzCtrlFrame::get_property() {
    struct ctrlFrameHeader *frame = reinterpret_cast<ctrlFrameHeader *>(vect.data());
    return frame->property;
}

uint8_t *TzCtrlFrame::frame() {
    update_size();
    return vect.data();
}

std::size_t TzCtrlFrame::frame_size() {
    return vect.size();
}

void TzCtrlFrame::swap_and_check_answer(std::vector<uint8_t> &x) {
    uint32_t req_property = get_property();
    struct ctrlFrameHeader *frame;

    if (x.size() < sizeof(ctrlFrameHeader))
        throw std::system_error(TZ_TOO_SHORT, TzError());

    vect.swap(x);
    frame = reinterpret_cast<ctrlFrameHeader *>(vect.data());
    if (frame->size != (vect.size() - sizeof(ctrlFrameHeader)))
        throw std::system_error(TZ_SIZE_MISMATCH, TzError());
    if (frame->property == TZ_UNKNOWN_CMD)
        throw std::system_error(TZ_NOT_IMPLEMENTED, TzError());
    if (frame->property == (req_property | TZ_FAILURE_FLAG))
        throw std::system_error(TZ_COMMAND_FAILED, TzError());
    if (frame->property != req_property)
        throw std::system_error(TZ_PROPERTY_MISMATCH, TzError());
}

void TzCtrlFrame::update_size() {
    int32_t payload_size = vect.size() - sizeof(ctrlFrameHeader);
    if (payload_size < 0)
        throw std::length_error("payload resized to less than 0");
    struct ctrlFrameHeader *frame = reinterpret_cast<ctrlFrameHeader *>(vect.data());
    frame->size                   = payload_size;
}

/*
 * Treuzell generic frame with most knowledge in the caller
 */

TzGenericCtrlFrame::TzGenericCtrlFrame(uint32_t property) : TzCtrlFrame(property) {}

TzGenericCtrlFrame::~TzGenericCtrlFrame() {}

uint8_t *TzGenericCtrlFrame::payload() {
    return vect.data() + sizeof(ctrlFrameHeader);
}

std::size_t TzGenericCtrlFrame::payload_size() {
    int32_t payload_size = vect.size() - sizeof(ctrlFrameHeader);
    if (payload_size < 0)
        throw std::length_error("payload resized to less than 0");
    return payload_size;
}

void TzGenericCtrlFrame::push_back32(const uint32_t &val) {
    vect.push_back(val & 0xFF);
    vect.push_back((val >> 8) & 0xFF);
    vect.push_back((val >> 16) & 0xFF);
    vect.push_back((val >> 24) & 0xFF);
}

void TzGenericCtrlFrame::push_back32(const std::vector<uint32_t> &src) {
    vect.reserve(vect.size() + (sizeof(uint32_t) * src.size()));
    for (auto const &val : src)
        push_back32(val);
}

uint32_t TzGenericCtrlFrame::get32(std::size_t index) {
    if (payload_size() < ((index + 1) * (sizeof(uint32_t))))
        throw std::system_error(TZ_TOO_SHORT, TzError());
    return *((uint32_t *)(payload() + (index * sizeof(uint32_t))));
}

uint64_t TzGenericCtrlFrame::get64(std::size_t index) {
    if (payload_size() < ((index + 1) * (sizeof(uint64_t))))
        throw std::system_error(TZ_TOO_SHORT, TzError());
    return *((uint64_t *)(payload() + (index * sizeof(uint64_t))));
}

TzDeviceStringsCtrlFrame::TzDeviceStringsCtrlFrame(uint32_t property, uint32_t device) : TzCtrlFrame(property) {
    vect.push_back(device & 0xFF);
    vect.push_back((device >> 8) & 0xFF);
    vect.push_back((device >> 16) & 0xFF);
    vect.push_back((device >> 24) & 0xFF);
}

std::vector<std::string> TzDeviceStringsCtrlFrame::get_strings() {
    std::vector<std::string> res;

    // TzCtrlFrame already checked request status
    uint8_t *frame        = vect.data();
    std::size_t remaining = vect.size();
    frame += sizeof(ctrlFrameHeader);
    remaining -= sizeof(ctrlFrameHeader);

    // check if the frame is large enough to contain the device id
    if (remaining < sizeof(uint32_t))
        throw std::system_error(TZ_TOO_SHORT, TzError());

    frame += sizeof(uint32_t);
    remaining -= sizeof(uint32_t);

    // the frame shall at least contain the NULL terminator
    if (!remaining)
        throw std::system_error(TZ_TOO_SHORT, TzError());
    if (*(frame + remaining - 1) != '\0')
        throw std::system_error(TZ_INVALID_ANSWER, TzError(), "compatible string shall be NULL-terminated");

    while (remaining) {
        std::string str((char *)frame);
        // str.size() doesn't include a NULL terminator
        frame += str.size() + 1;
        remaining -= str.size() + 1;
        res.push_back(str);
    }

    return res;
}

void TzDeviceStringsCtrlFrame::push_back(const std::string &str) {
    auto size = vect.size();
    vect.resize(vect.size() + str.size() + 1);
    memcpy(vect.data() + size, str.c_str(), str.size() + 1);
}

} // namespace Metavision
