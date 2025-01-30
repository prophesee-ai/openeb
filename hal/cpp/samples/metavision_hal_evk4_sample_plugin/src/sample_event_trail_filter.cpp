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

#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>


#include <metavision/hal/utils/hal_exception.h>
#include "sample_event_trail_filter.h"
#include "internal/sample_register_access.h"


SampleEventTrailFilter::SampleEventTrailFilter(std::shared_ptr<SampleUSBConnection> usb_connection) :
    usb_connection_(usb_connection) {}

std::set<Metavision::I_EventTrailFilterModule::Type> SampleEventTrailFilter::get_available_types() const {
    return {Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL, Metavision::I_EventTrailFilterModule::Type::STC_KEEP_TRAIL,
            Metavision::I_EventTrailFilterModule::Type::TRAIL};
}

bool SampleEventTrailFilter::set_threshold(uint32_t threshold) {
    if (threshold < get_min_supported_threshold() || threshold > get_max_supported_threshold()) {
        std::stringstream ss;
        ss << "Bad STC threshold value: " << threshold << ". Value should be in range [1000, 100000].";
    }

    threshold_ms_ = std::roundf(threshold / 1000.0);

    // Reset if needed
    if (is_enabled()) {
        enable(false);
        enable(true);
    }

    return true;
}

bool SampleEventTrailFilter::set_type(Metavision::I_EventTrailFilterModule::Type type) {
    filtering_type_ = type;
    // Reset if needed
    if (is_enabled()) {
        enable(false);
        enable(true);
    }
    return true;
}

bool SampleEventTrailFilter::enable(bool state) {
    // We write 101 (5) to stc/pipeline_control to bypass the block
    write_register(*usb_connection_, 0x0000D000, 0x05);
    enabled_ = false;

    if (!state) {
        return true;
    }

    if (filtering_type_ == I_EventTrailFilterModule::Type::STC_CUT_TRAIL ||
        filtering_type_ == I_EventTrailFilterModule::Type::STC_KEEP_TRAIL) {
        // stc_enable to 1
        write_register(*usb_connection_, 0x0000D004, 0x01);
        // trail_param to 0
        write_register(*usb_connection_, 0x0000D008, 0x00);
        // For a fully functional facility, stc_threshold and disable_stc_cut_trail should also be updated

    } else if (filtering_type_ == Metavision::I_EventTrailFilterModule::Type::TRAIL) {
        // stc_enable to 0
        write_register(*usb_connection_, 0x0000D004, 0x00);
        // trail_param to 1
        write_register(*usb_connection_, 0x0000D008, 0x01);
        // For a fully functional facility, trail_threshold should also be updated
    }

    // We write 101 (5) to stc/pipeline_control to enable the block
    write_register(*usb_connection_, 0x0000D000, 0x01);
    enabled_ = true;
    return true;
}

bool SampleEventTrailFilter::is_enabled() const {
    return enabled_;
}

Metavision::I_EventTrailFilterModule::Type SampleEventTrailFilter::get_type() const {
    return filtering_type_;
}

uint32_t SampleEventTrailFilter::get_threshold() const {
    return threshold_ms_ * 1000;
}

uint32_t SampleEventTrailFilter::get_max_supported_threshold() const {
    return 100000;
}

uint32_t SampleEventTrailFilter::get_min_supported_threshold() const {
    return 1000;
}
