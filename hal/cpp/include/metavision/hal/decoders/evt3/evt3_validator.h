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

#ifndef METAVISION_HAL_EVT3_VALIDATOR_H
#define METAVISION_HAL_EVT3_VALIDATOR_H

#include <functional>
#include <map>
#include <ostream>
#include <string>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/utils/decoder_protocol_violation.h"
#include "metavision/hal/utils/hal_error_code.h"
#include "metavision/hal/decoders/evt3/evt3_event_types.h"

namespace Metavision {
namespace decoder {
namespace evt3 {

template<class SelfType>
class ValidatorInterface {
protected:
    std::map<size_t, I_Decoder::ProtocolViolationCallback_t> notifiers_map_;
    size_t next_cb_idx_{0};

public:
    constexpr static int TIME_HIGH_MAX_VALUE = 0xFFF;
    constexpr static int LOOSE_TIME_HIGH_OVERFLOW_EPSILON =
        0xFF; // Arbitrary value to distinguish data swap from data drop

    ValidatorInterface(int height, int width) {}

    size_t add_protocol_violation_callback(const I_Decoder::ProtocolViolationCallback_t &cb) {
        notifiers_map_[next_cb_idx_] = cb;
        return next_cb_idx_++;
    }

    bool remove_protocol_violation_callback(size_t callback_id) {
        auto it = notifiers_map_.find(callback_id);
        if (it != notifiers_map_.end()) {
            notifiers_map_.erase(it);
            return true;
        }
        return false;
    }

    void notify(DecoderProtocolViolation violation) {
        if (notifiers_map_.empty()) {
            std::ostringstream oss;
            oss << "Evt3 protocol violation detected : " << violation;
            if (violation == DecoderProtocolViolation::NonMonotonicTimeHigh) {
                MV_HAL_LOG_ERROR() << oss.str();
            } else {
                MV_HAL_LOG_WARNING() << oss.str();
            }
        } else {
            for (auto &it : notifiers_map_) {
                it.second(violation);
            }
        }
    }

    bool validate_event_cd(const Evt3Raw::RawEvent *raw_events) {
        return static_cast<SelfType *>(this)->validate_event_cd_impl(raw_events);
    }

    bool validate_ext_trigger(const Evt3Raw::RawEvent *raw_events) {
        return static_cast<SelfType *>(this)->validate_ext_trigger_impl(raw_events);
    }

    bool has_valid_vect_base() {
        return static_cast<SelfType *>(this)->has_valid_vect_base_impl();
    }

    bool validate_vect_12_12_8_pattern(const Evt3Raw::RawEvent *raw_events, unsigned vect_base,
                                       int &next_valid_offset) {
        return static_cast<SelfType *>(this)->validate_vect_12_12_8_pattern_impl(raw_events, vect_base,
                                                                                 next_valid_offset);
    }

    bool validate_continue_12_12_4_pattern(const Evt3Raw::RawEvent *raw_events, int &next_valid_offset) {
        return static_cast<SelfType *>(this)->validate_continue_12_12_4_pattern_impl(raw_events, next_valid_offset);
    }

    void validate_time_high(timestamp prev_time_high, timestamp time_high) {
        static_cast<SelfType *>(this)->validate_time_high_impl(prev_time_high, time_high);
    }

    void state_update(const Evt3Raw::RawEvent *raw_event) {
        static_cast<SelfType *>(this)->state_update_impl(raw_event);
    }

protected:
    static bool is_strict_time_high_overflow(timestamp prev_time_high, timestamp time_high) {
        return prev_time_high == TIME_HIGH_MAX_VALUE && time_high == 0;
    }
    static bool is_loose_time_high_overflow(timestamp prev_time_high, timestamp time_high) {
        return (time_high - prev_time_high + TIME_HIGH_MAX_VALUE) < LOOSE_TIME_HIGH_OVERFLOW_EPSILON;
    }
};

class NullCheckValidator : public ValidatorInterface<NullCheckValidator> {
public:
    NullCheckValidator(int height, int width) : ValidatorInterface<NullCheckValidator>(height, width) {}

    bool validate_event_cd_impl(const Evt3Raw::RawEvent *raw_events) {
        return true;
    }

    bool validate_ext_trigger_impl(const Evt3Raw::RawEvent *raw_events) {
        return true;
    }

    bool has_valid_vect_base_impl() {
        return true;
    }

    bool validate_vect_12_12_8_pattern_impl(const Evt3Raw::RawEvent *raw_events, unsigned vect_base,
                                            int &next_valid_offset) {
        next_valid_offset = sizeof(Evt3Raw::Event_Vect12_12_8) / sizeof(Evt3Raw::RawEvent);
        return true;
    }

    bool validate_continue_12_12_4_pattern_impl(const Evt3Raw::RawEvent *raw_events, int &next_valid_offset) {
        next_valid_offset = sizeof(Evt3Raw::Event_Continue12_12_4) / sizeof(Evt3Raw::RawEvent);
        return true;
    }

    void validate_time_high_impl(timestamp prev_time_high, timestamp time_high) {}

    void state_update_impl(const Evt3Raw::RawEvent *raw_event) {}
};

class BasicCheckValidator : public ValidatorInterface<BasicCheckValidator> {
    uint32_t width_;
    bool has_vect_base_ = false;

public:
    BasicCheckValidator(int height, int width) :
        ValidatorInterface<BasicCheckValidator>(height, width), width_(width) {}

    bool validate_event_cd_impl(const Evt3Raw::RawEvent *raw_events) {
        return true;
    }

    bool validate_ext_trigger_impl(const Evt3Raw::RawEvent *raw_events) {
        return true;
    }

    bool has_valid_vect_base_impl() {
        return true;
    }

    bool validate_vect_12_12_8_pattern_impl(const Evt3Raw::RawEvent *raw_events, unsigned vect_base,
                                            int &next_valid_offset) {
        next_valid_offset = sizeof(Evt3Raw::Event_Vect12_12_8) / sizeof(Evt3Raw::RawEvent);

        if (!has_vect_base_ || vect_base + 32 > width_) {
            has_vect_base_ = false;
            notify(DecoderProtocolViolation::InvalidVectBase);
            return false;
        }

        return true;
    }

    bool validate_continue_12_12_4_pattern_impl(const Evt3Raw::RawEvent *raw_events, int &next_valid_offset) {
        next_valid_offset = sizeof(Evt3Raw::Event_Continue12_12_4) / sizeof(Evt3Raw::RawEvent);
        return true;
    }

    void validate_time_high_impl(timestamp prev_time_high, timestamp time_high) {
        if (is_strict_time_high_overflow(prev_time_high, time_high)) {
            return;
        }

        int timehigh_delta = time_high - prev_time_high;
        bool is_monotonic  = 0 <= timehigh_delta;
        if (!is_monotonic && !is_loose_time_high_overflow(prev_time_high, time_high)) {
            notify(DecoderProtocolViolation::NonMonotonicTimeHigh);
        }
    }

    void state_update_impl(const Evt3Raw::RawEvent *raw_event) {
        if (raw_event->type == uint8_t(Evt3EventTypes_4bits::VECT_BASE_X)) {
            has_vect_base_ = true;
        }
    }
};

class GrammarValidator : public ValidatorInterface<GrammarValidator> {
    uint32_t height_;
    uint32_t width_;
    bool is_valid_time_high_ = false;
    bool has_addr_y_         = false;
    bool has_vect_base_      = false;

public:
    GrammarValidator(int height, int width) :
        ValidatorInterface<GrammarValidator>(height, width), height_(height), width_(width) {}

    bool validate_event_cd_impl(const Evt3Raw::RawEvent *raw_events) {
        if (!has_addr_y_) {
            notify(DecoderProtocolViolation::MissingYAddr);
            return false;
        }
        return is_valid_time_high_;
    }

    bool validate_ext_trigger_impl(const Evt3Raw::RawEvent *raw_events) {
        return is_valid_time_high_;
    }

    bool has_valid_vect_base_impl() {
        return has_vect_base_;
    }

    bool validate_vect_12_12_8_pattern_impl(const Evt3Raw::RawEvent *raw_events, unsigned vect_base,
                                            int &next_valid_offset) {
        next_valid_offset = 0;
        if ((raw_events + 1)->type != uint8_t(Evt3EventTypes_4bits::VECT_12)) {
            next_valid_offset = 1;
        } else if ((raw_events + 2)->type != uint8_t(Evt3EventTypes_4bits::VECT_8)) {
            next_valid_offset = 2;
        }

        if (next_valid_offset > 0) {
            notify(DecoderProtocolViolation::PartialVect_12_12_8);
            has_vect_base_ = false;
            return false;
        }

        next_valid_offset = sizeof(Evt3Raw::Event_Vect12_12_8) / sizeof(Evt3Raw::RawEvent);

        if (!has_vect_base_ || vect_base + 32 > width_) {
            has_vect_base_ = false;
            notify(DecoderProtocolViolation::InvalidVectBase);
            return false;
        }

        if (!has_addr_y_) {
            notify(DecoderProtocolViolation::MissingYAddr);
            return false;
        }

        return is_valid_time_high_;
    }

    bool validate_continue_12_12_4_pattern_impl(const Evt3Raw::RawEvent *raw_events, int &next_valid_offset) {
        next_valid_offset = 0;
        if ((raw_events)->type != uint8_t(Evt3EventTypes_4bits::CONTINUED_12)) {
            notify(DecoderProtocolViolation::PartialContinued_12_12_4);
            return false;
        } else if ((raw_events + 1)->type != uint8_t(Evt3EventTypes_4bits::CONTINUED_12)) {
            next_valid_offset = 1;
        } else if ((raw_events + 2)->type != uint8_t(Evt3EventTypes_4bits::CONTINUED_4)) {
            next_valid_offset = 2;
        }

        if (next_valid_offset > 0) {
            notify(DecoderProtocolViolation::PartialContinued_12_12_4);
            return false;
        }

        next_valid_offset = sizeof(Evt3Raw::Event_Continue12_12_4) / sizeof(Evt3Raw::RawEvent);

        return is_valid_time_high_;
    }

    void validate_time_high_impl(timestamp prev_time_high, timestamp time_high) {
        int timehigh_delta = time_high - prev_time_high;
        bool is_monotonic  = 0 <= timehigh_delta;

        is_valid_time_high_ = is_monotonic || is_loose_time_high_overflow(prev_time_high, time_high);

        if (is_strict_time_high_overflow(prev_time_high, time_high)) {
            return;
        }

        if (!is_valid_time_high_) {
            notify(DecoderProtocolViolation::NonMonotonicTimeHigh);
        } else if (timehigh_delta != 0 && timehigh_delta != 1) {
            notify(DecoderProtocolViolation::NonContinuousTimeHigh);
        }
    }

    void state_update_impl(const Evt3Raw::RawEvent *raw_event) {
        if (raw_event->type == uint8_t(Evt3EventTypes_4bits::EVT_ADDR_Y)) {
            has_addr_y_ = true;
        }
        if (raw_event->type == uint8_t(Evt3EventTypes_4bits::VECT_BASE_X)) {
            has_vect_base_ = true;
        }
    }
};

} // namespace evt3
} // namespace decoder
} // namespace Metavision

#endif // METAVISION_HAL_EVT3_VALIDATOR_H
