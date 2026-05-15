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

#ifndef METAVISION_HAL_EVT4_VALIDATOR_H
#define METAVISION_HAL_EVT4_VALIDATOR_H

#include <functional>
#include <map>
#include <ostream>
#include <string>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/utils/decoder_protocol_violation.h"
#include "metavision/hal/utils/hal_error_code.h"
#include "metavision/hal/decoders/evt4/evt4_event_types.h"

namespace Metavision {
namespace decoder {
namespace evt4 {

template<class SelfType>
class ValidatorInterface {
protected:
    std::map<size_t, I_Decoder::ProtocolViolationCallback_t> notifiers_map_;
    size_t next_cb_idx_{0};
    std::uint32_t width_;
    std::uint32_t height_;

public:
    ValidatorInterface(std::uint32_t width, std::uint32_t height) : height_(height), width_(width) {}

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
            oss << "Evt4 protocol violation detected: " << violation;
            if (violation == DecoderProtocolViolation::NonMonotonicTimeHigh ||
                violation == DecoderProtocolViolation::OutOfBoundsEventCoordinate) {
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

    void reset() {
        return static_cast<SelfType *>(this)->reset_impl();
    }

    bool validate_event_cd(const Evt4Raw::EVT4EventCD *ev) {
        return static_cast<SelfType *>(this)->validate_event_cd_impl(ev);
    }

    bool validate_event_cd_vec(const Evt4Raw::EVT4EventCD *ev, const std::uint32_t *mask) {
        return static_cast<SelfType *>(this)->validate_event_cd_vec_impl(ev, mask);
    }

    bool validate_time_high(timestamp base_time, timestamp time_high) {
        return static_cast<SelfType *>(this)->validate_time_high_impl(base_time, time_high);
    }

    bool validate_ext_trigger(const Evt4Raw::RawEvent *raw_events) {
        return static_cast<SelfType *>(this)->validate_ext_trigger_impl(raw_events);
    }

    bool validate_event_other(const Evt4Raw::RawEvent *raw_events) {
        return static_cast<SelfType *>(this)->validate_event_other_impl(raw_events);
    }

    bool validate_event_continued(const Evt4Raw::RawEvent *raw_events) {
        return static_cast<SelfType *>(this)->validate_event_continued_impl(raw_events);
    }
};

class NullCheckValidator : public ValidatorInterface<NullCheckValidator> {
public:
    NullCheckValidator(std::uint32_t width, std::uint32_t height) :
        ValidatorInterface<NullCheckValidator>(width, height) {}

    void reset_impl() {}

    bool validate_event_cd_impl([[maybe_unused]] const Evt4Raw::EVT4EventCD *ev) {
        return true;
    }

    bool validate_event_cd_vec_impl([[maybe_unused]] const Evt4Raw::EVT4EventCD *ev,
                                    [[maybe_unused]] const std::uint32_t *mask) {
        return true;
    }

    bool validate_time_high_impl([[maybe_unused]] timestamp prev_time_high, [[maybe_unused]] timestamp time_high) {
        return true;
    }

    bool validate_ext_trigger_impl([[maybe_unused]] const Evt4Raw::RawEvent *raw_events) {
        return true;
    }

    bool validate_event_other_impl([[maybe_unused]] const Evt4Raw::RawEvent *raw_events) {
        return true;
    }

    bool validate_event_continued_impl([[maybe_unused]] const Evt4Raw::RawEvent *raw_events) {
        return true;
    }
};

class NotifyValidator : public ValidatorInterface<NotifyValidator> {
protected:
    static constexpr timestamp ThJumpThreshold{1'000};

public:
    NotifyValidator(std::uint32_t width, std::uint32_t height) : ValidatorInterface<NotifyValidator>(width, height) {}

    void reset_impl() {}

    bool validate_event_cd_impl(const Evt4Raw::EVT4EventCD *ev) {
        if (ev->y >= height_ || ev->x >= width_) {
            notify(DecoderProtocolViolation::OutOfBoundsEventCoordinate);
        }
        return true;
    }

    bool validate_event_cd_vec_impl(const Evt4Raw::EVT4EventCD *ev, [[maybe_unused]] const std::uint32_t *mask) {
        if (ev->y >= height_ || static_cast<std::uint32_t>(ev->x + 31) >= width_) {
            notify(DecoderProtocolViolation::OutOfBoundsEventCoordinate);
        }
        return true;
    }

    bool validate_time_high_impl(timestamp base_time, timestamp time_high) {
        if (time_high < base_time) {
            notify(DecoderProtocolViolation::NonMonotonicTimeHigh);
        } else if (time_high >= base_time + ThJumpThreshold) {
            notify(DecoderProtocolViolation::NonContinuousTimeHigh);
        }
        return true;
    }

    bool validate_ext_trigger_impl([[maybe_unused]] const Evt4Raw::RawEvent *raw_events) {
        return true;
    }

    bool validate_event_other_impl([[maybe_unused]] const Evt4Raw::RawEvent *raw_events) {
        return true;
    }

    bool validate_event_continued_impl([[maybe_unused]] const Evt4Raw::RawEvent *raw_events) {
        return true;
    }
};

class RobustValidator : public ValidatorInterface<RobustValidator> {
protected:
    bool ev_drop_{false};
    bool pending_th_jump_{false};
    timestamp pending_base_time{0};
    static constexpr timestamp ThIncrement{64};
    static constexpr timestamp ThJumpThreshold{1'000};

public:
    RobustValidator(std::uint32_t width, std::uint32_t height) : ValidatorInterface<RobustValidator>(width, height) {}

    void reset_impl() {
        ev_drop_          = false;
        pending_th_jump_  = false;
        pending_base_time = 0;
    }

    bool validate_event_cd_impl(const Evt4Raw::EVT4EventCD *ev) {
        if (ev_drop_) {
            return false;
        }
        if (ev->y >= height_ || ev->x >= width_) {
            notify(DecoderProtocolViolation::OutOfBoundsEventCoordinate);
            return false;
        }
        return true;
    }

    bool validate_event_cd_vec_impl(const Evt4Raw::EVT4EventCD *ev, [[maybe_unused]] const std::uint32_t *mask) {
        if (ev_drop_) {
            return false;
        } else if (ev->y >= height_ || static_cast<std::uint32_t>(ev->x + 31) >= width_) {
            notify(DecoderProtocolViolation::OutOfBoundsEventCoordinate);
            return false;
        }
        return true;
    }

    bool validate_time_high_impl(timestamp base_time, timestamp time_high) {
        // If there's a small return back (rollback) we drop events until we reach a state where we're above the
        // last timestamp. When a timehigh t arrives, if there's a jump forward of more than ThJumpThreshold we
        // wait to confirm the jump. We wait for a time high that is either t or t + 64 to confirm the jump, or
        // if we receive a time high that is equal to base_time or base_time + 64 we reject the jump.
        if (pending_th_jump_) {
            if ((time_high >= pending_base_time && time_high <= pending_base_time + ThIncrement) ||
                (time_high >= base_time && time_high <= base_time + ThIncrement)) {
                ev_drop_         = false;
                pending_th_jump_ = false;
                return true;
            } else if (time_high >= base_time) {
                pending_base_time = time_high;
            }
            return false;
        } else {
            if (time_high < base_time) {
                ev_drop_ = true;
                notify(DecoderProtocolViolation::NonMonotonicTimeHigh);
                return false;
            } else if (time_high >= base_time + ThJumpThreshold) {
                ev_drop_          = true;
                pending_base_time = time_high;
                pending_th_jump_  = true;
                notify(DecoderProtocolViolation::NonContinuousTimeHigh);
                return false;
            } else {
                ev_drop_ = false;
                return true;
            }
        }
    }

    bool validate_ext_trigger_impl([[maybe_unused]] const Evt4Raw::RawEvent *raw_events) {
        return !ev_drop_;
    }

    bool validate_event_other_impl([[maybe_unused]] const Evt4Raw::RawEvent *raw_events) {
        return !ev_drop_;
    }

    bool validate_event_continued_impl([[maybe_unused]] const Evt4Raw::RawEvent *raw_events) {
        return !ev_drop_;
    }
};

} // namespace evt4
} // namespace decoder
} // namespace Metavision

#endif // METAVISION_HAL_EVT4_VALIDATOR_H
