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
#include "metavision/hal/decoders/evt2/evt2_encoder.h"

namespace Metavision {

void Evt2Encoder::reset_state() {
    first_timehigh_written_ = false;
    ts_last_timehigh_       = std::numeric_limits<timestamp>::min();
    ts_last_ev_             = std::numeric_limits<timestamp>::min();
}

void Evt2Encoder::encode_event_cd(std::ofstream &ofs, const EventCD &ev) {
    if (ev.t < ts_last_ev_) {
        throw std::runtime_error("Input events must be encoded in increasing temporal order!");
    }
    update_timehigh(ofs, ev.t);
    write_cd(ofs, ev);
}

void Evt2Encoder::encode_event_trigger(std::ofstream &ofs, const EventExtTrigger &ev) {
    if (ev.t < ts_last_ev_) {
        throw std::runtime_error("Input events must be encoded in increasing temporal order!");
    }
    update_timehigh(ofs, ev.t);
    write_trigger(ofs, ev);
}

void Evt2Encoder::write_raw_event(std::ofstream &ofs, const EVT2RawEvent &raw_evt) const {
    ofs.write(reinterpret_cast<const char *>(&raw_evt.raw), sizeof(raw_evt.raw));
}

void Evt2Encoder::write_timehigh(std::ofstream &ofs, timestamp ts_timehigh_ev) {
    EVT2RawEvent raw_evt{0};
    raw_evt.th.type = static_cast<std::uint8_t>(EVT2EventTypes::EVT_TIME_HIGH);
    raw_evt.th.ts   = ts_timehigh_ev >> 6;
    write_raw_event(ofs, raw_evt);
    ts_last_ev_ = ts_timehigh_ev;
}

void Evt2Encoder::write_cd(std::ofstream &ofs, const EventCD &ev) {
    EVT2RawEvent raw_evt{0};
    raw_evt.cd.type      = static_cast<std::uint8_t>(ev.p == 1 ? EVT2EventTypes::CD_ON : EVT2EventTypes::CD_OFF);
    raw_evt.cd.timestamp = ev.t;
    raw_evt.cd.x         = ev.x;
    raw_evt.cd.y         = ev.y;
    write_raw_event(ofs, raw_evt);
    ts_last_ev_ = ev.t;
}

void Evt2Encoder::write_trigger(std::ofstream &ofs, const EventExtTrigger &ev) {
    EVT2RawEvent raw_evt{0};
    raw_evt.trig.type      = static_cast<std::uint8_t>(EVT2EventTypes::EXT_TRIGGER);
    raw_evt.trig.timestamp = ev.t;
    raw_evt.trig.id        = ev.id;
    raw_evt.trig.value     = ev.p;
    write_raw_event(ofs, raw_evt);
    ts_last_ev_ = ev.t;
}

void Evt2Encoder::update_timehigh(std::ofstream &ofs, timestamp ts) {
    if (!first_timehigh_written_) {
        first_timehigh_written_ = true;
        ts_last_timehigh_       = ts & (~kTime16usMask);
        Evt2Encoder::write_timehigh(ofs, ts_last_timehigh_);
    } else {
        while ((ts_last_timehigh_ >> 4) < (ts >> 4)) {
            ts_last_timehigh_ = (ts_last_timehigh_ & (~kTime16usMask)) + 16;
            Evt2Encoder::write_timehigh(ofs, ts_last_timehigh_);
        }
    }
}

} // namespace Metavision
