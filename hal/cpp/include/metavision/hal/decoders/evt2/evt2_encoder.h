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

#ifndef METAVISION_HAL_EVT2_ENCODER_H
#define METAVISION_HAL_EVT2_ENCODER_H

#include <cstdint>
#include <limits>
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/hal/decoders/evt2/evt2_event_types.h"

namespace Metavision {

/// @brief Class to encode events in EVT2 format
class Evt2Encoder {
public:
    /// @brief Constructor
    Evt2Encoder() = default;

    /// @brief resets the internal timehigh state of the encoder
    void reset_state();

    /// @brief Encodes a CD event and writes it to the output stream
    /// @param ofs Output stream to write the encoded event to
    /// @param ev CD event to encode
    void encode_event_cd(std::ofstream &ofs, const EventCD &ev);

    /// @brief Encodes a trigger event and writes it to the output stream
    /// @param ofs Output stream to write the encoded event to
    /// @param ev Trigger event to encode
    void encode_event_trigger(std::ofstream &ofs, const EventExtTrigger &ev);

private:
    void write_raw_event(std::ofstream &ofs, const EVT2RawEvent &raw_evt) const;
    void write_timehigh(std::ofstream &ofs, timestamp ts_timehigh_ev);
    void write_cd(std::ofstream &ofs, const EventCD &ev);
    void write_trigger(std::ofstream &ofs, const EventExtTrigger &ev);
    void update_timehigh(std::ofstream &ofs, timestamp ts);

    static constexpr timestamp kTime16usMask{(static_cast<timestamp>(1) << 4) - 1};
    bool first_timehigh_written_ = false;
    timestamp ts_last_timehigh_  = std::numeric_limits<timestamp>::min();
    timestamp ts_last_ev_        = std::numeric_limits<timestamp>::min();
};

} // namespace Metavision

#endif // METAVISION_HAL_EVT2_ENCODER_H
