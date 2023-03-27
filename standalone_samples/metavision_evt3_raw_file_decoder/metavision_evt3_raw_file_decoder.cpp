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

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

namespace Metavision {
namespace Evt3 {

// Event Data 3.0 is a 16 bits vectorized data format. It has been designed to comply with both data compactness
// necessity and vector event data support.
// This EVT3.0 event format avoids transmitting redundant event data for the time, y and x values.
enum class EventTypes : uint8_t {
    EVT_ADDR_Y = 0x0,  // Identifies a CD event and its y coordinate
    EVT_ADDR_X = 0x2,  // Marks a valid single event and identifies its polarity and X coordinate. The event's type and
                       // timestamp are considered to be the last ones sent
    VECT_BASE_X = 0x3, // Transmits the base address for a subsequent vector event and identifies its polarity and base
                       // X coordinate. This event does not represent a CD sensor event in itself and should not be
                       // processed as such,  it only sets the base x value for following VECT_12 and VECT_8 events.
    VECT_12 = 0x4, // Vector event with 12 valid bits. This event encodes the validity bits for events of the same type,
                   // timestamp and y coordinate as previously sent events, while consecutive in x coordinate with
                   // respect to the last sent VECT_BASE_X event. After processing this event, the X position value
                   // on the receiver side should be incremented by 12 with respect to the X position when the event was
                   // received, so that the VECT_BASE_X is updated like follows: VECT_BASE_X.x = VECT_BASE_X.x + 12
    VECT_8 = 0x5,  // Vector event with 8 valid bits. This event encodes the validity bits for events of the same type,
                   // timestamp and y coordinate as previously sent events, while consecutive in x coordinate with
                   // respect to the last sent VECT_BASE_X event. After processing this event, the X position value
                   // on the receiver side should be incremented by 8 with respect to the X position when the event was
                   // received, so that the VECT_BASE_X is updated like follows: VECT_BASE_X.x = VECT_BASE_X.x + 8
    EVT_TIME_LOW = 0x6, // Encodes the lower 12b of the timebase range (range 11 to 0). Note that the TIME_LOW value is
                        // only monotonic for a same event source, but can be non-monotonic when multiple event sources
                        // are considered. They should however refer to the same TIME_HIGH value. As the time low has
                        // 12b with a 1us resolution, it can encode time values from 0us to 4095us (4095 = 2^12 - 1).
                        // After 4095us, the time_low value wraps and returns to 0us, at which point the TIME_HIGH value
                        // should be incremented.
    EVT_TIME_HIGH = 0x8, // Encodes the higher portion of the timebase (range 23 to 12). Since it encodes the 12 higher
                         // bits over the 24 used to encode a timestamp, it has a resolution of 4096us (= 2^(24-12)) and
                         // it can encode time values from 0us to 16777215us (= 16.777215s). After 16773120us the
                         // time_high value wraps and returns to 0us.
    EXT_TRIGGER = 0xA    // External trigger output
};

// Evt3 raw events are 16-bit words
struct RawEvent {
    uint16_t pad : 12; // Padding
    uint16_t type : 4; // Event type
};

struct RawEventTime {
    uint16_t time : 12;
    uint16_t type : 4; // Event type : EventTypes::EVT_TIME_LOW OR EventTypes::EVT_TIME_HIGH
};

struct RawEventXAddr {
    uint16_t x : 11;   // Pixel X coordinate
    uint16_t pol : 1;  // Event polarity:
                       // '0': decrease in illumination
                       // '1': increase in illumination
    uint16_t type : 4; // Event type : EventTypes::EVT_ADDR_X
};

struct RawEventVect12 {
    uint16_t valid : 12; // Encodes the validity of the events in the vector :
                         // foreach i in 0 to 11
                         //   if valid[i] is '1'
                         //      valid event at X = VECT_BASE_X.x + i
    uint16_t type : 4;   // Event type : EventTypes::VECT_12
};

struct RawEventVect8 {
    uint16_t valid : 8; // Encodes the validity of the events in the vector :
                        // foreach i in  0 to 7
                        //   if valid[i] is '1'
                        //      valid event at X = VECT_BASE_X.x + i
    uint16_t unused : 4;
    uint16_t type : 4; // Event type : EventTypes::VECT_8
};

struct RawEventY {
    uint16_t y : 11;   // Pixel Y coordinate
    uint16_t orig : 1; // Identifies the System Type:
                       // '0': Master Camera (Left Camera in Stereo Systems)
                       // '1': Slave Camera (Right Camera in Stereo Systems)
    uint16_t type : 4; // Event type : EventTypes::EVT_ADDR_Y
};

struct RawEventXBase {
    uint16_t x : 11;   // Pixel X coordinate
    uint16_t pol : 1;  // Event polarity:
                       // '0': decrease in illumination
                       // '1': increase in illumination
    uint16_t type : 4; // Event type : EventTypes::VECT_BASE_X
};

struct RawEventExtTrigger {
    uint16_t value : 1; // Trigger current value (edge polarity):
                        // - '0' (falling edge);
                        // - '1' (rising edge).
    uint16_t unused : 7;
    uint16_t id : 4;   // Trigger channel ID.
    uint16_t type : 4; // Event type : EventTypes::EXT_TRIGGER
};

using timestamp_t = uint64_t; // Type for timestamp, in microseconds

} // namespace Evt3
} // namespace Metavision

int main(int argc, char *argv[]) {
    // Check input arguments validity
    if (argc < 3) {
        std::cerr << "Error : need input filename and output filename for CD events" << std::endl;
        std::cerr << std::endl
                  << "Usage : " << std::string(argv[0]) << " INPUT_FILENAME CD_OUTPUTFILE (TRIGGER_OUTPUTFILE)"
                  << std::endl;
        std::cerr << "Trigger output file will be written only if given as input" << std::endl;
        std::cerr << std::endl << "Example : " << std::string(argv[0]) << " input_file.raw cd_output.csv" << std::endl;
        std::cerr << std::endl;
        std::cerr << "The CD CSV file will have the format : x,y,polarity,timestamp" << std::endl;
        std::cerr << "The Trigger output CSV file will have the format : value,id,timestamp" << std::endl;
        return 1;
    }

    // Open input file
    std::ifstream input_file(argv[1], std::ios::in | std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Error : could not open file '" << argv[1] << "' for reading" << std::endl;
        return 1;
    }

    // Open CD csv output file
    std::ofstream cd_output_file(argv[2]);
    if (!cd_output_file.is_open()) {
        std::cerr << "Error : could not open file '" << argv[2] << "' for writing" << std::endl;
        return 1;
    }

    // Open External Trigger csv output file, if provided
    std::ofstream trigger_output_file;
    bool write_triggers = false;
    if (argc > 3) {
        trigger_output_file.open(argv[3]);
        if (!trigger_output_file.is_open()) {
            std::cerr << "Error : could not open file '" << argv[3] << "' for writing" << std::endl;
            return 1;
        }
        write_triggers = true;
    }

    // Skip the header of the input file, if present :
    int line_first_char = input_file.peek();
    while (line_first_char == '%') {
        std::string line;
        std::getline(input_file, line);
        if (line == "% end") {
            break;
        }
        line_first_char = input_file.peek();
    };

    // Vector where we'll read the raw data
    static constexpr uint32_t WORDS_TO_READ = 1000000; // Number of words to read at a time
    std::vector<Metavision::Evt3::RawEvent> buffer_read(WORDS_TO_READ);

    // State variables needed for decoding
    bool first_time_base_set                        = false;
    Metavision::Evt3::timestamp_t current_time_base = 0; // time high bits
    Metavision::Evt3::timestamp_t current_time_low  = 0;
    Metavision::Evt3::timestamp_t current_time      = 0;
    uint16_t current_ev_addr_y                      = 0;
    uint16_t current_base_x                         = 0;
    uint16_t current_polarity                       = 0;
    unsigned int n_time_high_loop                   = 0; // Counter of the time high loops

    std::string cd_str = "", trigg_str = "";

    while (input_file) {
        input_file.read(reinterpret_cast<char *>(buffer_read.data()),
                        WORDS_TO_READ * sizeof(Metavision::Evt3::RawEvent));
        Metavision::Evt3::RawEvent *current_word = buffer_read.data();
        Metavision::Evt3::RawEvent *last_word = current_word + input_file.gcount() / sizeof(Metavision::Evt3::RawEvent);

        // If the first event in the input file is not of type EVT_TIME_HIGH, then the times
        // of the first events might be wrong, because we don't have a time base yet. This is why
        // we skip the events until we find the first time high, so that we can correctly set
        // the current_time_base
        for (; !first_time_base_set && current_word != last_word; ++current_word) {
            Metavision::Evt3::EventTypes type = static_cast<Metavision::Evt3::EventTypes>(current_word->type);
            if (type == Metavision::Evt3::EventTypes::EVT_TIME_HIGH) {
                Metavision::Evt3::RawEventTime *ev_timehigh =
                    reinterpret_cast<Metavision::Evt3::RawEventTime *>(current_word);
                current_time_base   = (Metavision::Evt3::timestamp_t(ev_timehigh->time) << 12);
                first_time_base_set = true;
                break;
            }
        }
        for (; current_word != last_word; ++current_word) {
            Metavision::Evt3::EventTypes type = static_cast<Metavision::Evt3::EventTypes>(current_word->type);
            switch (type) {
            case Metavision::Evt3::EventTypes::EVT_ADDR_X: {
                Metavision::Evt3::RawEventXAddr *ev_addr_x =
                    reinterpret_cast<Metavision::Evt3::RawEventXAddr *>(current_word);
                // We have a new Event CD with
                // x = ev_addr_x->x
                // y = current_ev_addr_y
                // polarity = ev_addr_x->pol
                // time = current_time (in us)
                cd_str += std::to_string(ev_addr_x->x) + "," + std::to_string(current_ev_addr_y) + "," +
                          std::to_string(ev_addr_x->pol) + "," + std::to_string(current_time) + "\n";
                break;
            }
            case Metavision::Evt3::EventTypes::VECT_12: {
                uint16_t end = current_base_x + 12;

                Metavision::Evt3::RawEventVect12 *ev_vec_12 =
                    reinterpret_cast<Metavision::Evt3::RawEventVect12 *>(current_word);
                uint32_t valid = ev_vec_12->valid;
                for (uint16_t i = current_base_x; i != end; ++i) {
                    if (valid & 0x1) {
                        // We have a new Event CD with
                        // x = i
                        // y = current_ev_addr_y
                        // polarity = current_polarity
                        // time = current_time (in us)
                        cd_str += std::to_string(i) + "," + std::to_string(current_ev_addr_y) + "," +
                                  std::to_string(current_polarity) + "," + std::to_string(current_time) + "\n";
                    }
                    valid >>= 1;
                }
                current_base_x = end;
                break;
            }
            case Metavision::Evt3::EventTypes::VECT_8: {
                uint16_t end = current_base_x + 8;

                Metavision::Evt3::RawEventVect8 *ev_vec_8 =
                    reinterpret_cast<Metavision::Evt3::RawEventVect8 *>(current_word);
                uint32_t valid = ev_vec_8->valid;
                for (uint16_t i = current_base_x; i != end; ++i) {
                    if (valid & 0x1) {
                        // We have a new Event CD with
                        // x = i
                        // y = current_ev_addr_y
                        // polarity = current_polarity
                        // time = current_time (in us)
                        cd_str += std::to_string(i) + "," + std::to_string(current_ev_addr_y) + "," +
                                  std::to_string(current_polarity) + "," + std::to_string(current_time) + "\n";
                    }
                    valid >>= 1;
                }
                current_base_x = end;
                break;
            }
            case Metavision::Evt3::EventTypes::EVT_ADDR_Y: {
                Metavision::Evt3::RawEventY *ev_addr_y = reinterpret_cast<Metavision::Evt3::RawEventY *>(current_word);
                current_ev_addr_y                      = ev_addr_y->y;
                break;
            }
            case Metavision::Evt3::EventTypes::VECT_BASE_X: {
                Metavision::Evt3::RawEventXBase *ev_xbase =
                    reinterpret_cast<Metavision::Evt3::RawEventXBase *>(current_word);
                current_polarity = ev_xbase->pol;
                current_base_x   = ev_xbase->x;
                break;
            }
            case Metavision::Evt3::EventTypes::EVT_TIME_HIGH: {
                // Compute some useful constant variables :
                //
                // -> MaxTimestampBase is the maximum value that the variable current_time_base can have. It corresponds
                // to the case where an event Metavision::Evt3::RawEventTime of type EVT_TIME_HIGH has all the bits of
                // the field "timestamp" (12 bits total) set to 1 (value is (1 << 12) - 1). We then need to shift it by
                // 12 bits because this field represents the most significant bits of the event time base (range 23 to
                // 12). See the event description at the beginning of the file.
                //
                // -> TimeLoop is the loop duration (in us) before the time_high value wraps and returns to 0. Its value
                // is MaxTimestampBase + (1 << 12)
                //
                // -> LoopThreshold is a threshold value used to detect if a new value of the time high has decreased
                // because it looped. Theoretically, if the new value of the time high is lower than the last one, then
                // it means that is has looped. In practice, to protect ourselves from a transmission error, we use a
                // threshold value, so that we consider that the time high has looped only if it differs from the last
                // value by a sufficient difference (i.e. greater than the threshold)
                static constexpr Metavision::Evt3::timestamp_t MaxTimestampBase =
                    ((Metavision::Evt3::timestamp_t(1) << 12) - 1) << 12;                               // = 16773120us
                static constexpr Metavision::Evt3::timestamp_t TimeLoop = MaxTimestampBase + (1 << 12); // = 16777216us
                static constexpr Metavision::Evt3::timestamp_t LoopThreshold =
                    (10 << 12); // It could be another value too, as long as it is a big enough value that we can be
                                // sure that the time high looped

                Metavision::Evt3::RawEventTime *ev_timehigh =
                    reinterpret_cast<Metavision::Evt3::RawEventTime *>(current_word);
                Metavision::Evt3::timestamp_t new_time_base = (Metavision::Evt3::timestamp_t(ev_timehigh->time) << 12);
                new_time_base += n_time_high_loop * TimeLoop;

                if ((current_time_base > new_time_base) &&
                    (current_time_base - new_time_base >= MaxTimestampBase - LoopThreshold)) {
                    // Time High loop :  we consider that we went in the past because the timestamp looped
                    new_time_base += TimeLoop;
                    ++n_time_high_loop;
                }

                current_time_base = new_time_base;
                current_time      = current_time_base;
                break;
            }
            case Metavision::Evt3::EventTypes::EVT_TIME_LOW: {
                Metavision::Evt3::RawEventTime *ev_timelow =
                    reinterpret_cast<Metavision::Evt3::RawEventTime *>(current_word);
                current_time_low = ev_timelow->time;
                current_time     = current_time_base + current_time_low;
                break;
            }
            case Metavision::Evt3::EventTypes::EXT_TRIGGER: {
                if (write_triggers) {
                    Metavision::Evt3::RawEventExtTrigger *ev_trigg =
                        reinterpret_cast<Metavision::Evt3::RawEventExtTrigger *>(current_word);

                    // We have a new Event Trigger with
                    // value = ev_trigg->value
                    // id = ev_trigg->id
                    // time = current_time (in us)
                    trigg_str += std::to_string(ev_trigg->value) + "," + std::to_string(ev_trigg->id) + "," +
                                 std::to_string(current_time) + "\n";
                }
                break;
            }
            default:
                break;
            }
        }

        // Write in the files
        if (!cd_str.empty()) {
            cd_output_file << cd_str;
            cd_str.clear();
        }
        if (write_triggers && !trigg_str.empty()) {
            trigger_output_file << trigg_str;
            trigg_str.clear();
        }
    }
    return 0;
}
