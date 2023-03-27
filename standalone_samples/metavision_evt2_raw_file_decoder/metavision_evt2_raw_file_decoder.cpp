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
namespace Evt2 {

enum class EventTypes : uint8_t {
    CD_LOW        = 0x00, // CD event, decrease in illumination (polarity '0')
    CD_HIGH       = 0x01, // CD event, increase in illumination (polarity '1')
    EVT_TIME_HIGH = 0x08, // Encodes the higher portion of the timebase (range 33 to 6). Since it encodes the 28 higher
                          // bits over the 34 used to encode a timestamp, it has a resolution of 64us (= 2^(34-28)) and
                          // it can encode time values from 0us to 17179869183us (~ 4h46m20s). After
                          // 17179869120us the time_high value wraps and returns to 0us.
    EXT_TRIGGER = 0x0A,   // External trigger output
};

// Evt2 raw events are 32-bit words
struct RawEvent {
    unsigned int pad : 28; // Padding
    unsigned int type : 4; // Event type
};

struct RawEventTime {
    unsigned int timestamp : 28; // Most significant bits of the event time base (33..6)
    unsigned int type : 4;       // Event type : EventTypes::EVT_TIME_HIGH
};

struct RawEventCD {
    unsigned int y : 11;        // Pixel Y coordinate
    unsigned int x : 11;        // Pixel X coordinate
    unsigned int timestamp : 6; // Least significant bits of the event time ba
    unsigned int type : 4;      // Event type : EventTypes::CD_LOW or EventTypes::CD_HIGH
};
struct RawEventExtTrigger {
    unsigned int value : 1; // Trigger current value (edge polarity):
                            // - '0' (falling edge);
                            // - '1' (rising edge).
    unsigned int unused2 : 7;
    unsigned int id : 5; // Trigger channel ID.
    unsigned int unused1 : 9;
    unsigned int timestamp : 6;
    unsigned int type : 4; // Event type : EventTypes::EXT_TRIGGER
};

using timestamp_t = uint64_t; // Type for timestamp, in microseconds

} // namespace Evt2
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
    std::vector<Metavision::Evt2::RawEvent> buffer_read(WORDS_TO_READ);

    // State variables needed for decoding
    Metavision::Evt2::timestamp_t current_time_base = 0; // time high bits
    bool first_time_base_set                        = false;
    unsigned int n_time_high_loop                   = 0; // Counter of the time high loops

    std::string cd_str = "", trigg_str = "";

    while (input_file) {
        input_file.read(reinterpret_cast<char *>(buffer_read.data()),
                        WORDS_TO_READ * sizeof(Metavision::Evt2::RawEvent));
        Metavision::Evt2::RawEvent *current_word = buffer_read.data();
        Metavision::Evt2::RawEvent *last_word = current_word + input_file.gcount() / sizeof(Metavision::Evt2::RawEvent);

        // If the first event in the input file is not of type EVT_TIME_HIGH, then the times
        // of the first events might be wrong, because we don't have a time base yet. This is why
        // we skip the events until we find the first time high, so that we can correctly set
        // the current_time_base
        for (; !first_time_base_set && current_word != last_word; ++current_word) {
            Metavision::Evt2::EventTypes type = static_cast<Metavision::Evt2::EventTypes>(current_word->type);
            if (type == Metavision::Evt2::EventTypes::EVT_TIME_HIGH) {
                Metavision::Evt2::RawEventTime *ev_time_high =
                    reinterpret_cast<Metavision::Evt2::RawEventTime *>(current_word);
                current_time_base   = (Metavision::Evt2::timestamp_t(ev_time_high->timestamp) << 6);
                first_time_base_set = true;
                break;
            }
        }
        for (; current_word != last_word; ++current_word) {
            Metavision::Evt2::EventTypes type = static_cast<Metavision::Evt2::EventTypes>(current_word->type);
            switch (type) {
            case Metavision::Evt2::EventTypes::CD_LOW: {
                // CD events, decrease in illumination (polarity '0')
                Metavision::Evt2::RawEventCD *ev_cd = reinterpret_cast<Metavision::Evt2::RawEventCD *>(current_word);
                Metavision::Evt2::timestamp_t t     = current_time_base + ev_cd->timestamp;

                // We have a new Event CD with
                // x = ev_cd->x
                // y = ev_cd->y
                // polarity = 0
                // time = t (in us)
                cd_str += std::to_string(ev_cd->x) + "," + std::to_string(ev_cd->y) + ",0," + std::to_string(t) + "\n";
                break;
            }
            case Metavision::Evt2::EventTypes::CD_HIGH: {
                // CD events, increase in illumination (polarity '1')
                Metavision::Evt2::RawEventCD *ev_cd = reinterpret_cast<Metavision::Evt2::RawEventCD *>(current_word);
                Metavision::Evt2::timestamp_t t     = current_time_base + ev_cd->timestamp;

                // We have a new Event CD with
                // x = ev_cd->x
                // y = ev_cd->y
                // polarity = 1
                // time = t (in us)
                cd_str += std::to_string(ev_cd->x) + "," + std::to_string(ev_cd->y) + ",1," + std::to_string(t) + "\n";
                break;
            }
            case Metavision::Evt2::EventTypes::EVT_TIME_HIGH: {
                // Time high

                // Compute some useful constant variables :
                //
                // -> MaxTimestampBase is the maximum value that the variable current_time_base can have. It
                // corresponds to the case where an event of type Metavision::Evt2::RawEventTime has all the bits of the
                // field "timestamp" (28 bits total) set to 1 (value is (1 << 28) - 1). We then need to shift it by 6
                // bits because this field represents the most significant bits of the event time base (range 33 to 6).
                // See the event description at the beginning of the file.
                //
                // -> TimeLoop is the loop duration (in us) before the time_high value wraps and returns to 0. Its value
                // is MaxTimestampBase + (1 << 6)
                //
                // -> LoopThreshold is a threshold value used to detect if a new value of the time high has decreased
                // because it looped. Theoretically, if the new value of the time high is lower than the last one, then
                // it means that is has looped. In practice, to protect ourselves from a transmission error, we use a
                // threshold value, so that we consider that the time high has looped only if it differs from the last
                // value by a sufficient difference (i.e. greater than the threshold)
                static constexpr Metavision::Evt2::timestamp_t MaxTimestampBase =
                    ((Metavision::Evt2::timestamp_t(1) << 28) - 1) << 6; // = 17179869120us
                static constexpr Metavision::Evt2::timestamp_t TimeLoop =
                    MaxTimestampBase + (1 << 6); // = 17179869184us
                static constexpr Metavision::Evt2::timestamp_t LoopThreshold =
                    (10 << 6); // It could be another value too, as long as it is a big enough value that we can be sure
                               // that the time high looped

                Metavision::Evt2::RawEventTime *ev_time_high =
                    reinterpret_cast<Metavision::Evt2::RawEventTime *>(current_word);
                Metavision::Evt2::timestamp_t new_time_base =
                    (Metavision::Evt2::timestamp_t(ev_time_high->timestamp) << 6);
                new_time_base += n_time_high_loop * TimeLoop;

                if ((current_time_base > new_time_base) &&
                    (current_time_base - new_time_base >= MaxTimestampBase - LoopThreshold)) {
                    // Time High loop :  we consider that we went in the past because the timestamp looped
                    new_time_base += TimeLoop;
                    ++n_time_high_loop;
                }

                current_time_base = new_time_base;
                break;
            }
            case Metavision::Evt2::EventTypes::EXT_TRIGGER: {
                // External trigger output
                if (write_triggers) {
                    Metavision::Evt2::RawEventExtTrigger *ev_trigg =
                        reinterpret_cast<Metavision::Evt2::RawEventExtTrigger *>(current_word);
                    Metavision::Evt2::timestamp_t t = current_time_base + ev_trigg->timestamp;

                    // We have a new Event Trigger with
                    // value = ev_trigg->value
                    // id = ev_trigg->id
                    // time = t (in us)
                    trigg_str += std::to_string(ev_trigg->value) + "," + std::to_string(ev_trigg->id) + "," +
                                 std::to_string(t) + "\n";
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
