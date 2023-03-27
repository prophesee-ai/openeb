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
#include <sstream>
#include <cstdint>

namespace Metavision {
namespace Evt2 {

enum class EventTypes : uint8_t {
    CD_LOW        = 0x00, // CD event, decrease in illumination (polarity '0')
    CD_HIGH       = 0x01, // CD event, increase in illumination (polarity '1')
    EVT_TIME_HIGH = 0x08, // Encodes the higher portion of the timebase (bits 33..6). Since it encodes the 28 higher
                          // bits over the 34 used to encode a timestamp, it has a resolution of 64us (= 2^(34-28)) and
                          // it can encode time values from 0us to 17179869183us (~ 4h46m20s). After
                          // 17179869120us its value wraps and returns to 0us.
    EXT_TRIGGER = 0x0A,   // External trigger output
};

// Evt2 raw events are 32-bit words
struct RawEvent {
    unsigned int pad : 28; // Padding
    unsigned int type : 4; // Event type
};

struct RawEventTime {
    unsigned int timestamp : 28; // Most significant bits of the event timestamp (bits 33..6)
    unsigned int type : 4;       // Event type: EventTypes::EVT_TIME_HIGH
};

struct RawEventCD {
    unsigned int y : 11;        // Pixel Y coordinate
    unsigned int x : 11;        // Pixel X coordinate
    unsigned int timestamp : 6; // Least significant bits of the event timestamp (bits 5..0)
    unsigned int type : 4;      // Event type: EventTypes::CD_LOW or EventTypes::CD_HIGH
};

struct RawEventExtTrigger {
    unsigned int value : 1; // Trigger current value (edge polarity):
                            // - '0' (falling edge);
                            // - '1' (rising edge).
    unsigned int unused2 : 7;
    unsigned int id : 5; // Trigger channel ID.
    unsigned int unused1 : 9;
    unsigned int timestamp : 6; // Least significant bits of the event timestamp (bits 5..0)
    unsigned int type : 4;      // Event type: EventTypes::EXT_TRIGGER
};

using Timestamp = uint64_t; // Type for timestamp, in microseconds

/// @brief Class that reads CD events from a CSV file and encodes them in EVT2 format
struct EventCDEncoder {
    /// @brief Column position in the sensor at which the event happened
    unsigned short x;

    /// @brief Row position in the sensor at which the event happened
    unsigned short y;

    /// @brief Polarity
    ///
    /// The polarity represents the change of contrast
    ///     - 1: a positive contrast change
    ///     - 0: a negative contrast change
    short p;

    /// @brief Timestamp at which the event happened (in us)
    Timestamp t;

    /// @brief Reads next line of CSV file
    /// @param ifs Stream to the input file to read
    bool read_next_line(std::ifstream &ifs) {
        std::string line;
        if (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (std::getline(iss, token, ',')) {
                tokens.push_back(token);
            }
            if (tokens.size() != 4) {
                std::cerr << "Invalid line for CD event: <" << line << ">" << std::endl;
            } else {
                x = static_cast<unsigned short>(std::stoul(tokens[0]));
                y = static_cast<unsigned short>(std::stoul(tokens[1]));
                p = static_cast<short>(std::stoi(tokens[2]));
                t = std::stoll(tokens[3]);
                return true;
            }
        }
        return false;
    }

    /// @brief Encodes CD event
    /// @param raw_event Pointer to the data to write
    void encode(RawEvent *raw_event) {
        RawEventCD *raw_cd_event = reinterpret_cast<RawEventCD *>(raw_event);
        raw_cd_event->x          = x;
        raw_cd_event->y          = y;
        raw_cd_event->timestamp  = t;
        raw_cd_event->type = p ? static_cast<uint8_t>(EventTypes::CD_HIGH) : static_cast<uint8_t>(EventTypes::CD_LOW);
    }
};

/// @brief Class that reads Trigger events from a CSV file and encodes them in EVT2 format
struct EventTriggerEncoder {
    /// Polarity representing the change of contrast (1: positive, 0: negative)
    short p;

    /// Timestamp at which the event happened (in us)
    Timestamp t;

    /// ID of the external trigger
    short id;

    /// @brief Reads next line of CSV file
    /// @param ifs Stream to the input file to read
    bool read_next_line(std::ifstream &ifs) {
        std::string line;
        if (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (std::getline(iss, token, ',')) {
                tokens.push_back(token);
            }
            if (tokens.size() != 3) {
                std::cerr << "Invalid line for Trigger event: <" << line << ">" << std::endl;
            } else {
                p  = static_cast<short>(std::stoi(tokens[0]));
                id = static_cast<short>(std::stoi(tokens[1]));
                t  = std::stoll(tokens[2]);
                return true;
            }
        }
        return false;
    }

    /// @brief Encodes Trigger event
    /// @param raw_event Pointer to the data to write
    void encode(RawEvent *raw_event) {
        RawEventExtTrigger *raw_trigger_event = reinterpret_cast<RawEventExtTrigger *>(raw_event);
        raw_trigger_event->timestamp          = t;
        raw_trigger_event->id                 = id;
        raw_trigger_event->value              = p;
        raw_trigger_event->type               = static_cast<uint8_t>(EventTypes::EXT_TRIGGER);
    }
};

/// @brief Class that encodes Time High events in EVT2 format
struct EventTimeEncoder {
    /// @brief Constructor
    /// @param base Time (in us) of the first event to encode
    EventTimeEncoder(Timestamp base) : th((base / TH_NEXT_STEP) * TH_NEXT_STEP) {}

    /// @brief Encodes Time High
    /// @param raw_event Pointer to the data to write
    void encode(RawEvent *raw_event) {
        auto ev_th       = reinterpret_cast<RawEventTime *>(raw_event);
        ev_th->timestamp = th >> N_LOWER_BITS_TH;
        ev_th->type      = static_cast<uint8_t>(EventTypes::EVT_TIME_HIGH);
        th += TH_NEXT_STEP;
    }

    /// Next Time High to encode
    Timestamp th;

private:
    static constexpr char N_LOWER_BITS_TH           = 6;
    static constexpr unsigned int REDUNDANCY_FACTOR = 4;
    static constexpr Timestamp TH_STEP              = (1ul << N_LOWER_BITS_TH);
    static constexpr Timestamp TH_NEXT_STEP         = TH_STEP / REDUNDANCY_FACTOR;
};

} // namespace Evt2
} // namespace Metavision

int main(int argc, char *argv[]) {
    // Check input arguments validity
    if (argc < 3) {
        std::cerr << "Error: need output filename and input filename for CD events" << std::endl;
        std::cerr << std::endl
                  << "Usage: " << std::string(argv[0]) << " OUTPUT_FILENAME CD_INPUTFILE (TRIGGER_INPUTFILE)"
                  << std::endl;
        std::cerr << "Triggers will be encoded only if given trigger file has been given as input" << std::endl;
        std::cerr << std::endl << "Example: " << std::string(argv[0]) << " output_file.raw cd_input.csv" << std::endl;
        std::cerr << std::endl;
        std::cerr << "The CD CSV file needs to have the format: x,y,polarity,timestamp" << std::endl;
        std::cerr << "The Trigger input CSV file needs to have the format: value,id,timestamp" << std::endl;
        return 1;
    }

    // Open input files
    std::ifstream input_cd_file(argv[2]);
    if (!input_cd_file.is_open()) {
        std::cerr << "Error: could not open file '" << argv[2] << "' for reading" << std::endl;
        return 1;
    }
    std::ifstream input_trigger_file;
    if (argc > 3) {
        input_trigger_file.open(argv[3]);
        if (!input_trigger_file.is_open()) {
            std::cerr << "Error: could not open file '" << argv[3] << "' for reading" << std::endl;
            return 1;
        }
    }

    // Open raw output file
    std::ofstream output_raw_file(argv[1], std::ios::binary);
    if (!output_raw_file.is_open()) {
        std::cerr << "Error: could not open file '" << argv[1] << "' for writing" << std::endl;
        return 1;
    }

    // Write header: we write the header corresponding to Prophesee EVK3 Gen41 device (largest geometry)
    output_raw_file << "% Date 2020-09-04 13:14:05" << std::endl;
    output_raw_file << "% evt 2.0" << std::endl;
    output_raw_file << "% firmware_version 3.3.0" << std::endl;
    output_raw_file << "% integrator_name Prophesee" << std::endl;
    output_raw_file << "% plugin_name hal_plugin_gen41_evk3" << std::endl;
    output_raw_file << "% system_ID 48" << std::endl;
    output_raw_file << "% end" << std::endl;

    // Initialize encoders
    Metavision::Evt2::EventCDEncoder CD_events_encoder;
    Metavision::Evt2::EventTriggerEncoder trigger_events_encoder;
    bool cd_done      = !CD_events_encoder.read_next_line(input_cd_file);
    bool trigger_done = input_trigger_file ? !trigger_events_encoder.read_next_line(input_trigger_file) : true;
    if (cd_done && trigger_done) {
        std::cerr << "Error: no events in input file(s)" << std::endl;
        return 1;
    }

    // Create a buffer where to store the encoded data before writing them in the output file
    static constexpr size_t kSizeBuffer = 1000;
    std::vector<Metavision::Evt2::RawEvent> raw_events(kSizeBuffer);
    Metavision::Evt2::RawEvent *raw_events_current_ptr = raw_events.data();
    Metavision::Evt2::RawEvent *raw_events_end_ptr     = raw_events_current_ptr + kSizeBuffer;

    // Determine the timestamp of the oldest event
    Metavision::Evt2::Timestamp first_ts = 0;
    if (!cd_done) {
        first_ts = CD_events_encoder.t;
    }
    if (!trigger_done && trigger_events_encoder.t < first_ts) {
        first_ts = trigger_events_encoder.t;
    }

    // Time High encoder
    Metavision::Evt2::EventTimeEncoder time_high_encoder(first_ts);
    // Encode First Time High
    time_high_encoder.encode(raw_events_current_ptr);
    ++raw_events_current_ptr;

    while (!(cd_done && trigger_done)) {
        if (raw_events_current_ptr == raw_events_end_ptr) {
            // Write in output file
            output_raw_file.write(reinterpret_cast<const char *>(raw_events.data()),
                                  kSizeBuffer * sizeof(Metavision::Evt2::RawEvent));
            raw_events_current_ptr = raw_events.data();
        }

        if (!cd_done) {
            if (CD_events_encoder.t < time_high_encoder.th) {
                if (!trigger_done && trigger_events_encoder.t < CD_events_encoder.t) {
                    // Encode Trigger Event
                    trigger_events_encoder.encode(raw_events_current_ptr);
                    trigger_done = !trigger_events_encoder.read_next_line(input_trigger_file);
                } else {
                    // Encode CD Event
                    CD_events_encoder.encode(raw_events_current_ptr);
                    cd_done = !CD_events_encoder.read_next_line(input_cd_file);
                }
            } else {
                if (!trigger_done && trigger_events_encoder.t < time_high_encoder.th) {
                    // Encode Trigger Event
                    trigger_events_encoder.encode(raw_events_current_ptr);
                    trigger_done = !trigger_events_encoder.read_next_line(input_trigger_file);
                } else {
                    // Encode TH
                    time_high_encoder.encode(raw_events_current_ptr);
                }
            }
        } else {
            // If we arrive here it means that trigger_done = false (cf while condition)
            if (trigger_events_encoder.t < time_high_encoder.th) {
                // Encode Trigger Event
                trigger_events_encoder.encode(raw_events_current_ptr);
                trigger_done = !trigger_events_encoder.read_next_line(input_trigger_file);
            } else {
                // Encode TH
                time_high_encoder.encode(raw_events_current_ptr);
            }
        }
        ++raw_events_current_ptr;
    }

    // Write remaining encoded events in output file
    if (raw_events_current_ptr != raw_events.data()) {
        output_raw_file.write(reinterpret_cast<const char *>(raw_events.data()),
                              std::distance(raw_events.data(), raw_events_current_ptr) *
                                  sizeof(Metavision::Evt2::RawEvent));
    }

    return 0;
}
