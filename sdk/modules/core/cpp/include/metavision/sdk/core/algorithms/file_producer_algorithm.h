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

#ifndef METAVISION_SDK_CORE_FILE_PRODUCER_ALGORITHM_H
#define METAVISION_SDK_CORE_FILE_PRODUCER_ALGORITHM_H

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <fstream>
#include <iostream>
#include <sstream>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

template<class Event>
class FileProducerAlgorithmT {
public:
    /// @brief Builds a new FileProducerAlgorithmT object
    /// @param filename Name of the file to read data from
    /// @param loop If true, the reading from the file will be looped
    /// @param loop_delay Time interval (in us) between two consecutive loops
    FileProducerAlgorithmT(std::string filename, bool loop = false, timestamp loop_delay = 0);

    /// @brief Destructor
    ~FileProducerAlgorithmT();

    /// @brief Processes events until a given timestamp
    template<class OutputIt>
    inline void process_events(OutputIt d_first, timestamp ts);

    /// @brief If all events have been processed, returns true
    bool is_done();

    /// @brief Gets the timestamp in a given time window
    /// @param time_window Time difference (in us) with the first (or last) event in the file
    /// @param backward If true, search will be from the end of the file (thus giving time last event - time_window)
    inline timestamp get_time_at(timestamp time_window, bool backward);

    /// @brief Starts reproducing the file from a given time
    /// @param start_time Start time
    inline void start_at_time(timestamp start_time);

    /// @brief Gets the total number of events in the file
    uint64_t get_n_tot_ev() const;

    /// @brief Resets the loop length to zero
    static void reset_max_loop_length();

    /// @brief Loads file events to ram
    void load_to_ram();

    /// @brief Gets the width of the sensor producer that recorded the data
    /// @return Width of the sensor
    int get_width() const;

    /// @brief Gets the height of the sensor producer that recorded the data
    /// @return Height of the sensor
    int get_height() const;

    /// @brief Gets the date
    /// @return String of the form YYYY-MM-DD h:m:s (example: 2017-03-08 13:36:44 ) (UTC time)
    /// @note If the date was not found in the header of the file, an empty string is returned
    std::string get_date() const;

protected:
    inline void initialize();

private:
    static timestamp &get_max_loop_length() {
        static timestamp max_loop_length = 0;
        return max_loop_length;
    }

    static void set_max_loop_length(timestamp t) {
        get_max_loop_length() = t;
    }

    template<class OutputIt, class FunctionRead, class FunctionTime, class FunctionIncrement>
    inline void process_output(timestamp ts, OutputIt d_first, const FunctionRead &read_event,
                               const FunctionTime &get_time, const FunctionIncrement &increment_data);

    template<class FunctionRead, class FunctionIncrement>
    inline void loop_through_file(const FunctionRead &read_event, const FunctionIncrement &increment_data);

    // File info
    std::unique_ptr<std::ifstream> file_;
    std::string filename_;

    size_t ev_size_; // Size of a single event stored in the file

    std::streampos position_first_event_; // Position (in the file) of the first event
    std::streampos position_start_event;  // Position (in the file) of the start event (may be different
                                          // from the position_first_event_ if the function start_at_time is called)

    uint64_t start_event_  = 0; // First event to play (may be different than 0 if the function start_at_time is called)
    uint64_t n_tot_events_ = 0; // Tot number of events
    uint64_t current_event_         = 0;
    timestamp time_last_event_read_ = 0;

    timestamp time_first_event_ = 0, time_last_event_ = -1;
    timestamp time_start_event_ = 0; // May be different than time_first_event_ (f the function start_at_time is called)

    timestamp origin_ = 0;

    // OVERFLOW HANDLING
    static constexpr timestamp MAX_TIMESTAMP_32 = (1LL << 32) - 1;
    static constexpr timestamp threshold_       = (1LL << 30); // check only at the two highest bits
    static constexpr timestamp OVERFLOW_LENGTH  = MAX_TIMESTAMP_32 + 1;
    std::vector<uint64_t> events_overflows; // Keep trace of the first events after timestamp overflow
    unsigned int n_times_overflows_ = 0;    // number of times we have looped the timestamp
    timestamp delta_ts_overflow_    = 0;    // delta timestamp due to the overflow

    // LOOP HANDLING
    bool loop_;
    timestamp delta_ts_loop_ = 0;
    unsigned int n_loop      = 0; // Current loop number
    timestamp loop_delay_;        // Time between loops

    // READING FROM FILE
    void *last_data_;
    int version_ = 0;
    std::vector<char> last_buffer_;

    // READING FROM RAM
    bool events_from_ram_ = false;
    std::vector<uint8_t> vrawevents_;
    int times_event_in_vrawevents_ = 0; // number of elements of the vector vrawevents_ a single event occupies

    int width_ = 0, height_ = 0;

    std::string date_;
};

using FileProducerAlgorithm = FileProducerAlgorithmT<Metavision::Event2d>;

} // namespace Metavision

#include "detail/file_producer_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_FILE_PRODUCER_ALGORITHM_H
