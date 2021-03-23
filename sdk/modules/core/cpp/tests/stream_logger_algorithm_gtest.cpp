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
#include <random>
#include <array>

#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/sdk/core/algorithms/stream_logger_algorithm.h"
#include "metavision/sdk/base/events/event2d.h"
#include "metavision/sdk/base/events/detail/event_traits.h"

using namespace Metavision;

class StreamLoggerAlgorithm_GTest : public GTestWithTmpDir {
    using value_type = Event2d;

public:
    StreamLoggerAlgorithm_GTest() = default;

    void validate_file(const std::string &filename, const std::vector<value_type> &buffer) const {
        std::ifstream file(filename.c_str(), std::ios::binary | std::ios::in);
        ASSERT_TRUE(file.is_open());
        /*
         * Validate the header.
         *
         * Check if the header exist in the file.
         */
        GenericHeader header_parser(file);
        ASSERT_TRUE(!header_parser.empty());

        /*
         * Validate the format.
         *
         * To validate the format we expect a key in the header called "Version".
         * This version should be equal to the most recent one. In this case: 2
         */
        constexpr auto ExpectedFormat = "2";
        const auto val                = header_parser.get_field("Version");
        ASSERT_FALSE(val.empty());
        ASSERT_EQ(ExpectedFormat, val);

        /*
         * Validate the date.
         *
         * 1. Check that the DATE_KEY exist.
         * 2. Check until the day because precision on seconds may fail on windows
         */
        const auto date_created = header_parser.get_date();
        ASSERT_FALSE(date_created.empty());
        const auto date = date_created.substr(0, 10);
        ASSERT_EQ(date, (date_created).substr(0, date.size()));

        /*
         * Validate the Event type
         *
         * 1. Check that the event ID is the right one.
         * 2. Check that the event raw size is the right on.
         * Note that the first two characters are the info on event type and size
         */

        constexpr auto ExpectedSize = get_event_size<value_type>();
        constexpr auto ExpectedId   = get_event_id<value_type>();
        unsigned char event_type, event_size;

        file.read(reinterpret_cast<char *>(&event_type), sizeof(char));
        EXPECT_EQ(ExpectedId, event_type);

        file.read(reinterpret_cast<char *>(&event_size), sizeof(char));
        EXPECT_EQ(ExpectedSize, event_size);

        /*
         * Validate the number of events in the file.
         *
         * To do so, we read the total number of remaining bytes in the file,
         * and then we divide by the size of each event.
         *
         * We assume that the remaining data does not include the header size.
         *
         * 1. Check the total number of bytes in the file vs buffer of events.
         * 2. Check the total number of events in the file vs buffer of events.
         */
        std::streampos ev_start = file.tellg();
        file.seekg(0, std::ios::end);

        const auto data_length          = file.tellg() - ev_start;
        const auto expected_data_length = ExpectedSize * buffer.size();
        ASSERT_EQ(data_length, expected_data_length);

        const auto events_length          = static_cast<std::size_t>(data_length) / ExpectedSize;
        const auto expected_events_length = buffer.size();
        ASSERT_EQ(events_length, expected_events_length);

        /*
         * Validate the parsed events
         *
         * Go back to the point where the raw data was saved and start to analyze
         * each event one by one.
         */

        file.seekg(ev_start);
        std::array<char, ExpectedSize> data{};
        auto it_buff = buffer.cbegin();
        for (auto n_ev = 0ul; n_ev < expected_events_length && file.peek() != EOF; ++n_ev, ++it_buff) {
            file.read(data.data(), ExpectedSize);
            const auto ev_created = value_type::read_event(data.data());
            ASSERT_TRUE(it_buff->t == ev_created.t);
            ASSERT_TRUE(it_buff->x == ev_created.x);
            ASSERT_TRUE(it_buff->y == ev_created.y);
            ASSERT_TRUE(it_buff->p == ev_created.p);
        }

        /*
         * Validate that we are at the end of the file
         */
        ASSERT_TRUE(file.peek() == EOF);
    }

protected:
    virtual void SetUp() override {
        filename_ = tmpdir_handler_->get_full_path("tmp_td_mock.dat");
    }
    std::string filename_;
};

TEST_F(StreamLoggerAlgorithm_GTest, test_stream_logger_one_event_all_zeros) {
    std::vector<Event2d> buffer = {Event2d(0, 0, 0, 0)};

    StreamLoggerAlgorithm algo(filename_, 640, 480);
    algo.enable(true);
    algo.process_events(std::cbegin(buffer), std::cend(buffer), 0);
    algo.close();
    validate_file(filename_, buffer);
}

TEST_F(StreamLoggerAlgorithm_GTest, test_stream_logger_one_event_all_ones) {
    std::vector<Event2d> buffer;

    // Create the event
    const auto x_bits = 14;
    const auto y_bits = 14;
    const auto p_bits = 4;

    const auto ts = std::numeric_limits<uint32_t>::max();
    int x((1 << x_bits) - 1), y((1 << y_bits) - 1), p((1 << p_bits) - 1);
    buffer.emplace_back(x, y, p, ts);

    // Run the simulation
    const auto tmax = ts + 1;
    StreamLoggerAlgorithm algo(filename_, 640, 480);
    algo.enable(true);
    algo.process_events(std::cbegin(buffer), std::cend(buffer), tmax);
    algo.close();
    validate_file(filename_, buffer);
}

TEST_F(StreamLoggerAlgorithm_GTest, test_stream_logger_multiple_events) {
    std::vector<Event2d> buffer;

    // Create the events2d
    const auto t_bits = 32;
    const auto x_bits = 14;
    const auto y_bits = 14;
    const auto p_bits = 4;

    uint64_t ts(0);
    int x(0), y(0), p(0);

    for (unsigned int i_t = 0; i_t < t_bits; ++i_t) {
        ts = (1LL << i_t) - 1;
        for (unsigned int i_x = 0; i_x < x_bits; ++i_x) {
            x = (1 << i_x) - 1;
            for (unsigned int i_y = 0; i_y < y_bits; ++i_y) {
                y = (1 << i_y) - 1;
                for (unsigned int i_p = 0; i_p < p_bits; ++i_p) {
                    p = (1 << i_p) - 1;
                    buffer.emplace_back(x, y, p, ts);
                }
            }
        }
    }

    // Run the simulation
    const auto tmax = ts + 1;
    StreamLoggerAlgorithm algo(filename_, 640, 480);
    algo.enable(true);
    algo.process_events(std::cbegin(buffer), std::cend(buffer), tmax);
    algo.close();
    validate_file(filename_, buffer);
}

TEST_F(StreamLoggerAlgorithm_GTest, test_stream_logger_multiple_events_rand) {
    std::vector<Event2d> buffer;

    // Create the events2d
    const auto x_bits = 14;
    const auto y_bits = 14;
    const auto p_bits = 4;

    int x_max((1 << x_bits) - 1), y_max((1 << y_bits) - 1), p_max((1 << p_bits) - 1);
    std::mt19937 mt_rand; // Mersenne twister
    mt_rand.seed(91);

    auto ts = 0;
    for (auto j = 0ul, nev = 100ul; j < nev; ++j, ts += 100) {
        const auto x = static_cast<double>(mt_rand()) / mt_rand.max() * x_max;
        const auto y = static_cast<double>(mt_rand()) / mt_rand.max() * y_max;
        const auto p = static_cast<double>(mt_rand()) / mt_rand.max() * p_max;
        buffer.emplace_back(x, y, p, ts);
    }

    // Run the simulation
    const auto tmax = ts + 1;
    StreamLoggerAlgorithm algo(filename_, 640, 480);
    algo.enable(true);
    algo.process_events(std::cbegin(buffer), std::cend(buffer), tmax);
    algo.close();
    validate_file(filename_, buffer);
}
