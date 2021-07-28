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

#include <gtest/gtest.h>

#include "metavision/sdk/core/utils/rate_estimator.h"

using namespace Metavision;

TEST(RateEstimator_GTest, default_ctor) {
    RateEstimator estim;
    EXPECT_EQ(100000, estim.step_time());
    EXPECT_EQ(1000000, estim.window_time());
}

TEST(RateEstimator_GTest, custom_ctor_not_enough_data) {
    // GIVEN a default estimator (step=100ms)
    size_t num_calls = 0;
    RateEstimator estim([&num_calls](timestamp, double, double) { num_calls++; }, 100000, 1000000);

    // WHEN we add one count sample at t = 10us
    estim.add_data(10, 1);

    // THEN the callback is not supposed to be called
    EXPECT_EQ(0, num_calls);
}

TEST(RateEstimator_GTest, custom_ctor_called_once) {
    // GIVEN a default estimator (step=100ms)
    size_t num_calls = 0;
    RateEstimator estim([&num_calls](timestamp, double, double) { num_calls++; }, 100000, 1000000);

    // WHEN we add one count sample at t = 100ms
    estim.add_data(100000, 1);
    estim.add_data(100001, 0);

    // THEN the callback is supposed to be called once
    EXPECT_EQ(1, num_calls);
}

TEST(RateEstimator_GTest, custom_ctor_called_twice) {
    // GIVEN a default estimator (step=100ms)
    size_t num_calls = 0;
    RateEstimator estim([&num_calls](timestamp, double, double) { num_calls++; }, 100000, 1000000);

    // WHEN we add two counts sample at t = 100ms, and t = 200ms
    estim.add_data(100000, 1);
    estim.add_data(200000, 1);
    estim.add_data(200001, 0);

    // THEN the callback is supposed to be called twice, once for each t
    EXPECT_EQ(2, num_calls);
}

TEST(RateEstimator_GTest, custom_ctor_called_twice_bis) {
    // GIVEN a default estimator (step=100ms)
    size_t num_calls = 0;
    RateEstimator estim([&num_calls](timestamp, double, double) { num_calls++; }, 100000, 1000000);

    // WHEN we add counts sample at t = 100ms, and t = 200ms
    estim.add_data(100000, 1);
    estim.add_data(200000, 1);
    estim.add_data(300000, 0);

    // THEN the callback is supposed to be called twice, once for each t
    EXPECT_EQ(2, num_calls);
}

TEST(RateEstimator_GTest, custom_ctor_values_ok) {
    // GIVEN a default estimator (step=100ms, window=1000ms)
    timestamp cb_t;
    double avg_rate = 0, peak_rate = 0;
    RateEstimator estim(
        [&](timestamp t, double avg, double peak) {
            cb_t      = t;
            avg_rate  = avg;
            peak_rate = peak;
        },
        100000, 1000000);

    // WHEN we add a sample of 10 data at t = 100ms
    estim.add_data(100000, 10);
    estim.add_data(100001, 0);

    // THEN the average and peak rate is 100 data/s at t = 100ms
    EXPECT_EQ(100000, cb_t);
    EXPECT_DOUBLE_EQ(100, avg_rate);
    EXPECT_DOUBLE_EQ(100, peak_rate);
}

TEST(RateEstimator_GTest, custom_ctor_values_ok_same_time) {
    // GIVEN a default estimator (step=100ms, window=1000ms)
    timestamp cb_t;
    double avg_rate = 0, peak_rate = 0;
    RateEstimator estim(
        [&](timestamp t, double avg, double peak) {
            cb_t      = t;
            avg_rate  = avg;
            peak_rate = peak;
        },
        100000, 1000000);

    // WHEN we add a sample of 10 data at t = 10ms, and another of 20 data at same time
    estim.add_data(100000, 10);
    estim.add_data(100000, 20);
    estim.add_data(100001, 0);

    // THEN the average and peak rate is 300 data/s at t = 100ms
    EXPECT_EQ(100000, cb_t);
    EXPECT_DOUBLE_EQ(300, avg_rate);
    EXPECT_DOUBLE_EQ(300, peak_rate);
}

TEST(RateEstimator_GTest, custom_ctor_values_ok_2) {
    // GIVEN a default estimator (step=100ms, window=1000ms)
    timestamp cb_t;
    double avg_rate = 0, peak_rate = 0;
    RateEstimator estim(
        [&](timestamp t, double avg, double peak) {
            cb_t      = t;
            avg_rate  = avg;
            peak_rate = peak;
        },
        100000, 1000000);

    // WHEN we add one sample of 3 data at t = 100ms and samples of 1 data at t = 200, 300, ..., 1000 ms
    estim.add_data(100000, 3);
    for (int i = 200000; i <= 1000000; i += 100000) {
        estim.add_data(i, 1);
    }
    estim.add_data(10000001, 0);

    // THEN the average is (3+1+1+...+1)*10/0.1 = 12 data/s and peak rate is 3/0.1 = 30 data/s at t = 1000ms
    EXPECT_EQ(1000000, cb_t);
    EXPECT_DOUBLE_EQ(12, avg_rate);
    EXPECT_DOUBLE_EQ(30, peak_rate);
}

TEST(RateEstimator_GTest, custom_ctor_values_ok_3) {
    std::vector<std::tuple<timestamp, double, double>> values;
    // GIVEN a default estimator (step=100ms, window=1000ms)
    RateEstimator estim([&](timestamp t, double avg, double peak) { values.emplace_back(t, avg, peak); }, 100000,
                        1000000);

    // WHEN we add sample of 1,2,...,10 data at t = 100, 200, ..., 1000ms
    for (int i = 100000; i <= 1000000; i += 100000) {
        estim.add_data(i, i / 100000);
    }
    estim.add_data(10000001, 0);

    // THEN
    // the avg is 1/0.1 = 10 data/s and peak is 1/0.1 = 10 data/s at t = 100ms
    // the avg is (1+2)/0.2 = 15 data/s and peak is 2/0.1 = 20 data/s at t = 200ms
    // the avg is (1+2+3)/0.3 = 20 data/s and peak is 3/0.1 = 30 data/s at t = 300ms
    // ...
    // the avg is (1+2+3+...+i)/(i/10) = (i*(i+1)/2)/(i/10) = 5*(i+1) data/s and peak is i/0.1 data/s at t = i*100ms
    for (size_t i = 1; i <= 10; ++i) {
        EXPECT_EQ(100000 * i, std::get<0>(values[i - 1]));
        EXPECT_DOUBLE_EQ(5 * (i + 1), std::get<1>(values[i - 1]));
        EXPECT_DOUBLE_EQ(i / 0.1, std::get<2>(values[i - 1]));
    }
}

TEST(RateEstimator_GTest, custom_ctor_values_outside_window) {
    std::vector<std::tuple<timestamp, double, double>> values;
    // GIVEN a default estimator (step=100ms, window=1000ms)
    RateEstimator estim(
        [&](timestamp t, double avg, double peak) {
            if (t > 1000000) {
                values.emplace_back(t, avg, peak);
            }
        },
        100000, 1000000);

    // WHEN we add sample of 1, 2, ..., 20 data at t = 100, 200, ..., 2000ms
    for (int i = 100000; i <= 2000000; i += 100000) {
        estim.add_data(i, i / 100000);
    }
    estim.add_data(20000001, 0);

    // THEN
    // the avg is (2+3+...+10+11) data/s and peak is 11/0.1 data/s at t = 1100ms
    // the avg is (3+4+...+11+12) data/s and peak is 12/0.1 data/s at t = 1100ms
    // ...
    // the avg is ((i*(i+1)/2) - ((i-10)*(i-10+1)/2)) = 10*i - 45 data/s and peak is i/0.1 data/s at t = (10+i)*100ms
    for (size_t i = 1; i <= 10; ++i) {
        int k = 10 + i;
        EXPECT_EQ(100000 * k, std::get<0>(values[i - 1]));
        EXPECT_DOUBLE_EQ(10 * k - 45, std::get<1>(values[i - 1]));
        EXPECT_DOUBLE_EQ(k / 0.1, std::get<2>(values[i - 1]));
    }
}