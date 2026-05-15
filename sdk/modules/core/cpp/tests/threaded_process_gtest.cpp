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

#include <vector>
#include <atomic>
#include <fstream>
#include <gtest/gtest.h>

#include "metavision/sdk/core/utils/threaded_process.h"

class ThreadedProcess_GTest : public ::testing::Test {
public:
    ThreadedProcess_GTest() {}

    virtual ~ThreadedProcess_GTest() {}

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

TEST_F(ThreadedProcess_GTest, not_start_process_is_not_active) {
    // GIVEN a threaded process
    Metavision::ThreadedProcess threaded_process;

    // WHEN The process is not started
    // THEN the process is not active
    ASSERT_FALSE(threaded_process.is_active());
}

TEST_F(ThreadedProcess_GTest, can_not_start_several_times_the_processing_thread) {
    // GIVEN a threaded process
    Metavision::ThreadedProcess threaded_process;

    // WHEN we start it several times
    // THEN only the first time is successful
    ASSERT_TRUE(threaded_process.start());
    ASSERT_FALSE(threaded_process.start());
    ASSERT_FALSE(threaded_process.start());
    ASSERT_FALSE(threaded_process.start());
    ASSERT_FALSE(threaded_process.start());
    ASSERT_FALSE(threaded_process.start());
}

TEST_F(ThreadedProcess_GTest, process_is_not_active_if_stopped) {
    // GIVEN a threaded process
    Metavision::ThreadedProcess threaded_process;

    // WHEN we start it
    ASSERT_TRUE(threaded_process.start());
    // THEN it is active
    ASSERT_TRUE(threaded_process.is_active());

    // WHEN we stop it
    threaded_process.abort();
    ASSERT_FALSE(threaded_process.is_active());
}

TEST_F(ThreadedProcess_GTest, no_task_can_be_added_if_process_not_started) {
    // GIVEN a threaded process and a task
    int task_processed_count               = 0;
    Metavision::ThreadedProcess::Task task = [&]() { ++task_processed_count; };

    // WHEN The process is not started and a task pushed
    {
        Metavision::ThreadedProcess threaded_process;
        threaded_process.add_task(task);
    }
    // THEN the tasks are not processed
    ASSERT_EQ(0, task_processed_count);
}

TEST_F(ThreadedProcess_GTest, task_can_be_added_if_process_started) {
    // GIVEN a threaded process
    // AND GIVEN a task
    int task_processed_count = 0;
    std::condition_variable proceed_cond;
    std::mutex proceed_mutex;
    bool proceed{false};
    Metavision::ThreadedProcess::Task task = [&]() {
        ++task_processed_count;
        std::lock_guard<std::mutex> lock(proceed_mutex);
        proceed = true;
        proceed_cond.notify_all();
    };

    // WHEN The process is started and a task pushed
    {
        Metavision::ThreadedProcess threaded_process;
        threaded_process.start();
        ASSERT_TRUE(threaded_process.is_active());
        threaded_process.add_task(task);
        std::unique_lock<std::mutex> lock(proceed_mutex);
        ASSERT_TRUE(proceed_cond.wait_for(lock, std::chrono::seconds(1), [&]() { return proceed; }));
    }

    // THEN the task is processed
    ASSERT_EQ(1, task_processed_count);
}

TEST_F(ThreadedProcess_GTest, tasks_are_executed_in_order) {
    // GIVEN a threaded process
    // AND GIVEN 3 tasks
    std::vector<int> task_processed;

    std::condition_variable proceed_cond;
    std::mutex proceed_mutex;
    bool proceed{false};

    Metavision::ThreadedProcess::Task task1 = [&]() { task_processed.push_back(1); };
    Metavision::ThreadedProcess::Task task2 = [&]() { task_processed.push_back(2); };
    Metavision::ThreadedProcess::Task task3 = [&]() {
        task_processed.push_back(3);
        std::lock_guard<std::mutex> lock(proceed_mutex);
        proceed = true;
        proceed_cond.notify_all();
    };

    // WHEN The process is started and the tasks are pushed
    {
        Metavision::ThreadedProcess threaded_process;
        threaded_process.start();
        ASSERT_TRUE(threaded_process.is_active());
        threaded_process.add_task(task1);
        threaded_process.add_task(task2);
        threaded_process.add_task(task3);

        std::unique_lock<std::mutex> lock(proceed_mutex);
        ASSERT_TRUE(proceed_cond.wait_for(lock, std::chrono::seconds(1), [&]() { return proceed; }));
    }

    // THEN the 3 tasks are processed in the order they were pushed
    const std::vector<int> expected{{1, 2, 3}};
    ASSERT_EQ(expected, task_processed);
}

TEST_F(ThreadedProcess_GTest, calling_stop_prevents_adding_new_tasks) {
    // GIVEN a threaded process and 3 tasks
    std::vector<int> task_processed;

    std::condition_variable proceed_cond;
    std::mutex proceed_mutex;
    bool proceed{false};

    Metavision::ThreadedProcess::Task task1 = [&]() { task_processed.push_back(1); };
    Metavision::ThreadedProcess::Task task2 = [&]() {
        task_processed.push_back(2);

        std::lock_guard<std::mutex> lock(proceed_mutex);
        proceed = true;
        proceed_cond.notify_all();
    };

    Metavision::ThreadedProcess::Task task3 = [&]() { task_processed.push_back(3); };

    // WHEN The process is started and the 2 firsts tasks are pushed
    // AND WHEN stop is called before adding the third task
    {
        Metavision::ThreadedProcess threaded_process;
        threaded_process.start();
        ASSERT_TRUE(threaded_process.is_active());
        threaded_process.add_task(task1);
        threaded_process.add_task(task2);

        {
            std::unique_lock<std::mutex> lock(proceed_mutex);
            ASSERT_TRUE(proceed_cond.wait_for(lock, std::chrono::seconds(1), [&]() { return proceed; }));
        }
        threaded_process.abort();
        ASSERT_FALSE(threaded_process.is_active());
        threaded_process.add_task(task3);
    }

    // THEN Only 2 tasks are processed in the order they were pushed
    const std::vector<int> expected{{1, 2}};
    ASSERT_EQ(expected, task_processed);
}

TEST_F(ThreadedProcess_GTest, calling_stop_without_abort_prevents_adding_new_tasks) {
    // GIVEN a threaded process
    // AND GIVEN 3 tasks
    std::vector<int> task_processed;

    std::condition_variable proceed_cond;
    std::mutex proceed_mutex;
    bool proceed{false};

    Metavision::ThreadedProcess::Task task1 = [&]() { task_processed.push_back(1); };
    Metavision::ThreadedProcess::Task task2 = [&]() {
        task_processed.push_back(2);

        std::lock_guard<std::mutex> lock(proceed_mutex);
        proceed = true;
        proceed_cond.notify_all();
    };

    Metavision::ThreadedProcess::Task task3 = [&]() { task_processed.push_back(3); };

    // WHEN The process is started and the 2 firsts tasks are pushed
    // AND THEN stop without the request to abort pending tasks is called before adding the third task
    {
        Metavision::ThreadedProcess threaded_process;
        threaded_process.start();
        ASSERT_TRUE(threaded_process.is_active());
        threaded_process.add_task(task1);
        threaded_process.add_task(task2);
        {
            std::unique_lock<std::mutex> lock(proceed_mutex);
            ASSERT_TRUE(proceed_cond.wait_for(lock, std::chrono::seconds(1), [&]() { return proceed; }));
        }
        threaded_process.stop();
        ASSERT_FALSE(threaded_process.is_active());
        threaded_process.add_task(task3);
    }

    // THEN Only 2 tasks are processed in the order they were pushed
    const std::vector<int> expected{{1, 2}};
    ASSERT_EQ(expected, task_processed);
}

TEST_F(ThreadedProcess_GTest, calling_stop_does_not_run_pending_tasks) {
    // GIVEN a threaded process
    Metavision::ThreadedProcess threaded_process;
    std::condition_variable proceed_cond;
    std::mutex proceed_mutex;
    bool proceed{false};

    // AND GIVEN a task that adds a new task to the process when executed
    Metavision::ThreadedProcess::Task task;
    task = [&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // just to avoid adding too many tasks in the queue
        threaded_process.add_task(task);
    };

    // WHEN The process is started and the tasks are pushed
    // AND WHEN but stop is called
    {
        threaded_process.start();
        ASSERT_TRUE(threaded_process.is_active());
        threaded_process.add_task(task);
        threaded_process.abort();
        ASSERT_FALSE(threaded_process.is_active());
    }

    // THEN The abort request using stop breaks the repeating process of adding new tasks: the process leaves
    SUCCEED();
}

TEST_F(ThreadedProcess_GTest, calling_stop_without_abort_runs_pending_tasks) {
    // GIVEN a threaded process
    Metavision::ThreadedProcess threaded_process;
    std::vector<int> task_processed;

    std::condition_variable proceed_cond;
    std::mutex proceed_mutex;
    bool proceed{false};

    // AND GIVEN a task that adds a new task to the process when executed
    int tasks_to_add = 10;
    Metavision::ThreadedProcess::Task task;
    task = [&]() {
        if (tasks_to_add > 0) {
            threaded_process.add_task(task);
            task_processed.push_back(tasks_to_add);
            --tasks_to_add;
        }
    };

    // WHEN The process is started and the tasks are pushed
    // AND WHEN but stop is called
    {
        threaded_process.start();
        ASSERT_TRUE(threaded_process.is_active());
        threaded_process.add_task(task);
        threaded_process.stop();
        ASSERT_FALSE(threaded_process.is_active());
    }

    // THEN The request to not abort pending task using stop allows to run the third task even though it was pending.
    // The 3 tasks are processed in the order they were pushed
    const std::vector<int> expected{{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}};
    ASSERT_EQ(expected, task_processed);
}

TEST_F(ThreadedProcess_GTest, no_repeating_task_can_be_added_if_process_not_started) {
    // GIVEN a threaded process and a repeating task
    int task_processed_count                        = 0;
    Metavision::ThreadedProcess::RepeatingTask task = [&]() {
        ++task_processed_count;
        return task_processed_count < 3;
    };

    // WHEN The process is not started and a task pushed
    {
        Metavision::ThreadedProcess threaded_process;
        threaded_process.add_repeating_task(task);
    }
    // THEN the tasks are not processed
    ASSERT_EQ(0, task_processed_count);
}

TEST_F(ThreadedProcess_GTest, repeating_task_can_be_added_if_process_started) {
    // GIVEN a threaded process and a repeating task
    int task_processed_count = 0;
    std::condition_variable proceed_cond;
    std::mutex proceed_mutex;
    bool proceed{false};
    Metavision::ThreadedProcess::RepeatingTask task = [&]() {
        ++task_processed_count;
        if (task_processed_count == 3) {
            std::lock_guard<std::mutex> lock(proceed_mutex);
            proceed = true;
            proceed_cond.notify_all();
            return false;
        }

        return true;
    };

    // WHEN The process is started and a task pushed
    {
        Metavision::ThreadedProcess threaded_process;
        threaded_process.start();
        ASSERT_TRUE(threaded_process.is_active());
        threaded_process.add_repeating_task(task);
        std::unique_lock<std::mutex> lock(proceed_mutex);
        ASSERT_TRUE(proceed_cond.wait_for(lock, std::chrono::seconds(1), [&]() { return proceed; }));
    }

    // THEN the repeating task is called three times
    ASSERT_EQ(3, task_processed_count);
}

TEST_F(ThreadedProcess_GTest, calling_stop_stops_repeating_tasks_process) {
    // GIVEN a threaded process and repeating task
    std::condition_variable proceed_cond;
    std::mutex proceed_mutex;
    bool proceed{false};
    Metavision::ThreadedProcess::RepeatingTask task = [&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // just to avoid adding too many tasks in the queue
        return true;
    };

    // WHEN The process is started and the repeating task is pushed
    // AND THEN stop is called
    {
        Metavision::ThreadedProcess threaded_process;
        threaded_process.start();
        ASSERT_TRUE(threaded_process.is_active());
        threaded_process.add_repeating_task(task);
        threaded_process.abort();
        ASSERT_FALSE(threaded_process.is_active());
    }

    // THEN The abort request using stop prevents to continue running the repeating tasks
    SUCCEED();
}

TEST_F(ThreadedProcess_GTest, calling_stop_without_abort_wait_end_of_repeating_tasks_cycle) {
    // GIVEN a threaded process and repeating task
    int task_processed_count = 0;
    std::condition_variable proceed_cond;
    std::mutex proceed_mutex;
    bool proceed{false};
    Metavision::ThreadedProcess::RepeatingTask task = [&]() {
        ++task_processed_count;
        return task_processed_count < 3;
    };

    // WHEN The process is started and the repeating task is pushed
    // AND THEN stop without abort is called
    {
        Metavision::ThreadedProcess threaded_process;
        threaded_process.start();
        ASSERT_TRUE(threaded_process.is_active());
        threaded_process.add_repeating_task(task);
        threaded_process.stop();
        ASSERT_FALSE(threaded_process.is_active());
    }

    // THEN The request to not abort repeating task using stop waits the end of the repeating process.
    ASSERT_EQ(3, task_processed_count);
}
