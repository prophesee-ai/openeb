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

#include <atomic>
#include <thread>
#include <future>
#include <chrono>
#include <boost/any.hpp>
#include <gtest/gtest.h>
#include <stdexcept>

#include "metavision/sdk/core/pipeline/pipeline.h"
#include "metavision/sdk/core/pipeline/stage.h"

using namespace Metavision;

namespace {
struct MockAlgorithm {
    template<typename InputIt, typename OutputIt>
    void process_events(InputIt begin, InputIt end, OutputIt d_begin) {
        std::copy(begin, end, d_begin);
    }
};

struct MockProducingStage : public BaseStage {
    MockProducingStage() {
        set_starting_callback([this] {
            thread_ = std::thread([this] {
                while (true) {
                    if (stopped_)
                        break;
                    if (!produce_impl()) {
                        break;
                    }
                }
                if (!stopped_)
                    complete();
            });
        });
        set_stopping_callback([this] {
            stopped_ = true;
            if (thread_.joinable()) {
                thread_.join();
            }
        });
    }

    virtual bool produce_impl() {
        return true;
    }
    std::thread thread_;
    std::atomic<bool> stopped_{false};
};

struct VectorProducingStage : public MockProducingStage {
    VectorProducingStage(const std::vector<int> &datas) : datas(datas), step(0) {}
    bool produce_impl() override {
        if (step < datas.size()) {
            produce(datas[step++]);
            return true;
        }
        return false;
    }
    std::vector<int> datas;
    size_t step;
};

struct MockConsumingStage : public BaseStage {
    MockConsumingStage(bool delayed_start = false) {
        set_starting_callback([delayed_start, this] {
            if (delayed_start) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            started_ = true;
        });
        set_consuming_callback([this](const boost::any &data) {
            try {
                if (started_)
                    datas.emplace_back(boost::any_cast<int>(data));
            } catch (boost::bad_any_cast &c) {}
        });
    }
    std::vector<int> datas;
    std::atomic<bool> started_{false};
};
} // namespace

TEST(PipelineTest, no_stages) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a default constructed pipeline does not contain any stages
    Pipeline p;
    EXPECT_EQ(size_t(0), p.count());
    EXPECT_TRUE(p.empty());
}

TEST(PipelineTest, add_stage) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a pipeline with one added stage has the correct invariants
    Pipeline p;
    EXPECT_EQ(size_t(0), p.count());
    EXPECT_TRUE(p.empty());
    p.add_stage(std::make_unique<Stage>());
    EXPECT_EQ(size_t(1), p.count());
    EXPECT_FALSE(p.empty());
}

TEST(PipelineTest, add_algorithm_stage) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a pipeline with one added (algorithm) stage has the correct invariants
    Pipeline p;
    EXPECT_EQ(size_t(0), p.count());
    EXPECT_TRUE(p.empty());
    p.add_algorithm_stage(std::make_unique<MockAlgorithm>());
    EXPECT_EQ(size_t(1), p.count());
    EXPECT_FALSE(p.empty());
}

TEST(PipelineTest, add_stage_default_auto_detach) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a default constructed pipeline does not automatically detach stages
    Pipeline p;
    auto &stage = p.add_stage(std::make_unique<Stage>());
    EXPECT_FALSE(stage.is_detached());
}

TEST(PipelineTest, add_stage_auto_detach) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a pipeline with auto-detach automatically detaches stages
    Pipeline p(true);
    auto &stage = p.add_stage(std::make_unique<Stage>());
    EXPECT_TRUE(stage.is_detached());
}

TEST(PipelineTest, add_stage_no_auto_detach) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a pipeline with disabled auto-detach does not automatically detach stages
    Pipeline p(false);
    auto &stage = p.add_stage(std::make_unique<Stage>());
    EXPECT_FALSE(stage.is_detached());
}

TEST(PipelineTest, add_stage_auto_detach_precedence) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a pipeline with auto-detach can not detach un-detachable stages
    Pipeline p(true);
    auto &stage = p.add_stage(std::make_unique<Stage>(false));
    EXPECT_FALSE(stage.is_detached());
}

TEST(PipelineTest, add_stage_with_previous) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that add_stage correctly passes the previous stage arg
    Pipeline p;
    Stage s1;
    auto &s2 = p.add_stage(std::make_unique<Stage>(), s1);
    EXPECT_EQ(std::unordered_set<BaseStage *>{&s1}, s2.previous_stages());
}

TEST(PipelineTest, add_algorithm_stage_with_previous) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that add_algorithm_stage correctly passes the previous stage arg
    Pipeline p;
    Stage s1;
    auto &s2 = p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
    EXPECT_EQ(std::unordered_set<BaseStage *>{&s1}, s2.previous_stages());
}

TEST(PipelineTest, add_algorithm_stage_default_enabled) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that add_algorithm_stage correctly passes the enabled flag to the algo by default
    Pipeline p;
    auto &s = p.add_algorithm_stage(std::make_unique<MockAlgorithm>());
    EXPECT_TRUE(s.is_enabled());
}

TEST(PipelineTest, add_algorithm_stage_disabled) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that add_algorithm_stage correctly passes the enabled flag to the algo
    Pipeline p;
    auto &s = p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), false);
    EXPECT_FALSE(s.is_enabled());
}

TEST(PipelineTest, add_algorithm_stage_with_previous_disabled) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that add_algorithm_stage correctly passes the previous stages and
    // the enabled flag to the algo
    Pipeline p;
    Stage s1;
    auto &s2 = p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1, false);
    EXPECT_EQ(std::unordered_set<BaseStage *>{&s1}, s2.previous_stages());
    EXPECT_FALSE(s2.is_enabled());
}

TEST(PipelineTest, remove_stage) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that after removing a stage from a pipeline, the count is OK
    Pipeline p;
    auto &stage = p.add_stage(std::make_unique<Stage>(false));
    EXPECT_EQ(size_t(1), p.count());
    EXPECT_FALSE(p.empty());
    p.remove_stage(stage);
    EXPECT_EQ(size_t(0), p.count());
    EXPECT_TRUE(p.empty());
}

TEST(PipelineTest, destructor_when_empty) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the destructor does not block when the pipeline is empty
    Pipeline p;
}

TEST(PipelineTest, destructor_with_no_consumers) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the destructor does not block when the pipeline does not contain consumer
    Pipeline p;
    p.add_stage(std::make_unique<Stage>());
    p.add_stage(std::make_unique<Stage>());
}

TEST(PipelineTest, destructor_with_one_undetached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the destructor does not block if the producer and consumer are
    // both run on the main thread
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
}

TEST(PipelineTest, destructor_with_one_detached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the destructor does not block on a (well behaving) pipeline
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
}

TEST(PipelineTest, step_when_empty) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that step returns false when the pipeline is empty
    Pipeline p;
    EXPECT_FALSE(p.step());
}

TEST(PipelineTest, step_with_no_consumers) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that step finishes immediately when the pipeline does not contain consumer
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<Stage>());
    auto &s2 = p.add_stage(std::make_unique<Stage>());
    EXPECT_FALSE(p.step());
}

TEST(PipelineTest, step_with_one_undetached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that step returns true at least once if the producer and consumer are
    // both run on the main thread
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    auto &s2 = p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
    EXPECT_TRUE(p.step());
    while (p.step()) {}
}

TEST(PipelineTest, step_with_one_detached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that repeatedly calling step does not block (returns at some point) on a (well behaving) pipeline
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    auto &s2 = p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
    while (p.step()) {}
}

TEST(PipelineTest, run_when_empty) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that run does not block (returns at some point) when the pipeline is empty
    Pipeline p;
    p.run();
}

TEST(PipelineTest, run_with_no_consumers) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that run does not block (returns at some point) when the pipeline does not contain consumer
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1}));
    auto &s2 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{2}));
    p.run();
}

TEST(PipelineTest, run_with_one_undetached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that run does not block (returns at some point) if the producer and consumer are
    // both run on the main thread
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    auto &s2 = p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
    p.run();
}

TEST(PipelineTest, run_with_one_detached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that run does not block (returns at some point) on a (well behaving) pipeline
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    auto &s2 = p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
    p.run();
}

TEST(PipelineTest, destructor_when_empty_pipeline_started) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the destructor does not block when the pipeline is empty and started (with step)
    Pipeline p;
    p.step();
}

TEST(PipelineTest, destructor_with_no_consumers_pipeline_started) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the destructor does not block when the pipeline does not contain consumer and is started (with step)
    Pipeline p;
    p.add_stage(std::make_unique<Stage>());
    p.add_stage(std::make_unique<Stage>());
}

TEST(PipelineTest, destructor_with_one_undetached_consumer_pipeline_started) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the destructor does not block if the producer and consumer are
    // both run on the main thread and the pipeline is started (with step)
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
}

TEST(PipelineTest, destructor_with_one_detached_consumer_pipeline_started) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the destructor does not block on a (well behaving) pipeline, started (with step)
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
}

TEST(PipelineTest, detach_when_started) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that adding a stage on a started pipeline throws
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>());
    auto &s2 = p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
    p.step();
    EXPECT_FALSE(s1.detach());
}

TEST(PipelineTest, add_stage_when_started) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that adding a stage on a started pipeline throws
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>());
    auto &s2 = p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
    p.step();
    EXPECT_THROW(p.add_stage(std::make_unique<Stage>(), s1), std::runtime_error);
}

TEST(PipelineTest, remove_stage_when_started) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that removing a stage on a started pipeline throws
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>());
    auto &s2 = p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
    p.step();
    EXPECT_THROW(p.remove_stage(s2), std::runtime_error);
}

TEST(PipelineTest, status_when_empty) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the status of a default constructed pipeline is Inactive
    Pipeline p;
    EXPECT_EQ(Pipeline::Status::Inactive, p.status());
}

TEST(PipelineTest, status_when_started) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that status of a pipeline that was started but not finished is Started
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>());
    p.add_stage(std::make_unique<Stage>(), s1);
    p.step();
    EXPECT_EQ(Pipeline::Status::Started, p.status());
}

TEST(PipelineTest, status_when_cancelled) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that status of a pipeline that was cancelled is Cancelled
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>());
    p.add_stage(std::make_unique<Stage>(), s1);
    p.step();
    p.cancel();
    EXPECT_EQ(Pipeline::Status::Cancelled, p.status());
}

TEST(PipelineTest, status_when_completed) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that status of a pipeline that has finished is Completed
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    auto &s2 = p.add_algorithm_stage(std::make_unique<MockAlgorithm>(), s1);
    while (p.step()) {}
    EXPECT_EQ(Pipeline::Status::Completed, p.status());
}

TEST(PipelineTest, process_all_data_one_producer_one_undetached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a pipeline actually produces and forward data to all stages before
    // completing
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    auto &s2 = p.add_stage(std::make_unique<MockConsumingStage>(), s1);
    while (p.step()) {}
    EXPECT_EQ(Pipeline::Status::Completed, p.status());
    EXPECT_EQ(std::vector<int>({1, 2}), s2.datas);
}

TEST(PipelineTest, process_all_data_one_producer_one_detached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a pipeline actually produces and forward data to all stages before
    // completing
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    auto &s2 = p.add_stage(std::make_unique<MockConsumingStage>(), s1);
    while (p.step()) {}
    EXPECT_EQ(Pipeline::Status::Completed, p.status());
    EXPECT_EQ(std::vector<int>({1, 2}), s2.datas);
}

TEST(PipelineTest, process_all_data_two_producer_two_undetached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a pipeline actually produces and forward data to all stages before
    // completing
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    auto &s2 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{3, 4, 5}));
    auto &s3 = p.add_stage(std::make_unique<MockConsumingStage>(), s1);
    auto &s4 = p.add_stage(std::make_unique<MockConsumingStage>(), s2);
    while (p.step()) {}
    EXPECT_EQ(Pipeline::Status::Completed, p.status());
    EXPECT_EQ(std::vector<int>({1, 2}), s3.datas);
    EXPECT_EQ(std::vector<int>({3, 4, 5}), s4.datas);
}

TEST(PipelineTest, process_all_data_two_producer_two_detached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a pipeline actually produces and forward data to all stages before
    // completing
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    auto &s2 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{3, 4, 5}));
    auto &s3 = p.add_stage(std::make_unique<MockConsumingStage>(), s1);
    auto &s4 = p.add_stage(std::make_unique<MockConsumingStage>(), s2);
    while (p.step()) {}
    EXPECT_EQ(Pipeline::Status::Completed, p.status());
    EXPECT_EQ(std::vector<int>({1, 2}), s3.datas);
    EXPECT_EQ(std::vector<int>({3, 4, 5}), s4.datas);
}

TEST(PipelineTest, cancel_when_consuming_with_undetached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that calling cancel from the consuming callback of an undetached consumer
    // actually cancels the pipeline, at multiple stages of consumption
    size_t count = 5;
    std::vector<int> datas(count);
    for (size_t i = 0; i < count; ++i) {
        datas[i] = i;
    }
    for (size_t i = 1; i <= count; ++i) {
        Pipeline p;
        auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(datas));
        auto &s2 = p.add_stage(std::make_unique<Stage>(), s1);
        std::vector<int> cdatas;
        s2.set_consuming_callback([&p, &cdatas, i](const boost::any &data) {
            try {
                cdatas.emplace_back(boost::any_cast<int>(data));
                if (cdatas.size() == i)
                    p.cancel();
            } catch (boost::bad_any_cast &) { FAIL() << "Bad cast not supposed to be caught"; }
        });
        p.run();
        EXPECT_EQ(Pipeline::Status::Cancelled, p.status());
        EXPECT_EQ(std::vector<int>(datas.begin(), datas.begin() + i), cdatas);
    }
}

TEST(PipelineTest, cancel_when_consuming_with_detached_consumer) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that calling cancel from the consuming callback of an undetached consumer
    // actually cancels the pipeline, at multiple stages of consumption
    size_t count = 5;
    std::vector<int> datas(count);
    for (size_t i = 0; i < count; ++i) {
        datas[i] = i;
    }
    for (size_t i = 0; i < count; ++i) {
        Pipeline p(true);
        auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(datas));
        auto &s2 = p.add_stage(std::make_unique<Stage>(), s1);
        size_t j = 0;
        s2.set_consuming_callback([&p, &j, i](const boost::any &data) {
            try {
                if (j++ == i)
                    p.cancel();
            } catch (boost::bad_any_cast &) { FAIL() << "Bad cast not supposed to be caught"; }
        });
        p.run();
        EXPECT_EQ(Pipeline::Status::Cancelled, p.status());
    }
}

TEST(PipelineTest, cancel_with_no_remaining_tasks_on_main_thread) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that calling cancel when one stage is undetached and waiting for tasks
    // does not block forever
    Pipeline p(false);
    auto &s1 = p.add_stage(std::make_unique<Stage>());
    auto &s2 = p.add_stage(std::make_unique<Stage>(), s1);
    s2.detach();
    auto &s3 = p.add_stage(std::make_unique<Stage>(), s1);
    std::atomic<bool> run{false};
    // clang-format off
    std::thread([&run, &p] {
        while (!run) {
        }
        p.cancel();
    }).detach();
    // clang-format on
    while (p.step()) {
        run = true;
    }
    EXPECT_EQ(Pipeline::Status::Cancelled, p.status());
}

TEST(PipelineTest, DISABLED_consume_before_start) { // TODO Intermittently fails on Jenkins - To be fixed with MV-980";

    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that when a stage takes time to start, no data is lost, e.g. if the consuming
    // callback is called too soon.
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    auto &s2 = p.add_stage(std::make_unique<MockConsumingStage>(true), s1);
    while (p.step()) {}
    EXPECT_EQ(Pipeline::Status::Completed, p.status());
    EXPECT_EQ(std::vector<int>({1, 2}), s2.datas);
}

TEST(PipelineTest, step_callbacks_are_called_in_sequence) {
    static constexpr int PRE_STEP_DATA_ID  = 1; // Id pushed in the output vector by the pre step callback
    static constexpr int PRODUCED_DATA_ID  = 2; // Id pushed in the output vector by the consuming callback
    static constexpr int POST_STEP_DATA_ID = 3; // Id pushed in the output vector by the post callback

    // GIVEN a pipeline with:
    // - a global output vector
    // - a producer, executing from a separate thread, that produces integers of the same value (i.e. PRODUCED_DATA_ID)
    // - a consumer, executing from the main thread, that pushes the incoming data in the global output vector
    // - a pre-step callback, called from the main thread, that pushes its ID (i.e. PRE_STEP_DATA_ID) in the global
    // output vector
    // - a post-step callback, called from the main thread, that pushes its ID (i.e. POST_STEP_DATA_ID) in the global
    // output vector
    Pipeline p(true);

    std::vector<int> inputs(2, PRODUCED_DATA_ID);
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(inputs));

    std::vector<int> outputs;
    auto s2 = std::make_unique<Stage>(false);
    s2->set_consuming_callback([&outputs](const boost::any &in) {
        try {
            outputs.emplace_back(boost::any_cast<int>(in));
        } catch (boost::bad_any_cast &c) {}
    });
    p.add_stage(std::move(s2), s1);

    p.add_pre_step_callback([&outputs]() { outputs.emplace_back(PRE_STEP_DATA_ID); });
    p.add_post_step_callback([&outputs]() { outputs.emplace_back(POST_STEP_DATA_ID); });

    // WHEN we run the pipeline
    p.run();

    // THEN
    // - the pipeline completes
    // - the consumer is called in between a pre-step and a post-step callback
    // - because of the multi-threaded aspect of both the producer and the pipeline, there may be several calls to the
    // pre and post-step callbacks before and after the consumer is called. The final output data looks like
    // 1 3 1 3 ... 1 2 3 ... 1 2 3 ... 1 3 1 3
    EXPECT_EQ(Pipeline::Status::Completed, p.status());

    auto step_cb_data_it  = outputs.cbegin();
    auto consumed_data_it = outputs.cend();
    using SizeType        = std::vector<int>::size_type;
    for (SizeType i = 0; i < inputs.size(); ++i) {
        consumed_data_it = std::find(step_cb_data_it, outputs.cend(), PRODUCED_DATA_ID);
        EXPECT_TRUE(consumed_data_it != outputs.cend());            // the produced data is found in the output vector
        EXPECT_TRUE(std::prev(consumed_data_it) != outputs.cend()); // there is data before the produced data
        EXPECT_TRUE(std::next(consumed_data_it) != outputs.cend()); // there is data after the produced data
        EXPECT_EQ(1, *std::prev(consumed_data_it)); // the data before corresponds to the pre-step callback's
        EXPECT_EQ(3, *std::next(consumed_data_it)); // the data after corresponds to the post-step callback's

        const auto step_cb_data_range_begin = step_cb_data_it;
        const auto step_cb_data_range_end   = std::prev(consumed_data_it);
        const auto n_elts                   = std::distance(step_cb_data_range_begin, step_cb_data_range_end);

        // the pre and post-step callbacks may have been called several times before the consumer
        EXPECT_LE(0, n_elts);
        EXPECT_EQ(0, n_elts % 2); // if so, they have been called the same number of times

        // even indexes correspond to the pre-step callbacks while the odd ones correspond to the post-step callbacks
        for (int j = 0; j < n_elts; ++j)
            EXPECT_EQ((j % 2) ? POST_STEP_DATA_ID : PRE_STEP_DATA_ID, *(step_cb_data_it + j));

        step_cb_data_it = consumed_data_it + 2;
    }

    // the pre and post-step callbacks may also have been called several times after the consumer
    if (step_cb_data_it != outputs.cend()) {
        const auto n_elts = std::distance(step_cb_data_it, outputs.cend());
        EXPECT_EQ(0, n_elts % 2);

        for (int j = 0; j < n_elts; ++j)
            EXPECT_EQ((j % 2) ? POST_STEP_DATA_ID : PRE_STEP_DATA_ID, *(step_cb_data_it + j));
    }
}

TEST(PipelineTest, all_step_callbacks_are_called) {
    static constexpr int N_STEP_CBS = 3;

    // GIVEN a pipeline with:
    // - a producer
    // - a consumer
    // - a global output map where step callbacks' ids are registered
    // - N_STEP_CBS pre-step callbacks pushing their unique id in the global output map
    // - N_STEP_CBS post-step callbacks pushing their unique id in the global output map
    std::map<int, bool> called_step_cbs;

    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<VectorProducingStage>(std::vector<int>{1, 2}));
    auto &s2 = p.add_stage(std::make_unique<MockConsumingStage>(true), s1);

    for (int i = 0; i < N_STEP_CBS; ++i) {
        p.add_pre_step_callback([=, &called_step_cbs]() { called_step_cbs[i] = true; });
        p.add_post_step_callback([=, &called_step_cbs]() { called_step_cbs[N_STEP_CBS + i] = true; });
    }

    // WHEN we run the pipeline
    p.run();

    // THEN all the step callbacks' ids are found in the global output map meaning that all the registered step
    // callbacks have been called
    for (int i = 0; i < 2 * N_STEP_CBS; ++i) {
        const auto it = called_step_cbs.find(i);
        EXPECT_TRUE(it != called_step_cbs.cend());
        EXPECT_EQ(true, it->second);
    }
}

TEST(PipelineTest, on_setup_callback_is_called) {
    // GIVEN a pipeline with a single stage that sets a setup callback and a pre step callback from inside the latter
    Pipeline p;
    auto stage = std::make_unique<Stage>(false);

    bool has_setup_cb_been_called    = false;
    bool has_pre_step_cb_been_called = false;

    auto &stage_ref = *stage;

    stage->set_setup_callback([&]() {
        has_setup_cb_been_called = true;

        // We add this callback to detect potential deadlocks when setting callbacks from the setup one (if a deadlock
        // occurs, the test will hang on and a timeout will expire making this test fail, and as a result, make
        // the deadlock detectable).
        stage_ref.pipeline().add_post_step_callback([&]() { has_pre_step_cb_been_called = true; });
    });

    p.add_stage(std::move(stage));

    // WHEN we run the pipeline
    p.run();

    // THEN all the callbacks are called
    EXPECT_TRUE(has_setup_cb_been_called);
    EXPECT_TRUE(has_pre_step_cb_been_called);
}
