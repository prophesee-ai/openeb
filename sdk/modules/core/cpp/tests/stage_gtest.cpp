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
#include <gtest/gtest.h>
#include <stdexcept>

#include "metavision/sdk/core/pipeline/pipeline.h"
#include "metavision/sdk/core/pipeline/stage.h"

using namespace Metavision;

class MockStage : public Stage {
public:
    void complete() {
        Stage::complete();
    }
};

TEST(StageTest, default_constructor) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a default constructed stage is not detached
    Stage s;
    EXPECT_FALSE(s.is_detached());
}

TEST(StageTest, undetachable_constructor) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that an undetachable stage can not be detached
    Stage s(false);
    EXPECT_FALSE(s.is_detached());
    EXPECT_FALSE(s.detach());
    EXPECT_FALSE(s.is_detached());
}

TEST(StageTest, detachable_constructor) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a detachable stage can be detached
    Stage s;
    EXPECT_FALSE(s.is_detached());
    EXPECT_TRUE(s.detach());
    EXPECT_TRUE(s.is_detached());
}

TEST(StageTest, previous_stage_constructor) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that previous stage of a stage can be recovered
    Stage s1;
    Stage s2(s1);
    EXPECT_EQ(std::unordered_set<BaseStage *>({&s1}), s2.previous_stages());
}

TEST(StageTest, previous_stage_undetachable_constructor) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that previous stage of an undetachable stage can be recovered and can not be detached
    Stage s1;
    Stage s2(s1, false);
    EXPECT_FALSE(s2.is_detached());
    EXPECT_FALSE(s2.detach());
    EXPECT_FALSE(s2.is_detached());
    EXPECT_EQ(std::unordered_set<BaseStage *>({&s1}), s2.previous_stages());
}

TEST(StageTest, previous_stage_detachable_constructor) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that previous stage of a detachable stage can be recovered and can be detached
    Stage s1;
    Stage s2(s1);
    EXPECT_FALSE(s2.is_detached());
    EXPECT_TRUE(s2.detach());
    EXPECT_TRUE(s2.is_detached());
    EXPECT_EQ(std::unordered_set<BaseStage *>({&s1}), s2.previous_stages());
}

TEST(StageTest, empty_previous_stages) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a default constructed stage has no previous stages
    Stage s;
    EXPECT_EQ(std::unordered_set<BaseStage *>(), s.previous_stages());
}

TEST(StageTest, set_previous_stage) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that calling set_previous_stage sets the correct previous stage
    Stage s1, s2;
    s1.set_previous_stage(s2);
    EXPECT_EQ(std::unordered_set<BaseStage *>({&s2}), s1.previous_stages());
}

TEST(StageTest, add_previous_stages) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that adding multiple previous stages using consuming callbacks sets
    // the correct previous stages
    Stage s1, s2, s3;
    s1.set_consuming_callback(s2, [](const boost::any &) {});
    s1.set_consuming_callback(s3, [](const boost::any &) {});
    EXPECT_EQ(std::unordered_set<BaseStage *>({&s2, &s3}), s1.previous_stages());
}

TEST(StageTest, set_add_previous_stages) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that adding multiple previous stages using set_previous_stage and consuming
    // callbacks sets the correct previous stages
    Stage s1, s2, s3;
    s1.set_previous_stage(s2);
    s1.set_consuming_callback(s3, [](const boost::any &) {});
    EXPECT_EQ(std::unordered_set<BaseStage *>({&s2, &s3}), s1.previous_stages());
}

TEST(StageTest, empty_next_stages) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a default constructed stage has no next stages
    Stage s;
    EXPECT_EQ(std::unordered_set<BaseStage *>(), s.next_stages());
}

TEST(StageTest, set_next_stage) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that calling set_previous_stage sets the correct next stage
    Stage s1, s2;
    s1.set_previous_stage(s2);
    EXPECT_EQ(std::unordered_set<BaseStage *>({&s1}), s2.next_stages());
}

TEST(StageTest, add_next_stages) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that adding multiple previous stages using consuming callbacks sets
    // the correct next stages
    Stage s1, s2, s3;
    s1.set_consuming_callback(s2, [](const boost::any &) {});
    s1.set_consuming_callback(s3, [](const boost::any &) {});
    EXPECT_EQ(std::unordered_set<BaseStage *>({&s1}), s2.next_stages());
    EXPECT_EQ(std::unordered_set<BaseStage *>({&s1}), s3.next_stages());
}

TEST(StageTest, set_add_next_stages) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that adding multiple previous stages using set_previous_stage and consuming
    // callbacks sets the correct next stages
    Stage s1, s2, s3;
    s1.set_previous_stage(s2);
    s1.set_consuming_callback(s3, [](const boost::any &) {});
    EXPECT_EQ(std::unordered_set<BaseStage *>({&s1}), s2.next_stages());
    EXPECT_EQ(std::unordered_set<BaseStage *>({&s1}), s3.next_stages());
}

TEST(StageTest, invalid_pipeline) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a stage not added to a pipeline has a null pointer pipeline
    Stage s;
    EXPECT_THROW(s.pipeline(), std::runtime_error);
    EXPECT_THROW(const_cast<const Stage &>(s).pipeline(), std::runtime_error);
}

TEST(StageTest, valid_pipeline) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a stage not added to a pipeline has a null pointer pipeline
    Pipeline p;
    auto &s = p.add_stage(std::make_unique<Stage>());
    EXPECT_EQ(&p, &s.pipeline());
    EXPECT_EQ(&p, &const_cast<const Stage &>(s).pipeline());
}

TEST(StageTest, status_inactive) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the default status of a stage is inactive
    Stage s;
    EXPECT_EQ(BaseStage::Status::Inactive, s.status());
}

TEST(StageTest, status_completed) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a stage that completes has the completed status
    MockStage s;
    s.complete();
    EXPECT_EQ(BaseStage::Status::Completed, s.status());
}

TEST(StageTest, status_started) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a stage in a started pipeline has the started status
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<Stage>());
    // we need to add at least a consumer, otherwise the pipeline will immediately stop
    auto &s2 = p.add_stage(std::make_unique<Stage>(), s1);
    // start the pipeline
    p.step();
    EXPECT_EQ(BaseStage::Status::Started, s1.status());
}

TEST(StageTest, status_cancelled) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that a stage in a started pipeline has the started status
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<Stage>());
    // we need to add at least a consumer, otherwise the pipeline will immediately stop
    auto &s2 = p.add_stage(std::make_unique<Stage>(), s1);
    // start the pipeline
    p.step();
    p.cancel();
    EXPECT_EQ(BaseStage::Status::Cancelled, s1.status());
}

TEST(StageTest, remove_stage) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that removing a stage does not break the pipeline
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<Stage>());
    auto &s2 = p.add_stage(std::make_unique<Stage>(), s1);
    // remove a stage
    p.remove_stage(s2);
    // start the pipeline
    p.step();
    SUCCEED();
}

TEST(StageTest, receiving_callbacks) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that when a stage changes status, the next stages are notified
    std::atomic<bool> called_any_stage{false}, called_spec_stage{false}, called_no_stage{false};
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<MockStage>());
    auto &s2 = p.add_stage(std::make_unique<MockStage>());
    auto &s3 = p.add_stage(std::make_unique<MockStage>());
    auto &s4 = p.add_stage(std::make_unique<Stage>(), s1);
    auto &s5 = p.add_stage(std::make_unique<Stage>(), s2);
    auto &s6 = p.add_stage(std::make_unique<Stage>(), s3);
    s4.set_receiving_callback(s1, [&called_spec_stage](const BaseStage::NotificationType &t, const boost::any &d) {
        called_spec_stage = true;
    });
    s5.set_receiving_callback([&called_any_stage](const BaseStage &ps, const BaseStage::NotificationType &t,
                                                  const boost::any &d) { called_any_stage = true; });
    s6.set_receiving_callback(
        [&called_no_stage](const BaseStage::NotificationType &t, const boost::any &d) { called_no_stage = true; });
    // start the pipeline
    p.step();
    s1.complete();
    s2.complete();
    s3.complete();
    while (p.step()) {}
    EXPECT_TRUE(called_spec_stage);
    EXPECT_TRUE(called_any_stage);
    EXPECT_TRUE(called_no_stage);
}

TEST(StageTest, receiving_callback_started) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that when a stage changes status, the next stages are notified
    std::atomic<int> called{0};
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<Stage>());
    auto &s2 = p.add_stage(std::make_unique<Stage>(), s1);
    s2.set_receiving_callback(
        [&s1, &s2, &called](const BaseStage &ps, const BaseStage::NotificationType &t, const boost::any &d) {
            switch (called) {
            case 0:
                try {
                    EXPECT_EQ(&s1, &ps);
                    EXPECT_EQ(BaseStage::NotificationType::Status, t);
                    auto status = boost::any_cast<BaseStage::Status>(d);
                    EXPECT_EQ(BaseStage::Status::Started, status);
                } catch (boost::bad_any_cast &) { FAIL() << "Bad cast not supposed to be caught"; }
                called++;
                break;
            }
        });
    // start the pipeline
    p.step();
    while (called < 1) {}
}

TEST(StageTest, receiving_callback_completed) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that when a stage changes status, the next stages are notified
    std::atomic<int> called{0};
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<MockStage>());
    auto &s2 = p.add_stage(std::make_unique<Stage>(), s1);
    s2.set_receiving_callback(
        [&s1, &called](const BaseStage &ps, const BaseStage::NotificationType &t, const boost::any &d) {
            switch (called) {
            case 0:
                called++;
                break;
            case 1:
                try {
                    EXPECT_EQ(&s1, &ps);
                    EXPECT_EQ(BaseStage::NotificationType::Status, t);
                    auto status = boost::any_cast<BaseStage::Status>(d);
                    EXPECT_EQ(BaseStage::Status::Completed, status);
                } catch (boost::bad_any_cast &) { FAIL() << "Bad cast not supposed to be caught"; }
                called++;
                break;
            }
        });
    // start the pipeline
    p.step();
    s1.complete();
    while (called < 2) {}
}

TEST(StageTest, receiving_callback_cancelled) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that when a stage changes status, the next stages are notified
    std::atomic<int> called{0};
    Pipeline p(true);
    auto &s1 = p.add_stage(std::make_unique<Stage>());
    auto &s2 = p.add_stage(std::make_unique<Stage>(), s1);
    s2.set_receiving_callback(
        [&s1, &called](const BaseStage &ps, const BaseStage::NotificationType &t, const boost::any &d) {
            switch (called) {
            case 0:
                called++;
                break;
            case 1:
                try {
                    EXPECT_EQ(&s1, &ps);
                    EXPECT_EQ(BaseStage::NotificationType::Status, t);
                    auto status = boost::any_cast<BaseStage::Status>(d);
                    EXPECT_EQ(BaseStage::Status::Cancelled, status);
                } catch (boost::bad_any_cast &) { FAIL() << "Bad cast not supposed to be caught"; }
                called++;
                break;
            }
        });
    // start the pipeline
    p.step();
    p.cancel();
    while (called < 2) {}
}
