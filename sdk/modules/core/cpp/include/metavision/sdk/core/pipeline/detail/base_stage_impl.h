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

#ifndef METAVISION_SDK_CORE_DETAIL_BASE_STAGE_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_BASE_STAGE_IMPL_H

#include <stdexcept>

#include "metavision/sdk/core/pipeline/base_stage.h"
#include "metavision/sdk/core/pipeline/pipeline.h"

namespace Metavision {

BaseStage::BaseStage(bool detachable) : detachable_(detachable) {
    consuming_cbs_[nullptr] = [](BaseStage &, const boost::any &) {};
    receiving_cbs_[nullptr] = [](BaseStage &, const NotificationType &, const boost::any &) {};
}

BaseStage::BaseStage(BaseStage &prev_stage, bool detachable) : BaseStage(detachable) {
    set_previous_stage(prev_stage);
}

BaseStage::~BaseStage() {}

void BaseStage::set_previous_stage(BaseStage &prev_stage) {
    std::lock_guard<std::mutex> lock(mutex_);
    prev_stage.next_stages_.insert(this);
    prev_stages_.insert(&prev_stage);
}

const std::unordered_set<BaseStage *> &BaseStage::previous_stages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return prev_stages_;
}

const std::unordered_set<BaseStage *> &BaseStage::next_stages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_stages_;
}

BaseStage::Status BaseStage::status() const {
    return status_;
}

bool BaseStage::set_status(const Status &status) {
    if (status_.exchange(status) != status) {
        notify(NotificationType::Status, status);
        return true;
    }
    return false;
}

void BaseStage::complete() {
    set_status(Status::Completed);
    // we only set done_ = true only after notification tasks are queued
    // to make sure they are processed
    done_ = true;
    // then we make sure next stages wake up and notice this task is done
    signal();
}

void BaseStage::cancel() {
    set_status(Status::Cancelled);
    // we only set done_ = true only after notification tasks are queued
    // to make sure they are processed
    done_ = true;
    // then we make sure next stages wake up and notice this task is done
    signal();
}

bool BaseStage::is_done() const {
    return done_;
}

void BaseStage::start() {
    if (set_status(Status::Started)) {
        std::function<void()> cb;
        {
            std::lock_guard<std::mutex> lock(cbs_mutex_);
            cb = starting_cb_;
        }
        cb();
    }
}

void BaseStage::stop() {
    // no need to set any status, it will have been done independently via
    // set_status(Status::Completed) or set_status(Status::Cancelled)
    std::function<void()> cb;
    {
        std::lock_guard<std::mutex> lock(cbs_mutex_);
        cb = stopping_cb_;
    }
    cb();
}

void BaseStage::set_receiving_callback(
    const std::function<void(BaseStage &, const NotificationType &, const boost::any &)> &cb) {
    std::lock_guard<std::mutex> lock(cbs_mutex_);
    receiving_cbs_[nullptr] = cb;
}

void BaseStage::set_receiving_callback(const std::function<void(const NotificationType &, const boost::any &)> &cb) {
    std::lock_guard<std::mutex> lock(cbs_mutex_);
    receiving_cbs_[nullptr] = [cb](BaseStage &, const NotificationType &type, const boost::any &data) {
        cb(type, data);
    };
}

void BaseStage::set_receiving_callback(BaseStage &prev_stage,
                                       const std::function<void(const NotificationType &, const boost::any &)> &cb) {
    std::lock_guard<std::mutex> lock(cbs_mutex_);
    receiving_cbs_[&prev_stage] = [cb](BaseStage &, const NotificationType &type, const boost::any &data) {
        cb(type, data);
    };
}

void BaseStage::set_starting_callback(const std::function<void()> &cb) {
    std::lock_guard<std::mutex> lock(cbs_mutex_);
    starting_cb_ = cb;
}

void BaseStage::set_stopping_callback(const std::function<void()> &cb) {
    std::lock_guard<std::mutex> lock(cbs_mutex_);
    stopping_cb_ = cb;
}

void BaseStage::set_setup_callback(const std::function<void()> &cb) {
    std::lock_guard<std::mutex> lock(cbs_mutex_);
    setup_cb_ = cb;
}

void BaseStage::set_consuming_callback(const std::function<void(BaseStage &, const boost::any &)> &cb) {
    std::lock_guard<std::mutex> lock(cbs_mutex_);
    consuming_cbs_[nullptr] = cb;
}

void BaseStage::set_consuming_callback(const std::function<void(const boost::any &)> &cb) {
    std::lock_guard<std::mutex> lock(cbs_mutex_);
    consuming_cbs_[nullptr] = [cb](BaseStage &, const boost::any &data) { cb(data); };
}

void BaseStage::set_consuming_callback(BaseStage &prev_stage, const std::function<void(boost::any)> &cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::lock_guard<std::mutex> lock2(cbs_mutex_);
    consuming_cbs_[&prev_stage] = [cb](BaseStage &, const boost::any &data) { cb(data); };
    prev_stage.next_stages_.insert(this);
    prev_stages_.insert(&prev_stage);
}

void BaseStage::produce(const boost::any &data) {
    Pipeline *pipeline;
    std::unordered_set<BaseStage *> next_stages;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pipeline    = pipeline_;
        next_stages = next_stages_;
    }
    if (!pipeline)
        return;

    // If the pipeline has been cancelled, we can't produce anything
    if (pipeline->status() == Pipeline::Status::Cancelled)
        return;

    for (auto *stage : next_stages) {
        pipeline->schedule(
            *stage, [this, stage, data] { stage->consume(*this, data); }, stage->current_prod_id_++, true,
            stage->run_on_main_thread_);
    };
}

void BaseStage::consume(BaseStage &prev_stage, const boost::any &data) {
    std::function<void(BaseStage &, const boost::any &)> cb;
    {
        std::lock_guard<std::mutex> lock(cbs_mutex_);
        auto it = consuming_cbs_.find(&prev_stage);
        if (it != consuming_cbs_.end()) {
            cb = it->second;
        } else {
            cb = consuming_cbs_[nullptr];
        }
    }
    cb(prev_stage, data);
}

void BaseStage::signal() {
    Pipeline *pipeline;
    std::unordered_set<BaseStage *> next_stages;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pipeline    = pipeline_;
        next_stages = next_stages_;
    }
    if (!pipeline)
        return;

    // send dummy notification task
    for (auto *stage : next_stages) {
        pipeline->schedule(
            *stage, [] {}, stage->current_prod_id_++, true, stage->run_on_main_thread_);
    };
}

void BaseStage::notify(const NotificationType &type, const boost::any &data) {
    Pipeline *pipeline;
    std::unordered_set<BaseStage *> next_stages;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pipeline    = pipeline_;
        next_stages = next_stages_;
    }
    if (!pipeline)
        return;

    for (auto *stage : next_stages) {
        pipeline->schedule(
            *stage, [this, stage, type, data] { stage->receive(*this, type, data); }, stage->current_prod_id_++, false,
            stage->run_on_main_thread_);
    };
}

void BaseStage::receive(BaseStage &prev_stage, const NotificationType &type, const boost::any &data) {
    std::function<void(BaseStage &, const NotificationType, const boost::any &)> cb;
    {
        std::lock_guard<std::mutex> lock(cbs_mutex_);
        auto it = receiving_cbs_.find(&prev_stage);
        if (it != receiving_cbs_.end()) {
            cb = it->second;
        } else {
            cb = receiving_cbs_[nullptr];
        }
    }
    cb(prev_stage, type, data);
}

bool BaseStage::detach() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pipeline_ || pipeline_->status() != Pipeline::Status::Started) {
        if (detachable_) {
            run_on_main_thread_ = false;
            return true;
        }
    }
    return false;
}

bool BaseStage::is_detached() const {
    return !run_on_main_thread_;
}

void BaseStage::set_pipeline(Pipeline &pipeline) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pipeline_ && pipeline_->status() == Pipeline::Status::Started) {
            throw std::runtime_error("BaseStage : associated pipeline cannot be changed once started");
        }
        if (pipeline.status() == Pipeline::Status::Started) {
            throw std::runtime_error("BaseStage : pipeline already started");
        }
        pipeline_ = &pipeline;
    }

    std::function<void()> cb;
    {
        std::lock_guard<std::mutex> lock(cbs_mutex_);
        cb = setup_cb_;
    }
    cb();
}

Pipeline &BaseStage::pipeline() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pipeline_)
        throw std::runtime_error("BaseStage : no pipeline associated");
    return *pipeline_;
}

const Pipeline &BaseStage::pipeline() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pipeline_)
        throw std::runtime_error("BaseStage : no pipeline associated");
    return *pipeline_;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_BASE_STAGE_IMPL_H
