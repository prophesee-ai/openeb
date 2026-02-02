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

#include "metavision/sdk/stream/synced_camera_streams_slicer.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"

namespace Metavision {

enum class StreamingThreadStatus { OFFLINE, STARTED };

bool SyncedSlice::operator==(const SyncedSlice &other) const {
    return status == other.status && t == other.t && n_events == other.n_events &&
           master_events == other.master_events && master_triggers == other.master_triggers &&
           slave_events == other.slave_events;
}

class Source {
public:
    explicit Source(Camera &&camera) : camera_(std::move(camera)) {}

    [[nodiscard]] const Camera &camera() const {
        return camera_;
    }

protected:
    Camera camera_;
    std::mutex mtx_;
    StreamingThreadStatus state_ = StreamingThreadStatus::OFFLINE;
};

class Slave : public Source {
public:
    Slave(Camera &&camera, timestamp max_duration, std::size_t max_size) :
        Source(std::move(camera)), max_duration_(max_duration), max_size_(max_size) {}

    void start() {
        // Setup slave camera
        // While the slave camera is running, the slave thread simply buffers events in a vector for further processing
        // by the master thread, until the buffering size/duration threshold is reached.
        camera_.cd().add_callback([this](const auto &begin, const auto &end) {
            if (begin == end) {
                return;
            }
            std::unique_lock lock(mtx_);
            if (state_ == StreamingThreadStatus::OFFLINE) {
                return;
            }

            auto slave_is_too_fast = [this]() {
                return state_ == StreamingThreadStatus::STARTED && !events_queue_.empty() &&
                       (events_queue_.size() >= max_size_ ||
                        events_queue_.back().t - events_queue_.front().t >= max_duration_);
            };

            if (slave_is_too_fast()) {
                master_can_continue_cond_.notify_one();
                slave_can_continue_cond_.wait(lock, [=]() { return !slave_is_too_fast(); });
            }

            events_queue_.insert(events_queue_.end(), begin, end);
            master_can_continue_cond_.notify_one();
        });

        camera_.add_status_change_callback([this](const CameraStatus &status) {
            std::unique_lock lock(mtx_);
            if (status == CameraStatus::STARTED) {
                state_ = StreamingThreadStatus::STARTED;
            } else if (status == CameraStatus::STOPPED) {
                state_ = StreamingThreadStatus::OFFLINE;

                master_can_continue_cond_.notify_one();
            }
        });

        // Start the camera
        camera_.start();
    }

    void stop() {
        {
            std::unique_lock lock(mtx_);
            state_ = StreamingThreadStatus::OFFLINE;
            slave_can_continue_cond_.notify_one();
        }
        camera_.stop();
    }

    void get_sliced_events(std::vector<EventCD> &events, timestamp ts_upper_bound) {
        std::unique_lock lock(mtx_);

        auto has_slave_events_to_extract = [this, ts_upper_bound]() {
            return !events_queue_.empty() && events_queue_.front().t < ts_upper_bound;
        };
        auto depleted_slave_events_to_extract = [&]() {
            // We exhausted the events to be extracted for the current master slice if:
            // 1. there are currently no events to extract from the slave event buffer, AND
            //   2.a the slave event buffer is not empty (implying, due to condition 1. being true, that the first event
            //       in the slave buffer is to be part of the next slice), OR
            //   2.b the slave stream is offline (implying, due to 2.a being false, that we will not get anymore events
            //       to be included in the current slice).
            return !has_slave_events_to_extract() &&
                   (!events_queue_.empty() || state_ == StreamingThreadStatus::OFFLINE);
        };
        while (!depleted_slave_events_to_extract()) {
            if (has_slave_events_to_extract()) {
                auto it_events_to_copy_end =
                    std::lower_bound(events_queue_.begin(), events_queue_.end(), ts_upper_bound,
                                     [](const EventCD &ev, timestamp ts) { return ev.t < ts; });
                events.insert(events.end(), events_queue_.begin(), it_events_to_copy_end);
                events_queue_.erase(events_queue_.begin(), it_events_to_copy_end);
                slave_can_continue_cond_.notify_one();
            } else {
                master_can_continue_cond_.wait(lock);
            }
        }
    }

    void flush(std::vector<EventCD> &events) {
        if (state_ != StreamingThreadStatus::OFFLINE) {
            throw std::runtime_error("Cannot flush while the slave source is running");
        }

        events.insert(events.end(), events_queue_.begin(), events_queue_.end());
        events_queue_.clear();
    }

private:
    const timestamp max_duration_;
    const std::size_t max_size_;
    std::deque<EventCD> events_queue_;
    std::condition_variable master_can_continue_cond_;
    std::condition_variable slave_can_continue_cond_;
};

class SyncedCameraStreamsSlicer::Master : public Source {
public:
    using QueuePtr = std::shared_ptr<ConcurrentQueue<SyncedSlice>>;
    Master(QueuePtr queue, Camera &&camera, const SliceCondition &slice_condition) :
        Source(std::move(camera)), queue_(std::move(queue)) {
        event_buffer_pool_          = SharedObjectPool<std::vector<EventCD>>::make_unbounded();
        trigger_buffer_pool_        = SharedObjectPool<std::vector<EventExtTrigger>>::make_unbounded();
        curt_event_buffer_master_   = event_buffer_pool_.acquire();
        curt_trigger_buffer_master_ = trigger_buffer_pool_.acquire();
        slicer_.set_slicing_condition(slice_condition);
        slicer_.set_on_new_slice_callback([this](auto status, auto t_slice_master, auto nevents_slice_master) {
            on_new_slice(status, t_slice_master, nevents_slice_master);
        });
    }

    ~Master() {
        for (const auto &slave : slave_sources_) {
            slave->stop();
        }

        queue_->close();
        camera_.stop();
    }

    void start_slicing() {
        // Setup master camera & slicer
        camera_.cd().add_callback([this](const auto &begin, const auto &end) {
            std::unique_lock lock(mtx_);
            slicer_.process_events(begin, end, [this](const auto &slice_begin, const auto &slice_end) {
                curt_event_buffer_master_->insert(curt_event_buffer_master_->end(), slice_begin, slice_end);
            });
        });

        camera_.ext_trigger().add_callback([this](const auto &begin, const auto &end) {
            curt_trigger_buffer_master_->insert(curt_trigger_buffer_master_->end(), begin, end);
        });

        camera_.add_status_change_callback([this](const CameraStatus &status) {
            std::unique_lock lock(mtx_);
            if (status == CameraStatus::STARTED) {
                state_ = StreamingThreadStatus::STARTED;
            } else if (status == CameraStatus::STOPPED) {
                state_ = StreamingThreadStatus::OFFLINE;

                // The master went offline and won't be able to slice anymore, so let's stop the slave sources
                for (const auto &slave_source : slave_sources_) {
                    slave_source->stop();
                }

                // flush the remaining data
                slicer_.flush();

                queue_->close();
            }
        });

        try {
            auto decoder_master = camera_.get_device().get_facility<I_EventsStreamDecoder>();
            decoder_master->add_time_callback([this](timestamp t) { slicer_.notify_elapsed_time(t); });
        } catch (const CameraException &e) { MV_LOG_TRACE() << e.what(); }

        for (const auto &slave_source : slave_sources_) {
            slave_source->start();
        }
        camera_.start();
    }

    void add_slave(std::unique_ptr<Slave> slave_source) {
        slave_sources_.emplace_back(std::move(slave_source));
        curt_event_buffers_slave_.push_back(event_buffer_pool_.acquire());
    }

    size_t slaves_count() const {
        return slave_sources_.size();
    }

    [[nodiscard]] const Source &slave(size_t i) const {
        return *slave_sources_[i];
    }

private:
    void on_new_slice(EventBufferReslicerAlgorithm::ConditionStatus status, timestamp t_slice_master,
                      std::size_t nevents_slice_master) {
        if (state_ == StreamingThreadStatus::OFFLINE) {
            for (size_t i = 0; i < slave_sources_.size(); ++i) {
                slave_sources_[i]->flush(*curt_event_buffers_slave_[i]);
            }
        } else {
            for (size_t i = 0; i < slave_sources_.size(); ++i) {
                slave_sources_[i]->get_sliced_events(*curt_event_buffers_slave_[i], t_slice_master);
            }
        }

        SyncedSlice slice;
        slice.status          = status;
        slice.t               = t_slice_master;
        slice.n_events        = nevents_slice_master;
        slice.master_events   = std::move(curt_event_buffer_master_);
        slice.master_triggers = std::move(curt_trigger_buffer_master_);

        for (auto &slave_buffer : curt_event_buffers_slave_) {
            slice.slave_events.push_back(std::move(slave_buffer));
        }

        queue_->emplace(std::move(slice));

        curt_event_buffer_master_ = event_buffer_pool_.acquire();
        curt_event_buffer_master_->clear();
        curt_trigger_buffer_master_ = trigger_buffer_pool_.acquire();
        curt_trigger_buffer_master_->clear();

        for (auto &slave_buffer : curt_event_buffers_slave_) {
            slave_buffer = event_buffer_pool_.acquire();
            slave_buffer->clear();
        }
    }

    QueuePtr queue_;
    EventBufferReslicerAlgorithm slicer_;
    SharedObjectPool<std::vector<EventCD>> event_buffer_pool_;
    SharedObjectPool<std::vector<EventExtTrigger>> trigger_buffer_pool_;
    std::shared_ptr<EventBuffer> curt_event_buffer_master_;
    std::vector<std::shared_ptr<EventBuffer>> curt_event_buffers_slave_;
    std::shared_ptr<TriggerBuffer> curt_trigger_buffer_master_;
    std::vector<std::unique_ptr<Slave>> slave_sources_;
};

SyncedCameraStreamsSlicer::SyncedCameraStreamsSlicer() = default;

SyncedCameraStreamsSlicer::SyncedCameraStreamsSlicer(SyncedCameraStreamsSlicer &&slicer) noexcept {
    queue_         = std::move(slicer.queue_);
    master_source_ = std::move(slicer.master_source_);
}

SyncedCameraStreamsSlicer &SyncedCameraStreamsSlicer::operator=(SyncedCameraStreamsSlicer &&slicer) noexcept {
    queue_         = std::move(slicer.queue_);
    master_source_ = std::move(slicer.master_source_);
    return *this;
}

SyncedCameraStreamsSlicer::~SyncedCameraStreamsSlicer() = default;

SyncedCameraStreamsSlicer::SyncedCameraStreamsSlicer(Camera &&master_camera, std::vector<Camera> &&slave_cameras,
                                                     const SliceCondition &slice_condition, size_t max_queue_size) :
    queue_(std::make_unique<ConcurrentQueue<SyncedSlice>>(max_queue_size)) {
    if (slave_cameras.empty()) {
        throw std::invalid_argument("At least one slave camera must be provided");
    }

    // Initialize the master source
    master_source_ = std::make_unique<Master>(queue_, std::move(master_camera), slice_condition);

    // Initialize internal variables
    timestamp max_duration = std::numeric_limits<timestamp>::max();
    std::size_t max_size   = std::numeric_limits<std::size_t>::max();
    switch (slice_condition.type) {
    case Detail::ReslicingConditionType::IDENTITY:
        max_duration = 10000;
        break;
    case Detail::ReslicingConditionType::MIXED:
        max_duration = 2 * slice_condition.delta_ts;
        max_size     = 2 * slice_condition.delta_n_events;
        break;
    case Detail::ReslicingConditionType::N_EVENTS:
        max_size = 2 * slice_condition.delta_n_events;
        break;
    case Detail::ReslicingConditionType::N_US:
        max_duration = 2 * slice_condition.delta_ts;
        break;
    }

    // Initialize the slave sources
    for (auto &slave_camera : slave_cameras) {
        master_source_->add_slave(std::make_unique<Slave>(std::move(slave_camera), max_duration, max_size));
    }
}

SyncedCameraStreamsSlicer::SliceIterator SyncedCameraStreamsSlicer::begin() {
    master_source_->start_slicing();
    return SliceIterator(queue_);
}

SyncedCameraStreamsSlicer::SliceIterator SyncedCameraStreamsSlicer::end() {
    auto end = SliceIterator();
    return end;
}

const Camera &SyncedCameraStreamsSlicer::master() const {
    return master_source_->camera();
}

size_t SyncedCameraStreamsSlicer::slaves_count() const {
    return master_source_->slaves_count();
}

const Camera &SyncedCameraStreamsSlicer::slave(size_t i) const {
    return master_source_->slave(i).camera();
}

} // namespace Metavision