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

#ifndef METAVISION_SDK_CORE_DETAIL_FRAME_COMPOSITION_STAGE_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_FRAME_COMPOSITION_STAGE_IMPL_H

namespace Metavision {

inline bool FrameCompositionStage::SourceInfo::can_copy(timestamp next_frame_ts, timestamp max_ts_range) {
    return ts_last_frame_saved - next_frame_ts < max_ts_range;
}

inline void FrameCompositionStage::SourceInfo::add_frame(timestamp ts, FrameCompositionStage::FramePtr &ptr,
                                                         timestamp next_frame_ts, timestamp max_ts_range) {
    if (can_copy(next_frame_ts, max_ts_range)) {
        saved_frames_queue_.push({ts, ptr->clone()});
    } else {
        references_to_input_frames_queue_.push({ts, ptr});
    }
}

inline void FrameCompositionStage::SourceInfo::update_with_past_ptr(FrameCompositionStage::FramePtr &ptr) {
    if (ptr) {
        // In this case just update the reference of last frame received:
        // this is because we don't know yet if this frame is the one that will be used to update the
        // composed image or another one more recent (it all depends on the update frequency of this source
        // compared to the update frequency of the composer)
        // REMARK: keeping the reference to the frame will in fact block this pointer of the input pool
        // (that is ok if the input pool has a size >= 2, which we can safely assume)
        ptr_to_last_frame_received = ptr;
    }
}

bool FrameCompositionStage::SourceInfo::update_with_current_ptr(FrameCompositionStage::FramePtr &ptr,
                                                                timestamp source_next_frame_ts,
                                                                timestamp composer_next_frame_ts,
                                                                timestamp max_ts_range) {
    // Update info:
    ptr_to_last_frame_received = nullptr; // release shared pointer of the source frame pool (if any)
    ts_last_frame_saved        = source_next_frame_ts;

    if (ptr) {
        if (source_next_frame_ts == composer_next_frame_ts) {
            // Instead of saving the image (or storing the ptr) in the queue, we can directly update the
            // composed image, thus avoiding the copy (or avoiding to block the ptr from the input pool)
            return true;
        } else {
            // We have to copy/store this frame, it will be used to generate one of the next composed frames
            add_frame(source_next_frame_ts, ptr, composer_next_frame_ts, max_ts_range);
        }
    } // No need to handle the else condition: if ptr is a nullptr, then just by updating variable ts_last_frame_saved
      // (done above) then the composed frame will automatically keep last sub-image of this source (cf code of method
      // update_composed_image() in class FrameCompositionStage)
    return false;
}

void FrameCompositionStage::SourceInfo::update_with_future_ptr(FrameCompositionStage::FramePtr &ptr, timestamp ptr_ts,
                                                               timestamp source_next_frame_ts, timestamp frame_period,
                                                               timestamp composer_next_frame_ts,
                                                               timestamp max_ts_range) {
    // In this case the last frame received is the one we need to copy/store to update frame at time
    // source_next_frame_ts
    if (ptr_to_last_frame_received) {
        add_frame(source_next_frame_ts, ptr_to_last_frame_received, composer_next_frame_ts, max_ts_range);
    }

    // We're not going to have any frame between source_next_frame_ts and ptr_ts, so we can keep increasing
    // source_next_frame_ts until it becomes greater than or equal to ptr_ts
    while (ptr_ts > source_next_frame_ts) {
        source_next_frame_ts += frame_period;
    }

    // When arriving here there are two possibilities:
    //   1) ptr_ts == source_next_frame_ts: in this case we can copy the image / store the ptr
    //      in the queue
    //   2) ptr_ts < source_next_frame_ts: in this case just store the reference
    if (ptr_ts == source_next_frame_ts) {
        // No need to check the return value of the following call, because we already know that it will
        // return false (because we pass -1 as third parameter). We do indeed an extra check that we don't need
        // (check if source_next_frame_ts == -1 - cf code of method update_with_current_ptr()), but that's the cost
        // to have a common method to make the code more readable
        update_with_current_ptr(ptr, source_next_frame_ts, -1, max_ts_range);
    } else {
        ptr_to_last_frame_received = ptr;
        // Update ts_last_frame_saved to (source_next_frame_ts - frame_period) and not source_next_frame_ts because in
        // this case we still haven't found the frame to display for t = source_next_frame_ts
        ts_last_frame_saved = source_next_frame_ts - frame_period;
    }
}

FrameCompositionStage::FrameCompositionStage(int fps, timestamp max_ts_range, const cv::Vec3b &bg_color) :
    frame_period_(static_cast<timestamp>(1.e6 / fps + 0.5)),
    next_frame_ts_(static_cast<timestamp>(1.e6 / fps + 0.5)),
    max_ts_range_(max_ts_range),
    frame_composer_(bg_color),
    output_frame_pool_(FramePool::make_bounded()) {
    set_consuming_callback([](const boost::any &data) {
        MV_SDK_LOG_WARNING() << "For this stage to work properly, you need to call add_previous_frame_stage\n"
                             << "for each stage that produces frame to be composed.";
    });
    set_receiving_callback([this](BaseStage &stage, const BaseStage::NotificationType &type, const boost::any &data) {
        if (type == BaseStage::NotificationType::Status) {
            set_stage_status(stage, data);
        }
    });
}

void FrameCompositionStage::add_previous_frame_stage(BaseStage &prev_frame_stage, int x, int y, int width, int height,
                                                     bool enable_crop,
                                                     const FrameComposer::GrayToColorOptions &gray_to_color_options) {
    FrameComposer::ResizingOptions resize_o(width, height, enable_crop);
    const int ref                  = frame_composer_.add_new_subimage_parameters(x, y, resize_o, gray_to_color_options);
    stages_map_[&prev_frame_stage] = ref;
    sources_info_map_[ref]         = SourceInfo();
    set_consuming_callback(prev_frame_stage, [this, ref](const boost::any &data) { consume_frame(ref, data); });
}

FrameComposer &FrameCompositionStage::frame_composer() {
    return frame_composer_;
}

void FrameCompositionStage::consume_frame(int frame_ref, const boost::any &data) {
    try {
        auto ts_frame_ptr = boost::any_cast<Input>(data);

        SourceInfo &source_info        = sources_info_map_[frame_ref];
        timestamp source_next_frame_ts = source_info.ts_last_frame_saved + frame_period_;

        if (ts_frame_ptr.first < source_next_frame_ts) {
            source_info.update_with_past_ptr(ts_frame_ptr.second);
            // We can return directly, because in this case we already know that there is no new composed
            // image to generate
            return;
        } else if (ts_frame_ptr.first == source_next_frame_ts) {
            if (source_info.update_with_current_ptr(ts_frame_ptr.second, source_next_frame_ts, next_frame_ts_,
                                                    max_ts_range_)) {
                frame_composer_.update_subimage(frame_ref, *(ts_frame_ptr.second));
            }
        } else { // .i.e. ts_frame_ptr.first > source_next_frame_ts
            source_info.update_with_future_ptr(ts_frame_ptr.second, ts_frame_ptr.first, source_next_frame_ts,
                                               frame_period_, next_frame_ts_, max_ts_range_);
        }

        // Update and produce composed frames as long as we have all updates
        while (got_all_updates()) {
            update_composed_image(); // No need to check return value of this method because we already know that
                                     // have all updates
            produce_composed_frame();
        }
    } catch (boost::bad_any_cast &c) { MV_SDK_LOG_ERROR() << c.what(); }
}

void FrameCompositionStage::set_stage_status(BaseStage &stage, const boost::any &data) {
    try {
        auto status = boost::any_cast<const Status &>(data);
        if (status == Status::Completed) {
            SourceInfo &source_info        = sources_info_map_[stages_map_[&stage]];
            source_info.finished_producing = true;

            // In this case, if we had a pointer stored in source_info.ptr_to_last_frame_received,
            // we update the last frame saved:
            if (source_info.ptr_to_last_frame_received) {
                source_info.add_frame(source_info.ts_last_frame_saved + frame_period_,
                                      source_info.ptr_to_last_frame_received, next_frame_ts_, max_ts_range_);
            }

            // If all sources have finished producing, then we can update all the remaining
            // composed images:
            for (auto &p : sources_info_map_) {
                if (!p.second.finished_producing) {
                    return;
                }
            }

            // If we arrive here it means all sources have finished producing: we can keep updating and produce the
            // composed image until all input frames received have been consumed
            while (update_composed_image()) {
                produce_composed_frame();
            }
        }
    } catch (boost::bad_any_cast &c) { MV_SDK_LOG_ERROR() << c.what(); }
}

inline bool FrameCompositionStage::update_composed_image() {
    bool still_have_frames_to_consume = false;
    for (auto &p : sources_info_map_) {
        if (!p.second.saved_frames_queue_.empty()) {
            still_have_frames_to_consume = true; // Regardless of whether or not we enter the if condition below, if
                                                 // we arrive here it means the queue is not empty, i.e. we didn't
                                                 // consume all the frames of the source
            auto &next_frame_saved_in_queue = p.second.saved_frames_queue_.front();
            if (next_frame_saved_in_queue.first == next_frame_ts_) {
                frame_composer_.update_subimage(p.first, next_frame_saved_in_queue.second);
                p.second.saved_frames_queue_.pop();
                continue; // No need to look in the queue with the shared ptr: skip to next source
            }
        }
        if (!p.second.references_to_input_frames_queue_.empty()) {
            still_have_frames_to_consume = true; // Regardless of whether or not we enter the if condition below, if
                                                 // we arrive here it means the queue is not empty, i.e. we didn't
                                                 // consume all the frames of the source
            auto &next_in_queue = p.second.references_to_input_frames_queue_.front();
            if (next_in_queue.first == next_frame_ts_) {
                frame_composer_.update_subimage(p.first, *(next_in_queue.second));
                p.second.references_to_input_frames_queue_.pop();
            }
        }
    }
    return still_have_frames_to_consume;
}

inline void FrameCompositionStage::produce_composed_frame() {
    produced_frame_ptr_ = output_frame_pool_.acquire();
    frame_composer_.get_full_image().copyTo(*produced_frame_ptr_);
    produce(std::make_pair(next_frame_ts_, produced_frame_ptr_));
    next_frame_ts_ += frame_period_;
}

inline bool FrameCompositionStage::got_all_updates() {
    for (auto &p : sources_info_map_) {
        if (p.second.ts_last_frame_saved < next_frame_ts_ && !p.second.finished_producing) {
            return false;
        }
    }
    return true;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_FRAME_COMPOSITION_STAGE_IMPL_H
