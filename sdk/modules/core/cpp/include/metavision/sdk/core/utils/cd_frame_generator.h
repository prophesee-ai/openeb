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

#ifndef METAVISION_SDK_CORE_CD_FRAME_GENERATOR_H
#define METAVISION_SDK_CORE_CD_FRAME_GENERATOR_H

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h"
#include "metavision/sdk/core/utils/threaded_process.h"

namespace Metavision {

/// @brief Utility class to display CD events that handles multithreading and is built on the top
/// of PeriodicFrameGenerator
class CDFrameGenerator {
public:
    /// @brief Default constructor
    /// @param width Width of the image (in pixels)
    /// @param height Height of the image (in pixels)
    /// @param process_all_frames If true, it will process all frames, not just the latest one. Note that if true
    /// it can slow down the process.
    CDFrameGenerator(long width, long height, bool process_all_frames = false);

    /// @brief Destructor
    ~CDFrameGenerator();

    /// @brief Sets the color used to generate the frame
    ///
    /// By default, the frame generated will be 8 bits single channel with the default colors grey, white, black as
    /// background, on and off colors respectively.
    /// If the parameter @a colored is false, then the generated frame will be 8 bits single channel and the first
    /// channel will be used to define each colors (i.e. the other channels will be ignored).
    /// If the parameter @a colored is true, then the generated frame will be 8 bits three channels.
    ///
    /// @param background_color Color used as background, when no events were received for
    ///                         a pixel, default: grey cv::Scalar::all(128)
    /// @param on_color Color used for on events, default: white cv::Scalar::all(255)
    /// @param off_color Color used for off events, default: black cv:Scalar::all(0)
    /// @param colored If the generated frame should be single or three channels
    void set_colors(const cv::Scalar &background_color, const cv::Scalar &on_color, const cv::Scalar &off_color,
                    bool colored = false);

    /// @brief Sets the color used to generate the frame
    /// @param palette The Prophesee's color palette to use
    void set_color_palette(const Metavision::ColorPalette &palette);

    /// @brief Adds the buffer of events to be displayed
    /// @param begin Beginning of the buffer of events
    /// @param end End of the buffer of events
    void add_events(const Metavision::EventCD *begin, const Metavision::EventCD *end);

    /// @brief Sets the time interval to display events
    ///
    /// The events shown at each refresh are such that their timestamps are in the last 'display_accumulation_time_us'
    /// microseconds from the last received event timestamp.
    ///
    /// @param display_accumulation_time_us The time interval to display events from up to now (in us).
    void set_display_accumulation_time_us(timestamp display_accumulation_time_us);

    /// @brief Starts the generator thread
    ///
    /// @param fps Frame rate
    /// @param cb Function to call every time a new frame is available. It takes in input the time (in us) of the new
    /// frame and the frame. The frame passed as a parameter is guaranteed to be available and left untouched until the
    /// next time the callback is called. This means that you don't need to make a deep copy of it, if you only intend
    /// to use the frame until the next one is made available by the callback.
    /// @return true if the thread started successfully, false otherwise. Also returns false, if the thread is already
    /// started.
    bool start(std::uint16_t fps, const PeriodicFrameGenerationAlgorithm::OutputCb &cb);

    /// @brief Stops the generator thread
    ///
    /// @return true if the thread has been stopped successfully, false otherwise. Return false if the thread had not
    /// been previously started
    bool stop();

    /// @brief Resets the frame generator state
    void reset();

private:
    bool generate();

    // Image to display
    PeriodicFrameGenerationAlgorithm::OutputCb frame_cb_;

    struct FrameEvent {
        cv::Mat frame_;
        timestamp ts_us_;
    };
    std::vector<FrameEvent> frames_;
    std::vector<EventCD> events_front_, events_back_;
    size_t frames_count_{0};

    // Is frame dropping allowed?
    bool process_all_frames_ = false;
    bool events_available_   = false;
    timestamp notify_slice_us_{0};
    timestamp next_notify_us_;

    // Events to display
    std::mutex processing_mutex_;
    std::atomic<bool> stop_{true};
    std::condition_variable events_available_cond_;

    std::unique_ptr<PeriodicFrameGenerationAlgorithm> frame_generation_algo_;
    // Shadow params
    timestamp accumulation_time_us_;
    cv::Scalar background_color_, on_color_, off_color_;
    bool colored_;

    ThreadedProcess processing_thread_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_CD_FRAME_GENERATOR_H
