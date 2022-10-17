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

// This code sample demonstrate how to use the Metavision C++ SDK. The goal of this sample is to create a simple event
// counter and displayer by introducing some basic concepts of the Metavision SDK.

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/algorithms/flip_x_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

// this class will be used to analyze the events
class EventAnalyzer {
public:
    // class variables to store global information
    int global_counter                 = 0; // this will track how many events we processed
    Metavision::timestamp global_max_t = 0; // this will track the highest timestamp we processed

    // this function will be associated to the camera callback
    // it is used to compute statistics on the received events
    void analyze_events(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::cout << "----- New callback! -----" << std::endl;

        // time analysis
        // Note: events are ordered by timestamp in the callback, so the first event will have the lowest timestamp and
        // the last event will have the highest timestamp
        Metavision::timestamp min_t = begin->t;     // get the timestamp of the first event of this callback
        Metavision::timestamp max_t = (end - 1)->t; // get the timestamp of the last event of this callback
        global_max_t = max_t; // events are ordered by timestamp, so the current last event has the highest timestamp

        // counting analysis
        int counter = 0;
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            ++counter; // increasing local counter
        }
        global_counter += counter; // increase global counter

        // report
        std::cout << "There were " << counter << " events in this callback" << std::endl;
        std::cout << "There were " << global_counter << " total events up to now." << std::endl;
        std::cout << "The current callback included events from " << min_t << " up to " << max_t << " microseconds."
                  << std::endl;

        std::cout << "----- End of the callback! -----" << std::endl;
    }
};

// main loop
int main(int argc, char *argv[]) {
    Metavision::Camera cam;       // create the camera
    EventAnalyzer event_analyzer; // create the event analyzer

    if (argc >= 2) {
        // if we passed a file path, open it
        cam = Metavision::Camera::from_file(argv[1]);
    } else {
        // open the first available camera
        cam = Metavision::Camera::from_first_available();
    }

    // to analyze the events, we add a callback that will be called periodically to give access to the latest events
    cam.cd().add_callback([&event_analyzer](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
        event_analyzer.analyze_events(ev_begin, ev_end);
    });

    // to visualize the events, we will need to build frames and render them.
    // building frame will be done with a frame generator that will accumulate the events over time.
    // we need to provide it the camera resolution that we can retrieve from the camera instance
    int camera_width  = cam.geometry().width();
    int camera_height = cam.geometry().height();

    // we also need to choose an accumulation time and a frame rate (here of 20ms and 50 fps)
    const std::uint32_t acc = 20000;
    double fps              = 50;

    // now we can create our frame generator using previous variables
    auto frame_gen = Metavision::PeriodicFrameGenerationAlgorithm(camera_width, camera_height, acc, fps);

    // we create an algorithm that mirrors the X axis of the event stream
    auto algo = Metavision::FlipXAlgorithm(camera_width - 1);

    // we add the callback that will pass the events to the algo and then the frame generator
    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        // we use a vector of CD events to store the output of the algo
        std::vector<Metavision::EventCD> output;
        algo.process_events(begin, end, std::back_inserter(output));
        // we call the frame generator on the processed events
        frame_gen.process_events(output.begin(), output.end());
    });

    // to render the frames, we create a window using the Window class of the UI module
    Metavision::Window window("Metavision SDK Get Started", camera_width, camera_height,
                              Metavision::BaseWindow::RenderMode::BGR);

    // we set a callback on the windows to close it when the Escape or Q key is pressed
    window.set_keyboard_callback(
        [&window](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
            if (action == Metavision::UIAction::RELEASE &&
                (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                window.set_close_flag();
            }
        });

    // we set a callback on the frame generator so that it calls the window object to display the generated frames
    frame_gen.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) { window.show(frame); });

    // start the camera
    cam.start();

    // keep running until the camera is off, the recording is finished or the escape key was pressed
    while (cam.is_running() && !window.should_close()) {
        // we poll events (keyboard, mouse etc.) from the system with a 20ms sleep to avoid using 100% of a CPU's core
        // and we push them into the window where the callback on the escape key will ask the windows to close
        static constexpr std::int64_t kSleepPeriodMs = 20;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
    }

    // the recording is finished or the user wants to quit, stop the camera.
    cam.stop();

    // print the global statistics
    const double length_in_seconds = event_analyzer.global_max_t / 1000000.0;
    std::cout << "There were " << event_analyzer.global_counter << " events in total." << std::endl;
    std::cout << "The total duration was " << length_in_seconds << " seconds." << std::endl;
    if (length_in_seconds >= 1) { // no need to print this statistics if the video was too short
        std::cout << "There were " << event_analyzer.global_counter / length_in_seconds
                  << " events per second on average." << std::endl;
    }
}