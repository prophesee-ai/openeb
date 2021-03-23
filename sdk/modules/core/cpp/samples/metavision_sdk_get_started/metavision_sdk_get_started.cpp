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

// This code sample demonstrate how to use the Metavision SDK. The goal of this sample is to create a simple event
// counter and displayer by introducing some basic concepts of the Metavision SDK.

#include <opencv2/highgui/highgui.hpp>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>

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

    // add the event callback. This callback will be called periodically to provide access to the most recent events
    cam.cd().add_callback([&event_analyzer](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
        event_analyzer.analyze_events(ev_begin, ev_end);
    });

    // get camera resolution
    int camera_width  = cam.geometry().width();
    int camera_height = cam.geometry().height();

    // create a frame generator for visualization
    // this will get the events from the callback and accumulate them in a cv::Mat
    Metavision::CDFrameGenerator cd_frame_generator(camera_width, camera_height);

    // this callback tells the camera to pass the events to the frame generator, who will then create the frame
    cam.cd().add_callback(
        [&cd_frame_generator](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
            cd_frame_generator.add_events(ev_begin, ev_end);
        });

    const int fps       = 25; // event-based cameras do not have a frame rate, but we need one for visualization
    const int wait_time = static_cast<int>(std::round(1.f / fps * 1000)); // how much we should wait between two frames
    cv::Mat cd_frame;                                                     // frame where events will be accumulated
    const std::string window_name = "Metavision SDK Get Started";

    // this function is used to tell the frame generator what to do with the frame and how often to generate it
    cd_frame_generator.start(
        fps, [&cd_frame](const Metavision::timestamp &ts, const cv::Mat &frame) { frame.copyTo(cd_frame); });

    cv::namedWindow(window_name, cv::WINDOW_GUI_EXPANDED);
    cv::resizeWindow(window_name, camera_width, camera_height);

    // start the camera
    cam.start();

    // keep running while the camera is on or the video is not finished
    while (cam.is_running()) {
        // display the frame if it's not empty
        if (!cd_frame.empty()) {
            cv::imshow(window_name, cd_frame);
        }

        // if the user presses the `q` key, quit the loop
        if ((cv::waitKey(wait_time) & 0xff) == 'q') {
            break;
        }
    }

    // the video is finished or the user wants to quit, stop the camera.
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
