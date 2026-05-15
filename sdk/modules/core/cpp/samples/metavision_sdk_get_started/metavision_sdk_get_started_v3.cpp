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
// NOTE: this file is just here to ease the integration with the docs. The main sample is metavision_sdk_get_started.cpp
// NOTE: if you modify this file, please check that the docs references are correct (line numbers)

#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/base/events/event_cd.h>

// this class will be used to analyze the events
class EventAnalyzer {
public:
    // class variables to store global information
    int callback_counter               = 0; // this will track the number of callbacks
    int global_counter                 = 0; // this will track how many events we processed
    Metavision::timestamp global_max_t = 0; // this will track the highest timestamp we processed

    // this function will be associated to the camera callback
    // it is used to compute statistics on the received events
    void analyze_events(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
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

        // Uncomment next line to display the buffer report in the terminal
        // WARNING : logging in the terminal can drastically decrease the performances of your application, especially
        // on embedded platforms with low computational power
//        std::cout << "Cb n°" << callback_counter << ": " << counter << " events from t=" << min_t << " to t="
//                  << max_t << " us." << std::endl;

        // increment callbacks counter
        callback_counter++;
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

    // start the camera
    cam.start();

    // keep running while the camera is on or the recording is not finished
    while (cam.is_running()) {}

    // the recording is finished, stop the camera.
    // Note: we will never get here with a live camera
    cam.stop();

    // print the global statistics
    double length_in_seconds = event_analyzer.global_max_t / 1000000.0;
    std::cout << "There were " << event_analyzer.global_counter << " events in total." << std::endl;
    std::cout << "The total duration was " << length_in_seconds << " seconds." << std::endl;
    if (length_in_seconds >= 1) { // no need to print this statistics if the total duration was too short
        std::cout << "There were " << event_analyzer.global_counter / (event_analyzer.global_max_t / 1000000.0)
                  << " events per seconds on average." << std::endl;
    }
}
