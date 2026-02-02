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

// this function will be associated to the camera callback
void count_events(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
    int counter = 0;

    // this loop allows us to get access to each event received in this callback
    for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
        ++counter; // count each event

        // print each event
        std::cout << "Event received: coordinates (" << ev->x << ", " << ev->y << "), t: " << ev->t
                  << ", polarity: " << ev->p << std::endl;
    }

    // report
    std::cout << "There were " << counter << " events in this callback" << std::endl;
}

// main loop
int main(int argc, char *argv[]) {
    Metavision::Camera cam; // create the camera

    if (argc >= 2) {
        // if we passed a file path, open it
        cam = Metavision::Camera::from_file(argv[1]);
    } else {
        // open the first available camera
        cam = Metavision::Camera::from_first_available();
    }

    // to analyze the events, we add a callback that will be called periodically to give access to the latest events
    cam.cd().add_callback(count_events);

    // start the camera
    cam.start();

    // keep running while the camera is on or the recording is not finished
    while (cam.is_running()) {}

    // the recording is finished, stop the camera.
    // Note: we will never get here with a live camera
    cam.stop();
}