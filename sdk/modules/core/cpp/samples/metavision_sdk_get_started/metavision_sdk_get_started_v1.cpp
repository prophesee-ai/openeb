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

    // start the camera
    cam.start();

    // keep running while the camera is on or the recording is not finished
    while (cam.is_running()) {
        std::cout << "Camera is running!" << std::endl;
    }

    // the recording is finished, stop the camera.
    // Note: we will never get here with a live camera
    cam.stop();
}