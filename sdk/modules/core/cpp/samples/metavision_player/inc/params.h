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

#ifndef METAVISION_PLAYER_PARAMS_H
#define METAVISION_PLAYER_PARAMS_H

/// Structure to hold command line interface parameters.
struct Parameters {
    // Input.
    std::string in_bias_file;
    std::string in_raw_file;

    // Parameters for the exported files.
    std::string out_bias_file;
    std::string out_png_file;
    std::string out_avi_file;
    std::string out_raw_basename;
    int out_avi_fps = 25;

    // Buffer size in mega events.
    int buffer_size_mev = 100;

    // Show bias sliders
    bool show_biases = false;
};

#endif // #define METAVISION_PLAYER_PARAMS_H
