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

#ifndef METAVISION_HAL_BASE_EVENT_TYPES_H
#define METAVISION_HAL_BASE_EVENT_TYPES_H

#include <cstdint>

namespace Metavision {

typedef uint8_t EventTypesUnderlying_t;

// WARNING : in order to be able to manually insert events for every
// camera (Gen1, Gen2, Gen3, ...) those two types HAVE to be the
// same for all cameras (it is the case for now, but with every new
// sensor we have to check)
enum class BaseEventTypes : EventTypesUnderlying_t {
    LEFT_TD_LOW     = 0x00, // Left camera TD event, decrease in illumination (polarity '0')
    LEFT_TD_HIGH    = 0x01, // Left camera TD event, increase in illumination (polarity '1')
    LEFT_APS_END    = 0x02, // Left camera APS event, measurement end (polarity '0')
    LEFT_APS_START  = 0x03, // Left camera APS event, measurement start (polarity '1')
    RIGHT_TD_LOW    = 0x04, // Right camera TD event, decrease in illumination (polarity '0')
    RIGHT_TD_HIGH   = 0x05, // Right camera TD event, increase in illumination (polarity '1')
    RIGHT_APS_END   = 0x06, // Right camera APS event, measurement end (polarity '0')
    RIGHT_APS_START = 0x07, // Right camera APS event, measurement start (polarity '1')
    EVT_TIME_HIGH   = 0x08, // Timer high bits, also used to synchronize different event flows in the FPGA.
    STEREO_DISP     = 0x09, // Stereo disparity event
    EXT_TRIGGER     = 0x0A, // External trigger output
    IMU_EVT         = 0x0D, // Inertial Measurement Unit event that relays accelerometer and gyroscope information.
    OTHER           = 0x0E, // To be used for extensions in the event types
    CONTINUED       = 0X0F, // Extra data to previous events
};

} // namespace Metavision

#endif // METAVISION_HAL_BASE_EVENT_TYPES_H
