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

#ifndef METAVISION_SDK_CORE_EVENT_TRACKED_BOX_H
#define METAVISION_SDK_CORE_EVENT_TRACKED_BOX_H

#include <sstream>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/utils/detail/struct_pack.h"
#include "metavision/sdk/base/events/detail/event_traits.h"

namespace Metavision {

/// @brief Class representing a spatio-temporal tracked bounding-box event.
struct EventTrackedBox {
    /// @brief Constructor
    /// @param t Timestamp
    /// @param x X coordinate of the top-left corner of the box
    /// @param y Y coordinate of the top-left corner of the box
    /// @param w Box's Width
    /// @param h Box's height
    /// @param class_id Box's class label identifier
    /// @param track_id Track identifier
    /// @param class_confidence Confidence score of the detection
    inline EventTrackedBox(timestamp t = 0, float x = 0.f, float y = 0.f, float w = 0.f, float h = 0.f,
                           unsigned int class_id = 0, unsigned int track_id = 0, float class_confidence = 0) noexcept :
        t(t),
        x(x),
        y(y),
        w(w),
        h(h),
        class_id(class_id),
        track_id(track_id),
        class_confidence(class_confidence),
        tracking_confidence(class_confidence),
        last_detection_update_time(t),
        nb_detections(1) {}

    /// @brief Writes the tracked box into a csv format
    /// @param output Stream to write the tracked box
    /// @param sep The separator to use between attributes
    void write_csv_line(std::ostream &output, char sep = ' ') const {
        output << t << sep << class_id << sep << track_id << sep << x << sep << y << sep << w << sep << h << sep
               << class_confidence << sep << tracking_confidence << sep << last_detection_update_time << sep
               << nb_detections << std::endl;
    }

    /// @brief Serializes an EventBbox into a stream
    /// @param output Stream
    /// @param e EventBbox to be serialized
    /// @return Stream provided as input
    friend std::ostream &operator<<(std::ostream &output, const EventTrackedBox &e) {
        output << "EventTrackedBox: ("
               << "t: " << e.t << "   "
               << "x: " << e.x << "   "
               << "y: " << e.y << "   "
               << "w: " << e.w << "   "
               << "h: " << e.h << "   "
               << "class_id: " << e.class_id << "   "
               << "track_id: " << e.track_id << "   "
               << "class_confidence: " << e.class_confidence << "   "
               << "tracking_confidence: " << e.tracking_confidence << "   "
               << "last_detection_update_time: " << e.last_detection_update_time << "   "
               << "nb_detections: " << e.nb_detections << ")";
        return output;
    }

    /// @brief Updates the last detection timestamp and compute a new track confidence value
    /// @param t Timestamp of the new detection
    /// @param detection_confidence Detection confidence value
    /// @param similarity_box_track Weight applied on the detection to compute track detection
    inline void set_last_detection_update(timestamp t, float detection_confidence = 0.5f,
                                          float similarity_box_track = 1.0f) {
        last_detection_update_time = t;
        class_confidence           = detection_confidence;
        tracking_confidence += detection_confidence * similarity_box_track;
        tracking_confidence = std::min(tracking_confidence, 1.f);
    }

    // attributes
    timestamp t;                          ///< Timestamp of the box
    float x;                              ///< X position of the bounding box
    float y;                              ///< Y position of the bounding box
    float w;                              ///< Width of the bounding box
    float h;                              ///< Height of the bounding box
    unsigned int class_id;                ///< bounding box's class id
    int track_id;                         ///< Track identifier
    float class_confidence;               ///< Confidence of the detection
    float tracking_confidence;            ///< Confidence computed from previous detection and matching
    timestamp last_detection_update_time; ///< Time of last update of the detection.
    int nb_detections;                    ///< Number of time this box have been seen
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_EVENT_TRACKED_BOX_H
