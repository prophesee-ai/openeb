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

#ifndef METAVISION_SDK_ML_EVENT_BBOX_H
#define METAVISION_SDK_ML_EVENT_BBOX_H

#include <limits>
#include <ostream>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/utils/detail/struct_pack.h"
#include "metavision/sdk/base/events/detail/event_traits.h"

#include "metavision/sdk/core/utils/similarity_metrics.h"

namespace Metavision {

/// @brief Class representing a spatio-temporal bounding-box event.
///
/// The timestamp of the event (i.e. member variable 't') is by convention the detection timestamp.
/// Convention about the spatial position is that points with u in [x, x + w[ and v in [y, y + h[ are inside
/// the bounding-box and everything else is outside.
struct EventBbox {
    /// @brief Default constructor
    inline EventBbox() {
        t                = 0;
        x                = 0.f;
        y                = 0.f;
        w                = 0.f;
        h                = 0.f;
        class_id         = 0;
        track_id         = 0;
        class_confidence = 0.f;
    }

    /// @brief Constructs a Event Bounding box
    ///
    /// @param time Timestamp of last detection
    /// @param x Column index
    /// @param y Row index
    /// @param w Bounding box's width
    /// @param h Bounding box's height
    /// @param class_id Class identifier
    /// @param track_id Track identification number
    /// @param class_confidence Detection confidence
    inline EventBbox(timestamp time, float x, float y, float w, float h, unsigned int class_id, unsigned int track_id,
                     float class_confidence) :
        t(time), x(x), y(y), w(w), h(h), class_id(class_id), track_id(track_id), class_confidence(class_confidence) {}

    /// @brief Writes EventBbox in buffer
    ///
    /// @param buf Memory in which the bounding box will be serialized
    /// @param origin Reference timestamp cut away from the bounding box timestamp
    inline void write_event(void *buf, timestamp origin) const;

    /// @brief Reads EventBbox from a buffer
    ///
    /// @param buf Memory containing a serialized bounding box
    /// @param delta_ts Origin timestamp to be added to the serialized timestamp
    /// @return An unserialized event bounding box
    inline static EventBbox read_event(void *buf, const timestamp &delta_ts);

    /// @brief Gets x position of the bounding box
    /// @return X position of the bounding box
    inline float get_x() const {
        return x;
    }

    /// @brief Gets y position of the bounding box
    /// @return Y position of the bounding box
    inline float get_y() const {
        return y;
    }

    /// @brief Gets bounding box's width
    /// @return Width of the bounding box
    inline float get_width() const {
        return w;
    }

    /// @brief Gets bounding box's height
    /// @return Height of the bounding box
    inline float get_height() const {
        return h;
    }

    /// @brief Gets bounding box's class id
    /// @return bounding box's class id
    inline unsigned int get_class_id() const {
        return class_id;
    }

    /// @brief Computes the area recovered by both boxes
    /// @param bbox2 Box to be compared with
    /// @return The intersection area between current bbox and bbox 2 (in pixel * pixel)
    inline float intersection_area(const EventBbox &bbox2) const;

    /// @brief Computes the proportion of box overlap
    /// @param bbox2 Box to be compared with
    /// @return Percentage of overlap
    inline float intersection_area_over_union(const EventBbox &bbox2) const;

    /// @brief Serializes an EventBbox into a stream
    /// @param output Stream
    /// @param e EventBbox to be serialized
    /// @return Stream provided as input
    friend std::ostream &operator<<(std::ostream &output, const EventBbox &e) {
        output << "EventBbox: ("
               << "t: " << e.t << "   "
               << "x: " << e.x << "   "
               << "y: " << e.y << "   "
               << "w: " << e.w << "   "
               << "h: " << e.h << "   "
               << "class_id: " << e.class_id << "   "
               << "track_id: " << e.track_id << "   "
               << "class_confidence: " << e.class_confidence << ")";
        return output;
    }

    /// @brief Serialize a bounding box in csv format
    ///
    /// @param output Stream in which the csv of the bounding box will be written
    /// @param sep Character inserted between fields
    void write_csv_line(std::ostream &output, char sep = ' ') const {
        output << t << sep << class_id << sep << track_id << sep << x << sep << y << sep << w << sep << h << sep
               << class_confidence << std::endl;
    }

    FORCE_PACK(
        /// Packed struct to serialize events
        struct RawEvent {
            uint32_t ts;
            float x;
            float y;
            float w;
            float h;
            unsigned char class_id;
            unsigned int track_id;
            float class_confidence;
        });

    timestamp t;            ///< timestamp of the detection
    float x;                ///< X coordinate of top left corner
    float y;                ///< Y coordinate of top left corner
    float w;                ///< width of the bounding box
    float h;                ///< height of the bounding box
    unsigned int class_id;  ///< Class identifier of detected object
    unsigned int track_id;  ///< Track identifier
    float class_confidence; ///< Confidence of the detection
};

void EventBbox::write_event(void *buf, timestamp origin) const {
    RawEvent *buffer         = static_cast<RawEvent *>(buf);
    buffer->ts               = t - origin;
    buffer->x                = x;
    buffer->y                = y;
    buffer->w                = w;
    buffer->h                = h;
    buffer->class_id         = class_id;
    buffer->track_id         = track_id;
    buffer->class_confidence = class_confidence;
}

EventBbox EventBbox::read_event(void *buf, const timestamp &delta_ts) {
    RawEvent *buffer = static_cast<RawEvent *>(buf);
    return EventBbox(buffer->ts + delta_ts, buffer->x, buffer->y, buffer->w, buffer->h, buffer->class_id,
                     buffer->track_id, buffer->class_confidence);
}

float EventBbox::intersection_area(const EventBbox &bbox2) const {
    return Utils::intersection(*this, bbox2);
}

float EventBbox::intersection_area_over_union(const EventBbox &bbox2) const {
    return Utils::intersection_over_union(*this, bbox2);
}

} // namespace Metavision

METAVISION_DEFINE_EVENT_TRAIT(Metavision::EventBbox, 16, "Bbox")

#endif // METAVISION_SDK_CORE_EVENT_BBOX_H
