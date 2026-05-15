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

#ifndef METAVISION_SDK_BASE_DETAIL_EVENT_TRAITS_H
#define METAVISION_SDK_BASE_DETAIL_EVENT_TRAITS_H

#include <cstddef>

/// @brief Template function used to ensure ID uniqueness across events
namespace Metavision {
namespace detail {
template<std::size_t Id>
constexpr std::size_t unique_id();
} // namespace detail
} // namespace Metavision

#define METAVISION_DEFINE_EVENT_TRAIT(type_, id_, name_)            \
    /* this will fail to compile if the ID has already been used */ \
    namespace Metavision {                                          \
    namespace detail {                                              \
    template<>                                                      \
    constexpr std::size_t unique_id<id_>() {                        \
        return id_;                                                 \
    }                                                               \
    }                                                               \
    }                                                               \
                                                                    \
    namespace Metavision {                                          \
    template<>                                                      \
    struct event_traits<type_> {                                    \
        typedef type_ type;                                         \
        static constexpr const char *name() {                       \
            return name_;                                           \
        }                                                           \
        static constexpr unsigned char id() {                       \
            return id_;                                             \
        }                                                           \
        static constexpr unsigned char size() {                     \
            return sizeof(type_::RawEvent);                         \
        }                                                           \
    };                                                              \
    }

namespace Metavision {

/// @brief Trait class that describes each event property (id, size, name, etc.)
/// @tparam EventType Type of event for which the trait is defined
///
template<typename EventType>
struct event_traits;

/// @brief Convenience function to get an event name
/// @tparam EventType Type of event for which the name is requested
/// @return Name of the event
template<typename EventType>
constexpr const char *get_event_name() {
    return event_traits<EventType>::name();
}

///
/// @brief Convenience function to get an event id
/// @tparam EventType Type of event for which the ID is requested
/// @return The ID of the event
///
template<typename EventType>
constexpr unsigned char get_event_id() {
    return event_traits<EventType>::id();
}

/// @brief Convenience function to get an event size
/// @tparam EventType Type of event for which the size is requested
/// @return Size of the event in bytes
template<typename EventType>
constexpr unsigned char get_event_size() {
    return event_traits<EventType>::size();
}

} // namespace Metavision

#endif // METAVISION_SDK_BASE_DETAIL_EVENT_TRAITS_H
