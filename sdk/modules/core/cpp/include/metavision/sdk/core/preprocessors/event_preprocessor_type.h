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

#ifndef METAVISION_SDK_CORE_EVENT_PREPROCESSOR_TYPE_H
#define METAVISION_SDK_CORE_EVENT_PREPROCESSOR_TYPE_H

#include <string>
#include <unordered_map>

namespace Metavision {

enum class EventPreprocessorType : uint8_t { DIFF, HISTO, EVENT_CUBE, HARDWARE_HISTO, HARDWARE_DIFF, TIME_SURFACE };

static std::unordered_map<std::string, EventPreprocessorType> const stringToEventPreprocessorTypeMap = {
    {"diff", EventPreprocessorType::DIFF},
    {"diff3d", EventPreprocessorType::DIFF},
    {"histo", EventPreprocessorType::HISTO},
    {"histo3d", EventPreprocessorType::HISTO},
    {"event_cube", EventPreprocessorType::EVENT_CUBE},
    {"hardware_histo", EventPreprocessorType::HARDWARE_HISTO},
    {"hardware_diff", EventPreprocessorType::HARDWARE_DIFF},
    {"time_surface", EventPreprocessorType::TIME_SURFACE}};

static std::unordered_map<EventPreprocessorType, std::string> const eventPreprocessorTypeToStringMap = {
    {EventPreprocessorType::DIFF, "diff"},
    {EventPreprocessorType::HISTO, "histo"},
    {EventPreprocessorType::EVENT_CUBE, "event_cube"},
    {EventPreprocessorType::HARDWARE_HISTO, "hardware_histo"},
    {EventPreprocessorType::HARDWARE_DIFF, "hardware_diff"},
    {EventPreprocessorType::TIME_SURFACE, "time_surface"}};

} // namespace Metavision

namespace std {
std::istream &operator>>(std::istream &in, Metavision::EventPreprocessorType &type);
std::ostream &operator<<(std::ostream &os, const Metavision::EventPreprocessorType &type);
} // namespace std

#endif // METAVISION_SDK_CORE_EVENT_PREPROCESSOR_TYPE_H
