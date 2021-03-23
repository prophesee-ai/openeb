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

#ifndef METAVISION_SDK_UI_CONSTANTS_H
#define METAVISION_SDK_UI_CONSTANTS_H

#include <string>
#include <map>
#include <metavision/sdk/ui/utils/ui_event.h>

extern const std::string image_path;
extern const std::map<Metavision::UIKeyEvent, std::string> key_to_names;
extern const std::map<Metavision::UIMouseButton, std::string> button_to_names;

#endif // METAVISION_SDK_UI_CONSTANTS_H
