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

#ifndef SRC_INC_STEREO_FX3_HOSTIF_REGISTERMAP_H_
#define SRC_INC_STEREO_FX3_HOSTIF_REGISTERMAP_H_

RegmapData stereo_FX3HostIFRegisterMap[] = {{R, {{"PKT_END_ENABLE", 0x0}}},
                                            {F, {{"SHORT_PACKET_ENABLE", 0, 1, 0x1}}},
                                            {F, {{"SHORT_PACKET_ENABLE_SKIP", 1, 1, 0x0}}},
                                            {R, {{"PKT_END_INTERVAL_US", 0x4}}},
                                            {F, {{"", 0, 32, 0x400}}},
                                            {R, {{"PKT_END_DATA_COUNT", 0x8}}},
                                            {F, {{"", 0, 32, 0x400}}}

};
unsigned int stereo_FX3HostIFRegisterMapSize =
    sizeof(stereo_FX3HostIFRegisterMap) / sizeof(stereo_FX3HostIFRegisterMap[0]);

#endif
