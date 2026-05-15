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

#ifndef CONVERT_EVENT_FRAME_H
#define CONVERT_EVENT_FRAME_H

void convert_histogram(int n, const uint8_t *in, float *out, int pos_bits, int neg_bits);
void convert_histogram_padded(int n, const uint8_t *in, float *out, int pos_bits, int neg_bits);
void convert_diff(int n, const int8_t *in, float *out, int nbits);

#endif // CONVERT_EVENT_FRAME_H
