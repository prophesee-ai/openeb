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

#include <cstdint>

#include "convert_event_frame.h"

__global__ void cuda_convert_histogram(int n, const uint8_t *in, float *out, int pos_bits, int neg_bits) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float mult = 1 << (8 - max(pos_bits, neg_bits));

    for (int i = index; i < n; i += stride) {
        out[i * 3] = 0;
        out[i * 3 + 1] = (in[i] & ((1 << neg_bits) -1)) * mult;
        out[i * 3 + 2] = (in[i] >> pos_bits) * mult;
    }
}

__global__ void cuda_convert_histogram_padded(int n, const uint8_t *in, float *out, int pos_bits, int neg_bits) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float mult = 1 << (8 - max(pos_bits, neg_bits));

    for (int i = index; i < n / 2; i += stride) {
        out[i * 3] = 0;
        out[i * 3 + 1] = (in[i * 2] & ((1 << neg_bits) -1)) * mult;
        out[i * 3 + 2] = (in[i * 2 + 1] & ((1 << pos_bits) - 1)) * mult;
    }
}

__global__ void cuda_convert_diff(int n, const int8_t *in, float *out, int nbits) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float mult = 1 << (8 - nbits);

    for (int i = index; i < n; i += stride) {
        const float value = in[i] * mult + 127;
        out[i * 3] = value;
        out[i * 3 + 1] = value;
        out[i * 3 + 2] = value;
    }
}

void convert_histogram(int n, const uint8_t *in, float *out, int pos_bits, int neg_bits) {
    int nblocks = (n + 31) / 32;
    cuda_convert_histogram<<<nblocks, 32>>>(n, in, out, pos_bits, neg_bits);
}

void convert_histogram_padded(int n, const uint8_t *in, float *out, int pos_bits, int neg_bits) {
    int nblocks = (n / 2 + 31) / 32;
    cuda_convert_histogram_padded<<<nblocks, 32>>>(n, in, out, pos_bits, neg_bits);
}

void convert_diff(int n, const int8_t *in, float *out, int nbits) {
    int nblocks = (n + 31) / 32;
    cuda_convert_diff<<<nblocks, 32>>>(n, in, out, nbits);
}
