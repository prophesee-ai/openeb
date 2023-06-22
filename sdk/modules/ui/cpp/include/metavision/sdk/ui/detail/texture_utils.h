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

#ifndef METAVISION_TEXTURE_UTILS_H
#define METAVISION_TEXTURE_UTILS_H

#include <opencv2/core.hpp>

namespace Metavision {
namespace detail {

enum class TextureFormat { Gray, RGB, RGBA };
enum class TextureFilter { Nearest, Linear };

struct TextureOptions {
    std::uint32_t width;
    std::uint32_t height;
    TextureFormat format;
    TextureFilter minify_filter;
    TextureFilter magnify_filter;
};

unsigned int initialize_texture(const TextureOptions &options);

[[deprecated("This function is deprecated since version 4.2.0. Please use initialize_texture(const TextureOptions &) "
             "instead.")]] unsigned int
    initialize_texture(int width, int height, bool is_gray);

void upload_texture(const cv::Mat &img, const unsigned int &tex_id);

} // namespace detail
} // namespace Metavision

#endif // METAVISION_TEXTURE_UTILS_H
