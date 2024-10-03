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

#include <unordered_map>

#include "metavision/sdk/ui/utils/opengl_api.h"
#include "metavision/sdk/ui/detail/texture_utils.h"

namespace Metavision {
namespace detail {

static std::unordered_map<TextureFilter, int> to_gl_filter = {{TextureFilter::Nearest, GL_NEAREST},
                                                              {TextureFilter::Linear, GL_LINEAR}};

static std::unordered_map<TextureFormat, int> to_gl_internal_format = {
    {TextureFormat::Gray, GL_R8}, {TextureFormat::RGB, GL_RGB8}, {TextureFormat::RGBA, GL_RGBA8}};

static std::unordered_map<TextureFormat, int> to_gl_format = {
    {TextureFormat::Gray, GL_RED}, {TextureFormat::RGB, GL_RGB}, {TextureFormat::RGBA, GL_RGBA}};

unsigned int initialize_texture(const TextureOptions &options) {
    unsigned int tex_id;

    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, to_gl_filter.at(options.minify_filter));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, to_gl_filter.at(options.magnify_filter));

    const int width  = static_cast<int>(options.width);
    const int height = static_cast<int>(options.height);
    if (options.format == TextureFormat::Gray) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, to_gl_internal_format.at(options.format), width, height, 0,
                     to_gl_format.at(options.format), GL_UNSIGNED_BYTE, 0);
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    return tex_id;
}

static std::unordered_map<int, int> cv_to_gl_internal_format = {
    {CV_8UC1, GL_R8}, {CV_8UC3, GL_RGB8}, {CV_8UC4, GL_RGBA8}};

static std::unordered_map<int, int> cv_to_gl_format = {{CV_8UC1, GL_RED}, {CV_8UC3, GL_RGB}, {CV_8UC4, GL_RGBA}};

void upload_texture(const cv::Mat &img, const unsigned int &tex_id) {
    glBindTexture(GL_TEXTURE_2D, tex_id);

    if ((img.step) % 8 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ((img.step) % 4 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ((img.step) % 2 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glPixelStorei(GL_UNPACK_ROW_LENGTH, img.step / img.elemSize());

    glTexImage2D(GL_TEXTURE_2D, 0, cv_to_gl_internal_format.at(img.type()), img.cols, img.rows, 0,
                 cv_to_gl_format.at(img.type()), GL_UNSIGNED_BYTE, img.ptr());

    glBindTexture(GL_TEXTURE_2D, 0);
}

} // namespace detail
} // namespace Metavision
