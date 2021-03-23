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

#include "metavision/sdk/ui/detail/texture_utils.h"

#include <GL/glew.h>

namespace Metavision {
namespace detail {

unsigned int initialize_texture(int width, int height, bool is_gray) {
    unsigned int tex_id;

    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    float const_bkg_color[4] = {0, 0, 0, 0};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, const_bkg_color);

    if (is_gray) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, 0);
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    return tex_id;
}

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

    const auto internal_format = (img.type() == CV_8UC3) ? GL_RGB8 : GL_R8;
    const auto format          = (img.type() == CV_8UC3) ? GL_BGR : GL_RED;

    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, img.cols, img.rows, 0, format, GL_UNSIGNED_BYTE, img.ptr());

    glBindTexture(GL_TEXTURE_2D, 0);
}

} // namespace detail
} // namespace Metavision