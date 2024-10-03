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

#ifndef METAVISION_STREAM_CAMERA_GENERATION_INTERNAL_H
#define METAVISION_STREAM_CAMERA_GENERATION_INTERNAL_H

#include <string>
#include "metavision/sdk/stream/camera_generation.h"

namespace Metavision {
class Device;

struct CameraGeneration::Private {
    Private(const std::string &name);
    Private(short version_major, short version_minor, const std::string &name);

    virtual ~Private();

    // Remark : to determine if two cameras belong to the same generation, the type is not taken into account
    bool operator==(const CameraGeneration::Private &c) const;

    bool operator!=(const CameraGeneration::Private &c) const;

    bool operator<(const CameraGeneration::Private &c) const;

    bool operator<=(const CameraGeneration::Private &c) const;

    bool operator>(const CameraGeneration::Private &c) const;

    bool operator>=(const CameraGeneration::Private &c) const;

    static CameraGeneration *build(const std::string &name);
    static CameraGeneration *build(short version_major, short version_minor, const std::string &name = std::string());
    static CameraGeneration *build(Device &device);

    const short major_{-1};
    const short minor_{-1};
    const std::string name_;
};

} // namespace Metavision

#endif // METAVISION_STREAM_CAMERA_GENERATION_INTERNAL_H
