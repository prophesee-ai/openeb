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

#ifndef METAVISION_SDK_DRIVER_CAMERA_GENERATION_H
#define METAVISION_SDK_DRIVER_CAMERA_GENERATION_H

#include <memory>

namespace Metavision {

/// @brief Facility class to handle camera generation
class CameraGeneration {
public:
    /// @brief Destructor
    ///
    /// Deletes a CameraGeneration class instance.
    virtual ~CameraGeneration();

    /// @brief Returns the major version of the camera's generation
    short version_major() const;

    /// @brief Returns the minor version of the camera's generation
    short version_minor() const;

    /// @brief overrides "equal to" operator
    bool operator==(const CameraGeneration &c) const;

    /// @brief overrides "not equal to" operator
    bool operator!=(const CameraGeneration &c) const;

    /// @brief overrides "less than" operator
    bool operator<(const CameraGeneration &c) const;

    /// @brief overrides "less than or equal to" operator
    bool operator<=(const CameraGeneration &c) const;

    /// @brief overrides "greater than" operator
    bool operator>(const CameraGeneration &c) const;

    /// @brief overrides "greater than or equal to" operator
    bool operator>=(const CameraGeneration &c) const;

    /// @brief For internal use
    struct Private;
    /// @brief For internal use
    Private &get_pimpl();

private:
    CameraGeneration(Private *);
    std::unique_ptr<Private> pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_CAMERA_GENERATION_H
