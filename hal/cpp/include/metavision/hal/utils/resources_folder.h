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

#ifndef METAVISION_HAL_RESOURCES_FOLDER_H
#define METAVISION_HAL_RESOURCES_FOLDER_H

#include <filesystem>
#include <string>

namespace Metavision {

/// @brief A dedicated class to handle resources' installation paths
class ResourcesFolder {
public:
#ifndef __ANDROID__
    /// @brief Returns path where user settings are stored
    /// @return User settings path
    static std::filesystem::path get_user_path();
#endif

    /// @brief Returns installation path of support directories (like firmwares)
    /// @return Installation path of support directories
    static std::string get_install_path();

    /// @brief Returns the plugins' installation path
    /// @return Plugins' installation path
    static std::string get_plugin_install_path();
};

} // namespace Metavision

#endif // METAVISION_HAL_RESOURCES_FOLDER_H
