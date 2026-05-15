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

#ifndef METAVISION_HAL_PLUGIN_ENTRYPOINT_H
#define METAVISION_HAL_PLUGIN_ENTRYPOINT_H

#include <string>

#include "metavision/hal/plugin/plugin.h"
#include "metavision/hal/metavision_hal_export.h"

extern "C" {

/// @brief Main entry point of the plugin
///
/// This function should initialize the empty instantiated Plugin object
/// passed as argument through the raw (untyped) pointer @p plugin_ptr
///
/// For convenience, you may use the function @ref Metavision::plugin_cast to
/// cast the provided pointer back to a @ref Metavision::Plugin object.
///
/// @param plugin_ptr Pointer to the plugin to be initialized
METAVISION_HAL_EXTERN_EXPORT void initialize_plugin(void *plugin_ptr);
}

namespace Metavision {

/// @brief  @brief Convenience function to get the name of the plugin entrypoint
/// @return The plugin entry point function name
inline const std::string &get_plugin_entry_point() {
    static std::string entrypoint("initialize_plugin");
    return entrypoint;
}

/// @brief Convenience function to cast a raw untyped pointer to a plugin
/// @param plugin_ptr Raw pointer to an empty plugin
/// @return Reference to the pointer plugin
inline Metavision::Plugin &plugin_cast(void *plugin_ptr) {
    return *reinterpret_cast<Metavision::Plugin *>(plugin_ptr);
}
} // namespace Metavision

#endif // METAVISION_HAL_PLUGIN_ENTRYPOINT_H
