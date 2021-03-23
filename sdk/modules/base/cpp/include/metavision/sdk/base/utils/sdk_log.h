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

#ifndef METAVISION_SDK_BASE_SDK_LOG_H
#define METAVISION_SDK_BASE_SDK_LOG_H

#ifdef DOXYGEN_FULL_VERSION
/// @brief Convenience macro to return a logging operation of Debug level using the current logging stream, default
/// prefix format and automatically adding an end of line token at the end of the logging operation
/// @return A logging operation of Debug level
#define MV_SDK_LOG_DEBUG()

/// @brief Convenience macro to return a logging operation of Trace level using the current logging stream, default
/// prefix format and automatically adding an end of line token at the end of the logging operation
/// @return A logging operation of Trace level
#define MV_SDK_LOG_TRACE()

/// @brief Convenience function to return a logging operation of Info level using the current logging stream, default
/// prefix format and automatically adding an end of line token at the end of the logging operation
/// @return A logging operation of Info level
#define MV_SDK_LOG_INFO()

/// @brief Convenience function to return a logging operation of Warning level using the current logging stream, default
/// prefix format and automatically adding an end of line token at the end of the logging operation
/// @return A logging operation of Warning level
#define MV_SDK_LOG_WARNING()

/// @brief Convenience function to return a logging operation of Error level using the current logging stream, default
/// prefix format and automatically adding an end of line token at the end of the logging operation
/// @return A logging operation of Error level
#define MV_SDK_LOG_ERROR()
#endif

#include "detail/sdk_log_impl.h"

#endif // METAVISION_SDK_BASE_SDK_LOG_H
