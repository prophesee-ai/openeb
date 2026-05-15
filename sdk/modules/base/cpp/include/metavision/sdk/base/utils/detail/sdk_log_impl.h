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

#ifndef METAVISION_SDK_BASE_SDK_LOG_IMPL_H
#define METAVISION_SDK_BASE_SDK_LOG_IMPL_H

#include "metavision/sdk/base/utils/log.h"

namespace Metavision {
namespace detail {
namespace sdk {
static std::string PrefixFmt("[SDK][<LEVEL>] ");

template<Metavision::LogLevel Level>
Metavision::LoggingOperation<Level> log(const std::string &file, int line, const std::string &function) {
    return Metavision::LoggingOperation<Level>(Metavision::getLogOptions(), PrefixFmt, file, line, function);
}
} // namespace sdk
} // namespace detail
} // namespace Metavision

#define MV_SDK_LOG_WRAP_LEVEL(level) MV_LOG_WRAP(Metavision::detail::sdk::log, (level))

#define MV_SDK_LOG_DEBUG MV_SDK_LOG_WRAP_LEVEL(Metavision::LogLevel::Debug)
#define MV_SDK_LOG_TRACE MV_SDK_LOG_WRAP_LEVEL(Metavision::LogLevel::Trace)
#define MV_SDK_LOG_INFO MV_SDK_LOG_WRAP_LEVEL(Metavision::LogLevel::Info)
#define MV_SDK_LOG_WARNING MV_SDK_LOG_WRAP_LEVEL(Metavision::LogLevel::Warning)
#define MV_SDK_LOG_ERROR MV_SDK_LOG_WRAP_LEVEL(Metavision::LogLevel::Error)

#endif // METAVISION_SDK_BASE_SDK_LOG_IMPL_H
