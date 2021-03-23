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

#ifndef METAVISION_SDK_CORE_STAGE_H
#define METAVISION_SDK_CORE_STAGE_H

#include <atomic>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <boost/any.hpp>

#include "metavision/sdk/core/pipeline/base_stage.h"

namespace Metavision {

/// @brief Simple stage that can be customized
///
/// This class can be used to create a stage instance to be customized via
/// @ref set_consuming_callback, @ref set_starting_callback, @ref set_stopping_callback etc.
/// This can be an alternative to the creation of a new class inheriting @ref BaseStage.
class Stage : public BaseStage {
public:
    /// @brief Constructor
    /// @param detachable If this stage can be detached (i.e. can run on its own thread)
    inline Stage(bool detachable = true) : BaseStage(detachable) {}

    /// @brief Constructor
    ///
    /// The @p prev_stage is used to setup the consuming callback
    /// that will be called when the previous stage produces data.
    /// When the previous stage produces data, it will call the consuming
    /// callback (@ref set_consuming_callback) of this stage.
    /// This behavior is automatically handled by this constructor.
    /// If you need to customize the consuming callback of one (or all) of the previous stages,
    /// you should use @ref set_consuming_callback instead.
    ///
    /// @param prev_stage the stage that is executed before the created one
    /// @param detachable If this stage can be detached (i.e. can run on its own thread)
    inline Stage(BaseStage &prev_stage, bool detachable = true) : BaseStage(prev_stage, detachable) {}
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_STAGE_H
