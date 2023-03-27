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

#ifndef METAVISION_SDK_DRIVER_FRAME_DIFF_INTERNAL_H
#define METAVISION_SDK_DRIVER_FRAME_DIFF_INTERNAL_H

#include "metavision/sdk/core/utils/callback_manager.h"

namespace Metavision {

class IndexGenerator;

class FrameDiff::Private : public CallbackManager<RawEventFrameDiffCallback> {
public:
    Private(IndexManager &index_manager);

    virtual ~Private();

    static FrameDiff *build(IndexManager &index_manager);
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_FRAME_DIFF_INTERNAL_H
