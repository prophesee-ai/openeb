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

#include "metavision/sdk/stream/frame_histo.h"

#include "metavision/sdk/stream/internal/frame_histo_internal.h"
#include "metavision/sdk/core/utils/index_manager.h"
#include "metavision/sdk/stream/internal/callback_tag_ids.h"

namespace Metavision {

FrameHisto *FrameHisto::Private::build(IndexManager &index_manager) {
    return new FrameHisto(new Private(index_manager));
}

FrameHisto::Private::Private(IndexManager &index_manager) :
    CallbackManager<RawEventFrameHistoCallback>(index_manager, CallbackTagIds::DECODE_CALLBACK_TAG_ID) {}

FrameHisto::Private::~Private() {}

FrameHisto::~FrameHisto() {}

CallbackId FrameHisto::add_callback(const RawEventFrameHistoCallback &cb) {
    return pimpl_->add_callback(cb);
}

bool FrameHisto::remove_callback(CallbackId callback_id) {
    return pimpl_->remove_callback(callback_id);
}

FrameHisto::Private &FrameHisto::get_pimpl() {
    return *pimpl_;
}

FrameHisto::FrameHisto(Private *pimpl) : pimpl_(pimpl) {}

} // namespace Metavision
