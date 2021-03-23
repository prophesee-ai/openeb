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

#ifndef METAVISION_SDK_CORE_STREAM_LOGGING_STAGE_H
#define METAVISION_SDK_CORE_STREAM_LOGGING_STAGE_H

#include <boost/any.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "metavision/sdk/core/pipeline/base_stage.h"
#include "metavision/sdk/core/algorithms/stream_logger_algorithm.h"
#include "metavision/sdk/core/algorithms/stream_logger_algorithm.h"

namespace Metavision {

/// @brief Stage that runs @ref StreamLoggerAlgorithm
template<typename EventType>
class StreamLoggingStage : public BaseStage {
public:
    using EventBuffer     = std::vector<EventType>;
    using EventBufferPool = SharedObjectPool<EventBuffer>;
    using EventBufferPtr  = typename EventBufferPool::ptr_type;

    /// @brief Constructor
    /// @param filename Name of the output file
    /// @param width Width of the frame
    /// @param height Height of the frame
    StreamLoggingStage(const std::string &filename, int width, int height) :
        algo_(filename, static_cast<size_t>(width), static_cast<size_t>(height)) {
        set_starting_callback([this] { algo_.enable(true); });
        set_stopping_callback([this] { algo_.enable(false); });
        set_consuming_callback([this](const boost::any &data) {
            try {
                auto buffer = boost::any_cast<EventBufferPtr>(data);
                if (!buffer->empty())
                    algo_.process_events(buffer->begin(), buffer->end(), buffer->back().t);
            } catch (boost::bad_any_cast &) {}
        });
    }

    /// @brief Constructor
    /// @param prev_stage Previous Stage
    /// @param filename Name of the output file
    /// @param width Width of the frame
    /// @param height Height of the frame
    StreamLoggingStage(BaseStage &prev_stage, const std::string &filename, int width, int height) :
        StreamLoggingStage(filename, width, height) {
        set_previous_stage(prev_stage);
    }

    /// @brief Gets algo
    /// @return Algorithm class associated to this stage
    StreamLoggerAlgorithm &algo() {
        return algo_;
    }

private:
    StreamLoggerAlgorithm algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_STREAM_LOGGING_STAGE_H
