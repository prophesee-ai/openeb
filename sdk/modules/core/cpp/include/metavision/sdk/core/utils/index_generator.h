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

#ifndef METAVISION_SDK_CORE_INDEX_GENERATOR_H
#define METAVISION_SDK_CORE_INDEX_GENERATOR_H

#include <atomic>
#include <cstddef>

namespace Metavision {

/// @brief Utility class to create always incrementing indexes
class IndexGenerator {
public:
    /// @brief Constructor
    ///
    /// Creates an Indexer class instance.
    IndexGenerator() : next_idx_(0) {}

    /// @brief Destructor
    ///
    /// Deletes an Indexer class instance.
    ~IndexGenerator() {}

    /// @brief Return next available index
    size_t get_next_index() {
        return next_idx_++;
    }

private:
    std::atomic<size_t> next_idx_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_INDEX_GENERATOR_H
