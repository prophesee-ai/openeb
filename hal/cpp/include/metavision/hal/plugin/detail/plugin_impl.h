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

#ifndef METAVISION_HAL_DETAIL_PLUGIN_IMPL_H
#define METAVISION_HAL_DETAIL_PLUGIN_IMPL_H

#include <iterator>
#include <string>
#include <memory>
#include <vector>

#include "metavision/hal/plugin/plugin.h"

namespace Metavision {

class CameraDiscovery;
class FileDiscovery;

namespace detail {
/// @brief Content of a plugin
template<typename T>
class iterator {
    using container = typename std::vector<std::unique_ptr<T>>;

public:
    using difference_type   = typename container::difference_type;
    using value_type        = typename container::value_type;
    using reference         = T &;
    using pointer           = T *;
    using iterator_category = std::input_iterator_tag; // or another tag

    iterator(const typename container::iterator &it) : it_(it) {}

    bool operator!=(const iterator &it) const {
        return it_ != it.it_;
    }

    iterator &operator++() {
        ++it_;
        return *this;
    }

    reference operator*() const {
        return *it_->get();
    }

    pointer operator->() const {
        return it_->get();
    }

private:
    typename container::iterator it_;
};
} // namespace detail

class Plugin::CameraDiscoveryList {
public:
    CameraDiscoveryList(Plugin &);
    detail::iterator<CameraDiscovery> begin();
    detail::iterator<CameraDiscovery> end();
    size_t size() const;
    bool empty() const;

private:
    std::vector<std::unique_ptr<CameraDiscovery>> &camera_discovery_list_;
};

class Plugin::FileDiscoveryList {
public:
    FileDiscoveryList(Plugin &);
    detail::iterator<FileDiscovery> begin();
    detail::iterator<FileDiscovery> end();
    size_t size() const;
    bool empty() const;

private:
    std::vector<std::unique_ptr<FileDiscovery>> &file_discovery_list_;
};

} // namespace Metavision

#endif // METAVISION_HAL_DETAIL_PLUGIN_IMPL_H
