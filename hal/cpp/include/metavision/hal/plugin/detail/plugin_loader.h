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

#include <vector>
#include <string>

#ifndef METAVISION_HAL_DETAIL_PLUGIN_LOADER_H
#define METAVISION_HAL_DETAIL_PLUGIN_LOADER_H

namespace Metavision {

class Plugin;
class PluginLoader {
public:
    PluginLoader();
    ~PluginLoader();

    void clear_folders();
    void insert_folder(const std::string &folder);
    void insert_folders(const std::vector<std::string> &folders);

    void load_plugins();

    class PluginList;
    PluginList get_plugin_list();

private:
    struct PluginInfo;
    struct Library;

    void insert_plugin(const std::string &name, const std::string &library_path);
    void insert_plugin(const PluginInfo &info);

    std::vector<std::string> folders_;
    std::vector<std::unique_ptr<Library>> libraries_;

    static std::unique_ptr<Plugin> make_plugin(const std::string &plugin_name);

public:
    class PluginList {
        using container = typename std::vector<std::unique_ptr<Library>>;

    public:
        class iterator {
        public:
            using difference_type   = typename container::difference_type;
            using value_type        = typename container::value_type;
            using reference         = Plugin &;
            using pointer           = Plugin *;
            using iterator_category = std::input_iterator_tag; // or another tag

            iterator(const typename container::iterator &it);

            bool operator!=(const iterator &it) const;
            iterator &operator++();
            reference operator*() const;
            pointer operator->() const;

        private:
            typename container::iterator it_;
        };

        PluginList(std::vector<std::unique_ptr<Library>> &);
        iterator begin();
        iterator end();
        size_t size() const;
        bool empty() const;

    private:
        container &libraries_;
    };
}; // namespace Metavision

} // namespace Metavision

#endif // METAVISION_HAL_DETAIL_PLUGIN_LOADER_H