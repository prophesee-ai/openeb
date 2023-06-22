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

#ifndef METAVISION_HAL_GENX320_LL_ROI_H
#define METAVISION_HAL_GENX320_LL_ROI_H

#include <filesystem>
#include <vector>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

class DeviceConfig;
class RegisterMap;

class GenX320LowLevelRoi : public I_RegistrableFacility<GenX320LowLevelRoi> {
public:
    class Grid {
    public:
        Grid(int columns, int rows);

        void set_vector(const unsigned int &vector_id, const unsigned int &row, const unsigned int &val);
        unsigned int &get_vector(const unsigned int &vector_id, const unsigned int &row);
        void set_pixel(const unsigned int &column, const unsigned int &row, const bool &enable);

        /// @brief Returns the grid as a string
        /// @return Human readable string representation of a grid
        std::string to_string() const;

        std::tuple<unsigned int, unsigned int> get_size() const;

    private:
        std::vector<unsigned int> grid_;
        unsigned int rows_;
        unsigned int columns_;
    };

    GenX320LowLevelRoi(const DeviceConfig &config, const std::shared_ptr<RegisterMap> &regmap,
                       const std::string &sensor_prefix);

    void reset();
    bool apply(GenX320LowLevelRoi::Grid &user_grid);

    static std::filesystem::path default_calibration_path();
    bool load_calibration_file(const std::filesystem::path &path);

private:
    std::shared_ptr<RegisterMap> register_map_;
    std::string sensor_prefix_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_LL_ROI_H
