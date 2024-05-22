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

#ifndef METAVISION_HAL_GENX320_ROI_DRIVER_H
#define METAVISION_HAL_GENX320_ROI_DRIVER_H

#include <filesystem>
#include <vector>
#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/hal/utils/device_config.h"
#include "metavision/hal/facilities/i_roi.h"

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

class GenX320RoiDriver : public I_RegistrableFacility<GenX320RoiDriver> {
public:
    GenX320RoiDriver(int width, int height, const std::shared_ptr<RegisterMap> &regmap,
                     const std::string &sensor_prefix, const DeviceConfig &config);

    class Grid {
    public:
        Grid(int columns, int rows);

        void clear();
        void set_vector(const unsigned int &vector_id, const unsigned int &row, const unsigned int &val);
        unsigned int &get_vector(const unsigned int &vector_id, const unsigned int &row);
        unsigned int get_vector(const unsigned int &vector_id, const unsigned int &row) const;
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

    /// @brief ROI controller driver mode
    ///
    /// MASTER : Region of interest controlled by ROI Master state machine
    /// LATCH : Region of interest controlled by register access to latches
    /// IO : Region of interest controlled by pixel reset register or external input pin
    enum class DriverMode { MASTER, LATCH, IO };

    /// @brief Set ROI controller mode
    /// @param driver_mode driver mode to set
    /// @return true on success
    bool set_driver_mode(const DriverMode &driver_mode);

    /// @brief Gets ROI controller mode
    /// @return ROI driver mode
    DriverMode get_driver_mode() const;

    static std::filesystem::path default_calibration_path();
    bool load_calibration_file(const std::filesystem::path &path);

    /// @brief Returns the default device width
    ///
    /// This values is obtained from the default Device passed in the constructor of the class.
    int device_width() const;

    /// @brief Returns the default device height
    ///
    /// This values is obtained from the default Device passed in the constructor of the class.
    int device_height() const;

    /// @brief Sets the window mode
    /// @param mode window mode to set (ROI or RONI)
    /// @return true on success
    bool set_roi_mode(const I_ROI::Mode &mode);

    /// @brief Gets the ROI mode
    /// @param mode ROI mode set (ROI or RONI)
    /// @return ROI mode
    I_ROI::Mode get_roi_mode() const;

    /// @brief Gets the maximum number of windows
    /// @return the maximum number of windows that can be set via @ref set_windows
    unsigned int get_max_windows_count() const;

    /// @brief Sets multiple ROIs Windows
    ///
    /// Configure windows registers of ROI master block
    ///
    /// @param windows A vector of windows to set
    /// @return true on success
    /// @throw an exception if the size of the vector is higher than the maximum supported number
    ///        of windows (see @ref get_max_supported_windows_count)
    bool set_windows(const std::vector<I_ROI::Window> &windows);

    /// @brief Sets multiple ROIs from row and column binary maps
    ///
    /// The binary maps (std::vector<bool>) arguments must have the sensor's dimension
    /// Configure Grid underlying attribute with lines
    ///
    /// @param cols Vector of boolean of size sensor's width representing the binary map of the columns to
    /// enable
    /// @param rows Vector of boolean of size sensor's height representing the binary map of the rows to
    /// enable
    /// @return true if input have the correct dimension and the ROI is set correctly, false otherwise
    /// @warning For a pixel to be enabled, it must be enabled on both its row and column
    bool set_lines(const std::vector<bool> &cols, const std::vector<bool> &rows);

    bool set_pixel(const unsigned int &column, const unsigned int &row, const bool &enable);

    /// @brief Sets Grid ROI from user Grid configuration
    ///
    /// Configure Grid underlying attribute with user Grid object
    ///
    /// @param user_grid Grid
    /// @return true if input have the correct dimension and the ROI is set correctly, false otherwise
    bool set_grid(Grid &user_grid);

    /// @brief Apply configured ROI Windows to sensor
    /// @param window_count
    void apply_windows(unsigned int window_count);

    /// @brief Apply configured Grid config to sensor
    void apply_grid();

    /// @brief Apply ROI configuration based on driver mode
    /// @param state If true, enables ROI. If false, reset to full ROI
    /// @warning At least one ROI should have been set before calling this function
    /// @return true on success
    bool enable(bool state);

    void reset_to_full_roi();

    std::vector<I_ROI::Window> get_windows() const;
    bool get_lines(std::vector<bool> &cols, std::vector<bool> &rows) const;
    void print_windows_config();

    Grid get_grid() const;
    void print_grid_config();

    void pixel_reset(const bool &enable);

private:
    int device_height_{0};
    int device_width_{0};

    std::shared_ptr<RegisterMap> register_map_;
    std::string sensor_prefix_;

    DriverMode driver_mode_;
    I_ROI::Mode mode_;

    bool is_lines_;
    Grid grid_;
    I_ROI::Window main_window_;
    unsigned int roi_window_cnt_;

    void open_all_latches();
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_ROI_DRIVER_H
