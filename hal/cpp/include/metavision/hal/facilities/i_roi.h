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

#ifndef METAVISION_HAL_I_ROI_H
#define METAVISION_HAL_I_ROI_H

#include <cstdint>
#include <string>
#include <vector>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Interface facility for ROI (Region Of Interest)
class I_ROI : public I_RegistrableFacility<I_ROI> {
public:
    /// @brief Defines a simple rectangular window
    class Window {
    public:
        Window();

        /// @brief Creates a window defined by the corners {(x, y), (x + width, y + height)}
        Window(int x, int y, int width, int height);

        // defined for python bindings
        bool operator==(const Window &window) const;

        /// @brief Returns the window as a string
        /// @return Human readable string representation of a window
        std::string to_string() const;

        // defined to read from a string buffer
        friend std::istream &operator>>(std::istream &is, Window &rhs);

        // defined to output a string
        friend std::ostream &operator<<(std::ostream &lhs, Window &rhs);

        int x      = -1;
        int y      = -1;
        int width  = -1;
        int height = -1;
    };

    /// @brief Applies ROI
    /// @param state If true, enables ROI. If false, disables it
    /// @warning At least one ROI should have been set before calling this function
    /// @return true on success
    virtual bool enable(bool state) = 0;

    /// @brief Window mode
    ///
    /// ROI : Region of interest, any event outside the window will be discarded
    /// RONI : Region of non interest, any event inside the window will be discarded
    enum class Mode { ROI, RONI };

    /// @brief Sets the window mode
    /// @param mode window mode to set (ROI or RONI)
    /// @return true on success
    virtual bool set_mode(const Mode &mode) = 0;

    /// @brief Sets a window
    ///
    /// The window will be applied according to the current mode (ROI or RONI)
    /// @param window window to set
    /// @return true on success
    bool set_window(const Window &window);

    /// @brief Gets the maximum number of windows
    /// @return the maximum number of windows that can be set via @ref set_windows
    virtual size_t get_max_supported_windows_count() const = 0;

    /// @brief Sets multiple windows
    ///
    /// The windows will be applied according to the current mode (ROI or RONI)
    /// In ROI mode, enabled pixels are those inside the provided rectangles.
    /// In RONI mode, enabled pixels are those where row OR column are covered by the provided rectangle.
    ///
    /// @param windows A vector of windows to set
    /// @return true on success
    /// @throw an exception if the size of @p windows is higher than the maximum supported number
    ///        of windows (see @ref get_max_supported_windows_count)
    bool set_windows(const std::vector<Window> &windows);

    /// @brief Sets multiple lines and columns from row and column binary maps
    ///
    /// The binary maps (std::vector<bool>) arguments must have the sensor's dimension.
    /// The lines and columns will be applied according to the current mode (ROI or RONI).
    /// In ROI mode, enabled pixels are those where row AND column are set to true.
    /// In RONI mode, disabled pixels are those where row AND column are set to false.
    /// This means that conversely, enabled pixels are those where row OR column are set to true.
    ///
    /// @param cols Vector of boolean of size sensor's width representing the binary map of the columns to
    /// enable
    /// @param rows Vector of boolean of size sensor's height representing the binary map of the rows to
    /// enable
    /// @return true if input have the correct dimension and the ROI is set correctly, false otherwise
    virtual bool set_lines(const std::vector<bool> &cols, const std::vector<bool> &rows) = 0;

private:
    /// @brief Implementation of `set_windows`
    /// @param windows A vector of windows to set
    /// @return true on success
    /// @throw an exception if the size of the vector is higher than the maximum supported number
    ///        of windows (see @ref get_max_supported_windows_count)
    virtual bool set_windows_impl(const std::vector<Window> &windows) = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_ROI_H
