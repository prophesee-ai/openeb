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

#ifndef METAVISION_HAL_PSEE_ROI_H
#define METAVISION_HAL_PSEE_ROI_H

#include "metavision/hal/facilities/i_roi.h"

namespace Metavision {

class PseeROI : public I_ROI {
public:
    PseeROI(int width, int height);

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
    virtual bool set_mode(const Mode &mode) override;

    /// @brief Gets the window mode
    /// @return the window mode
    virtual Mode get_mode() const override;

    /// @brief Gets the maximum number of windows
    /// @return the maximum number of windows that can be set via @ref set_windows
    virtual size_t get_max_supported_windows_count() const override;

    /// @brief Implementation of `set_windows`
    /// @param windows A vector of windows to set
    /// @return true on success
    /// @throw an exception if the size of the vector is higher than the maximum supported number
    ///        of windows (see @ref get_max_supported_windows_count)
    virtual bool set_windows_impl(const std::vector<Window> &windows) override final;

    /// @brief Gets active ROI/RONI windows
    ///
    /// @return The vector of active windows
    virtual std::vector<Window> get_windows() const override;

    /// @brief Creates several rectangular ROI in bitword register format
    /// @param vroi Vector of ROI to transform to bitword register format
    virtual std::vector<uint32_t> create_ROIs(const std::vector<Window> &vroi);

    /// @brief Sets multiple ROIs from row and column binary maps
    ///
    /// The binary maps (std::vector<bool>) arguments must have the sensor's dimension
    ///
    /// @param cols Vector of boolean of size sensor's width representing the binary map of the columns to
    /// enable
    /// @param rows Vector of boolean of size sensor's height representing the binary map of the rows to
    /// enable
    /// @return true if input have the correct dimension and the ROI is set correctly, false otherwise
    /// @warning For a pixel to be enabled, it must be enabled on both its row and column
    virtual bool set_lines(const std::vector<bool> &cols, const std::vector<bool> &rows) override;

    /// @brief Get active ROI lines
    ///
    /// @param cols Vector of boolean to store active columns
    /// @param rows Vector of boolean to store active lines
    /// @return true on success
    virtual bool get_lines(std::vector<bool> &cols, std::vector<bool> &rows) const override;

protected:
    static std::vector<uint32_t> create_ROIs(const std::vector<Window> &vroi, int device_width, int device_height,
                                             bool x_flipped, int word_size, int x_offset = 0, int y_offset = 0);

    /// @brief Creates several rectangular ROI in bitword register format
    /// @param cols_to_enable Vector of boolean representing the binary map of the columns to disable (0) or to enable
    /// (1)
    /// @param rows_to_enable Vector of boolean representing the binary map of the rows to disable (0) or to enable (1)
    std::vector<uint32_t> create_ROIs(const std::vector<bool> &cols_to_enable, const std::vector<bool> &rows_to_enable,
                                      int x_offset = 0, int y_offset = 0);

    std::vector<Window> lines_to_windows(const std::vector<bool> &cols_to_enable, const std::vector<bool> &rows_to_enable);

    void lines_from_windows(const std::vector<Window> &windows, std::vector<bool> &cols, std::vector<bool> &rows) const;

    virtual bool write_ROI_windows(const std::vector<Window> &windows);

    /// Returns if x axis is flipped for coding the ROI
    virtual bool roi_x_flipped() const;

    /// Returns word size
    virtual int get_word_size() const;

    /// Writes ROI parameters
    virtual void write_ROI(const std::vector<uint32_t> &vroiparams) = 0;

    const int device_height_{0};
    const int device_width_{0};

private:

    bool is_lines;
    std::vector<Window> active_windows_;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_ROI_H
