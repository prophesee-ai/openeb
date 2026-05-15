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

#ifndef METAVISION_HAL_PSEE_PATTERN_GENERATOR_H
#define METAVISION_HAL_PSEE_PATTERN_GENERATOR_H

#include <stdint.h>

namespace Metavision {

/// @brief Virtual interface to handle a sensor's pattern generator
///
/// The pattern generator of the sensor's generates a controlled stream of events according to a fixed configuration and
/// dynamics periods. The pattern generator configuration defines the pattern model type, the type of event generated,
/// and the polarity of the events generated. It must be decided before enabling the pattern generator. A pattern
/// generator period is defined by a duration (in 10 * ns) during which it generates a fixed event rate (in Mev/s). 2
/// periods (called n and p) can be configured dynamically (before or after enabling)
class PseePatternGenerator {
public:
    /// @brief Hold all the pattern generator properties for enabling
    ///
    struct Configuration {
        /// @brief Holds pattern type scan type
        ///
        enum class PatternType : uint8_t {
            /// goes through all the pixels column wise
            Column = 0,

            /// goes through the diagonal: x = y (from 0);
            Slash = 1,

            /// Evt3 vectorized pattern
            Vector = 2,
        };

        /// @brief Holds generated pixel type
        ///
        enum class PixelType : uint8_t {
            CD = 0,
            EM = 1,
        };

        PatternType pattern_type{PatternType::Column};
        PixelType pixel_type{PixelType::CD};
        uint8_t pixel_polarity{0};
    }; /// struct Configuration

    virtual ~PseePatternGenerator() {}

    /// @brief Configures the pattern generator according to the input and enables it.
    ///
    /// @param configuration The pattern generator configuration to set.
    /// @return If the pattern was effectively enabled with the input configuration
    virtual bool enable(const Configuration &configuration) = 0;

    /// @brief Gets the sensor's pattern geometry
    ///
    virtual void get_pattern_geometry(int &width, int &height) const = 0;

    /// @brief Disables the pattern generator
    ///
    virtual void disable() = 0;

    /// @brief Returns if the pattern generator is enabled
    ///
    virtual bool is_enabled() = 0;

    /// @brief Sets the duration of the first and second period of the pattern generator.
    ///
    /// Step duration is 10 ns i.e. passing 100 as step_count is equivalent to set a period of 1000 ns.
    /// Max period length is (2^16 - 1) * 10 ns i.e. max step count is (2^16 - 1)
    ///
    /// During this period length, the pattern generator generates events at a fixed rate (\ref set_period_rate).
    /// Periods n and p alternate i.e. N Mev/s are generated during n_step_count*10ns and then P Mev/s are generated
    /// during p_step_count*10ns and so on.
    ///
    /// @param n_step_count the first period step count. Sets a period of n_step_count*10ns
    /// @param p_step_count the first period step count. Sets a period of p_step_count*10ns
    virtual void set_period_step_count(uint16_t n_step_count, uint16_t p_step_count) = 0;

    /// @brief Sets the event rate generated during the first and second periods.
    ///
    /// @param n_rate_Mev_s the rate of the first period in Mev/s
    /// @param p_rate_Mev_s the rate of the second period in Mev/s
    virtual void set_period_rate(uint8_t n_rate_Mev_s, uint8_t p_rate_Mev_s) = 0;

protected:
    /// state variable to know if default values are to be set
    bool is_period_rate_set_{false};
    bool is_period_length_set_{false};
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PATTERN_GENERATOR_H
