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

#ifndef METAVISION_SDK_CORE_PREPROCESSORS_TENSOR_H
#define METAVISION_SDK_CORE_PREPROCESSORS_TENSOR_H

#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace Metavision {

/// @brief Structure to store information on a tensor dimension
struct Dimension {
    static constexpr int kDynamic = -1;
    std::string name;
    int dim;
};

/// @brief Structure to store information on a tensor dimensions
/// Dimension values should be >= 1 or equal to -1 to indicate a dynamic value
struct TensorShape {
    /// @brief Default constructor
    TensorShape();

    /// @brief Constructor
    /// @param dimensions Vector indicating the dimensions of the tensor
    TensorShape(const std::vector<Dimension> &dimensions);

    /// @brief overrides "equal to" operator
    bool operator==(const TensorShape &other) const;

    /// @brief overrides "not equal to" operator
    bool operator!=(const TensorShape &other) const;

    /// @brief Provides the total number of values in the tensor
    /// @returns The tensor size in terms of values
    size_t get_nb_values() const;

    /// @brief Compares the dimensions and checks for mismatch.
    /// It tests the equality of shape dimensions, except when one shape has a dynamic dimension.
    /// In that case, the other can have any dimension.
    /// @returns True if the two shape are equal, with dynamic dimensions allowed and considered as matching any
    /// dimension
    bool matches(const TensorShape &other) const;

    /// @brief Indicates whether the shape is composed of valid dimensions (strictly positive)
    /// @returns False if any of the dimensions is <= 0, True otherwise
    bool is_valid() const;

    std::vector<Dimension> dimensions;
};

/// @brief Retrieves the length of the required dimension in the provided shape
/// @param shape The tensor shape to look into
/// @param dim_name The name of the dimension to retrieve
/// @returns The value of the required dimension
/// @throws std::runtime_error if the requested dimension does not exist
int get_dim(const TensorShape &shape, const std::string &dim_name);

/// @brief Sets the value of the required dimension in the provided shape
/// @param shape The tensor shape to update
/// @param dim_name The name of the dimension to update
/// @param value The new dimension value
/// @throws std::runtime_error if the requested dimension does not exist
void set_dim(TensorShape &shape, const std::string &dim_name, int value);

/// @brief Sets all dynamic dimensions in the shape to 1
/// @param shape The tensor shape to update
void set_dynamic_dimensions_to_one(TensorShape &shape);

/// @brief Enumerate of the different base type that a tensor can contain
enum class BaseType : std::uint8_t {
    BOOL,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    INT8,
    INT16,
    INT32,
    INT64,
    FLOAT16,
    FLOAT32,
    FLOAT64
};

/// @brief Provides a string representing the type
std::string to_string(const BaseType &type);

/// @brief Retrieves the @ref Metavision::BaseType from a string
/// @details The string must have the same nomenclature as the BaseType name
BaseType from_string(const std::string name);

/// @brief Provides the number of bytes on which data of this type is coded
/// @param type The type to give the byte size of
/// @returns The number of bytes coding a value of the input type
size_t byte_size(const BaseType &type);

/// @brief Generic class to store information about an N-dimension tensor and its content
class Tensor {
public:
    /// @brief Default constructor, no memory allocation
    Tensor();

    /// @brief Constructor with no input data
    /// Memory is allocated with respect to provided shape ad data type. Tensor values are initialized to 0.
    /// @param shape The shape of the tensor
    /// @param type The type of data stored in the tensor
    Tensor(const TensorShape &shape, const BaseType &type);

    /// @brief Main constructor for the tensor class
    /// @param shape The shape of the tensor
    /// @param type The type of data stored in the tensor
    /// @param ptr Pointer to the tensor data (use reinterpret_cast if necessary). The ownership of the pointer is never
    /// transferred.
    /// @param copy Indicates whether or not to copy the tensor data. If false, the pointer will be used directly and
    /// assumed to be valid.
    Tensor(const TensorShape &shape, const BaseType &type, std::byte *ptr, bool copy = false);

    /// @brief Copy constructor
    /// @param other Instance to copy from
    Tensor(const Tensor &other);

    /// @brief Move constructor
    /// @param other instance to move from
    Tensor(Tensor &&other) noexcept;

    /// @brief Copy assignment operator
    /// @param other Instance to copy from
    Tensor &operator=(const Tensor &other);

    /// @brief Move assignment operator
    /// @param other instance to move from
    Tensor &operator=(Tensor &&other) noexcept;

    /// @brief Updates the tensor with new shape, type and data
    /// @param shape New shape of the tensor
    /// @param type New type of the tensor's data
    /// @param ptr Pointer to the new tensor's data
    /// @param copy If true, the data pointed by ptr will be copied internally. If the new required memory is
    /// equal or lower than the previous one, no memory is reallocated.
    void create(const TensorShape &shape, const BaseType &type, std::byte *ptr, bool copy = false);

    /// @brief Allocates memory to the tensor for a given shape and type of data
    /// The tensor values are set to 0 and can be set later on thanks to the @ref Tensor::data() accessor.
    /// @param shape New shape of the tensor
    /// @param type New type of the tensor's data
    void create(const TensorShape &shape, const BaseType &type);

    /// @brief Gets the shape of the tensor
    /// @returns The tensor shape
    TensorShape shape() const;

    /// @brief Gets the base type of the data stored in the tensor
    /// @returns The base type of the tensor's data
    BaseType type() const;

    /// @brief Returns the amount of bytes necessary to store the tensor data
    /// @returns The tensor's data size in bytes
    size_t byte_size() const;

    ///  @brief  Indicates  whether  the  tensor  contains  any  data
    ///  @returns  True  if  the  tensor  is  empty  (contains  0  values)
    bool empty();

    /// @brief Gets the tensor data as const variable
    /// @returns A const pointer to the tensor data
    template<typename T = std::byte>
    const T *data() const;

    /// @brief Gets the tensor data
    /// @returns A pointer to the tensor data
    template<typename T = std::byte>
    T *data();

    /// @brief Sets the tensors values to the provided one
    /// @tparam Type of the input value
    /// @param val The new value to set the tensor values to.
    template<typename T>
    void set_to(T val);

    /// @brief Swaps the data between two Tensors
    /// @param other The other tensor to swap data with
    void swap(Tensor &other) {
        std::swap(shape_, other.shape_);
        std::swap(type_, other.type_);
        std::swap(ptr_, other.ptr_);
        std::swap(data_, other.data_);
    }

private:
    template<typename T>
    void set_to_impl(T typed_val, size_t n);

    TensorShape shape_;           // Shape of the tensor
    BaseType type_;               // Type of the base data stored in the tensor
    std::byte *ptr_;              // Pointer to the tensor data
    std::vector<std::byte> data_; // Used to store the data when ownership is given in the constructor (copy = true)
};

} // namespace Metavision

#include "metavision/sdk/core/preprocessors/detail/tensor_impl.h"

namespace std {
std::istream &operator>>(std::istream &in, Metavision::TensorShape &tensor_shape);
std::istream &operator>>(std::istream &in, Metavision::BaseType &tensor_type);
std::ostream &operator<<(std::ostream &os, const Metavision::TensorShape &tensor_shape);
std::ostream &operator<<(std::ostream &os, const Metavision::BaseType &tensor_type);
} // namespace std

#endif // METAVISION_SDK_CORE_PREPROCESSORS_TENSOR_H
