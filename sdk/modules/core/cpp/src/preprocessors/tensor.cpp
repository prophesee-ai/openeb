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

#include <cstring>
#include <sstream>
#include <utility>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/preprocessors/tensor.h"

namespace Metavision {

static std::unordered_map<BaseType, size_t> const baseTypeToByteSizeMap{
    {BaseType::BOOL, 1},   {BaseType::UINT8, 1},   {BaseType::UINT16, 2},  {BaseType::UINT32, 4},
    {BaseType::UINT64, 8}, {BaseType::INT8, 1},    {BaseType::INT16, 2},   {BaseType::INT32, 4},
    {BaseType::INT64, 8},  {BaseType::FLOAT16, 2}, {BaseType::FLOAT32, 4}, {BaseType::FLOAT64, 8}};

static std::unordered_map<BaseType, std::string> const baseTypeToStringMap{
    {BaseType::BOOL, "BOOL"},       {BaseType::UINT8, "UINT8"},     {BaseType::UINT16, "UINT16"},
    {BaseType::UINT32, "UINT32"},   {BaseType::UINT64, "UINT64"},   {BaseType::INT8, "INT8"},
    {BaseType::INT16, "INT16"},     {BaseType::INT32, "INT32"},     {BaseType::INT64, "INT64"},
    {BaseType::FLOAT16, "FLOAT16"}, {BaseType::FLOAT32, "FLOAT32"}, {BaseType::FLOAT64, "FLOAT64"}};

static std::unordered_map<std::string, BaseType> const stringToBaseTypeMap{
    {"BOOL", BaseType::BOOL},       {"UINT8", BaseType::UINT8},     {"UINT16", BaseType::UINT16},
    {"UINT32", BaseType::UINT32},   {"UINT64", BaseType::UINT64},   {"INT8", BaseType::INT8},
    {"INT16", BaseType::INT16},     {"INT32", BaseType::INT32},     {"INT64", BaseType::INT64},
    {"FLOAT16", BaseType::FLOAT16}, {"FLOAT32", BaseType::FLOAT32}, {"FLOAT64", BaseType::FLOAT64}};

TensorShape::TensorShape() {}

TensorShape::TensorShape(const std::vector<Dimension> &dimensions) : dimensions(dimensions) {}

bool TensorShape::operator==(const TensorShape &other) const {
    const auto &v1       = dimensions;
    const auto &v2       = other.dimensions;
    const size_t n       = v1.size();
    const bool equal_dim = n == v2.size();
    if (!equal_dim)
        return false;

    for (size_t i = 0; i < n; ++i)
        if (v1[i].dim != v2[i].dim || v1[i].name != v2[i].name)
            return false;
    return true;
}

bool TensorShape::operator!=(const TensorShape &other) const {
    return !(*this == other);
}

size_t TensorShape::get_nb_values() const {
    int nb_values = 1;
    for (auto k : dimensions)
        nb_values *= k.dim;
    return static_cast<size_t>(std::abs(nb_values));
}

bool TensorShape::matches(const TensorShape &other) const {
    const auto &v1       = dimensions;
    const auto &v2       = other.dimensions;
    const size_t n       = v1.size();
    const bool equal_dim = n == v2.size();
    if (!equal_dim)
        return false;

    for (size_t i = 0; i < n; ++i)
        if ((v1[i].dim != v2[i].dim) && (v1[i].dim != -1) && (v2[i].dim != -1))
            return false;

    return true;
}

bool TensorShape::is_valid() const {
    for (auto k : dimensions)
        if (k.dim <= 0)
            return false;
    return true;
}

int get_dim(const Metavision::TensorShape &shape, const std::string &dim_name) {
    for (const auto &d : shape.dimensions) {
        if (d.name == dim_name)
            return d.dim;
    }
    std::stringstream msg;
    msg << "Didn't find dimension " << dim_name << " in provided tensor shape " << shape << std::endl;
    throw std::runtime_error(msg.str());
}

void set_dim(Metavision::TensorShape &shape, const std::string &dim_name, int value) {
    for (auto &d : shape.dimensions) {
        if (d.name == dim_name) {
            d.dim = value;
            return;
        }
    }
    std::stringstream msg;
    msg << "Couldn't find dimension " << dim_name << " to set in provided tensor shape " << shape;
    MV_SDK_LOG_WARNING() << msg.str();
}

void set_dynamic_dimensions_to_one(Metavision::TensorShape &shape) {
    for (auto &d : shape.dimensions) {
        if (d.dim == Dimension::kDynamic)
            d.dim = 1;
    }
}

std::string to_string(const BaseType &type) {
    return baseTypeToStringMap.at(type);
}

BaseType from_string(const std::string name) {
    return stringToBaseTypeMap.at(name);
}

size_t byte_size(const BaseType &type) {
    return baseTypeToByteSizeMap.at(type);
}

Tensor::Tensor() : shape_(), type_(BaseType::BOOL), ptr_(nullptr) {}

Tensor::Tensor(const TensorShape &shape, const BaseType &type) {
    create(shape, type);
}

Tensor::Tensor(const TensorShape &shape, const BaseType &type, std::byte *ptr, bool copy) {
    create(shape, type, ptr, copy);
}

Tensor::Tensor(const Tensor &other) : shape_(other.shape_), type_(other.type_) {
    data_ = other.data_;
    ptr_  = data_.data();
}

Tensor::Tensor(Tensor &&other) noexcept :
    shape_(std::move(other.shape_)),
    type_(std::move(other.type_)),
    data_(std::move(other.data_)),
    ptr_(std::exchange(other.ptr_, nullptr)) {}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        shape_ = other.shape_;
        type_  = other.type_;
        data_  = other.data_;
        ptr_   = data_.data();
    }
    return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        type_  = std::move(other.type_);
        ptr_   = std::exchange(other.ptr_, nullptr);
        data_  = std::move(other.data_);
    }
    return *this;
}

void Tensor::create(const TensorShape &shape, const BaseType &type, std::byte *ptr, bool copy) {
    shape_ = shape;
    type_  = type;
    if (copy && shape_.is_valid()) {
        const size_t nb_val           = shape.get_nb_values();
        const size_t data_byte_length = Metavision::byte_size(type);
        const size_t nb_val_byte      = nb_val * data_byte_length;
        data_.resize(nb_val_byte);
        if (ptr)
            std::copy(ptr, ptr + nb_val_byte, data_.data());
        else
            memset(&data_[0], 0, nb_val_byte);
        ptr_ = data_.data();
    } else {
        MV_LOG_DEBUG() << "Tensor memory not allocated";
        data_.clear();
        ptr_ = ptr;
    }
}

void Tensor::create(const TensorShape &shape, const BaseType &type) {
    create(shape, type, nullptr, true);
}

TensorShape Tensor::shape() const {
    return shape_;
}

BaseType Tensor::type() const {
    return type_;
}

size_t Tensor::byte_size() const {
    return shape_.get_nb_values() * baseTypeToByteSizeMap.at(type_);
}

bool Tensor::empty() {
    return shape_.get_nb_values() == 0;
}

} // namespace Metavision

namespace std {

std::istream &operator>>(std::istream &in, Metavision::TensorShape &tensor_shape) {
    std::string tensor_shape_str;
    std::getline(in, tensor_shape_str);

    // Removes preceding and end brackets
    tensor_shape_str = tensor_shape_str.substr(1, tensor_shape_str.size() - 2);

    if (!tensor_shape_str.empty())
        tensor_shape = {};

    auto get_dimension = [](const std::string &s) {
        const std::string name = s.substr(0, s.find(":"));
        const int dim          = stoi(s.substr(s.find(":") + 1, s.length()));
        return Metavision::Dimension{name, dim};
    };

    // Loop on the string to find dimensions 'name:dim' separated by ', ' delimiter
    size_t pos = 0;
    std::string token;
    std::vector<Metavision::Dimension> dims;
    while ((pos = tensor_shape_str.find(", ")) != std::string::npos) {
        token = tensor_shape_str.substr(0, pos);
        dims.emplace_back(get_dimension(token));
        tensor_shape_str.erase(0, pos + 2);
    }

    // Read last dimension
    if (!tensor_shape_str.empty()) {
        token = tensor_shape_str;
        std::cout << token << std::endl;
        dims.emplace_back(get_dimension(token));
    }

    tensor_shape = {dims};

    return in;
}

std::istream &operator>>(std::istream &in, Metavision::BaseType &base_type) {
    std::string base_type_str;
    in >> base_type_str;

    if (Metavision::stringToBaseTypeMap.count(base_type_str) == 0)
        throw std::runtime_error("Unknown BaseType " + base_type_str);

    base_type = Metavision::stringToBaseTypeMap.at(base_type_str);

    return in;
}

std::ostream &operator<<(std::ostream &os, const Metavision::TensorShape &tensor_shape) {
    std::ostringstream oss;
    oss << "[";
    const auto &v = tensor_shape.dimensions;
    for (size_t i = 0; i + 1 < v.size(); ++i)
        oss << v[i].name << ":" << std::to_string(v[i].dim) << ", ";
    if (v.size() > 0)
        oss << v.back().name << ":" << std::to_string(v.back().dim);
    oss << "]";
    os << oss.str();
    return os;
}

std::ostream &operator<<(std::ostream &os, const Metavision::BaseType &tensor_type) {
    os << Metavision::baseTypeToStringMap.at(tensor_type);
    return os;
}

} // namespace std
