# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

project(metavision_event_frame_gpu_loading)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS base core driver REQUIRED)
find_package(Boost COMPONENTS program_options filesystem REQUIRED)
find_package(OpenCV COMPONENTS core imgproc imgcodecs REQUIRED)

enable_language(CUDA)

# Build CUDA kernels
set(cuda_source_files convert_event_frame.cu convert_event_frame.h)
add_library(cuda_event_frame_convert STATIC ${cuda_source_files})
set_target_properties(cuda_event_frame_convert PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(cuda_source_files convert_event_frame.cu convert_event_frame.h)
set (sample metavision_filtering)

set (sample metavision_event_frame_gpu_loading)
set(source_files ${sample}.cpp)
add_executable(${sample} ${source_files})
set_property(TARGET ${sample} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${sample} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(${sample} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${sample}
    Boost::filesystem
    Boost::program_options
    "${OpenCV_LIBS}"
    MetavisionSDK::base
    MetavisionSDK::core
    MetavisionSDK::driver
    cuda_event_frame_convert
)
