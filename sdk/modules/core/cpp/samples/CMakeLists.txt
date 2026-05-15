# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

if (NOT ANDROID)
    add_subdirectory(metavision_composed_viewer)
    add_subdirectory(metavision_csv_viewer)
    add_subdirectory(metavision_dummy_radar)
    add_subdirectory(metavision_event_frame_generation)
    add_subdirectory(metavision_event_frame_gpu_loading)
    add_subdirectory(metavision_events_integration)
    add_subdirectory(metavision_file_to_video)
    add_subdirectory(metavision_filtering)
    add_subdirectory(metavision_sdk_get_started)
    add_subdirectory(metavision_time_surface)
endif (NOT ANDROID)
