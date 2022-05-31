# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

##########################################################
#
#    Metavision SDK core ml - debian packages information
#

# File and package name of the components are automatically set, just need to set the package description
# and potential dependencies

# Python Development package (Open)
set(CPACK_COMPONENT_METAVISION-SDK-CORE-ML-PYTHON_DESCRIPTION "Metavision SDK CORE ML Python Modules.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-CORE-ML-PYTHON_DEPENDS metavision-sdk-base-python${PYTHON3_DEFAULT_VERSION} metavision-sdk-core-python${PYTHON3_DEFAULT_VERSION})

# Python samples of metavision-sdk-core-ml-python
set(CPACK_COMPONENT_METAVISION-SDK-CORE-ML-PYTHON-SAMPLES_DESCRIPTION "Samples for Metavision SDK CORE ML Python library.\n${PACKAGE_LICENSE}")
set(CPACK_COMPONENT_METAVISION-SDK-CORE-ML-PYTHON-SAMPLES_DEPENDS metavision-sdk-core-python${PYTHON3_DEFAULT_VERSION} metavision-sdk-core-ml-python)
