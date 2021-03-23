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

#ifndef METAVISION_HAL_PYTHON_BINDINGS_DOC_HAL_H
#define METAVISION_HAL_PYTHON_BINDINGS_DOC_HAL_H

#include <string>
#include "metavision/sdk/base/utils/python_bindings_doc.h"

namespace Metavision {
#ifdef GENERATE_DOC_PYTHON_BINDINGS_USING_CPP_COMMENTS
#include "python_doc_strings.hal.hpp"
static PythonBindingsDoc pybind_doc_hal(Metavision::PythonDoc::python_doc_strings_hal);
#else
static PythonBindingsDoc pybind_doc_hal;
#endif
} // namespace Metavision

#endif // METAVISION_HAL_PYTHON_BINDINGS_DOC_HAL_H
