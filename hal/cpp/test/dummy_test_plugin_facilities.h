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

#ifndef METAVISION_HAL_DUMMY_TEST_PLUGIN_FACILITIES_H
#define METAVISION_HAL_DUMMY_TEST_PLUGIN_FACILITIES_H

#include "metavision/hal/facilities/i_registrable_facility.h"

// Should not be added directly to dummy device, only here to be inherited by V2
class DummyFacilityV1 : public Metavision::I_RegistrableFacility<DummyFacilityV1> {};

// Should not be added directly to dummy device, only here to be inherited by V3
class DummyFacilityV2 : public Metavision::I_RegistrableFacility<DummyFacilityV2, DummyFacilityV1> {};

class DummyFacilityV3 : public Metavision::I_RegistrableFacility<DummyFacilityV3, DummyFacilityV2> {};

#endif // METAVISION_HAL_DUMMY_TEST_PLUGIN_FACILITIES_H
