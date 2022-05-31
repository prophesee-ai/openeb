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

#ifndef METAVISION_HAL_CCAM_GEN31_HELPERS_H
#define METAVISION_HAL_CCAM_GEN31_HELPERS_H
#define GEN31_WRITE_REGISTER(cmd, address, value)                \
    do {                                                         \
        if (!!getenv("LOG_REGISTERS")) {                         \
            printf("write %s -> 0x%x\n", #address, value);       \
            printf("      @0x%04x -> 0x%x\n\n", address, value); \
        }                                                        \
        cmd.write_register(address, value);                      \
    } while (0);

#define GEN31_SYSTEM_WRITE_REGISTER(cmd, address, value)                         \
    do {                                                                         \
        if (!!getenv("LOG_REGISTERS")) {                                         \
            printf("write %s -> 0x%x\n", #address, value);                       \
            printf("      @0x%04x -> 0x%x\n\n", base_address_ + address, value); \
        }                                                                        \
        cmd.write_register(base_address_ + address, value);                      \
    } while (0);

#define GEN31_SENSOR_WRITE_REGISTER(cmd, address, value)                                \
    do {                                                                                \
        if (!!getenv("LOG_REGISTERS")) {                                                \
            printf("write %s -> 0x%x\n", #address, value);                              \
            printf("      @0x%04x -> 0x%x\n\n", base_sensor_address_ + address, value); \
        }                                                                               \
        cmd.write_register(base_sensor_address_ + address, value);                      \
    } while (0);

#define GEN31_SEND_REGISTER_BIT(cmd, address, bit, value)                    \
    do {                                                                     \
        if (!!getenv("LOG_REGISTERS")) {                                     \
            printf("send  %s/%s -> 0x%x\n", #address, #bit, value);          \
            printf("      @0x%04x bit %d -> 0x%x\n\n", address, bit, value); \
        }                                                                    \
        cmd.send_register_bit(address, bit, value);                          \
    } while (0);

#define GEN31_SYSTEM_SEND_REGISTER_BIT(cmd, address, bit, value)                             \
    do {                                                                                     \
        if (!!getenv("LOG_REGISTERS")) {                                                     \
            printf("send  %s/%s -> 0x%x\n", #address, #bit, value);                          \
            printf("      @0x%04x bit %d -> 0x%x\n\n", base_address_ + address, bit, value); \
        }                                                                                    \
        cmd.send_register_bit(base_address_ + address, bit, value);                          \
    } while (0);

#define GEN31_SENSOR_SEND_REGISTER_BIT(cmd, address, bit, value)                                    \
    do {                                                                                            \
        if (!!getenv("LOG_REGISTERS")) {                                                            \
            printf("send  %s/%s -> 0x%x\n", #address, #bit, value);                                 \
            printf("      @0x%04x bit %d -> 0x%x\n\n", base_sensor_address_ + address, bit, value); \
        }                                                                                           \
        cmd.send_register_bit(base_sensor_address_ + address, bit, value);                          \
    } while (0);

#define GEN31_SET_REGISTER_BIT(cmd, address, bit, value)                     \
    do {                                                                     \
        if (!!getenv("LOG_REGISTERS")) {                                     \
            printf("set   %s/%s -> 0x%x\n", #address, #bit, value);          \
            printf("      @0x%04x bit %d -> 0x%x\n\n", address, bit, value); \
        }                                                                    \
        cmd.set_register_bit(address, bit, value);                           \
    } while (0);

#define GEN31_SYSTEM_SET_REGISTER_BIT(cmd, address, bit, value)                              \
    do {                                                                                     \
        if (!!getenv("LOG_REGISTERS")) {                                                     \
            printf("set   %s/%s -> 0x%x\n", #address, #bit, value);                          \
            printf("      @0x%04x bit %d -> 0x%x\n\n", base_address_ + address, bit, value); \
        }                                                                                    \
        cmd.set_register_bit(base_address_ + address, bit, value);                           \
    } while (0);

#define GEN31_SENSOR_SET_REGISTER_BIT(cmd, address, bit, value)                                     \
    do {                                                                                            \
        if (!!getenv("LOG_REGISTERS")) {                                                            \
            printf("set   %s/%s -> 0x%x\n", #address, #bit, value);                                 \
            printf("      @0x%04x bit %d -> 0x%x\n\n", base_sensor_address_ + address, bit, value); \
        }                                                                                           \
        cmd.set_register_bit(base_sensor_address_ + address, bit, value);                           \
    } while (0);

#define GEN31_SEND_REGISTER(cmd, address)                  \
    do {                                                   \
        if (!!getenv("LOG_REGISTERS"))                     \
            printf("send  %s mirrored value\n", #address); \
        cmd.send_register(address);                        \
    } while (0);

#define GEN31_SYSTEM_SEND_REGISTER(cmd, address)           \
    do {                                                   \
        if (!!getenv("LOG_REGISTERS"))                     \
            printf("send  %s mirrored value\n", #address); \
        cmd.send_register(base_address_ + address);        \
    } while (0);

#define GEN31_SENSOR_SEND_REGISTER(cmd, address)           \
    do {                                                   \
        if (!!getenv("LOG_REGISTERS"))                     \
            printf("send  %s mirrored value\n", #address); \
        cmd.send_register(base_sensor_address_ + address); \
    } while (0);

#endif // METAVISION_HAL_CCAM_GEN31_HELPERS_H
