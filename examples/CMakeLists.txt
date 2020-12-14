# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(FisherPrice_Alpaka)
add_subdirectory(FisherPrice_cpp)
if(COPCORE_ENABLE_CUDA)
  add_subdirectory(FisherPrice_cu)
endif()
#add_subdirectory(Raytracer_Benchmark)
add_subdirectory(Example1)
#add_subdirectory(Example2)
