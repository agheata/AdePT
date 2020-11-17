# Link specific libraries

# library Raytracer.cu
add_library(Raytracer_cuda Raytracer.cu RaytraceBenchmark.cu)
target_link_libraries(Raytracer_cuda VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda_static)
target_compile_options(Raytracer_cuda PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(Raytracer_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#library RaytraceBenchmark.cpp
add_executable(RaytraceBenchmark Raytracer.cpp RaytraceBenchmark.cpp)
target_link_libraries(RaytraceBenchmark VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml Raytracer_cuda)
target_compile_options(RaytraceBenchmark PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(RaytraceBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
