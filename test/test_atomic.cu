// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_atomic.cu
 * @brief Unit test for atomic operations.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <iostream>
#include <cassert>
#include <AdePT/Atomic.h>

#include "test_atomic.h"

///______________________________________________________________________________________
int main(void)
{
  const char *result[2] = {"FAILED", "OK"};
  bool success          = true;
  // Define the kernels granularity: 10K blocks of 32 treads each
  dim3 nblocks(10000), nthreads(32);

  // Allocate the content of SomeStruct in a buffer
  char *buffer = nullptr;
  cudaMallocManaged((void**)&buffer, sizeof(SomeStruct));
  SomeStruct *a = SomeStruct::MakeInstanceAt(buffer);

  // Launch a kernel doing additions
  bool testOK = true;
  std::cout << "   testAdd ... ";
  // Wait memory to reach device
  cudaDeviceSynchronize();
  #pragma omp parallel for collapse(2)
  COPCORE_KERNEL(nblocks.x, nthreads.x, testAdd, a);
  // Wait all warps to finish and sync memory
  cudaDeviceSynchronize();

  testOK &= a->var_int.load() == nblocks.x * nthreads.x;
  testOK &= a->var_float.load() == float(nblocks.x * nthreads.x);
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Launch a kernel doing subtraction
  testOK = true;
  std::cout << "   testSub ... ";
  a->var_int.store(nblocks.x * nthreads.x);
  a->var_float.store(nblocks.x * nthreads.x);
  cudaDeviceSynchronize();
  #pragma omp parallel for collapse(2)
  COPCORE_KERNEL(nblocks.x, nthreads.x, testSub, a);
  cudaDeviceSynchronize();

  testOK &= a->var_int.load() == 0;
  testOK &= a->var_float.load() == 0;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Launch a kernel testing compare and swap operations
  std::cout << "   testCAS ... ";
  a->var_int.store(99);
  cudaDeviceSynchronize();
  #pragma omp parallel for collapse(2)
  COPCORE_KERNEL(nblocks.x, nthreads.x, testCompareExchange, a);
  cudaDeviceSynchronize();
  testOK = a->var_int.load() == 99;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  cudaFree(buffer);
  if (!success) return 1;
  return 0;
}
