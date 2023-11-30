%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for square matrix multiplication
__global__ void MatrixMulSquareKernel(float* matrixA, float* matrixB, float* resultMatrix, int width) {
  // Thread indices
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if within matrix dimensions
  if ((row < width) && (col < width)) {
    float result = 0;
    for (int k = 0; k < width; ++k) {
      // Matrix multiplication
      result += matrixA[row * width + k] * matrixB[k * width + col];
    }
    resultMatrix[row * width + col] = result;
  }
}

// CUDA kernel for rectangular matrix multiplication
__global__ void MatrixMulRectangularKernel(float* matrixA, float* matrixB, float* resultMatrix,
                                           int widthA, int heightA, int widthB, int heightB) {
  // Thread indices
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if within matrix dimensions
  if ((row < heightA) && (col < widthB)) {
    float result = 0;
    for (int k = 0; k < widthA; ++k) {
      // Matrix multiplication
      result += matrixA[row * widthA + k] * matrixB[k * widthB + col];
    }
    resultMatrix[row * widthB + col] = result;
  }
}

int main() {
  // Matrix dimensions
  int widthA = 2048;
  int heightA = 1024;
  int widthB = 1024;
  int heightB = 2048;

  // Calculate matrix sizes
  size_t sizeA = widthA * heightA * sizeof(float);
  size_t sizeB = widthB * heightB * sizeof(float);
  size_t sizeResult;

  // Host matrices
  float *hostMatrixA, *hostMatrixB, *hostResultMatrix;
  hostMatrixA = (float*)malloc(sizeA);
  hostMatrixB = (float*)malloc(sizeB);
  hostResultMatrix = nullptr;

  // Device matrices
  float *deviceMatrixA, *deviceMatrixB, *deviceResultMatrix;
  cudaMalloc((void**)&deviceMatrixA, sizeA);
  cudaMalloc((void**)&deviceMatrixB, sizeB);

  // Copy data from host to device
  cudaMemcpy(deviceMatrixA, hostMatrixA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMatrixB, hostMatrixB, sizeB, cudaMemcpyHostToDevice);

  // Define CUDA grid and block dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim;

  // Check if matrices are square for optimal kernel selection
  if (widthA == heightA && widthB == heightB) {
    sizeResult = widthA * heightB * sizeof(float);
    hostResultMatrix = (float*)malloc(sizeResult);
    cudaMalloc((void**)&deviceResultMatrix, sizeResult);

    // Calculate grid dimensions for square matrices
    gridDim = dim3((widthA + blockDim.x - 1) / blockDim.x, (heightB + blockDim.y - 1) / blockDim.y);

    // Launch square matrix multiplication kernel
    MatrixMulSquareKernel<<<gridDim, blockDim>>>(deviceMatrixA, deviceMatrixB, deviceResultMatrix, widthA);

    // Copy result from device to host
    cudaMemcpy(hostResultMatrix, deviceResultMatrix, sizeResult, cudaMemcpyDeviceToHost);

    // Free device result matrix
    cudaFree(deviceResultMatrix);
  } else {
    sizeResult = heightA * widthB * sizeof(float);
    hostResultMatrix = (float*)malloc(sizeResult);
    cudaMalloc((void**)&deviceResultMatrix, sizeResult);

    // Calculate grid dimensions for rectangular matrices
    gridDim = dim3((widthB + blockDim.x - 1) / blockDim.x, (heightA + blockDim.y - 1) / blockDim.y);

    // Launch rectangular matrix multiplication kernel
    MatrixMulRectangularKernel<<<gridDim, blockDim>>>(deviceMatrixA, deviceMatrixB, deviceResultMatrix,
                                                      widthA, heightA, widthB, heightB);

    // Copy result from device to host
    cudaMemcpy(hostResultMatrix, deviceResultMatrix, sizeResult, cudaMemcpyDeviceToHost);

    // Free device result matrix
    cudaFree(deviceResultMatrix);
  }

  // Timing variables
  clock_t start, stop;
  start = clock();

  // Launch matrix multiplication kernel based on matrix type
  if (widthA == heightA && widthB == heightB) {
    MatrixMulSquareKernel<<<gridDim, blockDim>>>(deviceMatrixA, deviceMatrixB, deviceResultMatrix, widthA);
  } else {
    MatrixMulRectangularKernel<<<gridDim, blockDim>>>(deviceMatrixA, deviceMatrixB, deviceResultMatrix,
                                                      widthA, heightA, widthB, heightB);
  }

  // Synchronize device and calculate execution time
  cudaDeviceSynchronize();
  stop = clock();
  float milliseconds = ((float)(stop - start) / CLOCKS_PER_SEC);
  printf("Execution Time: %.4f seconds\n", milliseconds);

  // Free host memory
  free(hostMatrixA);
  free(hostMatrixB);
  free(hostResultMatrix);

  // Free device memory
  cudaFree(deviceMatrixA);
  cudaFree(deviceMatrixB);

  return 0;
}
