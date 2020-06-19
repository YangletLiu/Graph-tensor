# cuGraph-Tensor
cuGraph-Tensor library implements eight key graph-tensor operations on top of CUDA libraries including cuBLAS, cuSolver, and existing libraries including Magma and KBLAS. We encapsulate these operations into an opensource library and provide BLAS-like interfaces for ease of use. In addition, cuGraph-Tensor builds a graph data completiona pplicaitonf or fast and accurate reconstruction of incomplete graph data.

## References
[1] Tao Zhang, Xiao-Yang Liu, Xiaodong Wang, and Anwar Walid, “cuTensor-Tubal: Efficient primitives for tubal-rank tensor learning operations on GPUs,” IEEE Transactions on Parallel and Distributed Systems, vol. 31, no. 3, pp. 595–610, 2020.

[2] Tao Zhang and Xiao-Yang Liu, “cuTensor-tubal: Optimized GPU library for low-tubal-rank tensors,” in 44th IEEE Int. Conf. Acoust., Speech and Signal Proces. IEEE, 2019, pp. 8583–8587.

[3] Tao Zhang, Xiao-Yang Liu, and Xiaodong Wang, “High performance GPU tensor completion with tubal-sampling pattern,” IEEE Trans. Parallel Distrib. Syst., vol. 31, no. 7, pp. 1724–1739, 2020.

[4] Xiao-Yang Liu, Shuchin Aeron, Vaneet Aggarwal, Xiaodong Wang. Low-tubal-rank tensor completion using alternating minimization. IEEE Transactions on Information Theory, 2019.

## Graph-tensor Operations
graph shift (g-shift), graph Fourier transform (g-Ft), inverse graph Fourier transform (inverse g-Ft), graph filter (g-filter), graph convolution (g-convolution), graph-tensor product (g-product), graph-tensor SVD (g-SVD) and graph-tensor QR (g-QR)

## Application
The Tubal-Alter-Min algorithm [4] was proposed for robust data completion based on the low-tubal-rank tensor model. However, this algorithm is compute-intensive and its running time increases exponentially with the growing of tensor size and dimension. We exploit the high performance graph-tensor operartions in the cuGraph-Tensor library to solve this limitation.

## Results
Comparing with CPU-based GSPBOX and CPU MATLAB implementations running on two Xeon CPUs, g-shift, g-FT, inverse g-FT, g-filter, g-convolution, g-product, g-SVD and g-QR achieve up to 133.98X, 96.09X, 90.14X, 12.64X, 130.61X, 141.71X, 51.18X and 142.12X speedups, respectively (on average 35.27X, 18.54X, 18.12X, 4.47X, 38.16X, 27.60X, 7.91X and 23.83X faster over the GPU baseline implementation).

The graph data completion application achieves up to 174.38X speedup over the CPU MATLAB implementation, and up to 3.82X speedup with better accuracy over the GPU-based tensor completion in the cuTensor-tubal library.

## Folder directory
### CPU/Test_Reconstruction
This directory includes graph-tensor operation and graph-tensor resconstruction. test_reconstruction_real.m test the CPark video. test_reconstruction_simulataion.m test the walking data. test_reconstruction_simulataion.m test the simulation data.

### GPU
include: some header file (.h), define some interfaces

operation: implemente interface and implement graph-tensor operations

bin: generated .o file

lib: compressed library .a file

app: graph-tensor completion applications

test: each graph-tensor operation have a separate folder. For example, g-FT folder includes test_gFT.cpp and debug folder, that includes makefile and script to run the program

## Build & Run
On linux

```
cd Graph-tensor
make
#去对应的操作下,如g-FT/debug下make
cd /GPU/test/g-FT/debug/
make
#直接执行运行脚本
./run.sh

```

'''
