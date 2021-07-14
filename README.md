# VicsekModel
Molecular dynamics simulation of Vicsek model in 2 dimensional lattice with CUDA

1) how to compile
   $ nvcc -O2 cudaVicsek.cu -lcurand_static -lculibos -lm
   (both cudaVicsek.cu and subVicsek.c files should be in the same directory)
2) how to run
   $ ./a.out 1024 1000 10
   It requires three command line arguments L = 1024,  tmax = 1000, dt = 10
3) output
   It prints out "L noise t Vx Vy"
