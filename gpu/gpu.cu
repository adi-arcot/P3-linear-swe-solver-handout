#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "../common/common.hpp"
#include "../common/solver.hpp"

// Global variables for device memory
double *h, *u, *v;                     // Main fields
double *dh, *du, *dv;                  // Current derivatives
double *dh1, *du1, *dv1;               // Previous derivatives
double *dh2, *du2, *dv2;               // Second previous derivatives
int nx, ny;
double H, g, dx, dy, dt;
int t = 0;

// Combined kernel for all derivatives
__global__ void compute_all_derivatives_kernel(double *h, double *u, double *v,
                                             double *dh, double *du, double *dv,
                                             int nx, int ny, double H, double g, 
                                             double dx, double dy) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // faster moving index for coalescing
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        // Compute all derivatives at once
        dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
        du(i, j) = -g * dh_dx(i, j);
        dv(i, j) = -g * dh_dy(i, j);
    }
}

// Combined kernel for all ghost cells
__global__ void compute_all_ghost_cells_kernel(double *h, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handle horizontal ghost cells
    if (idx < ny) {
        h(nx, idx) = h(0, idx);
    }
    
    // Handle vertical ghost cells
    if (idx < nx) {
        h(idx, ny) = h(idx, 0);
    }
}

// Combined kernel for all boundaries
__global__ void compute_all_boundaries_kernel(double *u, double *v, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handle horizontal boundaries
    if (idx < ny) {
        u(0, idx) = u(nx, idx);
    }
    
    // Handle vertical boundaries
    if (idx < nx) {
        v(idx, 0) = v(idx, ny);
    }
}

// Multistep kernel remains mostly the same but with swapped indices
__global__ void multistep_kernel(double *h, double *u, double *v,
                                double *dh, double *du, double *dv,
                                double *dh1, double *du1, double *dv1,
                                double *dh2, double *du2, double *dv2,
                                int nx, int ny, double dt,
                                double a1, double a2, double a3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        // Update height field
        h(i, j) += (a1 * dh(i, j) + 
                    a2 * dh1(i, j) + 
                    a3 * dh2(i, j)) * dt;
        
        // Update u velocity field
        if (i < nx - 1) {
            u(i + 1, j) += (a1 * du(i, j) + 
                           a2 * du1(i, j) + 
                           a3 * du2(i, j)) * dt;
        }
        
        // Update v velocity field
        if (j < ny - 1) {
            v(i, j + 1) += (a1 * dv(i, j) + 
                           a2 * dv1(i, j) + 
                           a3 * dv2(i, j)) * dt;
        }
    }
}

void swap_buffers() {
    double *tmp;
    
    tmp = dh2;
    dh2 = dh1;
    dh1 = dh;
    dh = tmp;
    
    tmp = du2;
    du2 = du1;
    du1 = du;
    du = tmp;
    
    tmp = dv2;
    dv2 = dv1;
    dv1 = dv;
    dv = tmp;
}

void init(double *h0, double *u0, double *v0, double length_, double width_, 
          int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_) {
    nx = nx_;
    ny = ny_;
    H = H_;
    g = g_;
    dx = length_ / nx;
    dy = width_ / ny;
    dt = dt_;

    // Allocate device memory
    cudaMalloc(&h, (nx + 1) * (ny + 1) * sizeof(double));
    cudaMalloc(&u, (nx + 1) * ny * sizeof(double));
    cudaMalloc(&v, nx * (ny + 1) * sizeof(double));
    
    cudaMalloc(&dh, nx * ny * sizeof(double));
    cudaMalloc(&du, nx * ny * sizeof(double));
    cudaMalloc(&dv, nx * ny * sizeof(double));
    
    cudaMalloc(&dh1, nx * ny * sizeof(double));
    cudaMalloc(&du1, nx * ny * sizeof(double));
    cudaMalloc(&dv1, nx * ny * sizeof(double));
    
    cudaMalloc(&dh2, nx * ny * sizeof(double));
    cudaMalloc(&du2, nx * ny * sizeof(double));
    cudaMalloc(&dv2, nx * ny * sizeof(double));

    // Copy initial conditions to device
    cudaMemcpy(h, h0, (nx + 1) * (ny + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u, u0, (nx + 1) * ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v, v0, nx * (ny + 1) * sizeof(double), cudaMemcpyHostToDevice);
    
    // Initialize derivative arrays to zero
    cudaMemset(dh, 0, nx * ny * sizeof(double));
    cudaMemset(du, 0, nx * ny * sizeof(double));
    cudaMemset(dv, 0, nx * ny * sizeof(double));
    cudaMemset(dh1, 0, nx * ny * sizeof(double));
    cudaMemset(du1, 0, nx * ny * sizeof(double));
    cudaMemset(dv1, 0, nx * ny * sizeof(double));
    cudaMemset(dh2, 0, nx * ny * sizeof(double));
    cudaMemset(du2, 0, nx * ny * sizeof(double));
    cudaMemset(dv2, 0, nx * ny * sizeof(double));
}

void step() {
    dim3 blockDim(16, 16);
    dim3 gridDim((ny + blockDim.x - 1) / blockDim.x,  // Swapped nx/ny for coalescing
                 (nx + blockDim.y - 1) / blockDim.y);
    
    // Combined ghost cells kernel
    compute_all_ghost_cells_kernel<<<(max(nx,ny) + 255) / 256, 256>>>(h, nx, ny);
    cudaDeviceSynchronize();
    
    // Combined derivatives kernel
    compute_all_derivatives_kernel<<<gridDim, blockDim>>>(h, u, v, dh, du, dv, 
                                                         nx, ny, H, g, dx, dy);
    cudaDeviceSynchronize();
    
    // Set multistep coefficients
    double a1, a2 = 0.0, a3 = 0.0;
    if (t == 0) {
        a1 = 1.0;
    } else if (t == 1) {
        a1 = 3.0 / 2.0;
        a2 = -1.0 / 2.0;
    } else {
        a1 = 23.0 / 12.0;
        a2 = -16.0 / 12.0;
        a3 = 5.0 / 12.0;
    }
    
    // Multistep kernel
    multistep_kernel<<<gridDim, blockDim>>>(h, u, v, 
                                           dh, du, dv,
                                           dh1, du1, dv1,
                                           dh2, du2, dv2,
                                           nx, ny, dt, a1, a2, a3);
    cudaDeviceSynchronize();
    
    // Combined boundaries kernel
    compute_all_boundaries_kernel<<<(max(nx,ny) + 255) / 256, 256>>>(u, v, nx, ny);
    cudaDeviceSynchronize();
    
    swap_buffers();
    t++;
}

void transfer(double *h_host) {
    cudaMemcpy(h_host, h, (nx + 1) * (ny + 1) * sizeof(double), cudaMemcpyDeviceToHost);
}

void free_memory() {
    cudaFree(h);
    cudaFree(u);
    cudaFree(v);
    cudaFree(dh);
    cudaFree(du);
    cudaFree(dv);
    cudaFree(dh1);
    cudaFree(du1);
    cudaFree(dv1);
    cudaFree(dh2);
    cudaFree(du2);
    cudaFree(dv2);
}