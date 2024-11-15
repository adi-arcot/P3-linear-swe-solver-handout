#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../common/common.hpp"
#include "../common/solver.hpp"

using namespace std;

// Global variables
int nx, ny;
double *h, *u, *v;
double *dh, *du, *dv;
double *dh1, *du1, *dv1;
double *dh2, *du2, *dv2;
double H, g, dx, dy, dt;
int t = 0;


void compute_all_boundaries() { 
    // Combine all boundary computations into one loop
    // Horizontal boundaries
    for (int j = 0; j < ny; j++) {
        h(nx, j) = h(0, j);    // Ghost cells
        u(0, j) = u(nx, j);    // Boundaries
    }
     
    // Vertical boundaries
    for (int i = 0; i < nx; i++) {
        h(i, ny) = h(i, 0);    // Ghost cells
        v(i, 0) = v(i, ny);    // Boundaries
    }
}

void compute_derivatives() {
    const int L2_TILE = 128;  // Larger tile for L2 cache
    const int L1_TILE = 64;   // Smaller tile for L1 cache
    
    for (int it2 = 0; it2 < nx; it2 += L2_TILE) {
        for (int jt2 = 0; jt2 < ny; jt2 += L2_TILE) {
            // L2 cache level
            int iend2 = min(it2 + L2_TILE, nx);
            int jend2 = min(jt2 + L2_TILE, ny);
            
            // L1 cache level
            for (int it1 = it2; it1 < iend2; it1 += L1_TILE) {
                for (int jt1 = jt2; jt1 < jend2; jt1 += L1_TILE) {
                    int iend1 = min(it1 + L1_TILE, iend2);
                    int jend1 = min(jt1 + L1_TILE, jend2);
                    
                    // Process inner tile
                    for (int i = it1; i < iend1; i++) {
                        for (int j = jt1; j < jend1; j++) {
                            dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
                            du(i, j) = -g * dh_dx(i, j);
                            dv(i, j) = -g * dh_dy(i, j);
                        }
                    }
                }
            }
        }
    }
}

void multistep(double a1, double a2, double a3)
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j) + a3 * dh2(i, j)) * dt;
            u(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j) + a3 * du2(i, j)) * dt;
            v(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j) + a3 * dv2(i, j)) * dt;
        }
    }
}


void swap_buffers() {
    double *tmp;
    
    tmp = dh2; dh2 = dh1; dh1 = dh; dh = tmp;
    tmp = du2; du2 = du1; du1 = du; du = tmp;
    tmp = dv2; dv2 = dv1; dv1 = dv; dv = tmp;
}

void init(double *h0, double *u0, double *v0, double length_, double width_, 
          int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_) {
    // Set global parameters
    nx = nx_;
    ny = ny_;
    H = H_;
    g = g_;
    dx = length_ / nx;
    dy = width_ / ny;
    dt = dt_;
    
    // Set main field pointers
    h = h0;
    u = u0;
    v = v0;
    
    // Allocate memory for derivatives
    dh = (double *)calloc(nx * ny, sizeof(double));
    du = (double *)calloc(nx * ny, sizeof(double));
    dv = (double *)calloc(nx * ny, sizeof(double));
    
    dh1 = (double *)calloc(nx * ny, sizeof(double));
    du1 = (double *)calloc(nx * ny, sizeof(double));
    dv1 = (double *)calloc(nx * ny, sizeof(double));
    
    dh2 = (double *)calloc(nx * ny, sizeof(double));
    du2 = (double *)calloc(nx * ny, sizeof(double));
    dv2 = (double *)calloc(nx * ny, sizeof(double));
}

void step() {
    // Compute all boundaries in one go
    compute_all_boundaries();
    compute_derivatives();
    
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
    
    
    multistep(a1, a2, a3);
     
    // Swap buffers and increment time
    swap_buffers();
    t++;
}

void transfer(double *h_out) {
    return;  // No transfer needed in serial version
}

void free_memory() {
    free(dh);
    free(du);
    free(dv);
    free(dh1);
    free(du1);
    free(dv1);
    free(dh2);
    free(du2);
    free(dv2);
}