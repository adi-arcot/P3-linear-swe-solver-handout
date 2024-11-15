#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

// Here we hold the number of cells we have in the x and y directions
int nx, ny;

// This is where all of our points are. We need to keep track of our active
// height and velocity grids, but also the corresponding derivatives. The reason
// we have 2 copies for each derivative is that our multistep method uses the
// derivative from the last 2 time steps.
double *h, *u, *v, *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;
double H, g, dx, dy, dt;

void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{ 
    // We set the pointers to the arrays that were passed in
    h = h0;
    u = u0;
    v = v0;

    nx = nx_;
    ny = ny_;

    // We allocate memory for the derivatives
    dh = (double *)calloc(nx * ny, sizeof(double));
    du = (double *)calloc(nx * ny, sizeof(double));
    dv = (double *)calloc(nx * ny, sizeof(double));

    dh1 = (double *)calloc(nx * ny, sizeof(double));
    du1 = (double *)calloc(nx * ny, sizeof(double));
    dv1 = (double *)calloc(nx * ny, sizeof(double));

    dh2 = (double *)calloc(nx * ny, sizeof(double));
    du2 = (double *)calloc(nx * ny, sizeof(double));
    dv2 = (double *)calloc(nx * ny, sizeof(double));

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / ny;

    dt = dt_;
}

/**
 * This function computes the derivative of the height field
 * with respect to time. This is done by taking the divergence
 * of the velocity field and multiplying by -H.
 */
void compute_dh()
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
        }
    }
}

/**
 * This function computes the derivative of the x-component of the
 * velocity field with respect to time. This is done by taking the
 * derivative of the height field with respect to x and multiplying
 * by -g.
 */
void compute_du()
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            du(i, j) = -g * dh_dx(i, j);
        }
    }
}

/**
 * This function computes the derivative of the y-component of the
 * velocity field with respect to time. This is done by taking the
 * derivative of the height field with respect to y and multiplying
 * by -g.
 */
void compute_dv()
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dv(i, j) = -g * dh_dy(i, j);
        }
    }
}

/**
 * This function computes the next time step using a multistep method.
 * The coefficients a1, a2, and a3 are used to determine the weights
 * of the current and previous time steps.
 */
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

/**
 * This function computes the ghost cells for the horizontal boundaries.
 * This is done by copying the values from the opposite side of the domain.
 */
void compute_ghost_horizontal()
{
    for (int j = 0; j < ny; j++)
    {
        h(nx, j) = h(0, j);
    }
}

/**
 * This function computes the ghost cells for the vertical boundaries.
 * This is done by copying the values from the opposite side of the domain.
 */
void compute_ghost_vertical()
{
    for (int i = 0; i < nx; i++)
    {
        h(i, ny) = h(i, 0);
    }
}

/**
 * This function computes the boundaries for the horizontal boundaries.
 * We do this by copying the values from the opposite side of the domain.
 */
void compute_boundaries_horizontal()
{
    for (int j = 0; j < ny; j++)
    {
        u(0, j) = u(nx, j);
    }
}

/**
 * This function computes the boundaries for the vertical boundaries.
 * We do this by copying the values from the opposite side of the domain.
 */
void compute_boundaries_vertical()
{
    for (int i = 0; i < nx; i++)
    {
        v(i, 0) = v(i, ny);
    }
}

/**
 * This function swaps the buffers for the derivatives of our different fields.
 * This is done so that we can use the derivatives from the previous time steps
 * in our multistep method.
 */
void swap_buffers()
{
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

int t = 0;

void step()
{
    // First, we compute our ghost cells as we need them for our derivatives
    compute_ghost_horizontal();
    compute_ghost_vertical();

    // Next, we compute the derivatives of our fields
    compute_dh();
    compute_du();
    compute_dv();

    // We set the coefficients for our multistep method
    double a1, a2, a3;

    if (t == 0)
    {
        a1 = 1.0;
    }
    else if (t == 1)
    {
        a1 = 3.0 / 2.0;
        a2 = -1.0 / 2.0;
    }
    else
    {
        a1 = 23.0 / 12.0;
        a2 = -16.0 / 12.0;
        a3 = 5.0 / 12.0;
    }

    // Finally, we compute the next time step using our multistep method
    multistep(a1, a2, a3);

    // We compute the boundaries for our fields, as they are (1) needed for
    // the next time step, and (2) aren't explicitly set in our multistep method
    compute_boundaries_horizontal();
    compute_boundaries_vertical();

    // We swap the buffers for our derivatives so that we can use the derivatives
    // from the previous time steps in our multistep method, then increment
    // the time step counter
    swap_buffers();

    t++;
}

// Since all of our memory is already on the CPU, and specifically in the
// height field, we don't need to transfer anything
void transfer(double *h)
{
    return;
}

// We free all of the memory that we allocated. We didn't create the initial
// height or velocity fields, so we don't need to free them. They are the
// responsibility of the calling code.
void free_memory()
{
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