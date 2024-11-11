#include <mpi.h>

#include <stdlib.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

/**
 * This is your initialization function! It is very similar to the one in
 * serial.cpp, but with some difference. Firstly, only the process with rank 0
 * is going to actually generate the initial conditions h0, u0, and v0, so all
 * other processes are going to get nullptrs. Therefore, you'll need to find some
 * way to scatter the initial conditions to all processes. Secondly, now the
 * rank and num_procs arguments are passed to the function, so you can use them
 * to determine which rank the node running this process has, and how many
 * processes are running in total. This is useful to determine which part of the
 * domain each process is going to be responsible for.
 */
void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    // TODO: Your code here
}

/**
 * This is your step function! It is very similar to the one in serial.cpp, but
 * now the domain is divided among the processes, so you'll need to find some
 * way to communicate the ghost cells between processes.
 */
void step()
{
    // TODO: Your code here
}

/**
 * This is your transfer function! Similar to what you did in gpu.cu, you'll
 * need to get the data from the computers you're working on (there it was
 * the GPU, now its a bunch of CPU nodes), and send them all back to the process
 * which is actually running the main function (then it was the CPU, not it's
 * the node with rank 0).
 */
void transfer(double *h_recv)
{
    // TOOD: Your code here
}

/**
 * This is your finalization function! Since different nodes are going to be
 * initializing different chunks of memory, make sure to check which node
 * is running the code before you free some memory you haven't allocated, or
 * that you've actually freed memory that you have.
 */
void free_memory()
{
    // TODO: Your code here
}