#include <mpi.h>
#include <stdio.h>
#include <cstring>
#include <stdlib.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"


#define local_h(i,j) local_h[(i) * (ny + 1) + (j)]
#define local_u(i,j) local_u[(i) * (ny) + (j)]
#define local_v(i,j) local_v[(i) * (ny + 1) + (j)]

#define local_dh(i,j) local_dh[(i) * ny + (j)]
#define local_du(i,j) local_du[(i) * ny + (j)]
#define local_dv(i,j) local_dv[(i) * ny + (j)]

#define local_dh1(i,j) local_dh1[(i) * ny + (j)]
#define local_du1(i,j) local_du1[(i) * ny + (j)]
#define local_dv1(i,j) local_dv1[(i) * ny + (j)]

#define local_dh2(i,j) local_dh2[(i) * ny + (j)]
#define local_du2(i,j) local_du2[(i) * ny + (j)]
#define local_dv2(i,j) local_dv2[(i) * ny + (j)]

#define local_dh_dx(i,j) (local_h(i+1,j) - local_h(i,j)) /dx
#define local_dh_dy(i,j) (local_h(i, j+1) - local_h(i,j)) / dy

#define local_du_dx(i,j) (local_u(i+1,j) - local_u(i,j)) / dx
#define local_dv_dy(i,j) (local_v(i,j + 1) - local_v(i,j)) / dy



double *local_h , *local_u, *local_v, *local_dh, *local_du, *local_dv, *local_dh1, *local_du1, *local_dv1, *local_dh2, *local_du2, *local_dv2;
double H,g,dx,dy,dt;
int local_rows, rank_, num_procs, nx, ny, global_nx, global_ny;
int *hcounts, *hdisps;

void check_mpi_error(int err) {
    if (err != MPI_SUCCESS) {
        char error_string[1024];
        int length_of_error_string;
        MPI_Error_string(err, error_string, &length_of_error_string);
        printf("MPI Error: %s\n", error_string);
        MPI_Abort(MPI_COMM_WORLD, err);
    }
}



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
    global_nx = nx_; nx = nx_;
    global_ny = ny_; ny = ny_;
    num_procs = num_procs_;

    // dynamically calculate NProws and Npcols using mpi dims create
    //printf("Total number of processes: %d\n", num_procs); 
    //printf("Global nx: %d\n", global_nx); 
    int rows_per_proc = nx / num_procs;
    int remainder_rows = nx % num_procs;
    int start_row = rank_ * rows_per_proc;
    int end_row = start_row + rows_per_proc;
    if (rank_ == num_procs - 1) {
        end_row += remainder_rows; // last processor takes extra rows
    }
    local_rows = end_row - start_row;
    int last_proc_rows =rows_per_proc + remainder_rows;
   
    if (num_procs == 1) { printf("rows %d is equal (?) to nx %d", last_proc_rows, nx); }
    /* if(rank_ ==0) {
    printf("start_row: %d\n", start_row);
    printf("end_row: %d\n", end_row);
    fflush(stdout);
    }*/
    //Local dimensions for each process

    local_h = (double *)calloc((local_rows+1) * (ny+1), sizeof(double)); 
    local_u = (double *)calloc((local_rows+2) * ny, sizeof(double));
    local_v = (double *)calloc(local_rows * (ny+2), sizeof(double));
    
    local_dh = (double *)calloc((local_rows) * (ny), sizeof(double));    
    local_du = (double *)calloc((local_rows) * ny, sizeof(double));
    local_dv = (double *)calloc(local_rows * (ny), sizeof(double));

    local_dh1 = (double *)calloc((local_rows) * (ny), sizeof(double)); 
    local_du1 = (double *)calloc((local_rows) * ny, sizeof(double));
    local_dv1 = (double *)calloc(local_rows * (ny), sizeof(double));

    local_dh2 = (double *)calloc((local_rows) * (ny), sizeof(double)); 
    local_du2 = (double *)calloc((local_rows) * ny, sizeof(double));
    local_dv2 = (double *)calloc(local_rows * (ny), sizeof(double));




    // create displacements and counts for scattering
    hdisps = (int *) malloc(num_procs*sizeof(int));
    int *udisps = (int *) malloc(num_procs*sizeof(int));
    int *vdisps = (int *) malloc(num_procs*sizeof(int));
    hcounts = (int *) malloc(num_procs*sizeof(int));
    int *ucounts = (int *) malloc(num_procs*sizeof(int));
    int *vcounts = (int *) malloc(num_procs*sizeof(int));

    // compute displacements and counts based on dynamic block size
    for (int i = 0; i < num_procs; i++) {
        if (i < num_procs -1 ){
        hcounts[i] = (rows_per_proc+1) * (ny+1);
        ucounts[i] = (rows_per_proc+2)*ny;
        vcounts[i] = (rows_per_proc)*(ny+2);
        }
        if (i == num_procs -1) {
        hcounts[i] = (last_proc_rows + 1) * (ny+1);
        ucounts[i] = (last_proc_rows + 2) * ny;
        vcounts[i] = (last_proc_rows) * (ny + 2);
        }
    
        hdisps[i] = i * rows_per_proc * (ny + 1);
        udisps[i] = i * rows_per_proc * ny;
        vdisps[i] = i * rows_per_proc * (ny + 2);

    }
    double *h =  (double *)calloc((global_nx+1) * (global_ny+1), sizeof(double)); 
    double *u =  (double *)calloc((global_nx+2) * (global_ny), sizeof(double)); 
    double *v =  (double *)calloc((global_nx) * (global_ny+2), sizeof(double));    

    dx = length_ / nx;
    dy = width_ / ny;
    dt = dt_; 
    H = H_;
    g = g_;
    //printf("I am rank %3d. \n", rank_); 
    //rank 0 initializes global arrays and generates initial conditions
    if (rank_ == 0) {
        //initial conditions
        memcpy(h, h0, sizeof(double)*(global_nx +1) * (global_ny +1));
        memcpy(u, u0, sizeof(double)*(global_nx + 2) * (global_ny));
        memcpy(v, v0, sizeof(double)*(global_nx) * (global_ny+2));
    }    
        /*for (int i = 0; i < global_nx + 1; i++) {
            for (int j = 0; j < global_ny + 1; j++) {
                h(i,j) = i * (ny +1 ) + j;
            }
        }
        printf("Global Matrix h:\n");
        for (int i = 0; i < global_nx + 1; i++) {
            for (int j = 0; j < global_ny + 1; j++) {
                printf("%3f ", (double) h(i,j));
            }
            printf("\n");
        }*/
        /*printf("Global Matrix h0:\n");
        for (int i = 0; i < global_nx + 1; i++) {
            for (int j = 0; j < global_ny + 1; j++) {
                printf("%3f ", (double) h0[i *(global_ny + 1) +j]);
            }
            printf("\n");
        }
        printf("entry h[0]: %3f \n", h(0,0));
        printf("entry h[0]: %3f \n", h[0]);
  
        printf("Global Matrix u:\n"); 
        for (int i = 0; i < global_nx + 2; i++) {
             for (int j = 0; j < global_ny; j++) {
                 printf("%3f ", (double) u(i,j));
           }
           printf("\n");
        }
        printf("Global Matrix v:\n"); 
        for (int i = 0; i < global_nx; i++) {
             for (int j = 0; j < global_ny + 2; j++) {
                 printf("%3f ", (double) v(i,j));
           }
           printf("\n");
        }*/


     
        /*
        printf("Displacements (hdisps):\n");
        for (int i =0; i < num_procs; i++) {
            printf("hdisps[%d] = %d\n", i , hdisps[i]);
        }
        printf("Displacements (udisps):\n");
        for (int i =0; i < num_procs; i++) {
            printf("udisps[%d] = %d\n", i , udisps[i]);
        }
        printf("Displacements (vdisps):\n");
        for (int i =0; i < num_procs; i++) {
            printf("vdisps[%d] = %d\n", i , vdisps[i]);
        }
        fflush(stdout);
    }*/
     
    
     
     
     // broadcast parameters
     //MPI_Bcast(&H_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     //MPI_Bcast(&g_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     //MPI_Bcast(&dt_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

     // scatter initial conditions to all processes
     int err1, err2, err3;
     err1 = MPI_Scatterv(h, hcounts, hdisps, MPI_DOUBLE, local_h, hcounts[rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);
     err2 = MPI_Scatterv(u, ucounts, udisps, MPI_DOUBLE, local_u, ucounts[rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);
     err3 = MPI_Scatterv(v, vcounts, vdisps, MPI_DOUBLE, local_v, vcounts[rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    check_mpi_error(err1);
    check_mpi_error(err2);
    check_mpi_error(err3);

    /*if (rank_ == 0) {
    printf("entry localh(0): %3f \n", local_h[0]); 
    printf("entry localh(0,0): %3f \n", local_h(0,0));
    
    printf("Rank %d: Local Matrix h:\n", rank_); 
    for (int i = 0; i < local_rows + 1; i++) {
           for (int j = 0; j < ny + 1; j++) {
               printf("%3f ", (double) local_h[i*(ny+ 1) +j]);
           }
           printf("\n");
    }
    printf("\n");
    printf("Rank %d: Local Matrix u:\n", rank_); 
    for (int i = 0; i < local_rows + 2; i++) {
           for (int j = 0; j < ny; j++) {
               printf("%3f ", (double) local_u[i*(ny) +j]);
           }
           printf("\n");
    }
    printf("\n");
    printf("Rank %d: Local Matrix v:\n", rank_); 
        for (int i = 0; i < local_rows; i++) {
             for (int j = 0; j < ny + 2; j++) {
                 printf("%3f ", (double) local_v[i*(ny+2) +j]);
             }
        printf("\n");
    }
    printf("\n");
    }*/
    
    /*printf("Rank %d: Local Matrix h:\n", rank_); 
    for (int i = 0; i < local_rows + 1; i++) {
           for (int j = 0; j < ny + 1; j++) {
               printf("%3f ", (double) local_h[i*(ny+ 1) +j]);
           }
           printf("\n");
    }
    printf("\n");
    printf("Rank %d: Local Matrix u:\n", rank_); 
    for (int i = 0; i < local_rows + 2; i++) {
           for (int j = 0; j < ny; j++) {
               printf("%3f ", (double) local_u[i*(ny) +j]);
           }
           printf("\n");
    }
    printf("\n");
    
    fflush(stdout);*/
    if (rank_ == 0) {
        free(h); 
        free(u); 
        free(v); 
    }
    free(udisps); udisps=NULL; free(ucounts); ucounts = NULL;
    free(vdisps); vdisps = NULL; free(vcounts); vcounts = NULL;
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank_ == 0) {
        //printf("total number: %d\n", (nx + 1)*(ny+1));
        int total = 0;
        for (int i = 0; i < num_procs - 1; i++) // not last processor
        {
            hcounts[i] = hcounts[i] - (ny + 1); 
            total = total + hcounts[i];
            //printf("hcount is %d for rank %d \n", hcounts[i], i); fflush(stdout);
        }
        total = total + hcounts[num_procs - 1];
        //printf("total: %d \n", total); fflush(stdout);
    }

    MPI_Bcast(hcounts, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
    //printf("made it to the end of init with rank %d \n", rank_);
    //fflush(stdout);
}

/* bottom row of processor is top row of next
except that bottom row of last processor is top row of first
need to send top of row of each processor to bottom row of previous
except top of first processor goes to bottom of last
*/
void compute_ghost_horizontal() {
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    //printf("local rows for Rank %d is %d \n", rank_, local_rows);
    /*printf("ny is %d \n", ny); 
    printf("Rank %d before ghost exchange:\n", rank_);
    for (int i = 0; i < local_rows + 1; i++) {
        for (int j = 0; j < ny + 1; j++) {
            printf("%f ", local_h[i * (ny+1) + j]);
        }
        printf("\n");
    }
    fflush(stdout);
    */

    if (num_procs > 1) {
        if (rank_ > 0) {
        //printf("Rank %d sending its top row to previous rank %d.\n", rank_, rank_-1); fflush(stdout);
            MPI_Send(&local_h[(0) * (ny+1)], ny+1, MPI_DOUBLE, rank_ -1, 0, MPI_COMM_WORLD);
        }


        if (rank_ < num_procs - 1) {
            //printf("Rank %d receiving its bottom row from next rank %d.\n", rank_, rank_+1); fflush(stdout);
       
            MPI_Recv(&local_h[(local_rows) * (ny+1)], ny + 1, MPI_DOUBLE, rank_ + 1, 0, MPI_COMM_WORLD,&status);
 
        }

    
        if (rank_ == 0) {
            //printf("Rank 0 sending its top row to last processor %d.\n", num_procs-1); fflush(stdout);
            MPI_Send(&local_h[(0) * (ny+1)], ny + 1, MPI_DOUBLE, num_procs-1, 3, MPI_COMM_WORLD);
        }


        if (rank_ == num_procs -1) {
            //printf("Last rank receiving its bottom row from rank 0"); fflush(stdout);
            MPI_Recv(&local_h[(local_rows) * (ny+1)], ny+ 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);

    
        }
    }

    if (num_procs == 1) {
        for (int j = 0; j < ny; j++)
        {
            local_h(local_rows,j) = local_h(0,j);
        }
    }

        
    /*printf("Rank %d after ghost exchange:\n", rank_);
    for (int i = 0; i < local_rows + 1; i++) {
        for (int j = 0; j < ny + 1; j++) {
            printf("%f ", local_h[i * (ny+1) + j]);
        }
        printf("\n");
    }
    fflush(stdout);*/ 
 
}

/*
no need to share along processors 
*/
void compute_ghost_vertical()
{
    for (int i = 0; i < local_rows; i++) 
    {
        local_h(i,ny) = local_h(i, 0);
    }

}

/*
copy bottom row of u of a processor to top row of u of next processor
*/
void compute_boundaries_horizontal()
{

    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    
    if (num_procs > 1) {

        if (rank_ < num_procs -1) {
            MPI_Send(&local_u[(local_rows) * (ny)], ny, MPI_DOUBLE, rank_ +1, 4, MPI_COMM_WORLD);
        }


        if (rank_ >0) {
            MPI_Recv(&local_u[(0) * ny], ny, MPI_DOUBLE, rank_ - 1, 4,  MPI_COMM_WORLD, &status);

        }

    
        if (rank_ == num_procs -1) {
            MPI_Send(&local_u[(local_rows) * (ny)], ny, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
        }


        if (rank_ == 0) {
            MPI_Recv(&local_u[(0 ) * (ny)], ny, MPI_DOUBLE, num_procs-1, 5, MPI_COMM_WORLD, &status);    
        }

    }
    if (num_procs == 1) {
        for (int j = 0; j < ny; j++)
        {
            local_u(0,j) = local_u(local_rows, j);
        }
    }
    
}

/*
don't need to share here
*/
void compute_boundaries_vertical()
{
    for (int i = 0; i < local_rows; i ++) 
    {
        local_v(i, 0) = local_v(i, ny);
    }
}

/*
*/
void swap_buffers()
{
    double *tmp;
    
    tmp = local_dh2;
    local_dh2 = local_dh1;
    local_dh1 = local_dh;
    local_dh = tmp;

    tmp = local_du2;
    local_du2 = local_du1;
    local_du1 = local_du;
    local_du = tmp;
    
    tmp = local_dv2;
    local_dv2 = local_dv1;
    local_dv1 = local_dv;
    local_dv = tmp;
}

/*
 * This is your step function! It is very similar to the one in serial.cpp, but
 * now the domain is divided among the processes, so you'll need to find some
 * way to communicate the ghost cells between processes.
 */
int t = 0;
void step()
{
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    compute_ghost_horizontal();
    compute_ghost_vertical();
    MPI_Barrier(MPI_COMM_WORLD);
    //printf("\n");
    //printf("I am Rank %d in the step() function after computing ghosts.\n", rank_); 
    //printf("number of processes: %d \n", num_procs);

    for (int i = 0; i < local_rows; i++) 
    {
        for (int j = 0; j < ny; j++) 
        {
            local_dh(i,j) = -H * (local_du_dx(i,j) + local_dv_dy(i,j));
            local_du(i,j) = -g * local_dh_dx(i,j);
            local_dv(i,j) = -g * local_dh_dy(i,j);
        }
    }


    double a1, a2, a3;

    if (t==0)
    {
        a1 = 1.0;
    }
    else if (t==1)
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

    //multistep(a1,a2, a3); 
    for (int i = 0; i < local_rows; i++) 
    {
        for (int j = 0; j < ny; j++ ) 
        {
            local_h(i,j) += (a1 * local_dh(i,j) + a2 * local_dh1(i,j) + a3 * local_dh2(i,j)) * dt;
            local_u(i+1,j) += (a1 * local_du(i,j) + a2 * local_du1(i,j) + a3 * local_du2(i,j)) * dt; 
            local_v(i, j+1) += (a1 * local_dv(i,j) + a2 * local_dv1(i,j) + a3 * local_dv2(i,j)) * dt;
        }
    }

    compute_boundaries_horizontal();
    compute_boundaries_vertical();

    swap_buffers();
    MPI_Barrier(MPI_COMM_WORLD);
    t++;
    //printf("made it to end of step with rank %d \n", rank_);
    //fflush(stdout); 

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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    //printf("hcounts for rank %d is: %d\n", rank_, hcounts[rank_]);
    //printf("hdisps for rank %d is: %d\n", rank_, hdisps[rank_]);
    //fflush(stdout);
    MPI_Gatherv(&local_h[0], hcounts[rank_], MPI_DOUBLE, h_recv, hcounts, hdisps, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

/**
 * This is your finalization function! Since different nodes are going to be
 * initializing different chunks of memory, make sure to check which node
 * is running the code before you free some memory you haven't allocated, or
 * that you've actually freed memory that you have.
 */
void free_memory()
{
    free(local_h); free(local_u); free(local_v); free(local_dh); free(local_du); free(local_dv);
    free(local_dh1); free(local_du1); free(local_dv1); free(local_dh2); free(local_du2); free(local_dv2);
    free(hcounts); free(hdisps); 
}
