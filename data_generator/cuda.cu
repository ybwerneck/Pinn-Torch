#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>

// final time of simulation

// FHN cell model parameters
#define A 0.2
#define B 0.2
#define C 3.0
#define I_APP 1.0
#define CUDA_CHECK_ERROR() \
do { \
    cudaDeviceSynchronize(); \
    cudaError_t cudaError = cudaGetLastError(); \
    if (cudaError != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaError)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


void printMemoryUsage() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    double free_gb = static_cast<double>(free_mem) / (1 << 30); // Convert bytes to GB
    double total_gb = static_cast<double>(total_mem) / (1 << 30);

    printf("Memorya usage:\n");
    printf("Free memory: %.2f GB\n", free_gb);
    printf("Total memory: %.2f GB\n", total_gb);
}




// CUDA kernel to simulate FHN cell models
__global__ void fhn_kernel(float* ui, float* vi,float *ki, float* u_solution, float* v_solution, float DT, int NUM_CELLS, float T_FINAL, int rate, int N) {
    
    
    
          int I=999;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // calculate global index of thread
    float u_i = ui[idx];
    float v_i = vi[idx];
    float iapp= ki[idx];
 
    
    
    
    
    if (idx < NUM_CELLS) {
        int step = 0;
        
        float ro=N*rate;
       
        #pragma unroll 
  
        for (int t_i = 0; t_i < ro & step<N; t_i++) {

            // update FHN model equations for current cell

            float  u_new = u_i + DT * 10*((iapp + 1)*(u_i*(u_i-0.4)*(1-u_i))-v_i );
            float v_new = v_i + DT * ((u_i*0.04-0.16*v_i));

            // store solution for current time step
            int ind = idx + NUM_CELLS * step;

           
         

            u_solution[ind] = u_i;
            v_solution[ind] = v_i;
            
    
            step= (step == 0 || (t_i) % rate == 0 || (t_i) >= ro)?step+1:step;
            u_i = u_new;
            v_i = v_new;    
  

        }
      
    }
    
}

int main(int argc, char* argv[]) {

    float DT = 0.1;
    float T_FINAL =100;
    int rate = 2;

    if (argc != 4) {
        printf("Usage: program_name arg1 arg2\n");
  
    }
    else {
        DT = atof(argv[2]);
        T_FINAL = atof(argv[1]);
        rate = atoi(argv[3]);
                printf("\n DT %d t %f rate %f \n",DT,T_FINAL,rate);

    }
    int N=T_FINAL / (DT * rate) +1;
    //if(T_FINAL / (DT * rate)>int(T_FINAL / (DT * rate)))
      //  N++;
   

    int NUM_CELLS = 0;
    std::ifstream file("u.csv");
    std::string row;
    while (std::getline(file, row)) {
        NUM_CELLS++;
    }

    printf("%d CELLS   %d timepoints \n", NUM_CELLS,N)  ;


    // allocate memory on host for FHN model variables
    float* u_host = (float*)malloc(sizeof(float) * NUM_CELLS);
    float* v_host = (float*)malloc(sizeof(float) * NUM_CELLS);
    float* k_host = (float*)malloc(sizeof(float) * NUM_CELLS);


    float* u_solution = (float*)malloc(sizeof(float) * NUM_CELLS * N);
    float* v_solution = (float*)malloc(sizeof(float) * NUM_CELLS * N);
    float* t_solution = (float*)malloc(sizeof(float) * NUM_CELLS * N);
    printMemoryUsage();

    std::ifstream file2("u.csv"); // Assuming the CSV contains u,v,k values
    std::string line;
    int i = 0;
    while (std::getline(file2, line) && i < NUM_CELLS) {
        std::stringstream ss(line);
        std::string cell;
        if (std::getline(ss, cell, ',') && i < NUM_CELLS) {
            u_host[i] = std::stof(cell);
        }
        if (std::getline(ss, cell, ',') && i < NUM_CELLS) {
            v_host[i] = std::stof(cell);
        }
        if (std::getline(ss, cell, ',') && i < NUM_CELLS) {
            k_host[i] = std::stof(cell);
        }
        
        i++;
    }
    file2.close();
    
    
    // start measuring time for memory allocation and data transfer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    
    // allocate memory on device for FHN model variables
    float* u_dev, * v_dev, * t_dev;
    cudaMalloc((void**)&u_dev, sizeof(float) * NUM_CELLS);
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&v_dev, sizeof(float) * NUM_CELLS);
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&t_dev, sizeof(float) * NUM_CELLS);
    CUDA_CHECK_ERROR();

    float* u_solution_dev, * v_solution_dev, * t_solution_dev;
    cudaMalloc((void**)&u_solution_dev, sizeof(float) * NUM_CELLS * N);
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&v_solution_dev, sizeof(float) * NUM_CELLS * N);
    CUDA_CHECK_ERROR();
    //cudaMalloc((void**)&t_solution_dev, sizeof(float) * NUM_CELLS * N);
    CUDA_CHECK_ERROR();

    // copy FHN model variables from host to device
    cudaMemcpy(u_dev, u_host, sizeof(float) * NUM_CELLS, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(v_dev, v_host, sizeof(float) * NUM_CELLS, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(t_dev, k_host, sizeof(float) * NUM_CELLS, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    // calculate number of CUDA threads and blocks to use
    int threads_per_block = 1024;
    int blocks_per_grid = (NUM_CELLS + threads_per_block - 1) / threads_per_block;

    printf("%d block of 1024 threads \n", blocks_per_grid);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float allocation_time;
    cudaEventElapsedTime(&allocation_time, start, stop);

 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
printMemoryUsage();
    // simulate FHN cell models on device using CUDA kernel
    fhn_kernel << <blocks_per_grid, threads_per_block >> > (u_dev, v_dev,t_dev, u_solution_dev, v_solution_dev, DT, NUM_CELLS, T_FINAL, rate,N);
    CUDA_CHECK_ERROR();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float run_time;
    cudaEventElapsedTime(&run_time, start, stop);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy FHN model variables from device to host
    cudaMemcpy(u_solution, u_solution_dev, sizeof(float) * NUM_CELLS * N, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    cudaMemcpy(v_solution, v_solution_dev, sizeof(float) * NUM_CELLS * N, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    FILE* u_fp, * v_fp, * t_fp,*p_fp;
    u_fp = fopen("outputs/u.csv", "w");
    v_fp = fopen("outputs/v.csv", "w");
    t_fp = fopen("outputs/t.csv", "w");
    p_fp = fopen("outputs/p.csv", "w");

    
    for (int i = 0; i < NUM_CELLS; i++) {
        fprintf(u_fp, "%f", u_host[0 * NUM_CELLS]);
        fprintf(v_fp, "%f", v_host[0 * NUM_CELLS ]);

        for (int j = 0; j < N; j++) {
            int ind = j * NUM_CELLS + i;
            // update FHN model equations for current cell
            float U = u_solution[ind];
            float V = v_solution[ind];
            // print updated values to CSV files
            fprintf(u_fp, ",%f", U);
            fprintf(v_fp, ",%f", V);

        }
        fprintf(u_fp, "\n");
        fprintf(v_fp, "\n");
    }
    fclose(u_fp);
    fclose(v_fp);
    fprintf(t_fp, "%f", 0*DT*rate);

    for (int i = 0; i < N; i++) {
        fprintf(t_fp, ", %f ", (DT * rate)*i);
    }

    fclose(t_fp);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float w_time;
    cudaEventElapsedTime(&w_time, start, stop);
    
        
    
    fprintf(p_fp, " %f  ,%f , %f", allocation_time,run_time,w_time);
    fclose(p_fp);
    // free memory
    free(u_host);
    free(v_host);
    free(k_host);
    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(t_dev);

    printf("Run time in seconds: %f\n", run_time/1000);
    return 0;
}
