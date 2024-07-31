#include <hip/hip_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <mpi.h>

#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/un.h>
#include <netdb.h>
#include <time.h>

#define SAMPLE_VERSION "HIP-Examples-Application-v1.0"
#define SUCCESS 0
#define FAILURE 1

#define N 10*1024
// #define N 2560
// #define N 1280
// #define N 512
// #define N 1024*1024*1024

using namespace std;

// Helper macro for catching HIP errors
#define HIP_CALL(cmd)                                                                   \
    do {                                                                                \
        hipError_t error = (cmd);                                                       \
        if (error != hipSuccess)                                                        \
        {                                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)          \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << "\n";   \
            exit(-1);                                                                   \
        }                                                                               \
    } while (0)


struct kernelParam
{
    void *ptrs[7];
    void *ptrs_flag[7];
    void *ptrs_done[7];
};

__global__ void allred0Peer(float* in, float *out, int* done, int *flag, kernelParam kp, int myrank, int nranks, int iter)
{
    int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
	int tid = hipThreadIdx_x;

	/* sync before starting the reduction */
	if (tid < nranks) {
		if (hipBlockIdx_x == 0 && tid == 0) {
			flag[myrank] = iter;
#pragma unroll
			for (int j=0; j < (nranks-1); j++) {
				__atomic_store_n(&(((int*) kp.ptrs_flag[j])[myrank]), iter, __ATOMIC_RELEASE);
			}	
		}
		/* atomic load serializes the reads among the threads. Also, flag is uncached. */
		while(__atomic_load_n(&flag[tid], __ATOMIC_ACQUIRE) != iter) {}
	}
	__syncthreads();


	float sum;
	for (int i = num; i < N; i+= hipBlockDim_x * hipGridDim_x) {
		sum = in[i];
		out[i] = sum;
	}

	/* sync after finishing the reduction */
	if (tid < nranks) {
                if (hipBlockIdx_x == 0 && tid == 0) {
                        done[myrank] = iter;
#pragma unroll
                        for (int j=0; j < (nranks-1); j++) {
                __atomic_store_n(&(((int*) kp.ptrs_done[j])[myrank]), iter, __ATOMIC_RELEASE);
                        }
                }

                while(__atomic_load_n(&done[tid], __ATOMIC_ACQUIRE) != iter) {}
        }
        __syncthreads();

}

__global__ void allred1Peer(float* in, float *out, float* peer1, int* done, int *flag, kernelParam kp, int myrank, int nranks, int iter)
{
    int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
	int tid = hipThreadIdx_x;

	/* sync before starting the reduction */
	if (tid < nranks) {
		if (hipBlockIdx_x == 0 && tid == 0) {
			flag[myrank] = iter;
#pragma unroll
			for (int j=0; j < (nranks-1); j++) {
				__atomic_store_n(&(((int*) kp.ptrs_flag[j])[myrank]), iter, __ATOMIC_RELEASE);
			}	
		}
		/* atomic load serializes the reads among the threads. Also, flag is uncached. */
		while(__atomic_load_n(&flag[tid], __ATOMIC_ACQUIRE) != iter) {}
	}
	__syncthreads();


	float sum;
	for (int i = num; i < N; i+= hipBlockDim_x * hipGridDim_x) {
		sum = in[i];
		sum += peer1[i];
		out[i] = sum;
	}

	/* sync after finishing the reduction */
	if (tid < nranks) {
                if (hipBlockIdx_x == 0 && tid == 0) {
                        done[myrank] = iter;
#pragma unroll
                        for (int j=0; j < (nranks-1); j++) {
                __atomic_store_n(&(((int*) kp.ptrs_done[j])[myrank]), iter, __ATOMIC_RELEASE);
                        }
                }

                while(__atomic_load_n(&done[tid], __ATOMIC_ACQUIRE) != iter) {}
        }
        __syncthreads();

}

__global__ void allred2Peer(float* in, float *out, float* peer1, float* peer2, int* done, int *flag, kernelParam kp, int myrank, int nranks, int iter)
{
    int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
	int tid = hipThreadIdx_x;

	/* sync before starting the reduction */
	if (tid < nranks) {
		if (hipBlockIdx_x == 0 && tid == 0) {
			flag[myrank] = iter;
#pragma unroll
			for (int j=0; j < (nranks-1); j++) {
				__atomic_store_n(&(((int*) kp.ptrs_flag[j])[myrank]), iter, __ATOMIC_RELEASE);
			}	
		}
		/* atomic load serializes the reads among the threads. Also, flag is uncached. */
		while(__atomic_load_n(&flag[tid], __ATOMIC_ACQUIRE) != iter) {}
	}
	__syncthreads();


	float sum;
	for (int i = num; i < N; i+= hipBlockDim_x * hipGridDim_x) {
		sum = in[i];
		sum += peer1[i];
        sum += peer2[i];
		out[i] = sum;
	}

	/* sync after finishing the reduction */
	if (tid < nranks) {
                if (hipBlockIdx_x == 0 && tid == 0) {
                        done[myrank] = iter;
#pragma unroll
                        for (int j=0; j < (nranks-1); j++) {
                __atomic_store_n(&(((int*) kp.ptrs_done[j])[myrank]), iter, __ATOMIC_RELEASE);
                        }
                }

                while(__atomic_load_n(&done[tid], __ATOMIC_ACQUIRE) != iter) {}
        }
        __syncthreads();

}

__global__ void allred3Peer(float* in, float *out, float* peer1, float* peer2, float* peer3, int* done, int *flag, kernelParam kp, int myrank, int nranks, int iter)
{
    int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
	int tid = hipThreadIdx_x;

	/* sync before starting the reduction */
	if (tid < nranks) {
		if (hipBlockIdx_x == 0 && tid == 0) {
			flag[myrank] = iter;
#pragma unroll
			for (int j=0; j < (nranks-1); j++) {
				__atomic_store_n(&(((int*) kp.ptrs_flag[j])[myrank]), iter, __ATOMIC_RELEASE);
			}	
		}
		/* atomic load serializes the reads among the threads. Also, flag is uncached. */
		while(__atomic_load_n(&flag[tid], __ATOMIC_ACQUIRE) != iter) {}
	}
	__syncthreads();


	float sum;
	for (int i = num; i < N; i+= hipBlockDim_x * hipGridDim_x) {
		sum = in[i];
		sum += peer1[i];
        sum += peer2[i];
        sum += peer3[i];
		out[i] = sum;
	}

	/* sync after finishing the reduction */
	if (tid < nranks) {
                if (hipBlockIdx_x == 0 && tid == 0) {
                        done[myrank] = iter;
#pragma unroll
                        for (int j=0; j < (nranks-1); j++) {
                __atomic_store_n(&(((int*) kp.ptrs_done[j])[myrank]), iter, __ATOMIC_RELEASE);
                        }
                }

                while(__atomic_load_n(&done[tid], __ATOMIC_ACQUIRE) != iter) {}
        }
        __syncthreads();

}

__global__ void allred4Peer(float* in, float *out, float* peer1, float* peer2, float* peer3, float* peer4, int* done, int *flag, kernelParam kp, int myrank, int nranks, int iter)
{
    int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
	int tid = hipThreadIdx_x;

	/* sync before starting the reduction */
	if (tid < nranks) {
		if (hipBlockIdx_x == 0 && tid == 0) {
			flag[myrank] = iter;
#pragma unroll
			for (int j=0; j < (nranks-1); j++) {
				__atomic_store_n(&(((int*) kp.ptrs_flag[j])[myrank]), iter, __ATOMIC_RELEASE);
			}	
		}
		/* atomic load serializes the reads among the threads. Also, flag is uncached. */
		while(__atomic_load_n(&flag[tid], __ATOMIC_ACQUIRE) != iter) {}
	}
	__syncthreads();


	float sum;
	for (int i = num; i < N; i+= hipBlockDim_x * hipGridDim_x) {
		sum = in[i];
		sum += peer1[i];
        sum += peer2[i];
        sum += peer3[i];
        sum += peer4[i];
		out[i] = sum;
	}

	/* sync after finishing the reduction */
	if (tid < nranks) {
                if (hipBlockIdx_x == 0 && tid == 0) {
                        done[myrank] = iter;
#pragma unroll
                        for (int j=0; j < (nranks-1); j++) {
                __atomic_store_n(&(((int*) kp.ptrs_done[j])[myrank]), iter, __ATOMIC_RELEASE);
                        }
                }

                while(__atomic_load_n(&done[tid], __ATOMIC_ACQUIRE) != iter) {}
        }
        __syncthreads();

}

__global__ void allred5Peer(float* in, float *out, float* peer1, float* peer2, float* peer3, float* peer4, float* peer5, int* done, int *flag, kernelParam kp, int myrank, int nranks, int iter)
{
    int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
	int tid = hipThreadIdx_x;

	/* sync before starting the reduction */
	if (tid < nranks) {
		if (hipBlockIdx_x == 0 && tid == 0) {
			flag[myrank] = iter;
#pragma unroll
			for (int j=0; j < (nranks-1); j++) {
				__atomic_store_n(&(((int*) kp.ptrs_flag[j])[myrank]), iter, __ATOMIC_RELEASE);
			}	
		}
		/* atomic load serializes the reads among the threads. Also, flag is uncached. */
		while(__atomic_load_n(&flag[tid], __ATOMIC_ACQUIRE) != iter) {}
	}
	__syncthreads();


	float sum;
	for (int i = num; i < N; i+= hipBlockDim_x * hipGridDim_x) {
		sum = in[i];
		sum += peer1[i];
        sum += peer2[i];
        sum += peer3[i];
        sum += peer4[i];
        sum += peer5[i];
		out[i] = sum;
	}

	/* sync after finishing the reduction */
	if (tid < nranks) {
                if (hipBlockIdx_x == 0 && tid == 0) {
                        done[myrank] = iter;
#pragma unroll
                        for (int j=0; j < (nranks-1); j++) {
                __atomic_store_n(&(((int*) kp.ptrs_done[j])[myrank]), iter, __ATOMIC_RELEASE);
                        }
                }

                while(__atomic_load_n(&done[tid], __ATOMIC_ACQUIRE) != iter) {}
        }
        __syncthreads();

}

__global__ void allred6Peer(float* in, float *out, float* peer1, float* peer2, float* peer3, float* peer4, float* peer5, float* peer6, int* done, int *flag, kernelParam kp, int myrank, int nranks, int iter)
{
    int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
	int tid = hipThreadIdx_x;

	/* sync before starting the reduction */
	if (tid < nranks) {
		if (hipBlockIdx_x == 0 && tid == 0) {
			flag[myrank] = iter;
#pragma unroll
			for (int j=0; j < (nranks-1); j++) {
				__atomic_store_n(&(((int*) kp.ptrs_flag[j])[myrank]), iter, __ATOMIC_RELEASE);
			}	
		}
		/* atomic load serializes the reads among the threads. Also, flag is uncached. */
		while(__atomic_load_n(&flag[tid], __ATOMIC_ACQUIRE) != iter) {}
	}
	__syncthreads();


	float sum;
	for (int i = num; i < N; i+= hipBlockDim_x * hipGridDim_x) {
		sum = in[i];
		sum += peer1[i];
        sum += peer2[i];
        sum += peer3[i];
        sum += peer4[i];
        sum += peer5[i];
        sum += peer6[i];
		out[i] = sum;
	}

	/* sync after finishing the reduction */
	if (tid < nranks) {
                if (hipBlockIdx_x == 0 && tid == 0) {
                        done[myrank] = iter;
#pragma unroll
                        for (int j=0; j < (nranks-1); j++) {
                __atomic_store_n(&(((int*) kp.ptrs_done[j])[myrank]), iter, __ATOMIC_RELEASE);
                        }
                }

                while(__atomic_load_n(&done[tid], __ATOMIC_ACQUIRE) != iter) {}
        }
        __syncthreads();

}

__global__ void allred7Peer(float* in, float *out, float* peer1, float* peer2, float* peer3, float* peer4, float* peer5, float* peer6, float* peer7, 
		int* done, int *flag, kernelParam kp, int myrank, int nranks, int iter)
{
    int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

	int tid = hipThreadIdx_x;

	/* sync before starting the reduction */
	if (tid < nranks) {
		if (hipBlockIdx_x == 0 && tid == 0) {
			flag[myrank] = iter;
#pragma unroll
			for (int j=0; j < (nranks-1); j++) {
				__atomic_store_n(&(((int*) kp.ptrs_flag[j])[myrank]), iter, __ATOMIC_RELEASE);
			}	
		}


		/* atomic load serializes the reads among the threads. Also, flag is uncached. */
		while(__atomic_load_n(&flag[tid], __ATOMIC_ACQUIRE) != iter) {}

	}
	__syncthreads();


	float sum;
	for (int i = num; i < N; i+= hipBlockDim_x * hipGridDim_x) {
		sum = in[i];
		sum += peer1[i];
		sum += peer2[i];
		sum += peer3[i];
		sum += peer4[i];
		sum += peer5[i];
		sum += peer6[i];
		sum += peer7[i];

		out[i] = sum;
	}

	/* sync after finishing the reduction */
	if (tid < nranks) {
                if (hipBlockIdx_x == 0 && tid == 0) {
                        done[myrank] = iter;
#pragma unroll
                        for (int j=0; j < (nranks-1); j++) {
                __atomic_store_n(&(((int*) kp.ptrs_done[j])[myrank]), iter, __ATOMIC_RELEASE);
                        }
                }

                while(__atomic_load_n(&done[tid], __ATOMIC_ACQUIRE) != iter) {}
        }
    __syncthreads();

}

__global__ void allredPeer(float* in, float *out, int* done, int *flag, kernelParam kp, int myrank, int nranks, int iter, int numDevices) {
    int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
	int tid = hipThreadIdx_x;

	/* sync before starting the reduction */
	if (tid < nranks) {
		if (hipBlockIdx_x == 0 && tid == 0) {
			flag[myrank] = iter;
#pragma unroll
			for (int j=0; j < (nranks-1); j++) {
				__atomic_store_n(&(((int*) kp.ptrs_flag[j])[myrank]), iter, __ATOMIC_RELEASE);
			}	
		}
		/* atomic load serializes the reads among the threads. Also, flag is uncached. */
		while(__atomic_load_n(&flag[tid], __ATOMIC_ACQUIRE) != iter) {}
	}
	__syncthreads();


	float sum;
	for (int i = num; i < N; i+= hipBlockDim_x * hipGridDim_x) {
		sum = in[i];
        for (int j=0;j< numDevices; j++) {
            sum += ((float *)kp.ptrs[j])[i];
        }
		out[i] = sum;
	}

	/* sync after finishing the reduction */
	if (tid < nranks) {
                if (hipBlockIdx_x == 0 && tid == 0) {
                        done[myrank] = iter;
#pragma unroll
                        for (int j=0; j < (nranks-1); j++) {
                __atomic_store_n(&(((int*) kp.ptrs_done[j])[myrank]), iter, __ATOMIC_RELEASE);
                        }
                }

                while(__atomic_load_n(&done[tid], __ATOMIC_ACQUIRE) != iter) {}
        }
        __syncthreads();

}

int main(int argc, char* argv[])
{

    int myrank, nranks;
    // Declare timers
    std::chrono::high_resolution_clock::time_point t1, t2;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    kernelParam kp;

    hipDeviceProp_t devProp;

    HIP_CALL(hipSetDevice(myrank));
    hipGetDeviceProperties(&devProp, myrank);

    int numDevices = 0;
    hipError_t result = hipGetDeviceCount(&numDevices);
    if (result == hipSuccess) {
        std::cout << "Number of devices: " << numDevices << ", nranks= "<< nranks << std::endl;
    } else {
        std::cerr << "Error: " << result << std::endl;
    }

    float* input;
    input = (float*) malloc(N*sizeof(float));

    for (int i = 0; i < N; i++) {
	    input[i] = myrank + 1;
    }
    float *output = (float*) malloc(N*sizeof(float));
    for (int i = 0; i < N; i++) {
        output[i] = 0;
    }

    int *flag;
    flag= (int*) malloc (nranks * sizeof(int));
    for (int i = 0; i < nranks; i++) {
	    flag[i] = 0;
    }

    float* inputBuffer;
    float* outputBuffer;
    int* flagBuffer;
    int* doneBuffer;

    hipIpcMemHandle_t outHandle, inHandle, flagHandle, doneHandle;
    hipIpcMemHandle_t rcvOutHandle;
    hipIpcMemHandle_t *ipc_handles = NULL;
    hipIpcMemHandle_t *flag_handles = NULL;
    hipIpcMemHandle_t *done_handles = NULL;

    ipc_handles = (hipIpcMemHandle_t *) malloc(nranks * sizeof(hipIpcMemHandle_t));
    flag_handles = (hipIpcMemHandle_t *) malloc(nranks * sizeof(hipIpcMemHandle_t));
    done_handles = (hipIpcMemHandle_t *) malloc(nranks * sizeof(hipIpcMemHandle_t));

   
    hipExtMallocWithFlags((void**)&inputBuffer, N*sizeof(float), hipDeviceMallocDefault); //input buffer course grained. Reference: https://rocm.docs.amd.com/en/develop/conceptual/gpu-memory.html#coherence
    hipExtMallocWithFlags((void**)&outputBuffer, N*sizeof(float), hipDeviceMallocDefault); //input buffer course grained.
    hipExtMallocWithFlags((void**)&flagBuffer, nranks*sizeof(int), hipDeviceMallocUncached); 
    hipExtMallocWithFlags((void**)&doneBuffer, nranks*sizeof(int), hipDeviceMallocUncached); 


    /* Copy input to output */
    hipMemcpy(inputBuffer, input, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(outputBuffer, output, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(flagBuffer, flag, nranks * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(doneBuffer, flag, nranks * sizeof(int), hipMemcpyHostToDevice);


    int j = 0;

    /* Get IPC handle of inputbuf */
    HIP_CALL(hipIpcGetMemHandle(&inHandle, (void*)inputBuffer));
    HIP_CALL(hipIpcGetMemHandle(&flagHandle, (void*)flagBuffer));
    HIP_CALL(hipIpcGetMemHandle(&doneHandle, (void*)doneBuffer));

    /* Exchange IPC handles across the ranks */
    MPI_Allgather(&inHandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, ipc_handles, sizeof(hipIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(&flagHandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, flag_handles, sizeof(hipIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(&doneHandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, done_handles, sizeof(hipIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);


    for (int i = 0; i < nranks; i++) {
        if (i != myrank) {
    		HIP_CALL(hipIpcOpenMemHandle(&(kp.ptrs[j]), ipc_handles[i], hipIpcMemLazyEnablePeerAccess));
    		HIP_CALL(hipIpcOpenMemHandle(&(kp.ptrs_done[j]), done_handles[i], hipIpcMemLazyEnablePeerAccess));				
    		HIP_CALL(hipIpcOpenMemHandle(&(kp.ptrs_flag[j++]), flag_handles[i], hipIpcMemLazyEnablePeerAccess));				
	    }
    }

    hipStream_t stream;
    hipStreamCreate(&stream);
    MPI_Barrier(MPI_COMM_WORLD);

    // /* Warmup */
    // for (int iter = 0; iter < 10; iter++) {
	//     hipLaunchKernelGGL(allred7Peer,
    //         dim3(min(32,(N)/256)),
    //         dim3(256),
    //         0, stream,
    //         inputBuffer, outputBuffer, (float*)(kp.ptrs[0]), (float*)(kp.ptrs[1]), (float*)(kp.ptrs[2]),
    //             (float*)(kp.ptrs[3]), (float*)(kp.ptrs[4]), (float*)(kp.ptrs[5]), (float*)(kp.ptrs[6]), doneBuffer,
    //             flagBuffer, kp, myrank, nranks, iter+1);
    // }
    // hipStreamSynchronize(stream);
    // MPI_Barrier(MPI_COMM_WORLD);
    
    /* start time measurement */
    t1 = std::chrono::high_resolution_clock::now();
     
    /* Allreduce using read */
    for (int iter = 0; iter < 1; iter++) {	
        if(nranks==1) {
            hipLaunchKernelGGL(allred0Peer,
                dim3(min(32,(N)/256)),
                dim3(256),
                0, stream, 
                inputBuffer, outputBuffer, doneBuffer,
                    flagBuffer, kp, myrank, nranks, iter+1);
        }
        else if(nranks==2) {
            hipLaunchKernelGGL(allred1Peer,
                dim3(min(32,(N)/256)),
                dim3(256),
                0, stream, 
                inputBuffer, outputBuffer, (float*)(kp.ptrs[0]), doneBuffer,
                    flagBuffer, kp, myrank, nranks, iter+1);
        }
        else if(nranks==3) {
            hipLaunchKernelGGL(allred2Peer,
                dim3(min(32,(N)/256)),
                dim3(256),
                0, stream, 
                inputBuffer, outputBuffer, (float*)(kp.ptrs[0]), (float*)(kp.ptrs[1]), doneBuffer,
                    flagBuffer, kp, myrank, nranks, iter+1);
        }
        else if(nranks==4) {
            hipLaunchKernelGGL(allred3Peer,
                dim3(min(32,(N)/256)),
                dim3(256),
                0, stream, 
                inputBuffer, outputBuffer, (float*)(kp.ptrs[0]), (float*)(kp.ptrs[1]), (float*)(kp.ptrs[2]), doneBuffer,
                    flagBuffer, kp, myrank, nranks, iter+1);
        }
        else if(nranks==5) {
            hipLaunchKernelGGL(allred4Peer,
                dim3(min(32,(N)/256)),
                dim3(256),
                0, stream, 
                inputBuffer, outputBuffer, (float*)(kp.ptrs[0]), (float*)(kp.ptrs[1]), (float*)(kp.ptrs[2]), (float*)(kp.ptrs[3]), doneBuffer,
                    flagBuffer, kp, myrank, nranks, iter+1);
        }
        else if(nranks==6) {
            hipLaunchKernelGGL(allred5Peer,
                dim3(min(32,(N)/256)),
                dim3(256),
                0, stream, 
                inputBuffer, outputBuffer, (float*)(kp.ptrs[0]), (float*)(kp.ptrs[1]), (float*)(kp.ptrs[2]), (float*)(kp.ptrs[3]), (float*)(kp.ptrs[4]), doneBuffer,
                    flagBuffer, kp, myrank, nranks, iter+1);
        }
        else if(nranks==7) {
            hipLaunchKernelGGL(allred6Peer,
                dim3(min(32,(N)/256)),
                dim3(256),
                0, stream, 
                inputBuffer, outputBuffer, (float*)(kp.ptrs[0]), (float*)(kp.ptrs[1]), (float*)(kp.ptrs[2]), (float*)(kp.ptrs[3]), (float*)(kp.ptrs[4]), (float*)(kp.ptrs[5]), doneBuffer,
                    flagBuffer, kp, myrank, nranks, iter+1);
        }
        else if(nranks==8) {
            hipLaunchKernelGGL(allred7Peer,
                dim3(min(32,(N)/256)),
                dim3(256),
                0, stream, 
                inputBuffer, outputBuffer, (float*)(kp.ptrs[0]), (float*)(kp.ptrs[1]), (float*)(kp.ptrs[2]), (float*)(kp.ptrs[3]), (float*)(kp.ptrs[4]), (float*)(kp.ptrs[5]), (float*)(kp.ptrs[6]), doneBuffer,
                    flagBuffer, kp, myrank, nranks, iter+1);
        } 
        // else {
        //     cout<<"nranks > 8 used. no allreduce being run."<<endl;
        // }
        // hipLaunchKernelGGL(allredPeer,
        //     dim3(min(32,(N)/256)),
        //     dim3(256),
        //     0, stream, 
        //     inputBuffer, outputBuffer, doneBuffer,
        //         flagBuffer, kp, myrank, nranks, iter+1, numDevices);
    }
    hipStreamSynchronize(stream);

    t2 = std::chrono::high_resolution_clock::now();
    double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    // times = times/100;
    double out;

    /* Get the avg time among ranks */
    MPI_Allreduce(&times, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (myrank == 0) 
        cout<< "Allreduce time for "<< nranks << " ranks = " << out/nranks << " data size = " << N*sizeof(float) << endl; //checks performance


    hipMemcpy(output, outputBuffer, N * sizeof(float), hipMemcpyDeviceToHost);

    float *p = (float*) output;
    bool testRest = true;
    for (int i = 0; i < N; i++) {
    	if (p[i] != nranks*(nranks+1)/2) {
        	cout<< "Wrong result " << p[i] << " at index "<< i << endl; //checks correctness
            testRest = false;
            break;
        }
    }

    /* Cleanup */
    hipFree(inputBuffer);
    hipFree(outputBuffer);
    free(output);

    if(testRest) std::cout<<"Passed!\n";
    else std::cout<<"Failed!\n";


    MPI_Finalize();

    return SUCCESS;
}

// Sample output:
// Allred time for 8 ranks = 1.41391e-05 data size = 40960
// Passed!
// Passed!
// Passed!
// Passed!
// Passed!
// Passed!
// Passed!
// Passed!