

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern "C"
int setup(int argc, char** argv);

extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;


typedef struct result{
	float frag;
	size_t large_blank;
	size_t sum;
}result_t;

////////////////////////////////////////////////////////////////////////////////
// Calculate Fragmentation
////////////////////////////////////////////////////////////////////////////////
result_t calculate_array_fragmentation(int* array, size_t size) {
	size_t large_blank = 0, blank = 0, sum = 0;
	result_t res;

	for (int i = 0; i < size; i++) {
		if (array[i] == 0) {
			sum += 1;
			blank +=1;
		} else {
			if(blank > large_blank) {
				large_blank = blank;
				blank = 0;
			} else {
				blank = 0;
			}
		}
	}
	
	// the blank is the last part of the array
	if(blank > large_blank) {
		large_blank = blank;
	} 

	res.large_blank = large_blank;
	res.sum = sum;

	// no blank chunk
	if (sum == 0) {
		res.frag = 0.0;
		return res;
	}

	// only one blank chunk
	if (large_blank == sum) {
		res.frag = 0.0;
		return res;
	}

	res.frag = 1 - (float) large_blank / (float) sum;

	return res;
	
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
#ifdef GPU  
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 16;  
  dim3  grid( 1 , num_blocks);
  dim3  threads(16 , 16);
  
  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));
 
  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {	
   for (int j = 0; j <= hid; j++) {
	  input_weights_one_dim[m] = net->input_weights[k][j];
	  input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
	  m++;
    }
  }
  
  cudaMalloc((void**) &input_cuda, (in + 1) * sizeof(float));
  printf("Array: input_cuda, size: %luB\n", (in + 1) * sizeof(float));
  // ############################################################
	int * input_cuda_t;
	int * h_input_cuda_t = (int*) malloc((in + 1) * sizeof(int));
	cudaMalloc( (void**) &input_cuda_t, (in + 1) * sizeof(float)) ;
	cudaMemset(input_cuda_t, 0, (in + 1) * sizeof(float));
	// ############################################################

  
  cudaMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
  printf("Array: output_hidden_cuda, size: %luB\n", (hid + 1) * sizeof(float));
  // ############################################################
	int * output_hidden_cuda_t;
	int * h_output_hidden_cuda_t = (int*) malloc((hid + 1) * sizeof(int));
	cudaMalloc( (void**) &output_hidden_cuda_t, (hid + 1) * sizeof(float)) ;
	cudaMemset(output_hidden_cuda_t, 0, (hid + 1) * sizeof(float));
	// ############################################################


  cudaMalloc((void**) &input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  printf("Array: input_hidden_cuda, size: %luB\n", (in + 1) * (hid + 1) * sizeof(float));
  // ############################################################
	int * input_hidden_cuda_t;
	int * h_input_hidden_cuda_t = (int*) malloc((in + 1) * (hid + 1) * sizeof(int));
	cudaMalloc( (void**) &input_hidden_cuda_t, (in + 1) * (hid + 1) * sizeof(float)) ;
	cudaMemset(input_hidden_cuda_t, 0, (in + 1) * (hid + 1) * sizeof(float));
	// ############################################################

  cudaMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
  printf("Array: hidden_partial_sum, size: %luB\n", num_blocks * WIDTH * sizeof(float));
  // ############################################################
	int * hidden_partial_sum_t;
	int * h_hidden_partial_sum_t = (int*) malloc(num_blocks * WIDTH * sizeof(int));
	cudaMalloc( (void**) &hidden_partial_sum_t, num_blocks * WIDTH * sizeof(float)) ;
	cudaMemset(hidden_partial_sum_t, 0, num_blocks * WIDTH * sizeof(float));
	// ############################################################
  
  
#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);

#endif

#ifdef GPU
 
  printf("Performing GPU computation\n");
  
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
  
  cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

  
  
  bpnn_layerforward_CUDA<<< grid, threads >>>(input_cuda, input_cuda_t,
	                                          output_hidden_cuda, output_hidden_cuda_t,
											  input_hidden_cuda, input_hidden_cuda_t,
											  hidden_partial_sum, hidden_partial_sum_t,
											  in,
											  hid);
 
//  cudaThreadSynchronize();
//  
//  cudaError_t error = cudaGetLastError();
//	if (error != cudaSuccess) {
//		printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
//		exit(EXIT_FAILURE);
//	}
  
  cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
     
  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {	
      sum += partial_sum[k * hid + j-1] ;
    }
	sum += net->input_weights[0][j];
	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
  #endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#endif  


#ifdef GPU

  cudaMalloc((void**) &hidden_delta_cuda, (hid + 1) * sizeof(float));
  printf("Array: hidden_delta_cuda, size: %luB\n", (hid + 1) * sizeof(float));
  // ############################################################
	int * hidden_delta_cuda_t;
	int * h_hidden_delta_cuda_t = (int*) malloc((hid + 1) * sizeof(int));
	cudaMalloc( (void**) &hidden_delta_cuda_t, (hid + 1) * sizeof(float)) ;
	cudaMemset(hidden_delta_cuda_t, 0, (hid + 1) * sizeof(float));
	// ############################################################


  cudaMalloc((void**) &input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));
  printf("Array: input_prev_weights_cuda, size: %luB\n", (in + 1) * (hid + 1) * sizeof(float));
  // ############################################################
	int * input_prev_weights_cuda_t;
	int * h_input_prev_weights_cuda_t = (int*) malloc((in + 1) * (hid + 1) * sizeof(int));
	cudaMalloc( (void**) &input_prev_weights_cuda_t, (in + 1) * (hid + 1) * sizeof(float)) ;
	cudaMemset(input_prev_weights_cuda_t, 0, (in + 1) * (hid + 1) * sizeof(float));
	// ############################################################



  cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);


  bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda, hidden_delta_cuda_t, 
												hid, 
												input_cuda, input_cuda_t, 
												in,
												input_hidden_cuda, input_hidden_cuda_t,
												input_prev_weights_cuda, input_prev_weights_cuda_t
												);

  cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(input_weights_one_dim, input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);


  cudaMemcpy(h_input_cuda_t, input_cuda_t, (in + 1) * sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output_hidden_cuda_t, output_hidden_cuda_t, (hid + 1) * sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(h_input_hidden_cuda_t, input_hidden_cuda_t, (in + 1) * (hid + 1) * sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(h_hidden_partial_sum_t, hidden_partial_sum_t, num_blocks * WIDTH * sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(h_hidden_delta_cuda_t, hidden_delta_cuda_t, (hid + 1) * sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(h_input_prev_weights_cuda_t, input_prev_weights_cuda_t, (in + 1) * (hid + 1) * sizeof(int),cudaMemcpyDeviceToHost);


  printf("###############################################################################################\n");
	result_t result;
	result = calculate_array_fragmentation(h_input_cuda_t, in + 1);
	printf("input_cuda;	       \t size: %luB,\t large: %luB,\t sum: %luB,\t frag: %f\n", 
	(in + 1) * sizeof(float), result.large_blank*sizeof(float), result.sum*sizeof(float), result.frag);

	result = calculate_array_fragmentation(h_output_hidden_cuda_t, hid + 1);
	printf("output_hidden_cuda;    \t size: %luB,\t large: %luB,\t sum: %luB,\t frag: %f (unused)\n", 
	(hid + 1) * sizeof(float), result.large_blank*sizeof(float), result.sum*sizeof(float), result.frag);

  result = calculate_array_fragmentation(h_input_hidden_cuda_t, (in + 1) * (hid + 1));
	printf("input_hidden_cuda;     \t size: %luB, large: %luB,\t sum: %luB,\t frag: %f\n", 
	(in + 1) * (hid + 1) * sizeof(float), result.large_blank*sizeof(float), result.sum*sizeof(float), result.frag);

  result = calculate_array_fragmentation(h_hidden_partial_sum_t, num_blocks * WIDTH);
	printf("hidden_partial_sum;    \t size: %luB,\t large: %luB,\t sum: %luB,\t frag: %f\n", 
	(hid + 1) * sizeof(float), result.large_blank*sizeof(float), result.sum*sizeof(float), result.frag);

  result = calculate_array_fragmentation(h_hidden_delta_cuda_t, (in + 1) * (hid + 1));
	printf("hidden_delta_cuda;     \t size: %luB,\t large: %luB,\t sum: %luB,\t frag: %f\n", 
	num_blocks * WIDTH * sizeof(float), result.large_blank*sizeof(float), result.sum*sizeof(float), result.frag);

  result = calculate_array_fragmentation(h_input_prev_weights_cuda_t, hid + 1);
	printf("input_prev_weights_cuda; size: %luB, large: %luB,\t sum: %luB,\t frag: %f\n", 
	(in + 1) * (hid + 1) * sizeof(float), result.large_blank*sizeof(float), result.sum*sizeof(float), result.frag);
	
	printf("###############################################################################################\n");



  cudaFree(input_cuda_t);
  cudaFree(output_hidden_cuda_t); // not used in kernel1
  cudaFree(input_hidden_cuda_t);
  cudaFree(hidden_partial_sum_t);
  cudaFree(input_prev_weights_cuda_t);
  cudaFree(hidden_delta_cuda_t);

  free(h_input_cuda_t);
  free(h_output_hidden_cuda_t);
  free(h_input_hidden_cuda_t);
  free(h_hidden_partial_sum_t);
  free(h_input_prev_weights_cuda_t);
  free(h_hidden_delta_cuda_t);


  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);
  
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);

#endif   
  
  
  

}
