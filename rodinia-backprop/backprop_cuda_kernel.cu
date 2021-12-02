

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"


__global__ void
bpnn_layerforward_CUDA(float *input_cuda, int* input_cuda_t,
	                   float *output_hidden_cuda, int* output_hidden_cuda_t,
					   float *input_hidden_cuda, int* input_hidden_cuda_t,
					   float *hidden_partial_sum, int* hidden_partial_sum_t,
					   int in,
					   int hid) 
{
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

   int index_in = HEIGHT * by + ty + 1;
   
   __shared__ float input_node[HEIGHT];
   __shared__ float weight_matrix[HEIGHT][WIDTH];


   if ( tx == 0 )
   input_node[ty] = input_cuda[index_in] ;
   input_cuda_t[index_in] = 1;
   
   __syncthreads();

   weight_matrix[ty][tx] = input_hidden_cuda[index];
   input_hidden_cuda_t[index] = 1;

   __syncthreads();
   
   weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

   __syncthreads();   
   
   for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){
 
	   int power_two = __powf(2, i);

	   if( ty % power_two == 0 )
	   weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

	   __syncthreads();

   }
   
   //__syncthreads();

   input_hidden_cuda[index] = weight_matrix[ty][tx];
   input_hidden_cuda_t[index] = 1;
   
/*
   for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){
 
	   unsigned int power_two = i - 1;

	   if( (ty & power_two) == 0 ) {
		weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
	   }

   }
   */

   __syncthreads();

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
      hidden_partial_sum_t[by * hid + ty] = 1;
   }

}


__global__ void bpnn_adjust_weights_cuda(float * delta, int* delta_t,  
										 int hid,         
										 float * ly, int* ly_t,     
										 int in,          
										 float * w, int* w_t,      
										 float * oldw, int* oldw_t)  									
{
  
  
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;
	
   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   w_t[index] = 1;

   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   delta_t[index_x] = 1;
   ly_t[index_y] = 1;
   oldw_t[index] = 1;

   __syncthreads();

   if (ty == 0 && by ==0){
     w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
     w_t[index_x] = 1;

     oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
     delta_t[index_x] = 1;
     oldw_t[index_x] = 1;
   }
}
#endif 
