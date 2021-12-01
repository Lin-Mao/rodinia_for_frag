/*********************************************************************************
Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
OTHERWISE.

The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish
**********************************************************************************/
#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void
Kernel( Node* g_graph_nodes, int* g_graph_nodes_t, int* g_graph_edges, int* g_graph_edges_t, 
bool* g_graph_mask, int* g_graph_mask_t, bool* g_updating_graph_mask, int* g_updating_graph_mask_t, 
bool *g_graph_visited, int* g_graph_visited_t, int* g_cost, int* g_cost_t, int no_of_nodes) 
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid<no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid]=false;
		g_graph_mask_t[tid] = 1;

		g_graph_nodes_t[tid] = 1;
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++) {
			int id = g_graph_edges[i];
			g_graph_edges_t[i] = 1;


			if(!g_graph_visited[id]) {
					g_cost[id]=g_cost[tid]+1;
					g_cost_t[id] = 1;
					g_cost_t[tid] = 1;
					g_updating_graph_mask[id]=true;
					g_updating_graph_mask_t[id] = 1;
			}
			g_graph_visited_t[id] = 1;
		}
	}
}

#endif 
