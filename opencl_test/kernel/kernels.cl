
// =================================================================================================
// Project: 
// Exploring the performance of general matrix-multiplication on an NVIDIA Tesla K40m GPU.
//
// File information:
// Institution.... SURFsara <www.surfsara.nl>
// Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Changed at..... 2014-11-06
// License........ MIT license
// Tab-size....... 4 spaces
// Line length.... 100 characters
//
// =================================================================================================
//
// Matrices in column-major format
// A: K columns, M rows
// B: N columns, K rows
// C: N columns, M rows
//                         
//                   N     
//                o-----o  
//                |     |  
//              K | [B] |  
//                |     |  
//                o-----o  
//        K          N     
//    o-------o   o-----o  
//  M |  [A]  | M | [C] |  
//    |       |   |     |  
//    o-------o   o-----o  
//                         
//
// C-code for column-major matrix multiplication with alpha=1 and beta=0:
//
// for (int m=0; m<M; m++) {
//     for (int n=0; n<N; n++) {
//         float acc = 0.0f;
//         for (int k=0; k<K; k++) {
//             acc += A[k*M + m] * B[n*K + k];
//         }
//         C[n*M + m] = acc;
//     }
// }
//
// =================================================================================================

//#if KERNEL == 1

// First naive implementation
__kernel void myGEMM1(const int M, const int K, const int N, int img_n,
                      const __global float* A, const __global float* B, const __global float *C,
                      __global float* D, __global float * subA, __global float * subB) {
    
    // Thread identifiers
    const int row = get_local_id(0); //0 ~TS_h =[OH*OW]
    const int col = get_local_id(1); //1

    const int globalRow = TS_h*get_group_id(0)+row; // Row ID of C (0..M)
    const int globalCol = TS_w*get_group_id(1)+col; // Col ID of C (0..N)

    int img_size = M/img_n;
    int A_step = img_size *K;
    int C_step = img_size *N;

    
    for(int iter_n =0;iter_n<img_n;iter_n++){
        
            for(int k=0; k<K;k++){
               subA[globalRow*K+k] = A[globalRow*K+k+A_step*iter_n];
               subB[k*N+globalCol] = B[k*N+globalCol];
           }

           //Synchronise to make sure the tile is loaded
           barrier(CLK_LOCAL_MEM_FENCE);

           float acc =0.0f;
           for(int k=0;k<K;k++){
               acc += A[globalRow * K +k +A_step*iter_n]*B[k*N+globalCol];
           }
          
           //Synchronise to make sure the tile is loaded
           barrier(CLK_LOCAL_MEM_FENCE);

           float tmp_val = acc + C[globalCol];
           tmp_val = (tmp_val>0)? tmp_val:0;
           D[globalCol*img_size+globalRow+C_step*iter_n] = tmp_val;

           //Synchronise to make sure the tile is loaded
           barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// =================================================================================================

__kernel void im2col_gpu_kernel(const int img_n, const int img_c, const int img_h,const int img_w, const int col_h, const int col_w,
        const int ksize, const int pad, const int stride, __global float * data_im, __global float * data_col){

    const int localrow = get_local_id(0);
    const int localcol = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    int img_size = img_c * img_h * img_w;//[C*H*W]
    int col_height = img_c * ksize*ksize;//[C*FH*FW]
    int col_step =col_h * col_w; //[OH*OW]
    int col_size = col_h*col_w*col_height; //[OH*OW*C*FH*FW]

    for(int iter=0;iter<img_n;iter++)
    {
        int w_offset = globalRow % ksize;
        int h_offset = (globalRow/ksize)%ksize;
        int c_im = globalRow/ksize/ksize;

        int row = DIV2(globalCol,col_w);
        int col = MOD2(globalCol,col_w);

        int im_row = h_offset +row*stride -pad;
        int im_col = w_offset +col*stride -pad;
        int col_index = (globalRow * col_h +row)*col_w+col;
        float tmp_val = (im_row>=0 && im_col>=0 && im_row < img_h && im_col<img_w)?
                data_im[im_col+img_w*(im_row+img_h*c_im)+img_size*iter]:0;

        int row_t = col_index/col_step;
        int col_t = col_index%col_step;

        data_col[col_t*col_height+row_t+col_size*iter] = tmp_val;

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// =================================================================================================
__kernel void maxpool_gpu_kernel(const int col_height, const int col_h, const int col_w, const int img_c, const int kernel_size, 
        __global float * data_col, __global float * max_pool_out)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    int img_n = col_height/(col_h*col_w);
    int in_step = kernel_size * img_c;
    int in_size = kernel_size * img_c*col_h*col_w;
    int out_step = img_c;
    int out_size = img_c * col_h*col_w;
    int out_t_step = col_h*col_w;

    for(int iter_n =0; iter_n<img_n ;iter_n++)
    {
        float max_val = -1.0;
        for(int iter_row =0; iter_row < kernel_size;iter_row++) {
            float tmp_val = data_col[globalRow*in_step + globalCol*kernel_size+iter_row+iter_n*in_size];
            if(max_val < tmp_val)
            max_val =tmp_val;

        }

        barrier(CLK_LOCAL_MEM_FENCE);
        int col_index = globalRow*out_step+globalCol;

        int row_t = col_index/out_step;
        int col_t = col_index%out_step;
        max_pool_out[col_t*out_t_step + row_t + iter_n*out_size] = max_val;

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void myGEMM2(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C, __global float* subA, __global float* subB) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation register
    float acc = 0.0f;
    
	// subA[globalRow*K + globalCol] = A[globalRow*K + globalCol];
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t< numTiles ; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[row][col] = A[globalRow*K + tiledCol];
        Bsub[row][col] = B[tiledRow*N + globalCol];
		subA[globalRow*K + tiledCol] = A[globalRow*K + tiledCol];
		subB[tiledRow*N + globalCol] = B[tiledRow*N + globalCol];
		

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
		//C[globalRow*N + globalCol] = Asub[row][col] * Bsub[row][col];

        for (int k=0; k<TS; k++) {
			//acc += A[globalRow*K + k] * B[k*N + globalCol];
			acc += Asub[row][k] * Bsub[k][col];
			//acc += subA[row*K + k] * B[k*N + tiledCol];
        }


        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
		
    }

	C[globalRow*N + globalCol] = acc;

	//C[row*N + col] = acc;
	//C[row*N + col] = acc;
    // Store the final result in C
    //C[globalRow*N + globalCol] = acc;
}

/*
// Tiled and coalesced version
__kernel void myGEMM2(const int M, const int N, const int K,
	const __global float* A,
	const __global float* B,
	__global float* C) {

	// Thread identifiers
	const int row = get_local_id(0); // Local row ID (max: TS)
	const int col = get_local_id(1); // Local col ID (max: TS)
	const int globalRow = TS * get_group_id(0) + row; // Row ID of C (0..M)
	const int globalCol = TS * get_group_id(1) + col; // Col ID of C (0..N)

	// Local memory to fit a tile of TS*TS elements of A and B
	__local float Asub[TS][TS];
	__local float Bsub[TS][TS];

	// Initialise the accumulation register
	float acc = 0.0f;

	// Loop over all tiles
	const int numTiles = K / TS;
	for (int t = 0; t < numTiles; t++) {

		// Load one tile of A and B into local memory
		const int tiledRow = TS * t + row;
		const int tiledCol = TS * t + col;
		Asub[col][row] = A[tiledCol*M + globalRow];
		Bsub[col][row] = B[globalCol*K + tiledRow];

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k = 0; k < TS; k++) {
			acc += Asub[k][row] * Bsub[col][k];
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the final result in C
	C[globalCol*M + globalRow] = acc;
}
*/

//#endif
// =================================================================================================

#define WPT 8                        // The amount of work-per-thread, i.e. the thread-coarsening factor
#define RTS (TS/WPT)                 // The reduced tile-size in one dimension

// Increased the amount of work-per-thread by a factor WPT
__kernel void myGEMM3(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {


		//const int tiledRow = TS * t + row;
		//const int tiledCol = TS * t + col;
		//Asub[row][col] = A[globalRow*K + tiledCol];
		//Bsub[row][col] = B[tiledRow*N + globalCol];

        // Load one tile of A and B into local memory
        for (int w=0; w<WPT; w++) {
            const int tiledRow = TS*t + row;
            const int tiledCol = TS*t + col;
			Asub[row][col + w * RTS] = A[globalRow*K + (tiledCol + w * RTS)];
			Bsub[row][col + w * RTS] = B[tiledRow*N + (globalCol + w * RTS)];
            //Asub[col + w*RTS][row] = A[(tiledCol + w*RTS)*M + globalRow];
            //Bsub[col + w*RTS][row] = B[(globalCol + w*RTS)*K + tiledRow];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

		//for (int k = 0; k < TS; k++) {
		//	acc += Asub[row][k] * Bsub[k][col];
		//}

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        //C[(globalCol + w*RTS)*M + globalRow] = acc[w];
		C[globalRow*N + (globalCol + w * RTS)] = acc[w];
    }
}
/*
// Increased the amount of work-per-thread by a factor WPT
__kernel void myGEMM3(const int M, const int N, const int K,
	const __global float* A,
	const __global float* B,
	__global float* C) {

	// Thread identifiers
	const int row = get_local_id(0); // Local row ID (max: TS)
	const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
	const int globalRow = TS * get_group_id(0) + row; // Row ID of C (0..M)
	const int globalCol = TS * get_group_id(1) + col; // Col ID of C (0..N)

	// Local memory to fit a tile of TS*TS elements of A and B
	__local float Asub[TS][TS];
	__local float Bsub[TS][TS];

	// Initialise the accumulation registers
	float acc[WPT];
	for (int w = 0; w < WPT; w++) {
		acc[w] = 0.0f;
	}

	// Loop over all tiles
	const int numTiles = K / TS;
	for (int t = 0; t < numTiles; t++) {

		// Load one tile of A and B into local memory
		for (int w = 0; w < WPT; w++) {
			const int tiledRow = TS * t + row;
			const int tiledCol = TS * t + col;
			Asub[col + w * RTS][row] = A[(tiledCol + w * RTS)*M + globalRow];
			Bsub[col + w * RTS][row] = B[(globalCol + w * RTS)*K + tiledRow];
		}

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k = 0; k < TS; k++) {
			for (int w = 0; w < WPT; w++) {
				acc[w] += Asub[k][row] * Bsub[col + w * RTS][k];
			}
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the final results in C
	for (int w = 0; w < WPT; w++) {
		C[(globalCol + w * RTS)*M + globalRow] = acc[w];
	}
}
*/

// =================================================================================================
#if KERNEL == 6

// Use 2D register blocking (further increase in work per thread)
__kernel void myGEMM6(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK+2];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {

        // Load one tile of A and B into local memory
        #pragma unroll
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = MOD2(id,TSM);
            int col = DIV2(id,TSM);
            int tiledIndex = TSK*t + col;
            Asub[col][row] = A[tiledIndex*M + offsetM + row];
            Bsub[row][col] = B[tiledIndex*N + offsetN + row];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            #pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[col][k];
            }

            // Perform the computation
            #pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

#endif
// =================================================================================================


// =================================================================================================

/*
// Simple transpose kernel for a P * Q matrix
__kernel void transpose(const int P, const int Q,
                        const __global float* input,
                        __global float* output) {
    
    // Thread identifiers
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
    const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q

    // Set-up the local memory for shuffling
    __local float buffer[TRANSPOSEX][TRANSPOSEY];

    // Swap the x and y coordinates to perform the rotation (coalesced)
    if (ID0 < P && ID1 < Q) {
        buffer[ty][tx] = input[ID1*P + ID0];
    }

    // Synchronise all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    // We don't have to swap the x and y thread indices here,
    // because that's already done in the local memory
    const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
    const int newID1 = get_group_id(0)*TRANSPOSEX + ty;

    // Store the transposed result (coalesced)
    if (newID0 < Q && newID1 < P) {
        output[newID1*Q + newID0] = buffer[tx][ty];
    }
}
*/

// =================================================================================================
/*
// Pad the P * Q matrix with zeroes to form a P_XL * Q_XL matrix
__kernel void paddingAddZeroes(const int P, const int Q,
                               const __global float* input,
                               const int P_XL, const int Q_XL,
                               __global float* output) {
    
    // Thread identifiers
    const int tx = get_group_id(0)*PADDINGX + get_local_id(0); // 0..P_XL in blocks of PADDINGX
    const int ty = get_group_id(1)*PADDINGY + get_local_id(1); // 0..Q_XL in blocks of PADDINGY

    // Check whether we are within bounds of the XL matrix
    if (tx < P_XL && ty < Q_XL) {

        // Copy the input or pad a zero
        float value;
        if (tx < P && ty < Q) {
            value = input[ty*P + tx];
        }
        else {
            value = 0.0f;
        }

        // Store the result
        output[ty*P_XL + tx] = value;
    }
}
*/
// =================================================================================================
/*
// Remove padded values from a P_XL * Q_XL matrix to form a P * Q matrix
__kernel void paddingRemoveZeroes(const int P_XL, const int Q_XL,
                                  const __global float* input,
                                  const int P, const int Q,
                                  __global float* output) {
    
    // Thread identifiers
    const int tx = get_group_id(0)*PADDINGX + get_local_id(0); // 0..P in blocks of PADDINGX
    const int ty = get_group_id(1)*PADDINGY + get_local_id(1); // 0..Q in blocks of PADDINGY


    // Only store the result if within P * Q bounds
    if (tx < P && ty < Q) {
        output[ty*P + tx] = input[ty*P_XL + tx];
    }
}
*/





// =================================================================================================


// =================================================================================================


