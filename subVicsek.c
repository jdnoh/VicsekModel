#include <math.h>
#include <unistd.h>
#include <random>
#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <cuda_runtime.h>

#define two_ppi (6.28318530717958648)
#define ppi (3.14159265358979324)

struct VicsekParticle {
    float px, py, angle;
} ;

void error_output(const char *desc)
{
    printf("%s\n", desc) ; exit(-1) ;
}

// initializing RNG for all threads with the same seed
// each state setup will be the state after 2^{67}*tid calls 
__global__ void initialize_prng(const int ptlsNum, 
        unsigned int seed, curandState *state)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum)
        curand_init(seed, tid, 0, &state[tid]) ;
}

__global__ void init_random_config(curandState *state, const int lsize, 
        const int ptlsNum, struct VicsekParticle *ptls) 
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum) {
        curandState localState = state[tid] ;
        ptls[tid].px = curand_uniform(&localState)*lsize ;
        ptls[tid].py = curand_uniform(&localState)*lsize ;
        ptls[tid].angle = two_ppi*curand_uniform(&localState) ;
        state[tid] = localState ;
    }
}

__global__ void particles_move(struct VicsekParticle *ptls,
        const int lsize, const int ptlsNum, const float speed)
{
    // particle index
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum) {
        // updating X
        ptls[tid].px += speed*cosf(ptls[tid].angle) ;
        if(ptls[tid].px < 0.0)    ptls[tid].px += lsize ;
        if(ptls[tid].px >= lsize) ptls[tid].px -= lsize ;
        // updating Y
        ptls[tid].py += speed*sinf(ptls[tid].angle) ;
        if(ptls[tid].py < 0.0)    ptls[tid].py += lsize ;
        if(ptls[tid].py >= lsize) ptls[tid].py -= lsize ;
    }
}

__global__ void particles_rotate(struct VicsekParticle *ptls, 
        curandState *state, const float *angTmp, 
        const int ptlsNum, const float noise_pi)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum) {
        ptls[tid].angle = angTmp[tid] + 
            noise_pi*(2.*curand_uniform(&state[tid])-1.0) ;
    }
}

__global__ void local_average_velocity(struct VicsekParticle *ptls,
        const int lsize, const int ptlsNum, float *angTmp,
        int *cellHead, int *cellTail)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid < ptlsNum) {
        // my position
        float x = ptls[tid].px, y = ptls[tid].py ;
        // velocity of neigobors
        float vx=0.0, vy = 0.0;
        float dx, dy;
        for(int a=(int)x-1; a<=(int)x+1; a++) {
            for(int b=(int)y-1; b<=(int)y+1; b++) {
                // zz : index for neighboring cells
                int zz = (a+lsize)%lsize + ((b+lsize)%lsize)*lsize ;
                for(int k=cellHead[zz]; k<=cellTail[zz]; k++) {
                    // loop over particles in the cell zz
                    dx = fabsf(x-ptls[k].px) ;
                    if(dx>lsize/2.) dx = lsize-dx ;
                    dy = fabsf(y-ptls[k].py) ;
                    if(dy>lsize/2.) dy = lsize-dy ;
                    if(dx*dx+dy*dy < interactionRange2) {
                        vx += cosf(ptls[k].angle);
                        vy += sinf(ptls[k].angle);
                    }
                }
            }
        }
        // direction angle of local average velocity
        angTmp[tid] = atan2f(vy, vx);
    }
}

// make a table "cell[i]" for the cell index for a particle i
__global__ void find_address(struct VicsekParticle *ptls, 
        const int lsize, const int ptlsNum, int *cell)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum) {
         cell[tid] = ((int)ptls[tid].px)%lsize 
                    + lsize*(((int)ptls[tid].py)%lsize) ;
    }
}

// make tables "cellHead[c]" and "cellTail[c]" for the index 
// of the first and the last praticle in a cell c
// empty cells are not updated
__global__ void cell_head_tail(int ptlsNum, int *cell, 
        int *cellHead, int *cellTail)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum) {
        if(tid==0) cellHead[cell[tid]] = tid ;
        else {
            if(cell[tid]!=cell[tid-1]) cellHead[cell[tid]] = tid ;
        }
        if(tid==ptlsNum-1) cellTail[cell[tid]] = tid ;
        else {
            if(cell[tid]!=cell[tid+1]) cellTail[cell[tid]] = tid ;
        }
    }
}

__global__ void orderParameter(struct VicsekParticle *ptls,
        const int ptlsNum, float *vx, float *vy)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum) {
        vx[tid] = cosf(ptls[tid].angle);
        vy[tid] = sinf(ptls[tid].angle);
    }
}

void linked_list(struct VicsekParticle *ptls, const int lsize, 
        const int ptlsNum, const int cllsNum, int *cell,  
        int *head, int *tail, int nBlocks, int nThreads)
{
    // cell[ptl] = cell index of a particle
    find_address<<<nBlocks, nThreads>>>(ptls, lsize, ptlsNum, cell);
    // sort particles w.r.t the cell index
    thrust::sort_by_key(thrust::device_ptr<int>(cell),
                thrust::device_ptr<int>(cell)+ptlsNum,
                thrust::device_ptr<struct VicsekParticle>(ptls));
    thrust::fill(thrust::device_ptr<int>(head),
            thrust::device_ptr<int>(head)+cllsNum, 0);
    thrust::fill(thrust::device_ptr<int>(tail),
            thrust::device_ptr<int>(tail)+cllsNum, -1);
    // find the first (head) and the last (tail)  particle indices in each cell
    // head = -1 and tail = 0 for the empty cell
    cell_head_tail<<<nBlocks, nThreads>>>(ptlsNum, cell, head, tail);
}

void get_orderParameter(struct VicsekParticle *ptls, 
        const int ptlsNum, float *vx, float *vy, 
        float *odx, float *ody, const int nBlocks, const int nThreads)
{
    orderParameter<<<nBlocks, nThreads>>>(ptls, ptlsNum, vx, vy);
    *odx = thrust::reduce(thrust::device_ptr<float>(vx),
            thrust::device_ptr<float>(vx)+ptlsNum, 0.0, 
            thrust::plus<float>());
    *ody = thrust::reduce(thrust::device_ptr<float>(vy),
            thrust::device_ptr<float>(vy)+ptlsNum, 0.0, 
            thrust::plus<float>());
}
