#define interactionRange2 (1.0) 
#define MaxThreads (128)

#include "subVicsek.c" 

int main(int argc, char *argv[])
{
    if(argc!=4) error_output("command L tmax delt") ;
    const int    Lsize  = atoi(argv[1]);
    const int    tmax   = atoi(argv[2]);
    const int    delt   = atoi(argv[3]);
    const float  rho    = 1.0;
    const float  speed  = 0.5;

    // total number of particles
    const  int   ptlsNum = (int)(Lsize*Lsize*rho+0.01) ;
    // total number of cells
    const  int   cllsNum    = Lsize*Lsize ;

    // grid dimension
    const int nThreads = (MaxThreads<ptlsNum)? MaxThreads : ptlsNum;
    const int nBlocks  = (ptlsNum+nThreads-1)/nThreads; 

    struct VicsekParticle *devPtls ;
    // VicsekParticle in the device
    cudaMalloc(&devPtls, sizeof(struct VicsekParticle)*ptlsNum) ;

    // auxiliary memory for linked lists
    // linked list is managed with the THRUST library
    // corresponding device memory
    int *devCell, *devHead, *devTail ;
    cudaMalloc(&devCell, sizeof(int)*ptlsNum);
    cudaMalloc(&devHead, sizeof(int)*cllsNum);
    cudaMalloc(&devTail, sizeof(int)*cllsNum);

    // temporary angle variable
    float *devAngTmp;
    cudaMalloc(&devAngTmp, sizeof(float)*ptlsNum);

    // order parameter measurement
    float *devVx, *devVy;
    cudaMalloc(&devVx, sizeof(float)*ptlsNum);
    cudaMalloc(&devVy, sizeof(float)*ptlsNum);

    // set the PRNG seed with the device random number
    std::random_device rd;
    unsigned int seed = rd();
    // seed = 1234;
    // initialize the PRNGs
    curandState *devStates ;
    cudaMalloc((void **)&devStates, ptlsNum*sizeof(curandState)) ;
    initialize_prng<<<nBlocks, nThreads>>>(ptlsNum, seed, devStates) ;

    float noise = 0.5;
    printf("# L noise t Ox=<cos angle> Oy=<sin angle>\n"); 
    // for(float noise=0.1; noise<1.0; noise += 0.1) {
        float noise_pi = noise*ppi;

        // random initial configuration
        init_random_config<<<nBlocks,nThreads>>>(devStates, Lsize, 
                ptlsNum, devPtls) ;

        for(int t=1; t<=tmax; t++) {
            // position and angle update
            particles_move<<<nBlocks, nThreads>>>(devPtls, Lsize, ptlsNum,
                    speed); 

            // linked list
            linked_list(devPtls, Lsize, ptlsNum, cllsNum, devCell, devHead, devTail,
                    nBlocks, nThreads);
        
            // direction angle of local average velocity
            local_average_velocity<<<nBlocks, nThreads>>>(devPtls,
                    Lsize, ptlsNum, devAngTmp,devHead, devTail);

            // rotate the moving direction
            particles_rotate<<<nBlocks, nThreads>>>(devPtls, devStates, 
                    devAngTmp, ptlsNum, noise_pi);

            if(t%delt==0) {
                float odx, ody;
                get_orderParameter(devPtls, ptlsNum, devVx, devVy,
                        &odx, &ody, nBlocks, nThreads);
                printf("%8d %18e %12d %+.12e %+.12e\n", Lsize, noise, t, 
                        odx/(double)ptlsNum, ody/(double)ptlsNum);
            }
        }
    //}

    cudaFree(devPtls) ; cudaFree(devStates); cudaFree(devAngTmp);
    cudaFree(devCell) ; cudaFree(devHead) ; cudaFree(devTail) ; 
}
