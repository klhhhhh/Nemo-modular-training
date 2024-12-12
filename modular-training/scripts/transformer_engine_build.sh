export CUDNN_PATH=/global/common/software/nersc9/cudnn/8.9.3-cuda12
export CUDA_PATH=/global/homes/k/klhhhhh/cuda12.4
export CPATH=/global/common/software/nersc9/cudnn/8.9.3-cuda12/include:$CPATH
export PATH=/global/homes/k/klhhhhh/cuda12.4/bin:$PATH
export CUDA_HOME=/global/homes/k/klhhhhh/cuda12.4
export LD_LIBRARY_PATH=/global/homes/k/klhhhhh/cuda12.4/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/global/common/software/nersc9/cudnn/8.9.3-cuda12/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/global/common/software/nersc/pe/gpu/gnu/openmpi/5.0.0/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/global/common/software/nersc/pe/gpu/gnu/openmpi/5.0.0/lib/libmpi.so


export CUDACXX=/global/homes/k/klhhhhh/cuda12.4/bin/nvcc

git clone https://github.com/NVIDIA/TransformerEngine.git && \
cd TransformerEngine && \
git checkout $te_commit && \
git submodule init && git submodule update && \

export CXXFLAGS="-pthread"
export LDFLAGS="-pthread"
module load openmpi/5.0.0

NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/global/common/software/nersc/pe/gpu/gnu/openmpi/5.0.0/ pip install . --no-build-isolation