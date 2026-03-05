#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# 0. Check Environment
if [[ "$CONDA_DEFAULT_ENV" != "rewis3dModel" ]]; then
    echo "Error: Please activate the 'rewis3dModel' environment first."
    echo "Run: mamba activate rewis3dModel"
    exit 1
fi

echo "--- Setting up Build Environment ---"

# 1. Fix Cross-Device Link Errors
# We force build artifacts to stay on the same drive as the environment
mkdir -p "$CONDA_PREFIX/tmp"
export TMPDIR="$CONDA_PREFIX/tmp"
echo "TMPDIR set to: $TMPDIR"

# 2. Fix Missing Headers (cuda_runtime.h, cusparse.h, dense_hash_map)
# We add both the standard include path and the target-specific include path
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export PATH=$CONDA_PREFIX/bin:$PATH

# 3. Install PyTorch 2.6 (CUDA 12.4)
echo "--- Installing PyTorch 2.6 ---"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 4. Downgrade Setuptools (Fix for CLIP)
# CLIP relies on 'pkg_resources' which was removed in setuptools>=70
echo "--- Downgrading setuptools for legacy compatibility ---"
pip install "setuptools<70"

# 5. Install CLIP & Flash Attention
echo "--- Installing Complex Extensions ---"
# --no-build-isolation is crucial so they see our downgraded setuptools and env headers
pip install --no-build-isolation git+https://github.com/openai/CLIP.git

# Flash Attention 2.7.4 is pinned because 2.8+ has ABI issues with Torch 2.6 currently
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# 6. Install PyTorch Geometric
echo "--- Installing PyG ---"
# We try to find wheels, but will fallback to source build if needed
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch-geometric

# 7. Install SpConv (CUDA 12.x)
pip install spconv-cu120

# 8. Install Custom Ops (Pointcept)
echo "--- Compiling Pointcept Custom Ops ---"
# These require the CPATH exported in step 2

if [ -d "./libs/pointops" ]; then
    echo "Building pointops..."
    pip install ./libs/pointops --no-build-isolation --no-cache-dir
else
    echo "Warning: ./libs/pointops not found."
fi

if [ -d "./libs/pointgroup_ops" ]; then
    echo "Building pointgroup_ops..."
    pip install ./libs/pointgroup_ops --no-build-isolation --no-cache-dir
else
    echo "Warning: ./libs/pointgroup_ops not found."
fi
pip install torchmetrics
pip install transformers   
# Spconv is incompatible with numpy 2.0, so we ensure we have a compatible version
pip install "numpy<2.0.0"

# 9. Cleanup
# Optional: Restore setuptools, though sticking to <70 is often safer for older ML repos
# pip install --upgrade setuptools 

echo "---------------------------------------"
echo "SUCCESS: Environment 'rewis3dModel' is ready."
echo "Verify with: python -c 'import torch; import pointops; print(torch.__version__, pointops)'"