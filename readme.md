## 
### buding 
'''
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

nvcc -O3 -std=c++17 \
  -Xcompiler "-fPIC -fopenmp -march=native" \
  -shared \
  -I. $(python3 -m pybind11 --includes) \
  -gencode arch=compute_86,code=sm_86 \
  -Xptxas="-v -dlcm=ca -O3" \
  pybind_mnemonic2master.cu -o pybind_mnemonic2master${EXT} \
  -L/usr/local/cuda/lib64 -lcudart -lgomp

  python3 build_master_middleware.py mnemo.txt master_i_with_pass.bin  --batch-size 500000   --threads-per-block 128
'''
