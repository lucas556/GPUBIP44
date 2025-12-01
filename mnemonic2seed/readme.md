'''
nvcc -O3 -std=c++17 \
  -Xcompiler "-march=native" \
  -o mnemonic2master \
  pybind_mnemonic2master.cu \
  -L/usr/local/cuda/lib64 -lcudart
'''

'''
./mnemonic2master mnemo_256_100000000.txt master_i.bin \
    --batch-size 500000 \
    --threads-per-block 128 \
    --passphrase ""
'''
