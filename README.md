# mxnet-utils

## Installation

1. clone project
```bash
cd ~/api
git clone --recursive https://github.com/apache/incubator-mxnet mxnet
cd mxnet
```

2. Typical cmake process

```bash
mkdir build
cd build
```

If I don't have CUDA
```bash
cmake -DUSE_CUDA=0 -DUSE_MKLDNN=1   \
      -DUSE_CPP_PACKAGE=1           \
      -DCMAKE_BUILD_TYPE=Release    \
      -DCMAKE_INSTALL_PREFIX=~/api/mxnet-160 ..
```

If I have CUDA
```bash
cmake -DUSE_CUDA=1 -DUSE_CUDA_PATH=/usr/local/cuda  \
      -DUSE_CUDNN=1 -DUSE_MKLDNN=1                  \
      -DUSE_CPP_PACKAGE=1                           \
      -DCMAKE_BUILD_TYPE=Release                    \
      -DCMAKE_INSTALL_PREFIX=~/api/mxnet-160 ..
```

**cmake要一次完成 不然會怪怪的**
**如果要改設定 建議把build砍掉重建**

```bash
make -j8
make install
```

3. Add `LD_LIBRARY_PATH` setting to your bashrc
```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/api/mxnet-160/lib
```

4. Install python interface
```bash
cd ~/api/mxnet/python
python3 setup.py install --user
```