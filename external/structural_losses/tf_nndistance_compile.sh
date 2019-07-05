TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# <= tf 1.3
# /usr/local/cuda-8.0/bin/nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I ${TF_INC}  -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 && g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I ${TF_INC} -L /usr/local/cuda-8.0/lib64 -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# >= tf 1.4
/usr/local/cuda-8.0/bin/nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I ${TF_INC}  -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 && g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I ${TF_INC} -L /usr/local/cuda-8.0/lib64 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework

