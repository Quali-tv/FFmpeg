./configure --enable-libx264 --enable-gpl --disable-asm --extra-ldflags="-lstdc++ -lprotobuf -lgrpc++ -lgrpc -lkoku" --extra-cxxflags="-std=c++11" --extra-cflags="-fno-stack-check" && make -j 8