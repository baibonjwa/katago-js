# use latest-upstream of emscripten
pushd ${HOME}/goproblems/emsdk
#version=2.0.29
#version=2.0.30 # 動かない
version=3.1.52 # 動いたが,condition_variable#wait_forが動かない
# version=latest
./emsdk install $version
./emsdk activate $version
source emsdk_env.sh
popd

# build_dir=em_build
src_dir=../cpp

if [ ! -d $build_dir ]; then
    mkdir $build_dir
fi
cd $build_dir
if [ "$1" = "clean" ]; then
    emcmake cmake $src_dir -DBUILD_MCTS=1 -DUSE_BACKEND=TFJS && emmake make clean && emmake make
else
    emcmake cmake $src_dir -DBUILD_MCTS=1 -DUSE_BACKEND=TFJS && emmake make
fi

# Build ES Modules
# if [ "$1" = "clean" ]; then
#     emcmake cmake $src_dir -DBUILD_MCTS=1 -DUSE_BAgoproblems.developCKEND=TFJS \
#         -DCMAKE_EXECUTABLE_SUFFIX=".js" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
#     emmake make clean && \
#     emmake make EMCC_FLAGS="-s MODULARIZE=1 -s EXPORT_ES6=1"
# else
#     emcmake cmake $src_dir -DBUILD_MCTS=1 -DUSE_BACKEND=TFJS \
#         -DCMAKE_EXECUTABLE_SUFFIX=".js" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
#     emmake make EMCC_FLAGS="-s MODULARIZE=1 -s EXPORT_ES6=1"
# fi