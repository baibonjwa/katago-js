# use latest-upstream of emscripten
pushd ${HOME}/OpenSources/emsdk
version=2.0.29
#version=2.0.30 # 動かない
./emsdk install $version
./emsdk activate $version
source emsdk_env.sh
popd

build_dir=em_build
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