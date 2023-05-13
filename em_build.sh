# use latest-upstream of emscripten
pushd ${HOME}/OpenSources/emsdk
./emask activate 2.0.13
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