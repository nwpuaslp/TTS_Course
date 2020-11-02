#!/bin/bash

#########################################
######### Install Dependencies ##########
#########################################
#sudo apt-get install csh realpath

tools_dir=$(dirname $0)
cd $tools_dir

install_sptk=true
install_world=true

# 1. Get and compile SPTK
if [ "$install_sptk" = true ]; then
    echo "compiling SPTK..."
    (
        cd SPTK-3.9;
        ./configure --prefix=$PWD/build;
        make;
        make install
    )
fi


# 2. Getting WORLD
if [ "$install_world" = true ]; then
    echo "compiling WORLD..."
    (
        cd WORLD;
        make
        make analysis synth
        make clean
    )
fi

SPTK_BIN_DIR=bin/SPTK-3.9
WORLD_BIN_DIR=bin/WORLD

mkdir -p bin
mkdir -p $SPTK_BIN_DIR
mkdir -p $WORLD_BIN_DIR

cp SPTK-3.9/build/bin/* $SPTK_BIN_DIR/
cp WORLD/build/analysis $WORLD_BIN_DIR/
cp WORLD/build/synth $WORLD_BIN_DIR/



# 3. Getting WORLD

echo "compiling World..."
(
    cd World_v2;
    make
    cd examples/analysis_synthesis;
    make
)

WORLD2_BIN_DIR=bin/World_v2

mkdir -p $WORLD2_BIN_DIR

mv World_v2/build/analysis $WORLD2_BIN_DIR/
mv World_v2/build/synthesis $WORLD2_BIN_DIR/



if [[ ! -f ${SPTK_BIN_DIR}/x2x ]]; then
    echo "Error installing SPTK tools! Try installing dependencies!!"
    echo "sudo apt-get install csh"
    exit 1
elif [[ ! -f ${WORLD_BIN_DIR}/analysis ]]; then
    echo "Error installing WORLD tools"
    exit 1
else
    echo "All tools successfully compiled!!"
fi
