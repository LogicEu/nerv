#!/bin/bash

src=src/*.c
cc=gcc
name=libnerv

flags=(
    -std=c99
    -Wall
    -Wextra
    -O2
)

inc=(
    -I./
)

fail_op() {
    echo "Run with -dlib to build dynamically or -slib to build statically." && exit
}

fail_os() {
    echo "OS is not supported yet..." && exit
}

mac_dlib() {
    $cc ${flags[*]} ${inc[*]} ${lib[*]} -dynamiclib $src -o $name.dylib &&\
    install_name_tool -id @executable_path/../$name.dylib $name.dylib
}

linux_dlib() {
    $cc -shared ${flags[*]} ${inc[*]} ${lib[*]} -lm -fPIC $src -o $name.so 
}

dlib() {
    if echo "$OSTYPE" | grep -q "darwin"; then
        mac_dlib
    elif echo "$OSTYPE" | grep -q "linux"; then
        linux_dlib
    else
        fails_os
    fi
}

slib() {
    $cc ${flags[*]} ${inc[*]} -c $src && ar -crv $name.a *.o && rm *.o
}

cleanf() {
    if [ -f $1 ]; then 
        rm $1
    fi
}

clean() {
    cleanf $name.a
    cleanf $name.so
    cleanf $name.dylib
}

case "$1" in
    "-dlib")
        dlib;;
    "-slib")
        slib;;
    "-clean")
        clean;;
    *)
        fail_op;;
esac
