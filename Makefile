# nerv makefile

STD=-std=c99
WFLAGS=-Wall -Wextra
OPT=-O2
IDIR=-I.
CC=gcc
NAME=libnerv
SRC=src/*.c

CFLAGS=$(STD) $(WFLAGS) $(OPT) $(IDIR)
OS=$(shell uname -s)

ifeq ($(OS),Darwin)
	OSFLAGS=-dynamiclib
	LIB=$(NAME).dylib
else
	OSFLAGS=-lm -shared -fPIC
	LIB=$(NAME).so
endif

$(NAME).a: $(SRC)
	$(CC) $(CFLAGS) -c $(SRC) && ar -crv $(NAME).a *.o && rm *.o

shared: $(SRC)
	$(CC) -o $(LIB) $(SRC) $(CFLAGS) $(OSFLAGS)

clean: build.sh
	./$^ -$@
	
