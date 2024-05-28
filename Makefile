COMMON_OPTIONS = -Wall -O3 -march=skylake -D_GNU_SOURCE
LIBS = -lrt -lpthread

all:
	gcc $(COMMON_OPTIONS) *.c -o matrix.out $(LIBS)