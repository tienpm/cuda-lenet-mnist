TARGET=main
OBJECTS=lenet.o

CPPFLAGS=-std=c++11 -Wall -O3 -march=znver2 -fopenmp
LDFLAGS=
LDLIBS=-lm

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
