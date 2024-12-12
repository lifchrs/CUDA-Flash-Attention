NVCC=nvcc
NVCCFLAGS=-DCUDA -O3 -use_fast_math -arch=sm_80

# Output binary name
TARGET=attention

# Source files
SOURCES=main.cpp flash_attention.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(SOURCES) -o $@ $(NVCCFLAGS)

.PHONY: clean all

clean:
	rm -f $(TARGET)
	rm -f build/*.out