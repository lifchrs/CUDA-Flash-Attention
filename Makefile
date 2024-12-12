# Compiler settings
NVCC = nvcc
CXX = g++

# Directories
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

# Flags
NVCCFLAGS = -O3 -arch=sm_60  # Adjust the architecture based on your GPU
CXXFLAGS = -O3 -std=c++11

# Include directories
INCLUDES = -I/usr/local/cuda/include -I$(INCLUDE_DIR)

# Source files
CUDA_SRC = $(SRC_DIR)/flash_attention.cu
CPP_SRC = $(SRC_DIR)/main.cpp

# Object files
CUDA_OBJ = $(BUILD_DIR)/flash_attention.o
CPP_OBJ = $(BUILD_DIR)/main.o

# Output executable
TARGET = $(BUILD_DIR)/attention

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(CUDA_OBJ) $(CPP_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(CPP_OBJ): $(CPP_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean