NVCC = nvcc
ARCH = sm_75
TARGET = occupancy
SRC = occupancy.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

divergence: divergence.cu
	$(NVCC) -O3 -arch=$(ARCH) divergence.cu -o divergence

run-divergence: divergence
	./divergence