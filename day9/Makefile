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

register-ilp: register_ilp.cu
	$(NVCC) -O3 -arch=$(ARCH) -Xptxas -v register_ilp.cu -o register_ilp

run-register-ilp: register-ilp
	./register_ilp