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

DIST = dist

sass: register-ilp
	mkdir -p $(DIST)
	cuobjdump -sass \
		-fun '_Z10ilp_kernelILi32EEvPKfPfi' \
		register_ilp > $(DIST)/ilp32.sass
	cuobjdump -sass \
		-fun '_Z10ilp_kernelILi64EEvPKfPfi' \
		register_ilp > $(DIST)/ilp64.sass

sass-summary: sass
	@echo "===== ILP 32 ====="
	@echo -n "FFMA : "
	@grep -c "FFMA" $(DIST)/ilp32.sass || true
	@echo -n "LDG  : "
	@grep -c "LDG" $(DIST)/ilp32.sass || true
	@echo -n "STG  : "
	@grep -c "STG" $(DIST)/ilp32.sass || true
	@echo
	@echo "===== ILP 64 ====="
	@echo -n "FFMA : "
	@grep -c "FFMA" $(DIST)/ilp64.sass || true
	@echo -n "LDG  : "
	@grep -c "LDG" $(DIST)/ilp64.sass || true
	@echo -n "STG  : "
	@grep -c "STG" $(DIST)/ilp64.sass || true