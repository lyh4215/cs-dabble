CXX := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra

DIST := dist

EPOLL_TARGET := $(DIST)/epoll_server
POLL_TARGET := $(DIST)/poll_server
SELECT_TARGET := $(DIST)/select_server

COMMON_SRC := connection.cpp
COMMON_HDR := connection.hpp


.PHONY: \
	all \
	run \
	run-et \
	run-poll \
	run-select \
	run-debug \
	run-et-debug \
	run-poll-debug \
	run-select-debug \
	bench \
	bench-idle \
	clean


all: \
	$(EPOLL_TARGET) \
	$(POLL_TARGET) \
	$(SELECT_TARGET)


$(EPOLL_TARGET): epoll_server.cpp $(COMMON_SRC) $(COMMON_HDR)
	mkdir -p $(DIST)
	$(CXX) $(CXXFLAGS) epoll_server.cpp $(COMMON_SRC) -o $(EPOLL_TARGET)


$(POLL_TARGET): poll_server.cpp $(COMMON_SRC) $(COMMON_HDR)
	mkdir -p $(DIST)
	$(CXX) $(CXXFLAGS) poll_server.cpp $(COMMON_SRC) -o $(POLL_TARGET)


$(SELECT_TARGET): select_server.cpp $(COMMON_SRC) $(COMMON_HDR)
	mkdir -p $(DIST)
	$(CXX) $(CXXFLAGS) select_server.cpp $(COMMON_SRC) -o $(SELECT_TARGET)


run: $(EPOLL_TARGET)
	./$(EPOLL_TARGET)


run-et: $(EPOLL_TARGET)
	./$(EPOLL_TARGET) --et


run-poll: $(POLL_TARGET)
	./$(POLL_TARGET)


run-select: $(SELECT_TARGET)
	./$(SELECT_TARGET)


run-debug: $(EPOLL_TARGET)
	./$(EPOLL_TARGET) --debug


run-et-debug: $(EPOLL_TARGET)
	./$(EPOLL_TARGET) --et --debug


run-poll-debug: $(POLL_TARGET)
	./$(POLL_TARGET) --debug


run-select-debug: $(SELECT_TARGET)
	./$(SELECT_TARGET) --debug


bench: all
	python compare_benchmark.py

bench-idle: all
	python compare_idle_benchmark.py

clean:
	rm -f \
		$(EPOLL_TARGET) \
		$(POLL_TARGET) \
		$(SELECT_TARGET)