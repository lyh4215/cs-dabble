CXX := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra

TARGET := dist/epoll_server
SRC := epoll_server.cpp

.PHONY: all run run-et clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run:
	./$(TARGET)

run-debug:
	./$(TARGET) --debug

run-et:
	./$(TARGET) --et

run-et-debug:
	./$(TARGET) --et --debug

clean:
	rm -f $(TARGET)