#pragma once

#include <cstdint>
#include <string>


constexpr int BUF_SIZE = 65536;
constexpr uint32_t MAX_MESSAGE_SIZE = 1024 * 1024;


struct Connection {
    std::string in_buffer;
    std::string out_buffer;

    bool read_paused = false;
};


// 받은 byte들 중 완성된 frame들을 처리한다.
//
// true  = 정상
// false = protocol error
bool process_messages(
    int fd,
    Connection& conn
);


// non-blocking recv를 EAGAIN까지 수행한다.
//
// true  = connection 유지
// false = peer close 또는 error
bool read_from_socket(
    int fd,
    Connection& conn
);


// out_buffer를 가능한 만큼 send한다.
//
// true  = connection 유지
// false = send error
bool write_to_socket(
    int fd,
    Connection& conn
);