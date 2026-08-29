#include "connection.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <sys/socket.h>


bool process_messages(
    int fd,
    Connection& conn
) {
    while (true) {

        //
        // length header가 아직 덜 왔다.
        //
        if (conn.in_buffer.size() < 4) {
            return true;
        }

        uint32_t network_length;

        std::memcpy(
            &network_length,
            conn.in_buffer.data(),
            sizeof(network_length)
        );

        uint32_t payload_length =
            ntohl(network_length);

        if (payload_length > MAX_MESSAGE_SIZE) {
            std::cerr
                << "message too large: "
                << payload_length
                << '\n';

            return false;
        }

        size_t frame_size =
            sizeof(uint32_t)
            + payload_length;


        //
        // header는 받았는데 payload가 아직 덜 왔다.
        //
        if (conn.in_buffer.size() < frame_size) {
            return true;
        }


        //
        // complete payload
        //
        std::string payload(
            conn.in_buffer.data()
                + sizeof(uint32_t),
            payload_length
        );


        //
        // 처리한 frame 제거
        //
        conn.in_buffer.erase(
            0,
            frame_size
        );


        //
        // echo response framing
        //
        uint32_t response_length =
            htonl(
                static_cast<uint32_t>(
                    payload.size()
                )
            );

        conn.out_buffer.append(
            reinterpret_cast<const char*>(
                &response_length
            ),
            sizeof(response_length)
        );

        conn.out_buffer.append(payload);


        //
        // while 반복
        //
        // in_buffer에 frame이 여러 개 붙어 있을 수도 있음.
        //
    }
}


bool read_from_socket(
    int fd,
    Connection& conn
) {
    char buffer[BUF_SIZE];

    while (true) {

        ssize_t n = recv(
            fd,
            buffer,
            sizeof(buffer),
            0
        );

        if (n > 0) {
            conn.in_buffer.append(
                buffer,
                static_cast<size_t>(n)
            );

            continue;
        }


        //
        // peer가 connection 종료
        //
        if (n == 0) {
            return false;
        }


        //
        // non-blocking:
        // 현재 읽을 데이터는 전부 읽음
        //
        if (
            errno == EAGAIN ||
            errno == EWOULDBLOCK
        ) {
            break;
        }


        perror("recv");
        return false;
    }


    return process_messages(
        fd,
        conn
    );
}


bool write_to_socket(
    int fd,
    Connection& conn
) {
    while (!conn.out_buffer.empty()) {

        ssize_t n = send(
            fd,
            conn.out_buffer.data(),
            conn.out_buffer.size(),
            MSG_NOSIGNAL
        );

        if (n > 0) {
            conn.out_buffer.erase(
                0,
                static_cast<size_t>(n)
            );

            continue;
        }


        //
        // kernel send buffer가 현재 꽉 참
        //
        if (
            n == -1 &&
            (
                errno == EAGAIN ||
                errno == EWOULDBLOCK
            )
        ) {
            return true;
        }


        perror("send");
        return false;
    }


    return true;
}