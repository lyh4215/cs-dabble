#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>

constexpr int PORT = 5000;
constexpr int MAX_EVENTS = 1024;
constexpr int BUF_SIZE = 65536;
constexpr uint32_t MAX_MESSAGE_SIZE = 1024 * 1024; // 1 MiB


struct Connection {
    std::string in_buffer;
    std::string out_buffer;
};


std::unordered_map<int, Connection> connections;


void close_client(int epfd, int fd) {
    epoll_ctl(
        epfd,
        EPOLL_CTL_DEL,
        fd,
        nullptr
    );

    close(fd);
    connections.erase(fd);

    std::cout
        << "closed fd="
        << fd
        << '\n';
}


void update_events(
    int epfd,
    int fd,
    bool want_write
) {
    epoll_event ev{};

    ev.data.fd = fd;
    ev.events = EPOLLIN;

    if (want_write) {
        ev.events |= EPOLLOUT;
    }

    if (epoll_ctl(
            epfd,
            EPOLL_CTL_MOD,
            fd,
            &ev
        ) == -1) {

        perror("epoll_ctl MOD");
    }
}


bool process_messages(int fd) {
    Connection& conn = connections[fd];

    while (true) {

        // 4-byte length header조차 아직 다 안 왔음
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
                << "fd="
                << fd
                << " message too large: "
                << payload_length
                << '\n';

            return false;
        }

        size_t frame_size =
            sizeof(uint32_t)
            + payload_length;

        // header는 있지만 payload가 아직 덜 왔음
        if (conn.in_buffer.size() < frame_size) {
            return true;
        }

        // 완전한 frame 하나 확보
        std::string payload(
            conn.in_buffer.data()
                + sizeof(uint32_t),
            payload_length
        );

        std::cout
            << "fd="
            << fd
            << " message=\""
            << payload
            << "\"\n";

        // 처리한 frame 제거
        conn.in_buffer.erase(
            0,
            frame_size
        );

        //
        // Echo response 생성
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
        // in_buffer 뒤에 frame이 하나 더
        // 붙어 있을 수도 있기 때문
        //
    }
}


void handle_read(
    int epfd,
    int fd
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

            connections[fd]
                .in_buffer
                .append(
                    buffer,
                    static_cast<size_t>(n)
                );

            continue;
        }

        //
        // recv() == 0
        // peer가 connection 종료
        //
        if (n == 0) {
            close_client(
                epfd,
                fd
            );

            return;
        }

        //
        // non-blocking socket에서
        // 현재 더 읽을 데이터 없음
        //
        if (errno == EAGAIN ||
            errno == EWOULDBLOCK) {

            break;
        }

        perror("recv");

        close_client(
            epfd,
            fd
        );

        return;
    }

    //
    // raw byte stream에서
    // complete frame들을 꺼내 처리
    //
    if (!process_messages(fd)) {

        close_client(
            epfd,
            fd
        );

        return;
    }

    //
    // 보낼 데이터가 생겼으면
    // EPOLLOUT도 감시
    //
    if (!connections[fd]
             .out_buffer
             .empty()) {

        update_events(
            epfd,
            fd,
            true
        );
    }
}


void handle_write(
    int epfd,
    int fd
) {
    auto it = connections.find(fd);

    if (it == connections.end()) {
        return;
    }

    Connection& conn = it->second;

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

        if (n == -1 &&
            (errno == EAGAIN ||
             errno == EWOULDBLOCK)) {

            //
            // kernel send buffer에
            // 현재 공간 없음
            //
            // EPOLLOUT 이벤트가
            // 다시 올 때까지 기다림
            //
            break;
        }

        perror("send");

        close_client(
            epfd,
            fd
        );

        return;
    }

    //
    // 전부 보냈으면
    // EPOLLOUT 감시는 끈다
    //
    if (conn.out_buffer.empty()) {

        update_events(
            epfd,
            fd,
            false
        );
    }
}


int main() {

    //
    // listening socket
    //
    int listen_fd = socket(
        AF_INET,
        SOCK_STREAM | SOCK_NONBLOCK,
        0
    );

    if (listen_fd == -1) {
        perror("socket");
        return 1;
    }

    int reuse = 1;

    if (setsockopt(
            listen_fd,
            SOL_SOCKET,
            SO_REUSEADDR,
            &reuse,
            sizeof(reuse)
        ) == -1) {

        perror("setsockopt");
        return 1;
    }


    sockaddr_in addr{};

    addr.sin_family = AF_INET;

    addr.sin_addr.s_addr =
        htonl(INADDR_ANY);

    addr.sin_port =
        htons(PORT);


    if (bind(
            listen_fd,
            reinterpret_cast<sockaddr*>(
                &addr
            ),
            sizeof(addr)
        ) == -1) {

        perror("bind");
        return 1;
    }


    if (listen(
            listen_fd,
            SOMAXCONN
        ) == -1) {

        perror("listen");
        return 1;
    }


    //
    // epoll instance 생성
    //
    int epfd =
        epoll_create1(0);

    if (epfd == -1) {
        perror("epoll_create1");
        return 1;
    }


    //
    // listening fd 등록
    //
    epoll_event listen_event{};

    listen_event.data.fd =
        listen_fd;

    listen_event.events =
        EPOLLIN;


    if (epoll_ctl(
            epfd,
            EPOLL_CTL_ADD,
            listen_fd,
            &listen_event
        ) == -1) {

        perror(
            "epoll_ctl ADD listen"
        );

        return 1;
    }


    epoll_event events[MAX_EVENTS];


    std::cout
        << "epoll server listening on port "
        << PORT
        << '\n';


    //
    // event loop
    //
    while (true) {

        int ready =
            epoll_wait(
                epfd,
                events,
                MAX_EVENTS,
                -1
            );

        if (ready == -1) {

            if (errno == EINTR) {
                continue;
            }

            perror("epoll_wait");
            break;
        }


        for (
            int i = 0;
            i < ready;
            ++i
        ) {

            int fd =
                events[i].data.fd;

            uint32_t event =
                events[i].events;


            //
            // listening socket
            //
            if (fd == listen_fd) {

                while (true) {

                    sockaddr_in client_addr{};

                    socklen_t client_len =
                        sizeof(client_addr);


                    int client_fd =
                        accept4(
                            listen_fd,
                            reinterpret_cast<
                                sockaddr*
                            >(
                                &client_addr
                            ),
                            &client_len,
                            SOCK_NONBLOCK
                            | SOCK_CLOEXEC
                        );


                    if (client_fd == -1) {

                        if (
                            errno == EAGAIN ||
                            errno == EWOULDBLOCK
                        ) {
                            break;
                        }

                        perror("accept4");
                        break;
                    }


                    //
                    // 새 client fd를
                    // epoll에 등록
                    //
                    epoll_event client_event{};

                    client_event.data.fd =
                        client_fd;

                    client_event.events =
                        EPOLLIN;


                    if (epoll_ctl(
                            epfd,
                            EPOLL_CTL_ADD,
                            client_fd,
                            &client_event
                        ) == -1) {

                        perror(
                            "epoll_ctl ADD client"
                        );

                        close(client_fd);
                        continue;
                    }


                    connections[
                        client_fd
                    ] = {};


                    char ip[INET_ADDRSTRLEN];

                    inet_ntop(
                        AF_INET,
                        &client_addr.sin_addr,
                        ip,
                        sizeof(ip)
                    );


                    std::cout
                        << "accepted fd="
                        << client_fd
                        << " from "
                        << ip
                        << ":"
                        << ntohs(
                            client_addr.sin_port
                        )
                        << '\n';
                }

                continue;
            }


            //
            // socket error
            //
            if (event &
                (EPOLLERR |
                 EPOLLHUP)) {

                close_client(
                    epfd,
                    fd
                );

                continue;
            }


            //
            // readable
            //
            if (event & EPOLLIN) {

                handle_read(
                    epfd,
                    fd
                );

                //
                // handle_read에서
                // close됐을 수도 있음
                //
                if (
                    connections.find(fd)
                    ==
                    connections.end()
                ) {

                    continue;
                }
            }


            //
            // writable
            //
            if (event & EPOLLOUT) {

                handle_write(
                    epfd,
                    fd
                );
            }
        }
    }


    close(epfd);
    close(listen_fd);

    return 0;
}