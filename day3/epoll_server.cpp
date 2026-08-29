#include "connection.hpp"

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


std::unordered_map<int, Connection> connections;
bool use_et = false;
bool debug = false;

template <typename... Args>
void debug_log(Args&&... args) {
    if (!debug) {
        return;
    }

    (std::cout << ... << args) << '\n';
}

uint32_t read_events() {
    uint32_t events = EPOLLIN;

    if (use_et) {
        events |= EPOLLET;
    }

    return events;
}


void close_client(int epfd, int fd) {
    epoll_ctl(
        epfd,
        EPOLL_CTL_DEL,
        fd,
        nullptr
    );

    close(fd);
    connections.erase(fd);

    debug_log(
        "closed fd=",
        fd
    );
}


void update_events(
    int epfd,
    int fd,
    bool want_write
) {
    epoll_event ev{};

    ev.data.fd = fd;
    ev.events = read_events();

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


void handle_read(
    int epfd,
    int fd
) {
    Connection& conn =
        connections.at(fd);

    if (!read_from_socket(fd, conn)) {
        close_client(
            epfd,
            fd
        );

        return;
    }


    //
    // response가 생겼으면
    // EPOLLOUT 관심 등록
    //
    if (!conn.out_buffer.empty()) {
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
    Connection& conn =
        connections.at(fd);

    if (!write_to_socket(fd, conn)) {
        close_client(
            epfd,
            fd
        );

        return;
    }


    //
    // 전부 보냈으면
    // EPOLLOUT 관심 제거
    //
    if (conn.out_buffer.empty()) {
        update_events(
            epfd,
            fd,
            false
        );
    }
}


int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--et") {
            use_et = true;
        } else if (arg == "--debug") {
            debug = true;
        }
    }

    std::cout
        << "mode: "
        << (use_et ? "ET" : "LT")
        << '\n';

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

    listen_event.events = read_events();


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

                    client_event.events = read_events();


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


                    if (debug) {
                        char ip[INET_ADDRSTRLEN];

                        inet_ntop(
                            AF_INET,
                            &client_addr.sin_addr,
                            ip,
                            sizeof(ip)
                        );

                        debug_log(
                            "accepted fd=",
                            client_fd,
                            " from ",
                            ip,
                            ":",
                            ntohs(client_addr.sin_port)
                        );
                    }
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