#include "connection.hpp"

#include <arpa/inet.h>
#include <algorithm>
#include <cerrno>
#include <iostream>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>


constexpr int PORT = 5000;

bool debug = false;

std::unordered_map<int, Connection> connections;
std::vector<pollfd> poll_fds;


template <typename... Args>
void debug_log(Args&&... args) {
    if (!debug) {
        return;
    }

    (std::cout << ... << args) << '\n';
}


//
// client connection 종료
//
// poll_fds에서 즉시 erase하면
// event loop를 순회하는 중 vector iterator/index가 깨질 수 있다.
//
// 따라서 fd=-1로 표시만 하고,
// event loop가 끝난 뒤 실제로 제거한다.
//
void close_client(int fd) {
    if (connections.find(fd) == connections.end()) {
        return;
    }

    close(fd);
    connections.erase(fd);

    for (auto& pfd : poll_fds) {
        if (pfd.fd == fd) {
            pfd.fd = -1;
            pfd.events = 0;
            pfd.revents = 0;
            break;
        }
    }

    debug_log(
        "closed fd=",
        fd
    );
}


//
// fd=-1로 표시된 pollfd들을
// 실제 vector에서 제거
//
void cleanup_closed_fds() {
    poll_fds.erase(
        std::remove_if(
            poll_fds.begin(),
            poll_fds.end(),
            [](const pollfd& pfd) {
                return pfd.fd < 0;
            }
        ),
        poll_fds.end()
    );
}


//
// 특정 client의 POLLOUT 관심 여부 변경
//
void set_write_interest(
    int fd,
    bool want_write
) {
    for (auto& pfd : poll_fds) {
        if (pfd.fd != fd) {
            continue;
        }

        if (want_write) {
            pfd.events |= POLLOUT;
        } else {
            pfd.events &= ~POLLOUT;
        }

        return;
    }
}


//
// readable client 처리
//
void handle_read(int fd) {
    auto it = connections.find(fd);

    if (it == connections.end()) {
        return;
    }

    Connection& conn = it->second;

    //
    // connection.cpp의 공통 함수
    //
    // recv()를 EAGAIN까지 수행하고
    // length-prefix frame까지 parsing한다.
    //
    if (!read_from_socket(fd, conn)) {
        close_client(fd);
        return;
    }

    //
    // echo response가 만들어졌으면
    // POLLOUT도 감시
    //
    if (!conn.out_buffer.empty()) {
        set_write_interest(
            fd,
            true
        );
    }
}


//
// writable client 처리
//
void handle_write(int fd) {
    auto it = connections.find(fd);

    if (it == connections.end()) {
        return;
    }

    Connection& conn = it->second;

    //
    // connection.cpp의 공통 함수
    //
    // out_buffer를 가능한 만큼 send()
    //
    if (!write_to_socket(fd, conn)) {
        close_client(fd);
        return;
    }

    //
    // 다 보냈으면 POLLOUT 감시 종료
    //
    // socket은 대부분 writable이므로
    // 계속 POLLOUT을 켜두면 poll()이
    // 불필요하게 계속 깨어날 수 있다.
    //
    if (conn.out_buffer.empty()) {
        set_write_interest(
            fd,
            false
        );
    }
}


//
// listening socket의 pending connection들을
// EAGAIN이 나올 때까지 전부 accept
//
void accept_clients(int listen_fd) {
    while (true) {
        sockaddr_in client_addr{};

        socklen_t client_len =
            sizeof(client_addr);

        int client_fd =
            accept4(
                listen_fd,
                reinterpret_cast<sockaddr*>(
                    &client_addr
                ),
                &client_len,
                SOCK_NONBLOCK |
                SOCK_CLOEXEC
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
        // application connection state 생성
        //
        connections[client_fd] = {};


        //
        // poll에서 감시할 fd 배열에 추가
        //
        pollfd client_pollfd{};

        client_pollfd.fd =
            client_fd;

        client_pollfd.events =
            POLLIN;

        client_pollfd.revents =
            0;

        poll_fds.push_back(
            client_pollfd
        );


        if (debug) {
            char ip[
                INET_ADDRSTRLEN
            ];

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
                ntohs(
                    client_addr.sin_port
                )
            );
        }
    }
}


int main(
    int argc,
    char* argv[]
) {
    //
    // command line arguments
    //
    for (
        int i = 1;
        i < argc;
        ++i
    ) {
        std::string arg =
            argv[i];

        if (arg == "--debug") {
            debug = true;
        }
    }


    //
    // listening socket 생성
    //
    int listen_fd =
        socket(
            AF_INET,
            SOCK_STREAM |
                SOCK_NONBLOCK |
                SOCK_CLOEXEC,
            0
        );

    if (listen_fd == -1) {
        perror("socket");
        return 1;
    }


    //
    // address 재사용 허용
    //
    int reuse = 1;

    if (
        setsockopt(
            listen_fd,
            SOL_SOCKET,
            SO_REUSEADDR,
            &reuse,
            sizeof(reuse)
        ) == -1
    ) {
        perror("setsockopt");

        close(listen_fd);
        return 1;
    }


    //
    // 0.0.0.0:5000
    //
    sockaddr_in addr{};

    addr.sin_family =
        AF_INET;

    addr.sin_addr.s_addr =
        htonl(INADDR_ANY);

    addr.sin_port =
        htons(PORT);


    if (
        bind(
            listen_fd,
            reinterpret_cast<
                sockaddr*
            >(&addr),
            sizeof(addr)
        ) == -1
    ) {
        perror("bind");

        close(listen_fd);
        return 1;
    }


    if (
        listen(
            listen_fd,
            SOMAXCONN
        ) == -1
    ) {
        perror("listen");

        close(listen_fd);
        return 1;
    }


    //
    // poll에서는 epoll_create1() 같은
    // kernel-side instance를 만들지 않는다.
    //
    // 그냥 application이 pollfd 배열을 가진다.
    //
    pollfd listen_pollfd{};

    listen_pollfd.fd =
        listen_fd;

    listen_pollfd.events =
        POLLIN;

    listen_pollfd.revents =
        0;

    poll_fds.push_back(
        listen_pollfd
    );


    std::cout
        << "poll server listening on port "
        << PORT
        << '\n';


    //
    // main event loop
    //
    while (true) {

        //
        // 핵심:
        //
        // poll_fds 배열 전체를
        // 매 호출마다 kernel에 넘긴다.
        //
        int ready =
            poll(
                poll_fds.data(),
                static_cast<nfds_t>(
                    poll_fds.size()
                ),
                -1
            );


        if (ready == -1) {
            if (errno == EINTR) {
                continue;
            }

            perror("poll");
            break;
        }


        //
        // accept_clients()에서 push_back()될 수 있으므로
        // 이번 poll()이 감시했던 fd 개수만 순회한다.
        //
        size_t current_count =
            poll_fds.size();


        //
        // poll의 핵심적인 특징:
        //
        // ready가 2개뿐이어도
        // poll_fds 전체를 순회해서
        // revents를 확인해야 한다.
        //
        for (
            size_t i = 0;
            i < current_count;
            ++i
        ) {
            int fd =
                poll_fds[i].fd;

            short revents =
                poll_fds[i].revents;


            if (
                fd < 0 ||
                revents == 0
            ) {
                continue;
            }


            //
            // listening socket
            //
            if (fd == listen_fd) {

                if (
                    revents &
                    POLLIN
                ) {
                    accept_clients(
                        listen_fd
                    );
                }

                if (
                    revents &
                    (
                        POLLERR |
                        POLLHUP |
                        POLLNVAL
                    )
                ) {
                    std::cerr
                        << "listening socket error\n";

                    close(listen_fd);
                    return 1;
                }

                continue;
            }


            //
            // invalid/error
            //
            if (
                revents &
                (
                    POLLERR |
                    POLLNVAL
                )
            ) {
                close_client(fd);
                continue;
            }


            //
            // readable
            //
            if (
                revents &
                POLLIN
            ) {
                handle_read(fd);

                //
                // handle_read에서
                // connection이 닫혔을 수도 있음
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
            if (
                revents &
                POLLOUT
            ) {
                handle_write(fd);

                if (
                    connections.find(fd)
                    ==
                    connections.end()
                ) {
                    continue;
                }
            }


            //
            // peer hangup
            //
            if (
                revents &
                POLLHUP
            ) {
                close_client(fd);
            }
        }


        //
        // iteration 중 삭제했던 fd들을
        // 이제 실제 vector에서 제거
        //
        cleanup_closed_fds();
    }


    //
    // cleanup
    //
    for (
        const auto& [fd, conn]
        : connections
    ) {
        (void)conn;
        close(fd);
    }

    close(listen_fd);

    return 0;
}