#include "connection.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <iostream>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>


constexpr int PORT = 5000;

bool debug = false;

std::unordered_map<int, Connection> connections;


// select가 지속적으로 관리할 "원본" fd set
fd_set master_read_set;
fd_set master_write_set;

int max_fd = -1;


template <typename... Args>
void debug_log(Args&&... args) {
    if (!debug) {
        return;
    }

    (std::cout << ... << args) << '\n';
}


void recompute_max_fd(int listen_fd) {
    int new_max = listen_fd;

    for (const auto& [fd, conn] : connections) {
        (void)conn;

        if (fd > new_max) {
            new_max = fd;
        }
    }

    max_fd = new_max;
}


void close_client(
    int listen_fd,
    int fd
) {
    if (
        connections.find(fd)
        ==
        connections.end()
    ) {
        return;
    }

    FD_CLR(
        fd,
        &master_read_set
    );

    FD_CLR(
        fd,
        &master_write_set
    );

    close(fd);

    connections.erase(fd);

    debug_log(
        "closed fd=",
        fd
    );

    //
    // max_fd였던 socket이 닫혔다면
    // 새로운 max_fd 계산
    //
    if (fd == max_fd) {
        recompute_max_fd(
            listen_fd
        );
    }
}


void set_write_interest(
    int fd,
    bool want_write
) {
    if (want_write) {
        FD_SET(
            fd,
            &master_write_set
        );
    } else {
        FD_CLR(
            fd,
            &master_write_set
        );
    }
}


void handle_read(
    int listen_fd,
    int fd
) {
    auto it =
        connections.find(fd);

    if (
        it ==
        connections.end()
    ) {
        return;
    }

    Connection& conn =
        it->second;

    if (
        !read_from_socket(
            fd,
            conn
        )
    ) {
        close_client(
            listen_fd,
            fd
        );

        return;
    }

    if (!conn.out_buffer.empty()) {
        set_write_interest(
            fd,
            true
        );
    }
}


void handle_write(
    int listen_fd,
    int fd
) {
    auto it =
        connections.find(fd);

    if (
        it ==
        connections.end()
    ) {
        return;
    }

    Connection& conn =
        it->second;

    if (
        !write_to_socket(
            fd,
            conn
        )
    ) {
        close_client(
            listen_fd,
            fd
        );

        return;
    }

    if (conn.out_buffer.empty()) {
        set_write_interest(
            fd,
            false
        );
    }
}


void accept_clients(
    int listen_fd
) {
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
        // select의 고전적인 한계
        //
        if (client_fd >= FD_SETSIZE) {
            std::cerr
                << "fd="
                << client_fd
                << " exceeds FD_SETSIZE="
                << FD_SETSIZE
                << '\n';

            close(client_fd);
            continue;
        }

        connections[
            client_fd
        ] = {};

        //
        // 새 socket을 read set에 등록
        //
        FD_SET(
            client_fd,
            &master_read_set
        );

        if (client_fd > max_fd) {
            max_fd =
                client_fd;
        }

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
    // select용 fd_set 초기화
    //
    FD_ZERO(
        &master_read_set
    );

    FD_ZERO(
        &master_write_set
    );

    //
    // listen socket 감시
    //
    FD_SET(
        listen_fd,
        &master_read_set
    );

    max_fd =
        listen_fd;


    std::cout
        << "select server listening on port "
        << PORT
        << '\n';


    while (true) {
        //
        // 중요:
        //
        // select()는 fd_set을 직접 수정한다.
        //
        // 그래서 master set을 복사해서
        // working set을 만들어야 한다.
        //
        fd_set read_set =
            master_read_set;

        fd_set write_set =
            master_write_set;


        int ready =
            select(
                max_fd + 1,
                &read_set,
                &write_set,
                nullptr,
                nullptr
            );


        if (ready == -1) {
            if (errno == EINTR) {
                continue;
            }

            perror("select");
            break;
        }


        //
        // select의 핵심:
        //
        // 0 ~ max_fd까지 직접 검사해야 한다.
        //
        for (
            int fd = 0;
            fd <= max_fd;
            ++fd
        ) {
            bool readable =
                FD_ISSET(
                    fd,
                    &read_set
                );

            bool writable =
                FD_ISSET(
                    fd,
                    &write_set
                );

            if (
                !readable &&
                !writable
            ) {
                continue;
            }


            //
            // listening socket
            //
            if (fd == listen_fd) {
                if (readable) {
                    accept_clients(
                        listen_fd
                    );
                }

                continue;
            }


            //
            // readable client
            //
            if (readable) {
                handle_read(
                    listen_fd,
                    fd
                );

                if (
                    connections.find(fd)
                    ==
                    connections.end()
                ) {
                    continue;
                }
            }


            //
            // writable client
            //
            if (writable) {
                handle_write(
                    listen_fd,
                    fd
                );
            }
        }
    }


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