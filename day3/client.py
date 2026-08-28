import socket
import time


HOST = "127.0.0.1"
PORT = 5000


def main():
    message = b"hello"

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as sock:

        sock.connect((HOST, PORT))

        start = time.perf_counter()

        sock.sendall(message)

        received = sock.recv(len(message))

        end = time.perf_counter()

    print(f"sent     : {message!r}")
    print(f"received : {received!r}")
    print(
        f"RTT      : "
        f"{(end - start) * 1_000_000:.1f} us"
    )


if __name__ == "__main__":
    main()