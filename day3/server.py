import socket


HOST = "127.0.0.1"
PORT = 5000


def handle_client(conn, addr):
    print(f"connected: {addr}")

    with conn:
        while True:
            data = conn.recv(65536)

            if not data:
                break

            conn.sendall(data)

    print(f"disconnected: {addr}")


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        server.bind((HOST, PORT))
        server.listen()

        print(f"listening on {HOST}:{PORT}")

        while True:
            conn, addr = server.accept()
            handle_client(conn, addr)


if __name__ == "__main__":
    main()