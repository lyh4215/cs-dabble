import socket
import ssl


HOST = "127.0.0.1"
PORT = 5443


context = ssl.SSLContext(
    ssl.PROTOCOL_TLS_SERVER
)

context.load_cert_chain(
    certfile="certs/server.crt",
    keyfile="certs/server.key",
)


server = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
)

server.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR,
    1,
)

server.bind(
    (HOST, PORT)
)

server.listen()

print(
    f"TLS server listening "
    f"on {HOST}:{PORT}"
)


while True:
    raw_conn, addr = server.accept()

    print(
        "TCP connected:",
        addr,
    )

    try:
        conn = context.wrap_socket(
            raw_conn,
            server_side=True,
        )

        print(
            "TLS established:",
            conn.version(),
            conn.cipher(),
        )

        data = conn.recv(4096)

        print(
            "received:",
            data.decode(),
        )

        conn.sendall(
            b"hello over TLS"
        )

        conn.close()

    except ssl.SSLError as e:
        print(
            "TLS error:",
            e,
        )

        raw_conn.close()