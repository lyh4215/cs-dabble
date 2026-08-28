import socket
import struct


HOST = "127.0.0.1"
PORT = 5000


def recv_exact(sock, n):
    data = bytearray()

    while len(data) < n:
        chunk = sock.recv(n - len(data))

        if not chunk:
            raise ConnectionError(
                "connection closed"
            )

        data.extend(chunk)

    return bytes(data)


def send_message(sock, message):
    payload = message.encode()

    header = struct.pack(
        "!I",
        len(payload),
    )

    sock.sendall(
        header + payload
    )

# 이렇게 보내도, 원복 가능
# def send_message(sock, message):
#     payload = message.encode()

#     frame = (
#         struct.pack("!I", len(payload))
#         + payload
#     )

#     for byte in frame:
#         sock.sendall(bytes([byte]))
#         time.sleep(0.01)


def recv_message(sock):
    header = recv_exact(sock, 4)

    (length,) = struct.unpack(
        "!I",
        header,
    )

    payload = recv_exact(
        sock,
        length,
    )

    return payload.decode()


with socket.create_connection(
    (HOST, PORT)
) as sock:

    send_message(sock, "hello")
    print(recv_message(sock))

    send_message(sock, "network systems")
    print(recv_message(sock))

    send_message(sock, "epoll!")
    print(recv_message(sock))