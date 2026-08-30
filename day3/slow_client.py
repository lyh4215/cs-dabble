import socket
import struct
import threading
import time


HOST = "127.0.0.1"
PORT = 5000

MESSAGE_SIZE = 4096
MESSAGE_COUNT = 10000


def recv_exact(sock, size):
    data = b""

    while len(data) < size:
        chunk = sock.recv(
            size - len(data)
        )

        if not chunk:
            raise ConnectionError(
                "connection closed"
            )

        data += chunk

    return data


def delayed_reader(
    sock,
    payload,
):
    #
    # 처음 3초 동안은 일부러
    # response를 전혀 읽지 않는다.
    #
    print(
        "reader: sleeping for 3 seconds"
    )

    time.sleep(3)

    print(
        "reader: START reading"
    )

    for i in range(MESSAGE_COUNT):

        header = recv_exact(
            sock,
            4
        )

        (length,) = struct.unpack(
            "!I",
            header
        )

        response = recv_exact(
            sock,
            length
        )

        if response != payload:
            raise RuntimeError(
                "invalid response"
            )

        if i % 100 == 0:
            print(
                f"received={i}"
            )

    print(
        "reader: finished"
    )


sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
)

sock.connect(
    (HOST, PORT)
)

payload = (
    b"x" * MESSAGE_SIZE
)

frame = (
    struct.pack(
        "!I",
        len(payload)
    )
    + payload
)


#
# reader는 별도 thread.
#
reader_thread = threading.Thread(
    target=delayed_reader,
    args=(
        sock,
        payload,
    ),
)

reader_thread.start()


#
# main thread는 계속 request 전송.
#
for i in range(MESSAGE_COUNT):

    sock.sendall(
        frame
    )

    if i % 100 == 0:
        print(
            f"sent={i}"
        )


print(
    "sender: finished"
)

reader_thread.join()

sock.close()