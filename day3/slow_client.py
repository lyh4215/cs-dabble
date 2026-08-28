import socket
import time

HOST = "127.0.0.1"
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

print("connected")

time.sleep(30)

sock.close()