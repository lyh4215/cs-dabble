import torch

torch.set_printoptions(
    precision=2,
    sci_mode=False,
)


# --------------------------------------------------
# message passing
#
# h_new[i] = sum of neighbor features
#
# 학습은 일부러 빼고,
# connectivity 효과만 본다.
# --------------------------------------------------

def propagate(A, h, steps):

    history = [h.clone()]

    for _ in range(steps):
        h = A @ h
        history.append(h.clone())

    return history


# ==================================================
# 1. Plain graph
#
# community A: 0 ... 7
# community B: 8 ... 15
# ==================================================

N = 16

A_plain = torch.zeros(N, N)


def connect(A, i, j):

    A[i, j] = 1
    A[j, i] = 1


for i in range(7):
    connect(A_plain, i, i + 1)

for i in range(8, 15):
    connect(A_plain, i, i + 1)


# source information
h_plain = torch.zeros(N, 1)

h_plain[0] = +1.0
h_plain[8] = -1.0


# ==================================================
# 2. Global Virtual Node
#
# node 16 하나가
# 모든 노드와 연결
# ==================================================

A_global = torch.zeros(17, 17)

A_global[:16, :16] = A_plain

GLOBAL_VN = 16

for i in range(16):
    connect(A_global, i, GLOBAL_VN)


h_global = torch.zeros(17, 1)

h_global[0] = +1.0
h_global[8] = -1.0


# ==================================================
# 3. Selective Virtual Nodes
#
# VN_A = 16
# VN_B = 17
# ==================================================

A_selective = torch.zeros(18, 18)

A_selective[:16, :16] = A_plain

VN_A = 16
VN_B = 17


# community A만 VN_A에 연결
for i in range(8):
    connect(A_selective, i, VN_A)


# community B만 VN_B에 연결
for i in range(8, 16):
    connect(A_selective, i, VN_B)


h_selective = torch.zeros(18, 1)

h_selective[0] = +1.0
h_selective[8] = -1.0


# ==================================================
# propagate
# ==================================================

plain = propagate(
    A_plain,
    h_plain,
    steps=2,
)

global_vn = propagate(
    A_global,
    h_global,
    steps=2,
)

selective = propagate(
    A_selective,
    h_selective,
    steps=2,
)


print("After 2 message-passing layers\n")


print("Plain")
print(
    "node 7 :",
    plain[2][7].item(),
)
print(
    "node 15:",
    plain[2][15].item(),
)


print("\nGlobal VN")
print(
    "VN:",
    global_vn[1][GLOBAL_VN].item(),
)

print(
    "node 7 :",
    global_vn[2][7].item(),
)

print(
    "node 15:",
    global_vn[2][15].item(),
)


print("\nSelective VN")

print(
    "VN A:",
    selective[1][VN_A].item(),
)

print(
    "VN B:",
    selective[1][VN_B].item(),
)

print(
    "node 7 :",
    selective[2][7].item(),
)

print(
    "node 15:",
    selective[2][15].item(),
)