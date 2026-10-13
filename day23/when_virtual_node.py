import torch

torch.manual_seed(0)

# ==================================================
# 1. 두 개의 chain
#
# A: 0 - 1 - 2 - 3
# B: 4 - 5 - 6 - 7
#
# 0과 4만 처음에 자기 community identity를 앎
# ==================================================

N = 8

A = torch.zeros(N, N)


def connect(i, j):
    A[i, j] = 1
    A[j, i] = 1


connect(0, 1)
connect(1, 2)
connect(2, 3)

connect(4, 5)
connect(5, 6)
connect(6, 7)


# ==================================================
# 2. routing용 representation
#
# A identity = -1
# B identity = +1
#
# 처음에는 anchor만 알고 있음
# ==================================================

route_feature = torch.zeros(N, 1)

route_feature[0] = -1.0
route_feature[4] = +1.0


# ==================================================
# 3. 실제 전달해야 할 payload
#
# A source = +1
# B source = -1
#
# target:
# node 3 -> +1
# node 7 -> -1
# ==================================================

payload = torch.zeros(N, 1)

payload[0] = +1.0
payload[4] = -1.0

target = torch.tensor([
    +1.0,
    -1.0,
])


# ==================================================
# 4. 각 layer에서
# VN을 사용한다면 결과가 어떻게 되는지 계산
# ==================================================

candidate_predictions = []

r = route_feature.clone()

for layer in range(4):

    # ----------------------------------------------
    # routing
    #
    # r < 0 -> VN0
    # r > 0 -> VN1
    #
    # 아직 r=0이면
    # [0.5, 0.5]
    # ----------------------------------------------

    logits = torch.cat(
        [
            -3.0 * r,
            +3.0 * r,
        ],
        dim=1,
    )

    assign = torch.softmax(
        logits,
        dim=1,
    )


    # ----------------------------------------------
    # node -> VN
    # ----------------------------------------------

    vn = assign.T @ payload


    # ----------------------------------------------
    # VN -> node
    # ----------------------------------------------

    received = assign @ vn


    pred = torch.stack([
        received[3, 0],
        received[7, 0],
    ])

    candidate_predictions.append(pred)


    print(
        f"\nLayer {layer}"
    )

    print(
        "target route features:",
        r[3].item(),
        r[7].item(),
    )

    print(
        "target assignments:"
    )

    print(
        "node 3:",
        assign[3].tolist(),
    )

    print(
        "node 7:",
        assign[7].tolist(),
    )

    print(
        "VN prediction:",
        pred.tolist(),
    )


    # ----------------------------------------------
    # local message passing
    #
    # representation 정보가
    # graph edge를 따라 한 hop씩 이동
    # ----------------------------------------------

    r = r + A @ r


# ==================================================
# 5. "어느 layer에서 VN을 쓸까?"
#
# layer 0~3 각각에 learnable score
# ==================================================

P = torch.stack(
    candidate_predictions
)

# shape:
# [4 layers, 2 target nodes]

when_logits = torch.nn.Parameter(
    torch.zeros(4)
)

optimizer = torch.optim.Adam(
    [when_logits],
    lr=0.1,
)


print(
    "\n\n--- Learning WHEN to use VN ---"
)


for step in range(201):

    # layer 선택 확률
    layer_weight = torch.softmax(
        when_logits,
        dim=0,
    )

    # 각 layer의 VN output을 weighted combination
    pred = (
        layer_weight[:, None]
        * P
    ).sum(dim=0)

    loss = torch.mean(
        (pred - target) ** 2
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if step % 50 == 0:

        print(
            f"\nstep={step}"
        )

        print(
            "loss:",
            loss.item(),
        )

        print(
            "layer weights:",
            layer_weight.detach().tolist(),
        )

        print(
            "prediction:",
            pred.detach().tolist(),
        )