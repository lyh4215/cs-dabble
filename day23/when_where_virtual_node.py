import torch

torch.manual_seed(0)

# ==================================================
# 1. graph
#
# A: 0 - 1 - 2 - 3
# B: 4 - 5 - 6 - 7
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
# 2. initial node representation
#
# community identity:
# node 0 = -1
# node 4 = +1
#
# 나머지는 아직 모름
# ==================================================

r0 = torch.zeros(N, 1)

r0[0] = -1.0
r0[4] = +1.0


# ==================================================
# 3. payload
#
# 실제 전달해야 할 정보
# ==================================================

payload = torch.zeros(N, 1)

payload[0] = +1.0
payload[4] = -1.0

target = torch.tensor([
    +1.0,   # node 3
    -1.0,   # node 7
])


# ==================================================
# 4. 각 layer의 representation 미리 생성
#
# local message passing
# ==================================================

representations = []

r = r0.clone()

for layer in range(4):

    representations.append(r.clone())

    # local propagation
    r = r + A @ r


# ==================================================
# 5. learnable WHERE router
#
# node representation 1D
# → VN0 / VN1 score
# ==================================================

where_router = torch.nn.Linear(
    1,
    2,
)


# ==================================================
# 6. learnable WHEN logits
#
# layer 0,1,2,3 중
# 어느 layer를 얼마나 사용할지
# ==================================================

when_logits = torch.nn.Parameter(
    torch.zeros(4)
)


optimizer = torch.optim.Adam(
    list(where_router.parameters())
    + [when_logits],
    lr=0.05,
)


# ==================================================
# 7. joint training
# ==================================================

for step in range(1001):

    layer_predictions = []

    layer_assignments = []

    for layer in range(4):

        r = representations[layer]

        # ------------------------------------------
        # WHERE
        #
        # 각 node가 VN0/VN1에 갈 확률
        # ------------------------------------------

        logits = where_router(r)

        assign = torch.softmax(
            logits,
            dim=1,
        )

        layer_assignments.append(assign)


        # ------------------------------------------
        # node -> VN
        # ------------------------------------------

        vn = assign.T @ payload


        # ------------------------------------------
        # VN -> node
        # ------------------------------------------

        received = assign @ vn


        # target nodes 3, 7
        pred = torch.stack([
            received[3, 0],
            received[7, 0],
        ])

        layer_predictions.append(pred)


    # shape:
    # [4 layers, 2 target nodes]
    P = torch.stack(
        layer_predictions
    )


    # ----------------------------------------------
    # WHEN
    #
    # 어떤 layer의 VN output을 믿을지
    # ----------------------------------------------

    layer_weight = torch.softmax(
        when_logits,
        dim=0,
    )


    # 모든 layer 결과를 weighted combination
    final_pred = (
        layer_weight[:, None]
        * P
    ).sum(dim=0)


    loss = torch.mean(
        (final_pred - target) ** 2
    )


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    # ----------------------------------------------
    # monitoring
    # ----------------------------------------------

    if step % 200 == 0:

        print(
            f"\nstep={step}"
        )

        print(
            "loss:",
            loss.item(),
        )

        print(
            "WHEN layer weights:"
        )

        print(
            layer_weight.detach().tolist()
        )

        print(
            "final prediction:",
            final_pred.detach().tolist()
        )


        print(
            "\nWHERE assignments"
        )

        for layer in range(4):

            assign = layer_assignments[layer]

            print(
                f"Layer {layer}:"
            )

            print(
                " node 3:",
                assign[3].detach().tolist(),
            )

            print(
                " node 7:",
                assign[7].detach().tolist(),
            )