import torch

torch.manual_seed(0)


# ==================================================
# 1. graph
#
# A community: node 0 ~ 7
# B community: node 8 ~ 15
#
# routing에 사용할 node representation
# A는 -1 근처
# B는 +1 근처
# ==================================================

route_feature = torch.cat([
    torch.full((8, 1), -1.0),
    torch.full((8, 1), +1.0),
])


# ==================================================
# 2. 실제 전달해야 하는 message
#
# node 0  : +1
# node 8  : -1
#
# 나머지는 정보 없음
# ==================================================

h = torch.zeros(16, 1)

h[0] = +1.0
h[8] = -1.0


# 최종적으로
#
# node 7  → +1을 받아야 함
# node 15 → -1을 받아야 함
# ==================================================

target = torch.tensor([
    [+1.0],
    [-1.0],
])


# ==================================================
# 3. node → VN router
#
# 입력:
# node representation 1개
#
# 출력:
# VN0 / VN1 score
# ==================================================

router = torch.nn.Linear(
    1,
    2,
)

optimizer = torch.optim.Adam(
    router.parameters(),
    lr=0.05,
)


# ==================================================
# 4. training
# ==================================================

for step in range(501):

    # ----------------------------------------------
    # 각 node가 VN0/VN1을 얼마나 선호하는지
    # ----------------------------------------------

    logits = router(route_feature)

    assign = torch.softmax(
        logits,
        dim=1,
    )

    # assign shape:
    #
    # node 0 → [p(VN0), p(VN1)]
    # node 1 → [...]
    # ...
    # ----------------------------------------------


    # ----------------------------------------------
    # node → VN message aggregation
    #
    # assign.T : [2, 16]
    # h        : [16, 1]
    #
    # 결과:
    # VN0, VN1 각각 어떤 message를 모았는지
    # ----------------------------------------------

    vn = assign.T @ h


    # ----------------------------------------------
    # VN → node broadcast
    #
    # 자신이 연결된 VN의 정보를 다시 받음
    # ----------------------------------------------

    received = assign @ vn


    # 우리가 관심 있는 마지막 node
    pred = torch.stack([
        received[7, 0],
        received[15, 0],
    ]).reshape(2, 1)


    loss = torch.mean(
        (pred - target) ** 2
    )


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    # ----------------------------------------------
    # monitoring
    # ----------------------------------------------

    if step % 100 == 0:

        with torch.no_grad():

            print(
                f"\nstep={step}",
                f"loss={loss.item():.5f}",
            )

            print(
                "A assignment:",
                assign[0].tolist(),
            )

            print(
                "B assignment:",
                assign[8].tolist(),
            )

            print(
                "VN values:",
                vn.squeeze().tolist(),
            )

            print(
                "prediction:",
                pred.squeeze().tolist(),
            )