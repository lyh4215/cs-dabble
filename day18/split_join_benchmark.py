import time


CHUNK = 128

SIZES_KB = [
    1, 2, 4, 8, 16,
    32, 64, 128, 256,
]


def split_copy(data):
    """
    실제 bytes 조각을 새로 생성.
    → 각 chunk마다 data copy 발생
    """

    return [
        data[i:i + CHUNK]
        for i in range(0, len(data), CHUNK)
    ]


def split_view(data):
    """
    원본 data를 가리키는 memoryview만 생성.
    → payload 자체는 copy하지 않음.
    """

    view = memoryview(data)

    return [
        view[i:i + CHUNK]
        for i in range(0, len(data), CHUNK)
    ]


def benchmark(func, data, iterations):

    start = time.perf_counter()

    for _ in range(iterations):
        func(data)

    end = time.perf_counter()

    return (
        (end - start)
        / iterations
        * 1_000_000
    )


def copy_split_join(data):

    parts = split_copy(data)

    result = b"".join(parts)

    assert result == data


def view_split_only(data):

    parts = split_view(data)

    # PIM이 각 piece를 직접 소비한다고 가정.
    # 전체 block으로 다시 join하지 않음.

    _ = parts[-1][0]


print(
    f"{'KB':>6}"
    f"{'Pieces':>10}"
    f"{'Copy+Join(us)':>16}"
    f"{'View(us)':>12}"
)

for kb in SIZES_KB:

    size = kb * 1024

    data = bytes(
        i % 256
        for i in range(size)
    )

    pieces = size // CHUNK

    # 큰 block일수록 iteration 줄이기
    iterations = max(
        50,
        10000 // kb,
    )

    copy_time = benchmark(
        copy_split_join,
        data,
        iterations,
    )

    view_time = benchmark(
        view_split_only,
        data,
        iterations,
    )

    print(
        f"{kb:6d}"
        f"{pieces:10d}"
        f"{copy_time:16.2f}"
        f"{view_time:12.2f}"
    )