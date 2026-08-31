import asyncio
import random
from dataclasses import dataclass


@dataclass
class Value:
    data: str
    version: int
    coordinator: str


class Replica:
    def __init__(self, name):
        self.name = name
        self.store = {}

    async def write(self, key, value):
        await asyncio.sleep(
            random.uniform(0.05, 0.20)
        )

        current = self.store.get(key)

        #
        # 단순히 version이 더 크거나 같으면
        # overwrite한다고 해보자.
        #
        if (
            current is None
            or value.version >= current.version
        ):
            self.store[key] = value

        print(
            f"{self.name}: received "
            f"{value.data} "
            f"v{value.version} "
            f"from {value.coordinator}"
        )

    def read_local(self, key):
        return self.store.get(key)


class Coordinator:
    def __init__(
        self,
        name,
        replicas,
        write_quorum,
    ):
        self.name = name
        self.replicas = replicas
        self.W = write_quorum

        #
        # 각 coordinator가 자기 version을
        # 따로 갖고 있음
        #
        self.version = 1

    async def write(
        self,
        key,
        data,
    ):
        self.version += 1

        value = Value(
            data=data,
            version=self.version,
            coordinator=self.name,
        )

        print()
        print(
            f"{self.name}: WRITE "
            f"{key}={data} "
            f"v{value.version}"
        )

        tasks = [
            asyncio.create_task(
                replica.write(
                    key,
                    value,
                )
            )
            for replica in self.replicas
        ]

        acks = 0

        for task in asyncio.as_completed(tasks):
            await task

            acks += 1

            if acks >= self.W:
                print(
                    f"{self.name}: "
                    f"WRITE SUCCESS"
                )

                return


def show(replicas, key):
    print()
    print("=== FINAL STATE ===")

    for replica in replicas:
        value = replica.read_local(key)

        if value is None:
            print(
                f"{replica.name}: None"
            )
            continue

        print(
            f"{replica.name}: "
            f"{value.data} "
            f"v{value.version} "
            f"from {value.coordinator}"
        )


async def main():
    a = Replica("A")
    b = Replica("B")
    c = Replica("C")

    replicas = [
        a,
        b,
        c,
    ]

    #
    # 처음엔 모두 같은 상태
    #
    initial = Value(
        data="10",
        version=1,
        coordinator="initial",
    )

    for replica in replicas:
        replica.store["x"] = initial

    #
    # Coordinator 두 개
    #
    x = Coordinator(
        "X",
        replicas,
        write_quorum=2,
    )

    y = Coordinator(
        "Y",
        replicas,
        write_quorum=2,
    )

    #
    # 거의 동시에 write
    #
    await asyncio.gather(
        x.write(
            "x",
            "20",
        ),

        y.write(
            "x",
            "30",
        ),
    )

    #
    # 아직 background write들이
    # 남아 있을 수 있으니
    #
    await asyncio.sleep(0.3)

    show(
        replicas,
        "x",
    )


if __name__ == "__main__":
    asyncio.run(main())