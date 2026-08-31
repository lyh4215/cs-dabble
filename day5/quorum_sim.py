import asyncio
import random
from dataclasses import dataclass


@dataclass
class Value:
    data: str
    version: int


class Replica:
    def __init__(self, name):
        self.name = name
        self.store = {}
        self.alive = True

    async def write(self, key, value):
        if not self.alive:
            raise ConnectionError(
                f"{self.name} is down"
            )

        await asyncio.sleep(
            random.uniform(0.05, 0.20)
        )

        self.store[key] = value

        print(
            f"{self.name}: ACK write "
            f"{key}={value.data} "
            f"v{value.version}"
        )

    async def read(self, key):
        if not self.alive:
            raise ConnectionError(
                f"{self.name} is down"
            )

        await asyncio.sleep(
            random.uniform(0.05, 0.20)
        )

        value = self.store.get(key)

        print(
            f"{self.name}: READ "
            f"{value}"
        )

        return value


class Coordinator:
    def __init__(
        self,
        replicas,
        read_quorum,
        write_quorum,
    ):
        self.replicas = replicas

        self.R = read_quorum
        self.W = write_quorum

        self.version = 0

    async def write(
        self,
        key,
        data,
    ):
        self.version += 1

        value = Value(
            data=data,
            version=self.version,
        )

        print()
        print(
            f"WRITE {key}={data} "
            f"v{value.version}"
        )

        tasks = [
            asyncio.create_task(
                replica.write(
                    key,
                    value,
                )
            )
            for replica
            in self.replicas
        ]

        acks = 0

        for task in asyncio.as_completed(
            tasks
        ):
            try:
                await task
                acks += 1

                if acks >= self.W:
                    print(
                        f"WRITE SUCCESS "
                        f"({acks}/{self.W})"
                    )

                    return

            except ConnectionError as e:
                print(e)

        raise RuntimeError(
            "write quorum not reached"
        )

    async def read(self, key):
        print()
        print(
            f"READ {key}"
        )

        tasks = [
            asyncio.create_task(
                replica.read(key)
            )
            for replica
            in self.replicas
        ]

        responses = []

        for task in asyncio.as_completed(
            tasks
        ):
            try:
                value = await task

                responses.append(
                    value
                )

                if len(responses) >= self.R:
                    break

            except ConnectionError as e:
                print(e)

        if len(responses) < self.R:
            raise RuntimeError(
                "read quorum not reached"
            )

        #
        # 응답 중 가장 높은 version 선택
        #
        values = [
            value
            for value in responses
            if value is not None
        ]

        if not values:
            return None

        latest = max(
            values,
            key=lambda value:
                value.version,
        )

        print(
            f"READ RESULT: "
            f"{latest.data} "
            f"v{latest.version}"
        )

        return latest


async def main():
    a = Replica("A")
    b = Replica("B")
    c = Replica("C")

    replicas = [a, b, c]

    coordinator = Coordinator(
        replicas,
        read_quorum=1,
        write_quorum=1,
    )

    #
    # 처음에는 모두 x=10
    #
    await coordinator.write(
        "x",
        "10",
    )

    # 첫 write task들이 모두 끝날 시간을 잠깐 줌
    await asyncio.sleep(0.3)

    print()
    print("=== PARTITION ===")

    #
    # A만 접근 가능
    #
    b.alive = False
    c.alive = False

    await coordinator.write(
        "x",
        "20",
    )

    print()
    print("=== PARTITION HEALED ===")

    #
    # B/C는 x=10인 상태로 다시 돌아옴
    #
    b.alive = True
    c.alive = True

    await coordinator.read("x")


if __name__ == "__main__":
    asyncio.run(main())