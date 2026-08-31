import time
from dataclasses import dataclass


@dataclass
class Value:
    data: str
    version: int
    node: str


class Replica:
    def __init__(self, name):
        self.name = name
        self.store = {}

    def write(self, key, value):
        old = self.store.get(key)

        if old is None:
            version = 1
        else:
            version = old.version + 1

        self.store[key] = Value(
            data=value,
            version=version,
            node=self.name,
        )

        print(
            f"{self.name}: write "
            f"{key}={value} "
            f"(v{version})"
        )

    def read(self, key):
        return self.store.get(key)

    def merge_from(self, other):
        for key, incoming in other.store.items():
            current = self.store.get(key)

            if current is None:
                self.store[key] = incoming

            elif incoming.version > current.version:
                self.store[key] = incoming


def show(replicas, key):
    print()
    print(f"=== {key} ===")

    for replica in replicas:
        value = replica.read(key)

        if value is None:
            print(
                f"{replica.name}: None"
            )

        else:
            print(
                f"{replica.name}: "
                f"{value.data} "
                f"(v{value.version}, "
                f"from {value.node})"
            )


def main():
    a = Replica("A")
    b = Replica("B")
    c = Replica("C")

    replicas = [a, b, c]

    #
    # 처음에는 모두 같은 값
    #
    for replica in replicas:
        replica.write(
            "x",
            "10",
        )

    show(
        replicas,
        "x",
    )

    print()
    print(
        "=== NETWORK PARTITION ==="
    )

    #
    # A와 C가 서로 통신할 수 없다고 하자.
    #
    a.write(
        "x",
        "20",
    )

    c.write(
        "x",
        "30",
    )

    show(
        replicas,
        "x",
    )

    print()
    print(
        "=== PARTITION HEALED ==="
    )

    #
    # 서로 다시 sync
    #
    a.merge_from(c)
    c.merge_from(a)

    show(
        replicas,
        "x",
    )


if __name__ == "__main__":
    main()