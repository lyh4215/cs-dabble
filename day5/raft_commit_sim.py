from dataclasses import dataclass


@dataclass
class Entry:
    term: int
    command: str


class Node:
    def __init__(self, name):
        self.name = name
        self.log = []
        self.commit_index = -1

    def append(self, entry):
        self.log.append(entry)

    def show(self):
        committed = []

        for i, entry in enumerate(self.log):
            mark = "*" if i <= self.commit_index else " "

            committed.append(
                f"{mark}{entry.command}(t{entry.term})"
            )

        print(
            f"{self.name}: {committed}"
        )


def show_all(nodes):
    print()

    for node in nodes:
        node.show()


def main():
    a = Node("A")
    b = Node("B")
    c = Node("C")

    nodes = [a, b, c]

    print("=== TERM 1: A LEADER ===")

    entry = Entry(
        term=1,
        command="SET x=20",
    )

    #
    # Leader A가 자신의 log에 추가
    #
    a.append(entry)

    #
    # B에게만 replication 성공
    #
    b.append(entry)

    show_all(nodes)

    print()
    print("A crashes BEFORE commit")

    #
    # 이 시점에는:
    #
    # A: x=20
    # B: x=20
    # C: 없음
    #
    # 그런데 commit_index는 모두 -1
    #

    print()
    print("=== TERM 2 ===")

    #
    # C가 새 leader가 됐다고
    # 가정해보자.
    #
    leader = c

    print(
        f"new leader: {leader.name}"
    )

    #
    # leader의 log가 authoritative하다고
    # 단순화해서 followers를 맞춘다.
    #
    for node in [b]:
        node.log = list(
            leader.log
        )

    show_all([b, c])


if __name__ == "__main__":
    main()