from dataclasses import dataclass


@dataclass
class Entry:
    term: int
    command: str


class Node:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def show(self):
        print(f"{self.name}:")
        for i, entry in enumerate(self.log, start=1):
            print(
                f"  index={i} "
                f"term={entry.term} "
                f"cmd={entry.command}"
            )


def append_entries(
    leader,
    follower,
    prev_log_index,
    prev_log_term,
    new_entries,
):
    print()
    print(
        f"Leader asks: "
        f"prev_index={prev_log_index}, "
        f"prev_term={prev_log_term}"
    )

    #
    # index는 Raft처럼 1부터 시작한다고 가정
    #
    if prev_log_index > 0:
        if len(follower.log) < prev_log_index:
            print(
                "Follower: missing prev entry"
            )
            return False

        follower_prev = (
            follower.log[
                prev_log_index - 1
            ]
        )

        if follower_prev.term != prev_log_term:
            print(
                "Follower: prev term mismatch"
            )
            return False

    print(
        "Follower: prefix matches"
    )

    #
    # prev_log_index 뒤에 있는
    # conflicting suffix 제거
    #
    follower.log = (
        follower.log[:prev_log_index]
    )

    #
    # leader의 suffix 붙이기
    #
    follower.log.extend(
        new_entries
    )

    return True


def main():
    leader = Node(
        "A",
        [
            Entry(1, "x"),
            Entry(2, "y"),
            Entry(4, "z"),
        ],
    )

    follower = Node(
        "B",
        [
            Entry(1, "x"),
            Entry(3, "p"),
            Entry(3, "q"),
        ],
    )

    print(
        "=== INITIAL ==="
    )

    leader.show()
    follower.show()

    #
    # leader는 처음에는
    # follower가 index2까지
    # 자기와 같을 거라고 생각
    #
    success = append_entries(
        leader,
        follower,
        prev_log_index=2,
        prev_log_term=2,
        new_entries=[
            Entry(4, "z"),
        ],
    )

    print(
        f"success = {success}"
    )

    #
    # 실패했으니 한 칸 뒤로 가서
    # index1에서 다시 확인
    #
    success = append_entries(
        leader,
        follower,
        prev_log_index=1,
        prev_log_term=1,
        new_entries=[
            Entry(2, "y"),
            Entry(4, "z"),
        ],
    )

    print(
        f"success = {success}"
    )

    print()
    print(
        "=== AFTER REPAIR ==="
    )

    leader.show()
    follower.show()


if __name__ == "__main__":
    main()