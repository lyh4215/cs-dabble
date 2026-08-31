import asyncio
from enum import Enum


class TxState(Enum):
    IDLE = "IDLE"
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


class Participant:
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance

        self.state = TxState.IDLE
        self.pending_delta = 0

    async def prepare(self, delta):
        print(
            f"{self.name}: PREPARE delta={delta}"
        )

        #
        # 예시:
        # 돈이 음수가 되는 transaction은 거부
        #
        if self.balance + delta < 0:
            print(
                f"{self.name}: VOTE NO"
            )
            return False

        #
        # 아직 실제 balance에는 반영하지 않음.
        #
        # 대신:
        # "나중에 COMMIT이 오면 이 변경을
        # 반드시 수행할 준비가 됐다"
        #
        self.pending_delta = delta
        self.state = TxState.PREPARED

        print(
            f"{self.name}: VOTE YES "
            f"-> PREPARED"
        )

        return True

    async def commit(self):
        if self.state != TxState.PREPARED:
            raise RuntimeError(
                f"{self.name}: not prepared"
            )

        self.balance += self.pending_delta
        self.pending_delta = 0

        self.state = TxState.COMMITTED

        print(
            f"{self.name}: COMMIT "
            f"balance={self.balance}"
        )

    async def abort(self):
        if self.state == TxState.COMMITTED:
            raise RuntimeError(
                f"{self.name}: already committed"
            )

        self.pending_delta = 0
        self.state = TxState.ABORTED

        print(
            f"{self.name}: ABORT"
        )

    def show(self):
        print(
            f"{self.name}: "
            f"balance={self.balance}, "
            f"state={self.state.value}, "
            f"pending={self.pending_delta}"
        )


class Coordinator:
    def __init__(self, participants):
        self.participants = participants

    async def transfer(
        self,
        from_account,
        to_account,
        amount,
        crash_after_prepare=False,
    ):
        print()
        print(
            f"=== TRANSFER {amount} ==="
        )

        votes = await asyncio.gather(
            from_account.prepare(
                -amount
            ),
            to_account.prepare(
                +amount
            ),
        )

        if not all(votes):
            print()
            print(
                "Coordinator: ABORT"
            )

            await asyncio.gather(
                *[
                    p.abort()
                    for p in self.participants
                    if p.state
                    == TxState.PREPARED
                ]
            )

            return

        print()
        print(
            "Coordinator: "
            "ALL PARTICIPANTS PREPARED"
        )

        if crash_after_prepare:
            print()
            print(
                "💥 Coordinator crashed "
                "before sending COMMIT"
            )

            return

        print()
        print(
            "Coordinator: COMMIT"
        )

        await asyncio.gather(
            *[
                p.commit()
                for p in self.participants
            ]
        )


async def main():
    shard_a = Participant(
        "Shard-A",
        balance=100,
    )

    shard_b = Participant(
        "Shard-B",
        balance=50,
    )

    coordinator = Coordinator(
        [
            shard_a,
            shard_b,
        ]
    )

    await coordinator.transfer(
        from_account=shard_a,
        to_account=shard_b,
        amount=30,
        crash_after_prepare=True,
    )

    print()
    print(
        "=== AFTER COORDINATOR CRASH ==="
    )

    shard_a.show()
    shard_b.show()


if __name__ == "__main__":
    asyncio.run(main())