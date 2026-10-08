from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    action: str
    target: str


CURRENT = Capability("read", "current_page")
MAIL_READ = Capability("read", "mail")
MAIL_SEND = Capability("send", "mail")
CAL_READ = Capability("read", "calendar")
CAL_WRITE = Capability("write", "calendar")
BANK_READ = Capability("read", "bank")


@dataclass
class Task:
    name: str
    actually_required: set[Capability]
    inferred: set[Capability]


tasks = [
    # 정확히 맞춤
    Task(
        name="Summarize article",
        actually_required={CURRENT},
        inferred={CURRENT},
    ),

    # 너무 적게 줌
    Task(
        name="Email article summary",
        actually_required={CURRENT, MAIL_SEND},
        inferred={CURRENT},
    ),

    # 너무 많이 줌
    Task(
        name="Check calendar",
        actually_required={CAL_READ},
        inferred={CAL_READ, CAL_WRITE, MAIL_READ},
    ),

    # 정확히 맞춤
    Task(
        name="Send calendar summary",
        actually_required={CAL_READ, MAIL_SEND},
        inferred={CAL_READ, MAIL_SEND},
    ),
]


for task in tasks:
    missing = task.actually_required - task.inferred
    extra = task.inferred - task.actually_required

    print(f"\n=== {task.name} ===")

    print("required :", task.actually_required)
    print("inferred :", task.inferred)

    if missing:
        print("FALSE DENY RISK")
        print("missing  :", missing)

    if extra:
        print("OVER-PRIVILEGE RISK")
        print("extra    :", extra)

    if not missing and not extra:
        print("EXACT")