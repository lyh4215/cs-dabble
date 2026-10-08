from dataclasses import dataclass, field


USER = "USER"
PAGE = "PAGE"


@dataclass
class Action:
    kind: str
    target: str

    # 이 행동을 "해도 된다"고 결정하게 만든 출처
    authority_sources: set[str] = field(default_factory=set)

    # 행동에 사용되는 내용의 출처
    data_sources: set[str] = field(default_factory=set)


SENSITIVE_TARGETS = {
    "https://mail.local",
    "https://bank.local",
}


def authorize(action: Action):
    # sensitive cross-origin action은
    # 반드시 사용자에게서 authority가 와야 한다.
    if action.target in SENSITIVE_TARGETS:
        if USER not in action.authority_sources:
            return False, "sensitive action has no USER authority"

    return True, "authorized"


def test(name, action):
    ok, reason = authorize(action)

    print(f"\n=== {name} ===")
    print("action:", action.kind, action.target)
    print("authority:", action.authority_sources)
    print("data:", action.data_sources)
    print("result:", "ALLOW" if ok else "DENY")
    print("reason:", reason)


# --------------------------------------------------
# 1. 사용자가 직접 mail을 읽어달라고 요청
# --------------------------------------------------

test(
    "1. Explicit user request",
    Action(
        kind="read",
        target="https://mail.local",
        authority_sources={USER},
    ),
)


# --------------------------------------------------
# 2. webpage가 agent에게 mail을 읽으라고 지시
# --------------------------------------------------

test(
    "2. Page-injected request",
    Action(
        kind="read",
        target="https://mail.local",
        authority_sources={PAGE},
    ),
)


# --------------------------------------------------
# 3. 사용자가 기사 요약만 요청했는데
#    page가 mail 접근을 유도함
# --------------------------------------------------

test(
    "3. Indirect prompt injection",
    Action(
        kind="read",
        target="https://mail.local",
        authority_sources={PAGE},
        data_sources={PAGE},
    ),
)


# --------------------------------------------------
# 4. 사용자가
#    "이 기사 요약해서 메일 보내줘"
#    라고 명시적으로 요청
#
# mail 전송 권한 = USER
# 기사 내용       = PAGE
# --------------------------------------------------

test(
    "4. Legitimate cross-site task",
    Action(
        kind="send",
        target="https://mail.local",
        authority_sources={USER},
        data_sources={PAGE},
    ),
)