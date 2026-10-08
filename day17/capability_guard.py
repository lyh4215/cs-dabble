from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    action: str
    target: str


@dataclass
class AgentAction:
    action: str
    target: str
    reason: str


class CapabilityGuard:
    def __init__(self, capabilities):
        self.capabilities = set(capabilities)

    def authorize(self, action):
        requested = Capability(
            action=action.action,
            target=action.target,
        )

        if requested not in self.capabilities:
            return False, f"capability not granted: {requested}"

        return True, "authorized"


# --------------------------------------------------
# 사용자의 원래 요청:
#
# "현재 기사를 요약해서 나한테 메일 보내줘"
#
# 여기서 최초 권한을 만든다고 가정.
# --------------------------------------------------

user_capabilities = {
    Capability("read", "current_page"),
    Capability("send", "https://mail.local"),
}

guard = CapabilityGuard(user_capabilities)


def run(name, action):
    ok, reason = guard.authorize(action)

    print(f"\n=== {name} ===")
    print("agent wants :", action.action, action.target)
    print("reason      :", action.reason)
    print("result      :", "ALLOW" if ok else "DENY")
    print("guard       :", reason)


# 정상:
# 현재 페이지를 읽는다.
run(
    "1. Read article",
    AgentAction(
        action="read",
        target="current_page",
        reason="Need article text for user's summary request",
    ),
)


# 정상:
# 사용자가 메일 전송을 요청했다.
run(
    "2. Send summary",
    AgentAction(
        action="send",
        target="https://mail.local",
        reason="User asked to email the summary",
    ),
)


# 공격 페이지의 instruction을 LLM이 따라버렸다고 가정.
run(
    "3. Injected mail read",
    AgentAction(
        action="read",
        target="https://mail.local",
        reason="Web page says to read the inbox",
    ),
)


# 공격 페이지가 다른 사이트 접근까지 유도
run(
    "4. Injected bank access",
    AgentAction(
        action="read",
        target="https://bank.local",
        reason="Web page says bank information is needed",
    ),
)