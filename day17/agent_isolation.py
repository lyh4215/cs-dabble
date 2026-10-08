from dataclasses import dataclass


PAGES = {
    "https://attacker.local": "attacker-controlled page",
    "https://news.local": "Today's public news article",
    "https://mail.local": "PRIVATE MAIL: project meeting at 10 AM",
}


@dataclass(frozen=True)
class Component:
    name: str
    origin: str
    trusted: bool


class Browser:
    # 일반 renderer가 직접 페이지를 읽는 경우
    def renderer_read(self, renderer, target_origin):
        if renderer.origin != target_origin:
            raise PermissionError(
                f"{renderer.origin} cannot directly read {target_origin}"
            )

        return PAGES[target_origin]

    # agent background는 사용자 대신 여러 origin을 다룰 수 있음
    def privileged_read(self, target_origin):
        return PAGES[target_origin]


class AgentBackground:
    def __init__(self, browser, check_origin):
        self.browser = browser
        self.check_origin = check_origin

    def on_message(self, sender, message):
        if message["type"] != "user_task":
            return "ignored"

        # 방어가 있는 버전
        if self.check_origin:
            if not sender.trusted or sender.name != "task-panel":
                return (
                    f"REJECTED: user task from "
                    f"{sender.name} ({sender.origin})"
                )

        if message["action"] == "read":
            return self.browser.privileged_read(
                message["target"]
            )

        return "unsupported action"


browser = Browser()

attacker_renderer = Component(
    name="renderer",
    origin="https://attacker.local",
    trusted=False,
)

task_panel = Component(
    name="task-panel",
    origin="chrome-extension://agent",
    trusted=True,
)


print("=== 1. Ordinary browser isolation ===")

try:
    result = browser.renderer_read(
        attacker_renderer,
        "https://mail.local",
    )
    print(result)
except PermissionError as e:
    print("BLOCKED:", e)


task = {
    "type": "user_task",
    "action": "read",
    "target": "https://mail.local",
}


print("\n=== 2. Legitimate agent request ===")

safe_agent = AgentBackground(
    browser,
    check_origin=True,
)

print(
    safe_agent.on_message(
        task_panel,
        task,
    )
)


print("\n=== 3. Vulnerable IPC design ===")

vulnerable_agent = AgentBackground(
    browser,
    check_origin=False,
)

print(
    vulnerable_agent.on_message(
        attacker_renderer,
        task,
    )
)


print("\n=== 4. Origin-checked IPC ===")

fixed_agent = AgentBackground(
    browser,
    check_origin=True,
)

print(
    fixed_agent.on_message(
        attacker_renderer,
        task,
    )
)