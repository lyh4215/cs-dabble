from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    execution: int   # C
    period: int      # T
    critical: bool


BASE_TASKS = [
    Task("A", 20, 50,  True),
    Task("B", 20, 70,  True),
    Task("C", 30, 120, True),
]

D = Task("D", 5, 100, False)   # U=0.05
E = Task("E", 5, 100, True)    # U=0.05

HORIZON = 600


def simulate(tasks, horizon, policy, switch_cost=True):
    jobs = []

    # FCFS가 현재 잡고 있는 non-preemptive job
    current = None

    # 직전에 CPU에 올라가 있던 job id
    last_job_id = None

    next_job_id = 0

    timeline = []
    misses = []
    context_switches = 0

    for t in range(horizon):

        # 1. periodic task release
        for task in tasks:
            if t % task.period == 0:
                jobs.append({
                    "id": next_job_id,
                    "task": task,
                    "release": t,
                    "deadline": t + task.period,
                    "remaining": task.execution,
                    "missed": False,
                })

                next_job_id += 1

        # 2. deadline 확인
        for job in jobs:
            if (
                job["remaining"] > 0
                and job["deadline"] <= t
                and not job["missed"]
            ):
                job["missed"] = True

                # miss 순간의 상태를 복사해서 저장
                misses.append({
                    "task": job["task"],
                    "release": job["release"],
                    "deadline": job["deadline"],
                    "remaining": job["remaining"],
                })

        ready = [
            job
            for job in jobs
            if job["release"] <= t
            and job["remaining"] > 0
        ]

        # 3. scheduler가 이번 tick에 실행할 job 선택
        selected = None

        if policy == "FCFS":

            # 현재 job이 아직 안 끝났으면 계속 실행
            if current is not None and current["remaining"] > 0:
                selected = current

            elif ready:
                selected = min(
                    ready,
                    key=lambda j: (
                        j["release"],
                        j["id"],
                    ),
                )

                current = selected

            else:
                current = None

        elif policy == "RM":

            if ready:
                selected = min(
                    ready,
                    key=lambda j: (
                        j["task"].period,
                        j["release"],
                        j["id"],
                    ),
                )

        elif policy == "EDF":

            if ready:
                selected = min(
                    ready,
                    key=lambda j: (
                        j["deadline"],
                        j["release"],
                        j["id"],
                    ),
                )

        else:
            raise ValueError(policy)

        # 4. 실행할 job이 없음
        if selected is None:
            timeline.append(".")
            last_job_id = None
            continue

        # 5. 다른 job으로 CPU가 넘어가면 context switch
        #
        # 첫 dispatch와 idle -> task는 무료로 두고,
        # task A -> task B일 때만 1 tick 비용 발생
        if (
            switch_cost
            and last_job_id is not None
            and selected["id"] != last_job_id
        ):
            timeline.append("S")
            context_switches += 1

            # switch가 끝났다고 간주
            last_job_id = selected["id"]

            # 이 tick에는 실제 계산 안 함
            continue

        # 6. 실제 CPU 계산 1 tick
        timeline.append(selected["task"].name)
        selected["remaining"] -= 1

        last_job_id = selected["id"]

    # horizon 끝까지 보면서 deadline이 지난 job 확인
    for job in jobs:
        if (
            job["remaining"] > 0
            and job["deadline"] <= horizon
            and not job["missed"]
        ):
            job["missed"] = True

            misses.append({
                "task": job["task"],
                "release": job["release"],
                "deadline": job["deadline"],
                "remaining": job["remaining"],
            })

    return timeline, misses, context_switches

def print_result(
    policy,
    timeline,
    misses,
    context_switches,
):
    print(f"\n=== {policy} ===")

    print("time : ", end="")
    for t in range(len(timeline)):
        print(t % 10, end="")
    print()

    print("CPU  : ", "".join(timeline))

    print(f"context switches: {context_switches}")
    print(f"deadline misses : {len(misses)}")

    for job in misses:
        print(
            f"  {job['task'].name} "
            f"release={job['release']} "
            f"deadline={job['deadline']} "
            f"remaining={job['remaining']}"
        )

def utilization(tasks):
    return sum(
        task.execution / task.period
        for task in tasks
    )


def admit_task(tasks, new_task):
    candidate = tasks + [new_task]
    candidate_util = utilization(candidate)

    print(f"\nAdmission request: {new_task.name}")
    print(f"critical           : {new_task.critical}")
    print(f"current utilization: {utilization(tasks):.3f}")
    print(f"after admission    : {candidate_util:.3f}")

    # 그냥 들어갈 수 있으면 바로 accept
    if candidate_util <= 1.0:
        print(f"ACCEPT {new_task.name}")
        return candidate

    # CPU가 부족한데 새 task가 non-critical이면 거절
    if not new_task.critical:
        print(
            f"REJECT {new_task.name}: "
            "non-critical task would overload CPU"
        )
        return tasks

    # 여기부터는 새 task가 critical
    print(
        f"{new_task.name} is critical. "
        "Trying to evict non-critical tasks..."
    )

    # 기존 non-critical task들
    non_critical = [
        task
        for task in tasks
        if not task.critical
    ]

    # CPU를 많이 먹는 non-critical task부터 제거
    non_critical.sort(
        key=lambda task:
            task.execution / task.period,
        reverse=True,
    )

    remaining = list(tasks)
    evicted = []

    for task in non_critical:
        remaining.remove(task)
        evicted.append(task)

        new_util = utilization(
            remaining + [new_task]
        )

        print(
            f"  evict {task.name}"
            f" -> utilization={new_util:.3f}"
        )

        if new_util <= 1.0:
            print(
                f"ACCEPT {new_task.name}"
            )

            print(
                "evicted:",
                ", ".join(
                    task.name
                    for task in evicted
                ),
            )

            return remaining + [new_task]

    # non-critical을 전부 제거해도 불가능
    print(
        f"REJECT {new_task.name}: "
        "not enough capacity even after eviction"
    )

    return tasks


def main():
    tasks = list(BASE_TASKS)

    print(
        f"Base utilization: "
        f"{utilization(tasks):.3f}"
    )

    # non-critical D 요청
    tasks = admit_task(
        tasks,
        D,
    )

    # critical E 요청
    tasks = admit_task(
        tasks,
        E,
    )

    print("\nFinal admitted tasks:")

    for task in tasks:
        kind = (
            "CRITICAL"
            if task.critical
            else "NON-CRITICAL"
        )

        print(
            f"  {task.name}: "
            f"C={task.execution}, "
            f"T={task.period}, "
            f"{kind}"
        )

    print(
        f"\nFinal utilization: "
        f"{utilization(tasks):.3f}"
    )

    for policy in ["FCFS", "RM", "EDF"]:
        (
            timeline,
            misses,
            context_switches,
        ) = simulate(
            tasks,
            HORIZON,
            policy,
            switch_cost=True,
        )

        print_result(
            policy,
            timeline,
            misses,
            context_switches,
        )

if __name__ == "__main__":
    main()
