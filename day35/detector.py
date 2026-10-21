
import pandas as pd

df = pd.read_csv("sensor_data.csv")

SENSORS = ["hr", "acc", "gps"]

# 1. 각 센서의 결측 여부
missing = df[SENSORS].isna()

# 2. 동시에 누락된 센서 수
df["missing_count"] = missing.sum(axis=1)

# 3. 시간별 결측 패턴
df["pattern"] = missing.apply(
    lambda row: "".join(
        "1" if row[s] else "0"
        for s in SENSORS
    ),
    axis=1
)

print(df[["time", "pattern", "missing_count"]].iloc[15:40])

print("\nPattern distribution:")
print(df["pattern"].value_counts())


def find_missing_runs(patterns: list[str]) -> list[dict]:
    """
    연속된 동일 결측 패턴을 찾아 반환한다.

    정상 패턴("000")은 제외한다.

    Returns:
        [
            {
                "pattern": "100",
                "start": 2,
                "end": 4,
                "length": 3
            },
            ...
        ]
    """
    runs = []

    # TODO: 연속 결측 구간 탐지 구현

    return runs


test = [
    "000", "000", "100", "100",
    "100", "000", "111", "111",
    "000", "001", "000"
]

print(find_missing_runs(test))
