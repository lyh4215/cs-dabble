# ============================================
# Reported numbers from ICST 2026 Table VI
# ============================================

music = {
    "compilable": 309_800,
    "uncompilable": 110_940,
    "build_cpu_h": 825.0,
    "test_cpu_h": 268.9,
    "total_cpu_h": 1093.3,
    "top1": 60.3,
}

irradiate = {
    "compilable": 224_148,
    "uncompilable": 0,
    "build_cpu_h": 0.0,
    "test_cpu_h": 507.8,
    "total_cpu_h": 507.8,
    "top1": 65.5,
}


def pct(x):
    return f"{x * 100:.1f}%"


# --------------------------------------------
# 1. How much CPU time was removed?
# --------------------------------------------

saved = (
    music["total_cpu_h"]
    - irradiate["total_cpu_h"]
)

reduction = (
    saved
    / music["total_cpu_h"]
)


# --------------------------------------------
# 2. How much of MUSIC cost was build?
# --------------------------------------------

music_build_fraction = (
    music["build_cpu_h"]
    / music["total_cpu_h"]
)


# --------------------------------------------
# 3. How many source mutants failed to compile?
# --------------------------------------------

known_music_mutants = (
    music["compilable"]
    + music["uncompilable"]
)

uncompilable_ratio = (
    music["uncompilable"]
    / known_music_mutants
)


# --------------------------------------------
# 4. Accuracy difference
# --------------------------------------------

top1_gain_pp = (
    irradiate["top1"]
    - music["top1"]
)

relative_gain = (
    top1_gain_pp
    / music["top1"]
)


print("================================")
print("MUSIC — SOURCE LEVEL")
print("================================")

print(
    "compilable mutants:",
    music["compilable"],
)

print(
    "uncompilable mutants:",
    music["uncompilable"],
)

print(
    "uncompilable ratio:",
    pct(uncompilable_ratio),
)

print(
    "build CPU-hours:",
    music["build_cpu_h"],
)

print(
    "test CPU-hours:",
    music["test_cpu_h"],
)

print(
    "total CPU-hours:",
    music["total_cpu_h"],
)


print("\n================================")
print("IRRADIATE — LLVM IR LEVEL")
print("================================")

print(
    "compilable mutants:",
    irradiate["compilable"],
)

print(
    "uncompilable mutants:",
    irradiate["uncompilable"],
)

print(
    "build CPU-hours:",
    irradiate["build_cpu_h"],
)

print(
    "test CPU-hours:",
    irradiate["test_cpu_h"],
)

print(
    "total CPU-hours:",
    irradiate["total_cpu_h"],
)


print("\n================================")
print("COST ANALYSIS")
print("================================")

print(
    "CPU-hours saved:",
    round(saved, 1),
)

print(
    "total reduction:",
    pct(reduction),
)

print(
    "MUSIC cost spent on build:",
    pct(music_build_fraction),
)


print("\n================================")
print("FAULT LOCALIZATION")
print("================================")

print(
    "MUSIC top-1:",
    f'{music["top1"]:.1f}%',
)

print(
    "IRradiate top-1:",
    f'{irradiate["top1"]:.1f}%',
)

print(
    "absolute gain:",
    f"{top1_gain_pp:.1f} percentage points",
)

print(
    "relative gain:",
    pct(relative_gain),
)