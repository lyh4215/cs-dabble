import subprocess
import tempfile
from pathlib import Path


VERSIONS = {

"01_raw_pointer": r'''
fn inc(p: *mut i32) {
    unsafe {
        *p += 1;
    }
}

fn sum(xs: *const i32, n: usize) -> i32 {
    let mut result = 0;

    unsafe {
        for i in 0..n {
            result += *xs.add(i);
        }
    }

    result
}

fn process(xs: *mut i32, n: usize) -> i32 {
    inc(xs);

    sum(
        xs as *const i32,
        n
    )
}

fn main() {
    let mut xs = vec![1, 2, 3];

    let result = process(
        xs.as_mut_ptr(),
        xs.len()
    );

    println!("{result}");
}
''',


"02_migrated_but_broken": r'''
fn inc(p: &mut i32) {
    *p += 1;
}

fn sum(xs: &[i32]) -> i32 {
    xs.iter().sum()
}

/*
callee들의 type은 안전하게 migration했지만
caller는 아직 예전 C-style 호출 구조를 유지한다고 가정.
*/

fn process(xs: *mut i32, n: usize) -> i32 {

    inc(xs);

    sum(xs, n)
}

fn main() {
    let mut xs = vec![1, 2, 3];

    let result = process(
        xs.as_mut_ptr(),
        xs.len()
    );

    println!("{result}");
}
''',


"03_migrated_fixed": r'''
fn inc(p: &mut i32) {
    *p += 1;
}

fn sum(xs: &[i32]) -> i32 {
    xs.iter().sum()
}

/*
callee의 새 Rust signature에 맞춰
caller 자체도 restructure.
*/

fn process(xs: &mut [i32]) -> i32 {

    inc(&mut xs[0]);

    sum(xs)
}

fn main() {
    let mut xs = vec![1, 2, 3];

    let result = process(
        &mut xs
    );

    println!("{result}");
}
'''
}


def compile_rust(name, source):

    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        src = tmp / f"{name}.rs"
        exe = tmp / name

        src.write_text(source)

        result = subprocess.run(
            [
                "rustc",
                "--edition=2021",
                str(src),
                "-o",
                str(exe)
            ],
            text=True,
            capture_output=True
        )

        if result.returncode != 0:

            return {
                "compile": False,
                "stderr": result.stderr,
                "output": None
            }


        run = subprocess.run(
            [str(exe)],
            text=True,
            capture_output=True
        )


        return {
            "compile": True,
            "stderr": "",
            "output": run.stdout.strip()
        }


def static_metrics(source):

    return {
        "raw_ptr": (
            source.count("*mut ")
            + source.count("*const ")
        ),

        "unsafe": (
            source.count("unsafe")
        ),

        "slice": (
            source.count("&[")
            + source.count("&mut [")
        ),

        "reference": (
            source.count("&mut i32")
            + source.count("&i32")
        )
    }


print(
    "=== C → RUST TYPE MIGRATION ==="
)

print()


for name, source in VERSIONS.items():

    print(
        f"[{name}]"
    )

    metrics = static_metrics(
        source
    )

    result = compile_rust(
        name,
        source
    )


    print(
        f"  raw pointers : "
        f"{metrics['raw_ptr']}"
    )

    print(
        f"  unsafe       : "
        f"{metrics['unsafe']}"
    )

    print(
        f"  slices       : "
        f"{metrics['slice']}"
    )

    print(
        f"  references   : "
        f"{metrics['reference']}"
    )


    if result["compile"]:

        print(
            "  compile      : OK"
        )

        print(
            f"  output       : "
            f"{result['output']}"
        )

    else:

        print(
            "  compile      : FAIL"
        )

        print()

        # diagnostics가 길어서 앞부분만
        lines = (
            result["stderr"]
            .splitlines()
        )

        for line in lines[:20]:
            print(
                "   ",
                line
            )


    print()