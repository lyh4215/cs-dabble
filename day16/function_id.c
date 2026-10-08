#include <stdio.h>
#include <stdlib.h>

typedef int (*fn_t)(int);


// 직접 호출되지 않고
// function pointer를 통해서만 호출될 함수들
__attribute__((noinline))
int alpha(int x) {
    return x + 10;
}


__attribute__((noinline))
int beta(int x) {
    return x * 3;
}


fn_t table[] = {
    alpha,
    beta,
};


// indirect call
__attribute__((noinline))
int indirect_dispatch(
    int which,
    int x
) {
    fn_t f =
        table[which & 1];

    return f(x);
}


// tail-call optimization 후보
__attribute__((noinline))
int tail_leaf(int x) {
    return x - 7;
}


__attribute__((noinline))
int tail_wrapper(int x) {
    return tail_leaf(x);
}


// 평범한 direct-call 함수
__attribute__((noinline))
int ordinary(int x) {
    return x * x;
}


int main(
    int argc,
    char **argv
) {
    int n =
        argc > 1
        ? atoi(argv[1])
        : 3;

    int a =
        ordinary(n);

    int b =
        indirect_dispatch(
            n,
            n
        );

    int c =
        tail_wrapper(n);

    printf(
        "%d\n",
        a + b + c
    );

    return 0;
}