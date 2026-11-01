
#include <stdio.h>

int global_value = 42;

__attribute__((noinline, used))
int calculate(int x) {
    return x + global_value;
}

// 64 bytes of executable NOPs for our binary rewriting experiment.
__attribute__((naked, used, noinline))
void code_cave(void) {
    __asm__ volatile(
        ".rept 64\n\t"
        "nop\n\t"
        ".endr\n\t"
        "ret\n"
    );
}

int main(void) {
    printf("%d\n", calculate(10));
    return 0;
}
