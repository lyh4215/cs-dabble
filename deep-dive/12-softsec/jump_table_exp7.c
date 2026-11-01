
#include <stdio.h>

volatile int global_value = 42;

__attribute__((noinline, used))
int calculate(int x) {
    switch (x) {
        case 0: return x + global_value;
        case 1: return x - global_value;
        case 2: return x * global_value;
        case 3: return x ^ global_value;
        case 4: return x | global_value;
        case 5: return x & global_value;
        case 6: return x << (global_value & 7);
        case 7: return x + global_value * 3;
        default: return -1;
    }
}

int main(int argc, char **argv) {
    int x = argc - 1;
    printf("%d\n", calculate(x));
    return 0;
}
__attribute__((naked, used, noinline))
void code_cave(void) {
    __asm__ volatile(
        ".rept 64\n\t"
        "nop\n\t"
        ".endr\n\t"
        "ret\n\t"
    );
}
