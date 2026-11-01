
#include <stdio.h>

int global_value = 42;

__attribute__((noinline))
int calculate(int x) {
    int result;

    if (x > 10) {
        result = x + global_value;
    } else {
        result = x - global_value;
    }

    return result;
}

int main(void) {
    printf("%d\n", calculate(15));
    return 0;
}
