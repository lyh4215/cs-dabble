// A harmless workload for observing Linux system-call traces.
// Intentionally invokes additional legitimate or irrelevant calls per mode.
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <unistd.h>

static void steady_work(void) {
    const char *path = "/etc/hostname";
    char buf[256] = {0};
    long fd = syscall(SYS_openat, AT_FDCWD, path, O_RDONLY, 0);
    if (fd >= 0) {
        long n = syscall(SYS_read, fd, buf, sizeof(buf) - 1);
        if (n >= 0) syscall(SYS_write, STDOUT_FILENO, buf, (size_t)n);
        syscall(SYS_close, fd);
    }
    (void)syscall(SYS_getpid);
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "steady";
    if (!strcmp(mode, "startup")) return 0;
    if (!strcmp(mode, "steady") || !strcmp(mode, "rare") || !strcmp(mode, "poison")) steady_work();
    if (!strcmp(mode, "rare")) {
        struct rusage usage;
        (void)syscall(SYS_getrusage, RUSAGE_SELF, &usage); // legitimate, rarely used operation
    }
    if (!strcmp(mode, "poison")) {
        struct utsname system_info;
        (void)syscall(SYS_uname, &system_info);  // extra calls from an untrusted observation
        (void)syscall(SYS_getppid);
    }
    return 0;
}
