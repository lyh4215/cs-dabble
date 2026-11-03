// Exp 2: install a REAL per-process seccomp-BPF allowlist (x86_64 Linux only).
// Policy is loaded after dynamic startup, before our test workload runs.
// This executable never installs host-wide filtering or changes container settings.
#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>

#ifndef AUDIT_ARCH_X86_64
#error This experiment requires x86-64 Linux.
#endif
#if !defined(__x86_64__)
#error This experiment only supports x86-64 Linux; do not reuse the BPF policy for another ABI.
#endif

#include "allowed_syscalls.h" // generated from Exp 1 JSON by the Python runner

static int install_allowlist(void) {
    size_t allow_count = sizeof(allowed_syscalls) / sizeof(allowed_syscalls[0]);
    size_t filter_capacity = 7 + 2 * allow_count;
    struct sock_filter *ins = calloc(filter_capacity, sizeof(*ins));
    if (!ins) { perror("calloc"); return -1; }
    size_t n = 0;
    // Architecture check: kill on wrong ABI, avoiding syscall number confusion.
    ins[n++] = (struct sock_filter)BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch));
    ins[n++] = (struct sock_filter)BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0);
    ins[n++] = (struct sock_filter)BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS);
    ins[n++] = (struct sock_filter)BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr));
    // Reject x32 ABI syscall numbers, which share AUDIT_ARCH_X86_64.
    ins[n++] = (struct sock_filter)BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, 0x40000000, 0, 1);
    ins[n++] = (struct sock_filter)BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS);
    for (size_t i = 0; i < allow_count; ++i) {
        ins[n++] = (struct sock_filter)BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (unsigned int)allowed_syscalls[i], 0, 1);
        ins[n++] = (struct sock_filter)BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
    }
    ins[n++] = (struct sock_filter)BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA));
    struct sock_fprog prog = { .len = (unsigned short)n, .filter = ins };
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        perror("prctl(PR_SET_NO_NEW_PRIVS)"); free(ins); return -1;
    }
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) != 0) {
        perror("prctl(PR_SET_SECCOMP)"); free(ins); return -1;
    }
    free(ins);
    return 0;
}

static int show_result(const char *name, long rv) {
    int saved_errno = errno;
    if (rv >= 0) {
        printf("CALL %-10s => OK\n", name);
        return 0;
    }
    printf("CALL %-10s => DENIED/FAILED errno=%d (%s)\n", name, saved_errno, strerror(saved_errno));
    return 1;
}

static int steady_work(void) {
    int failed = 0;
    const char *path = "/etc/hostname";
    char buf[256] = {0};
    errno = 0;
    long fd = syscall(SYS_openat, AT_FDCWD, path, O_RDONLY, 0);
    failed |= show_result("openat", fd);
    if (fd >= 0) {
        errno = 0;
        long r = syscall(SYS_read, fd, buf, sizeof(buf) - 1);
        failed |= show_result("read", r);
        if (r >= 0) {
            errno = 0;
            failed |= show_result("write", syscall(SYS_write, STDOUT_FILENO, buf, (size_t)r));
        }
        errno = 0;
        failed |= show_result("close", syscall(SYS_close, fd));
    }
    errno = 0;
    failed |= show_result("getpid", syscall(SYS_getpid));
    return failed;
}

int main(int argc, char **argv) {
    if (argc != 2 || (strcmp(argv[1], "steady") && strcmp(argv[1], "rare") && strcmp(argv[1], "poison"))) {
        fprintf(stderr, "Usage: %s steady|rare|poison\n", argv[0]);
        return 2;
    }
    // Flush before loading so buffered output does not mix with error handling.
    setvbuf(stdout, NULL, _IONBF, 0);
    if (install_allowlist() != 0) return 90;
    int failed = steady_work();
    if (!strcmp(argv[1], "rare")) {
        struct rusage u;
        errno = 0;
        failed |= show_result("getrusage", syscall(SYS_getrusage, RUSAGE_SELF, &u));
    }
    if (!strcmp(argv[1], "poison")) {
        struct utsname info;
        errno = 0;
        failed |= show_result("uname", syscall(SYS_uname, &info));
        errno = 0;
        failed |= show_result("getppid", syscall(SYS_getppid));
    }
    printf("RESULT=%s\n", failed ? "FAIL" : "PASS");
    return failed ? 23 : 0;
}
