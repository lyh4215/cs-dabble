#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int new_memfd(const char *name) {
#ifdef SYS_memfd_create
    return (int)syscall(SYS_memfd_create, name, 0);
#else
    errno = ENOSYS;
    return -1;
#endif
}

static int check_child_write(volatile unsigned char *addr, bool expect_fault) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == -1) { perror("fork"); return -1; }
    if (pid == 0) {
        *addr = (unsigned char)0xC7;
        _exit(0);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) { perror("waitpid"); return -1; }
    if (expect_fault) {
        if (WIFSIGNALED(status) && WTERMSIG(status) == SIGSEGV) {
            printf("  result: SIGSEGV (as expected)\n");
            return 0;
        }
        printf("  result: unexpected status=0x%x (expected SIGSEGV)\n", status);
        return -1;
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        printf("  result: write succeeded (as expected)\n");
        return 0;
    }
    printf("  result: unexpected status=0x%x (expected success)\n", status);
    return -1;
}

int main(void) {
    long n = sysconf(_SC_PAGESIZE);
    if (n <= 0) { fprintf(stderr, "invalid page size\n"); return 1; }
    size_t sz = (size_t)n;
    int fd = new_memfd("genesis-shadow-alias-demo");
    if (fd < 0) { perror("memfd_create"); return 1; }
    if (ftruncate(fd, (off_t)sz) != 0) { perror("ftruncate"); close(fd); return 1; }

    /* Two distinct virtual mappings of one MAP_SHARED physical backing. */
    unsigned char *read_only = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    unsigned char *shadow_rw = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (read_only == MAP_FAILED || shadow_rw == MAP_FAILED) {
        perror("mmap");
        if (read_only != MAP_FAILED) munmap(read_only, sz);
        if (shadow_rw != MAP_FAILED) munmap(shadow_rw, sz);
        close(fd);
        return 1;
    }
    close(fd);

    printf("=== GENESIS Exp 1: two virtual mappings of one backing page ===\n");
    printf("page size              = %zu\n", sz);
    printf("read-only alias VA     = %p, permissions=R--\n", (void *)read_only);
    printf("writable shadow VA     = %p, permissions=RW-\n", (void *)shadow_rw);
    if (read_only == shadow_rw) { fprintf(stderr, "aliases unexpectedly identical\n"); return 1; }

    shadow_rw[0] = 0x2A;
    printf("\n[1] Inner-style write via RW alias -> observation via RO alias\n");
    printf("  write(shadow_rw[0])=0x%02x; read(read_only[0])=0x%02x\n", shadow_rw[0], read_only[0]);
    if (read_only[0] != 0x2A) return 1;

    printf("\n[2] Attempt direct write via RO alias in a child process\n");
    if (check_child_write(read_only, true) != 0) return 1;
    if (read_only[0] != 0x2A) return 1;

    printf("\n[3] Attempt write via RW alias in a child process\n");
    if (check_child_write(shadow_rw, false) != 0) return 1;
    printf("  after child write: read_only[0]=0x%02x\n", read_only[0]);
    if (read_only[0] != 0xC7) return 1;

    printf("\n[4] Security interpretation\n");
    puts("  RO alias alone is protected, but the RW alias is a bypass if untrusted code can access it.");
    puts("  GENESIS must hide access to its user-marked shadow mapping from the outer kernel with SMAP.");
    puts("  This user-space demo DOES NOT implement SMAP, secure gates, or kernel privilege separation.");
    puts("\nRESULT: PASS (3 behavioral checks)");
    munmap(read_only, sz);
    munmap(shadow_rw, sz);
    return 0;
}
