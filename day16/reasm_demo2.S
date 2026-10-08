.section .text.start,"ax"
.global _start

_start:
    # 진짜 function pointer
    mov table(%rip), %rax
    mov $7, %edi
    call *%rax

    # 반환값 저장
    mov %eax, %ebx

    # 그냥 숫자 상수 읽기
    mov magic(%rip), %rax

    # magic 값이 원래 0x402000인지 확인
    cmp $0x402000, %rax
    jne wrong

    mov %ebx, %edi
    mov $60, %eax
    syscall

wrong:
    mov $99, %edi
    mov $60, %eax
    syscall


.section .text.alpha,"ax"
.global alpha

alpha:
    lea 10(%rdi), %eax
    ret


.section .data,"aw"
.align 8

table:
    .quad alpha

magic:
    .quad 0x402000