.section .text.start,"ax"
.global _start
.type _start,@function

_start:
    mov table(%rip), %rax     # table[0]의 함수 주소를 읽음
    mov $7, %edi
    call *%rax                # 그 함수로 indirect call

    # 함수 반환값을 그대로 exit code로 사용
    mov %eax, %edi
    mov $60, %eax
    syscall


.section .text.alpha,"ax"
.global alpha
.type alpha,@function

alpha:
    lea 10(%rdi), %eax        # alpha(7) = 17
    ret


.section .text.beta,"ax"
.global beta
.type beta,@function

beta:
    lea 20(%rdi), %eax
    ret


.section .data,"aw"
.align 8
.global table

table:
    .quad alpha               # alpha 주소 저장