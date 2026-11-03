/* Educational x86-64 layout; imitates GENESIS's gate ordering.
 * For assembly/disassembly ONLY. DO NOT execute from userspace.
 */
.text
.globl demo_gateway_entry
.type demo_gateway_entry, @function
demo_gateway_entry:
    pushfq
    cli
    stac
    movq %rsp, %r9
    leaq demo_inner_stack_top(%rip), %rsp
    pushq %r9
    call demo_inner_handler
.globl demo_gateway_exit
.type demo_gateway_exit, @function
demo_gateway_exit:
    movq (%rsp), %rsp
    popfq
    clac
    ret
.globl demo_inner_handler
.type demo_inner_handler, @function
demo_inner_handler:
    jmp demo_gateway_exit
.bss
.balign 16
demo_inner_stack:
    .skip 4096
demo_inner_stack_top:
