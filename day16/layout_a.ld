ENTRY(_start)

SECTIONS
{
    . = 0x401000;
    .text.start : { *(.text.start) }

    . = 0x402000;
    .text.alpha : { *(.text.alpha) }

    . = 0x403000;
    .text.beta : { *(.text.beta) }

    . = 0x404000;
    .data : { *(.data) }
}