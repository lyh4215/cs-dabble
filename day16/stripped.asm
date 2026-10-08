
stripped:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <.init>:
    1000:	f3 0f 1e fa          	endbr64
    1004:	48 83 ec 08          	sub    rsp,0x8
    1008:	48 8b 05 d9 2f 00 00 	mov    rax,QWORD PTR [rip+0x2fd9]        # 3fe8 <__cxa_finalize@plt+0x2f98>
    100f:	48 85 c0             	test   rax,rax
    1012:	74 02                	je     1016 <strtol@plt-0x1a>
    1014:	ff d0                	call   rax
    1016:	48 83 c4 08          	add    rsp,0x8
    101a:	c3                   	ret

Disassembly of section .plt:

0000000000001020 <strtol@plt-0x10>:
    1020:	ff 35 92 2f 00 00    	push   QWORD PTR [rip+0x2f92]        # 3fb8 <__cxa_finalize@plt+0x2f68>
    1026:	ff 25 94 2f 00 00    	jmp    QWORD PTR [rip+0x2f94]        # 3fc0 <__cxa_finalize@plt+0x2f70>
    102c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]

0000000000001030 <strtol@plt>:
    1030:	ff 25 92 2f 00 00    	jmp    QWORD PTR [rip+0x2f92]        # 3fc8 <__cxa_finalize@plt+0x2f78>
    1036:	68 00 00 00 00       	push   0x0
    103b:	e9 e0 ff ff ff       	jmp    1020 <strtol@plt-0x10>

0000000000001040 <__printf_chk@plt>:
    1040:	ff 25 8a 2f 00 00    	jmp    QWORD PTR [rip+0x2f8a]        # 3fd0 <__cxa_finalize@plt+0x2f80>
    1046:	68 01 00 00 00       	push   0x1
    104b:	e9 d0 ff ff ff       	jmp    1020 <strtol@plt-0x10>

Disassembly of section .plt.got:

0000000000001050 <__cxa_finalize@plt>:
    1050:	ff 25 a2 2f 00 00    	jmp    QWORD PTR [rip+0x2fa2]        # 3ff8 <__cxa_finalize@plt+0x2fa8>
    1056:	66 90                	xchg   ax,ax

Disassembly of section .text:

0000000000001060 <.text>:
    1060:	55                   	push   rbp
    1061:	bd 03 00 00 00       	mov    ebp,0x3
    1066:	53                   	push   rbx
    1067:	48 83 ec 08          	sub    rsp,0x8
    106b:	83 ff 01             	cmp    edi,0x1
    106e:	7e 12                	jle    1082 <__cxa_finalize@plt+0x32>
    1070:	48 8b 7e 08          	mov    rdi,QWORD PTR [rsi+0x8]
    1074:	ba 0a 00 00 00       	mov    edx,0xa
    1079:	31 f6                	xor    esi,esi
    107b:	e8 b0 ff ff ff       	call   1030 <strtol@plt>
    1080:	89 c5                	mov    ebp,eax
    1082:	89 ef                	mov    edi,ebp
    1084:	89 ee                	mov    esi,ebp
    1086:	e8 85 01 00 00       	call   1210 <__cxa_finalize@plt+0x1c0>
    108b:	89 ef                	mov    edi,ebp
    108d:	89 c3                	mov    ebx,eax
    108f:	e8 3c 01 00 00       	call   11d0 <__cxa_finalize@plt+0x180>
    1094:	89 ef                	mov    edi,ebp
    1096:	48 8d 35 67 0f 00 00 	lea    rsi,[rip+0xf67]        # 2004 <__cxa_finalize@plt+0xfb4>
    109d:	89 c2                	mov    edx,eax
    109f:	e8 5c 01 00 00       	call   1200 <__cxa_finalize@plt+0x1b0>
    10a4:	01 d3                	add    ebx,edx
    10a6:	bf 02 00 00 00       	mov    edi,0x2
    10ab:	8d 14 03             	lea    edx,[rbx+rax*1]
    10ae:	31 c0                	xor    eax,eax
    10b0:	e8 8b ff ff ff       	call   1040 <__printf_chk@plt>
    10b5:	48 83 c4 08          	add    rsp,0x8
    10b9:	31 c0                	xor    eax,eax
    10bb:	5b                   	pop    rbx
    10bc:	5d                   	pop    rbp
    10bd:	c3                   	ret
    10be:	66 90                	xchg   ax,ax
    10c0:	f3 0f 1e fa          	endbr64
    10c4:	31 ed                	xor    ebp,ebp
    10c6:	49 89 d1             	mov    r9,rdx
    10c9:	5e                   	pop    rsi
    10ca:	48 89 e2             	mov    rdx,rsp
    10cd:	48 83 e4 f0          	and    rsp,0xfffffffffffffff0
    10d1:	50                   	push   rax
    10d2:	54                   	push   rsp
    10d3:	45 31 c0             	xor    r8d,r8d
    10d6:	31 c9                	xor    ecx,ecx
    10d8:	48 8d 3d 81 ff ff ff 	lea    rdi,[rip+0xffffffffffffff81]        # 1060 <__cxa_finalize@plt+0x10>
    10df:	ff 15 f3 2e 00 00    	call   QWORD PTR [rip+0x2ef3]        # 3fd8 <__cxa_finalize@plt+0x2f88>
    10e5:	f4                   	hlt
    10e6:	66 2e 0f 1f 84 00 00 	cs nop WORD PTR [rax+rax*1+0x0]
    10ed:	00 00 00 
    10f0:	48 8d 3d 29 2f 00 00 	lea    rdi,[rip+0x2f29]        # 4020 <__cxa_finalize@plt+0x2fd0>
    10f7:	48 8d 05 22 2f 00 00 	lea    rax,[rip+0x2f22]        # 4020 <__cxa_finalize@plt+0x2fd0>
    10fe:	48 39 f8             	cmp    rax,rdi
    1101:	74 15                	je     1118 <__cxa_finalize@plt+0xc8>
    1103:	48 8b 05 d6 2e 00 00 	mov    rax,QWORD PTR [rip+0x2ed6]        # 3fe0 <__cxa_finalize@plt+0x2f90>
    110a:	48 85 c0             	test   rax,rax
    110d:	74 09                	je     1118 <__cxa_finalize@plt+0xc8>
    110f:	ff e0                	jmp    rax
    1111:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
    1118:	c3                   	ret
    1119:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
    1120:	48 8d 3d f9 2e 00 00 	lea    rdi,[rip+0x2ef9]        # 4020 <__cxa_finalize@plt+0x2fd0>
    1127:	48 8d 35 f2 2e 00 00 	lea    rsi,[rip+0x2ef2]        # 4020 <__cxa_finalize@plt+0x2fd0>
    112e:	48 29 fe             	sub    rsi,rdi
    1131:	48 89 f0             	mov    rax,rsi
    1134:	48 c1 ee 3f          	shr    rsi,0x3f
    1138:	48 c1 f8 03          	sar    rax,0x3
    113c:	48 01 c6             	add    rsi,rax
    113f:	48 d1 fe             	sar    rsi,1
    1142:	74 14                	je     1158 <__cxa_finalize@plt+0x108>
    1144:	48 8b 05 a5 2e 00 00 	mov    rax,QWORD PTR [rip+0x2ea5]        # 3ff0 <__cxa_finalize@plt+0x2fa0>
    114b:	48 85 c0             	test   rax,rax
    114e:	74 08                	je     1158 <__cxa_finalize@plt+0x108>
    1150:	ff e0                	jmp    rax
    1152:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]
    1158:	c3                   	ret
    1159:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
    1160:	f3 0f 1e fa          	endbr64
    1164:	80 3d b5 2e 00 00 00 	cmp    BYTE PTR [rip+0x2eb5],0x0        # 4020 <__cxa_finalize@plt+0x2fd0>
    116b:	75 2b                	jne    1198 <__cxa_finalize@plt+0x148>
    116d:	55                   	push   rbp
    116e:	48 83 3d 82 2e 00 00 	cmp    QWORD PTR [rip+0x2e82],0x0        # 3ff8 <__cxa_finalize@plt+0x2fa8>
    1175:	00 
    1176:	48 89 e5             	mov    rbp,rsp
    1179:	74 0c                	je     1187 <__cxa_finalize@plt+0x137>
    117b:	48 8b 3d 86 2e 00 00 	mov    rdi,QWORD PTR [rip+0x2e86]        # 4008 <__cxa_finalize@plt+0x2fb8>
    1182:	e8 c9 fe ff ff       	call   1050 <__cxa_finalize@plt>
    1187:	e8 64 ff ff ff       	call   10f0 <__cxa_finalize@plt+0xa0>
    118c:	c6 05 8d 2e 00 00 01 	mov    BYTE PTR [rip+0x2e8d],0x1        # 4020 <__cxa_finalize@plt+0x2fd0>
    1193:	5d                   	pop    rbp
    1194:	c3                   	ret
    1195:	0f 1f 00             	nop    DWORD PTR [rax]
    1198:	c3                   	ret
    1199:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
    11a0:	f3 0f 1e fa          	endbr64
    11a4:	e9 77 ff ff ff       	jmp    1120 <__cxa_finalize@plt+0xd0>
    11a9:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
    11b0:	8d 47 0a             	lea    eax,[rdi+0xa]
    11b3:	c3                   	ret
    11b4:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
    11bb:	00 00 00 00 
    11bf:	90                   	nop
    11c0:	8d 04 7f             	lea    eax,[rdi+rdi*2]
    11c3:	c3                   	ret
    11c4:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
    11cb:	00 00 00 00 
    11cf:	90                   	nop
    11d0:	89 f8                	mov    eax,edi
    11d2:	48 8d 15 37 2e 00 00 	lea    rdx,[rip+0x2e37]        # 4010 <__cxa_finalize@plt+0x2fc0>
    11d9:	89 f7                	mov    edi,esi
    11db:	83 e0 01             	and    eax,0x1
    11de:	ff 24 c2             	jmp    QWORD PTR [rdx+rax*8]
    11e1:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
    11e8:	00 00 00 00 
    11ec:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
    11f0:	8d 47 f9             	lea    eax,[rdi-0x7]
    11f3:	c3                   	ret
    11f4:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
    11fb:	00 00 00 00 
    11ff:	90                   	nop
    1200:	eb ee                	jmp    11f0 <__cxa_finalize@plt+0x1a0>
    1202:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1209:	00 00 00 00 
    120d:	0f 1f 00             	nop    DWORD PTR [rax]
    1210:	0f af ff             	imul   edi,edi
    1213:	89 f8                	mov    eax,edi
    1215:	c3                   	ret

Disassembly of section .fini:

0000000000001218 <.fini>:
    1218:	f3 0f 1e fa          	endbr64
    121c:	48 83 ec 08          	sub    rsp,0x8
    1220:	48 83 c4 08          	add    rsp,0x8
    1224:	c3                   	ret
