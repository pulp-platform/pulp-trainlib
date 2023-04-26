riscv32-unknown-elf-objdump -Mmarch=rv32imcxgap9 -d /home/alberto/pulp-trainlib/tests/test_conv2d/BUILD/PULP/GCC_RISCV//conv2d_fp32/conv2d_fp32

/home/alberto/pulp-trainlib/tests/test_conv2d/BUILD/PULP/GCC_RISCV//conv2d_fp32/conv2d_fp32:     file format elf32-littleriscv


Disassembly of section .vectors:

1c008000 <__irq_vector_base>:
1c008000:	08c0006f          	j	1c00808c <pos_illegal_instr>
1c008004:	08c0006f          	j	1c008090 <pos_no_irq_handler>
1c008008:	0880006f          	j	1c008090 <pos_no_irq_handler>
1c00800c:	0840006f          	j	1c008090 <pos_no_irq_handler>
1c008010:	0800006f          	j	1c008090 <pos_no_irq_handler>
1c008014:	07c0006f          	j	1c008090 <pos_no_irq_handler>
1c008018:	0780006f          	j	1c008090 <pos_no_irq_handler>
1c00801c:	0740006f          	j	1c008090 <pos_no_irq_handler>
1c008020:	0700006f          	j	1c008090 <pos_no_irq_handler>
1c008024:	06c0006f          	j	1c008090 <pos_no_irq_handler>
1c008028:	0680006f          	j	1c008090 <pos_no_irq_handler>
1c00802c:	0640006f          	j	1c008090 <pos_no_irq_handler>
1c008030:	0600006f          	j	1c008090 <pos_no_irq_handler>
1c008034:	05c0006f          	j	1c008090 <pos_no_irq_handler>
1c008038:	0580006f          	j	1c008090 <pos_no_irq_handler>
1c00803c:	0540006f          	j	1c008090 <pos_no_irq_handler>
1c008040:	0500006f          	j	1c008090 <pos_no_irq_handler>
1c008044:	04c0006f          	j	1c008090 <pos_no_irq_handler>
1c008048:	0480006f          	j	1c008090 <pos_no_irq_handler>
1c00804c:	0440006f          	j	1c008090 <pos_no_irq_handler>
1c008050:	0400006f          	j	1c008090 <pos_no_irq_handler>
1c008054:	03c0006f          	j	1c008090 <pos_no_irq_handler>
1c008058:	0380006f          	j	1c008090 <pos_no_irq_handler>
1c00805c:	0340006f          	j	1c008090 <pos_no_irq_handler>
1c008060:	0300006f          	j	1c008090 <pos_no_irq_handler>
1c008064:	02c0006f          	j	1c008090 <pos_no_irq_handler>
1c008068:	0280006f          	j	1c008090 <pos_no_irq_handler>
1c00806c:	0240006f          	j	1c008090 <pos_no_irq_handler>
1c008070:	0200006f          	j	1c008090 <pos_no_irq_handler>
1c008074:	01c0006f          	j	1c008090 <pos_no_irq_handler>
1c008078:	0180006f          	j	1c008090 <pos_no_irq_handler>
1c00807c:	0140006f          	j	1c008090 <pos_no_irq_handler>

1c008080 <_start>:
1c008080:	0000d517          	auipc	a0,0xd
1c008084:	e5650513          	addi	a0,a0,-426 # 1c014ed6 <pos_init_entry>
1c008088:	00050067          	jr	a0

1c00808c <pos_illegal_instr>:
1c00808c:	0000006f          	j	1c00808c <pos_illegal_instr>

1c008090 <pos_no_irq_handler>:
1c008090:	30200073          	mret

1c008094 <pos_semihosting_call>:
1c008094:	00100073          	ebreak
1c008098:	00008067          	ret

Disassembly of section .text:

1c00809c <__udivdi3>:
1c00809c:	87b2                	mv	a5,a2
1c00809e:	8736                	mv	a4,a3
1c0080a0:	88aa                	mv	a7,a0
1c0080a2:	882e                	mv	a6,a1
1c0080a4:	1e069d63          	bnez	a3,1c00829e <__udivdi3+0x202>
1c0080a8:	1c001337          	lui	t1,0x1c001
1c0080ac:	86830313          	addi	t1,t1,-1944 # 1c000868 <__DTOR_END__>
1c0080b0:	0ac5fd63          	bleu	a2,a1,1c00816a <__udivdi3+0xce>
1c0080b4:	6741                	lui	a4,0x10
1c0080b6:	0ae67363          	bleu	a4,a2,1c00815c <__udivdi3+0xc0>
1c0080ba:	0ff00693          	li	a3,255
1c0080be:	00c6b6b3          	sltu	a3,a3,a2
1c0080c2:	068e                	slli	a3,a3,0x3
1c0080c4:	00d65733          	srl	a4,a2,a3
1c0080c8:	933a                	add	t1,t1,a4
1c0080ca:	00034703          	lbu	a4,0(t1)
1c0080ce:	02000313          	li	t1,32
1c0080d2:	96ba                	add	a3,a3,a4
1c0080d4:	40d30333          	sub	t1,t1,a3
1c0080d8:	00030c63          	beqz	t1,1c0080f0 <__udivdi3+0x54>
1c0080dc:	00659733          	sll	a4,a1,t1
1c0080e0:	00d556b3          	srl	a3,a0,a3
1c0080e4:	006617b3          	sll	a5,a2,t1
1c0080e8:	00e6e833          	or	a6,a3,a4
1c0080ec:	006518b3          	sll	a7,a0,t1
1c0080f0:	0107d513          	srli	a0,a5,0x10
1c0080f4:	02a85733          	divu	a4,a6,a0
1c0080f8:	1007d5b3          	p.exthz	a1,a5
1c0080fc:	0108d693          	srli	a3,a7,0x10
1c008100:	02a87633          	remu	a2,a6,a0
1c008104:	02e58833          	mul	a6,a1,a4
1c008108:	0642                	slli	a2,a2,0x10
1c00810a:	8ed1                	or	a3,a3,a2
1c00810c:	863a                	mv	a2,a4
1c00810e:	0106fc63          	bleu	a6,a3,1c008126 <__udivdi3+0x8a>
1c008112:	96be                	add	a3,a3,a5
1c008114:	fff70613          	addi	a2,a4,-1 # ffff <__l1_heap_size+0x17cb>
1c008118:	00f6e763          	bltu	a3,a5,1c008126 <__udivdi3+0x8a>
1c00811c:	0106f563          	bleu	a6,a3,1c008126 <__udivdi3+0x8a>
1c008120:	ffe70613          	addi	a2,a4,-2
1c008124:	96be                	add	a3,a3,a5
1c008126:	410686b3          	sub	a3,a3,a6
1c00812a:	02a6f833          	remu	a6,a3,a0
1c00812e:	02a6d6b3          	divu	a3,a3,a0
1c008132:	df0828b3          	p.insert	a7,a6,15,16
1c008136:	02d58733          	mul	a4,a1,a3
1c00813a:	8536                	mv	a0,a3
1c00813c:	00e8fb63          	bleu	a4,a7,1c008152 <__udivdi3+0xb6>
1c008140:	98be                	add	a7,a7,a5
1c008142:	fff68513          	addi	a0,a3,-1
1c008146:	00f8e663          	bltu	a7,a5,1c008152 <__udivdi3+0xb6>
1c00814a:	00e8f463          	bleu	a4,a7,1c008152 <__udivdi3+0xb6>
1c00814e:	ffe68513          	addi	a0,a3,-2
1c008152:	01061793          	slli	a5,a2,0x10
1c008156:	8fc9                	or	a5,a5,a0
1c008158:	4801                	li	a6,0
1c00815a:	a06d                	j	1c008204 <__udivdi3+0x168>
1c00815c:	01000737          	lui	a4,0x1000
1c008160:	46c1                	li	a3,16
1c008162:	f6e661e3          	bltu	a2,a4,1c0080c4 <__udivdi3+0x28>
1c008166:	46e1                	li	a3,24
1c008168:	bfb1                	j	1c0080c4 <__udivdi3+0x28>
1c00816a:	e601                	bnez	a2,1c008172 <__udivdi3+0xd6>
1c00816c:	4685                	li	a3,1
1c00816e:	02c6d7b3          	divu	a5,a3,a2
1c008172:	66c1                	lui	a3,0x10
1c008174:	08d7fb63          	bleu	a3,a5,1c00820a <__udivdi3+0x16e>
1c008178:	0ff00693          	li	a3,255
1c00817c:	00f6f363          	bleu	a5,a3,1c008182 <__udivdi3+0xe6>
1c008180:	4721                	li	a4,8
1c008182:	00e7d6b3          	srl	a3,a5,a4
1c008186:	9336                	add	t1,t1,a3
1c008188:	00034683          	lbu	a3,0(t1)
1c00818c:	02000613          	li	a2,32
1c008190:	96ba                	add	a3,a3,a4
1c008192:	8e15                	sub	a2,a2,a3
1c008194:	e251                	bnez	a2,1c008218 <__udivdi3+0x17c>
1c008196:	40f58733          	sub	a4,a1,a5
1c00819a:	4805                	li	a6,1
1c00819c:	0107d513          	srli	a0,a5,0x10
1c0081a0:	1007d5b3          	p.exthz	a1,a5
1c0081a4:	0108d693          	srli	a3,a7,0x10
1c0081a8:	02a77633          	remu	a2,a4,a0
1c0081ac:	02a75733          	divu	a4,a4,a0
1c0081b0:	0642                	slli	a2,a2,0x10
1c0081b2:	8ed1                	or	a3,a3,a2
1c0081b4:	02e58333          	mul	t1,a1,a4
1c0081b8:	863a                	mv	a2,a4
1c0081ba:	0066fc63          	bleu	t1,a3,1c0081d2 <__udivdi3+0x136>
1c0081be:	96be                	add	a3,a3,a5
1c0081c0:	fff70613          	addi	a2,a4,-1 # ffffff <__l1_heap_size+0xff17cb>
1c0081c4:	00f6e763          	bltu	a3,a5,1c0081d2 <__udivdi3+0x136>
1c0081c8:	0066f563          	bleu	t1,a3,1c0081d2 <__udivdi3+0x136>
1c0081cc:	ffe70613          	addi	a2,a4,-2
1c0081d0:	96be                	add	a3,a3,a5
1c0081d2:	406686b3          	sub	a3,a3,t1
1c0081d6:	02a6f333          	remu	t1,a3,a0
1c0081da:	02a6d6b3          	divu	a3,a3,a0
1c0081de:	df0328b3          	p.insert	a7,t1,15,16
1c0081e2:	02d58733          	mul	a4,a1,a3
1c0081e6:	8536                	mv	a0,a3
1c0081e8:	00e8fb63          	bleu	a4,a7,1c0081fe <__udivdi3+0x162>
1c0081ec:	98be                	add	a7,a7,a5
1c0081ee:	fff68513          	addi	a0,a3,-1 # ffff <__l1_heap_size+0x17cb>
1c0081f2:	00f8e663          	bltu	a7,a5,1c0081fe <__udivdi3+0x162>
1c0081f6:	00e8f463          	bleu	a4,a7,1c0081fe <__udivdi3+0x162>
1c0081fa:	ffe68513          	addi	a0,a3,-2
1c0081fe:	01061793          	slli	a5,a2,0x10
1c008202:	8fc9                	or	a5,a5,a0
1c008204:	853e                	mv	a0,a5
1c008206:	85c2                	mv	a1,a6
1c008208:	8082                	ret
1c00820a:	010006b7          	lui	a3,0x1000
1c00820e:	4741                	li	a4,16
1c008210:	f6d7e9e3          	bltu	a5,a3,1c008182 <__udivdi3+0xe6>
1c008214:	4761                	li	a4,24
1c008216:	b7b5                	j	1c008182 <__udivdi3+0xe6>
1c008218:	00c797b3          	sll	a5,a5,a2
1c00821c:	00d5d333          	srl	t1,a1,a3
1c008220:	0107de13          	srli	t3,a5,0x10
1c008224:	00c59733          	sll	a4,a1,a2
1c008228:	00c518b3          	sll	a7,a0,a2
1c00822c:	00d555b3          	srl	a1,a0,a3
1c008230:	03c35533          	divu	a0,t1,t3
1c008234:	8dd9                	or	a1,a1,a4
1c008236:	1007d733          	p.exthz	a4,a5
1c00823a:	0105d693          	srli	a3,a1,0x10
1c00823e:	03c37633          	remu	a2,t1,t3
1c008242:	882a                	mv	a6,a0
1c008244:	02a70333          	mul	t1,a4,a0
1c008248:	0642                	slli	a2,a2,0x10
1c00824a:	8ed1                	or	a3,a3,a2
1c00824c:	0066fc63          	bleu	t1,a3,1c008264 <__udivdi3+0x1c8>
1c008250:	96be                	add	a3,a3,a5
1c008252:	fff50813          	addi	a6,a0,-1
1c008256:	00f6e763          	bltu	a3,a5,1c008264 <__udivdi3+0x1c8>
1c00825a:	0066f563          	bleu	t1,a3,1c008264 <__udivdi3+0x1c8>
1c00825e:	ffe50813          	addi	a6,a0,-2
1c008262:	96be                	add	a3,a3,a5
1c008264:	406686b3          	sub	a3,a3,t1
1c008268:	03c6f633          	remu	a2,a3,t3
1c00826c:	03c6d6b3          	divu	a3,a3,t3
1c008270:	df0625b3          	p.insert	a1,a2,15,16
1c008274:	02d70733          	mul	a4,a4,a3
1c008278:	8636                	mv	a2,a3
1c00827a:	00e5fc63          	bleu	a4,a1,1c008292 <__udivdi3+0x1f6>
1c00827e:	95be                	add	a1,a1,a5
1c008280:	fff68613          	addi	a2,a3,-1 # ffffff <__l1_heap_size+0xff17cb>
1c008284:	00f5e763          	bltu	a1,a5,1c008292 <__udivdi3+0x1f6>
1c008288:	00e5f563          	bleu	a4,a1,1c008292 <__udivdi3+0x1f6>
1c00828c:	ffe68613          	addi	a2,a3,-2
1c008290:	95be                	add	a1,a1,a5
1c008292:	0842                	slli	a6,a6,0x10
1c008294:	40e58733          	sub	a4,a1,a4
1c008298:	00c86833          	or	a6,a6,a2
1c00829c:	b701                	j	1c00819c <__udivdi3+0x100>
1c00829e:	12d5ea63          	bltu	a1,a3,1c0083d2 <__udivdi3+0x336>
1c0082a2:	67c1                	lui	a5,0x10
1c0082a4:	02f6fd63          	bleu	a5,a3,1c0082de <__udivdi3+0x242>
1c0082a8:	0ff00793          	li	a5,255
1c0082ac:	00d7b8b3          	sltu	a7,a5,a3
1c0082b0:	088e                	slli	a7,a7,0x3
1c0082b2:	1c001737          	lui	a4,0x1c001
1c0082b6:	0116d7b3          	srl	a5,a3,a7
1c0082ba:	86870713          	addi	a4,a4,-1944 # 1c000868 <__DTOR_END__>
1c0082be:	97ba                	add	a5,a5,a4
1c0082c0:	0007c783          	lbu	a5,0(a5) # 10000 <__l1_heap_size+0x17cc>
1c0082c4:	02000813          	li	a6,32
1c0082c8:	97c6                	add	a5,a5,a7
1c0082ca:	40f80833          	sub	a6,a6,a5
1c0082ce:	00081f63          	bnez	a6,1c0082ec <__udivdi3+0x250>
1c0082d2:	4785                	li	a5,1
1c0082d4:	f2b6e8e3          	bltu	a3,a1,1c008204 <__udivdi3+0x168>
1c0082d8:	04a637b3          	p.sletu	a5,a2,a0
1c0082dc:	b725                	j	1c008204 <__udivdi3+0x168>
1c0082de:	010007b7          	lui	a5,0x1000
1c0082e2:	48c1                	li	a7,16
1c0082e4:	fcf6e7e3          	bltu	a3,a5,1c0082b2 <__udivdi3+0x216>
1c0082e8:	48e1                	li	a7,24
1c0082ea:	b7e1                	j	1c0082b2 <__udivdi3+0x216>
1c0082ec:	00f658b3          	srl	a7,a2,a5
1c0082f0:	010696b3          	sll	a3,a3,a6
1c0082f4:	00d8e6b3          	or	a3,a7,a3
1c0082f8:	00f5d333          	srl	t1,a1,a5
1c0082fc:	0106de13          	srli	t3,a3,0x10
1c008300:	00f55733          	srl	a4,a0,a5
1c008304:	03c377b3          	remu	a5,t1,t3
1c008308:	1006d8b3          	p.exthz	a7,a3
1c00830c:	010595b3          	sll	a1,a1,a6
1c008310:	8f4d                	or	a4,a4,a1
1c008312:	01075593          	srli	a1,a4,0x10
1c008316:	01061633          	sll	a2,a2,a6
1c00831a:	03c35333          	divu	t1,t1,t3
1c00831e:	07c2                	slli	a5,a5,0x10
1c008320:	8ddd                	or	a1,a1,a5
1c008322:	02688eb3          	mul	t4,a7,t1
1c008326:	879a                	mv	a5,t1
1c008328:	01d5fc63          	bleu	t4,a1,1c008340 <__udivdi3+0x2a4>
1c00832c:	95b6                	add	a1,a1,a3
1c00832e:	fff30793          	addi	a5,t1,-1
1c008332:	00d5e763          	bltu	a1,a3,1c008340 <__udivdi3+0x2a4>
1c008336:	01d5f563          	bleu	t4,a1,1c008340 <__udivdi3+0x2a4>
1c00833a:	ffe30793          	addi	a5,t1,-2
1c00833e:	95b6                	add	a1,a1,a3
1c008340:	41d585b3          	sub	a1,a1,t4
1c008344:	03c5f333          	remu	t1,a1,t3
1c008348:	03c5d5b3          	divu	a1,a1,t3
1c00834c:	df032733          	p.insert	a4,t1,15,16
1c008350:	02b888b3          	mul	a7,a7,a1
1c008354:	832e                	mv	t1,a1
1c008356:	01177c63          	bleu	a7,a4,1c00836e <__udivdi3+0x2d2>
1c00835a:	9736                	add	a4,a4,a3
1c00835c:	fff58313          	addi	t1,a1,-1
1c008360:	00d76763          	bltu	a4,a3,1c00836e <__udivdi3+0x2d2>
1c008364:	01177563          	bleu	a7,a4,1c00836e <__udivdi3+0x2d2>
1c008368:	ffe58313          	addi	t1,a1,-2
1c00836c:	9736                	add	a4,a4,a3
1c00836e:	07c2                	slli	a5,a5,0x10
1c008370:	6e41                	lui	t3,0x10
1c008372:	0067e7b3          	or	a5,a5,t1
1c008376:	fffe0593          	addi	a1,t3,-1 # ffff <__l1_heap_size+0x17cb>
1c00837a:	00b7f6b3          	and	a3,a5,a1
1c00837e:	41170733          	sub	a4,a4,a7
1c008382:	8df1                	and	a1,a1,a2
1c008384:	0107d893          	srli	a7,a5,0x10
1c008388:	02b68333          	mul	t1,a3,a1
1c00838c:	8241                	srli	a2,a2,0x10
1c00838e:	02b885b3          	mul	a1,a7,a1
1c008392:	8eae                	mv	t4,a1
1c008394:	42c68eb3          	p.mac	t4,a3,a2
1c008398:	01035693          	srli	a3,t1,0x10
1c00839c:	96f6                	add	a3,a3,t4
1c00839e:	02c888b3          	mul	a7,a7,a2
1c0083a2:	00b6f363          	bleu	a1,a3,1c0083a8 <__udivdi3+0x30c>
1c0083a6:	98f2                	add	a7,a7,t3
1c0083a8:	0106d613          	srli	a2,a3,0x10
1c0083ac:	98b2                	add	a7,a7,a2
1c0083ae:	03176063          	bltu	a4,a7,1c0083ce <__udivdi3+0x332>
1c0083b2:	db1713e3          	bne	a4,a7,1c008158 <__udivdi3+0xbc>
1c0083b6:	6741                	lui	a4,0x10
1c0083b8:	177d                	addi	a4,a4,-1
1c0083ba:	8ef9                	and	a3,a3,a4
1c0083bc:	06c2                	slli	a3,a3,0x10
1c0083be:	00e37333          	and	t1,t1,a4
1c0083c2:	01051533          	sll	a0,a0,a6
1c0083c6:	969a                	add	a3,a3,t1
1c0083c8:	4801                	li	a6,0
1c0083ca:	e2d57de3          	bleu	a3,a0,1c008204 <__udivdi3+0x168>
1c0083ce:	17fd                	addi	a5,a5,-1
1c0083d0:	b361                	j	1c008158 <__udivdi3+0xbc>
1c0083d2:	4801                	li	a6,0
1c0083d4:	4781                	li	a5,0
1c0083d6:	b53d                	j	1c008204 <__udivdi3+0x168>

1c0083d8 <__umoddi3>:
1c0083d8:	88b2                	mv	a7,a2
1c0083da:	8736                	mv	a4,a3
1c0083dc:	87aa                	mv	a5,a0
1c0083de:	882e                	mv	a6,a1
1c0083e0:	1a069963          	bnez	a3,1c008592 <__umoddi3+0x1ba>
1c0083e4:	1c0016b7          	lui	a3,0x1c001
1c0083e8:	86868693          	addi	a3,a3,-1944 # 1c000868 <__DTOR_END__>
1c0083ec:	0ac5f463          	bleu	a2,a1,1c008494 <__umoddi3+0xbc>
1c0083f0:	6341                	lui	t1,0x10
1c0083f2:	08667a63          	bleu	t1,a2,1c008486 <__umoddi3+0xae>
1c0083f6:	0ff00313          	li	t1,255
1c0083fa:	00c37363          	bleu	a2,t1,1c008400 <__umoddi3+0x28>
1c0083fe:	4721                	li	a4,8
1c008400:	00e65333          	srl	t1,a2,a4
1c008404:	969a                	add	a3,a3,t1
1c008406:	0006c683          	lbu	a3,0(a3)
1c00840a:	02000313          	li	t1,32
1c00840e:	9736                	add	a4,a4,a3
1c008410:	40e30333          	sub	t1,t1,a4
1c008414:	00030c63          	beqz	t1,1c00842c <__umoddi3+0x54>
1c008418:	006595b3          	sll	a1,a1,t1
1c00841c:	00e55733          	srl	a4,a0,a4
1c008420:	006618b3          	sll	a7,a2,t1
1c008424:	00b76833          	or	a6,a4,a1
1c008428:	006517b3          	sll	a5,a0,t1
1c00842c:	0108d613          	srli	a2,a7,0x10
1c008430:	02c87733          	remu	a4,a6,a2
1c008434:	1008d533          	p.exthz	a0,a7
1c008438:	0107d693          	srli	a3,a5,0x10
1c00843c:	02c85833          	divu	a6,a6,a2
1c008440:	0742                	slli	a4,a4,0x10
1c008442:	8ed9                	or	a3,a3,a4
1c008444:	03050833          	mul	a6,a0,a6
1c008448:	0106f863          	bleu	a6,a3,1c008458 <__umoddi3+0x80>
1c00844c:	96c6                	add	a3,a3,a7
1c00844e:	0116e563          	bltu	a3,a7,1c008458 <__umoddi3+0x80>
1c008452:	0106f363          	bleu	a6,a3,1c008458 <__umoddi3+0x80>
1c008456:	96c6                	add	a3,a3,a7
1c008458:	410686b3          	sub	a3,a3,a6
1c00845c:	02c6f733          	remu	a4,a3,a2
1c008460:	02c6d6b3          	divu	a3,a3,a2
1c008464:	df0727b3          	p.insert	a5,a4,15,16
1c008468:	02d506b3          	mul	a3,a0,a3
1c00846c:	00d7f863          	bleu	a3,a5,1c00847c <__umoddi3+0xa4>
1c008470:	97c6                	add	a5,a5,a7
1c008472:	0117e563          	bltu	a5,a7,1c00847c <__umoddi3+0xa4>
1c008476:	00d7f363          	bleu	a3,a5,1c00847c <__umoddi3+0xa4>
1c00847a:	97c6                	add	a5,a5,a7
1c00847c:	8f95                	sub	a5,a5,a3
1c00847e:	0067d533          	srl	a0,a5,t1
1c008482:	4581                	li	a1,0
1c008484:	8082                	ret
1c008486:	01000337          	lui	t1,0x1000
1c00848a:	4741                	li	a4,16
1c00848c:	f6666ae3          	bltu	a2,t1,1c008400 <__umoddi3+0x28>
1c008490:	4761                	li	a4,24
1c008492:	b7bd                	j	1c008400 <__umoddi3+0x28>
1c008494:	e601                	bnez	a2,1c00849c <__umoddi3+0xc4>
1c008496:	4605                	li	a2,1
1c008498:	031658b3          	divu	a7,a2,a7
1c00849c:	6641                	lui	a2,0x10
1c00849e:	06c8ff63          	bleu	a2,a7,1c00851c <__umoddi3+0x144>
1c0084a2:	0ff00613          	li	a2,255
1c0084a6:	01167363          	bleu	a7,a2,1c0084ac <__umoddi3+0xd4>
1c0084aa:	4721                	li	a4,8
1c0084ac:	00e8d633          	srl	a2,a7,a4
1c0084b0:	96b2                	add	a3,a3,a2
1c0084b2:	0006c603          	lbu	a2,0(a3)
1c0084b6:	02000313          	li	t1,32
1c0084ba:	963a                	add	a2,a2,a4
1c0084bc:	40c30333          	sub	t1,t1,a2
1c0084c0:	06031563          	bnez	t1,1c00852a <__umoddi3+0x152>
1c0084c4:	411585b3          	sub	a1,a1,a7
1c0084c8:	0108d713          	srli	a4,a7,0x10
1c0084cc:	1008d533          	p.exthz	a0,a7
1c0084d0:	0107d613          	srli	a2,a5,0x10
1c0084d4:	02e5f6b3          	remu	a3,a1,a4
1c0084d8:	02e5d5b3          	divu	a1,a1,a4
1c0084dc:	06c2                	slli	a3,a3,0x10
1c0084de:	8ed1                	or	a3,a3,a2
1c0084e0:	02b505b3          	mul	a1,a0,a1
1c0084e4:	00b6f863          	bleu	a1,a3,1c0084f4 <__umoddi3+0x11c>
1c0084e8:	96c6                	add	a3,a3,a7
1c0084ea:	0116e563          	bltu	a3,a7,1c0084f4 <__umoddi3+0x11c>
1c0084ee:	00b6f363          	bleu	a1,a3,1c0084f4 <__umoddi3+0x11c>
1c0084f2:	96c6                	add	a3,a3,a7
1c0084f4:	40b685b3          	sub	a1,a3,a1
1c0084f8:	02e5f6b3          	remu	a3,a1,a4
1c0084fc:	02e5d5b3          	divu	a1,a1,a4
1c008500:	df06a7b3          	p.insert	a5,a3,15,16
1c008504:	02b505b3          	mul	a1,a0,a1
1c008508:	00b7f863          	bleu	a1,a5,1c008518 <__umoddi3+0x140>
1c00850c:	97c6                	add	a5,a5,a7
1c00850e:	0117e563          	bltu	a5,a7,1c008518 <__umoddi3+0x140>
1c008512:	00b7f363          	bleu	a1,a5,1c008518 <__umoddi3+0x140>
1c008516:	97c6                	add	a5,a5,a7
1c008518:	8f8d                	sub	a5,a5,a1
1c00851a:	b795                	j	1c00847e <__umoddi3+0xa6>
1c00851c:	01000637          	lui	a2,0x1000
1c008520:	4741                	li	a4,16
1c008522:	f8c8e5e3          	bltu	a7,a2,1c0084ac <__umoddi3+0xd4>
1c008526:	4761                	li	a4,24
1c008528:	b751                	j	1c0084ac <__umoddi3+0xd4>
1c00852a:	006898b3          	sll	a7,a7,t1
1c00852e:	00c5d733          	srl	a4,a1,a2
1c008532:	006517b3          	sll	a5,a0,t1
1c008536:	00c55633          	srl	a2,a0,a2
1c00853a:	006595b3          	sll	a1,a1,t1
1c00853e:	0108d513          	srli	a0,a7,0x10
1c008542:	8dd1                	or	a1,a1,a2
1c008544:	02a77633          	remu	a2,a4,a0
1c008548:	1008d833          	p.exthz	a6,a7
1c00854c:	0105d693          	srli	a3,a1,0x10
1c008550:	02a75733          	divu	a4,a4,a0
1c008554:	0642                	slli	a2,a2,0x10
1c008556:	8ed1                	or	a3,a3,a2
1c008558:	02e80733          	mul	a4,a6,a4
1c00855c:	00e6f863          	bleu	a4,a3,1c00856c <__umoddi3+0x194>
1c008560:	96c6                	add	a3,a3,a7
1c008562:	0116e563          	bltu	a3,a7,1c00856c <__umoddi3+0x194>
1c008566:	00e6f363          	bleu	a4,a3,1c00856c <__umoddi3+0x194>
1c00856a:	96c6                	add	a3,a3,a7
1c00856c:	8e99                	sub	a3,a3,a4
1c00856e:	02a6f733          	remu	a4,a3,a0
1c008572:	02a6d6b3          	divu	a3,a3,a0
1c008576:	df0725b3          	p.insert	a1,a4,15,16
1c00857a:	02d806b3          	mul	a3,a6,a3
1c00857e:	00d5f863          	bleu	a3,a1,1c00858e <__umoddi3+0x1b6>
1c008582:	95c6                	add	a1,a1,a7
1c008584:	0115e563          	bltu	a1,a7,1c00858e <__umoddi3+0x1b6>
1c008588:	00d5f363          	bleu	a3,a1,1c00858e <__umoddi3+0x1b6>
1c00858c:	95c6                	add	a1,a1,a7
1c00858e:	8d95                	sub	a1,a1,a3
1c008590:	bf25                	j	1c0084c8 <__umoddi3+0xf0>
1c008592:	eed5e9e3          	bltu	a1,a3,1c008484 <__umoddi3+0xac>
1c008596:	6741                	lui	a4,0x10
1c008598:	04e6f563          	bleu	a4,a3,1c0085e2 <__umoddi3+0x20a>
1c00859c:	0ff00e93          	li	t4,255
1c0085a0:	00deb733          	sltu	a4,t4,a3
1c0085a4:	070e                	slli	a4,a4,0x3
1c0085a6:	1c001337          	lui	t1,0x1c001
1c0085aa:	00e6d8b3          	srl	a7,a3,a4
1c0085ae:	86830313          	addi	t1,t1,-1944 # 1c000868 <__DTOR_END__>
1c0085b2:	989a                	add	a7,a7,t1
1c0085b4:	0008ce83          	lbu	t4,0(a7)
1c0085b8:	02000e13          	li	t3,32
1c0085bc:	9eba                	add	t4,t4,a4
1c0085be:	41de0e33          	sub	t3,t3,t4
1c0085c2:	020e1763          	bnez	t3,1c0085f0 <__umoddi3+0x218>
1c0085c6:	00b6e463          	bltu	a3,a1,1c0085ce <__umoddi3+0x1f6>
1c0085ca:	00c56963          	bltu	a0,a2,1c0085dc <__umoddi3+0x204>
1c0085ce:	40c507b3          	sub	a5,a0,a2
1c0085d2:	8d95                	sub	a1,a1,a3
1c0085d4:	00f53533          	sltu	a0,a0,a5
1c0085d8:	40a58833          	sub	a6,a1,a0
1c0085dc:	853e                	mv	a0,a5
1c0085de:	85c2                	mv	a1,a6
1c0085e0:	b555                	j	1c008484 <__umoddi3+0xac>
1c0085e2:	010008b7          	lui	a7,0x1000
1c0085e6:	4741                	li	a4,16
1c0085e8:	fb16efe3          	bltu	a3,a7,1c0085a6 <__umoddi3+0x1ce>
1c0085ec:	4761                	li	a4,24
1c0085ee:	bf65                	j	1c0085a6 <__umoddi3+0x1ce>
1c0085f0:	01d65733          	srl	a4,a2,t4
1c0085f4:	01c696b3          	sll	a3,a3,t3
1c0085f8:	8ed9                	or	a3,a3,a4
1c0085fa:	01d5d7b3          	srl	a5,a1,t4
1c0085fe:	0106d813          	srli	a6,a3,0x10
1c008602:	0307f333          	remu	t1,a5,a6
1c008606:	1006d733          	p.exthz	a4,a3
1c00860a:	01d558b3          	srl	a7,a0,t4
1c00860e:	01c595b3          	sll	a1,a1,t3
1c008612:	00b8e5b3          	or	a1,a7,a1
1c008616:	0105d893          	srli	a7,a1,0x10
1c00861a:	01c61633          	sll	a2,a2,t3
1c00861e:	01c51533          	sll	a0,a0,t3
1c008622:	0307d7b3          	divu	a5,a5,a6
1c008626:	0342                	slli	t1,t1,0x10
1c008628:	011368b3          	or	a7,t1,a7
1c00862c:	02f70f33          	mul	t5,a4,a5
1c008630:	833e                	mv	t1,a5
1c008632:	01e8fc63          	bleu	t5,a7,1c00864a <__umoddi3+0x272>
1c008636:	98b6                	add	a7,a7,a3
1c008638:	fff78313          	addi	t1,a5,-1 # ffffff <__l1_heap_size+0xff17cb>
1c00863c:	00d8e763          	bltu	a7,a3,1c00864a <__umoddi3+0x272>
1c008640:	01e8f563          	bleu	t5,a7,1c00864a <__umoddi3+0x272>
1c008644:	ffe78313          	addi	t1,a5,-2
1c008648:	98b6                	add	a7,a7,a3
1c00864a:	41e888b3          	sub	a7,a7,t5
1c00864e:	0308f7b3          	remu	a5,a7,a6
1c008652:	0308d8b3          	divu	a7,a7,a6
1c008656:	df07a5b3          	p.insert	a1,a5,15,16
1c00865a:	03170733          	mul	a4,a4,a7
1c00865e:	87c6                	mv	a5,a7
1c008660:	00e5fc63          	bleu	a4,a1,1c008678 <__umoddi3+0x2a0>
1c008664:	95b6                	add	a1,a1,a3
1c008666:	fff88793          	addi	a5,a7,-1 # ffffff <__l1_heap_size+0xff17cb>
1c00866a:	00d5e763          	bltu	a1,a3,1c008678 <__umoddi3+0x2a0>
1c00866e:	00e5f563          	bleu	a4,a1,1c008678 <__umoddi3+0x2a0>
1c008672:	ffe88793          	addi	a5,a7,-2
1c008676:	95b6                	add	a1,a1,a3
1c008678:	0342                	slli	t1,t1,0x10
1c00867a:	6f41                	lui	t5,0x10
1c00867c:	00f36333          	or	t1,t1,a5
1c008680:	ffff0793          	addi	a5,t5,-1 # ffff <__l1_heap_size+0x17cb>
1c008684:	00f37833          	and	a6,t1,a5
1c008688:	01035313          	srli	t1,t1,0x10
1c00868c:	8ff1                	and	a5,a5,a2
1c00868e:	02f808b3          	mul	a7,a6,a5
1c008692:	8d99                	sub	a1,a1,a4
1c008694:	01065713          	srli	a4,a2,0x10
1c008698:	02f307b3          	mul	a5,t1,a5
1c00869c:	8fbe                	mv	t6,a5
1c00869e:	42e80fb3          	p.mac	t6,a6,a4
1c0086a2:	0108d813          	srli	a6,a7,0x10
1c0086a6:	987e                	add	a6,a6,t6
1c0086a8:	02e30333          	mul	t1,t1,a4
1c0086ac:	00f87363          	bleu	a5,a6,1c0086b2 <__umoddi3+0x2da>
1c0086b0:	937a                	add	t1,t1,t5
1c0086b2:	01085713          	srli	a4,a6,0x10
1c0086b6:	933a                	add	t1,t1,a4
1c0086b8:	6741                	lui	a4,0x10
1c0086ba:	177d                	addi	a4,a4,-1
1c0086bc:	00e87833          	and	a6,a6,a4
1c0086c0:	0842                	slli	a6,a6,0x10
1c0086c2:	00e8f733          	and	a4,a7,a4
1c0086c6:	9742                	add	a4,a4,a6
1c0086c8:	0065e663          	bltu	a1,t1,1c0086d4 <__umoddi3+0x2fc>
1c0086cc:	00659d63          	bne	a1,t1,1c0086e6 <__umoddi3+0x30e>
1c0086d0:	00e57b63          	bleu	a4,a0,1c0086e6 <__umoddi3+0x30e>
1c0086d4:	40c70633          	sub	a2,a4,a2
1c0086d8:	00c73733          	sltu	a4,a4,a2
1c0086dc:	40d30333          	sub	t1,t1,a3
1c0086e0:	40e30333          	sub	t1,t1,a4
1c0086e4:	8732                	mv	a4,a2
1c0086e6:	40e50733          	sub	a4,a0,a4
1c0086ea:	00e53533          	sltu	a0,a0,a4
1c0086ee:	406585b3          	sub	a1,a1,t1
1c0086f2:	8d89                	sub	a1,a1,a0
1c0086f4:	01d597b3          	sll	a5,a1,t4
1c0086f8:	01c75733          	srl	a4,a4,t3
1c0086fc:	00e7e533          	or	a0,a5,a4
1c008700:	01c5d5b3          	srl	a1,a1,t3
1c008704:	b341                	j	1c008484 <__umoddi3+0xac>

1c008706 <__gedf2>:
1c008706:	002027f3          	csrr	a5,0x2
1c00870a:	7ff00e93          	li	t4,2047
1c00870e:	d5459733          	p.extractu	a4,a1,10,20
1c008712:	e60597b3          	p.extractu	a5,a1,19,0
1c008716:	e60698b3          	p.extractu	a7,a3,19,0
1c00871a:	d5469833          	p.extractu	a6,a3,10,20
1c00871e:	832a                	mv	t1,a0
1c008720:	81fd                	srli	a1,a1,0x1f
1c008722:	8e32                	mv	t3,a2
1c008724:	82fd                	srli	a3,a3,0x1f
1c008726:	01d71a63          	bne	a4,t4,1c00873a <__gedf2+0x34>
1c00872a:	00a7eeb3          	or	t4,a5,a0
1c00872e:	060e8a63          	beqz	t4,1c0087a2 <__gedf2+0x9c>
1c008732:	00186073          	csrsi	0x1,16
1c008736:	5579                	li	a0,-2
1c008738:	8082                	ret
1c00873a:	01d81a63          	bne	a6,t4,1c00874e <__gedf2+0x48>
1c00873e:	00c8e633          	or	a2,a7,a2
1c008742:	fa65                	bnez	a2,1c008732 <__gedf2+0x2c>
1c008744:	ef19                	bnez	a4,1c008762 <__gedf2+0x5c>
1c008746:	8d5d                	or	a0,a0,a5
1c008748:	00153513          	seqz	a0,a0
1c00874c:	a811                	j	1c008760 <__gedf2+0x5a>
1c00874e:	ef21                	bnez	a4,1c0087a6 <__gedf2+0xa0>
1c008750:	8d5d                	or	a0,a0,a5
1c008752:	00153513          	seqz	a0,a0
1c008756:	00081563          	bnez	a6,1c008760 <__gedf2+0x5a>
1c00875a:	00c8e633          	or	a2,a7,a2
1c00875e:	c221                	beqz	a2,1c00879e <__gedf2+0x98>
1c008760:	e901                	bnez	a0,1c008770 <__gedf2+0x6a>
1c008762:	00d59463          	bne	a1,a3,1c00876a <__gedf2+0x64>
1c008766:	00e85863          	ble	a4,a6,1c008776 <__gedf2+0x70>
1c00876a:	4505                	li	a0,1
1c00876c:	e199                	bnez	a1,1c008772 <__gedf2+0x6c>
1c00876e:	8082                	ret
1c008770:	fefd                	bnez	a3,1c00876e <__gedf2+0x68>
1c008772:	557d                	li	a0,-1
1c008774:	8082                	ret
1c008776:	01075663          	ble	a6,a4,1c008782 <__gedf2+0x7c>
1c00877a:	557d                	li	a0,-1
1c00877c:	d9ed                	beqz	a1,1c00876e <__gedf2+0x68>
1c00877e:	4505                	li	a0,1
1c008780:	8082                	ret
1c008782:	fef8e4e3          	bltu	a7,a5,1c00876a <__gedf2+0x64>
1c008786:	01179863          	bne	a5,a7,1c008796 <__gedf2+0x90>
1c00878a:	fe6e60e3          	bltu	t3,t1,1c00876a <__gedf2+0x64>
1c00878e:	4501                	li	a0,0
1c008790:	fdc37fe3          	bleu	t3,t1,1c00876e <__gedf2+0x68>
1c008794:	b7dd                	j	1c00877a <__gedf2+0x74>
1c008796:	ff17e2e3          	bltu	a5,a7,1c00877a <__gedf2+0x74>
1c00879a:	4501                	li	a0,0
1c00879c:	8082                	ret
1c00879e:	fd75                	bnez	a0,1c00879a <__gedf2+0x94>
1c0087a0:	b7e9                	j	1c00876a <__gedf2+0x64>
1c0087a2:	f8e80ee3          	beq	a6,a4,1c00873e <__gedf2+0x38>
1c0087a6:	4501                	li	a0,0
1c0087a8:	fa0809e3          	beqz	a6,1c00875a <__gedf2+0x54>
1c0087ac:	bf5d                	j	1c008762 <__gedf2+0x5c>

1c0087ae <__ledf2>:
1c0087ae:	002027f3          	csrr	a5,0x2
1c0087b2:	7ff00e93          	li	t4,2047
1c0087b6:	d5459733          	p.extractu	a4,a1,10,20
1c0087ba:	e60597b3          	p.extractu	a5,a1,19,0
1c0087be:	e60698b3          	p.extractu	a7,a3,19,0
1c0087c2:	d5469833          	p.extractu	a6,a3,10,20
1c0087c6:	832a                	mv	t1,a0
1c0087c8:	81fd                	srli	a1,a1,0x1f
1c0087ca:	8e32                	mv	t3,a2
1c0087cc:	82fd                	srli	a3,a3,0x1f
1c0087ce:	01d71a63          	bne	a4,t4,1c0087e2 <__ledf2+0x34>
1c0087d2:	00a7eeb3          	or	t4,a5,a0
1c0087d6:	060e8a63          	beqz	t4,1c00884a <__ledf2+0x9c>
1c0087da:	00186073          	csrsi	0x1,16
1c0087de:	4509                	li	a0,2
1c0087e0:	8082                	ret
1c0087e2:	01d81a63          	bne	a6,t4,1c0087f6 <__ledf2+0x48>
1c0087e6:	00c8e633          	or	a2,a7,a2
1c0087ea:	fa65                	bnez	a2,1c0087da <__ledf2+0x2c>
1c0087ec:	ef19                	bnez	a4,1c00880a <__ledf2+0x5c>
1c0087ee:	8d5d                	or	a0,a0,a5
1c0087f0:	00153513          	seqz	a0,a0
1c0087f4:	a811                	j	1c008808 <__ledf2+0x5a>
1c0087f6:	ef21                	bnez	a4,1c00884e <__ledf2+0xa0>
1c0087f8:	8d5d                	or	a0,a0,a5
1c0087fa:	00153513          	seqz	a0,a0
1c0087fe:	00081563          	bnez	a6,1c008808 <__ledf2+0x5a>
1c008802:	00c8e633          	or	a2,a7,a2
1c008806:	c221                	beqz	a2,1c008846 <__ledf2+0x98>
1c008808:	e901                	bnez	a0,1c008818 <__ledf2+0x6a>
1c00880a:	00d59463          	bne	a1,a3,1c008812 <__ledf2+0x64>
1c00880e:	00e85863          	ble	a4,a6,1c00881e <__ledf2+0x70>
1c008812:	4505                	li	a0,1
1c008814:	e199                	bnez	a1,1c00881a <__ledf2+0x6c>
1c008816:	8082                	ret
1c008818:	fefd                	bnez	a3,1c008816 <__ledf2+0x68>
1c00881a:	557d                	li	a0,-1
1c00881c:	8082                	ret
1c00881e:	01075663          	ble	a6,a4,1c00882a <__ledf2+0x7c>
1c008822:	557d                	li	a0,-1
1c008824:	d9ed                	beqz	a1,1c008816 <__ledf2+0x68>
1c008826:	4505                	li	a0,1
1c008828:	8082                	ret
1c00882a:	fef8e4e3          	bltu	a7,a5,1c008812 <__ledf2+0x64>
1c00882e:	01179863          	bne	a5,a7,1c00883e <__ledf2+0x90>
1c008832:	fe6e60e3          	bltu	t3,t1,1c008812 <__ledf2+0x64>
1c008836:	4501                	li	a0,0
1c008838:	fdc37fe3          	bleu	t3,t1,1c008816 <__ledf2+0x68>
1c00883c:	b7dd                	j	1c008822 <__ledf2+0x74>
1c00883e:	ff17e2e3          	bltu	a5,a7,1c008822 <__ledf2+0x74>
1c008842:	4501                	li	a0,0
1c008844:	8082                	ret
1c008846:	fd75                	bnez	a0,1c008842 <__ledf2+0x94>
1c008848:	b7e9                	j	1c008812 <__ledf2+0x64>
1c00884a:	f8e80ee3          	beq	a6,a4,1c0087e6 <__ledf2+0x38>
1c00884e:	4501                	li	a0,0
1c008850:	fa0809e3          	beqz	a6,1c008802 <__ledf2+0x54>
1c008854:	bf5d                	j	1c00880a <__ledf2+0x5c>

1c008856 <__extendsfdf2>:
1c008856:	002027f3          	csrr	a5,0x2
1c00885a:	cf751633          	p.extractu	a2,a0,7,23
1c00885e:	00160713          	addi	a4,a2,1 # 1000001 <__l1_heap_size+0xff17cd>
1c008862:	ee873733          	p.bclr	a4,a4,23,8
1c008866:	4685                	li	a3,1
1c008868:	ec0517b3          	p.extractu	a5,a0,22,0
1c00886c:	817d                	srli	a0,a0,0x1f
1c00886e:	00e6d963          	ble	a4,a3,1c008880 <__extendsfdf2+0x2a>
1c008872:	0037d713          	srli	a4,a5,0x3
1c008876:	38060613          	addi	a2,a2,896
1c00887a:	07f6                	slli	a5,a5,0x1d
1c00887c:	4681                	li	a3,0
1c00887e:	a899                	j	1c0088d4 <__extendsfdf2+0x7e>
1c008880:	ee05                	bnez	a2,1c0088b8 <__extendsfdf2+0x62>
1c008882:	c7ad                	beqz	a5,1c0088ec <__extendsfdf2+0x96>
1c008884:	477d                	li	a4,31
1c008886:	100796b3          	p.fl1	a3,a5
1c00888a:	40d706b3          	sub	a3,a4,a3
1c00888e:	4729                	li	a4,10
1c008890:	00d74e63          	blt	a4,a3,1c0088ac <__extendsfdf2+0x56>
1c008894:	472d                	li	a4,11
1c008896:	8f15                	sub	a4,a4,a3
1c008898:	01568613          	addi	a2,a3,21
1c00889c:	00e7d733          	srl	a4,a5,a4
1c0088a0:	00c797b3          	sll	a5,a5,a2
1c0088a4:	38900613          	li	a2,905
1c0088a8:	8e15                	sub	a2,a2,a3
1c0088aa:	bfc9                	j	1c00887c <__extendsfdf2+0x26>
1c0088ac:	ff568713          	addi	a4,a3,-11
1c0088b0:	00e79733          	sll	a4,a5,a4
1c0088b4:	4781                	li	a5,0
1c0088b6:	b7fd                	j	1c0088a4 <__extendsfdf2+0x4e>
1c0088b8:	cf8d                	beqz	a5,1c0088f2 <__extendsfdf2+0x9c>
1c0088ba:	004006b7          	lui	a3,0x400
1c0088be:	8efd                	and	a3,a3,a5
1c0088c0:	0037d713          	srli	a4,a5,0x3
1c0088c4:	0016b693          	seqz	a3,a3
1c0088c8:	0692                	slli	a3,a3,0x4
1c0088ca:	c1374733          	p.bset	a4,a4,0,19
1c0088ce:	07f6                	slli	a5,a5,0x1d
1c0088d0:	7ff00613          	li	a2,2047
1c0088d4:	4581                	li	a1,0
1c0088d6:	e60725b3          	p.insert	a1,a4,19,0
1c0088da:	d54625b3          	p.insert	a1,a2,10,20
1c0088de:	c1f525b3          	p.insert	a1,a0,0,31
1c0088e2:	853e                	mv	a0,a5
1c0088e4:	c299                	beqz	a3,1c0088ea <__extendsfdf2+0x94>
1c0088e6:	0016a073          	csrs	0x1,a3
1c0088ea:	8082                	ret
1c0088ec:	4701                	li	a4,0
1c0088ee:	4601                	li	a2,0
1c0088f0:	b771                	j	1c00887c <__extendsfdf2+0x26>
1c0088f2:	4701                	li	a4,0
1c0088f4:	7ff00613          	li	a2,2047
1c0088f8:	b751                	j	1c00887c <__extendsfdf2+0x26>

1c0088fa <main>:
1c0088fa:	1c001537          	lui	a0,0x1c001
1c0088fe:	711d                	addi	sp,sp,-96
1c008900:	96850513          	addi	a0,a0,-1688 # 1c000968 <__DTOR_END__+0x100>
1c008904:	ce86                	sw	ra,92(sp)
1c008906:	2420a0ef          	jal	ra,1c012b48 <puts>
1c00890a:	0808                	addi	a0,sp,16
1c00890c:	31c0c0ef          	jal	ra,1c014c28 <pi_cluster_conf_init>
1c008910:	0048                	addi	a0,sp,4
1c008912:	080c                	addi	a1,sp,16
1c008914:	6d70b0ef          	jal	ra,1c0147ea <pi_open_from_conf>
1c008918:	0048                	addi	a0,sp,4
1c00891a:	31a0c0ef          	jal	ra,1c014c34 <pi_cluster_open>
1c00891e:	c509                	beqz	a0,1c008928 <main+0x2e>
1c008920:	40f6                	lw	ra,92(sp)
1c008922:	557d                	li	a0,-1
1c008924:	6125                	addi	sp,sp,96
1c008926:	8082                	ret
1c008928:	1c001537          	lui	a0,0x1c001
1c00892c:	98c50513          	addi	a0,a0,-1652 # 1c00098c <__DTOR_END__+0x124>
1c008930:	2180a0ef          	jal	ra,1c012b48 <puts>
1c008934:	1c0097b7          	lui	a5,0x1c009
1c008938:	97078793          	addi	a5,a5,-1680 # 1c008970 <net_step>
1c00893c:	102c                	addi	a1,sp,40
1c00893e:	d43e                	sw	a5,40(sp)
1c008940:	0048                	addi	a0,sp,4
1c008942:	47a1                	li	a5,8
1c008944:	de3e                	sw	a5,60(sp)
1c008946:	d602                	sw	zero,44(sp)
1c008948:	d802                	sw	zero,48(sp)
1c00894a:	da02                	sw	zero,52(sp)
1c00894c:	4e20c0ef          	jal	ra,1c014e2e <pi_cluster_send_task_to_cl>
1c008950:	1c001537          	lui	a0,0x1c001
1c008954:	9b050513          	addi	a0,a0,-1616 # 1c0009b0 <__DTOR_END__+0x148>
1c008958:	1f00a0ef          	jal	ra,1c012b48 <puts>
1c00895c:	0048                	addi	a0,sp,4
1c00895e:	44a0c0ef          	jal	ra,1c014da8 <pi_cluster_close>
1c008962:	1c0017b7          	lui	a5,0x1c001
1c008966:	4501                	li	a0,0
1c008968:	7c07a023          	sw	zero,1984(a5) # 1c0017c0 <pos_kernel_pmsis_exit_value>
1c00896c:	2760a0ef          	jal	ra,1c012be2 <exit>

1c008970 <net_step>:
1c008970:	711d                	addi	sp,sp,-96
1c008972:	000317b7          	lui	a5,0x31
1c008976:	ce86                	sw	ra,92(sp)
1c008978:	cca2                	sw	s0,88(sp)
1c00897a:	caa6                	sw	s1,84(sp)
1c00897c:	c8ca                	sw	s2,80(sp)
1c00897e:	c6ce                	sw	s3,76(sp)
1c008980:	c4d2                	sw	s4,72(sp)
1c008982:	c2d6                	sw	s5,68(sp)
1c008984:	c0da                	sw	s6,64(sp)
1c008986:	de5e                	sw	s7,60(sp)
1c008988:	dc62                	sw	s8,56(sp)
1c00898a:	da66                	sw	s9,52(sp)
1c00898c:	d86a                	sw	s10,48(sp)
1c00898e:	d66e                	sw	s11,44(sp)
1c008990:	07dd                	addi	a5,a5,23
1c008992:	cc079073          	csrw	0xcc0,a5
1c008996:	1c015ab7          	lui	s5,0x1c015
1c00899a:	284a8a93          	addi	s5,s5,644 # 1c015284 <L1_memocc_bytes>
1c00899e:	000aa783          	lw	a5,0(s5)
1c0089a2:	6689                	lui	a3,0x2
1c0089a4:	004aa703          	lw	a4,4(s5)
1c0089a8:	a8068693          	addi	a3,a3,-1408 # 1a80 <_l1_preload_size+0x2b8>
1c0089ac:	97b6                	add	a5,a5,a3
1c0089ae:	6685                	lui	a3,0x1
1c0089b0:	e6068693          	addi	a3,a3,-416 # e60 <pos_soc_event_callback+0xa18>
1c0089b4:	1c001537          	lui	a0,0x1c001
1c0089b8:	9736                	add	a4,a4,a3
1c0089ba:	85be                	mv	a1,a5
1c0089bc:	a2c50513          	addi	a0,a0,-1492 # 1c000a2c <__DTOR_END__+0x1c4>
1c0089c0:	00faa023          	sw	a5,0(s5)
1c0089c4:	00eaa223          	sw	a4,4(s5)
1c0089c8:	24e0a0ef          	jal	ra,1c012c16 <printf>
1c0089cc:	004aa583          	lw	a1,4(s5)
1c0089d0:	1c001537          	lui	a0,0x1c001
1c0089d4:	a5050513          	addi	a0,a0,-1456 # 1c000a50 <__DTOR_END__+0x1e8>
1c0089d8:	23e0a0ef          	jal	ra,1c012c16 <printf>
1c0089dc:	100007b7          	lui	a5,0x10000
1c0089e0:	01c78913          	addi	s2,a5,28 # 1000001c <l1_in>
1c0089e4:	008a8713          	addi	a4,s5,8
1c0089e8:	01c78793          	addi	a5,a5,28
1c0089ec:	0326d0fb          	lp.setupi	x1,50,1c008a06 <net_step+0x96>
1c0089f0:	00072803          	lw	a6,0(a4) # 10000 <__l1_heap_size+0x17cc>
1c0089f4:	4348                	lw	a0,4(a4)
1c0089f6:	470c                	lw	a1,8(a4)
1c0089f8:	4750                	lw	a2,12(a4)
1c0089fa:	0107a023          	sw	a6,0(a5)
1c0089fe:	c3c8                	sw	a0,4(a5)
1c008a00:	c78c                	sw	a1,8(a5)
1c008a02:	c7d0                	sw	a2,12(a5)
1c008a04:	0741                	addi	a4,a4,16
1c008a06:	07c1                	addi	a5,a5,16
1c008a08:	100007b7          	lui	a5,0x10000
1c008a0c:	1c015737          	lui	a4,0x1c015
1c008a10:	4a478993          	addi	s3,a5,1188 # 100004a4 <l1_ker>
1c008a14:	6cc70713          	addi	a4,a4,1740 # 1c0156cc <WEIGHTS>
1c008a18:	4a478793          	addi	a5,a5,1188
1c008a1c:	0906d0fb          	lp.setupi	x1,144,1c008a36 <net_step+0xc6>
1c008a20:	00072803          	lw	a6,0(a4)
1c008a24:	4348                	lw	a0,4(a4)
1c008a26:	470c                	lw	a1,8(a4)
1c008a28:	4750                	lw	a2,12(a4)
1c008a2a:	0107a023          	sw	a6,0(a5)
1c008a2e:	c3c8                	sw	a0,4(a5)
1c008a30:	c78c                	sw	a1,8(a5)
1c008a32:	c7d0                	sw	a2,12(a5)
1c008a34:	0741                	addi	a4,a4,16
1c008a36:	07c1                	addi	a5,a5,16
1c008a38:	10001b37          	lui	s6,0x10001
1c008a3c:	6605                	lui	a2,0x1
1c008a3e:	a2060613          	addi	a2,a2,-1504 # a20 <pos_soc_event_callback+0x5d8>
1c008a42:	4581                	li	a1,0
1c008a44:	da4b0513          	addi	a0,s6,-604 # 10000da4 <im2col_buffer>
1c008a48:	0b80a0ef          	jal	ra,1c012b00 <memset>
1c008a4c:	32090413          	addi	s0,s2,800
1c008a50:	12000613          	li	a2,288
1c008a54:	4581                	li	a1,0
1c008a56:	8522                	mv	a0,s0
1c008a58:	0a80a0ef          	jal	ra,1c012b00 <memset>
1c008a5c:	4695                	li	a3,5
1c008a5e:	44d92623          	sw	a3,1100(s2)
1c008a62:	44d92823          	sw	a3,1104(s2)
1c008a66:	f14024f3          	csrr	s1,mhartid
1c008a6a:	04800693          	li	a3,72
1c008a6e:	1c001537          	lui	a0,0x1c001
1c008a72:	478d                	li	a5,3
1c008a74:	4721                	li	a4,8
1c008a76:	0c800613          	li	a2,200
1c008a7a:	46d92023          	sw	a3,1120(s2)
1c008a7e:	a7450513          	addi	a0,a0,-1420 # 1c000a74 <__DTOR_END__+0x20c>
1c008a82:	24000693          	li	a3,576
1c008a86:	44e92a23          	sw	a4,1108(s2)
1c008a8a:	46f92223          	sw	a5,1124(s2)
1c008a8e:	46f92423          	sw	a5,1128(s2)
1c008a92:	46e92623          	sw	a4,1132(s2)
1c008a96:	46f92e23          	sw	a5,1148(s2)
1c008a9a:	48f92023          	sw	a5,1152(s2)
1c008a9e:	48e92223          	sw	a4,1156(s2)
1c008aa2:	45292023          	sw	s2,1088(s2)
1c008aa6:	44c92423          	sw	a2,1096(s2)
1c008aaa:	44892c23          	sw	s0,1112(s2)
1c008aae:	47392823          	sw	s3,1136(s2)
1c008ab2:	46d92c23          	sw	a3,1144(s2)
1c008ab6:	0920a0ef          	jal	ra,1c012b48 <puts>
1c008aba:	ca549733          	p.extractu	a4,s1,5,5
1c008abe:	47fd                	li	a5,31
1c008ac0:	38f70e63          	beq	a4,a5,1c008e5c <net_step+0x4ec>
1c008ac4:	102007b7          	lui	a5,0x10200
1c008ac8:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1fec34>
1c008acc:	0007a023          	sw	zero,0(a5)
1c008ad0:	4781                	li	a5,0
1c008ad2:	cc179073          	csrw	0xcc1,a5
1c008ad6:	ca5497b3          	p.extractu	a5,s1,5,5
1c008ada:	477d                	li	a4,31
1c008adc:	36e78763          	beq	a5,a4,1c008e4a <net_step+0x4da>
1c008ae0:	102007b7          	lui	a5,0x10200
1c008ae4:	4705                	li	a4,1
1c008ae6:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1fec34>
1c008aea:	02e7a023          	sw	a4,32(a5)
1c008aee:	4781                	li	a5,0
1c008af0:	79f79073          	csrw	pccr31,a5
1c008af4:	ca5497b3          	p.extractu	a5,s1,5,5
1c008af8:	477d                	li	a4,31
1c008afa:	32e78f63          	beq	a5,a4,1c008e38 <net_step+0x4c8>
1c008afe:	102007b7          	lui	a5,0x10200
1c008b02:	4705                	li	a4,1
1c008b04:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1fec34>
1c008b08:	00e7ac23          	sw	a4,24(a5)
1c008b0c:	478d                	li	a5,3
1c008b0e:	cc179073          	csrw	0xcc1,a5
1c008b12:	10000637          	lui	a2,0x10000
1c008b16:	100005b7          	lui	a1,0x10000
1c008b1a:	10000537          	lui	a0,0x10000
1c008b1e:	4785                	li	a5,1
1c008b20:	4705                	li	a4,1
1c008b22:	4881                	li	a7,0
1c008b24:	da4b0813          	addi	a6,s6,-604
1c008b28:	4681                	li	a3,0
1c008b2a:	47460613          	addi	a2,a2,1140 # 10000474 <layer1_out>
1c008b2e:	48c58593          	addi	a1,a1,1164 # 1000048c <layer1_wgt>
1c008b32:	45c50513          	addi	a0,a0,1116 # 1000045c <layer1_in>
1c008b36:	2fe090ef          	jal	ra,1c011e34 <pulp_conv2d_fp32_fw_cl>
1c008b3a:	ca5497b3          	p.extractu	a5,s1,5,5
1c008b3e:	477d                	li	a4,31
1c008b40:	2ce78b63          	beq	a5,a4,1c008e16 <net_step+0x4a6>
1c008b44:	102007b7          	lui	a5,0x10200
1c008b48:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1fec34>
1c008b4c:	0007a023          	sw	zero,0(a5)
1c008b50:	4781                	li	a5,0
1c008b52:	cc179073          	csrw	0xcc1,a5
1c008b56:	477d                	li	a4,31
1c008b58:	ca5497b3          	p.extractu	a5,s1,5,5
1c008b5c:	2ce78963          	beq	a5,a4,1c008e2e <net_step+0x4be>
1c008b60:	102007b7          	lui	a5,0x10200
1c008b64:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1fec34>
1c008b68:	0087a783          	lw	a5,8(a5)
1c008b6c:	1c001737          	lui	a4,0x1c001
1c008b70:	7a470993          	addi	s3,a4,1956 # 1c0017a4 <_cycles>
1c008b74:	00f9a023          	sw	a5,0(s3)
1c008b78:	781027f3          	csrr	a5,pccr1
1c008b7c:	1c001db7          	lui	s11,0x1c001
1c008b80:	7acd8d93          	addi	s11,s11,1964 # 1c0017ac <_instr>
1c008b84:	00fda023          	sw	a5,0(s11)
1c008b88:	780027f3          	csrr	a5,pccr0
1c008b8c:	1c001d37          	lui	s10,0x1c001
1c008b90:	7a0d0d13          	addi	s10,s10,1952 # 1c0017a0 <_edata>
1c008b94:	00fd2023          	sw	a5,0(s10)
1c008b98:	78c027f3          	csrr	a5,pccr12
1c008b9c:	1c001cb7          	lui	s9,0x1c001
1c008ba0:	7b0c8c93          	addi	s9,s9,1968 # 1c0017b0 <_ldext>
1c008ba4:	00fca023          	sw	a5,0(s9)
1c008ba8:	790027f3          	csrr	a5,pccr16
1c008bac:	1c001c37          	lui	s8,0x1c001
1c008bb0:	7b8c0c13          	addi	s8,s8,1976 # 1c0017b8 <_tcdmcont>
1c008bb4:	00fc2023          	sw	a5,0(s8)
1c008bb8:	782027f3          	csrr	a5,pccr2
1c008bbc:	1c001bb7          	lui	s7,0x1c001
1c008bc0:	7b4b8b93          	addi	s7,s7,1972 # 1c0017b4 <_ldstall>
1c008bc4:	00fba023          	sw	a5,0(s7)
1c008bc8:	784026f3          	csrr	a3,pccr4
1c008bcc:	1c0014b7          	lui	s1,0x1c001
1c008bd0:	1c001b37          	lui	s6,0x1c001
1c008bd4:	7bc48493          	addi	s1,s1,1980 # 1c0017bc <id>
1c008bd8:	7a8b0b13          	addi	s6,s6,1960 # 1c0017a8 <_imiss>
1c008bdc:	f14027f3          	csrr	a5,mhartid
1c008be0:	4529                	li	a0,10
1c008be2:	f457b7b3          	p.bclr	a5,a5,26,5
1c008be6:	c09c                	sw	a5,0(s1)
1c008be8:	00db2023          	sw	a3,0(s6)
1c008bec:	7cf090ef          	jal	ra,1c012bba <putchar>
1c008bf0:	408c                	lw	a1,0(s1)
1c008bf2:	0009a603          	lw	a2,0(s3)
1c008bf6:	1c001537          	lui	a0,0x1c001
1c008bfa:	a8450513          	addi	a0,a0,-1404 # 1c000a84 <__DTOR_END__+0x21c>
1c008bfe:	0180a0ef          	jal	ra,1c012c16 <printf>
1c008c02:	408c                	lw	a1,0(s1)
1c008c04:	000da603          	lw	a2,0(s11)
1c008c08:	1c001537          	lui	a0,0x1c001
1c008c0c:	a9850513          	addi	a0,a0,-1384 # 1c000a98 <__DTOR_END__+0x230>
1c008c10:	0060a0ef          	jal	ra,1c012c16 <printf>
1c008c14:	408c                	lw	a1,0(s1)
1c008c16:	000d2603          	lw	a2,0(s10)
1c008c1a:	1c001537          	lui	a0,0x1c001
1c008c1e:	aac50513          	addi	a0,a0,-1364 # 1c000aac <__DTOR_END__+0x244>
1c008c22:	7f5090ef          	jal	ra,1c012c16 <printf>
1c008c26:	408c                	lw	a1,0(s1)
1c008c28:	000ca603          	lw	a2,0(s9)
1c008c2c:	1c001537          	lui	a0,0x1c001
1c008c30:	ac850513          	addi	a0,a0,-1336 # 1c000ac8 <__DTOR_END__+0x260>
1c008c34:	7e3090ef          	jal	ra,1c012c16 <printf>
1c008c38:	408c                	lw	a1,0(s1)
1c008c3a:	000c2603          	lw	a2,0(s8)
1c008c3e:	1c001537          	lui	a0,0x1c001
1c008c42:	ae050513          	addi	a0,a0,-1312 # 1c000ae0 <__DTOR_END__+0x278>
1c008c46:	7d1090ef          	jal	ra,1c012c16 <printf>
1c008c4a:	408c                	lw	a1,0(s1)
1c008c4c:	000ba603          	lw	a2,0(s7)
1c008c50:	1c001537          	lui	a0,0x1c001
1c008c54:	af850513          	addi	a0,a0,-1288 # 1c000af8 <__DTOR_END__+0x290>
1c008c58:	7bf090ef          	jal	ra,1c012c16 <printf>
1c008c5c:	408c                	lw	a1,0(s1)
1c008c5e:	000b2603          	lw	a2,0(s6)
1c008c62:	1c001537          	lui	a0,0x1c001
1c008c66:	b1050513          	addi	a0,a0,-1264 # 1c000b10 <__DTOR_END__+0x2a8>
1c008c6a:	7ad090ef          	jal	ra,1c012c16 <printf>
1c008c6e:	1c001537          	lui	a0,0x1c001
1c008c72:	b2450513          	addi	a0,a0,-1244 # 1c000b24 <__DTOR_END__+0x2bc>
1c008c76:	6d3090ef          	jal	ra,1c012b48 <puts>
1c008c7a:	1c0017b7          	lui	a5,0x1c001
1c008c7e:	a287a883          	lw	a7,-1496(a5) # 1c000a28 <__DTOR_END__+0x1c0>
1c008c82:	00000513          	li	a0,0
1c008c86:	328a8a93          	addi	s5,s5,808
1c008c8a:	8e46                	mv	t3,a7
1c008c8c:	862a                	mv	a2,a0
1c008c8e:	84a2                	mv	s1,s0
1c008c90:	85d6                	mv	a1,s5
1c008c92:	8a22                	mv	s4,s0
1c008c94:	04800693          	li	a3,72
1c008c98:	004a270b          	p.lw	a4,4(s4!)
1c008c9c:	0045a78b          	p.lw	a5,4(a1!)
1c008ca0:	a0e79853          	flt.s	a6,a5,a4
1c008ca4:	08f777d3          	fsub.s	a5,a4,a5
1c008ca8:	10080663          	beqz	a6,1c008db4 <net_step+0x444>
1c008cac:	a0e61853          	flt.s	a6,a2,a4
1c008cb0:	10080263          	beqz	a6,1c008db4 <net_step+0x444>
1c008cb4:	a0f61853          	flt.s	a6,a2,a5
1c008cb8:	00081463          	bnez	a6,1c008cc0 <net_step+0x350>
1c008cbc:	20f797d3          	fneg.s	a5,a5a5
1c008cc0:	18e7f7d3          	fdiv.s	a5,a5,a4
1c008cc4:	19c7f7d3          	fdiv.s	a5,a5,t3
1c008cc8:	00f57553          	fadd.s	a0,a0,a5
1c008ccc:	16fd                	addi	a3,a3,-1
1c008cce:	f6e9                	bnez	a3,1c008c98 <net_step+0x328>
1c008cd0:	1c001b37          	lui	s6,0x1c001
1c008cd4:	3649                	jal	1c008856 <__extendsfdf2>
1c008cd6:	9d0b0b13          	addi	s6,s6,-1584 # 1c0009d0 <__DTOR_END__+0x168>
1c008cda:	000b2603          	lw	a2,0(s6)
1c008cde:	004b2683          	lw	a3,4(s6)
1c008ce2:	34f1                	jal	1c0087ae <__ledf2>
1c008ce4:	12054263          	bltz	a0,1c008e08 <net_step+0x498>
1c008ce8:	1c001537          	lui	a0,0x1c001
1c008cec:	b4850513          	addi	a0,a0,-1208 # 1c000b48 <__DTOR_END__+0x2e0>
1c008cf0:	659090ef          	jal	ra,1c012b48 <puts>
1c008cf4:	004b2b83          	lw	s7,4(s6)
1c008cf8:	00000d13          	li	s10,0
1c008cfc:	000b2b03          	lw	s6,0(s6)
1c008d00:	89a2                	mv	s3,s0
1c008d02:	4d81                	li	s11,0
1c008d04:	4c81                	li	s9,0
1c008d06:	1c001437          	lui	s0,0x1c001
1c008d0a:	0049a90b          	p.lw	s2,4(s3!)
1c008d0e:	004aac0b          	p.lw	s8,4(s5!)
1c008d12:	09897553          	fsub.s	a0,s2,s8
1c008d16:	a0ad16d3          	flt.s	a3,s10,a0
1c008d1a:	e299                	bnez	a3,1c008d20 <net_step+0x3b0>
1c008d1c:	20a51553          	fneg.s	a0,a0a0
1c008d20:	3e1d                	jal	1c008856 <__extendsfdf2>
1c008d22:	865a                	mv	a2,s6
1c008d24:	86de                	mv	a3,s7
1c008d26:	32c5                	jal	1c008706 <__gedf2>
1c008d28:	02a05763          	blez	a0,1c008d56 <net_step+0x3e6>
1c008d2c:	0c0d8663          	beqz	s11,1c008df8 <net_step+0x488>
1c008d30:	854a                	mv	a0,s2
1c008d32:	4d85                	li	s11,1
1c008d34:	360d                	jal	1c008856 <__extendsfdf2>
1c008d36:	882a                	mv	a6,a0
1c008d38:	8562                	mv	a0,s8
1c008d3a:	ce42                	sw	a6,28(sp)
1c008d3c:	cc2e                	sw	a1,24(sp)
1c008d3e:	3e21                	jal	1c008856 <__extendsfdf2>
1c008d40:	4872                	lw	a6,28(sp)
1c008d42:	48e2                	lw	a7,24(sp)
1c008d44:	862a                	mv	a2,a0
1c008d46:	86ae                	mv	a3,a1
1c008d48:	c04a                	sw	s2,0(sp)
1c008d4a:	85e6                	mv	a1,s9
1c008d4c:	8762                	mv	a4,s8
1c008d4e:	9d840513          	addi	a0,s0,-1576 # 1c0009d8 <__DTOR_END__+0x170>
1c008d52:	6c5090ef          	jal	ra,1c012c16 <printf>
1c008d56:	0c85                	addi	s9,s9,1
1c008d58:	fb3a19e3          	bne	s4,s3,1c008d0a <net_step+0x39a>
1c008d5c:	100006b7          	lui	a3,0x10000
1c008d60:	10000637          	lui	a2,0x10000
1c008d64:	100005b7          	lui	a1,0x10000
1c008d68:	1c001537          	lui	a0,0x1c001
1c008d6c:	47468693          	addi	a3,a3,1140 # 10000474 <layer1_out>
1c008d70:	48c60613          	addi	a2,a2,1164 # 1000048c <layer1_wgt>
1c008d74:	45c58593          	addi	a1,a1,1116 # 1000045c <layer1_in>
1c008d78:	b6050513          	addi	a0,a0,-1184 # 1c000b60 <__DTOR_END__+0x2f8>
1c008d7c:	69b090ef          	jal	ra,1c012c16 <printf>
1c008d80:	4401                	li	s0,0
1c008d82:	1c001a37          	lui	s4,0x1c001
1c008d86:	498d                	li	s3,3
1c008d88:	04800913          	li	s2,72
1c008d8c:	a829                	j	1c008da6 <net_step+0x436>
1c008d8e:	0044a50b          	p.lw	a0,4(s1!)
1c008d92:	0405                	addi	s0,s0,1
1c008d94:	34c9                	jal	1c008856 <__extendsfdf2>
1c008d96:	862a                	mv	a2,a0
1c008d98:	86ae                	mv	a3,a1
1c008d9a:	b80a0513          	addi	a0,s4,-1152 # 1c000b80 <__DTOR_END__+0x318>
1c008d9e:	679090ef          	jal	ra,1c012c16 <printf>
1c008da2:	03240c63          	beq	s0,s2,1c008dda <net_step+0x46a>
1c008da6:	033467b3          	rem	a5,s0,s3
1c008daa:	f3f5                	bnez	a5,1c008d8e <net_step+0x41e>
1c008dac:	4529                	li	a0,10
1c008dae:	60d090ef          	jal	ra,1c012bba <putchar>
1c008db2:	bff1                	j	1c008d8e <net_step+0x41e>
1c008db4:	a0f61853          	flt.s	a6,a2,a5
1c008db8:	00081463          	bnez	a6,1c008dc0 <net_step+0x450>
1c008dbc:	20f797d3          	fneg.s	a5,a5a5
1c008dc0:	a0e61853          	flt.s	a6,a2,a4
1c008dc4:	00081463          	bnez	a6,1c008dcc <net_step+0x45c>
1c008dc8:	20e71753          	fneg.s	a4,a4a4
1c008dcc:	18e7f7d3          	fdiv.s	a5,a5,a4
1c008dd0:	1917f7d3          	fdiv.s	a5,a5,a7
1c008dd4:	00f57553          	fadd.s	a0,a0,a5
1c008dd8:	bdd5                	j	1c008ccc <net_step+0x35c>
1c008dda:	40f6                	lw	ra,92(sp)
1c008ddc:	4466                	lw	s0,88(sp)
1c008dde:	44d6                	lw	s1,84(sp)
1c008de0:	4946                	lw	s2,80(sp)
1c008de2:	49b6                	lw	s3,76(sp)
1c008de4:	4a26                	lw	s4,72(sp)
1c008de6:	4a96                	lw	s5,68(sp)
1c008de8:	4b06                	lw	s6,64(sp)
1c008dea:	5bf2                	lw	s7,60(sp)
1c008dec:	5c62                	lw	s8,56(sp)
1c008dee:	5cd2                	lw	s9,52(sp)
1c008df0:	5d42                	lw	s10,48(sp)
1c008df2:	5db2                	lw	s11,44(sp)
1c008df4:	6125                	addi	sp,sp,96
1c008df6:	8082                	ret
1c008df8:	4529                	li	a0,10
1c008dfa:	5c1090ef          	jal	ra,1c012bba <putchar>
1c008dfe:	ffcaac03          	lw	s8,-4(s5)
1c008e02:	ffc9a903          	lw	s2,-4(s3)
1c008e06:	b72d                	j	1c008d30 <net_step+0x3c0>
1c008e08:	1c001537          	lui	a0,0x1c001
1c008e0c:	b3450513          	addi	a0,a0,-1228 # 1c000b34 <__DTOR_END__+0x2cc>
1c008e10:	539090ef          	jal	ra,1c012b48 <puts>
1c008e14:	b5c5                	j	1c008cf4 <net_step+0x384>
1c008e16:	1a10b7b7          	lui	a5,0x1a10b
1c008e1a:	0007a023          	sw	zero,0(a5) # 1a10b000 <__l1_end+0xa109834>
1c008e1e:	4781                	li	a5,0
1c008e20:	cc179073          	csrw	0xcc1,a5
1c008e24:	477d                	li	a4,31
1c008e26:	ca5497b3          	p.extractu	a5,s1,5,5
1c008e2a:	d2e79be3          	bne	a5,a4,1c008b60 <net_step+0x1f0>
1c008e2e:	1a10b7b7          	lui	a5,0x1a10b
1c008e32:	0087a783          	lw	a5,8(a5) # 1a10b008 <__l1_end+0xa10983c>
1c008e36:	bb1d                	j	1c008b6c <net_step+0x1fc>
1c008e38:	4785                	li	a5,1
1c008e3a:	1a10b737          	lui	a4,0x1a10b
1c008e3e:	00f72c23          	sw	a5,24(a4) # 1a10b018 <__l1_end+0xa10984c>
1c008e42:	478d                	li	a5,3
1c008e44:	cc179073          	csrw	0xcc1,a5
1c008e48:	b1e9                	j	1c008b12 <net_step+0x1a2>
1c008e4a:	4785                	li	a5,1
1c008e4c:	1a10b737          	lui	a4,0x1a10b
1c008e50:	02f72023          	sw	a5,32(a4) # 1a10b020 <__l1_end+0xa109854>
1c008e54:	4781                	li	a5,0
1c008e56:	79f79073          	csrw	pccr31,a5
1c008e5a:	b969                	j	1c008af4 <net_step+0x184>
1c008e5c:	1a10b7b7          	lui	a5,0x1a10b
1c008e60:	0007a023          	sw	zero,0(a5) # 1a10b000 <__l1_end+0xa109834>
1c008e64:	4781                	li	a5,0
1c008e66:	cc179073          	csrw	0xcc1,a5
1c008e6a:	b1b5                	j	1c008ad6 <net_step+0x166>
1c008e6c:	0000                	unimp
	...

1c008e70 <mm>:
1c008e70:	00c52303          	lw	t1,12(a0)
1c008e74:	f14028f3          	csrr	a7,mhartid
1c008e78:	f458b8b3          	p.bclr	a7,a7,26,5
1c008e7c:	00730793          	addi	a5,t1,7
1c008e80:	41f7d693          	srai	a3,a5,0x1f
1c008e84:	f836b6b3          	p.bclr	a3,a3,28,3
1c008e88:	06f6a6db          	p.addn	a3,a3,a5,3
1c008e8c:	02d888b3          	mul	a7,a7,a3
1c008e90:	4d0c                	lw	a1,24(a0)
1c008e92:	1141                	addi	sp,sp,-16
1c008e94:	c426                	sw	s1,8(sp)
1c008e96:	c622                	sw	s0,12(sp)
1c008e98:	c24a                	sw	s2,4(sp)
1c008e9a:	4104                	lw	s1,0(a0)
1c008e9c:	00452e03          	lw	t3,4(a0)
1c008ea0:	451c                	lw	a5,8(a0)
1c008ea2:	4910                	lw	a2,16(a0)
1c008ea4:	96c6                	add	a3,a3,a7
1c008ea6:	4958                	lw	a4,20(a0)
1c008ea8:	0466c333          	p.min	t1,a3,t1
1c008eac:	e9c1                	bnez	a1,1c008f3c <mm+0xcc>
1c008eae:	12172063          	p.beqimm	a4,1,1c008fce <mm+0x15e>
1c008eb2:	08e05063          	blez	a4,1c008f32 <mm+0xc2>
1c008eb6:	0668de63          	ble	t1,a7,1c008f32 <mm+0xc2>
1c008eba:	06c05c63          	blez	a2,1c008f32 <mm+0xc2>
1c008ebe:	031706b3          	mul	a3,a4,a7
1c008ec2:	00261413          	slli	s0,a2,0x2
1c008ec6:	00271393          	slli	t2,a4,0x2
1c008eca:	008e0933          	add	s2,t3,s0
1c008ece:	03160633          	mul	a2,a2,a7
1c008ed2:	00e682b3          	add	t0,a3,a4
1c008ed6:	028a                	slli	t0,t0,0x2
1c008ed8:	068a                	slli	a3,a3,0x2
1c008eda:	00d48f33          	add	t5,s1,a3
1c008ede:	92a6                	add	t0,t0,s1
1c008ee0:	00261493          	slli	s1,a2,0x2
1c008ee4:	94be                	add	s1,s1,a5
1c008ee6:	41c90833          	sub	a6,s2,t3
1c008eea:	1871                	addi	a6,a6,-4
1c008eec:	00285813          	srli	a6,a6,0x2
1c008ef0:	8ef2                	mv	t4,t3
1c008ef2:	8fa6                	mv	t6,s1
1c008ef4:	0805                	addi	a6,a6,1
1c008ef6:	0001                	nop
1c008ef8:	0168407b          	lp.setup	x0,a6,1c008f24 <mm+0xb4>
1c008efc:	41e287b3          	sub	a5,t0,t5
1c008f00:	17f1                	addi	a5,a5,-4
1c008f02:	00000713          	li	a4,0
1c008f06:	8389                	srli	a5,a5,0x2
1c008f08:	8676                	mv	a2,t4
1c008f0a:	86fa                	mv	a3,t5
1c008f0c:	0785                	addi	a5,a5,1
1c008f0e:	0001                	nop
1c008f10:	0067c0fb          	lp.setup	x1,a5,1c008f1c <mm+0xac>
1c008f14:	0046a50b          	p.lw	a0,4(a3!)
1c008f18:	2086758b          	p.lw	a1,s0(a2!)
1c008f1c:	70b57743          	fmadd.s	a4,a0,a1,a4
1c008f20:	00efa22b          	p.sw	a4,4(t6!)
1c008f24:	0e91                	addi	t4,t4,4
1c008f26:	0885                	addi	a7,a7,1
1c008f28:	9f1e                	add	t5,t5,t2
1c008f2a:	929e                	add	t0,t0,t2
1c008f2c:	94a2                	add	s1,s1,s0
1c008f2e:	fb131ce3          	bne	t1,a7,1c008ee6 <mm+0x76>
1c008f32:	4432                	lw	s0,12(sp)
1c008f34:	44a2                	lw	s1,8(sp)
1c008f36:	4912                	lw	s2,4(sp)
1c008f38:	0141                	addi	sp,sp,16
1c008f3a:	8082                	ret
1c008f3c:	0e172c63          	p.beqimm	a4,1,1c009034 <mm+0x1c4>
1c008f40:	fee059e3          	blez	a4,1c008f32 <mm+0xc2>
1c008f44:	fe68d7e3          	ble	t1,a7,1c008f32 <mm+0xc2>
1c008f48:	fec055e3          	blez	a2,1c008f32 <mm+0xc2>
1c008f4c:	031706b3          	mul	a3,a4,a7
1c008f50:	40c003b3          	neg	t2,a2
1c008f54:	00271293          	slli	t0,a4,0x2
1c008f58:	00239413          	slli	s0,t2,0x2
1c008f5c:	038e                	slli	t2,t2,0x3
1c008f5e:	43160633          	p.mac	a2,a2,a7
1c008f62:	9736                	add	a4,a4,a3
1c008f64:	070a                	slli	a4,a4,0x2
1c008f66:	068a                	slli	a3,a3,0x2
1c008f68:	00d48f33          	add	t5,s1,a3
1c008f6c:	94ba                	add	s1,s1,a4
1c008f6e:	00261713          	slli	a4,a2,0x2
1c008f72:	97ba                	add	a5,a5,a4
1c008f74:	00878933          	add	s2,a5,s0
1c008f78:	41278833          	sub	a6,a5,s2
1c008f7c:	1871                	addi	a6,a6,-4
1c008f7e:	00285813          	srli	a6,a6,0x2
1c008f82:	8ef2                	mv	t4,t3
1c008f84:	8fca                	mv	t6,s2
1c008f86:	0805                	addi	a6,a6,1
1c008f88:	0168407b          	lp.setup	x0,a6,1c008fb4 <mm+0x144>
1c008f8c:	41e487b3          	sub	a5,s1,t5
1c008f90:	17f1                	addi	a5,a5,-4
1c008f92:	00000713          	li	a4,0
1c008f96:	8389                	srli	a5,a5,0x2
1c008f98:	8676                	mv	a2,t4
1c008f9a:	86fa                	mv	a3,t5
1c008f9c:	0785                	addi	a5,a5,1
1c008f9e:	0001                	nop
1c008fa0:	0067c0fb          	lp.setup	x1,a5,1c008fac <mm+0x13c>
1c008fa4:	0046a50b          	p.lw	a0,4(a3!)
1c008fa8:	0046258b          	p.lw	a1,4(a2!)
1c008fac:	70b57743          	fmadd.s	a4,a0,a1,a4
1c008fb0:	00efa22b          	p.sw	a4,4(t6!)
1c008fb4:	9e96                	add	t4,t4,t0
1c008fb6:	0885                	addi	a7,a7,1
1c008fb8:	407907b3          	sub	a5,s2,t2
1c008fbc:	9f16                	add	t5,t5,t0
1c008fbe:	9496                	add	s1,s1,t0
1c008fc0:	fb131ae3          	bne	t1,a7,1c008f74 <mm+0x104>
1c008fc4:	4432                	lw	s0,12(sp)
1c008fc6:	44a2                	lw	s1,8(sp)
1c008fc8:	4912                	lw	s2,4(sp)
1c008fca:	0141                	addi	sp,sp,16
1c008fcc:	8082                	ret
1c008fce:	f668d2e3          	ble	t1,a7,1c008f32 <mm+0xc2>
1c008fd2:	f6c050e3          	blez	a2,1c008f32 <mm+0xc2>
1c008fd6:	03160733          	mul	a4,a2,a7
1c008fda:	00289513          	slli	a0,a7,0x2
1c008fde:	00231813          	slli	a6,t1,0x2
1c008fe2:	40a80833          	sub	a6,a6,a0
1c008fe6:	1871                	addi	a6,a6,-4
1c008fe8:	00261313          	slli	t1,a2,0x2
1c008fec:	00285813          	srli	a6,a6,0x2
1c008ff0:	9526                	add	a0,a0,s1
1c008ff2:	006e0eb3          	add	t4,t3,t1
1c008ff6:	0805                	addi	a6,a6,1
1c008ff8:	00271893          	slli	a7,a4,0x2
1c008ffc:	98be                	add	a7,a7,a5
1c008ffe:	0001                	nop
1c009000:	0148407b          	lp.setup	x0,a6,1c009028 <mm+0x1b8>
1c009004:	41ce87b3          	sub	a5,t4,t3
1c009008:	17f1                	addi	a5,a5,-4
1c00900a:	8389                	srli	a5,a5,0x2
1c00900c:	8646                	mv	a2,a7
1c00900e:	86f2                	mv	a3,t3
1c009010:	0785                	addi	a5,a5,1
1c009012:	0001                	nop
1c009014:	0077c0fb          	lp.setup	x1,a5,1c009022 <mm+0x1b2>
1c009018:	0046a58b          	p.lw	a1,4(a3!)
1c00901c:	4118                	lw	a4,0(a0)
1c00901e:	10b77753          	fmul.s	a4,a4,a1
1c009022:	00e6222b          	p.sw	a4,4(a2!)
1c009026:	989a                	add	a7,a7,t1
1c009028:	0511                	addi	a0,a0,4
1c00902a:	4432                	lw	s0,12(sp)
1c00902c:	44a2                	lw	s1,8(sp)
1c00902e:	4912                	lw	s2,4(sp)
1c009030:	0141                	addi	sp,sp,16
1c009032:	8082                	ret
1c009034:	ee68dfe3          	ble	t1,a7,1c008f32 <mm+0xc2>
1c009038:	eec05de3          	blez	a2,1c008f32 <mm+0xc2>
1c00903c:	03160733          	mul	a4,a2,a7
1c009040:	00289513          	slli	a0,a7,0x2
1c009044:	00231813          	slli	a6,t1,0x2
1c009048:	40a80833          	sub	a6,a6,a0
1c00904c:	1871                	addi	a6,a6,-4
1c00904e:	00261313          	slli	t1,a2,0x2
1c009052:	00285813          	srli	a6,a6,0x2
1c009056:	9526                	add	a0,a0,s1
1c009058:	006e0eb3          	add	t4,t3,t1
1c00905c:	0805                	addi	a6,a6,1
1c00905e:	00271893          	slli	a7,a4,0x2
1c009062:	98be                	add	a7,a7,a5
1c009064:	0148407b          	lp.setup	x0,a6,1c00908c <mm+0x21c>
1c009068:	41ce87b3          	sub	a5,t4,t3
1c00906c:	17f1                	addi	a5,a5,-4
1c00906e:	8389                	srli	a5,a5,0x2
1c009070:	8646                	mv	a2,a7
1c009072:	86f2                	mv	a3,t3
1c009074:	0785                	addi	a5,a5,1
1c009076:	0001                	nop
1c009078:	0077c0fb          	lp.setup	x1,a5,1c009086 <mm+0x216>
1c00907c:	0046a58b          	p.lw	a1,4(a3!)
1c009080:	4118                	lw	a4,0(a0)
1c009082:	10b77753          	fmul.s	a4,a4,a1
1c009086:	00e6222b          	p.sw	a4,4(a2!)
1c00908a:	989a                	add	a7,a7,t1
1c00908c:	0511                	addi	a0,a0,4
1c00908e:	4432                	lw	s0,12(sp)
1c009090:	44a2                	lw	s1,8(sp)
1c009092:	4912                	lw	s2,4(sp)
1c009094:	0141                	addi	sp,sp,16
1c009096:	8082                	ret

1c009098 <mm_M>:
1c009098:	01052e03          	lw	t3,16(a0)
1c00909c:	f1402373          	csrr	t1,mhartid
1c0090a0:	f4533333          	p.bclr	t1,t1,26,5
1c0090a4:	007e0713          	addi	a4,t3,7
1c0090a8:	41f75793          	srai	a5,a4,0x1f
1c0090ac:	f837b7b3          	p.bclr	a5,a5,28,3
1c0090b0:	06e7a7db          	p.addn	a5,a5,a4,3
1c0090b4:	02f30333          	mul	t1,t1,a5
1c0090b8:	01852383          	lw	t2,24(a0)
1c0090bc:	1101                	addi	sp,sp,-32
1c0090be:	cc26                	sw	s1,24(sp)
1c0090c0:	ce22                	sw	s0,28(sp)
1c0090c2:	ca4a                	sw	s2,20(sp)
1c0090c4:	c84e                	sw	s3,16(sp)
1c0090c6:	c652                	sw	s4,12(sp)
1c0090c8:	c456                	sw	s5,8(sp)
1c0090ca:	c25a                	sw	s6,4(sp)
1c0090cc:	979a                	add	a5,a5,t1
1c0090ce:	00052f83          	lw	t6,0(a0)
1c0090d2:	4144                	lw	s1,4(a0)
1c0090d4:	00852f03          	lw	t5,8(a0)
1c0090d8:	00c52e83          	lw	t4,12(a0)
1c0090dc:	01452883          	lw	a7,20(a0)
1c0090e0:	05c7c7b3          	p.min	a5,a5,t3
1c0090e4:	08038963          	beqz	t2,1c009176 <mm_M+0xde>
1c0090e8:	07d05e63          	blez	t4,1c009164 <mm_M+0xcc>
1c0090ec:	06f35c63          	ble	a5,t1,1c009164 <mm_M+0xcc>
1c0090f0:	00289393          	slli	t2,a7,0x2
1c0090f4:	427304b3          	p.mac	s1,t1,t2
1c0090f8:	078a                	slli	a5,a5,0x2
1c0090fa:	002e1a93          	slli	s5,t3,0x2
1c0090fe:	00ff0433          	add	s0,t5,a5
1c009102:	007f8933          	add	s2,t6,t2
1c009106:	4a01                	li	s4,0
1c009108:	4981                	li	s3,0
1c00910a:	006a02b3          	add	t0,s4,t1
1c00910e:	028a                	slli	t0,t0,0x2
1c009110:	92fa                	add	t0,t0,t5
1c009112:	40540833          	sub	a6,s0,t0
1c009116:	1871                	addi	a6,a6,-4
1c009118:	00285813          	srli	a6,a6,0x2
1c00911c:	8b26                	mv	s6,s1
1c00911e:	0805                	addi	a6,a6,1
1c009120:	01a8407b          	lp.setup	x0,a6,1c009154 <mm_M+0xbc>
1c009124:	00000713          	li	a4,0
1c009128:	03105463          	blez	a7,1c009150 <mm_M+0xb8>
1c00912c:	41f907b3          	sub	a5,s2,t6
1c009130:	17f1                	addi	a5,a5,-4
1c009132:	00000713          	li	a4,0
1c009136:	8389                	srli	a5,a5,0x2
1c009138:	865a                	mv	a2,s6
1c00913a:	86fe                	mv	a3,t6
1c00913c:	0785                	addi	a5,a5,1
1c00913e:	0001                	nop
1c009140:	0067c0fb          	lp.setup	x1,a5,1c00914c <mm_M+0xb4>
1c009144:	0046a50b          	p.lw	a0,4(a3!)
1c009148:	0046258b          	p.lw	a1,4(a2!)
1c00914c:	70b57743          	fmadd.s	a4,a0,a1,a4
1c009150:	00e2a22b          	p.sw	a4,4(t0!)
1c009154:	9b1e                	add	s6,s6,t2
1c009156:	0985                	addi	s3,s3,1
1c009158:	9a72                	add	s4,s4,t3
1c00915a:	9456                	add	s0,s0,s5
1c00915c:	9f9e                	add	t6,t6,t2
1c00915e:	991e                	add	s2,s2,t2
1c009160:	fb3e95e3          	bne	t4,s3,1c00910a <mm_M+0x72>
1c009164:	4472                	lw	s0,28(sp)
1c009166:	44e2                	lw	s1,24(sp)
1c009168:	4952                	lw	s2,20(sp)
1c00916a:	49c2                	lw	s3,16(sp)
1c00916c:	4a32                	lw	s4,12(sp)
1c00916e:	4aa2                	lw	s5,8(sp)
1c009170:	4b12                	lw	s6,4(sp)
1c009172:	6105                	addi	sp,sp,32
1c009174:	8082                	ret
1c009176:	ffd057e3          	blez	t4,1c009164 <mm_M+0xcc>
1c00917a:	fef355e3          	ble	a5,t1,1c009164 <mm_M+0xcc>
1c00917e:	00289993          	slli	s3,a7,0x2
1c009182:	00231913          	slli	s2,t1,0x2
1c009186:	078a                	slli	a5,a5,0x2
1c009188:	9926                	add	s2,s2,s1
1c00918a:	013f82b3          	add	t0,t6,s3
1c00918e:	94be                	add	s1,s1,a5
1c009190:	002e1a13          	slli	s4,t3,0x2
1c009194:	4401                	li	s0,0
1c009196:	41248833          	sub	a6,s1,s2
1c00919a:	00640ab3          	add	s5,s0,t1
1c00919e:	1871                	addi	a6,a6,-4
1c0091a0:	0a8a                	slli	s5,s5,0x2
1c0091a2:	00285813          	srli	a6,a6,0x2
1c0091a6:	9afa                	add	s5,s5,t5
1c0091a8:	8b4a                	mv	s6,s2
1c0091aa:	0805                	addi	a6,a6,1
1c0091ac:	01a8407b          	lp.setup	x0,a6,1c0091e0 <mm_M+0x148>
1c0091b0:	00000713          	li	a4,0
1c0091b4:	03105463          	blez	a7,1c0091dc <mm_M+0x144>
1c0091b8:	41f287b3          	sub	a5,t0,t6
1c0091bc:	17f1                	addi	a5,a5,-4
1c0091be:	00000713          	li	a4,0
1c0091c2:	8389                	srli	a5,a5,0x2
1c0091c4:	865a                	mv	a2,s6
1c0091c6:	86fe                	mv	a3,t6
1c0091c8:	0785                	addi	a5,a5,1
1c0091ca:	0001                	nop
1c0091cc:	0067c0fb          	lp.setup	x1,a5,1c0091d8 <mm_M+0x140>
1c0091d0:	0046a50b          	p.lw	a0,4(a3!)
1c0091d4:	2146758b          	p.lw	a1,s4(a2!)
1c0091d8:	70b57743          	fmadd.s	a4,a0,a1,a4
1c0091dc:	00eaa22b          	p.sw	a4,4(s5!)
1c0091e0:	0b11                	addi	s6,s6,4
1c0091e2:	0385                	addi	t2,t2,1
1c0091e4:	9472                	add	s0,s0,t3
1c0091e6:	9fce                	add	t6,t6,s3
1c0091e8:	92ce                	add	t0,t0,s3
1c0091ea:	fa7e96e3          	bne	t4,t2,1c009196 <mm_M+0xfe>
1c0091ee:	bf9d                	j	1c009164 <mm_M+0xcc>

1c0091f0 <mm_dw>:
1c0091f0:	4558                	lw	a4,12(a0)
1c0091f2:	1101                	addi	sp,sp,-32
1c0091f4:	c84e                	sw	s3,16(sp)
1c0091f6:	00770693          	addi	a3,a4,7
1c0091fa:	41f6d793          	srai	a5,a3,0x1f
1c0091fe:	f837b7b3          	p.bclr	a5,a5,28,3
1c009202:	06d7a7db          	p.addn	a5,a5,a3,3
1c009206:	f14029f3          	csrr	s3,mhartid
1c00920a:	f459b9b3          	p.bclr	s3,s3,26,5
1c00920e:	02f989b3          	mul	s3,s3,a5
1c009212:	c652                	sw	s4,12(sp)
1c009214:	01052a03          	lw	s4,16(a0)
1c009218:	ce22                	sw	s0,28(sp)
1c00921a:	cc26                	sw	s1,24(sp)
1c00921c:	ca4a                	sw	s2,20(sp)
1c00921e:	c456                	sw	s5,8(sp)
1c009220:	4110                	lw	a2,0(a0)
1c009222:	4154                	lw	a3,4(a0)
1c009224:	00852283          	lw	t0,8(a0)
1c009228:	97ce                	add	a5,a5,s3
1c00922a:	01852e83          	lw	t4,24(a0)
1c00922e:	04e7c433          	p.min	s0,a5,a4
1c009232:	09405363          	blez	s4,1c0092b8 <mm_dw+0xc8>
1c009236:	03d70733          	mul	a4,a4,t4
1c00923a:	0689df63          	ble	s0,s3,1c0092b8 <mm_dw+0xc8>
1c00923e:	8976                	mv	s2,t4
1c009240:	002e9f13          	slli	t5,t4,0x2
1c009244:	41e686b3          	sub	a3,a3,t5
1c009248:	41d00fb3          	neg	t6,t4
1c00924c:	00271493          	slli	s1,a4,0x2
1c009250:	40c68e33          	sub	t3,a3,a2
1c009254:	002a1a93          	slli	s5,s4,0x2
1c009258:	0f8a                	slli	t6,t6,0x2
1c00925a:	033a07b3          	mul	a5,s4,s3
1c00925e:	4381                	li	t2,0
1c009260:	43d98933          	p.mac	s2,s3,t4
1c009264:	078a                	slli	a5,a5,0x2
1c009266:	92be                	add	t0,t0,a5
1c009268:	090a                	slli	s2,s2,0x2
1c00926a:	9932                	add	s2,s2,a2
1c00926c:	884a                	mv	a6,s2
1c00926e:	8316                	mv	t1,t0
1c009270:	413408b3          	sub	a7,s0,s3
1c009274:	01c8c07b          	lp.setup	x0,a7,1c0092ac <mm_dw+0xbc>
1c009278:	00000713          	li	a4,0
1c00927c:	03d05663          	blez	t4,1c0092a8 <mm_dw+0xb8>
1c009280:	010f86b3          	add	a3,t6,a6
1c009284:	40d807b3          	sub	a5,a6,a3
1c009288:	17f1                	addi	a5,a5,-4
1c00928a:	00000713          	li	a4,0
1c00928e:	8389                	srli	a5,a5,0x2
1c009290:	01c80633          	add	a2,a6,t3
1c009294:	0785                	addi	a5,a5,1
1c009296:	0001                	nop
1c009298:	0067c0fb          	lp.setup	x1,a5,1c0092a4 <mm_dw+0xb4>
1c00929c:	0046a50b          	p.lw	a0,4(a3!)
1c0092a0:	0046258b          	p.lw	a1,4(a2!)
1c0092a4:	70b57743          	fmadd.s	a4,a0,a1,a4
1c0092a8:	00e36aab          	p.sw	a4,s5(t1!)
1c0092ac:	987a                	add	a6,a6,t5
1c0092ae:	0385                	addi	t2,t2,1
1c0092b0:	0291                	addi	t0,t0,4
1c0092b2:	9e26                	add	t3,t3,s1
1c0092b4:	fa7a1ce3          	bne	s4,t2,1c00926c <mm_dw+0x7c>
1c0092b8:	4472                	lw	s0,28(sp)
1c0092ba:	44e2                	lw	s1,24(sp)
1c0092bc:	4952                	lw	s2,20(sp)
1c0092be:	49c2                	lw	s3,16(sp)
1c0092c0:	4a32                	lw	s4,12(sp)
1c0092c2:	4aa2                	lw	s5,8(sp)
1c0092c4:	6105                	addi	sp,sp,32
1c0092c6:	8082                	ret
1c0092c8:	0000                	unimp
	...

1c0092cc <mm_dw_in_grad>:
1c0092cc:	7179                	addi	sp,sp,-48
1c0092ce:	d426                	sw	s1,40(sp)
1c0092d0:	01852883          	lw	a7,24(a0)
1c0092d4:	4944                	lw	s1,20(a0)
1c0092d6:	f14023f3          	csrr	t2,mhartid
1c0092da:	f453b3b3          	p.bclr	t2,t2,26,5
1c0092de:	0314c2b3          	div	t0,s1,a7
1c0092e2:	ca5a                	sw	s6,20(sp)
1c0092e4:	00c52b03          	lw	s6,12(a0)
1c0092e8:	d622                	sw	s0,44(sp)
1c0092ea:	ce52                	sw	s4,28(sp)
1c0092ec:	cc56                	sw	s5,24(sp)
1c0092ee:	d24a                	sw	s2,36(sp)
1c0092f0:	d04e                	sw	s3,32(sp)
1c0092f2:	c85e                	sw	s7,16(sp)
1c0092f4:	c662                	sw	s8,12(sp)
1c0092f6:	c466                	sw	s9,8(sp)
1c0092f8:	4118                	lw	a4,0(a0)
1c0092fa:	00452a83          	lw	s5,4(a0)
1c0092fe:	00852a03          	lw	s4,8(a0)
1c009302:	4900                	lw	s0,16(a0)
1c009304:	00728693          	addi	a3,t0,7
1c009308:	41f6d793          	srai	a5,a3,0x1f
1c00930c:	f837b7b3          	p.bclr	a5,a5,28,3
1c009310:	06d7a7db          	p.addn	a5,a5,a3,3
1c009314:	02f383b3          	mul	t2,t2,a5
1c009318:	979e                	add	a5,a5,t2
1c00931a:	0457c2b3          	p.min	t0,a5,t0
1c00931e:	09605663          	blez	s6,1c0093aa <mm_dw_in_grad+0xde>
1c009322:	08805463          	blez	s0,1c0093aa <mm_dw_in_grad+0xde>
1c009326:	0853d263          	ble	t0,t2,1c0093aa <mm_dw_in_grad+0xde>
1c00932a:	027407b3          	mul	a5,s0,t2
1c00932e:	00289313          	slli	t1,a7,0x2
1c009332:	ffc70913          	addi	s2,a4,-4
1c009336:	048a                	slli	s1,s1,0x2
1c009338:	00241b93          	slli	s7,s0,0x2
1c00933c:	4981                	li	s3,0
1c00933e:	40ea8ab3          	sub	s5,s5,a4
1c009342:	42638933          	p.mac	s2,t2,t1
1c009346:	078a                	slli	a5,a5,0x2
1c009348:	9a3e                	add	s4,s4,a5
1c00934a:	8fd6                	mv	t6,s5
1c00934c:	8f52                	mv	t5,s4
1c00934e:	4e81                	li	t4,0
1c009350:	874a                	mv	a4,s2
1c009352:	887a                	mv	a6,t5
1c009354:	004f8e13          	addi	t3,t6,4
1c009358:	40728533          	sub	a0,t0,t2
1c00935c:	01e5407b          	lp.setup	x0,a0,1c009398 <mm_dw_in_grad+0xcc>
1c009360:	00000693          	li	a3,0
1c009364:	00e305b3          	add	a1,t1,a4
1c009368:	03105663          	blez	a7,1c009394 <mm_dw_in_grad+0xc8>
1c00936c:	00e305b3          	add	a1,t1,a4
1c009370:	40e587b3          	sub	a5,a1,a4
1c009374:	17f1                	addi	a5,a5,-4
1c009376:	00000693          	li	a3,0
1c00937a:	8389                	srli	a5,a5,0x2
1c00937c:	862e                	mv	a2,a1
1c00937e:	9772                	add	a4,a4,t3
1c009380:	0785                	addi	a5,a5,1
1c009382:	0001                	nop
1c009384:	0067c0fb          	lp.setup	x1,a5,1c009390 <mm_dw_in_grad+0xc4>
1c009388:	ffc62c8b          	p.lw	s9,-4(a2!)
1c00938c:	00472c0b          	p.lw	s8,4(a4!)
1c009390:	698cf6c3          	fmadd.s	a3,s9,s8,a3
1c009394:	00d86bab          	p.sw	a3,s7(a6!)
1c009398:	872e                	mv	a4,a1
1c00939a:	0e85                	addi	t4,t4,1
1c00939c:	0f11                	addi	t5,t5,4
1c00939e:	9fa6                	add	t6,t6,s1
1c0093a0:	fbd418e3          	bne	s0,t4,1c009350 <mm_dw_in_grad+0x84>
1c0093a4:	0985                	addi	s3,s3,1
1c0093a6:	fb3b12e3          	bne	s6,s3,1c00934a <mm_dw_in_grad+0x7e>
1c0093aa:	5432                	lw	s0,44(sp)
1c0093ac:	54a2                	lw	s1,40(sp)
1c0093ae:	5912                	lw	s2,36(sp)
1c0093b0:	5982                	lw	s3,32(sp)
1c0093b2:	4a72                	lw	s4,28(sp)
1c0093b4:	4ae2                	lw	s5,24(sp)
1c0093b6:	4b52                	lw	s6,20(sp)
1c0093b8:	4bc2                	lw	s7,16(sp)
1c0093ba:	4c32                	lw	s8,12(sp)
1c0093bc:	4ca2                	lw	s9,8(sp)
1c0093be:	6145                	addi	sp,sp,48
1c0093c0:	8082                	ret
	...

1c0093c4 <mm_u2>:
1c0093c4:	7139                	addi	sp,sp,-64
1c0093c6:	de22                	sw	s0,60(sp)
1c0093c8:	4540                	lw	s0,12(a0)
1c0093ca:	d456                	sw	s5,40(sp)
1c0093cc:	f1402af3          	csrr	s5,mhartid
1c0093d0:	00740713          	addi	a4,s0,7
1c0093d4:	41f75793          	srai	a5,a4,0x1f
1c0093d8:	f837b7b3          	p.bclr	a5,a5,28,3
1c0093dc:	06e7a7db          	p.addn	a5,a5,a4,3
1c0093e0:	f45abab3          	p.bclr	s5,s5,26,5
1c0093e4:	02fa8ab3          	mul	s5,s5,a5
1c0093e8:	4114                	lw	a3,0(a0)
1c0093ea:	d652                	sw	s4,44(sp)
1c0093ec:	01452a03          	lw	s4,20(a0)
1c0093f0:	c036                	sw	a3,0(sp)
1c0093f2:	4514                	lw	a3,8(a0)
1c0093f4:	4d18                	lw	a4,24(a0)
1c0093f6:	d25a                	sw	s6,36(sp)
1c0093f8:	c236                	sw	a3,4(sp)
1c0093fa:	fc1a36b3          	p.bclr	a3,s4,30,1
1c0093fe:	cc66                	sw	s9,24(sp)
1c009400:	97d6                	add	a5,a5,s5
1c009402:	dc26                	sw	s1,56(sp)
1c009404:	da4a                	sw	s2,52(sp)
1c009406:	d84e                	sw	s3,48(sp)
1c009408:	d05e                	sw	s7,32(sp)
1c00940a:	ce62                	sw	s8,28(sp)
1c00940c:	ca6a                	sw	s10,20(sp)
1c00940e:	c86e                	sw	s11,16(sp)
1c009410:	c436                	sw	a3,8(sp)
1c009412:	00452b03          	lw	s6,4(a0)
1c009416:	01052c83          	lw	s9,16(a0)
1c00941a:	0487c433          	p.min	s0,a5,s0
1c00941e:	12070163          	beqz	a4,1c009540 <mm_u2+0x17c>
1c009422:	108ad163          	ble	s0,s5,1c009524 <mm_u2+0x160>
1c009426:	c00a3fb3          	p.bclr	t6,s4,0,0
1c00942a:	0f905d63          	blez	s9,1c009524 <mm_u2+0x160>
1c00942e:	034a8733          	mul	a4,s5,s4
1c009432:	4902                	lw	s2,0(sp)
1c009434:	002a1393          	slli	t2,s4,0x2
1c009438:	4692                	lw	a3,4(sp)
1c00943a:	41900c33          	neg	s8,s9
1c00943e:	002c1d13          	slli	s10,s8,0x2
1c009442:	8bd6                	mv	s7,s5
1c009444:	0c0e                	slli	s8,s8,0x3
1c009446:	039a8db3          	mul	s11,s5,s9
1c00944a:	00170493          	addi	s1,a4,1
1c00944e:	c63a                	sw	a4,12(sp)
1c009450:	4702                	lw	a4,0(sp)
1c009452:	048a                	slli	s1,s1,0x2
1c009454:	94ba                	add	s1,s1,a4
1c009456:	427a8933          	p.mac	s2,s5,t2
1c00945a:	019d87b3          	add	a5,s11,s9
1c00945e:	078a                	slli	a5,a5,0x2
1c009460:	97b6                	add	a5,a5,a3
1c009462:	00fd02b3          	add	t0,s10,a5
1c009466:	40578e33          	sub	t3,a5,t0
1c00946a:	1e71                	addi	t3,t3,-4
1c00946c:	002e5e13          	srli	t3,t3,0x2
1c009470:	8eda                	mv	t4,s6
1c009472:	8f16                	mv	t5,t0
1c009474:	0e05                	addi	t3,t3,1
1c009476:	0001                	nop
1c009478:	026e407b          	lp.setup	x0,t3,1c0094c4 <mm_u2+0x100>
1c00947c:	00000793          	li	a5,0
1c009480:	040f8063          	beqz	t6,1c0094c0 <mm_u2+0xfc>
1c009484:	ffff8693          	addi	a3,t6,-1
1c009488:	8285                	srli	a3,a3,0x1
1c00948a:	001f8713          	addi	a4,t6,1
1c00948e:	4889                	li	a7,2
1c009490:	004e8613          	addi	a2,t4,4
1c009494:	85a6                	mv	a1,s1
1c009496:	8576                	mv	a0,t4
1c009498:	884a                	mv	a6,s2
1c00949a:	00000793          	li	a5,0
1c00949e:	0685                	addi	a3,a3,1
1c0094a0:	1b176363          	bltu	a4,a7,1c009646 <mm_u2+0x282>
1c0094a4:	00c6c0fb          	lp.setup	x1,a3,1c0094bc <mm_u2+0xf8>
1c0094a8:	0088298b          	p.lw	s3,8(a6!)
1c0094ac:	0085230b          	p.lw	t1,8(a0!)
1c0094b0:	0085a88b          	p.lw	a7,8(a1!)
1c0094b4:	0086270b          	p.lw	a4,8(a2!)
1c0094b8:	7869f7c3          	fmadd.s	a5,s3,t1,a5
1c0094bc:	78e8f7c3          	fmadd.s	a5,a7,a4,a5
1c0094c0:	00ff222b          	p.sw	a5,4(t5!)
1c0094c4:	9e9e                	add	t4,t4,t2
1c0094c6:	0b85                	addi	s7,s7,1
1c0094c8:	418287b3          	sub	a5,t0,s8
1c0094cc:	991e                	add	s2,s2,t2
1c0094ce:	949e                	add	s1,s1,t2
1c0094d0:	f97419e3          	bne	s0,s7,1c009462 <mm_u2+0x9e>
1c0094d4:	47a2                	lw	a5,8(sp)
1c0094d6:	c7b9                	beqz	a5,1c009524 <mm_u2+0x160>
1c0094d8:	4732                	lw	a4,12(sp)
1c0094da:	fffa0813          	addi	a6,s4,-1
1c0094de:	4692                	lw	a3,4(sp)
1c0094e0:	983a                	add	a6,a6,a4
1c0094e2:	4702                	lw	a4,0(sp)
1c0094e4:	002d9593          	slli	a1,s11,0x2
1c0094e8:	080a                	slli	a6,a6,0x2
1c0094ea:	ffc38793          	addi	a5,t2,-4
1c0094ee:	0c8a                	slli	s9,s9,0x2
1c0094f0:	95b6                	add	a1,a1,a3
1c0094f2:	983a                	add	a6,a6,a4
1c0094f4:	9b3e                	add	s6,s6,a5
1c0094f6:	41540433          	sub	s0,s0,s5
1c0094fa:	0001                	nop
1c0094fc:	0134407b          	lp.setup	x0,s0,1c009522 <mm_u2+0x15e>
1c009500:	ffcc8713          	addi	a4,s9,-4
1c009504:	8309                	srli	a4,a4,0x2
1c009506:	86da                	mv	a3,s6
1c009508:	0705                	addi	a4,a4,1
1c00950a:	0001                	nop
1c00950c:	009740fb          	lp.setup	x1,a4,1c00951e <mm_u2+0x15a>
1c009510:	4190                	lw	a2,0(a1)
1c009512:	00082783          	lw	a5,0(a6)
1c009516:	2076f50b          	p.lw	a0,t2(a3!)
1c00951a:	60a7f7c3          	fmadd.s	a5,a5,a0,a2
1c00951e:	00f5a22b          	p.sw	a5,4(a1!)
1c009522:	981e                	add	a6,a6,t2
1c009524:	5472                	lw	s0,60(sp)
1c009526:	54e2                	lw	s1,56(sp)
1c009528:	5952                	lw	s2,52(sp)
1c00952a:	59c2                	lw	s3,48(sp)
1c00952c:	5a32                	lw	s4,44(sp)
1c00952e:	5aa2                	lw	s5,40(sp)
1c009530:	5b12                	lw	s6,36(sp)
1c009532:	5b82                	lw	s7,32(sp)
1c009534:	4c72                	lw	s8,28(sp)
1c009536:	4ce2                	lw	s9,24(sp)
1c009538:	4d52                	lw	s10,20(sp)
1c00953a:	4dc2                	lw	s11,16(sp)
1c00953c:	6121                	addi	sp,sp,64
1c00953e:	8082                	ret
1c009540:	fe8ad2e3          	ble	s0,s5,1c009524 <mm_u2+0x160>
1c009544:	c00a3fb3          	p.bclr	t6,s4,0,0
1c009548:	fd905ee3          	blez	s9,1c009524 <mm_u2+0x160>
1c00954c:	4682                	lw	a3,0(sp)
1c00954e:	034a87b3          	mul	a5,s5,s4
1c009552:	4b92                	lw	s7,4(sp)
1c009554:	002c9993          	slli	s3,s9,0x2
1c009558:	002a1393          	slli	t2,s4,0x2
1c00955c:	8936                	mv	s2,a3
1c00955e:	013b0d33          	add	s10,s6,s3
1c009562:	003c9d93          	slli	s11,s9,0x3
1c009566:	8c56                	mv	s8,s5
1c009568:	433a8bb3          	p.mac	s7,s5,s3
1c00956c:	00178493          	addi	s1,a5,1
1c009570:	048a                	slli	s1,s1,0x2
1c009572:	c63e                	sw	a5,12(sp)
1c009574:	94b6                	add	s1,s1,a3
1c009576:	427a8933          	p.mac	s2,s5,t2
1c00957a:	416d0e33          	sub	t3,s10,s6
1c00957e:	1e71                	addi	t3,t3,-4
1c009580:	002e5e13          	srli	t3,t3,0x2
1c009584:	8eda                	mv	t4,s6
1c009586:	8f5e                	mv	t5,s7
1c009588:	0e05                	addi	t3,t3,1
1c00958a:	0001                	nop
1c00958c:	026e407b          	lp.setup	x0,t3,1c0095d8 <mm_u2+0x214>
1c009590:	00000793          	li	a5,0
1c009594:	040f8063          	beqz	t6,1c0095d4 <mm_u2+0x210>
1c009598:	ffff8693          	addi	a3,t6,-1
1c00959c:	8285                	srli	a3,a3,0x1
1c00959e:	001f8713          	addi	a4,t6,1
1c0095a2:	4889                	li	a7,2
1c0095a4:	01d985b3          	add	a1,s3,t4
1c0095a8:	8626                	mv	a2,s1
1c0095aa:	8576                	mv	a0,t4
1c0095ac:	884a                	mv	a6,s2
1c0095ae:	00000793          	li	a5,0
1c0095b2:	0685                	addi	a3,a3,1
1c0095b4:	09176763          	bltu	a4,a7,1c009642 <mm_u2+0x27e>
1c0095b8:	00c6c0fb          	lp.setup	x1,a3,1c0095d0 <mm_u2+0x20c>
1c0095bc:	0088228b          	p.lw	t0,8(a6!)
1c0095c0:	21b5730b          	p.lw	t1,s11(a0!)
1c0095c4:	0086288b          	p.lw	a7,8(a2!)
1c0095c8:	21b5f70b          	p.lw	a4,s11(a1!)
1c0095cc:	7862f7c3          	fmadd.s	a5,t0,t1,a5
1c0095d0:	78e8f7c3          	fmadd.s	a5,a7,a4,a5
1c0095d4:	00ff222b          	p.sw	a5,4(t5!)
1c0095d8:	0e91                	addi	t4,t4,4
1c0095da:	0c05                	addi	s8,s8,1
1c0095dc:	9bce                	add	s7,s7,s3
1c0095de:	991e                	add	s2,s2,t2
1c0095e0:	949e                	add	s1,s1,t2
1c0095e2:	f9841ce3          	bne	s0,s8,1c00957a <mm_u2+0x1b6>
1c0095e6:	4722                	lw	a4,8(sp)
1c0095e8:	df15                	beqz	a4,1c009524 <mm_u2+0x160>
1c0095ea:	fffa0793          	addi	a5,s4,-1
1c0095ee:	035c85b3          	mul	a1,s9,s5
1c0095f2:	46b2                	lw	a3,12(sp)
1c0095f4:	40000837          	lui	a6,0x40000
1c0095f8:	187d                	addi	a6,a6,-1
1c0095fa:	9852                	add	a6,a6,s4
1c0095fc:	9836                	add	a6,a6,a3
1c0095fe:	4702                	lw	a4,0(sp)
1c009600:	4692                	lw	a3,4(sp)
1c009602:	080a                	slli	a6,a6,0x2
1c009604:	983a                	add	a6,a6,a4
1c009606:	039787b3          	mul	a5,a5,s9
1c00960a:	058a                	slli	a1,a1,0x2
1c00960c:	41540433          	sub	s0,s0,s5
1c009610:	95b6                	add	a1,a1,a3
1c009612:	078a                	slli	a5,a5,0x2
1c009614:	9b3e                	add	s6,s6,a5
1c009616:	0001                	nop
1c009618:	0134407b          	lp.setup	x0,s0,1c00963e <mm_u2+0x27a>
1c00961c:	ffc98713          	addi	a4,s3,-4
1c009620:	8309                	srli	a4,a4,0x2
1c009622:	86da                	mv	a3,s6
1c009624:	0705                	addi	a4,a4,1
1c009626:	0001                	nop
1c009628:	009740fb          	lp.setup	x1,a4,1c00963a <mm_u2+0x276>
1c00962c:	4190                	lw	a2,0(a1)
1c00962e:	0046a50b          	p.lw	a0,4(a3!)
1c009632:	00082783          	lw	a5,0(a6) # 40000000 <__l2_shared_end+0x23fe7af4>
1c009636:	60a7f7c3          	fmadd.s	a5,a5,a0,a2
1c00963a:	00f5a22b          	p.sw	a5,4(a1!)
1c00963e:	981e                	add	a6,a6,t2
1c009640:	b5d5                	j	1c009524 <mm_u2+0x160>
1c009642:	4685                	li	a3,1
1c009644:	bf95                	j	1c0095b8 <mm_u2+0x1f4>
1c009646:	4685                	li	a3,1
1c009648:	bdb1                	j	1c0094a4 <mm_u2+0xe0>
	...

1c00964c <mm_unroll_1x2>:
1c00964c:	7139                	addi	sp,sp,-64
1c00964e:	dc26                	sw	s1,56(sp)
1c009650:	4544                	lw	s1,12(a0)
1c009652:	d456                	sw	s5,40(sp)
1c009654:	f1402af3          	csrr	s5,mhartid
1c009658:	00748693          	addi	a3,s1,7
1c00965c:	41f6d793          	srai	a5,a3,0x1f
1c009660:	f837b7b3          	p.bclr	a5,a5,28,3
1c009664:	06d7a7db          	p.addn	a5,a5,a3,3
1c009668:	f45abab3          	p.bclr	s5,s5,26,5
1c00966c:	02fa8ab3          	mul	s5,s5,a5
1c009670:	ce62                	sw	s8,28(sp)
1c009672:	01052c03          	lw	s8,16(a0)
1c009676:	de22                	sw	s0,60(sp)
1c009678:	da4a                	sw	s2,52(sp)
1c00967a:	d84e                	sw	s3,48(sp)
1c00967c:	d652                	sw	s4,44(sp)
1c00967e:	d25a                	sw	s6,36(sp)
1c009680:	d05e                	sw	s7,32(sp)
1c009682:	cc66                	sw	s9,24(sp)
1c009684:	97d6                	add	a5,a5,s5
1c009686:	ca6a                	sw	s10,20(sp)
1c009688:	c86e                	sw	s11,16(sp)
1c00968a:	4685                	li	a3,1
1c00968c:	0497c4b3          	p.min	s1,a5,s1
1c009690:	2586dc63          	ble	s8,a3,1c0098e8 <mm_unroll_1x2+0x29c>
1c009694:	4514                	lw	a3,8(a0)
1c009696:	4d1c                	lw	a5,24(a0)
1c009698:	fc1c3733          	p.bclr	a4,s8,30,1
1c00969c:	c436                	sw	a3,8(sp)
1c00969e:	c63a                	sw	a4,12(sp)
1c0096a0:	00052f03          	lw	t5,0(a0)
1c0096a4:	00452c83          	lw	s9,4(a0)
1c0096a8:	01452f83          	lw	t6,20(a0)
1c0096ac:	12079363          	bnez	a5,1c0097d2 <mm_unroll_1x2+0x186>
1c0096b0:	109ad363          	ble	s1,s5,1c0097b6 <mm_unroll_1x2+0x16a>
1c0096b4:	c00c3b33          	p.bclr	s6,s8,0,0
1c0096b8:	080b0b63          	beqz	s6,1c00974e <mm_unroll_1x2+0x102>
1c0096bc:	897e                	mv	s2,t6
1c0096be:	43fa8933          	p.mac	s2,s5,t6
1c0096c2:	89b6                	mv	s3,a3
1c0096c4:	002c1d13          	slli	s10,s8,0x2
1c0096c8:	002f9b93          	slli	s7,t6,0x2
1c0096cc:	847a                	mv	s0,t5
1c0096ce:	8a56                	mv	s4,s5
1c0096d0:	43aa89b3          	p.mac	s3,s5,s10
1c0096d4:	090a                	slli	s2,s2,0x2
1c0096d6:	997a                	add	s2,s2,t5
1c0096d8:	437a8433          	p.mac	s0,s5,s7
1c0096dc:	fffb0313          	addi	t1,s6,-1
1c0096e0:	00135313          	srli	t1,t1,0x1
1c0096e4:	001b0793          	addi	a5,s6,1
1c0096e8:	4709                	li	a4,2
1c0096ea:	88e6                	mv	a7,s9
1c0096ec:	00498e13          	addi	t3,s3,4
1c0096f0:	8ece                	mv	t4,s3
1c0096f2:	0305                	addi	t1,t1,1
1c0096f4:	20e7e963          	bltu	a5,a4,1c009906 <mm_unroll_1x2+0x2ba>
1c0096f8:	0243407b          	lp.setup	x0,t1,1c009740 <mm_unroll_1x2+0xf4>
1c0096fc:	00000613          	li	a2,0
1c009700:	86b2                	mv	a3,a2
1c009702:	03f05b63          	blez	t6,1c009738 <mm_unroll_1x2+0xec>
1c009706:	00000613          	li	a2,0
1c00970a:	408907b3          	sub	a5,s2,s0
1c00970e:	17f1                	addi	a5,a5,-4
1c009710:	86b2                	mv	a3,a2
1c009712:	8389                	srli	a5,a5,0x2
1c009714:	00488813          	addi	a6,a7,4
1c009718:	8546                	mv	a0,a7
1c00971a:	85a2                	mv	a1,s0
1c00971c:	0785                	addi	a5,a5,1
1c00971e:	0001                	nop
1c009720:	00a7c0fb          	lp.setup	x1,a5,1c009734 <mm_unroll_1x2+0xe8>
1c009724:	0045a70b          	p.lw	a4,4(a1!)
1c009728:	21a5738b          	p.lw	t2,s10(a0!)
1c00972c:	21a8728b          	p.lw	t0,s10(a6!)
1c009730:	68e3f6c3          	fmadd.s	a3,t2,a4,a3
1c009734:	60e2f643          	fmadd.s	a2,t0,a4,a2
1c009738:	00dea42b          	p.sw	a3,8(t4!)
1c00973c:	00ce242b          	p.sw	a2,8(t3!)
1c009740:	08a1                	addi	a7,a7,8
1c009742:	0a05                	addi	s4,s4,1
1c009744:	99ea                	add	s3,s3,s10
1c009746:	945e                	add	s0,s0,s7
1c009748:	995e                	add	s2,s2,s7
1c00974a:	f94499e3          	bne	s1,s4,1c0096dc <mm_unroll_1x2+0x90>
1c00974e:	4732                	lw	a4,12(sp)
1c009750:	c33d                	beqz	a4,1c0097b6 <mm_unroll_1x2+0x16a>
1c009752:	40000537          	lui	a0,0x40000
1c009756:	001a8793          	addi	a5,s5,1
1c00975a:	157d                	addi	a0,a0,-1
1c00975c:	43878533          	p.mac	a0,a5,s8
1c009760:	002f9893          	slli	a7,t6,0x2
1c009764:	0c0a                	slli	s8,s8,0x2
1c009766:	ffcc0793          	addi	a5,s8,-4
1c00976a:	9cbe                	add	s9,s9,a5
1c00976c:	47a2                	lw	a5,8(sp)
1c00976e:	415484b3          	sub	s1,s1,s5
1c009772:	431a8f33          	p.mac	t5,s5,a7
1c009776:	050a                	slli	a0,a0,0x2
1c009778:	953e                	add	a0,a0,a5
1c00977a:	877a                	mv	a4,t5
1c00977c:	01c4c07b          	lp.setup	x0,s1,1c0097b4 <mm_unroll_1x2+0x168>
1c009780:	00000693          	li	a3,0
1c009784:	00e88333          	add	t1,a7,a4
1c009788:	03f05463          	blez	t6,1c0097b0 <mm_unroll_1x2+0x164>
1c00978c:	00e88333          	add	t1,a7,a4
1c009790:	40e307b3          	sub	a5,t1,a4
1c009794:	17f1                	addi	a5,a5,-4
1c009796:	00000693          	li	a3,0
1c00979a:	8389                	srli	a5,a5,0x2
1c00979c:	8666                	mv	a2,s9
1c00979e:	0785                	addi	a5,a5,1
1c0097a0:	0067c0fb          	lp.setup	x1,a5,1c0097ac <mm_unroll_1x2+0x160>
1c0097a4:	0047280b          	p.lw	a6,4(a4!)
1c0097a8:	2186758b          	p.lw	a1,s8(a2!)
1c0097ac:	68b876c3          	fmadd.s	a3,a6,a1,a3
1c0097b0:	00d56c2b          	p.sw	a3,s8(a0!)
1c0097b4:	871a                	mv	a4,t1
1c0097b6:	5472                	lw	s0,60(sp)
1c0097b8:	54e2                	lw	s1,56(sp)
1c0097ba:	5952                	lw	s2,52(sp)
1c0097bc:	59c2                	lw	s3,48(sp)
1c0097be:	5a32                	lw	s4,44(sp)
1c0097c0:	5aa2                	lw	s5,40(sp)
1c0097c2:	5b12                	lw	s6,36(sp)
1c0097c4:	5b82                	lw	s7,32(sp)
1c0097c6:	4c72                	lw	s8,28(sp)
1c0097c8:	4ce2                	lw	s9,24(sp)
1c0097ca:	4d52                	lw	s10,20(sp)
1c0097cc:	4dc2                	lw	s11,16(sp)
1c0097ce:	6121                	addi	sp,sp,64
1c0097d0:	8082                	ret
1c0097d2:	fe9ad2e3          	ble	s1,s5,1c0097b6 <mm_unroll_1x2+0x16a>
1c0097d6:	c00c3d33          	p.bclr	s10,s8,0,0
1c0097da:	080d0e63          	beqz	s10,1c009876 <mm_unroll_1x2+0x22a>
1c0097de:	897e                	mv	s2,t6
1c0097e0:	43fa8933          	p.mac	s2,s5,t6
1c0097e4:	002c1d93          	slli	s11,s8,0x2
1c0097e8:	002f9993          	slli	s3,t6,0x2
1c0097ec:	8b36                	mv	s6,a3
1c0097ee:	847a                	mv	s0,t5
1c0097f0:	003f9a13          	slli	s4,t6,0x3
1c0097f4:	8bd6                	mv	s7,s5
1c0097f6:	43ba8b33          	p.mac	s6,s5,s11
1c0097fa:	090a                	slli	s2,s2,0x2
1c0097fc:	997a                	add	s2,s2,t5
1c0097fe:	433a8433          	p.mac	s0,s5,s3
1c009802:	fffd0893          	addi	a7,s10,-1
1c009806:	0018d893          	srli	a7,a7,0x1
1c00980a:	001d0793          	addi	a5,s10,1
1c00980e:	4709                	li	a4,2
1c009810:	8866                	mv	a6,s9
1c009812:	004b0313          	addi	t1,s6,4
1c009816:	8e5a                	mv	t3,s6
1c009818:	0885                	addi	a7,a7,1
1c00981a:	0ee7e863          	bltu	a5,a4,1c00990a <mm_unroll_1x2+0x2be>
1c00981e:	0001                	nop
1c009820:	0248c07b          	lp.setup	x0,a7,1c009868 <mm_unroll_1x2+0x21c>
1c009824:	00000613          	li	a2,0
1c009828:	86b2                	mv	a3,a2
1c00982a:	03f05b63          	blez	t6,1c009860 <mm_unroll_1x2+0x214>
1c00982e:	00000613          	li	a2,0
1c009832:	408907b3          	sub	a5,s2,s0
1c009836:	17f1                	addi	a5,a5,-4
1c009838:	86b2                	mv	a3,a2
1c00983a:	8389                	srli	a5,a5,0x2
1c00983c:	01098eb3          	add	t4,s3,a6
1c009840:	8542                	mv	a0,a6
1c009842:	85a2                	mv	a1,s0
1c009844:	0785                	addi	a5,a5,1
1c009846:	0001                	nop
1c009848:	00a7c0fb          	lp.setup	x1,a5,1c00985c <mm_unroll_1x2+0x210>
1c00984c:	0045a70b          	p.lw	a4,4(a1!)
1c009850:	0045238b          	p.lw	t2,4(a0!) # 40000004 <__l2_shared_end+0x23fe7af8>
1c009854:	004ea28b          	p.lw	t0,4(t4!)
1c009858:	68e3f6c3          	fmadd.s	a3,t2,a4,a3
1c00985c:	60e2f643          	fmadd.s	a2,t0,a4,a2
1c009860:	00de242b          	p.sw	a3,8(t3!)
1c009864:	00c3242b          	p.sw	a2,8(t1!)
1c009868:	9852                	add	a6,a6,s4
1c00986a:	0b85                	addi	s7,s7,1
1c00986c:	9b6e                	add	s6,s6,s11
1c00986e:	944e                	add	s0,s0,s3
1c009870:	994e                	add	s2,s2,s3
1c009872:	f97498e3          	bne	s1,s7,1c009802 <mm_unroll_1x2+0x1b6>
1c009876:	46b2                	lw	a3,12(sp)
1c009878:	de9d                	beqz	a3,1c0097b6 <mm_unroll_1x2+0x16a>
1c00987a:	40000537          	lui	a0,0x40000
1c00987e:	001a8713          	addi	a4,s5,1
1c009882:	fffc0793          	addi	a5,s8,-1
1c009886:	157d                	addi	a0,a0,-1
1c009888:	43870533          	p.mac	a0,a4,s8
1c00988c:	002f9893          	slli	a7,t6,0x2
1c009890:	46a2                	lw	a3,8(sp)
1c009892:	0c0a                	slli	s8,s8,0x2
1c009894:	415484b3          	sub	s1,s1,s5
1c009898:	03f787b3          	mul	a5,a5,t6
1c00989c:	050a                	slli	a0,a0,0x2
1c00989e:	9536                	add	a0,a0,a3
1c0098a0:	431a8f33          	p.mac	t5,s5,a7
1c0098a4:	078a                	slli	a5,a5,0x2
1c0098a6:	9cbe                	add	s9,s9,a5
1c0098a8:	877a                	mv	a4,t5
1c0098aa:	0001                	nop
1c0098ac:	01c4c07b          	lp.setup	x0,s1,1c0098e4 <mm_unroll_1x2+0x298>
1c0098b0:	00000693          	li	a3,0
1c0098b4:	00e88333          	add	t1,a7,a4
1c0098b8:	03f05463          	blez	t6,1c0098e0 <mm_unroll_1x2+0x294>
1c0098bc:	00e88333          	add	t1,a7,a4
1c0098c0:	40e307b3          	sub	a5,t1,a4
1c0098c4:	17f1                	addi	a5,a5,-4
1c0098c6:	00000693          	li	a3,0
1c0098ca:	8389                	srli	a5,a5,0x2
1c0098cc:	8666                	mv	a2,s9
1c0098ce:	0785                	addi	a5,a5,1
1c0098d0:	0067c0fb          	lp.setup	x1,a5,1c0098dc <mm_unroll_1x2+0x290>
1c0098d4:	0047280b          	p.lw	a6,4(a4!)
1c0098d8:	0046258b          	p.lw	a1,4(a2!)
1c0098dc:	68b876c3          	fmadd.s	a3,a6,a1,a3
1c0098e0:	00d56c2b          	p.sw	a3,s8(a0!)
1c0098e4:	871a                	mv	a4,t1
1c0098e6:	bdc1                	j	1c0097b6 <mm_unroll_1x2+0x16a>
1c0098e8:	5472                	lw	s0,60(sp)
1c0098ea:	54e2                	lw	s1,56(sp)
1c0098ec:	5952                	lw	s2,52(sp)
1c0098ee:	59c2                	lw	s3,48(sp)
1c0098f0:	5a32                	lw	s4,44(sp)
1c0098f2:	5aa2                	lw	s5,40(sp)
1c0098f4:	5b12                	lw	s6,36(sp)
1c0098f6:	5b82                	lw	s7,32(sp)
1c0098f8:	4c72                	lw	s8,28(sp)
1c0098fa:	4ce2                	lw	s9,24(sp)
1c0098fc:	4d52                	lw	s10,20(sp)
1c0098fe:	4dc2                	lw	s11,16(sp)
1c009900:	6121                	addi	sp,sp,64
1c009902:	d6eff06f          	j	1c008e70 <mm>
1c009906:	4305                	li	t1,1
1c009908:	bbc5                	j	1c0096f8 <mm_unroll_1x2+0xac>
1c00990a:	4885                	li	a7,1
1c00990c:	bf09                	j	1c00981e <mm_unroll_1x2+0x1d2>
	...

1c009910 <mm_unroll_1x4>:
1c009910:	715d                	addi	sp,sp,-80
1c009912:	d26a                	sw	s10,36(sp)
1c009914:	01052d03          	lw	s10,16(a0) # 40000010 <__l2_shared_end+0x23fe7b04>
1c009918:	c6a2                	sw	s0,76(sp)
1c00991a:	c4a6                	sw	s1,72(sp)
1c00991c:	c2ca                	sw	s2,68(sp)
1c00991e:	c0ce                	sw	s3,64(sp)
1c009920:	de52                	sw	s4,60(sp)
1c009922:	dc56                	sw	s5,56(sp)
1c009924:	da5a                	sw	s6,52(sp)
1c009926:	d85e                	sw	s7,48(sp)
1c009928:	d662                	sw	s8,44(sp)
1c00992a:	d466                	sw	s9,40(sp)
1c00992c:	d06e                	sw	s11,32(sp)
1c00992e:	468d                	li	a3,3
1c009930:	f14027f3          	csrr	a5,mhartid
1c009934:	35a6dc63          	ble	s10,a3,1c009c8c <mm_unroll_1x4+0x37c>
1c009938:	00c52b83          	lw	s7,12(a0)
1c00993c:	f457b6b3          	p.bclr	a3,a5,26,5
1c009940:	4d0c                	lw	a1,24(a0)
1c009942:	007b8613          	addi	a2,s7,7
1c009946:	41f65793          	srai	a5,a2,0x1f
1c00994a:	f837b7b3          	p.bclr	a5,a5,28,3
1c00994e:	06c7a7db          	p.addn	a5,a5,a2,3
1c009952:	02f686b3          	mul	a3,a3,a5
1c009956:	4110                	lw	a2,0(a0)
1c009958:	01452983          	lw	s3,20(a0)
1c00995c:	c632                	sw	a2,12(sp)
1c00995e:	4510                	lw	a2,8(a0)
1c009960:	c832                	sw	a2,16(sp)
1c009962:	c036                	sw	a3,0(sp)
1c009964:	4154                	lw	a3,4(a0)
1c009966:	4702                	lw	a4,0(sp)
1c009968:	c236                	sw	a3,4(sp)
1c00996a:	fa2d36b3          	p.bclr	a3,s10,29,2
1c00996e:	97ba                	add	a5,a5,a4
1c009970:	ca36                	sw	a3,20(sp)
1c009972:	8dba                	mv	s11,a4
1c009974:	0577cbb3          	p.min	s7,a5,s7
1c009978:	16059163          	bnez	a1,1c009ada <mm_unroll_1x4+0x1ca>
1c00997c:	15775163          	ble	s7,a4,1c009abe <mm_unroll_1x4+0x1ae>
1c009980:	c20d3cb3          	p.bclr	s9,s10,1,0
1c009984:	0a0c8563          	beqz	s9,1c009a2e <mm_unroll_1x4+0x11e>
1c009988:	47b2                	lw	a5,12(sp)
1c00998a:	894e                	mv	s2,s3
1c00998c:	43370933          	p.mac	s2,a4,s3
1c009990:	8a32                	mv	s4,a2
1c009992:	002d1293          	slli	t0,s10,0x2
1c009996:	00299b13          	slli	s6,s3,0x2
1c00999a:	84be                	mv	s1,a5
1c00999c:	4f92                	lw	t6,4(sp)
1c00999e:	8aba                	mv	s5,a4
1c0099a0:	42570a33          	p.mac	s4,a4,t0
1c0099a4:	090a                	slli	s2,s2,0x2
1c0099a6:	993e                	add	s2,s2,a5
1c0099a8:	436704b3          	p.mac	s1,a4,s6
1c0099ac:	fffc8e93          	addi	t4,s9,-1
1c0099b0:	002ede93          	srli	t4,t4,0x2
1c0099b4:	003c8793          	addi	a5,s9,3
1c0099b8:	4711                	li	a4,4
1c0099ba:	8f7e                	mv	t5,t6
1c0099bc:	8e52                	mv	t3,s4
1c0099be:	0e85                	addi	t4,t4,1
1c0099c0:	2ee7e463          	bltu	a5,a4,1c009ca8 <mm_unroll_1x4+0x398>
1c0099c4:	2b305163          	blez	s3,1c009c66 <mm_unroll_1x4+0x356>
1c0099c8:	00000813          	li	a6,0
1c0099cc:	409906b3          	sub	a3,s2,s1
1c0099d0:	16f1                	addi	a3,a3,-4
1c0099d2:	8542                	mv	a0,a6
1c0099d4:	85c2                	mv	a1,a6
1c0099d6:	8642                	mv	a2,a6
1c0099d8:	8289                	srli	a3,a3,0x2
1c0099da:	87fa                	mv	a5,t5
1c0099dc:	88a6                	mv	a7,s1
1c0099de:	0685                	addi	a3,a3,1
1c0099e0:	0136c0fb          	lp.setup	x1,a3,1c009a06 <mm_unroll_1x4+0xf6>
1c0099e4:	0048a70b          	p.lw	a4,4(a7!)
1c0099e8:	0007ac03          	lw	s8,0(a5)
1c0099ec:	43c0                	lw	s0,4(a5)
1c0099ee:	0087a383          	lw	t2,8(a5)
1c0099f2:	00c7a303          	lw	t1,12(a5)
1c0099f6:	60ec7643          	fmadd.s	a2,s8,a4,a2
1c0099fa:	58e475c3          	fmadd.s	a1,s0,a4,a1
1c0099fe:	50e3f543          	fmadd.s	a0,t2,a4,a0
1c009a02:	80e37843          	fmadd.s	a6,t1,a4,a6
1c009a06:	9796                	add	a5,a5,t0
1c009a08:	00ce2023          	sw	a2,0(t3)
1c009a0c:	00be2223          	sw	a1,4(t3)
1c009a10:	00ae2423          	sw	a0,8(t3)
1c009a14:	010e2623          	sw	a6,12(t3)
1c009a18:	0f41                	addi	t5,t5,16
1c009a1a:	0e41                	addi	t3,t3,16
1c009a1c:	1efd                	addi	t4,t4,-1
1c009a1e:	fa0e93e3          	bnez	t4,1c0099c4 <mm_unroll_1x4+0xb4>
1c009a22:	0a85                	addi	s5,s5,1
1c009a24:	9a16                	add	s4,s4,t0
1c009a26:	94da                	add	s1,s1,s6
1c009a28:	995a                	add	s2,s2,s6
1c009a2a:	f95b91e3          	bne	s7,s5,1c0099ac <mm_unroll_1x4+0x9c>
1c009a2e:	47d2                	lw	a5,20(sp)
1c009a30:	c7d9                	beqz	a5,1c009abe <mm_unroll_1x4+0x1ae>
1c009a32:	09acd663          	ble	s10,s9,1c009abe <mm_unroll_1x4+0x1ae>
1c009a36:	4602                	lw	a2,0(sp)
1c009a38:	4732                	lw	a4,12(sp)
1c009a3a:	88ce                	mv	a7,s3
1c009a3c:	433608b3          	p.mac	a7,a2,s3
1c009a40:	00299e93          	slli	t4,s3,0x2
1c009a44:	82ba                	mv	t0,a4
1c009a46:	4692                	lw	a3,4(sp)
1c009a48:	43c2                	lw	t2,16(sp)
1c009a4a:	002c9e13          	slli	t3,s9,0x2
1c009a4e:	002d1f13          	slli	t5,s10,0x2
1c009a52:	9e36                	add	t3,t3,a3
1c009a54:	01e68fb3          	add	t6,a3,t5
1c009a58:	03a60333          	mul	t1,a2,s10
1c009a5c:	088a                	slli	a7,a7,0x2
1c009a5e:	98ba                	add	a7,a7,a4
1c009a60:	43d602b3          	p.mac	t0,a2,t4
1c009a64:	41cf8833          	sub	a6,t6,t3
1c009a68:	01930433          	add	s0,t1,s9
1c009a6c:	1871                	addi	a6,a6,-4
1c009a6e:	040a                	slli	s0,s0,0x2
1c009a70:	00285813          	srli	a6,a6,0x2
1c009a74:	941e                	add	s0,s0,t2
1c009a76:	84f2                	mv	s1,t3
1c009a78:	0805                	addi	a6,a6,1
1c009a7a:	0001                	nop
1c009a7c:	01a8407b          	lp.setup	x0,a6,1c009ab0 <mm_unroll_1x4+0x1a0>
1c009a80:	00000713          	li	a4,0
1c009a84:	03305463          	blez	s3,1c009aac <mm_unroll_1x4+0x19c>
1c009a88:	405887b3          	sub	a5,a7,t0
1c009a8c:	17f1                	addi	a5,a5,-4
1c009a8e:	00000713          	li	a4,0
1c009a92:	8389                	srli	a5,a5,0x2
1c009a94:	8626                	mv	a2,s1
1c009a96:	8696                	mv	a3,t0
1c009a98:	0785                	addi	a5,a5,1
1c009a9a:	0001                	nop
1c009a9c:	0067c0fb          	lp.setup	x1,a5,1c009aa8 <mm_unroll_1x4+0x198>
1c009aa0:	0046a50b          	p.lw	a0,4(a3!)
1c009aa4:	21e6758b          	p.lw	a1,t5(a2!)
1c009aa8:	70b57743          	fmadd.s	a4,a0,a1,a4
1c009aac:	00e4222b          	p.sw	a4,4(s0!)
1c009ab0:	0491                	addi	s1,s1,4
1c009ab2:	0d85                	addi	s11,s11,1
1c009ab4:	936a                	add	t1,t1,s10
1c009ab6:	92f6                	add	t0,t0,t4
1c009ab8:	98f6                	add	a7,a7,t4
1c009aba:	fbbb95e3          	bne	s7,s11,1c009a64 <mm_unroll_1x4+0x154>
1c009abe:	4436                	lw	s0,76(sp)
1c009ac0:	44a6                	lw	s1,72(sp)
1c009ac2:	4916                	lw	s2,68(sp)
1c009ac4:	4986                	lw	s3,64(sp)
1c009ac6:	5a72                	lw	s4,60(sp)
1c009ac8:	5ae2                	lw	s5,56(sp)
1c009aca:	5b52                	lw	s6,52(sp)
1c009acc:	5bc2                	lw	s7,48(sp)
1c009ace:	5c32                	lw	s8,44(sp)
1c009ad0:	5ca2                	lw	s9,40(sp)
1c009ad2:	5d12                	lw	s10,36(sp)
1c009ad4:	5d82                	lw	s11,32(sp)
1c009ad6:	6161                	addi	sp,sp,80
1c009ad8:	8082                	ret
1c009ada:	ff7752e3          	ble	s7,a4,1c009abe <mm_unroll_1x4+0x1ae>
1c009ade:	c20d3cb3          	p.bclr	s9,s10,1,0
1c009ae2:	0c0c8163          	beqz	s9,1c009ba4 <mm_unroll_1x4+0x294>
1c009ae6:	002d1793          	slli	a5,s10,0x2
1c009aea:	c43e                	sw	a5,8(sp)
1c009aec:	46b2                	lw	a3,12(sp)
1c009aee:	87b2                	mv	a5,a2
1c009af0:	4622                	lw	a2,8(sp)
1c009af2:	8b4e                	mv	s6,s3
1c009af4:	43370b33          	p.mac	s6,a4,s3
1c009af8:	00299a13          	slli	s4,s3,0x2
1c009afc:	8ab6                	mv	s5,a3
1c009afe:	cc6a                	sw	s10,24(sp)
1c009b00:	00499c13          	slli	s8,s3,0x4
1c009b04:	8d3a                	mv	s10,a4
1c009b06:	ce3a                	sw	a4,28(sp)
1c009b08:	42c707b3          	p.mac	a5,a4,a2
1c009b0c:	0b0a                	slli	s6,s6,0x2
1c009b0e:	9b36                	add	s6,s6,a3
1c009b10:	43470ab3          	p.mac	s5,a4,s4
1c009b14:	8dbe                	mv	s11,a5
1c009b16:	fffc8913          	addi	s2,s9,-1
1c009b1a:	00295913          	srli	s2,s2,0x2
1c009b1e:	003c8793          	addi	a5,s9,3
1c009b22:	4711                	li	a4,4
1c009b24:	4492                	lw	s1,4(sp)
1c009b26:	846e                	mv	s0,s11
1c009b28:	0905                	addi	s2,s2,1
1c009b2a:	18e7e163          	bltu	a5,a4,1c009cac <mm_unroll_1x4+0x39c>
1c009b2e:	11305963          	blez	s3,1c009c40 <mm_unroll_1x4+0x330>
1c009b32:	00000513          	li	a0,0
1c009b36:	415b0733          	sub	a4,s6,s5
1c009b3a:	009a08b3          	add	a7,s4,s1
1c009b3e:	1771                	addi	a4,a4,-4
1c009b40:	85aa                	mv	a1,a0
1c009b42:	862a                	mv	a2,a0
1c009b44:	86aa                	mv	a3,a0
1c009b46:	011a0833          	add	a6,s4,a7
1c009b4a:	8309                	srli	a4,a4,0x2
1c009b4c:	010a0eb3          	add	t4,s4,a6
1c009b50:	8e26                	mv	t3,s1
1c009b52:	8356                	mv	t1,s5
1c009b54:	0705                	addi	a4,a4,1
1c009b56:	0001                	nop
1c009b58:	012740fb          	lp.setup	x1,a4,1c009b7c <mm_unroll_1x4+0x26c>
1c009b5c:	0043278b          	p.lw	a5,4(t1!)
1c009b60:	004e238b          	p.lw	t2,4(t3!)
1c009b64:	0048a28b          	p.lw	t0,4(a7!)
1c009b68:	00482f8b          	p.lw	t6,4(a6!)
1c009b6c:	004eaf0b          	p.lw	t5,4(t4!)
1c009b70:	68f3f6c3          	fmadd.s	a3,t2,a5,a3
1c009b74:	60f2f643          	fmadd.s	a2,t0,a5,a2
1c009b78:	58fff5c3          	fmadd.s	a1,t6,a5,a1
1c009b7c:	50ff7543          	fmadd.s	a0,t5,a5,a0
1c009b80:	c014                	sw	a3,0(s0)
1c009b82:	c050                	sw	a2,4(s0)
1c009b84:	c40c                	sw	a1,8(s0)
1c009b86:	c448                	sw	a0,12(s0)
1c009b88:	94e2                	add	s1,s1,s8
1c009b8a:	0441                	addi	s0,s0,16
1c009b8c:	197d                	addi	s2,s2,-1
1c009b8e:	fa0910e3          	bnez	s2,1c009b2e <mm_unroll_1x4+0x21e>
1c009b92:	47a2                	lw	a5,8(sp)
1c009b94:	0d05                	addi	s10,s10,1
1c009b96:	9ad2                	add	s5,s5,s4
1c009b98:	9dbe                	add	s11,s11,a5
1c009b9a:	9b52                	add	s6,s6,s4
1c009b9c:	f7ab9de3          	bne	s7,s10,1c009b16 <mm_unroll_1x4+0x206>
1c009ba0:	4d62                	lw	s10,24(sp)
1c009ba2:	4df2                	lw	s11,28(sp)
1c009ba4:	4652                	lw	a2,20(sp)
1c009ba6:	de01                	beqz	a2,1c009abe <mm_unroll_1x4+0x1ae>
1c009ba8:	f1acdbe3          	ble	s10,s9,1c009abe <mm_unroll_1x4+0x1ae>
1c009bac:	4602                	lw	a2,0(sp)
1c009bae:	82ce                	mv	t0,s3
1c009bb0:	4fb2                	lw	t6,12(sp)
1c009bb2:	03a607b3          	mul	a5,a2,s10
1c009bb6:	4392                	lw	t2,4(sp)
1c009bb8:	00299313          	slli	t1,s3,0x2
1c009bbc:	46c2                	lw	a3,16(sp)
1c009bbe:	4732                	lw	a4,12(sp)
1c009bc0:	002d1f13          	slli	t5,s10,0x2
1c009bc4:	8436                	mv	s0,a3
1c009bc6:	433602b3          	p.mac	t0,a2,s3
1c009bca:	00fd0eb3          	add	t4,s10,a5
1c009bce:	002e9e13          	slli	t3,t4,0x2
1c009bd2:	9e36                	add	t3,t3,a3
1c009bd4:	42660fb3          	p.mac	t6,a2,t1
1c009bd8:	028a                	slli	t0,t0,0x2
1c009bda:	92ba                	add	t0,t0,a4
1c009bdc:	426c83b3          	p.mac	t2,s9,t1
1c009be0:	019788b3          	add	a7,a5,s9
1c009be4:	088a                	slli	a7,a7,0x2
1c009be6:	98a2                	add	a7,a7,s0
1c009be8:	411e0833          	sub	a6,t3,a7
1c009bec:	1871                	addi	a6,a6,-4
1c009bee:	00285813          	srli	a6,a6,0x2
1c009bf2:	849e                	mv	s1,t2
1c009bf4:	0805                	addi	a6,a6,1
1c009bf6:	0001                	nop
1c009bf8:	01a8407b          	lp.setup	x0,a6,1c009c2c <mm_unroll_1x4+0x31c>
1c009bfc:	00000713          	li	a4,0
1c009c00:	03305463          	blez	s3,1c009c28 <mm_unroll_1x4+0x318>
1c009c04:	41f287b3          	sub	a5,t0,t6
1c009c08:	17f1                	addi	a5,a5,-4
1c009c0a:	00000713          	li	a4,0
1c009c0e:	8389                	srli	a5,a5,0x2
1c009c10:	8626                	mv	a2,s1
1c009c12:	86fe                	mv	a3,t6
1c009c14:	0785                	addi	a5,a5,1
1c009c16:	0001                	nop
1c009c18:	0067c0fb          	lp.setup	x1,a5,1c009c24 <mm_unroll_1x4+0x314>
1c009c1c:	0046a50b          	p.lw	a0,4(a3!)
1c009c20:	0046258b          	p.lw	a1,4(a2!)
1c009c24:	70b57743          	fmadd.s	a4,a0,a1,a4
1c009c28:	00e8a22b          	p.sw	a4,4(a7!)
1c009c2c:	949a                	add	s1,s1,t1
1c009c2e:	0d85                	addi	s11,s11,1
1c009c30:	87f6                	mv	a5,t4
1c009c32:	9e7a                	add	t3,t3,t5
1c009c34:	9f9a                	add	t6,t6,t1
1c009c36:	929a                	add	t0,t0,t1
1c009c38:	9eea                	add	t4,t4,s10
1c009c3a:	fbbb93e3          	bne	s7,s11,1c009be0 <mm_unroll_1x4+0x2d0>
1c009c3e:	b541                	j	1c009abe <mm_unroll_1x4+0x1ae>
1c009c40:	00000513          	li	a0,0
1c009c44:	94e2                	add	s1,s1,s8
1c009c46:	0441                	addi	s0,s0,16
1c009c48:	85aa                	mv	a1,a0
1c009c4a:	862a                	mv	a2,a0
1c009c4c:	86aa                	mv	a3,a0
1c009c4e:	feb42c23          	sw	a1,-8(s0)
1c009c52:	fec42a23          	sw	a2,-12(s0)
1c009c56:	fed42823          	sw	a3,-16(s0)
1c009c5a:	fea42e23          	sw	a0,-4(s0)
1c009c5e:	197d                	addi	s2,s2,-1
1c009c60:	ec0917e3          	bnez	s2,1c009b2e <mm_unroll_1x4+0x21e>
1c009c64:	b73d                	j	1c009b92 <mm_unroll_1x4+0x282>
1c009c66:	00000813          	li	a6,0
1c009c6a:	0f41                	addi	t5,t5,16
1c009c6c:	0e41                	addi	t3,t3,16
1c009c6e:	8542                	mv	a0,a6
1c009c70:	85c2                	mv	a1,a6
1c009c72:	8642                	mv	a2,a6
1c009c74:	feae2c23          	sw	a0,-8(t3)
1c009c78:	febe2a23          	sw	a1,-12(t3)
1c009c7c:	fece2823          	sw	a2,-16(t3)
1c009c80:	ff0e2e23          	sw	a6,-4(t3)
1c009c84:	1efd                	addi	t4,t4,-1
1c009c86:	d20e9fe3          	bnez	t4,1c0099c4 <mm_unroll_1x4+0xb4>
1c009c8a:	bb61                	j	1c009a22 <mm_unroll_1x4+0x112>
1c009c8c:	4436                	lw	s0,76(sp)
1c009c8e:	44a6                	lw	s1,72(sp)
1c009c90:	4916                	lw	s2,68(sp)
1c009c92:	4986                	lw	s3,64(sp)
1c009c94:	5a72                	lw	s4,60(sp)
1c009c96:	5ae2                	lw	s5,56(sp)
1c009c98:	5b52                	lw	s6,52(sp)
1c009c9a:	5bc2                	lw	s7,48(sp)
1c009c9c:	5c32                	lw	s8,44(sp)
1c009c9e:	5ca2                	lw	s9,40(sp)
1c009ca0:	5d12                	lw	s10,36(sp)
1c009ca2:	5d82                	lw	s11,32(sp)
1c009ca4:	6161                	addi	sp,sp,80
1c009ca6:	b25d                	j	1c00964c <mm_unroll_1x2>
1c009ca8:	4e85                	li	t4,1
1c009caa:	bb29                	j	1c0099c4 <mm_unroll_1x4+0xb4>
1c009cac:	4905                	li	s2,1
1c009cae:	b541                	j	1c009b2e <mm_unroll_1x4+0x21e>

1c009cb0 <mm_unroll_1x8>:
1c009cb0:	4910                	lw	a2,16(a0)
1c009cb2:	7119                	addi	sp,sp,-128
1c009cb4:	dea2                	sw	s0,124(sp)
1c009cb6:	dca6                	sw	s1,120(sp)
1c009cb8:	daca                	sw	s2,116(sp)
1c009cba:	d8ce                	sw	s3,112(sp)
1c009cbc:	d6d2                	sw	s4,108(sp)
1c009cbe:	d4d6                	sw	s5,104(sp)
1c009cc0:	d2da                	sw	s6,100(sp)
1c009cc2:	d0de                	sw	s7,96(sp)
1c009cc4:	cee2                	sw	s8,92(sp)
1c009cc6:	cce6                	sw	s9,88(sp)
1c009cc8:	caea                	sw	s10,84(sp)
1c009cca:	c8ee                	sw	s11,80(sp)
1c009ccc:	d832                	sw	a2,48(sp)
1c009cce:	469d                	li	a3,7
1c009cd0:	f14027f3          	csrr	a5,mhartid
1c009cd4:	44c6d463          	ble	a2,a3,1c00a11c <mm_unroll_1x8+0x46c>
1c009cd8:	4550                	lw	a2,12(a0)
1c009cda:	f457b6b3          	p.bclr	a3,a5,26,5
1c009cde:	872a                	mv	a4,a0
1c009ce0:	00760593          	addi	a1,a2,7
1c009ce4:	41f5d793          	srai	a5,a1,0x1f
1c009ce8:	f837b7b3          	p.bclr	a5,a5,28,3
1c009cec:	06b7a7db          	p.addn	a5,a5,a1,3
1c009cf0:	02f686b3          	mul	a3,a3,a5
1c009cf4:	00472d03          	lw	s10,4(a4)
1c009cf8:	4d08                	lw	a0,24(a0)
1c009cfa:	dc6a                	sw	s10,56(sp)
1c009cfc:	d636                	sw	a3,44(sp)
1c009cfe:	4314                	lw	a3,0(a4)
1c009d00:	5d32                	lw	s10,44(sp)
1c009d02:	c0b6                	sw	a3,64(sp)
1c009d04:	4714                	lw	a3,8(a4)
1c009d06:	4b58                	lw	a4,20(a4)
1c009d08:	97ea                	add	a5,a5,s10
1c009d0a:	04c7c7b3          	p.min	a5,a5,a2
1c009d0e:	cc3a                	sw	a4,24(sp)
1c009d10:	5742                	lw	a4,48(sp)
1c009d12:	c2b6                	sw	a3,68(sp)
1c009d14:	c6ea                	sw	s10,76(sp)
1c009d16:	f8373733          	p.bclr	a4,a4,28,3
1c009d1a:	c4ba                	sw	a4,72(sp)
1c009d1c:	da3e                	sw	a5,52(sp)
1c009d1e:	20050c63          	beqz	a0,1c009f36 <mm_unroll_1x8+0x286>
1c009d22:	1efd5c63          	ble	a5,s10,1c009f1a <mm_unroll_1x8+0x26a>
1c009d26:	5642                	lw	a2,48(sp)
1c009d28:	46e2                	lw	a3,24(sp)
1c009d2a:	c4063633          	p.bclr	a2,a2,2,0
1c009d2e:	068a                	slli	a3,a3,0x2
1c009d30:	d032                	sw	a2,32(sp)
1c009d32:	c236                	sw	a3,4(sp)
1c009d34:	12060e63          	beqz	a2,1c009e70 <mm_unroll_1x8+0x1c0>
1c009d38:	47e2                	lw	a5,24(sp)
1c009d3a:	5742                	lw	a4,48(sp)
1c009d3c:	4696                	lw	a3,68(sp)
1c009d3e:	0796                	slli	a5,a5,0x5
1c009d40:	ce3e                	sw	a5,28(sp)
1c009d42:	47e2                	lw	a5,24(sp)
1c009d44:	070a                	slli	a4,a4,0x2
1c009d46:	42ed06b3          	p.mac	a3,s10,a4
1c009d4a:	de3a                	sw	a4,60(sp)
1c009d4c:	4706                	lw	a4,64(sp)
1c009d4e:	d26a                	sw	s10,36(sp)
1c009d50:	42fd07b3          	p.mac	a5,s10,a5
1c009d54:	d436                	sw	a3,40(sp)
1c009d56:	078a                	slli	a5,a5,0x2
1c009d58:	97ba                	add	a5,a5,a4
1c009d5a:	ca3e                	sw	a5,20(sp)
1c009d5c:	4792                	lw	a5,4(sp)
1c009d5e:	42fd0733          	p.mac	a4,s10,a5
1c009d62:	c83a                	sw	a4,16(sp)
1c009d64:	5782                	lw	a5,32(sp)
1c009d66:	5d62                	lw	s10,56(sp)
1c009d68:	5622                	lw	a2,40(sp)
1c009d6a:	fff78713          	addi	a4,a5,-1
1c009d6e:	830d                	srli	a4,a4,0x3
1c009d70:	0705                	addi	a4,a4,1
1c009d72:	079d                	addi	a5,a5,7
1c009d74:	46a1                	li	a3,8
1c009d76:	c46a                	sw	s10,8(sp)
1c009d78:	c032                	sw	a2,0(sp)
1c009d7a:	c63a                	sw	a4,12(sp)
1c009d7c:	3cd7e163          	bltu	a5,a3,1c00a13e <mm_unroll_1x8+0x48e>
1c009d80:	47e2                	lw	a5,24(sp)
1c009d82:	38f05363          	blez	a5,1c00a108 <mm_unroll_1x8+0x458>
1c009d86:	4692                	lw	a3,4(sp)
1c009d88:	4722                	lw	a4,8(sp)
1c009d8a:	47d2                	lw	a5,20(sp)
1c009d8c:	4d42                	lw	s10,16(sp)
1c009d8e:	00e68433          	add	s0,a3,a4
1c009d92:	00000e13          	li	t3,0
1c009d96:	008683b3          	add	t2,a3,s0
1c009d9a:	007682b3          	add	t0,a3,t2
1c009d9e:	41a78733          	sub	a4,a5,s10
1c009da2:	4792                	lw	a5,4(sp)
1c009da4:	00568fb3          	add	t6,a3,t0
1c009da8:	01f68f33          	add	t5,a3,t6
1c009dac:	1771                	addi	a4,a4,-4
1c009dae:	8372                	mv	t1,t3
1c009db0:	88f2                	mv	a7,t3
1c009db2:	8872                	mv	a6,t3
1c009db4:	8572                	mv	a0,t3
1c009db6:	85f2                	mv	a1,t3
1c009db8:	8672                	mv	a2,t3
1c009dba:	86f2                	mv	a3,t3
1c009dbc:	4922                	lw	s2,8(sp)
1c009dbe:	01e78eb3          	add	t4,a5,t5
1c009dc2:	8309                	srli	a4,a4,0x2
1c009dc4:	01d789b3          	add	s3,a5,t4
1c009dc8:	84ea                	mv	s1,s10
1c009dca:	0705                	addi	a4,a4,1
1c009dcc:	022740fb          	lp.setup	x1,a4,1c009e10 <mm_unroll_1x8+0x160>
1c009dd0:	0044a78b          	p.lw	a5,4(s1!)
1c009dd4:	00492d8b          	p.lw	s11,4(s2!)
1c009dd8:	00442d0b          	p.lw	s10,4(s0!)
1c009ddc:	0043ac8b          	p.lw	s9,4(t2!)
1c009de0:	0042ac0b          	p.lw	s8,4(t0!)
1c009de4:	004fab8b          	p.lw	s7,4(t6!)
1c009de8:	004f2b0b          	p.lw	s6,4(t5!)
1c009dec:	004eaa8b          	p.lw	s5,4(t4!)
1c009df0:	0049aa0b          	p.lw	s4,4(s3!)
1c009df4:	68fdf6c3          	fmadd.s	a3,s11,a5,a3
1c009df8:	60fd7643          	fmadd.s	a2,s10,a5,a2
1c009dfc:	58fcf5c3          	fmadd.s	a1,s9,a5,a1
1c009e00:	50fc7543          	fmadd.s	a0,s8,a5,a0
1c009e04:	80fbf843          	fmadd.s	a6,s7,a5,a6
1c009e08:	88fb78c3          	fmadd.s	a7,s6,a5,a7
1c009e0c:	30faf343          	fmadd.s	t1,s5,a5,t1
1c009e10:	e0fa7e43          	fmadd.s	t3,s4,a5,t3
1c009e14:	4d02                	lw	s10,0(sp)
1c009e16:	4732                	lw	a4,12(sp)
1c009e18:	00dd2023          	sw	a3,0(s10)
1c009e1c:	00cd2223          	sw	a2,4(s10)
1c009e20:	46f2                	lw	a3,28(sp)
1c009e22:	4622                	lw	a2,8(sp)
1c009e24:	00bd2423          	sw	a1,8(s10)
1c009e28:	00ad2623          	sw	a0,12(s10)
1c009e2c:	010d2823          	sw	a6,16(s10)
1c009e30:	011d2a23          	sw	a7,20(s10)
1c009e34:	006d2c23          	sw	t1,24(s10)
1c009e38:	01cd2e23          	sw	t3,28(s10)
1c009e3c:	9636                	add	a2,a2,a3
1c009e3e:	020d0d13          	addi	s10,s10,32
1c009e42:	177d                	addi	a4,a4,-1
1c009e44:	c432                	sw	a2,8(sp)
1c009e46:	c06a                	sw	s10,0(sp)
1c009e48:	c63a                	sw	a4,12(sp)
1c009e4a:	fb1d                	bnez	a4,1c009d80 <mm_unroll_1x8+0xd0>
1c009e4c:	5d12                	lw	s10,36(sp)
1c009e4e:	5622                	lw	a2,40(sp)
1c009e50:	56f2                	lw	a3,60(sp)
1c009e52:	0d05                	addi	s10,s10,1
1c009e54:	d26a                	sw	s10,36(sp)
1c009e56:	4742                	lw	a4,16(sp)
1c009e58:	4792                	lw	a5,4(sp)
1c009e5a:	4d52                	lw	s10,20(sp)
1c009e5c:	9636                	add	a2,a2,a3
1c009e5e:	d432                	sw	a2,40(sp)
1c009e60:	5692                	lw	a3,36(sp)
1c009e62:	5652                	lw	a2,52(sp)
1c009e64:	973e                	add	a4,a4,a5
1c009e66:	9d3e                	add	s10,s10,a5
1c009e68:	c83a                	sw	a4,16(sp)
1c009e6a:	ca6a                	sw	s10,20(sp)
1c009e6c:	eed61ce3          	bne	a2,a3,1c009d64 <mm_unroll_1x8+0xb4>
1c009e70:	4726                	lw	a4,72(sp)
1c009e72:	c745                	beqz	a4,1c009f1a <mm_unroll_1x8+0x26a>
1c009e74:	5742                	lw	a4,48(sp)
1c009e76:	5782                	lw	a5,32(sp)
1c009e78:	0ae7d163          	ble	a4,a5,1c009f1a <mm_unroll_1x8+0x26a>
1c009e7c:	5d32                	lw	s10,44(sp)
1c009e7e:	4fe2                	lw	t6,24(sp)
1c009e80:	5602                	lw	a2,32(sp)
1c009e82:	02ed07b3          	mul	a5,s10,a4
1c009e86:	89ba                	mv	s3,a4
1c009e88:	00271e93          	slli	t4,a4,0x2
1c009e8c:	52e2                	lw	t0,56(sp)
1c009e8e:	4f06                	lw	t5,64(sp)
1c009e90:	4696                	lw	a3,68(sp)
1c009e92:	4462                	lw	s0,24(sp)
1c009e94:	5a52                	lw	s4,52(sp)
1c009e96:	44b6                	lw	s1,76(sp)
1c009e98:	8936                	mv	s2,a3
1c009e9a:	00f70e33          	add	t3,a4,a5
1c009e9e:	4712                	lw	a4,4(sp)
1c009ea0:	43fd0fb3          	p.mac	t6,s10,t6
1c009ea4:	002e1313          	slli	t1,t3,0x2
1c009ea8:	9336                	add	t1,t1,a3
1c009eaa:	83ba                	mv	t2,a4
1c009eac:	8ab2                	mv	s5,a2
1c009eae:	42e602b3          	p.mac	t0,a2,a4
1c009eb2:	0f8a                	slli	t6,t6,0x2
1c009eb4:	42ed0f33          	p.mac	t5,s10,a4
1c009eb8:	4d06                	lw	s10,64(sp)
1c009eba:	9fea                	add	t6,t6,s10
1c009ebc:	015788b3          	add	a7,a5,s5
1c009ec0:	088a                	slli	a7,a7,0x2
1c009ec2:	98ca                	add	a7,a7,s2
1c009ec4:	41130833          	sub	a6,t1,a7
1c009ec8:	1871                	addi	a6,a6,-4
1c009eca:	00285813          	srli	a6,a6,0x2
1c009ece:	8b16                	mv	s6,t0
1c009ed0:	0805                	addi	a6,a6,1
1c009ed2:	0001                	nop
1c009ed4:	01a8407b          	lp.setup	x0,a6,1c009f08 <mm_unroll_1x8+0x258>
1c009ed8:	00000713          	li	a4,0
1c009edc:	02805463          	blez	s0,1c009f04 <mm_unroll_1x8+0x254>
1c009ee0:	41ef87b3          	sub	a5,t6,t5
1c009ee4:	17f1                	addi	a5,a5,-4
1c009ee6:	00000713          	li	a4,0
1c009eea:	8389                	srli	a5,a5,0x2
1c009eec:	865a                	mv	a2,s6
1c009eee:	86fa                	mv	a3,t5
1c009ef0:	0785                	addi	a5,a5,1
1c009ef2:	0001                	nop
1c009ef4:	0067c0fb          	lp.setup	x1,a5,1c009f00 <mm_unroll_1x8+0x250>
1c009ef8:	0046a50b          	p.lw	a0,4(a3!)
1c009efc:	0046258b          	p.lw	a1,4(a2!)
1c009f00:	70b57743          	fmadd.s	a4,a0,a1,a4
1c009f04:	00e8a22b          	p.sw	a4,4(a7!)
1c009f08:	9b1e                	add	s6,s6,t2
1c009f0a:	0485                	addi	s1,s1,1
1c009f0c:	87f2                	mv	a5,t3
1c009f0e:	9376                	add	t1,t1,t4
1c009f10:	9f1e                	add	t5,t5,t2
1c009f12:	9f9e                	add	t6,t6,t2
1c009f14:	9e4e                	add	t3,t3,s3
1c009f16:	fa9a13e3          	bne	s4,s1,1c009ebc <mm_unroll_1x8+0x20c>
1c009f1a:	5476                	lw	s0,124(sp)
1c009f1c:	54e6                	lw	s1,120(sp)
1c009f1e:	5956                	lw	s2,116(sp)
1c009f20:	59c6                	lw	s3,112(sp)
1c009f22:	5a36                	lw	s4,108(sp)
1c009f24:	5aa6                	lw	s5,104(sp)
1c009f26:	5b16                	lw	s6,100(sp)
1c009f28:	5b86                	lw	s7,96(sp)
1c009f2a:	4c76                	lw	s8,92(sp)
1c009f2c:	4ce6                	lw	s9,88(sp)
1c009f2e:	4d56                	lw	s10,84(sp)
1c009f30:	4dc6                	lw	s11,80(sp)
1c009f32:	6109                	addi	sp,sp,128
1c009f34:	8082                	ret
1c009f36:	fefd52e3          	ble	a5,s10,1c009f1a <mm_unroll_1x8+0x26a>
1c009f3a:	56c2                	lw	a3,48(sp)
1c009f3c:	c406b6b3          	p.bclr	a3,a3,2,0
1c009f40:	d036                	sw	a3,32(sp)
1c009f42:	10068763          	beqz	a3,1c00a050 <mm_unroll_1x8+0x3a0>
1c009f46:	5742                	lw	a4,48(sp)
1c009f48:	47e2                	lw	a5,24(sp)
1c009f4a:	4296                	lw	t0,68(sp)
1c009f4c:	8dea                	mv	s11,s10
1c009f4e:	00271b13          	slli	s6,a4,0x2
1c009f52:	436d82b3          	p.mac	t0,s11,s6
1c009f56:	8d3e                	mv	s10,a5
1c009f58:	4606                	lw	a2,64(sp)
1c009f5a:	00279f93          	slli	t6,a5,0x2
1c009f5e:	847e                	mv	s0,t6
1c009f60:	8cb2                	mv	s9,a2
1c009f62:	83ee                	mv	t2,s11
1c009f64:	8936                	mv	s2,a3
1c009f66:	42fd8d33          	p.mac	s10,s11,a5
1c009f6a:	8496                	mv	s1,t0
1c009f6c:	52d2                	lw	t0,52(sp)
1c009f6e:	43fd8cb3          	p.mac	s9,s11,t6
1c009f72:	5fe2                	lw	t6,56(sp)
1c009f74:	0d0a                	slli	s10,s10,0x2
1c009f76:	9d32                	add	s10,s10,a2
1c009f78:	8dbe                	mv	s11,a5
1c009f7a:	fff90b93          	addi	s7,s2,-1
1c009f7e:	003bdb93          	srli	s7,s7,0x3
1c009f82:	00790793          	addi	a5,s2,7
1c009f86:	4721                	li	a4,8
1c009f88:	8c7e                	mv	s8,t6
1c009f8a:	8aa6                	mv	s5,s1
1c009f8c:	0b85                	addi	s7,s7,1
1c009f8e:	1ae7e663          	bltu	a5,a4,1c00a13a <mm_unroll_1x8+0x48a>
1c009f92:	0001                	nop
1c009f94:	056bc07b          	lp.setup	x0,s7,1c00a040 <mm_unroll_1x8+0x390>
1c009f98:	15b05e63          	blez	s11,1c00a0f4 <mm_unroll_1x8+0x444>
1c009f9c:	00000e93          	li	t4,0
1c009fa0:	419d06b3          	sub	a3,s10,s9
1c009fa4:	16f1                	addi	a3,a3,-4
1c009fa6:	8e76                	mv	t3,t4
1c009fa8:	8376                	mv	t1,t4
1c009faa:	88f6                	mv	a7,t4
1c009fac:	8876                	mv	a6,t4
1c009fae:	8576                	mv	a0,t4
1c009fb0:	85f6                	mv	a1,t4
1c009fb2:	8676                	mv	a2,t4
1c009fb4:	8289                	srli	a3,a3,0x2
1c009fb6:	8a22                	mv	s4,s0
1c009fb8:	87e2                	mv	a5,s8
1c009fba:	841e                	mv	s0,t2
1c009fbc:	8f66                	mv	t5,s9
1c009fbe:	8396                	mv	t2,t0
1c009fc0:	0685                	addi	a3,a3,1
1c009fc2:	89ea                	mv	s3,s10
1c009fc4:	82ee                	mv	t0,s11
1c009fc6:	0001                	nop
1c009fc8:	0236c0fb          	lp.setup	x1,a3,1c00a00e <mm_unroll_1x8+0x35e>
1c009fcc:	004f270b          	p.lw	a4,4(t5!)
1c009fd0:	0047ad03          	lw	s10,4(a5)
1c009fd4:	0007ad83          	lw	s11,0(a5)
1c009fd8:	58ed75c3          	fmadd.s	a1,s10,a4,a1
1c009fdc:	0087ad03          	lw	s10,8(a5)
1c009fe0:	60edf643          	fmadd.s	a2,s11,a4,a2
1c009fe4:	50ed7543          	fmadd.s	a0,s10,a4,a0
1c009fe8:	00c7ad03          	lw	s10,12(a5)
1c009fec:	80ed7843          	fmadd.s	a6,s10,a4,a6
1c009ff0:	0107ad03          	lw	s10,16(a5)
1c009ff4:	88ed78c3          	fmadd.s	a7,s10,a4,a7
1c009ff8:	0147ad03          	lw	s10,20(a5)
1c009ffc:	30ed7343          	fmadd.s	t1,s10,a4,t1
1c00a000:	0187ad03          	lw	s10,24(a5)
1c00a004:	e0ed7e43          	fmadd.s	t3,s10,a4,t3
1c00a008:	01c7ad03          	lw	s10,28(a5)
1c00a00c:	97da                	add	a5,a5,s6
1c00a00e:	e8ed7ec3          	fmadd.s	t4,s10,a4,t4
1c00a012:	8d96                	mv	s11,t0
1c00a014:	8d4e                	mv	s10,s3
1c00a016:	829e                	mv	t0,t2
1c00a018:	83a2                	mv	t2,s0
1c00a01a:	8452                	mv	s0,s4
1c00a01c:	00caa023          	sw	a2,0(s5)
1c00a020:	00baa223          	sw	a1,4(s5)
1c00a024:	00aaa423          	sw	a0,8(s5)
1c00a028:	010aa623          	sw	a6,12(s5)
1c00a02c:	011aa823          	sw	a7,16(s5)
1c00a030:	006aaa23          	sw	t1,20(s5)
1c00a034:	01caac23          	sw	t3,24(s5)
1c00a038:	01daae23          	sw	t4,28(s5)
1c00a03c:	020c0c13          	addi	s8,s8,32
1c00a040:	020a8a93          	addi	s5,s5,32
1c00a044:	0385                	addi	t2,t2,1
1c00a046:	94da                	add	s1,s1,s6
1c00a048:	9ca2                	add	s9,s9,s0
1c00a04a:	9d22                	add	s10,s10,s0
1c00a04c:	f27297e3          	bne	t0,t2,1c009f7a <mm_unroll_1x8+0x2ca>
1c00a050:	46a6                	lw	a3,72(sp)
1c00a052:	ec0684e3          	beqz	a3,1c009f1a <mm_unroll_1x8+0x26a>
1c00a056:	5642                	lw	a2,48(sp)
1c00a058:	5682                	lw	a3,32(sp)
1c00a05a:	ecc6d0e3          	ble	a2,a3,1c009f1a <mm_unroll_1x8+0x26a>
1c00a05e:	5732                	lw	a4,44(sp)
1c00a060:	48e2                	lw	a7,24(sp)
1c00a062:	00261f13          	slli	t5,a2,0x2
1c00a066:	8932                	mv	s2,a2
1c00a068:	02c70333          	mul	t1,a4,a2
1c00a06c:	4606                	lw	a2,64(sp)
1c00a06e:	87c6                	mv	a5,a7
1c00a070:	00279e93          	slli	t4,a5,0x2
1c00a074:	82b2                	mv	t0,a2
1c00a076:	5d62                	lw	s10,56(sp)
1c00a078:	4496                	lw	s1,68(sp)
1c00a07a:	59d2                	lw	s3,52(sp)
1c00a07c:	4436                	lw	s0,76(sp)
1c00a07e:	00269e13          	slli	t3,a3,0x2
1c00a082:	431708b3          	p.mac	a7,a4,a7
1c00a086:	9e6a                	add	t3,t3,s10
1c00a088:	01ed0fb3          	add	t6,s10,t5
1c00a08c:	83be                	mv	t2,a5
1c00a08e:	8a36                	mv	s4,a3
1c00a090:	43d702b3          	p.mac	t0,a4,t4
1c00a094:	088a                	slli	a7,a7,0x2
1c00a096:	98b2                	add	a7,a7,a2
1c00a098:	41cf8833          	sub	a6,t6,t3
1c00a09c:	01430ab3          	add	s5,t1,s4
1c00a0a0:	1871                	addi	a6,a6,-4
1c00a0a2:	0a8a                	slli	s5,s5,0x2
1c00a0a4:	00285813          	srli	a6,a6,0x2
1c00a0a8:	9aa6                	add	s5,s5,s1
1c00a0aa:	8b72                	mv	s6,t3
1c00a0ac:	0805                	addi	a6,a6,1
1c00a0ae:	0001                	nop
1c00a0b0:	01a8407b          	lp.setup	x0,a6,1c00a0e4 <mm_unroll_1x8+0x434>
1c00a0b4:	00000713          	li	a4,0
1c00a0b8:	02705463          	blez	t2,1c00a0e0 <mm_unroll_1x8+0x430>
1c00a0bc:	405887b3          	sub	a5,a7,t0
1c00a0c0:	17f1                	addi	a5,a5,-4
1c00a0c2:	00000713          	li	a4,0
1c00a0c6:	8389                	srli	a5,a5,0x2
1c00a0c8:	865a                	mv	a2,s6
1c00a0ca:	8696                	mv	a3,t0
1c00a0cc:	0785                	addi	a5,a5,1
1c00a0ce:	0001                	nop
1c00a0d0:	0067c0fb          	lp.setup	x1,a5,1c00a0dc <mm_unroll_1x8+0x42c>
1c00a0d4:	0046a50b          	p.lw	a0,4(a3!)
1c00a0d8:	21e6758b          	p.lw	a1,t5(a2!)
1c00a0dc:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00a0e0:	00eaa22b          	p.sw	a4,4(s5!)
1c00a0e4:	0b11                	addi	s6,s6,4
1c00a0e6:	0405                	addi	s0,s0,1
1c00a0e8:	934a                	add	t1,t1,s2
1c00a0ea:	92f6                	add	t0,t0,t4
1c00a0ec:	98f6                	add	a7,a7,t4
1c00a0ee:	fa8995e3          	bne	s3,s0,1c00a098 <mm_unroll_1x8+0x3e8>
1c00a0f2:	b525                	j	1c009f1a <mm_unroll_1x8+0x26a>
1c00a0f4:	00000e93          	li	t4,0
1c00a0f8:	8e76                	mv	t3,t4
1c00a0fa:	8376                	mv	t1,t4
1c00a0fc:	88f6                	mv	a7,t4
1c00a0fe:	8876                	mv	a6,t4
1c00a100:	8576                	mv	a0,t4
1c00a102:	85f6                	mv	a1,t4
1c00a104:	8676                	mv	a2,t4
1c00a106:	bf19                	j	1c00a01c <mm_unroll_1x8+0x36c>
1c00a108:	00000e13          	li	t3,0
1c00a10c:	8372                	mv	t1,t3
1c00a10e:	88f2                	mv	a7,t3
1c00a110:	8872                	mv	a6,t3
1c00a112:	8572                	mv	a0,t3
1c00a114:	85f2                	mv	a1,t3
1c00a116:	8672                	mv	a2,t3
1c00a118:	86f2                	mv	a3,t3
1c00a11a:	b9ed                	j	1c009e14 <mm_unroll_1x8+0x164>
1c00a11c:	5476                	lw	s0,124(sp)
1c00a11e:	54e6                	lw	s1,120(sp)
1c00a120:	5956                	lw	s2,116(sp)
1c00a122:	59c6                	lw	s3,112(sp)
1c00a124:	5a36                	lw	s4,108(sp)
1c00a126:	5aa6                	lw	s5,104(sp)
1c00a128:	5b16                	lw	s6,100(sp)
1c00a12a:	5b86                	lw	s7,96(sp)
1c00a12c:	4c76                	lw	s8,92(sp)
1c00a12e:	4ce6                	lw	s9,88(sp)
1c00a130:	4d56                	lw	s10,84(sp)
1c00a132:	4dc6                	lw	s11,80(sp)
1c00a134:	6109                	addi	sp,sp,128
1c00a136:	fdaff06f          	j	1c009910 <mm_unroll_1x4>
1c00a13a:	4b85                	li	s7,1
1c00a13c:	bd99                	j	1c009f92 <mm_unroll_1x8+0x2e2>
1c00a13e:	4685                	li	a3,1
1c00a140:	c636                	sw	a3,12(sp)
1c00a142:	b93d                	j	1c009d80 <mm_unroll_1x8+0xd0>
1c00a144:	0000                	unimp
	...

1c00a148 <mm_unroll_2x1>:
1c00a148:	715d                	addi	sp,sp,-80
1c00a14a:	da5a                	sw	s6,52(sp)
1c00a14c:	00c52b03          	lw	s6,12(a0)
1c00a150:	dc56                	sw	s5,56(sp)
1c00a152:	c6a2                	sw	s0,76(sp)
1c00a154:	c4a6                	sw	s1,72(sp)
1c00a156:	c2ca                	sw	s2,68(sp)
1c00a158:	c0ce                	sw	s3,64(sp)
1c00a15a:	de52                	sw	s4,60(sp)
1c00a15c:	d85e                	sw	s7,48(sp)
1c00a15e:	d662                	sw	s8,44(sp)
1c00a160:	d466                	sw	s9,40(sp)
1c00a162:	d26a                	sw	s10,36(sp)
1c00a164:	d06e                	sw	s11,32(sp)
1c00a166:	473d                	li	a4,15
1c00a168:	c00b3ab3          	p.bclr	s5,s6,0,0
1c00a16c:	f14026f3          	csrr	a3,mhartid
1c00a170:	31575963          	ble	s5,a4,1c00a482 <mm_unroll_2x1+0x33a>
1c00a174:	007a8713          	addi	a4,s5,7
1c00a178:	f456b6b3          	p.bclr	a3,a3,26,5
1c00a17c:	870d                	srai	a4,a4,0x3
1c00a17e:	02e68933          	mul	s2,a3,a4
1c00a182:	4110                	lw	a2,0(a0)
1c00a184:	ca36                	sw	a3,20(sp)
1c00a186:	4d14                	lw	a3,24(a0)
1c00a188:	c632                	sw	a2,12(sp)
1c00a18a:	4510                	lw	a2,8(a0)
1c00a18c:	415b07b3          	sub	a5,s6,s5
1c00a190:	ce3e                	sw	a5,28(sp)
1c00a192:	c832                	sw	a2,16(sp)
1c00a194:	4910                	lw	a2,16(a0)
1c00a196:	974a                	add	a4,a4,s2
1c00a198:	4140                	lw	s0,4(a0)
1c00a19a:	c432                	sw	a2,8(sp)
1c00a19c:	01452e83          	lw	t4,20(a0)
1c00a1a0:	05574a33          	p.min	s4,a4,s5
1c00a1a4:	16069763          	bnez	a3,1c00a312 <mm_unroll_2x1+0x1ca>
1c00a1a8:	0b495863          	ble	s4,s2,1c00a258 <mm_unroll_2x1+0x110>
1c00a1ac:	0ac05663          	blez	a2,1c00a258 <mm_unroll_2x1+0x110>
1c00a1b0:	00261d93          	slli	s11,a2,0x2
1c00a1b4:	00361d13          	slli	s10,a2,0x3
1c00a1b8:	84f6                	mv	s1,t4
1c00a1ba:	4632                	lw	a2,12(sp)
1c00a1bc:	432e84b3          	p.mac	s1,t4,s2
1c00a1c0:	4c42                	lw	s8,16(sp)
1c00a1c2:	4bc2                	lw	s7,16(sp)
1c00a1c4:	00190713          	addi	a4,s2,1
1c00a1c8:	002e9793          	slli	a5,t4,0x2
1c00a1cc:	83b2                	mv	t2,a2
1c00a1ce:	008d86b3          	add	a3,s11,s0
1c00a1d2:	cc5a                	sw	s6,24(sp)
1c00a1d4:	003e9c93          	slli	s9,t4,0x3
1c00a1d8:	8b36                	mv	s6,a3
1c00a1da:	432d8c33          	p.mac	s8,s11,s2
1c00a1de:	048a                	slli	s1,s1,0x2
1c00a1e0:	94b2                	add	s1,s1,a2
1c00a1e2:	89a6                	mv	s3,s1
1c00a1e4:	43b70bb3          	p.mac	s7,a4,s11
1c00a1e8:	432783b3          	p.mac	t2,a5,s2
1c00a1ec:	408b0e33          	sub	t3,s6,s0
1c00a1f0:	1e71                	addi	t3,t3,-4
1c00a1f2:	002e5e13          	srli	t3,t3,0x2
1c00a1f6:	8f22                	mv	t5,s0
1c00a1f8:	82de                	mv	t0,s7
1c00a1fa:	8fe2                	mv	t6,s8
1c00a1fc:	0e05                	addi	t3,t3,1
1c00a1fe:	0001                	nop
1c00a200:	022e407b          	lp.setup	x0,t3,1c00a244 <mm_unroll_2x1+0xfc>
1c00a204:	00000613          	li	a2,0
1c00a208:	86b2                	mv	a3,a2
1c00a20a:	03d05963          	blez	t4,1c00a23c <mm_unroll_2x1+0xf4>
1c00a20e:	00000613          	li	a2,0
1c00a212:	407987b3          	sub	a5,s3,t2
1c00a216:	17f1                	addi	a5,a5,-4
1c00a218:	86b2                	mv	a3,a2
1c00a21a:	8389                	srli	a5,a5,0x2
1c00a21c:	8826                	mv	a6,s1
1c00a21e:	851e                	mv	a0,t2
1c00a220:	85fa                	mv	a1,t5
1c00a222:	0785                	addi	a5,a5,1
1c00a224:	00a7c0fb          	lp.setup	x1,a5,1c00a238 <mm_unroll_2x1+0xf0>
1c00a228:	21b5f70b          	p.lw	a4,s11(a1!)
1c00a22c:	0045230b          	p.lw	t1,4(a0!)
1c00a230:	0048288b          	p.lw	a7,4(a6!)
1c00a234:	68e376c3          	fmadd.s	a3,t1,a4,a3
1c00a238:	60e8f643          	fmadd.s	a2,a7,a4,a2
1c00a23c:	00dfa22b          	p.sw	a3,4(t6!)
1c00a240:	00c2a22b          	p.sw	a2,4(t0!)
1c00a244:	0f11                	addi	t5,t5,4
1c00a246:	0909                	addi	s2,s2,2
1c00a248:	9c6a                	add	s8,s8,s10
1c00a24a:	9bea                	add	s7,s7,s10
1c00a24c:	93e6                	add	t2,t2,s9
1c00a24e:	94e6                	add	s1,s1,s9
1c00a250:	99e6                	add	s3,s3,s9
1c00a252:	f9494de3          	blt	s2,s4,1c00a1ec <mm_unroll_2x1+0xa4>
1c00a256:	4b62                	lw	s6,24(sp)
1c00a258:	46f2                	lw	a3,28(sp)
1c00a25a:	08d05e63          	blez	a3,1c00a2f6 <mm_unroll_2x1+0x1ae>
1c00a25e:	47a2                	lw	a5,8(sp)
1c00a260:	4652                	lw	a2,20(sp)
1c00a262:	46a2                	lw	a3,8(sp)
1c00a264:	00778713          	addi	a4,a5,7
1c00a268:	41f75793          	srai	a5,a4,0x1f
1c00a26c:	f837b7b3          	p.bclr	a5,a5,28,3
1c00a270:	06e7a7db          	p.addn	a5,a5,a4,3
1c00a274:	02f60733          	mul	a4,a2,a5
1c00a278:	97ba                	add	a5,a5,a4
1c00a27a:	04d7c7b3          	p.min	a5,a5,a3
1c00a27e:	06f75c63          	ble	a5,a4,1c00a2f6 <mm_unroll_2x1+0x1ae>
1c00a282:	076ada63          	ble	s6,s5,1c00a2f6 <mm_unroll_2x1+0x1ae>
1c00a286:	8fba                	mv	t6,a4
1c00a288:	42b2                	lw	t0,12(sp)
1c00a28a:	43568fb3          	p.mac	t6,a3,s5
1c00a28e:	002e9f13          	slli	t5,t4,0x2
1c00a292:	070a                	slli	a4,a4,0x2
1c00a294:	00e40e33          	add	t3,s0,a4
1c00a298:	4742                	lw	a4,16(sp)
1c00a29a:	078a                	slli	a5,a5,0x2
1c00a29c:	943e                	add	s0,s0,a5
1c00a29e:	00269393          	slli	t2,a3,0x2
1c00a2a2:	43ea82b3          	p.mac	t0,s5,t5
1c00a2a6:	0f8a                	slli	t6,t6,0x2
1c00a2a8:	9fba                	add	t6,t6,a4
1c00a2aa:	8716                	mv	a4,t0
1c00a2ac:	837e                	mv	t1,t6
1c00a2ae:	415b0833          	sub	a6,s6,s5
1c00a2b2:	0001                	nop
1c00a2b4:	01c8407b          	lp.setup	x0,a6,1c00a2ec <mm_unroll_2x1+0x1a4>
1c00a2b8:	00000693          	li	a3,0
1c00a2bc:	00ef08b3          	add	a7,t5,a4
1c00a2c0:	03d05463          	blez	t4,1c00a2e8 <mm_unroll_2x1+0x1a0>
1c00a2c4:	00ef08b3          	add	a7,t5,a4
1c00a2c8:	40e887b3          	sub	a5,a7,a4
1c00a2cc:	17f1                	addi	a5,a5,-4
1c00a2ce:	00000693          	li	a3,0
1c00a2d2:	8389                	srli	a5,a5,0x2
1c00a2d4:	8672                	mv	a2,t3
1c00a2d6:	0785                	addi	a5,a5,1
1c00a2d8:	0067c0fb          	lp.setup	x1,a5,1c00a2e4 <mm_unroll_2x1+0x19c>
1c00a2dc:	0047250b          	p.lw	a0,4(a4!)
1c00a2e0:	2076758b          	p.lw	a1,t2(a2!)
1c00a2e4:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00a2e8:	00d363ab          	p.sw	a3,t2(t1!)
1c00a2ec:	8746                	mv	a4,a7
1c00a2ee:	0e11                	addi	t3,t3,4
1c00a2f0:	0f91                	addi	t6,t6,4
1c00a2f2:	fbc41ce3          	bne	s0,t3,1c00a2aa <mm_unroll_2x1+0x162>
1c00a2f6:	4436                	lw	s0,76(sp)
1c00a2f8:	44a6                	lw	s1,72(sp)
1c00a2fa:	4916                	lw	s2,68(sp)
1c00a2fc:	4986                	lw	s3,64(sp)
1c00a2fe:	5a72                	lw	s4,60(sp)
1c00a300:	5ae2                	lw	s5,56(sp)
1c00a302:	5b52                	lw	s6,52(sp)
1c00a304:	5bc2                	lw	s7,48(sp)
1c00a306:	5c32                	lw	s8,44(sp)
1c00a308:	5ca2                	lw	s9,40(sp)
1c00a30a:	5d12                	lw	s10,36(sp)
1c00a30c:	5d82                	lw	s11,32(sp)
1c00a30e:	6161                	addi	sp,sp,80
1c00a310:	8082                	ret
1c00a312:	0b495f63          	ble	s4,s2,1c00a3d0 <mm_unroll_2x1+0x288>
1c00a316:	0ac05d63          	blez	a2,1c00a3d0 <mm_unroll_2x1+0x288>
1c00a31a:	00261713          	slli	a4,a2,0x2
1c00a31e:	40e607b3          	sub	a5,a2,a4
1c00a322:	00279b93          	slli	s7,a5,0x2
1c00a326:	87b2                	mv	a5,a2
1c00a328:	40c00db3          	neg	s11,a2
1c00a32c:	432607b3          	p.mac	a5,a2,s2
1c00a330:	00361d13          	slli	s10,a2,0x3
1c00a334:	4642                	lw	a2,16(sp)
1c00a336:	00190693          	addi	a3,s2,1
1c00a33a:	83f6                	mv	t2,t4
1c00a33c:	8c32                	mv	s8,a2
1c00a33e:	002e9993          	slli	s3,t4,0x2
1c00a342:	003e9c93          	slli	s9,t4,0x3
1c00a346:	0d8a                	slli	s11,s11,0x2
1c00a348:	cc5a                	sw	s6,24(sp)
1c00a34a:	42d70c33          	p.mac	s8,a4,a3
1c00a34e:	46b2                	lw	a3,12(sp)
1c00a350:	078a                	slli	a5,a5,0x2
1c00a352:	97b2                	add	a5,a5,a2
1c00a354:	84b6                	mv	s1,a3
1c00a356:	432e83b3          	p.mac	t2,t4,s2
1c00a35a:	432984b3          	p.mac	s1,s3,s2
1c00a35e:	038a                	slli	t2,t2,0x2
1c00a360:	93b6                	add	t2,t2,a3
1c00a362:	01b78b33          	add	s6,a5,s11
1c00a366:	41678e33          	sub	t3,a5,s6
1c00a36a:	1e71                	addi	t3,t3,-4
1c00a36c:	002e5e13          	srli	t3,t3,0x2
1c00a370:	86a2                	mv	a3,s0
1c00a372:	82e2                	mv	t0,s8
1c00a374:	8fda                	mv	t6,s6
1c00a376:	0e05                	addi	t3,t3,1
1c00a378:	0fd05763          	blez	t4,1c00a466 <mm_unroll_2x1+0x31e>
1c00a37c:	00000593          	li	a1,0
1c00a380:	01368533          	add	a0,a3,s3
1c00a384:	40d507b3          	sub	a5,a0,a3
1c00a388:	17f1                	addi	a5,a5,-4
1c00a38a:	862e                	mv	a2,a1
1c00a38c:	8389                	srli	a5,a5,0x2
1c00a38e:	889e                	mv	a7,t2
1c00a390:	8826                	mv	a6,s1
1c00a392:	0785                	addi	a5,a5,1
1c00a394:	00a7c0fb          	lp.setup	x1,a5,1c00a3a8 <mm_unroll_2x1+0x260>
1c00a398:	0046a70b          	p.lw	a4,4(a3!)
1c00a39c:	00482f0b          	p.lw	t5,4(a6!)
1c00a3a0:	0048a30b          	p.lw	t1,4(a7!)
1c00a3a4:	60ef7643          	fmadd.s	a2,t5,a4,a2
1c00a3a8:	58e375c3          	fmadd.s	a1,t1,a4,a1
1c00a3ac:	00cfa22b          	p.sw	a2,4(t6!)
1c00a3b0:	00b2a22b          	p.sw	a1,4(t0!)
1c00a3b4:	86aa                	mv	a3,a0
1c00a3b6:	1e7d                	addi	t3,t3,-1
1c00a3b8:	fc0e10e3          	bnez	t3,1c00a378 <mm_unroll_2x1+0x230>
1c00a3bc:	0909                	addi	s2,s2,2
1c00a3be:	417b07b3          	sub	a5,s6,s7
1c00a3c2:	9c6a                	add	s8,s8,s10
1c00a3c4:	94e6                	add	s1,s1,s9
1c00a3c6:	93e6                	add	t2,t2,s9
1c00a3c8:	f9494de3          	blt	s2,s4,1c00a362 <mm_unroll_2x1+0x21a>
1c00a3cc:	4b62                	lw	s6,24(sp)
1c00a3ce:	47f2                	lw	a5,28(sp)
1c00a3d0:	f2f053e3          	blez	a5,1c00a2f6 <mm_unroll_2x1+0x1ae>
1c00a3d4:	4622                	lw	a2,8(sp)
1c00a3d6:	46d2                	lw	a3,20(sp)
1c00a3d8:	00760713          	addi	a4,a2,7
1c00a3dc:	41f75793          	srai	a5,a4,0x1f
1c00a3e0:	f837b7b3          	p.bclr	a5,a5,28,3
1c00a3e4:	06e7a7db          	p.addn	a5,a5,a4,3
1c00a3e8:	02f68fb3          	mul	t6,a3,a5
1c00a3ec:	97fe                	add	a5,a5,t6
1c00a3ee:	04c7c2b3          	p.min	t0,a5,a2
1c00a3f2:	f05fd2e3          	ble	t0,t6,1c00a2f6 <mm_unroll_2x1+0x1ae>
1c00a3f6:	f16ad0e3          	ble	s6,s5,1c00a2f6 <mm_unroll_2x1+0x1ae>
1c00a3fa:	8f7e                	mv	t5,t6
1c00a3fc:	43b2                	lw	t2,12(sp)
1c00a3fe:	43560f33          	p.mac	t5,a2,s5
1c00a402:	002e9e13          	slli	t3,t4,0x2
1c00a406:	4742                	lw	a4,16(sp)
1c00a408:	00261493          	slli	s1,a2,0x2
1c00a40c:	43cf8433          	p.mac	s0,t6,t3
1c00a410:	0f0a                	slli	t5,t5,0x2
1c00a412:	9f3a                	add	t5,t5,a4
1c00a414:	43ca83b3          	p.mac	t2,s5,t3
1c00a418:	871e                	mv	a4,t2
1c00a41a:	837a                	mv	t1,t5
1c00a41c:	415b0833          	sub	a6,s6,s5
1c00a420:	01c8407b          	lp.setup	x0,a6,1c00a458 <mm_unroll_2x1+0x310>
1c00a424:	00000693          	li	a3,0
1c00a428:	00ee08b3          	add	a7,t3,a4
1c00a42c:	03d05463          	blez	t4,1c00a454 <mm_unroll_2x1+0x30c>
1c00a430:	00ee08b3          	add	a7,t3,a4
1c00a434:	40e887b3          	sub	a5,a7,a4
1c00a438:	17f1                	addi	a5,a5,-4
1c00a43a:	00000693          	li	a3,0
1c00a43e:	8389                	srli	a5,a5,0x2
1c00a440:	8622                	mv	a2,s0
1c00a442:	0785                	addi	a5,a5,1
1c00a444:	0067c0fb          	lp.setup	x1,a5,1c00a450 <mm_unroll_2x1+0x308>
1c00a448:	0047250b          	p.lw	a0,4(a4!)
1c00a44c:	0046258b          	p.lw	a1,4(a2!)
1c00a450:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00a454:	00d364ab          	p.sw	a3,s1(t1!)
1c00a458:	8746                	mv	a4,a7
1c00a45a:	0f85                	addi	t6,t6,1
1c00a45c:	0f11                	addi	t5,t5,4
1c00a45e:	9472                	add	s0,s0,t3
1c00a460:	fbf29ce3          	bne	t0,t6,1c00a418 <mm_unroll_2x1+0x2d0>
1c00a464:	bd49                	j	1c00a2f6 <mm_unroll_2x1+0x1ae>
1c00a466:	00000593          	li	a1,0
1c00a46a:	01368533          	add	a0,a3,s3
1c00a46e:	86aa                	mv	a3,a0
1c00a470:	862e                	mv	a2,a1
1c00a472:	00cfa22b          	p.sw	a2,4(t6!)
1c00a476:	00b2a22b          	p.sw	a1,4(t0!)
1c00a47a:	1e7d                	addi	t3,t3,-1
1c00a47c:	ee0e1ee3          	bnez	t3,1c00a378 <mm_unroll_2x1+0x230>
1c00a480:	bf35                	j	1c00a3bc <mm_unroll_2x1+0x274>
1c00a482:	4436                	lw	s0,76(sp)
1c00a484:	44a6                	lw	s1,72(sp)
1c00a486:	4916                	lw	s2,68(sp)
1c00a488:	4986                	lw	s3,64(sp)
1c00a48a:	5a72                	lw	s4,60(sp)
1c00a48c:	5ae2                	lw	s5,56(sp)
1c00a48e:	5b52                	lw	s6,52(sp)
1c00a490:	5bc2                	lw	s7,48(sp)
1c00a492:	5c32                	lw	s8,44(sp)
1c00a494:	5ca2                	lw	s9,40(sp)
1c00a496:	5d12                	lw	s10,36(sp)
1c00a498:	5d82                	lw	s11,32(sp)
1c00a49a:	6161                	addi	sp,sp,80
1c00a49c:	9d5fe06f          	j	1c008e70 <mm>
1c00a4a0:	0000                	unimp
	...

1c00a4a4 <mm_unroll_4x1>:
1c00a4a4:	00c52303          	lw	t1,12(a0)
1c00a4a8:	7119                	addi	sp,sp,-128
1c00a4aa:	dea2                	sw	s0,124(sp)
1c00a4ac:	c20335b3          	p.bclr	a1,t1,1,0
1c00a4b0:	dca6                	sw	s1,120(sp)
1c00a4b2:	daca                	sw	s2,116(sp)
1c00a4b4:	d8ce                	sw	s3,112(sp)
1c00a4b6:	d6d2                	sw	s4,108(sp)
1c00a4b8:	d4d6                	sw	s5,104(sp)
1c00a4ba:	d2da                	sw	s6,100(sp)
1c00a4bc:	d0de                	sw	s7,96(sp)
1c00a4be:	cee2                	sw	s8,92(sp)
1c00a4c0:	cce6                	sw	s9,88(sp)
1c00a4c2:	caea                	sw	s10,84(sp)
1c00a4c4:	c8ee                	sw	s11,80(sp)
1c00a4c6:	de1a                	sw	t1,60(sp)
1c00a4c8:	477d                	li	a4,31
1c00a4ca:	d62e                	sw	a1,44(sp)
1c00a4cc:	f14026f3          	csrr	a3,mhartid
1c00a4d0:	46b75763          	ble	a1,a4,1c00a93e <mm_unroll_4x1+0x49a>
1c00a4d4:	00758713          	addi	a4,a1,7
1c00a4d8:	870d                	srai	a4,a4,0x3
1c00a4da:	f456b6b3          	p.bclr	a3,a3,26,5
1c00a4de:	c4b6                	sw	a3,72(sp)
1c00a4e0:	02e686b3          	mul	a3,a3,a4
1c00a4e4:	87aa                	mv	a5,a0
1c00a4e6:	851a                	mv	a0,t1
1c00a4e8:	862e                	mv	a2,a1
1c00a4ea:	8d0d                	sub	a0,a0,a1
1c00a4ec:	0007a803          	lw	a6,0(a5)
1c00a4f0:	0047a883          	lw	a7,4(a5)
1c00a4f4:	0087ae03          	lw	t3,8(a5)
1c00a4f8:	0107a303          	lw	t1,16(a5)
1c00a4fc:	c0c2                	sw	a6,64(sp)
1c00a4fe:	c836                	sw	a3,16(sp)
1c00a500:	45c2                	lw	a1,16(sp)
1c00a502:	4f94                	lw	a3,24(a5)
1c00a504:	d446                	sw	a7,40(sp)
1c00a506:	972e                	add	a4,a4,a1
1c00a508:	04c74733          	p.min	a4,a4,a2
1c00a50c:	c2f2                	sw	t3,68(sp)
1c00a50e:	d81a                	sw	t1,48(sp)
1c00a510:	c6aa                	sw	a0,76(sp)
1c00a512:	da3a                	sw	a4,52(sp)
1c00a514:	4bc4                	lw	s1,20(a5)
1c00a516:	1e069c63          	bnez	a3,1c00a70e <mm_unroll_4x1+0x26a>
1c00a51a:	852e                	mv	a0,a1
1c00a51c:	55d2                	lw	a1,52(sp)
1c00a51e:	12b55663          	ble	a1,a0,1c00a64a <mm_unroll_4x1+0x1a6>
1c00a522:	00149793          	slli	a5,s1,0x1
1c00a526:	00978533          	add	a0,a5,s1
1c00a52a:	12605063          	blez	t1,1c00a64a <mm_unroll_4x1+0x1a6>
1c00a52e:	46c2                	lw	a3,16(sp)
1c00a530:	00231d93          	slli	s11,t1,0x2
1c00a534:	8e46                	mv	t3,a7
1c00a536:	02d48cb3          	mul	s9,s1,a3
1c00a53a:	00431813          	slli	a6,t1,0x4
1c00a53e:	00449893          	slli	a7,s1,0x4
1c00a542:	9e6e                	add	t3,t3,s11
1c00a544:	4306                	lw	t1,64(sp)
1c00a546:	ca42                	sw	a6,20(sp)
1c00a548:	c646                	sw	a7,12(sp)
1c00a54a:	dc72                	sw	t3,56(sp)
1c00a54c:	4816                	lw	a6,68(sp)
1c00a54e:	4896                	lw	a7,68(sp)
1c00a550:	01978c33          	add	s8,a5,s9
1c00a554:	01948d33          	add	s10,s1,s9
1c00a558:	47c2                	lw	a5,16(sp)
1c00a55a:	9caa                	add	s9,s9,a0
1c00a55c:	4e16                	lw	t3,68(sp)
1c00a55e:	4516                	lw	a0,68(sp)
1c00a560:	00168593          	addi	a1,a3,1
1c00a564:	00268613          	addi	a2,a3,2
1c00a568:	068d                	addi	a3,a3,3
1c00a56a:	00249713          	slli	a4,s1,0x2
1c00a56e:	42fd8533          	p.mac	a0,s11,a5
1c00a572:	8b9a                	mv	s7,t1
1c00a574:	0d0a                	slli	s10,s10,0x2
1c00a576:	9d1a                	add	s10,s10,t1
1c00a578:	0c0a                	slli	s8,s8,0x2
1c00a57a:	0c8a                	slli	s9,s9,0x2
1c00a57c:	9c1a                	add	s8,s8,t1
1c00a57e:	9c9a                	add	s9,s9,t1
1c00a580:	8b6a                	mv	s6,s10
1c00a582:	c426                	sw	s1,8(sp)
1c00a584:	43b58833          	p.mac	a6,a1,s11
1c00a588:	d22a                	sw	a0,36(sp)
1c00a58a:	43b608b3          	p.mac	a7,a2,s11
1c00a58e:	d042                	sw	a6,32(sp)
1c00a590:	43b68e33          	p.mac	t3,a3,s11
1c00a594:	ce46                	sw	a7,28(sp)
1c00a596:	42f70bb3          	p.mac	s7,a4,a5
1c00a59a:	cc72                	sw	t3,24(sp)
1c00a59c:	5322                	lw	t1,40(sp)
1c00a59e:	5e62                	lw	t3,56(sp)
1c00a5a0:	4ae2                	lw	s5,24(sp)
1c00a5a2:	4a72                	lw	s4,28(sp)
1c00a5a4:	406e0433          	sub	s0,t3,t1
1c00a5a8:	1471                	addi	s0,s0,-4
1c00a5aa:	5982                	lw	s3,32(sp)
1c00a5ac:	5912                	lw	s2,36(sp)
1c00a5ae:	8009                	srli	s0,s0,0x2
1c00a5b0:	849a                	mv	s1,t1
1c00a5b2:	0405                	addi	s0,s0,1
1c00a5b4:	4322                	lw	t1,8(sp)
1c00a5b6:	32605e63          	blez	t1,1c00a8f2 <mm_unroll_4x1+0x44e>
1c00a5ba:	00000513          	li	a0,0
1c00a5be:	417b0733          	sub	a4,s6,s7
1c00a5c2:	1771                	addi	a4,a4,-4
1c00a5c4:	85aa                	mv	a1,a0
1c00a5c6:	862a                	mv	a2,a0
1c00a5c8:	86aa                	mv	a3,a0
1c00a5ca:	8309                	srli	a4,a4,0x2
1c00a5cc:	8ee6                	mv	t4,s9
1c00a5ce:	8e62                	mv	t3,s8
1c00a5d0:	836a                	mv	t1,s10
1c00a5d2:	88de                	mv	a7,s7
1c00a5d4:	8826                	mv	a6,s1
1c00a5d6:	0705                	addi	a4,a4,1
1c00a5d8:	012740fb          	lp.setup	x1,a4,1c00a5fc <mm_unroll_4x1+0x158>
1c00a5dc:	21b8778b          	p.lw	a5,s11(a6!)
1c00a5e0:	0048a38b          	p.lw	t2,4(a7!)
1c00a5e4:	0043228b          	p.lw	t0,4(t1!)
1c00a5e8:	004e2f8b          	p.lw	t6,4(t3!)
1c00a5ec:	004eaf0b          	p.lw	t5,4(t4!)
1c00a5f0:	68f3f6c3          	fmadd.s	a3,t2,a5,a3
1c00a5f4:	60f2f643          	fmadd.s	a2,t0,a5,a2
1c00a5f8:	58fff5c3          	fmadd.s	a1,t6,a5,a1
1c00a5fc:	50ff7543          	fmadd.s	a0,t5,a5,a0
1c00a600:	00d9222b          	p.sw	a3,4(s2!)
1c00a604:	00c9a22b          	p.sw	a2,4(s3!)
1c00a608:	00ba222b          	p.sw	a1,4(s4!)
1c00a60c:	00aaa22b          	p.sw	a0,4(s5!)
1c00a610:	0491                	addi	s1,s1,4
1c00a612:	147d                	addi	s0,s0,-1
1c00a614:	f045                	bnez	s0,1c00a5b4 <mm_unroll_4x1+0x110>
1c00a616:	4652                	lw	a2,20(sp)
1c00a618:	4542                	lw	a0,16(sp)
1c00a61a:	5592                	lw	a1,36(sp)
1c00a61c:	5682                	lw	a3,32(sp)
1c00a61e:	4772                	lw	a4,28(sp)
1c00a620:	47e2                	lw	a5,24(sp)
1c00a622:	4832                	lw	a6,12(sp)
1c00a624:	58d2                	lw	a7,52(sp)
1c00a626:	0511                	addi	a0,a0,4
1c00a628:	95b2                	add	a1,a1,a2
1c00a62a:	96b2                	add	a3,a3,a2
1c00a62c:	9732                	add	a4,a4,a2
1c00a62e:	97b2                	add	a5,a5,a2
1c00a630:	c82a                	sw	a0,16(sp)
1c00a632:	d22e                	sw	a1,36(sp)
1c00a634:	d036                	sw	a3,32(sp)
1c00a636:	ce3a                	sw	a4,28(sp)
1c00a638:	cc3e                	sw	a5,24(sp)
1c00a63a:	9bc2                	add	s7,s7,a6
1c00a63c:	9d42                	add	s10,s10,a6
1c00a63e:	9c42                	add	s8,s8,a6
1c00a640:	9cc2                	add	s9,s9,a6
1c00a642:	9b42                	add	s6,s6,a6
1c00a644:	f5154ce3          	blt	a0,a7,1c00a59c <mm_unroll_4x1+0xf8>
1c00a648:	44a2                	lw	s1,8(sp)
1c00a64a:	4536                	lw	a0,76(sp)
1c00a64c:	0aa05363          	blez	a0,1c00a6f2 <mm_unroll_4x1+0x24e>
1c00a650:	55c2                	lw	a1,48(sp)
1c00a652:	4626                	lw	a2,72(sp)
1c00a654:	00758713          	addi	a4,a1,7
1c00a658:	41f75793          	srai	a5,a4,0x1f
1c00a65c:	f837b7b3          	p.bclr	a5,a5,28,3
1c00a660:	06e7a7db          	p.addn	a5,a5,a4,3
1c00a664:	02f60733          	mul	a4,a2,a5
1c00a668:	97ba                	add	a5,a5,a4
1c00a66a:	04b7c7b3          	p.min	a5,a5,a1
1c00a66e:	08f75263          	ble	a5,a4,1c00a6f2 <mm_unroll_4x1+0x24e>
1c00a672:	56f2                	lw	a3,60(sp)
1c00a674:	5832                	lw	a6,44(sp)
1c00a676:	06d85e63          	ble	a3,a6,1c00a6f2 <mm_unroll_4x1+0x24e>
1c00a67a:	8fba                	mv	t6,a4
1c00a67c:	4286                	lw	t0,64(sp)
1c00a67e:	43058fb3          	p.mac	t6,a1,a6
1c00a682:	00249e93          	slli	t4,s1,0x2
1c00a686:	58a2                	lw	a7,40(sp)
1c00a688:	4316                	lw	t1,68(sp)
1c00a68a:	070a                	slli	a4,a4,0x2
1c00a68c:	078a                	slli	a5,a5,0x2
1c00a68e:	00e88e33          	add	t3,a7,a4
1c00a692:	00f88f33          	add	t5,a7,a5
1c00a696:	00259393          	slli	t2,a1,0x2
1c00a69a:	8436                	mv	s0,a3
1c00a69c:	43d802b3          	p.mac	t0,a6,t4
1c00a6a0:	0f8a                	slli	t6,t6,0x2
1c00a6a2:	9f9a                	add	t6,t6,t1
1c00a6a4:	8942                	mv	s2,a6
1c00a6a6:	8716                	mv	a4,t0
1c00a6a8:	837e                	mv	t1,t6
1c00a6aa:	41240833          	sub	a6,s0,s2
1c00a6ae:	0001                	nop
1c00a6b0:	01c8407b          	lp.setup	x0,a6,1c00a6e8 <mm_unroll_4x1+0x244>
1c00a6b4:	00000693          	li	a3,0
1c00a6b8:	00ee88b3          	add	a7,t4,a4
1c00a6bc:	02905463          	blez	s1,1c00a6e4 <mm_unroll_4x1+0x240>
1c00a6c0:	00ee88b3          	add	a7,t4,a4
1c00a6c4:	40e887b3          	sub	a5,a7,a4
1c00a6c8:	17f1                	addi	a5,a5,-4
1c00a6ca:	00000693          	li	a3,0
1c00a6ce:	8389                	srli	a5,a5,0x2
1c00a6d0:	8672                	mv	a2,t3
1c00a6d2:	0785                	addi	a5,a5,1
1c00a6d4:	0067c0fb          	lp.setup	x1,a5,1c00a6e0 <mm_unroll_4x1+0x23c>
1c00a6d8:	0047250b          	p.lw	a0,4(a4!)
1c00a6dc:	2076758b          	p.lw	a1,t2(a2!)
1c00a6e0:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00a6e4:	00d363ab          	p.sw	a3,t2(t1!)
1c00a6e8:	8746                	mv	a4,a7
1c00a6ea:	0e11                	addi	t3,t3,4
1c00a6ec:	0f91                	addi	t6,t6,4
1c00a6ee:	fbcf1ce3          	bne	t5,t3,1c00a6a6 <mm_unroll_4x1+0x202>
1c00a6f2:	5476                	lw	s0,124(sp)
1c00a6f4:	54e6                	lw	s1,120(sp)
1c00a6f6:	5956                	lw	s2,116(sp)
1c00a6f8:	59c6                	lw	s3,112(sp)
1c00a6fa:	5a36                	lw	s4,108(sp)
1c00a6fc:	5aa6                	lw	s5,104(sp)
1c00a6fe:	5b16                	lw	s6,100(sp)
1c00a700:	5b86                	lw	s7,96(sp)
1c00a702:	4c76                	lw	s8,92(sp)
1c00a704:	4ce6                	lw	s9,88(sp)
1c00a706:	4d56                	lw	s10,84(sp)
1c00a708:	4dc6                	lw	s11,80(sp)
1c00a70a:	6109                	addi	sp,sp,128
1c00a70c:	8082                	ret
1c00a70e:	14e5d163          	ble	a4,a1,1c00a850 <mm_unroll_4x1+0x3ac>
1c00a712:	861a                	mv	a2,t1
1c00a714:	00149793          	slli	a5,s1,0x1
1c00a718:	00978333          	add	t1,a5,s1
1c00a71c:	12c05a63          	blez	a2,1c00a850 <mm_unroll_4x1+0x3ac>
1c00a720:	02b488b3          	mul	a7,s1,a1
1c00a724:	86ae                	mv	a3,a1
1c00a726:	00261713          	slli	a4,a2,0x2
1c00a72a:	00168813          	addi	a6,a3,1
1c00a72e:	8e32                	mv	t3,a2
1c00a730:	41c00533          	neg	a0,t3
1c00a734:	0e12                	slli	t3,t3,0x4
1c00a736:	d072                	sw	t3,32(sp)
1c00a738:	4e16                	lw	t3,68(sp)
1c00a73a:	00c705b3          	add	a1,a4,a2
1c00a73e:	01178bb3          	add	s7,a5,a7
1c00a742:	4786                	lw	a5,64(sp)
1c00a744:	01148cb3          	add	s9,s1,a7
1c00a748:	0c8a                	slli	s9,s9,0x2
1c00a74a:	0b8a                	slli	s7,s7,0x2
1c00a74c:	01130c33          	add	s8,t1,a7
1c00a750:	9cbe                	add	s9,s9,a5
1c00a752:	48c2                	lw	a7,16(sp)
1c00a754:	9bbe                	add	s7,s7,a5
1c00a756:	57c2                	lw	a5,48(sp)
1c00a758:	00449313          	slli	t1,s1,0x4
1c00a75c:	c41a                	sw	t1,8(sp)
1c00a75e:	42f887b3          	p.mac	a5,a7,a5
1c00a762:	4316                	lw	t1,68(sp)
1c00a764:	4896                	lw	a7,68(sp)
1c00a766:	00268613          	addi	a2,a3,2
1c00a76a:	068d                	addi	a3,a3,3
1c00a76c:	00249d93          	slli	s11,s1,0x2
1c00a770:	058a                	slli	a1,a1,0x2
1c00a772:	0c0a                	slli	s8,s8,0x2
1c00a774:	050a                	slli	a0,a0,0x2
1c00a776:	40b005b3          	neg	a1,a1
1c00a77a:	078a                	slli	a5,a5,0x2
1c00a77c:	979a                	add	a5,a5,t1
1c00a77e:	42e80333          	p.mac	t1,a6,a4
1c00a782:	4806                	lw	a6,64(sp)
1c00a784:	d22a                	sw	a0,36(sp)
1c00a786:	dc2e                	sw	a1,56(sp)
1c00a788:	8d42                	mv	s10,a6
1c00a78a:	9c42                	add	s8,s8,a6
1c00a78c:	ce1a                	sw	t1,28(sp)
1c00a78e:	4342                	lw	t1,16(sp)
1c00a790:	42e608b3          	p.mac	a7,a2,a4
1c00a794:	42e68e33          	p.mac	t3,a3,a4
1c00a798:	cc46                	sw	a7,24(sp)
1c00a79a:	426d8d33          	p.mac	s10,s11,t1
1c00a79e:	ca72                	sw	t3,20(sp)
1c00a7a0:	5512                	lw	a0,36(sp)
1c00a7a2:	56a2                	lw	a3,40(sp)
1c00a7a4:	4b52                	lw	s6,20(sp)
1c00a7a6:	953e                	add	a0,a0,a5
1c00a7a8:	40a78433          	sub	s0,a5,a0
1c00a7ac:	1471                	addi	s0,s0,-4
1c00a7ae:	4ae2                	lw	s5,24(sp)
1c00a7b0:	4a72                	lw	s4,28(sp)
1c00a7b2:	8009                	srli	s0,s0,0x2
1c00a7b4:	c62a                	sw	a0,12(sp)
1c00a7b6:	89aa                	mv	s3,a0
1c00a7b8:	0405                	addi	s0,s0,1
1c00a7ba:	14905e63          	blez	s1,1c00a916 <mm_unroll_4x1+0x472>
1c00a7be:	00000813          	li	a6,0
1c00a7c2:	00dd8f33          	add	t5,s11,a3
1c00a7c6:	40df0733          	sub	a4,t5,a3
1c00a7ca:	1771                	addi	a4,a4,-4
1c00a7cc:	8542                	mv	a0,a6
1c00a7ce:	85c2                	mv	a1,a6
1c00a7d0:	8642                	mv	a2,a6
1c00a7d2:	8309                	srli	a4,a4,0x2
1c00a7d4:	8ee2                	mv	t4,s8
1c00a7d6:	8e5e                	mv	t3,s7
1c00a7d8:	8366                	mv	t1,s9
1c00a7da:	88ea                	mv	a7,s10
1c00a7dc:	0705                	addi	a4,a4,1
1c00a7de:	0001                	nop
1c00a7e0:	012740fb          	lp.setup	x1,a4,1c00a804 <mm_unroll_4x1+0x360>
1c00a7e4:	0046a78b          	p.lw	a5,4(a3!)
1c00a7e8:	0048a90b          	p.lw	s2,4(a7!)
1c00a7ec:	0043238b          	p.lw	t2,4(t1!)
1c00a7f0:	004e228b          	p.lw	t0,4(t3!)
1c00a7f4:	004eaf8b          	p.lw	t6,4(t4!)
1c00a7f8:	60f97643          	fmadd.s	a2,s2,a5,a2
1c00a7fc:	58f3f5c3          	fmadd.s	a1,t2,a5,a1
1c00a800:	50f2f543          	fmadd.s	a0,t0,a5,a0
1c00a804:	80fff843          	fmadd.s	a6,t6,a5,a6
1c00a808:	00c9a22b          	p.sw	a2,4(s3!)
1c00a80c:	00ba222b          	p.sw	a1,4(s4!)
1c00a810:	00aaa22b          	p.sw	a0,4(s5!)
1c00a814:	010b222b          	p.sw	a6,4(s6!)
1c00a818:	86fa                	mv	a3,t5
1c00a81a:	147d                	addi	s0,s0,-1
1c00a81c:	fc59                	bnez	s0,1c00a7ba <mm_unroll_4x1+0x316>
1c00a81e:	5702                	lw	a4,32(sp)
1c00a820:	4542                	lw	a0,16(sp)
1c00a822:	46f2                	lw	a3,28(sp)
1c00a824:	4862                	lw	a6,24(sp)
1c00a826:	48d2                	lw	a7,20(sp)
1c00a828:	4e22                	lw	t3,8(sp)
1c00a82a:	45b2                	lw	a1,12(sp)
1c00a82c:	5662                	lw	a2,56(sp)
1c00a82e:	5352                	lw	t1,52(sp)
1c00a830:	0511                	addi	a0,a0,4
1c00a832:	96ba                	add	a3,a3,a4
1c00a834:	983a                	add	a6,a6,a4
1c00a836:	98ba                	add	a7,a7,a4
1c00a838:	c82a                	sw	a0,16(sp)
1c00a83a:	ce36                	sw	a3,28(sp)
1c00a83c:	cc42                	sw	a6,24(sp)
1c00a83e:	ca46                	sw	a7,20(sp)
1c00a840:	40c587b3          	sub	a5,a1,a2
1c00a844:	9d72                	add	s10,s10,t3
1c00a846:	9cf2                	add	s9,s9,t3
1c00a848:	9bf2                	add	s7,s7,t3
1c00a84a:	9c72                	add	s8,s8,t3
1c00a84c:	f4654ae3          	blt	a0,t1,1c00a7a0 <mm_unroll_4x1+0x2fc>
1c00a850:	45b6                	lw	a1,76(sp)
1c00a852:	eab050e3          	blez	a1,1c00a6f2 <mm_unroll_4x1+0x24e>
1c00a856:	5642                	lw	a2,48(sp)
1c00a858:	46a6                	lw	a3,72(sp)
1c00a85a:	00760713          	addi	a4,a2,7
1c00a85e:	41f75793          	srai	a5,a4,0x1f
1c00a862:	f837b7b3          	p.bclr	a5,a5,28,3
1c00a866:	06e7a7db          	p.addn	a5,a5,a4,3
1c00a86a:	02f68fb3          	mul	t6,a3,a5
1c00a86e:	97fe                	add	a5,a5,t6
1c00a870:	04c7c3b3          	p.min	t2,a5,a2
1c00a874:	e67fdfe3          	ble	t2,t6,1c00a6f2 <mm_unroll_4x1+0x24e>
1c00a878:	5772                	lw	a4,60(sp)
1c00a87a:	57b2                	lw	a5,44(sp)
1c00a87c:	e6e7dbe3          	ble	a4,a5,1c00a6f2 <mm_unroll_4x1+0x24e>
1c00a880:	8f7e                	mv	t5,t6
1c00a882:	5e22                	lw	t3,40(sp)
1c00a884:	4286                	lw	t0,64(sp)
1c00a886:	42f60f33          	p.mac	t5,a2,a5
1c00a88a:	00249e93          	slli	t4,s1,0x2
1c00a88e:	4816                	lw	a6,68(sp)
1c00a890:	00261413          	slli	s0,a2,0x2
1c00a894:	893a                	mv	s2,a4
1c00a896:	89be                	mv	s3,a5
1c00a898:	43df8e33          	p.mac	t3,t6,t4
1c00a89c:	0f0a                	slli	t5,t5,0x2
1c00a89e:	9f42                	add	t5,t5,a6
1c00a8a0:	43d782b3          	p.mac	t0,a5,t4
1c00a8a4:	8716                	mv	a4,t0
1c00a8a6:	837a                	mv	t1,t5
1c00a8a8:	41390833          	sub	a6,s2,s3
1c00a8ac:	01c8407b          	lp.setup	x0,a6,1c00a8e4 <mm_unroll_4x1+0x440>
1c00a8b0:	00000693          	li	a3,0
1c00a8b4:	00ee88b3          	add	a7,t4,a4
1c00a8b8:	02905463          	blez	s1,1c00a8e0 <mm_unroll_4x1+0x43c>
1c00a8bc:	00ee88b3          	add	a7,t4,a4
1c00a8c0:	40e887b3          	sub	a5,a7,a4
1c00a8c4:	17f1                	addi	a5,a5,-4
1c00a8c6:	00000693          	li	a3,0
1c00a8ca:	8389                	srli	a5,a5,0x2
1c00a8cc:	8672                	mv	a2,t3
1c00a8ce:	0785                	addi	a5,a5,1
1c00a8d0:	0067c0fb          	lp.setup	x1,a5,1c00a8dc <mm_unroll_4x1+0x438>
1c00a8d4:	0047250b          	p.lw	a0,4(a4!)
1c00a8d8:	0046258b          	p.lw	a1,4(a2!)
1c00a8dc:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00a8e0:	00d3642b          	p.sw	a3,s0(t1!)
1c00a8e4:	8746                	mv	a4,a7
1c00a8e6:	0f85                	addi	t6,t6,1
1c00a8e8:	0f11                	addi	t5,t5,4
1c00a8ea:	9e76                	add	t3,t3,t4
1c00a8ec:	fbf39ce3          	bne	t2,t6,1c00a8a4 <mm_unroll_4x1+0x400>
1c00a8f0:	b509                	j	1c00a6f2 <mm_unroll_4x1+0x24e>
1c00a8f2:	00000513          	li	a0,0
1c00a8f6:	0491                	addi	s1,s1,4
1c00a8f8:	86aa                	mv	a3,a0
1c00a8fa:	862a                	mv	a2,a0
1c00a8fc:	85aa                	mv	a1,a0
1c00a8fe:	00d9222b          	p.sw	a3,4(s2!)
1c00a902:	00c9a22b          	p.sw	a2,4(s3!)
1c00a906:	00ba222b          	p.sw	a1,4(s4!)
1c00a90a:	00aaa22b          	p.sw	a0,4(s5!)
1c00a90e:	147d                	addi	s0,s0,-1
1c00a910:	ca0412e3          	bnez	s0,1c00a5b4 <mm_unroll_4x1+0x110>
1c00a914:	b309                	j	1c00a616 <mm_unroll_4x1+0x172>
1c00a916:	00000813          	li	a6,0
1c00a91a:	00dd8f33          	add	t5,s11,a3
1c00a91e:	86fa                	mv	a3,t5
1c00a920:	8642                	mv	a2,a6
1c00a922:	85c2                	mv	a1,a6
1c00a924:	8542                	mv	a0,a6
1c00a926:	00c9a22b          	p.sw	a2,4(s3!)
1c00a92a:	00ba222b          	p.sw	a1,4(s4!)
1c00a92e:	00aaa22b          	p.sw	a0,4(s5!)
1c00a932:	010b222b          	p.sw	a6,4(s6!)
1c00a936:	147d                	addi	s0,s0,-1
1c00a938:	e80411e3          	bnez	s0,1c00a7ba <mm_unroll_4x1+0x316>
1c00a93c:	b5cd                	j	1c00a81e <mm_unroll_4x1+0x37a>
1c00a93e:	5476                	lw	s0,124(sp)
1c00a940:	54e6                	lw	s1,120(sp)
1c00a942:	5956                	lw	s2,116(sp)
1c00a944:	59c6                	lw	s3,112(sp)
1c00a946:	5a36                	lw	s4,108(sp)
1c00a948:	5aa6                	lw	s5,104(sp)
1c00a94a:	5b16                	lw	s6,100(sp)
1c00a94c:	5b86                	lw	s7,96(sp)
1c00a94e:	4c76                	lw	s8,92(sp)
1c00a950:	4ce6                	lw	s9,88(sp)
1c00a952:	4d56                	lw	s10,84(sp)
1c00a954:	4dc6                	lw	s11,80(sp)
1c00a956:	6109                	addi	sp,sp,128
1c00a958:	ff0ff06f          	j	1c00a148 <mm_unroll_2x1>

1c00a95c <mm_unroll_8x1>:
1c00a95c:	00c52283          	lw	t0,12(a0)
1c00a960:	7151                	addi	sp,sp,-240
1c00a962:	d7a2                	sw	s0,236(sp)
1c00a964:	c402b333          	p.bclr	t1,t0,2,0
1c00a968:	d5a6                	sw	s1,232(sp)
1c00a96a:	d3ca                	sw	s2,228(sp)
1c00a96c:	d1ce                	sw	s3,224(sp)
1c00a96e:	cfd2                	sw	s4,220(sp)
1c00a970:	cdd6                	sw	s5,216(sp)
1c00a972:	cbda                	sw	s6,212(sp)
1c00a974:	c9de                	sw	s7,208(sp)
1c00a976:	c7e2                	sw	s8,204(sp)
1c00a978:	c5e6                	sw	s9,200(sp)
1c00a97a:	c3ea                	sw	s10,196(sp)
1c00a97c:	c1ee                	sw	s11,192(sp)
1c00a97e:	d316                	sw	t0,164(sp)
1c00a980:	03f00713          	li	a4,63
1c00a984:	d49a                	sw	t1,104(sp)
1c00a986:	f14026f3          	csrr	a3,mhartid
1c00a98a:	64675163          	ble	t1,a4,1c00afcc <mm_unroll_8x1+0x670>
1c00a98e:	f456b6b3          	p.bclr	a3,a3,26,5
1c00a992:	40335713          	srai	a4,t1,0x3
1c00a996:	02e68433          	mul	s0,a3,a4
1c00a99a:	87aa                	mv	a5,a0
1c00a99c:	478c                	lw	a1,8(a5)
1c00a99e:	4b90                	lw	a2,16(a5)
1c00a9a0:	4bdc                	lw	a5,20(a5)
1c00a9a2:	4104                	lw	s1,0(a0)
1c00a9a4:	dd36                	sw	a3,184(sp)
1c00a9a6:	da3e                	sw	a5,52(sp)
1c00a9a8:	579a                	lw	a5,164(sp)
1c00a9aa:	4d14                	lw	a3,24(a0)
1c00a9ac:	4148                	lw	a0,4(a0)
1c00a9ae:	9722                	add	a4,a4,s0
1c00a9b0:	406787b3          	sub	a5,a5,t1
1c00a9b4:	04674733          	p.min	a4,a4,t1
1c00a9b8:	cca2                	sw	s0,88(sp)
1c00a9ba:	d526                	sw	s1,168(sp)
1c00a9bc:	d2aa                	sw	a0,100(sp)
1c00a9be:	d8ae                	sw	a1,112(sp)
1c00a9c0:	d6b2                	sw	a2,108(sp)
1c00a9c2:	df3e                	sw	a5,188(sp)
1c00a9c4:	daba                	sw	a4,116(sp)
1c00a9c6:	30069463          	bnez	a3,1c00acce <mm_unroll_8x1+0x372>
1c00a9ca:	84a2                	mv	s1,s0
1c00a9cc:	22e45d63          	ble	a4,s0,1c00ac06 <mm_unroll_8x1+0x2aa>
1c00a9d0:	55d2                	lw	a1,52(sp)
1c00a9d2:	86b2                	mv	a3,a2
1c00a9d4:	00259613          	slli	a2,a1,0x2
1c00a9d8:	00159e13          	slli	t3,a1,0x1
1c00a9dc:	ceb2                	sw	a2,92(sp)
1c00a9de:	00be0833          	add	a6,t3,a1
1c00a9e2:	22d05263          	blez	a3,1c00ac06 <mm_unroll_8x1+0x2aa>
1c00a9e6:	53aa                	lw	t2,168(sp)
1c00a9e8:	02958eb3          	mul	t4,a1,s1
1c00a9ec:	8432                	mv	s0,a2
1c00a9ee:	00269713          	slli	a4,a3,0x2
1c00a9f2:	8fae                	mv	t6,a1
1c00a9f4:	dcba                	sw	a4,120(sp)
1c00a9f6:	00248613          	addi	a2,s1,2
1c00a9fa:	00348693          	addi	a3,s1,3
1c00a9fe:	00448713          	addi	a4,s1,4
1c00aa02:	00359513          	slli	a0,a1,0x3
1c00aa06:	00548793          	addi	a5,s1,5
1c00aa0a:	429403b3          	p.mac	t2,s0,s1
1c00aa0e:	00148593          	addi	a1,s1,1
1c00aa12:	00648893          	addi	a7,s1,6
1c00aa16:	00748313          	addi	t1,s1,7
1c00aa1a:	54aa                	lw	s1,168(sp)
1c00aa1c:	01df8f33          	add	t5,t6,t4
1c00aa20:	9e76                	add	t3,t3,t4
1c00aa22:	41f50533          	sub	a0,a0,t6
1c00aa26:	002f1e93          	slli	t4,t5,0x2
1c00aa2a:	050a                	slli	a0,a0,0x2
1c00aa2c:	0e0a                	slli	t3,t3,0x2
1c00aa2e:	9ea6                	add	t4,t4,s1
1c00aa30:	9e26                	add	t3,t3,s1
1c00aa32:	d61e                	sw	t2,44(sp)
1c00aa34:	db2a                	sw	a0,180(sp)
1c00aa36:	53e6                	lw	t2,120(sp)
1c00aa38:	54c6                	lw	s1,112(sp)
1c00aa3a:	5546                	lw	a0,112(sp)
1c00aa3c:	005f9293          	slli	t0,t6,0x5
1c00aa40:	427584b3          	p.mac	s1,a1,t2
1c00aa44:	55c6                	lw	a1,112(sp)
1c00aa46:	d096                	sw	t0,96(sp)
1c00aa48:	4466                	lw	s0,88(sp)
1c00aa4a:	52c6                	lw	t0,112(sp)
1c00aa4c:	de76                	sw	t4,60(sp)
1c00aa4e:	dc72                	sw	t3,56(sp)
1c00aa50:	5ee6                	lw	t4,120(sp)
1c00aa52:	5e36                	lw	t3,108(sp)
1c00aa54:	5f16                	lw	t5,100(sp)
1c00aa56:	42760533          	p.mac	a0,a2,t2
1c00aa5a:	5646                	lw	a2,112(sp)
1c00aa5c:	00281f93          	slli	t6,a6,0x2
1c00aa60:	0e16                	slli	t3,t3,0x5
1c00aa62:	080e                	slli	a6,a6,0x3
1c00aa64:	9efa                	add	t4,t4,t5
1c00aa66:	d172                	sw	t3,160(sp)
1c00aa68:	d776                	sw	t4,172(sp)
1c00aa6a:	defe                	sw	t6,124(sp)
1c00aa6c:	d802                	sw	zero,48(sp)
1c00aa6e:	427685b3          	p.mac	a1,a3,t2
1c00aa72:	56c6                	lw	a3,112(sp)
1c00aa74:	d942                	sw	a6,176(sp)
1c00aa76:	cd26                	sw	s1,152(sp)
1c00aa78:	cb2a                	sw	a0,148(sp)
1c00aa7a:	42770633          	p.mac	a2,a4,t2
1c00aa7e:	5746                	lw	a4,112(sp)
1c00aa80:	c92e                	sw	a1,144(sp)
1c00aa82:	428382b3          	p.mac	t0,t2,s0
1c00aa86:	c732                	sw	a2,140(sp)
1c00aa88:	427786b3          	p.mac	a3,a5,t2
1c00aa8c:	cf16                	sw	t0,156(sp)
1c00aa8e:	42788733          	p.mac	a4,a7,t2
1c00aa92:	c536                	sw	a3,136(sp)
1c00aa94:	c33a                	sw	a4,132(sp)
1c00aa96:	5846                	lw	a6,112(sp)
1c00aa98:	42730833          	p.mac	a6,t1,t2
1c00aa9c:	c142                	sw	a6,128(sp)
1c00aa9e:	5332                	lw	t1,44(sp)
1c00aaa0:	52f6                	lw	t0,124(sp)
1c00aaa2:	43f6                	lw	t2,92(sp)
1c00aaa4:	4476                	lw	s0,92(sp)
1c00aaa6:	929a                	add	t0,t0,t1
1c00aaa8:	54ba                	lw	s1,172(sp)
1c00aaaa:	5516                	lw	a0,100(sp)
1c00aaac:	45f6                	lw	a1,92(sp)
1c00aaae:	9396                	add	t2,t2,t0
1c00aab0:	4676                	lw	a2,92(sp)
1c00aab2:	5742                	lw	a4,48(sp)
1c00aab4:	ca96                	sw	t0,84(sp)
1c00aab6:	941e                	add	s0,s0,t2
1c00aab8:	46fa                	lw	a3,156(sp)
1c00aaba:	486a                	lw	a6,152(sp)
1c00aabc:	48da                	lw	a7,148(sp)
1c00aabe:	4e4a                	lw	t3,144(sp)
1c00aac0:	4eba                	lw	t4,140(sp)
1c00aac2:	4f2a                	lw	t5,136(sp)
1c00aac4:	4f9a                	lw	t6,132(sp)
1c00aac6:	428a                	lw	t0,128(sp)
1c00aac8:	534a                	lw	t1,176(sp)
1c00aaca:	95a2                	add	a1,a1,s0
1c00aacc:	40a487b3          	sub	a5,s1,a0
1c00aad0:	962e                	add	a2,a2,a1
1c00aad2:	17f1                	addi	a5,a5,-4
1c00aad4:	8389                	srli	a5,a5,0x2
1c00aad6:	96ba                	add	a3,a3,a4
1c00aad8:	983a                	add	a6,a6,a4
1c00aada:	98ba                	add	a7,a7,a4
1c00aadc:	9e3a                	add	t3,t3,a4
1c00aade:	9eba                	add	t4,t4,a4
1c00aae0:	9f3a                	add	t5,t5,a4
1c00aae2:	9fba                	add	t6,t6,a4
1c00aae4:	92ba                	add	t0,t0,a4
1c00aae6:	40660333          	sub	t1,a2,t1
1c00aaea:	c89e                	sw	t2,80(sp)
1c00aaec:	c6a2                	sw	s0,76(sp)
1c00aaee:	c4ae                	sw	a1,72(sp)
1c00aaf0:	c2b2                	sw	a2,68(sp)
1c00aaf2:	d436                	sw	a3,40(sp)
1c00aaf4:	d242                	sw	a6,36(sp)
1c00aaf6:	d046                	sw	a7,32(sp)
1c00aaf8:	ce72                	sw	t3,28(sp)
1c00aafa:	cc76                	sw	t4,24(sp)
1c00aafc:	ca7a                	sw	t5,20(sp)
1c00aafe:	c87e                	sw	t6,16(sp)
1c00ab00:	c616                	sw	t0,12(sp)
1c00ab02:	c09a                	sw	t1,64(sp)
1c00ab04:	0785                	addi	a5,a5,1
1c00ab06:	c42a                	sw	a0,8(sp)
1c00ab08:	c23e                	sw	a5,4(sp)
1c00ab0a:	5652                	lw	a2,52(sp)
1c00ab0c:	48c05963          	blez	a2,1c00af9e <mm_unroll_8x1+0x642>
1c00ab10:	4886                	lw	a7,64(sp)
1c00ab12:	5e32                	lw	t3,44(sp)
1c00ab14:	4996                	lw	s3,68(sp)
1c00ab16:	4926                	lw	s2,72(sp)
1c00ab18:	41c88733          	sub	a4,a7,t3
1c00ab1c:	00000e13          	li	t3,0
1c00ab20:	1771                	addi	a4,a4,-4
1c00ab22:	8309                	srli	a4,a4,0x2
1c00ab24:	8372                	mv	t1,t3
1c00ab26:	88f2                	mv	a7,t3
1c00ab28:	8872                	mv	a6,t3
1c00ab2a:	8572                	mv	a0,t3
1c00ab2c:	85f2                	mv	a1,t3
1c00ab2e:	8672                	mv	a2,t3
1c00ab30:	86f2                	mv	a3,t3
1c00ab32:	44b6                	lw	s1,76(sp)
1c00ab34:	4446                	lw	s0,80(sp)
1c00ab36:	43d6                	lw	t2,84(sp)
1c00ab38:	52e2                	lw	t0,56(sp)
1c00ab3a:	5ff2                	lw	t6,60(sp)
1c00ab3c:	5f32                	lw	t5,44(sp)
1c00ab3e:	4ea2                	lw	t4,8(sp)
1c00ab40:	0705                	addi	a4,a4,1
1c00ab42:	8dba                	mv	s11,a4
1c00ab44:	023dc0fb          	lp.setup	x1,s11,1c00ab8a <mm_unroll_8x1+0x22e>
1c00ab48:	5766                	lw	a4,120(sp)
1c00ab4a:	004f2d0b          	p.lw	s10,4(t5!)
1c00ab4e:	004fac8b          	p.lw	s9,4(t6!)
1c00ab52:	20eef78b          	p.lw	a5,a4(t4!)
1c00ab56:	0042ac0b          	p.lw	s8,4(t0!)
1c00ab5a:	0043ab8b          	p.lw	s7,4(t2!)
1c00ab5e:	00442b0b          	p.lw	s6,4(s0!)
1c00ab62:	0044aa8b          	p.lw	s5,4(s1!)
1c00ab66:	00492a0b          	p.lw	s4,4(s2!)
1c00ab6a:	0049a70b          	p.lw	a4,4(s3!)
1c00ab6e:	68fd76c3          	fmadd.s	a3,s10,a5,a3
1c00ab72:	60fcf643          	fmadd.s	a2,s9,a5,a2
1c00ab76:	58fc75c3          	fmadd.s	a1,s8,a5,a1
1c00ab7a:	50fbf543          	fmadd.s	a0,s7,a5,a0
1c00ab7e:	80fb7843          	fmadd.s	a6,s6,a5,a6
1c00ab82:	88faf8c3          	fmadd.s	a7,s5,a5,a7
1c00ab86:	30fa7343          	fmadd.s	t1,s4,a5,t1
1c00ab8a:	e0f77e43          	fmadd.s	t3,a4,a5,t3
1c00ab8e:	57a2                	lw	a5,40(sp)
1c00ab90:	5e92                	lw	t4,36(sp)
1c00ab92:	5f02                	lw	t5,32(sp)
1c00ab94:	00d7a22b          	p.sw	a3,4(a5!)
1c00ab98:	4ff2                	lw	t6,28(sp)
1c00ab9a:	00cea22b          	p.sw	a2,4(t4!)
1c00ab9e:	42e2                	lw	t0,24(sp)
1c00aba0:	00bf222b          	p.sw	a1,4(t5!)
1c00aba4:	43d2                	lw	t2,20(sp)
1c00aba6:	00afa22b          	p.sw	a0,4(t6!)
1c00abaa:	4442                	lw	s0,16(sp)
1c00abac:	0102a22b          	p.sw	a6,4(t0!)
1c00abb0:	44b2                	lw	s1,12(sp)
1c00abb2:	4522                	lw	a0,8(sp)
1c00abb4:	4592                	lw	a1,4(sp)
1c00abb6:	0113a22b          	p.sw	a7,4(t2!)
1c00abba:	0064222b          	p.sw	t1,4(s0!)
1c00abbe:	01c4a22b          	p.sw	t3,4(s1!)
1c00abc2:	0511                	addi	a0,a0,4
1c00abc4:	15fd                	addi	a1,a1,-1
1c00abc6:	d43e                	sw	a5,40(sp)
1c00abc8:	d276                	sw	t4,36(sp)
1c00abca:	d07a                	sw	t5,32(sp)
1c00abcc:	ce7e                	sw	t6,28(sp)
1c00abce:	cc16                	sw	t0,24(sp)
1c00abd0:	ca1e                	sw	t2,20(sp)
1c00abd2:	c822                	sw	s0,16(sp)
1c00abd4:	c626                	sw	s1,12(sp)
1c00abd6:	c42a                	sw	a0,8(sp)
1c00abd8:	c22e                	sw	a1,4(sp)
1c00abda:	f985                	bnez	a1,1c00ab0a <mm_unroll_8x1+0x1ae>
1c00abdc:	46e6                	lw	a3,88(sp)
1c00abde:	5872                	lw	a6,60(sp)
1c00abe0:	5886                	lw	a7,96(sp)
1c00abe2:	5e62                	lw	t3,56(sp)
1c00abe4:	5ec2                	lw	t4,48(sp)
1c00abe6:	575a                	lw	a4,180(sp)
1c00abe8:	4786                	lw	a5,64(sp)
1c00abea:	5f0a                	lw	t5,160(sp)
1c00abec:	5fd6                	lw	t6,116(sp)
1c00abee:	06a1                	addi	a3,a3,8
1c00abf0:	973e                	add	a4,a4,a5
1c00abf2:	9846                	add	a6,a6,a7
1c00abf4:	9e46                	add	t3,t3,a7
1c00abf6:	9efa                	add	t4,t4,t5
1c00abf8:	ccb6                	sw	a3,88(sp)
1c00abfa:	d63a                	sw	a4,44(sp)
1c00abfc:	de42                	sw	a6,60(sp)
1c00abfe:	dc72                	sw	t3,56(sp)
1c00ac00:	d876                	sw	t4,48(sp)
1c00ac02:	e9f6cee3          	blt	a3,t6,1c00aa9e <mm_unroll_8x1+0x142>
1c00ac06:	53fa                	lw	t2,188(sp)
1c00ac08:	0a705563          	blez	t2,1c00acb2 <mm_unroll_8x1+0x356>
1c00ac0c:	5436                	lw	s0,108(sp)
1c00ac0e:	54ea                	lw	s1,184(sp)
1c00ac10:	00740713          	addi	a4,s0,7
1c00ac14:	41f75793          	srai	a5,a4,0x1f
1c00ac18:	f837b7b3          	p.bclr	a5,a5,28,3
1c00ac1c:	06e7a7db          	p.addn	a5,a5,a4,3
1c00ac20:	02f48733          	mul	a4,s1,a5
1c00ac24:	97ba                	add	a5,a5,a4
1c00ac26:	0487c7b3          	p.min	a5,a5,s0
1c00ac2a:	08f75463          	ble	a5,a4,1c00acb2 <mm_unroll_8x1+0x356>
1c00ac2e:	559a                	lw	a1,164(sp)
1c00ac30:	5626                	lw	a2,104(sp)
1c00ac32:	08b65063          	ble	a1,a2,1c00acb2 <mm_unroll_8x1+0x356>
1c00ac36:	56d2                	lw	a3,52(sp)
1c00ac38:	8fba                	mv	t6,a4
1c00ac3a:	52aa                	lw	t0,168(sp)
1c00ac3c:	42c40fb3          	p.mac	t6,s0,a2
1c00ac40:	00269e93          	slli	t4,a3,0x2
1c00ac44:	5816                	lw	a6,100(sp)
1c00ac46:	58c6                	lw	a7,112(sp)
1c00ac48:	070a                	slli	a4,a4,0x2
1c00ac4a:	078a                	slli	a5,a5,0x2
1c00ac4c:	00241393          	slli	t2,s0,0x2
1c00ac50:	00e80e33          	add	t3,a6,a4
1c00ac54:	00f80f33          	add	t5,a6,a5
1c00ac58:	84ae                	mv	s1,a1
1c00ac5a:	43d602b3          	p.mac	t0,a2,t4
1c00ac5e:	0f8a                	slli	t6,t6,0x2
1c00ac60:	9fc6                	add	t6,t6,a7
1c00ac62:	8436                	mv	s0,a3
1c00ac64:	8932                	mv	s2,a2
1c00ac66:	8716                	mv	a4,t0
1c00ac68:	837e                	mv	t1,t6
1c00ac6a:	41248833          	sub	a6,s1,s2
1c00ac6e:	0001                	nop
1c00ac70:	01c8407b          	lp.setup	x0,a6,1c00aca8 <mm_unroll_8x1+0x34c>
1c00ac74:	00000693          	li	a3,0
1c00ac78:	00ee88b3          	add	a7,t4,a4
1c00ac7c:	02805463          	blez	s0,1c00aca4 <mm_unroll_8x1+0x348>
1c00ac80:	00ee88b3          	add	a7,t4,a4
1c00ac84:	40e887b3          	sub	a5,a7,a4
1c00ac88:	17f1                	addi	a5,a5,-4
1c00ac8a:	00000693          	li	a3,0
1c00ac8e:	8389                	srli	a5,a5,0x2
1c00ac90:	8672                	mv	a2,t3
1c00ac92:	0785                	addi	a5,a5,1
1c00ac94:	0067c0fb          	lp.setup	x1,a5,1c00aca0 <mm_unroll_8x1+0x344>
1c00ac98:	0047250b          	p.lw	a0,4(a4!)
1c00ac9c:	2076758b          	p.lw	a1,t2(a2!)
1c00aca0:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00aca4:	00d363ab          	p.sw	a3,t2(t1!)
1c00aca8:	8746                	mv	a4,a7
1c00acaa:	0e11                	addi	t3,t3,4
1c00acac:	0f91                	addi	t6,t6,4
1c00acae:	fbcf1ce3          	bne	t5,t3,1c00ac66 <mm_unroll_8x1+0x30a>
1c00acb2:	543e                	lw	s0,236(sp)
1c00acb4:	54ae                	lw	s1,232(sp)
1c00acb6:	591e                	lw	s2,228(sp)
1c00acb8:	598e                	lw	s3,224(sp)
1c00acba:	4a7e                	lw	s4,220(sp)
1c00acbc:	4aee                	lw	s5,216(sp)
1c00acbe:	4b5e                	lw	s6,212(sp)
1c00acc0:	4bce                	lw	s7,208(sp)
1c00acc2:	4c3e                	lw	s8,204(sp)
1c00acc4:	4cae                	lw	s9,200(sp)
1c00acc6:	4d1e                	lw	s10,196(sp)
1c00acc8:	4d8e                	lw	s11,192(sp)
1c00acca:	616d                	addi	sp,sp,240
1c00accc:	8082                	ret
1c00acce:	22e45563          	ble	a4,s0,1c00aef8 <mm_unroll_8x1+0x59c>
1c00acd2:	5852                	lw	a6,52(sp)
1c00acd4:	00281893          	slli	a7,a6,0x2
1c00acd8:	00181793          	slli	a5,a6,0x1
1c00acdc:	d646                	sw	a7,44(sp)
1c00acde:	01078333          	add	t1,a5,a6
1c00ace2:	20c05b63          	blez	a2,1c00aef8 <mm_unroll_8x1+0x59c>
1c00ace6:	02880eb3          	mul	t4,a6,s0
1c00acea:	8fc2                	mv	t6,a6
1c00acec:	84a2                	mv	s1,s0
1c00acee:	00361813          	slli	a6,a2,0x3
1c00acf2:	00261713          	slli	a4,a2,0x2
1c00acf6:	83b2                	mv	t2,a2
1c00acf8:	9832                	add	a6,a6,a2
1c00acfa:	40c00e33          	neg	t3,a2
1c00acfe:	00140513          	addi	a0,s0,1
1c00ad02:	00240593          	addi	a1,s0,2
1c00ad06:	97f6                	add	a5,a5,t4
1c00ad08:	01df8f33          	add	t5,t6,t4
1c00ad0c:	00279e93          	slli	t4,a5,0x2
1c00ad10:	57b6                	lw	a5,108(sp)
1c00ad12:	00340613          	addi	a2,s0,3
1c00ad16:	00440693          	addi	a3,s0,4
1c00ad1a:	42f487b3          	p.mac	a5,s1,a5
1c00ad1e:	00540893          	addi	a7,s0,5
1c00ad22:	00640f93          	addi	t6,s0,6
1c00ad26:	00740293          	addi	t0,s0,7
1c00ad2a:	5452                	lw	s0,52(sp)
1c00ad2c:	0396                	slli	t2,t2,0x5
1c00ad2e:	cd1e                	sw	t2,152(sp)
1c00ad30:	0416                	slli	s0,s0,0x5
1c00ad32:	d0a2                	sw	s0,96(sp)
1c00ad34:	5446                	lw	s0,112(sp)
1c00ad36:	078a                	slli	a5,a5,0x2
1c00ad38:	53aa                	lw	t2,168(sp)
1c00ad3a:	97a2                	add	a5,a5,s0
1c00ad3c:	5432                	lw	s0,44(sp)
1c00ad3e:	0f0a                	slli	t5,t5,0x2
1c00ad40:	080a                	slli	a6,a6,0x2
1c00ad42:	429403b3          	p.mac	t2,s0,s1
1c00ad46:	54aa                	lw	s1,168(sp)
1c00ad48:	0e0a                	slli	t3,t3,0x2
1c00ad4a:	030a                	slli	t1,t1,0x2
1c00ad4c:	9f26                	add	t5,t5,s1
1c00ad4e:	9ea6                	add	t4,t4,s1
1c00ad50:	41000833          	neg	a6,a6
1c00ad54:	de7a                	sw	t5,60(sp)
1c00ad56:	dc76                	sw	t4,56(sp)
1c00ad58:	cf72                	sw	t3,156(sp)
1c00ad5a:	c09e                	sw	t2,64(sp)
1c00ad5c:	dc9a                	sw	t1,120(sp)
1c00ad5e:	d142                	sw	a6,160(sp)
1c00ad60:	5e46                	lw	t3,112(sp)
1c00ad62:	5846                	lw	a6,112(sp)
1c00ad64:	5ec6                	lw	t4,112(sp)
1c00ad66:	5f46                	lw	t5,112(sp)
1c00ad68:	5346                	lw	t1,112(sp)
1c00ad6a:	53c6                	lw	t2,112(sp)
1c00ad6c:	5446                	lw	s0,112(sp)
1c00ad6e:	42e50833          	p.mac	a6,a0,a4
1c00ad72:	d802                	sw	zero,48(sp)
1c00ad74:	42e58e33          	p.mac	t3,a1,a4
1c00ad78:	cb42                	sw	a6,148(sp)
1c00ad7a:	42e60eb3          	p.mac	t4,a2,a4
1c00ad7e:	c972                	sw	t3,144(sp)
1c00ad80:	42e68f33          	p.mac	t5,a3,a4
1c00ad84:	c776                	sw	t4,140(sp)
1c00ad86:	42e88333          	p.mac	t1,a7,a4
1c00ad8a:	c57a                	sw	t5,136(sp)
1c00ad8c:	42ef83b3          	p.mac	t2,t6,a4
1c00ad90:	c31a                	sw	t1,132(sp)
1c00ad92:	42e28433          	p.mac	s0,t0,a4
1c00ad96:	c11e                	sw	t2,128(sp)
1c00ad98:	dea2                	sw	s0,124(sp)
1c00ad9a:	457a                	lw	a0,156(sp)
1c00ad9c:	55e6                	lw	a1,120(sp)
1c00ad9e:	4606                	lw	a2,64(sp)
1c00ada0:	56b2                	lw	a3,44(sp)
1c00ada2:	5732                	lw	a4,44(sp)
1c00ada4:	953e                	add	a0,a0,a5
1c00ada6:	95b2                	add	a1,a1,a2
1c00ada8:	5832                	lw	a6,44(sp)
1c00adaa:	5e42                	lw	t3,48(sp)
1c00adac:	96ae                	add	a3,a3,a1
1c00adae:	8f89                	sub	a5,a5,a0
1c00adb0:	48da                	lw	a7,148(sp)
1c00adb2:	4eca                	lw	t4,144(sp)
1c00adb4:	4f3a                	lw	t5,140(sp)
1c00adb6:	4faa                	lw	t6,136(sp)
1c00adb8:	429a                	lw	t0,132(sp)
1c00adba:	430a                	lw	t1,128(sp)
1c00adbc:	53f6                	lw	t2,124(sp)
1c00adbe:	5432                	lw	s0,44(sp)
1c00adc0:	9736                	add	a4,a4,a3
1c00adc2:	17f1                	addi	a5,a5,-4
1c00adc4:	983a                	add	a6,a6,a4
1c00adc6:	c8b6                	sw	a3,80(sp)
1c00adc8:	8389                	srli	a5,a5,0x2
1c00adca:	5696                	lw	a3,100(sp)
1c00adcc:	98f2                	add	a7,a7,t3
1c00adce:	9ef2                	add	t4,t4,t3
1c00add0:	9f72                	add	t5,t5,t3
1c00add2:	9ff2                	add	t6,t6,t3
1c00add4:	92f2                	add	t0,t0,t3
1c00add6:	9372                	add	t1,t1,t3
1c00add8:	93f2                	add	t2,t2,t3
1c00adda:	9442                	add	s0,s0,a6
1c00addc:	0785                	addi	a5,a5,1
1c00adde:	ceaa                	sw	a0,92(sp)
1c00ade0:	caae                	sw	a1,84(sp)
1c00ade2:	c6ba                	sw	a4,76(sp)
1c00ade4:	c4c2                	sw	a6,72(sp)
1c00ade6:	d246                	sw	a7,36(sp)
1c00ade8:	d076                	sw	t4,32(sp)
1c00adea:	ce7a                	sw	t5,28(sp)
1c00adec:	cc7e                	sw	t6,24(sp)
1c00adee:	ca16                	sw	t0,20(sp)
1c00adf0:	c81a                	sw	t1,16(sp)
1c00adf2:	c61e                	sw	t2,12(sp)
1c00adf4:	c2a2                	sw	s0,68(sp)
1c00adf6:	d42a                	sw	a0,40(sp)
1c00adf8:	c23e                	sw	a5,4(sp)
1c00adfa:	5752                	lw	a4,52(sp)
1c00adfc:	1ae05b63          	blez	a4,1c00afb2 <mm_unroll_8x1+0x656>
1c00ae00:	5e32                	lw	t3,44(sp)
1c00ae02:	00000e93          	li	t4,0
1c00ae06:	4996                	lw	s3,68(sp)
1c00ae08:	9e36                	add	t3,t3,a3
1c00ae0a:	40de0733          	sub	a4,t3,a3
1c00ae0e:	c472                	sw	t3,8(sp)
1c00ae10:	1771                	addi	a4,a4,-4
1c00ae12:	8e76                	mv	t3,t4
1c00ae14:	8376                	mv	t1,t4
1c00ae16:	88f6                	mv	a7,t4
1c00ae18:	8876                	mv	a6,t4
1c00ae1a:	8576                	mv	a0,t4
1c00ae1c:	85f6                	mv	a1,t4
1c00ae1e:	8676                	mv	a2,t4
1c00ae20:	4926                	lw	s2,72(sp)
1c00ae22:	44b6                	lw	s1,76(sp)
1c00ae24:	4446                	lw	s0,80(sp)
1c00ae26:	43d6                	lw	t2,84(sp)
1c00ae28:	52e2                	lw	t0,56(sp)
1c00ae2a:	5ff2                	lw	t6,60(sp)
1c00ae2c:	4f06                	lw	t5,64(sp)
1c00ae2e:	8309                	srli	a4,a4,0x2
1c00ae30:	0705                	addi	a4,a4,1
1c00ae32:	0001                	nop
1c00ae34:	022740fb          	lp.setup	x1,a4,1c00ae78 <mm_unroll_8x1+0x51c>
1c00ae38:	0046a78b          	p.lw	a5,4(a3!)
1c00ae3c:	004f2d8b          	p.lw	s11,4(t5!)
1c00ae40:	004fad0b          	p.lw	s10,4(t6!)
1c00ae44:	0042ac8b          	p.lw	s9,4(t0!)
1c00ae48:	0043ac0b          	p.lw	s8,4(t2!)
1c00ae4c:	00442b8b          	p.lw	s7,4(s0!)
1c00ae50:	0044ab0b          	p.lw	s6,4(s1!)
1c00ae54:	00492a8b          	p.lw	s5,4(s2!)
1c00ae58:	0049aa0b          	p.lw	s4,4(s3!)
1c00ae5c:	60fdf643          	fmadd.s	a2,s11,a5,a2
1c00ae60:	58fd75c3          	fmadd.s	a1,s10,a5,a1
1c00ae64:	50fcf543          	fmadd.s	a0,s9,a5,a0
1c00ae68:	80fc7843          	fmadd.s	a6,s8,a5,a6
1c00ae6c:	88fbf8c3          	fmadd.s	a7,s7,a5,a7
1c00ae70:	30fb7343          	fmadd.s	t1,s6,a5,t1
1c00ae74:	e0fafe43          	fmadd.s	t3,s5,a5,t3
1c00ae78:	e8fa7ec3          	fmadd.s	t4,s4,a5,t4
1c00ae7c:	5f22                	lw	t5,40(sp)
1c00ae7e:	5f92                	lw	t6,36(sp)
1c00ae80:	5282                	lw	t0,32(sp)
1c00ae82:	00cf222b          	p.sw	a2,4(t5!)
1c00ae86:	43f2                	lw	t2,28(sp)
1c00ae88:	00bfa22b          	p.sw	a1,4(t6!)
1c00ae8c:	4462                	lw	s0,24(sp)
1c00ae8e:	00a2a22b          	p.sw	a0,4(t0!)
1c00ae92:	44d2                	lw	s1,20(sp)
1c00ae94:	0103a22b          	p.sw	a6,4(t2!)
1c00ae98:	4542                	lw	a0,16(sp)
1c00ae9a:	0114222b          	p.sw	a7,4(s0!)
1c00ae9e:	45b2                	lw	a1,12(sp)
1c00aea0:	4612                	lw	a2,4(sp)
1c00aea2:	0064a22b          	p.sw	t1,4(s1!)
1c00aea6:	01c5222b          	p.sw	t3,4(a0!)
1c00aeaa:	01d5a22b          	p.sw	t4,4(a1!)
1c00aeae:	167d                	addi	a2,a2,-1
1c00aeb0:	d47a                	sw	t5,40(sp)
1c00aeb2:	d27e                	sw	t6,36(sp)
1c00aeb4:	d016                	sw	t0,32(sp)
1c00aeb6:	ce1e                	sw	t2,28(sp)
1c00aeb8:	cc22                	sw	s0,24(sp)
1c00aeba:	ca26                	sw	s1,20(sp)
1c00aebc:	c82a                	sw	a0,16(sp)
1c00aebe:	c62e                	sw	a1,12(sp)
1c00aec0:	c232                	sw	a2,4(sp)
1c00aec2:	46a2                	lw	a3,8(sp)
1c00aec4:	fa1d                	bnez	a2,1c00adfa <mm_unroll_8x1+0x49e>
1c00aec6:	4866                	lw	a6,88(sp)
1c00aec8:	5ff2                	lw	t6,60(sp)
1c00aeca:	5286                	lw	t0,96(sp)
1c00aecc:	5362                	lw	t1,56(sp)
1c00aece:	53c2                	lw	t2,48(sp)
1c00aed0:	5eb2                	lw	t4,44(sp)
1c00aed2:	4f16                	lw	t5,68(sp)
1c00aed4:	446a                	lw	s0,152(sp)
1c00aed6:	48f6                	lw	a7,92(sp)
1c00aed8:	5e0a                	lw	t3,160(sp)
1c00aeda:	54d6                	lw	s1,116(sp)
1c00aedc:	0821                	addi	a6,a6,8
1c00aede:	9efa                	add	t4,t4,t5
1c00aee0:	9f96                	add	t6,t6,t0
1c00aee2:	9316                	add	t1,t1,t0
1c00aee4:	93a2                	add	t2,t2,s0
1c00aee6:	ccc2                	sw	a6,88(sp)
1c00aee8:	c0f6                	sw	t4,64(sp)
1c00aeea:	de7e                	sw	t6,60(sp)
1c00aeec:	dc1a                	sw	t1,56(sp)
1c00aeee:	d81e                	sw	t2,48(sp)
1c00aef0:	41c887b3          	sub	a5,a7,t3
1c00aef4:	ea9843e3          	blt	a6,s1,1c00ad9a <mm_unroll_8x1+0x43e>
1c00aef8:	54fa                	lw	s1,188(sp)
1c00aefa:	da905ce3          	blez	s1,1c00acb2 <mm_unroll_8x1+0x356>
1c00aefe:	5536                	lw	a0,108(sp)
1c00af00:	55ea                	lw	a1,184(sp)
1c00af02:	00750713          	addi	a4,a0,7
1c00af06:	41f75793          	srai	a5,a4,0x1f
1c00af0a:	f837b7b3          	p.bclr	a5,a5,28,3
1c00af0e:	06e7a7db          	p.addn	a5,a5,a4,3
1c00af12:	02f58fb3          	mul	t6,a1,a5
1c00af16:	97fe                	add	a5,a5,t6
1c00af18:	04a7c3b3          	p.min	t2,a5,a0
1c00af1c:	d87fdbe3          	ble	t2,t6,1c00acb2 <mm_unroll_8x1+0x356>
1c00af20:	561a                	lw	a2,164(sp)
1c00af22:	56a6                	lw	a3,104(sp)
1c00af24:	d8c6d7e3          	ble	a2,a3,1c00acb2 <mm_unroll_8x1+0x356>
1c00af28:	5752                	lw	a4,52(sp)
1c00af2a:	8f7e                	mv	t5,t6
1c00af2c:	5e16                	lw	t3,100(sp)
1c00af2e:	52aa                	lw	t0,168(sp)
1c00af30:	42d50f33          	p.mac	t5,a0,a3
1c00af34:	00271e93          	slli	t4,a4,0x2
1c00af38:	57c6                	lw	a5,112(sp)
1c00af3a:	00251413          	slli	s0,a0,0x2
1c00af3e:	8932                	mv	s2,a2
1c00af40:	84ba                	mv	s1,a4
1c00af42:	89b6                	mv	s3,a3
1c00af44:	43df8e33          	p.mac	t3,t6,t4
1c00af48:	0f0a                	slli	t5,t5,0x2
1c00af4a:	9f3e                	add	t5,t5,a5
1c00af4c:	43d682b3          	p.mac	t0,a3,t4
1c00af50:	8716                	mv	a4,t0
1c00af52:	837a                	mv	t1,t5
1c00af54:	41390833          	sub	a6,s2,s3
1c00af58:	01c8407b          	lp.setup	x0,a6,1c00af90 <mm_unroll_8x1+0x634>
1c00af5c:	00000693          	li	a3,0
1c00af60:	00ee88b3          	add	a7,t4,a4
1c00af64:	02905463          	blez	s1,1c00af8c <mm_unroll_8x1+0x630>
1c00af68:	00ee88b3          	add	a7,t4,a4
1c00af6c:	40e887b3          	sub	a5,a7,a4
1c00af70:	17f1                	addi	a5,a5,-4
1c00af72:	00000693          	li	a3,0
1c00af76:	8389                	srli	a5,a5,0x2
1c00af78:	8672                	mv	a2,t3
1c00af7a:	0785                	addi	a5,a5,1
1c00af7c:	0067c0fb          	lp.setup	x1,a5,1c00af88 <mm_unroll_8x1+0x62c>
1c00af80:	0047250b          	p.lw	a0,4(a4!)
1c00af84:	0046258b          	p.lw	a1,4(a2!)
1c00af88:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00af8c:	00d3642b          	p.sw	a3,s0(t1!)
1c00af90:	8746                	mv	a4,a7
1c00af92:	0f85                	addi	t6,t6,1
1c00af94:	0f11                	addi	t5,t5,4
1c00af96:	9e76                	add	t3,t3,t4
1c00af98:	fbf39ce3          	bne	t2,t6,1c00af50 <mm_unroll_8x1+0x5f4>
1c00af9c:	bb19                	j	1c00acb2 <mm_unroll_8x1+0x356>
1c00af9e:	00000e13          	li	t3,0
1c00afa2:	8372                	mv	t1,t3
1c00afa4:	88f2                	mv	a7,t3
1c00afa6:	8872                	mv	a6,t3
1c00afa8:	8572                	mv	a0,t3
1c00afaa:	85f2                	mv	a1,t3
1c00afac:	8672                	mv	a2,t3
1c00afae:	86f2                	mv	a3,t3
1c00afb0:	bef9                	j	1c00ab8e <mm_unroll_8x1+0x232>
1c00afb2:	57b2                	lw	a5,44(sp)
1c00afb4:	00000e93          	li	t4,0
1c00afb8:	96be                	add	a3,a3,a5
1c00afba:	8e76                	mv	t3,t4
1c00afbc:	8376                	mv	t1,t4
1c00afbe:	88f6                	mv	a7,t4
1c00afc0:	8876                	mv	a6,t4
1c00afc2:	8576                	mv	a0,t4
1c00afc4:	85f6                	mv	a1,t4
1c00afc6:	8676                	mv	a2,t4
1c00afc8:	c436                	sw	a3,8(sp)
1c00afca:	bd4d                	j	1c00ae7c <mm_unroll_8x1+0x520>
1c00afcc:	543e                	lw	s0,236(sp)
1c00afce:	54ae                	lw	s1,232(sp)
1c00afd0:	591e                	lw	s2,228(sp)
1c00afd2:	598e                	lw	s3,224(sp)
1c00afd4:	4a7e                	lw	s4,220(sp)
1c00afd6:	4aee                	lw	s5,216(sp)
1c00afd8:	4b5e                	lw	s6,212(sp)
1c00afda:	4bce                	lw	s7,208(sp)
1c00afdc:	4c3e                	lw	s8,204(sp)
1c00afde:	4cae                	lw	s9,200(sp)
1c00afe0:	4d1e                	lw	s10,196(sp)
1c00afe2:	4d8e                	lw	s11,192(sp)
1c00afe4:	616d                	addi	sp,sp,240
1c00afe6:	cbeff06f          	j	1c00a4a4 <mm_unroll_4x1>
	...

1c00afec <mm_unroll_2x2>:
1c00afec:	454c                	lw	a1,12(a0)
1c00afee:	7159                	addi	sp,sp,-112
1c00aff0:	d6a2                	sw	s0,108(sp)
1c00aff2:	d4a6                	sw	s1,104(sp)
1c00aff4:	d2ca                	sw	s2,100(sp)
1c00aff6:	d0ce                	sw	s3,96(sp)
1c00aff8:	ced2                	sw	s4,92(sp)
1c00affa:	ccd6                	sw	s5,88(sp)
1c00affc:	cada                	sw	s6,84(sp)
1c00affe:	c8de                	sw	s7,80(sp)
1c00b000:	c6e2                	sw	s8,76(sp)
1c00b002:	c4e6                	sw	s9,72(sp)
1c00b004:	c2ea                	sw	s10,68(sp)
1c00b006:	c0ee                	sw	s11,64(sp)
1c00b008:	d42e                	sw	a1,40(sp)
1c00b00a:	473d                	li	a4,15
1c00b00c:	c005b6b3          	p.bclr	a3,a1,0,0
1c00b010:	f1402673          	csrr	a2,mhartid
1c00b014:	4ad75b63          	ble	a3,a4,1c00b4ca <mm_unroll_2x2+0x4de>
1c00b018:	00768713          	addi	a4,a3,7
1c00b01c:	f4563633          	p.bclr	a2,a2,26,5
1c00b020:	870d                	srai	a4,a4,0x3
1c00b022:	02e60bb3          	mul	s7,a2,a4
1c00b026:	410c                	lw	a1,0(a0)
1c00b028:	57a2                	lw	a5,40(sp)
1c00b02a:	da32                	sw	a2,52(sp)
1c00b02c:	d22e                	sw	a1,36(sp)
1c00b02e:	414c                	lw	a1,4(a0)
1c00b030:	4d10                	lw	a2,24(a0)
1c00b032:	8f95                	sub	a5,a5,a3
1c00b034:	ce2e                	sw	a1,28(sp)
1c00b036:	450c                	lw	a1,8(a0)
1c00b038:	975e                	add	a4,a4,s7
1c00b03a:	04d74733          	p.min	a4,a4,a3
1c00b03e:	d62e                	sw	a1,44(sp)
1c00b040:	490c                	lw	a1,16(a0)
1c00b042:	de3e                	sw	a5,60(sp)
1c00b044:	cc3a                	sw	a4,24(sp)
1c00b046:	ca2e                	sw	a1,20(sp)
1c00b048:	4940                	lw	s0,20(a0)
1c00b04a:	1c061e63          	bnez	a2,1c00b226 <mm_unroll_2x2+0x23a>
1c00b04e:	10ebdc63          	ble	a4,s7,1c00b166 <mm_unroll_2x2+0x17a>
1c00b052:	400007b7          	lui	a5,0x40000
1c00b056:	001b8713          	addi	a4,s7,1
1c00b05a:	17fd                	addi	a5,a5,-1
1c00b05c:	42e587b3          	p.mac	a5,a1,a4
1c00b060:	862e                	mv	a2,a1
1c00b062:	fff58693          	addi	a3,a1,-1
1c00b066:	00241593          	slli	a1,s0,0x2
1c00b06a:	d02e                	sw	a1,32(sp)
1c00b06c:	c00635b3          	p.bclr	a1,a2,0,0
1c00b070:	c42e                	sw	a1,8(sp)
1c00b072:	45d2                	lw	a1,20(sp)
1c00b074:	00261d93          	slli	s11,a2,0x2
1c00b078:	fc163633          	p.bclr	a2,a2,30,1
1c00b07c:	c832                	sw	a2,16(sp)
1c00b07e:	00341613          	slli	a2,s0,0x3
1c00b082:	058e                	slli	a1,a1,0x3
1c00b084:	c232                	sw	a2,4(sp)
1c00b086:	5632                	lw	a2,44(sp)
1c00b088:	c02e                	sw	a1,0(sp)
1c00b08a:	45f2                	lw	a1,28(sp)
1c00b08c:	078a                	slli	a5,a5,0x2
1c00b08e:	97b2                	add	a5,a5,a2
1c00b090:	068a                	slli	a3,a3,0x2
1c00b092:	96ae                	add	a3,a3,a1
1c00b094:	c63e                	sw	a5,12(sp)
1c00b096:	5792                	lw	a5,36(sp)
1c00b098:	d836                	sw	a3,48(sp)
1c00b09a:	8b22                	mv	s6,s0
1c00b09c:	5682                	lw	a3,32(sp)
1c00b09e:	5a92                	lw	s5,36(sp)
1c00b0a0:	43740b33          	p.mac	s6,s0,s7
1c00b0a4:	8d32                	mv	s10,a2
1c00b0a6:	8cb2                	mv	s9,a2
1c00b0a8:	8c3e                	mv	s8,a5
1c00b0aa:	43768ab3          	p.mac	s5,a3,s7
1c00b0ae:	0b0a                	slli	s6,s6,0x2
1c00b0b0:	9b3e                	add	s6,s6,a5
1c00b0b2:	437d8d33          	p.mac	s10,s11,s7
1c00b0b6:	42ed8cb3          	p.mac	s9,s11,a4
1c00b0ba:	42e68c33          	p.mac	s8,a3,a4
1c00b0be:	45a2                	lw	a1,8(sp)
1c00b0c0:	c1d1                	beqz	a1,1c00b144 <mm_unroll_2x2+0x158>
1c00b0c2:	fff58393          	addi	t2,a1,-1
1c00b0c6:	0013d393          	srli	t2,t2,0x1
1c00b0ca:	00158793          	addi	a5,a1,1
1c00b0ce:	4709                	li	a4,2
1c00b0d0:	42f2                	lw	t0,28(sp)
1c00b0d2:	004d0913          	addi	s2,s10,4
1c00b0d6:	004c8493          	addi	s1,s9,4
1c00b0da:	89e6                	mv	s3,s9
1c00b0dc:	8a6a                	mv	s4,s10
1c00b0de:	0385                	addi	t2,t2,1
1c00b0e0:	40e7e663          	bltu	a5,a4,1c00b4ec <mm_unroll_2x2+0x500>
1c00b0e4:	30805663          	blez	s0,1c00b3f0 <mm_unroll_2x2+0x404>
1c00b0e8:	00000893          	li	a7,0
1c00b0ec:	415b07b3          	sub	a5,s6,s5
1c00b0f0:	17f1                	addi	a5,a5,-4
1c00b0f2:	8846                	mv	a6,a7
1c00b0f4:	8546                	mv	a0,a7
1c00b0f6:	85c6                	mv	a1,a7
1c00b0f8:	8389                	srli	a5,a5,0x2
1c00b0fa:	00428f13          	addi	t5,t0,4
1c00b0fe:	8ee2                	mv	t4,s8
1c00b100:	8e16                	mv	t3,t0
1c00b102:	8356                	mv	t1,s5
1c00b104:	0785                	addi	a5,a5,1
1c00b106:	0001                	nop
1c00b108:	0107c0fb          	lp.setup	x1,a5,1c00b128 <mm_unroll_2x2+0x13c>
1c00b10c:	00432f8b          	p.lw	t6,4(t1!)
1c00b110:	21be760b          	p.lw	a2,s11(t3!)
1c00b114:	21bf768b          	p.lw	a3,s11(t5!)
1c00b118:	004ea70b          	p.lw	a4,4(t4!)
1c00b11c:	58cff5c3          	fmadd.s	a1,t6,a2,a1
1c00b120:	50dff543          	fmadd.s	a0,t6,a3,a0
1c00b124:	80e67843          	fmadd.s	a6,a2,a4,a6
1c00b128:	88e6f8c3          	fmadd.s	a7,a3,a4,a7
1c00b12c:	00ba242b          	p.sw	a1,8(s4!)
1c00b130:	00a9242b          	p.sw	a0,8(s2!)
1c00b134:	0109a42b          	p.sw	a6,8(s3!)
1c00b138:	0114a42b          	p.sw	a7,8(s1!)
1c00b13c:	02a1                	addi	t0,t0,8
1c00b13e:	13fd                	addi	t2,t2,-1
1c00b140:	fa0392e3          	bnez	t2,1c00b0e4 <mm_unroll_2x2+0xf8>
1c00b144:	45c2                	lw	a1,16(sp)
1c00b146:	32059e63          	bnez	a1,1c00b482 <mm_unroll_2x2+0x496>
1c00b14a:	4682                	lw	a3,0(sp)
1c00b14c:	4632                	lw	a2,12(sp)
1c00b14e:	4712                	lw	a4,4(sp)
1c00b150:	47e2                	lw	a5,24(sp)
1c00b152:	9636                	add	a2,a2,a3
1c00b154:	0b89                	addi	s7,s7,2
1c00b156:	c632                	sw	a2,12(sp)
1c00b158:	9aba                	add	s5,s5,a4
1c00b15a:	9d36                	add	s10,s10,a3
1c00b15c:	9cb6                	add	s9,s9,a3
1c00b15e:	9c3a                	add	s8,s8,a4
1c00b160:	9b3a                	add	s6,s6,a4
1c00b162:	f4fbcee3          	blt	s7,a5,1c00b0be <mm_unroll_2x2+0xd2>
1c00b166:	5672                	lw	a2,60(sp)
1c00b168:	0ac05163          	blez	a2,1c00b20a <mm_unroll_2x2+0x21e>
1c00b16c:	46d2                	lw	a3,20(sp)
1c00b16e:	55d2                	lw	a1,52(sp)
1c00b170:	00768713          	addi	a4,a3,7
1c00b174:	41f75793          	srai	a5,a4,0x1f
1c00b178:	f837b7b3          	p.bclr	a5,a5,28,3
1c00b17c:	06e7a7db          	p.addn	a5,a5,a4,3
1c00b180:	02f58733          	mul	a4,a1,a5
1c00b184:	97ba                	add	a5,a5,a4
1c00b186:	04d7c7b3          	p.min	a5,a5,a3
1c00b18a:	08f75063          	ble	a5,a4,1c00b20a <mm_unroll_2x2+0x21e>
1c00b18e:	85b6                	mv	a1,a3
1c00b190:	56a2                	lw	a3,40(sp)
1c00b192:	00271313          	slli	t1,a4,0x2
1c00b196:	00279813          	slli	a6,a5,0x2
1c00b19a:	fff68613          	addi	a2,a3,-1
1c00b19e:	028606b3          	mul	a3,a2,s0
1c00b1a2:	40680833          	sub	a6,a6,t1
1c00b1a6:	47f2                	lw	a5,28(sp)
1c00b1a8:	1871                	addi	a6,a6,-4
1c00b1aa:	00285813          	srli	a6,a6,0x2
1c00b1ae:	933e                	add	t1,t1,a5
1c00b1b0:	00259f13          	slli	t5,a1,0x2
1c00b1b4:	0805                	addi	a6,a6,1
1c00b1b6:	42b60733          	p.mac	a4,a2,a1
1c00b1ba:	5612                	lw	a2,36(sp)
1c00b1bc:	00d40eb3          	add	t4,s0,a3
1c00b1c0:	068a                	slli	a3,a3,0x2
1c00b1c2:	00d60e33          	add	t3,a2,a3
1c00b1c6:	56b2                	lw	a3,44(sp)
1c00b1c8:	0e8a                	slli	t4,t4,0x2
1c00b1ca:	9eb2                	add	t4,t4,a2
1c00b1cc:	00271893          	slli	a7,a4,0x2
1c00b1d0:	98b6                	add	a7,a7,a3
1c00b1d2:	0001                	nop
1c00b1d4:	01a8407b          	lp.setup	x0,a6,1c00b208 <mm_unroll_2x2+0x21c>
1c00b1d8:	00000713          	li	a4,0
1c00b1dc:	02805463          	blez	s0,1c00b204 <mm_unroll_2x2+0x218>
1c00b1e0:	41ce87b3          	sub	a5,t4,t3
1c00b1e4:	17f1                	addi	a5,a5,-4
1c00b1e6:	00000713          	li	a4,0
1c00b1ea:	8389                	srli	a5,a5,0x2
1c00b1ec:	861a                	mv	a2,t1
1c00b1ee:	86f2                	mv	a3,t3
1c00b1f0:	0785                	addi	a5,a5,1
1c00b1f2:	0001                	nop
1c00b1f4:	0067c0fb          	lp.setup	x1,a5,1c00b200 <mm_unroll_2x2+0x214>
1c00b1f8:	0046a50b          	p.lw	a0,4(a3!)
1c00b1fc:	21e6758b          	p.lw	a1,t5(a2!)
1c00b200:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00b204:	00e8a22b          	p.sw	a4,4(a7!)
1c00b208:	0311                	addi	t1,t1,4
1c00b20a:	5436                	lw	s0,108(sp)
1c00b20c:	54a6                	lw	s1,104(sp)
1c00b20e:	5916                	lw	s2,100(sp)
1c00b210:	5986                	lw	s3,96(sp)
1c00b212:	4a76                	lw	s4,92(sp)
1c00b214:	4ae6                	lw	s5,88(sp)
1c00b216:	4b56                	lw	s6,84(sp)
1c00b218:	4bc6                	lw	s7,80(sp)
1c00b21a:	4c36                	lw	s8,76(sp)
1c00b21c:	4ca6                	lw	s9,72(sp)
1c00b21e:	4d16                	lw	s10,68(sp)
1c00b220:	4d86                	lw	s11,64(sp)
1c00b222:	6165                	addi	sp,sp,112
1c00b224:	8082                	ret
1c00b226:	12ebd463          	ble	a4,s7,1c00b34e <mm_unroll_2x2+0x362>
1c00b22a:	400006b7          	lui	a3,0x40000
1c00b22e:	001b8713          	addi	a4,s7,1
1c00b232:	16fd                	addi	a3,a3,-1
1c00b234:	42e586b3          	p.mac	a3,a1,a4
1c00b238:	862e                	mv	a2,a1
1c00b23a:	fff58793          	addi	a5,a1,-1
1c00b23e:	058a                	slli	a1,a1,0x2
1c00b240:	d82e                	sw	a1,48(sp)
1c00b242:	45d2                	lw	a1,20(sp)
1c00b244:	8c22                	mv	s8,s0
1c00b246:	c0063633          	p.bclr	a2,a2,0,0
1c00b24a:	fc15b5b3          	p.bclr	a1,a1,30,1
1c00b24e:	d02e                	sw	a1,32(sp)
1c00b250:	028787b3          	mul	a5,a5,s0
1c00b254:	55b2                	lw	a1,44(sp)
1c00b256:	068a                	slli	a3,a3,0x2
1c00b258:	c432                	sw	a2,8(sp)
1c00b25a:	96ae                	add	a3,a3,a1
1c00b25c:	c836                	sw	a3,16(sp)
1c00b25e:	46f2                	lw	a3,28(sp)
1c00b260:	4652                	lw	a2,20(sp)
1c00b262:	8dae                	mv	s11,a1
1c00b264:	5c92                	lw	s9,36(sp)
1c00b266:	43740c33          	p.mac	s8,s0,s7
1c00b26a:	078a                	slli	a5,a5,0x2
1c00b26c:	97b6                	add	a5,a5,a3
1c00b26e:	dc3e                	sw	a5,56(sp)
1c00b270:	57c2                	lw	a5,48(sp)
1c00b272:	060e                	slli	a2,a2,0x3
1c00b274:	c232                	sw	a2,4(sp)
1c00b276:	5612                	lw	a2,36(sp)
1c00b278:	00241d13          	slli	s10,s0,0x2
1c00b27c:	c05e                	sw	s7,0(sp)
1c00b27e:	42fb8db3          	p.mac	s11,s7,a5
1c00b282:	87ae                	mv	a5,a1
1c00b284:	55c2                	lw	a1,48(sp)
1c00b286:	0c0a                	slli	s8,s8,0x2
1c00b288:	9c32                	add	s8,s8,a2
1c00b28a:	00341b13          	slli	s6,s0,0x3
1c00b28e:	43ab8633          	p.mac	a2,s7,s10
1c00b292:	42b707b3          	p.mac	a5,a4,a1
1c00b296:	c632                	sw	a2,12(sp)
1c00b298:	8ab2                	mv	s5,a2
1c00b29a:	43a70cb3          	p.mac	s9,a4,s10
1c00b29e:	8bbe                	mv	s7,a5
1c00b2a0:	47a2                	lw	a5,8(sp)
1c00b2a2:	c3c9                	beqz	a5,1c00b324 <mm_unroll_2x2+0x338>
1c00b2a4:	fff78393          	addi	t2,a5,-1 # 3fffffff <__l2_shared_end+0x23fe7af3>
1c00b2a8:	0013d393          	srli	t2,t2,0x1
1c00b2ac:	0785                	addi	a5,a5,1
1c00b2ae:	4709                	li	a4,2
1c00b2b0:	42f2                	lw	t0,28(sp)
1c00b2b2:	004d8913          	addi	s2,s11,4
1c00b2b6:	004b8493          	addi	s1,s7,4
1c00b2ba:	89de                	mv	s3,s7
1c00b2bc:	8a6e                	mv	s4,s11
1c00b2be:	0385                	addi	t2,t2,1
1c00b2c0:	22e7e463          	bltu	a5,a4,1c00b4e8 <mm_unroll_2x2+0x4fc>
1c00b2c4:	14805863          	blez	s0,1c00b414 <mm_unroll_2x2+0x428>
1c00b2c8:	00000893          	li	a7,0
1c00b2cc:	415c07b3          	sub	a5,s8,s5
1c00b2d0:	17f1                	addi	a5,a5,-4
1c00b2d2:	8846                	mv	a6,a7
1c00b2d4:	8546                	mv	a0,a7
1c00b2d6:	85c6                	mv	a1,a7
1c00b2d8:	8389                	srli	a5,a5,0x2
1c00b2da:	01a28f33          	add	t5,t0,s10
1c00b2de:	8ee6                	mv	t4,s9
1c00b2e0:	8e16                	mv	t3,t0
1c00b2e2:	8356                	mv	t1,s5
1c00b2e4:	0785                	addi	a5,a5,1
1c00b2e6:	0001                	nop
1c00b2e8:	0107c0fb          	lp.setup	x1,a5,1c00b308 <mm_unroll_2x2+0x31c>
1c00b2ec:	00432f8b          	p.lw	t6,4(t1!)
1c00b2f0:	004e260b          	p.lw	a2,4(t3!)
1c00b2f4:	004f268b          	p.lw	a3,4(t5!)
1c00b2f8:	004ea70b          	p.lw	a4,4(t4!)
1c00b2fc:	58cff5c3          	fmadd.s	a1,t6,a2,a1
1c00b300:	50dff543          	fmadd.s	a0,t6,a3,a0
1c00b304:	80e67843          	fmadd.s	a6,a2,a4,a6
1c00b308:	88e6f8c3          	fmadd.s	a7,a3,a4,a7
1c00b30c:	00ba242b          	p.sw	a1,8(s4!)
1c00b310:	00a9242b          	p.sw	a0,8(s2!)
1c00b314:	0109a42b          	p.sw	a6,8(s3!)
1c00b318:	0114a42b          	p.sw	a7,8(s1!)
1c00b31c:	92da                	add	t0,t0,s6
1c00b31e:	13fd                	addi	t2,t2,-1
1c00b320:	fa0392e3          	bnez	t2,1c00b2c4 <mm_unroll_2x2+0x2d8>
1c00b324:	5702                	lw	a4,32(sp)
1c00b326:	10071963          	bnez	a4,1c00b438 <mm_unroll_2x2+0x44c>
1c00b32a:	4612                	lw	a2,4(sp)
1c00b32c:	4782                	lw	a5,0(sp)
1c00b32e:	45c2                	lw	a1,16(sp)
1c00b330:	46b2                	lw	a3,12(sp)
1c00b332:	4762                	lw	a4,24(sp)
1c00b334:	0789                	addi	a5,a5,2
1c00b336:	95b2                	add	a1,a1,a2
1c00b338:	96da                	add	a3,a3,s6
1c00b33a:	c03e                	sw	a5,0(sp)
1c00b33c:	c82e                	sw	a1,16(sp)
1c00b33e:	c636                	sw	a3,12(sp)
1c00b340:	9db2                	add	s11,s11,a2
1c00b342:	9bb2                	add	s7,s7,a2
1c00b344:	9ada                	add	s5,s5,s6
1c00b346:	9cda                	add	s9,s9,s6
1c00b348:	9c5a                	add	s8,s8,s6
1c00b34a:	f4e7cbe3          	blt	a5,a4,1c00b2a0 <mm_unroll_2x2+0x2b4>
1c00b34e:	55f2                	lw	a1,60(sp)
1c00b350:	eab05de3          	blez	a1,1c00b20a <mm_unroll_2x2+0x21e>
1c00b354:	4652                	lw	a2,20(sp)
1c00b356:	00760713          	addi	a4,a2,7
1c00b35a:	41f75793          	srai	a5,a4,0x1f
1c00b35e:	f837b7b3          	p.bclr	a5,a5,28,3
1c00b362:	06e7a7db          	p.addn	a5,a5,a4,3
1c00b366:	5752                	lw	a4,52(sp)
1c00b368:	02f706b3          	mul	a3,a4,a5
1c00b36c:	97b6                	add	a5,a5,a3
1c00b36e:	04c7c7b3          	p.min	a5,a5,a2
1c00b372:	e8f6dce3          	ble	a5,a3,1c00b20a <mm_unroll_2x2+0x21e>
1c00b376:	55a2                	lw	a1,40(sp)
1c00b378:	4872                	lw	a6,28(sp)
1c00b37a:	00241e13          	slli	t3,s0,0x2
1c00b37e:	fff58713          	addi	a4,a1,-1
1c00b382:	02e608b3          	mul	a7,a2,a4
1c00b386:	5632                	lw	a2,44(sp)
1c00b388:	02e40733          	mul	a4,s0,a4
1c00b38c:	00d88eb3          	add	t4,a7,a3
1c00b390:	98be                	add	a7,a7,a5
1c00b392:	0e8a                	slli	t4,t4,0x2
1c00b394:	088a                	slli	a7,a7,0x2
1c00b396:	41d888b3          	sub	a7,a7,t4
1c00b39a:	18f1                	addi	a7,a7,-4
1c00b39c:	0028d893          	srli	a7,a7,0x2
1c00b3a0:	9eb2                	add	t4,t4,a2
1c00b3a2:	0885                	addi	a7,a7,1
1c00b3a4:	43c68833          	p.mac	a6,a3,t3
1c00b3a8:	5692                	lw	a3,36(sp)
1c00b3aa:	00870f33          	add	t5,a4,s0
1c00b3ae:	0f0a                	slli	t5,t5,0x2
1c00b3b0:	070a                	slli	a4,a4,0x2
1c00b3b2:	00e68333          	add	t1,a3,a4
1c00b3b6:	9f36                	add	t5,t5,a3
1c00b3b8:	01a8c07b          	lp.setup	x0,a7,1c00b3ec <mm_unroll_2x2+0x400>
1c00b3bc:	00000713          	li	a4,0
1c00b3c0:	02805463          	blez	s0,1c00b3e8 <mm_unroll_2x2+0x3fc>
1c00b3c4:	406f07b3          	sub	a5,t5,t1
1c00b3c8:	17f1                	addi	a5,a5,-4
1c00b3ca:	00000713          	li	a4,0
1c00b3ce:	8389                	srli	a5,a5,0x2
1c00b3d0:	8642                	mv	a2,a6
1c00b3d2:	869a                	mv	a3,t1
1c00b3d4:	0785                	addi	a5,a5,1
1c00b3d6:	0001                	nop
1c00b3d8:	0067c0fb          	lp.setup	x1,a5,1c00b3e4 <mm_unroll_2x2+0x3f8>
1c00b3dc:	0046a50b          	p.lw	a0,4(a3!) # 40000004 <__l2_shared_end+0x23fe7af8>
1c00b3e0:	0046258b          	p.lw	a1,4(a2!)
1c00b3e4:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00b3e8:	00eea22b          	p.sw	a4,4(t4!)
1c00b3ec:	9872                	add	a6,a6,t3
1c00b3ee:	bd31                	j	1c00b20a <mm_unroll_2x2+0x21e>
1c00b3f0:	00000893          	li	a7,0
1c00b3f4:	02a1                	addi	t0,t0,8
1c00b3f6:	85c6                	mv	a1,a7
1c00b3f8:	8546                	mv	a0,a7
1c00b3fa:	8846                	mv	a6,a7
1c00b3fc:	00ba242b          	p.sw	a1,8(s4!)
1c00b400:	00a9242b          	p.sw	a0,8(s2!)
1c00b404:	0109a42b          	p.sw	a6,8(s3!)
1c00b408:	0114a42b          	p.sw	a7,8(s1!)
1c00b40c:	13fd                	addi	t2,t2,-1
1c00b40e:	cc039be3          	bnez	t2,1c00b0e4 <mm_unroll_2x2+0xf8>
1c00b412:	bb0d                	j	1c00b144 <mm_unroll_2x2+0x158>
1c00b414:	00000893          	li	a7,0
1c00b418:	92da                	add	t0,t0,s6
1c00b41a:	85c6                	mv	a1,a7
1c00b41c:	8546                	mv	a0,a7
1c00b41e:	8846                	mv	a6,a7
1c00b420:	00ba242b          	p.sw	a1,8(s4!)
1c00b424:	00a9242b          	p.sw	a0,8(s2!)
1c00b428:	0109a42b          	p.sw	a6,8(s3!)
1c00b42c:	0114a42b          	p.sw	a7,8(s1!)
1c00b430:	13fd                	addi	t2,t2,-1
1c00b432:	e80399e3          	bnez	t2,1c00b2c4 <mm_unroll_2x2+0x2d8>
1c00b436:	b5fd                	j	1c00b324 <mm_unroll_2x2+0x338>
1c00b438:	4732                	lw	a4,12(sp)
1c00b43a:	4642                	lw	a2,16(sp)
1c00b43c:	5ec2                	lw	t4,48(sp)
1c00b43e:	5e62                	lw	t3,56(sp)
1c00b440:	4589                	li	a1,2
1c00b442:	00000693          	li	a3,0
1c00b446:	01a70333          	add	t1,a4,s10
1c00b44a:	02805563          	blez	s0,1c00b474 <mm_unroll_2x2+0x488>
1c00b44e:	01a70333          	add	t1,a4,s10
1c00b452:	40e307b3          	sub	a5,t1,a4
1c00b456:	17f1                	addi	a5,a5,-4
1c00b458:	00000693          	li	a3,0
1c00b45c:	8389                	srli	a5,a5,0x2
1c00b45e:	8572                	mv	a0,t3
1c00b460:	0785                	addi	a5,a5,1
1c00b462:	0001                	nop
1c00b464:	0067c0fb          	lp.setup	x1,a5,1c00b470 <mm_unroll_2x2+0x484>
1c00b468:	0047288b          	p.lw	a7,4(a4!)
1c00b46c:	0045280b          	p.lw	a6,4(a0!)
1c00b470:	6908f6c3          	fmadd.s	a3,a7,a6,a3
1c00b474:	00d66eab          	p.sw	a3,t4(a2!)
1c00b478:	871a                	mv	a4,t1
1c00b47a:	ea15a8e3          	p.beqimm	a1,1,1c00b32a <mm_unroll_2x2+0x33e>
1c00b47e:	4585                	li	a1,1
1c00b480:	b7c9                	j	1c00b442 <mm_unroll_2x2+0x456>
1c00b482:	45b2                	lw	a1,12(sp)
1c00b484:	5e02                	lw	t3,32(sp)
1c00b486:	5ec2                	lw	t4,48(sp)
1c00b488:	8756                	mv	a4,s5
1c00b48a:	4509                	li	a0,2
1c00b48c:	00000693          	li	a3,0
1c00b490:	00ee0333          	add	t1,t3,a4
1c00b494:	02805463          	blez	s0,1c00b4bc <mm_unroll_2x2+0x4d0>
1c00b498:	00ee0333          	add	t1,t3,a4
1c00b49c:	40e307b3          	sub	a5,t1,a4
1c00b4a0:	17f1                	addi	a5,a5,-4
1c00b4a2:	00000693          	li	a3,0
1c00b4a6:	8389                	srli	a5,a5,0x2
1c00b4a8:	8676                	mv	a2,t4
1c00b4aa:	0785                	addi	a5,a5,1
1c00b4ac:	0067c0fb          	lp.setup	x1,a5,1c00b4b8 <mm_unroll_2x2+0x4cc>
1c00b4b0:	0047288b          	p.lw	a7,4(a4!)
1c00b4b4:	21b6780b          	p.lw	a6,s11(a2!)
1c00b4b8:	6908f6c3          	fmadd.s	a3,a7,a6,a3
1c00b4bc:	00d5edab          	p.sw	a3,s11(a1!)
1c00b4c0:	871a                	mv	a4,t1
1c00b4c2:	c81524e3          	p.beqimm	a0,1,1c00b14a <mm_unroll_2x2+0x15e>
1c00b4c6:	4505                	li	a0,1
1c00b4c8:	b7d1                	j	1c00b48c <mm_unroll_2x2+0x4a0>
1c00b4ca:	5436                	lw	s0,108(sp)
1c00b4cc:	54a6                	lw	s1,104(sp)
1c00b4ce:	5916                	lw	s2,100(sp)
1c00b4d0:	5986                	lw	s3,96(sp)
1c00b4d2:	4a76                	lw	s4,92(sp)
1c00b4d4:	4ae6                	lw	s5,88(sp)
1c00b4d6:	4b56                	lw	s6,84(sp)
1c00b4d8:	4bc6                	lw	s7,80(sp)
1c00b4da:	4c36                	lw	s8,76(sp)
1c00b4dc:	4ca6                	lw	s9,72(sp)
1c00b4de:	4d16                	lw	s10,68(sp)
1c00b4e0:	4d86                	lw	s11,64(sp)
1c00b4e2:	6165                	addi	sp,sp,112
1c00b4e4:	fccfe06f          	j	1c009cb0 <mm_unroll_1x8>
1c00b4e8:	4385                	li	t2,1
1c00b4ea:	bbe9                	j	1c00b2c4 <mm_unroll_2x2+0x2d8>
1c00b4ec:	4385                	li	t2,1
1c00b4ee:	bedd                	j	1c00b0e4 <mm_unroll_2x2+0xf8>

1c00b4f0 <mm_unroll_2x4>:
1c00b4f0:	454c                	lw	a1,12(a0)
1c00b4f2:	7175                	addi	sp,sp,-144
1c00b4f4:	c722                	sw	s0,140(sp)
1c00b4f6:	c526                	sw	s1,136(sp)
1c00b4f8:	c34a                	sw	s2,132(sp)
1c00b4fa:	c14e                	sw	s3,128(sp)
1c00b4fc:	ded2                	sw	s4,124(sp)
1c00b4fe:	dcd6                	sw	s5,120(sp)
1c00b500:	dada                	sw	s6,116(sp)
1c00b502:	d8de                	sw	s7,112(sp)
1c00b504:	d6e2                	sw	s8,108(sp)
1c00b506:	d4e6                	sw	s9,104(sp)
1c00b508:	d2ea                	sw	s10,100(sp)
1c00b50a:	d0ee                	sw	s11,96(sp)
1c00b50c:	de2e                	sw	a1,60(sp)
1c00b50e:	46bd                	li	a3,15
1c00b510:	c005b733          	p.bclr	a4,a1,0,0
1c00b514:	f1402673          	csrr	a2,mhartid
1c00b518:	54e6d963          	ble	a4,a3,1c00ba6a <mm_unroll_2x4+0x57a>
1c00b51c:	4914                	lw	a3,16(a0)
1c00b51e:	d436                	sw	a3,40(sp)
1c00b520:	55a2                	lw	a1,40(sp)
1c00b522:	468d                	li	a3,3
1c00b524:	56b6d263          	ble	a1,a3,1c00ba88 <mm_unroll_2x4+0x598>
1c00b528:	00770693          	addi	a3,a4,7
1c00b52c:	868d                	srai	a3,a3,0x3
1c00b52e:	f4563633          	p.bclr	a2,a2,26,5
1c00b532:	c8b2                	sw	a2,80(sp)
1c00b534:	02d60633          	mul	a2,a2,a3
1c00b538:	410c                	lw	a1,0(a0)
1c00b53a:	57f2                	lw	a5,60(sp)
1c00b53c:	01452c03          	lw	s8,20(a0)
1c00b540:	c2ae                	sw	a1,68(sp)
1c00b542:	414c                	lw	a1,4(a0)
1c00b544:	8f99                	sub	a5,a5,a4
1c00b546:	cabe                	sw	a5,84(sp)
1c00b548:	d82e                	sw	a1,48(sp)
1c00b54a:	450c                	lw	a1,8(a0)
1c00b54c:	cc32                	sw	a2,24(sp)
1c00b54e:	4d10                	lw	a2,24(a0)
1c00b550:	c6ae                	sw	a1,76(sp)
1c00b552:	45e2                	lw	a1,24(sp)
1c00b554:	96ae                	add	a3,a3,a1
1c00b556:	04e6c6b3          	p.min	a3,a3,a4
1c00b55a:	da36                	sw	a3,52(sp)
1c00b55c:	22060163          	beqz	a2,1c00b77e <mm_unroll_2x4+0x28e>
1c00b560:	16d5d463          	ble	a3,a1,1c00b6c8 <mm_unroll_2x4+0x1d8>
1c00b564:	5622                	lw	a2,40(sp)
1c00b566:	56a2                	lw	a3,40(sp)
1c00b568:	57a2                	lw	a5,40(sp)
1c00b56a:	c2063633          	p.bclr	a2,a2,1,0
1c00b56e:	068a                	slli	a3,a3,0x2
1c00b570:	c0b6                	sw	a3,64(sp)
1c00b572:	ce32                	sw	a2,28(sp)
1c00b574:	40f606b3          	sub	a3,a2,a5
1c00b578:	003c1613          	slli	a2,s8,0x3
1c00b57c:	d232                	sw	a2,36(sp)
1c00b57e:	4662                	lw	a2,24(sp)
1c00b580:	00158713          	addi	a4,a1,1
1c00b584:	fa27b5b3          	p.bclr	a1,a5,29,2
1c00b588:	078e                	slli	a5,a5,0x3
1c00b58a:	dc3e                	sw	a5,56(sp)
1c00b58c:	87e2                	mv	a5,s8
1c00b58e:	42cc07b3          	p.mac	a5,s8,a2
1c00b592:	c4ae                	sw	a1,72(sp)
1c00b594:	004c1593          	slli	a1,s8,0x4
1c00b598:	ca2e                	sw	a1,20(sp)
1c00b59a:	4596                	lw	a1,68(sp)
1c00b59c:	068a                	slli	a3,a3,0x2
1c00b59e:	ccb6                	sw	a3,88(sp)
1c00b5a0:	46b6                	lw	a3,76(sp)
1c00b5a2:	4d16                	lw	s10,68(sp)
1c00b5a4:	002c1d93          	slli	s11,s8,0x2
1c00b5a8:	078a                	slli	a5,a5,0x2
1c00b5aa:	97ae                	add	a5,a5,a1
1c00b5ac:	c43e                	sw	a5,8(sp)
1c00b5ae:	4786                	lw	a5,64(sp)
1c00b5b0:	42cd8d33          	p.mac	s10,s11,a2
1c00b5b4:	45b6                	lw	a1,76(sp)
1c00b5b6:	c862                	sw	s8,16(sp)
1c00b5b8:	42c786b3          	p.mac	a3,a5,a2
1c00b5bc:	4606                	lw	a2,64(sp)
1c00b5be:	47f2                	lw	a5,28(sp)
1c00b5c0:	d036                	sw	a3,32(sp)
1c00b5c2:	4696                	lw	a3,68(sp)
1c00b5c4:	42e605b3          	p.mac	a1,a2,a4
1c00b5c8:	42ed86b3          	p.mac	a3,s11,a4
1c00b5cc:	5742                	lw	a4,48(sp)
1c00b5ce:	d62e                	sw	a1,44(sp)
1c00b5d0:	42fd8733          	p.mac	a4,s11,a5
1c00b5d4:	c636                	sw	a3,12(sp)
1c00b5d6:	ceba                	sw	a4,92(sp)
1c00b5d8:	47f2                	lw	a5,28(sp)
1c00b5da:	cfcd                	beqz	a5,1c00b694 <mm_unroll_2x4+0x1a4>
1c00b5dc:	fff78c93          	addi	s9,a5,-1
1c00b5e0:	002cdc93          	srli	s9,s9,0x2
1c00b5e4:	078d                	addi	a5,a5,3
1c00b5e6:	4711                	li	a4,4
1c00b5e8:	5c42                	lw	s8,48(sp)
1c00b5ea:	5b32                	lw	s6,44(sp)
1c00b5ec:	5b82                	lw	s7,32(sp)
1c00b5ee:	0c85                	addi	s9,s9,1
1c00b5f0:	4ae7ed63          	bltu	a5,a4,1c00baaa <mm_unroll_2x4+0x5ba>
1c00b5f4:	04fcc07b          	lp.setup	x0,s9,1c00b692 <mm_unroll_2x4+0x1a2>
1c00b5f8:	45c2                	lw	a1,16(sp)
1c00b5fa:	36b05763          	blez	a1,1c00b968 <mm_unroll_2x4+0x478>
1c00b5fe:	4722                	lw	a4,8(sp)
1c00b600:	00000393          	li	t2,0
1c00b604:	018d84b3          	add	s1,s11,s8
1c00b608:	41a706b3          	sub	a3,a4,s10
1c00b60c:	16f1                	addi	a3,a3,-4
1c00b60e:	829e                	mv	t0,t2
1c00b610:	8f9e                	mv	t6,t2
1c00b612:	8f1e                	mv	t5,t2
1c00b614:	8e9e                	mv	t4,t2
1c00b616:	8e1e                	mv	t3,t2
1c00b618:	831e                	mv	t1,t2
1c00b61a:	889e                	mv	a7,t2
1c00b61c:	4a32                	lw	s4,12(sp)
1c00b61e:	009d8433          	add	s0,s11,s1
1c00b622:	8289                	srli	a3,a3,0x2
1c00b624:	008d8ab3          	add	s5,s11,s0
1c00b628:	89e2                	mv	s3,s8
1c00b62a:	896a                	mv	s2,s10
1c00b62c:	0685                	addi	a3,a3,1
1c00b62e:	0001                	nop
1c00b630:	01c6c0fb          	lp.setup	x1,a3,1c00b668 <mm_unroll_2x4+0x178>
1c00b634:	0049270b          	p.lw	a4,4(s2!)
1c00b638:	004a278b          	p.lw	a5,4(s4!)
1c00b63c:	0049a80b          	p.lw	a6,4(s3!)
1c00b640:	0044a50b          	p.lw	a0,4(s1!)
1c00b644:	0044258b          	p.lw	a1,4(s0!)
1c00b648:	004aa60b          	p.lw	a2,4(s5!)
1c00b64c:	890778c3          	fmadd.s	a7,a4,a6,a7
1c00b650:	30a77343          	fmadd.s	t1,a4,a0,t1
1c00b654:	e0b77e43          	fmadd.s	t3,a4,a1,t3
1c00b658:	e8c77ec3          	fmadd.s	t4,a4,a2,t4
1c00b65c:	f0f87f43          	fmadd.s	t5,a6,a5,t5
1c00b660:	f8f57fc3          	fmadd.s	t6,a0,a5,t6
1c00b664:	28f5f2c3          	fmadd.s	t0,a1,a5,t0
1c00b668:	38f673c3          	fmadd.s	t2,a2,a5,t2
1c00b66c:	47d2                	lw	a5,20(sp)
1c00b66e:	011ba023          	sw	a7,0(s7)
1c00b672:	006ba223          	sw	t1,4(s7)
1c00b676:	01cba423          	sw	t3,8(s7)
1c00b67a:	01dba623          	sw	t4,12(s7)
1c00b67e:	01eb2023          	sw	t5,0(s6)
1c00b682:	01fb2223          	sw	t6,4(s6)
1c00b686:	005b2423          	sw	t0,8(s6)
1c00b68a:	007b2623          	sw	t2,12(s6)
1c00b68e:	0bc1                	addi	s7,s7,16
1c00b690:	0b41                	addi	s6,s6,16
1c00b692:	9c3e                	add	s8,s8,a5
1c00b694:	4626                	lw	a2,72(sp)
1c00b696:	34061e63          	bnez	a2,1c00b9f2 <mm_unroll_2x4+0x502>
1c00b69a:	56e2                	lw	a3,56(sp)
1c00b69c:	47e2                	lw	a5,24(sp)
1c00b69e:	5602                	lw	a2,32(sp)
1c00b6a0:	5732                	lw	a4,44(sp)
1c00b6a2:	0789                	addi	a5,a5,2
1c00b6a4:	9636                	add	a2,a2,a3
1c00b6a6:	5592                	lw	a1,36(sp)
1c00b6a8:	cc3e                	sw	a5,24(sp)
1c00b6aa:	d032                	sw	a2,32(sp)
1c00b6ac:	47b2                	lw	a5,12(sp)
1c00b6ae:	4622                	lw	a2,8(sp)
1c00b6b0:	9736                	add	a4,a4,a3
1c00b6b2:	d63a                	sw	a4,44(sp)
1c00b6b4:	56d2                	lw	a3,52(sp)
1c00b6b6:	4762                	lw	a4,24(sp)
1c00b6b8:	97ae                	add	a5,a5,a1
1c00b6ba:	962e                	add	a2,a2,a1
1c00b6bc:	c63e                	sw	a5,12(sp)
1c00b6be:	c432                	sw	a2,8(sp)
1c00b6c0:	9d2e                	add	s10,s10,a1
1c00b6c2:	f0d74be3          	blt	a4,a3,1c00b5d8 <mm_unroll_2x4+0xe8>
1c00b6c6:	4c42                	lw	s8,16(sp)
1c00b6c8:	45d6                	lw	a1,84(sp)
1c00b6ca:	08b05c63          	blez	a1,1c00b762 <mm_unroll_2x4+0x272>
1c00b6ce:	5622                	lw	a2,40(sp)
1c00b6d0:	46c6                	lw	a3,80(sp)
1c00b6d2:	55a2                	lw	a1,40(sp)
1c00b6d4:	00760793          	addi	a5,a2,7
1c00b6d8:	878d                	srai	a5,a5,0x3
1c00b6da:	02f68633          	mul	a2,a3,a5
1c00b6de:	97b2                	add	a5,a5,a2
1c00b6e0:	04b7c733          	p.min	a4,a5,a1
1c00b6e4:	06e65f63          	ble	a4,a2,1c00b762 <mm_unroll_2x4+0x272>
1c00b6e8:	57f2                	lw	a5,60(sp)
1c00b6ea:	5842                	lw	a6,48(sp)
1c00b6ec:	002c1e13          	slli	t3,s8,0x2
1c00b6f0:	fff78693          	addi	a3,a5,-1
1c00b6f4:	02d587b3          	mul	a5,a1,a3
1c00b6f8:	45b6                	lw	a1,76(sp)
1c00b6fa:	038686b3          	mul	a3,a3,s8
1c00b6fe:	00c78eb3          	add	t4,a5,a2
1c00b702:	00f708b3          	add	a7,a4,a5
1c00b706:	0e8a                	slli	t4,t4,0x2
1c00b708:	088a                	slli	a7,a7,0x2
1c00b70a:	41d888b3          	sub	a7,a7,t4
1c00b70e:	18f1                	addi	a7,a7,-4
1c00b710:	0028d893          	srli	a7,a7,0x2
1c00b714:	9eae                	add	t4,t4,a1
1c00b716:	0885                	addi	a7,a7,1
1c00b718:	43c60833          	p.mac	a6,a2,t3
1c00b71c:	4616                	lw	a2,68(sp)
1c00b71e:	00dc0f33          	add	t5,s8,a3
1c00b722:	0f0a                	slli	t5,t5,0x2
1c00b724:	068a                	slli	a3,a3,0x2
1c00b726:	00d60333          	add	t1,a2,a3
1c00b72a:	9f32                	add	t5,t5,a2
1c00b72c:	01a8c07b          	lp.setup	x0,a7,1c00b760 <mm_unroll_2x4+0x270>
1c00b730:	00000713          	li	a4,0
1c00b734:	03805463          	blez	s8,1c00b75c <mm_unroll_2x4+0x26c>
1c00b738:	406f07b3          	sub	a5,t5,t1
1c00b73c:	17f1                	addi	a5,a5,-4
1c00b73e:	00000713          	li	a4,0
1c00b742:	8389                	srli	a5,a5,0x2
1c00b744:	8642                	mv	a2,a6
1c00b746:	869a                	mv	a3,t1
1c00b748:	0785                	addi	a5,a5,1
1c00b74a:	0001                	nop
1c00b74c:	0067c0fb          	lp.setup	x1,a5,1c00b758 <mm_unroll_2x4+0x268>
1c00b750:	0046a50b          	p.lw	a0,4(a3!)
1c00b754:	0046258b          	p.lw	a1,4(a2!)
1c00b758:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00b75c:	00eea22b          	p.sw	a4,4(t4!)
1c00b760:	9872                	add	a6,a6,t3
1c00b762:	443a                	lw	s0,140(sp)
1c00b764:	44aa                	lw	s1,136(sp)
1c00b766:	491a                	lw	s2,132(sp)
1c00b768:	498a                	lw	s3,128(sp)
1c00b76a:	5a76                	lw	s4,124(sp)
1c00b76c:	5ae6                	lw	s5,120(sp)
1c00b76e:	5b56                	lw	s6,116(sp)
1c00b770:	5bc6                	lw	s7,112(sp)
1c00b772:	5c36                	lw	s8,108(sp)
1c00b774:	5ca6                	lw	s9,104(sp)
1c00b776:	5d16                	lw	s10,100(sp)
1c00b778:	5d86                	lw	s11,96(sp)
1c00b77a:	6149                	addi	sp,sp,144
1c00b77c:	8082                	ret
1c00b77e:	12d5de63          	ble	a3,a1,1c00b8ba <mm_unroll_2x4+0x3ca>
1c00b782:	56a2                	lw	a3,40(sp)
1c00b784:	002c1793          	slli	a5,s8,0x2
1c00b788:	d23e                	sw	a5,36(sp)
1c00b78a:	c206b6b3          	p.bclr	a3,a3,1,0
1c00b78e:	00158793          	addi	a5,a1,1
1c00b792:	c436                	sw	a3,8(sp)
1c00b794:	00269593          	slli	a1,a3,0x2
1c00b798:	003c1693          	slli	a3,s8,0x3
1c00b79c:	c836                	sw	a3,16(sp)
1c00b79e:	46e2                	lw	a3,24(sp)
1c00b7a0:	8d62                	mv	s10,s8
1c00b7a2:	5722                	lw	a4,40(sp)
1c00b7a4:	43868d33          	p.mac	s10,a3,s8
1c00b7a8:	d62e                	sw	a1,44(sp)
1c00b7aa:	fa273633          	p.bclr	a2,a4,29,2
1c00b7ae:	d032                	sw	a2,32(sp)
1c00b7b0:	5642                	lw	a2,48(sp)
1c00b7b2:	00271a93          	slli	s5,a4,0x2
1c00b7b6:	070e                	slli	a4,a4,0x3
1c00b7b8:	95b2                	add	a1,a1,a2
1c00b7ba:	ce3a                	sw	a4,28(sp)
1c00b7bc:	c0ae                	sw	a1,64(sp)
1c00b7be:	5712                	lw	a4,36(sp)
1c00b7c0:	4596                	lw	a1,68(sp)
1c00b7c2:	4c96                	lw	s9,68(sp)
1c00b7c4:	0d0a                	slli	s10,s10,0x2
1c00b7c6:	9656                	add	a2,a2,s5
1c00b7c8:	42d70cb3          	p.mac	s9,a4,a3
1c00b7cc:	9d2e                	add	s10,s10,a1
1c00b7ce:	4736                	lw	a4,76(sp)
1c00b7d0:	45b6                	lw	a1,76(sp)
1c00b7d2:	dc32                	sw	a2,56(sp)
1c00b7d4:	4d96                	lw	s11,68(sp)
1c00b7d6:	5612                	lw	a2,36(sp)
1c00b7d8:	42da8733          	p.mac	a4,s5,a3
1c00b7dc:	42fa85b3          	p.mac	a1,s5,a5
1c00b7e0:	c63a                	sw	a4,12(sp)
1c00b7e2:	42f60db3          	p.mac	s11,a2,a5
1c00b7e6:	ca2e                	sw	a1,20(sp)
1c00b7e8:	4722                	lw	a4,8(sp)
1c00b7ea:	c74d                	beqz	a4,1c00b894 <mm_unroll_2x4+0x3a4>
1c00b7ec:	fff70b13          	addi	s6,a4,-1
1c00b7f0:	002b5b13          	srli	s6,s6,0x2
1c00b7f4:	00370793          	addi	a5,a4,3
1c00b7f8:	4711                	li	a4,4
1c00b7fa:	5bc2                	lw	s7,48(sp)
1c00b7fc:	49d2                	lw	s3,20(sp)
1c00b7fe:	4a32                	lw	s4,12(sp)
1c00b800:	0b05                	addi	s6,s6,1
1c00b802:	2ae7e263          	bltu	a5,a4,1c00baa6 <mm_unroll_2x4+0x5b6>
1c00b806:	0001                	nop
1c00b808:	045b407b          	lp.setup	x0,s6,1c00b892 <mm_unroll_2x4+0x3a2>
1c00b80c:	15805463          	blez	s8,1c00b954 <mm_unroll_2x4+0x464>
1c00b810:	00000393          	li	t2,0
1c00b814:	419d0633          	sub	a2,s10,s9
1c00b818:	1671                	addi	a2,a2,-4
1c00b81a:	829e                	mv	t0,t2
1c00b81c:	8f9e                	mv	t6,t2
1c00b81e:	8f1e                	mv	t5,t2
1c00b820:	8e9e                	mv	t4,t2
1c00b822:	8e1e                	mv	t3,t2
1c00b824:	831e                	mv	t1,t2
1c00b826:	889e                	mv	a7,t2
1c00b828:	8209                	srli	a2,a2,0x2
1c00b82a:	84ee                	mv	s1,s11
1c00b82c:	87de                	mv	a5,s7
1c00b82e:	8466                	mv	s0,s9
1c00b830:	0605                	addi	a2,a2,1
1c00b832:	0001                	nop
1c00b834:	01c640fb          	lp.setup	x1,a2,1c00b86c <mm_unroll_2x4+0x37c>
1c00b838:	0044268b          	p.lw	a3,4(s0!)
1c00b83c:	0044a70b          	p.lw	a4,4(s1!)
1c00b840:	0007a903          	lw	s2,0(a5)
1c00b844:	0047a803          	lw	a6,4(a5)
1c00b848:	4788                	lw	a0,8(a5)
1c00b84a:	47cc                	lw	a1,12(a5)
1c00b84c:	8926f8c3          	fmadd.s	a7,a3,s2,a7
1c00b850:	3106f343          	fmadd.s	t1,a3,a6,t1
1c00b854:	e0a6fe43          	fmadd.s	t3,a3,a0,t3
1c00b858:	e8b6fec3          	fmadd.s	t4,a3,a1,t4
1c00b85c:	f0e97f43          	fmadd.s	t5,s2,a4,t5
1c00b860:	f8e87fc3          	fmadd.s	t6,a6,a4,t6
1c00b864:	28e572c3          	fmadd.s	t0,a0,a4,t0
1c00b868:	38e5f3c3          	fmadd.s	t2,a1,a4,t2
1c00b86c:	97d6                	add	a5,a5,s5
1c00b86e:	011a2023          	sw	a7,0(s4)
1c00b872:	006a2223          	sw	t1,4(s4)
1c00b876:	01ca2423          	sw	t3,8(s4)
1c00b87a:	01da2623          	sw	t4,12(s4)
1c00b87e:	01e9a023          	sw	t5,0(s3)
1c00b882:	01f9a223          	sw	t6,4(s3)
1c00b886:	0059a423          	sw	t0,8(s3)
1c00b88a:	0079a623          	sw	t2,12(s3)
1c00b88e:	0a41                	addi	s4,s4,16
1c00b890:	09c1                	addi	s3,s3,16
1c00b892:	0bc1                	addi	s7,s7,16
1c00b894:	5702                	lw	a4,32(sp)
1c00b896:	e37d                	bnez	a4,1c00b97c <mm_unroll_2x4+0x48c>
1c00b898:	46c2                	lw	a3,16(sp)
1c00b89a:	4662                	lw	a2,24(sp)
1c00b89c:	4732                	lw	a4,12(sp)
1c00b89e:	47f2                	lw	a5,28(sp)
1c00b8a0:	45d2                	lw	a1,20(sp)
1c00b8a2:	9cb6                	add	s9,s9,a3
1c00b8a4:	9db6                	add	s11,s11,a3
1c00b8a6:	9d36                	add	s10,s10,a3
1c00b8a8:	56d2                	lw	a3,52(sp)
1c00b8aa:	0609                	addi	a2,a2,2
1c00b8ac:	973e                	add	a4,a4,a5
1c00b8ae:	95be                	add	a1,a1,a5
1c00b8b0:	cc32                	sw	a2,24(sp)
1c00b8b2:	c63a                	sw	a4,12(sp)
1c00b8b4:	ca2e                	sw	a1,20(sp)
1c00b8b6:	f2d649e3          	blt	a2,a3,1c00b7e8 <mm_unroll_2x4+0x2f8>
1c00b8ba:	47d6                	lw	a5,84(sp)
1c00b8bc:	eaf053e3          	blez	a5,1c00b762 <mm_unroll_2x4+0x272>
1c00b8c0:	55a2                	lw	a1,40(sp)
1c00b8c2:	4646                	lw	a2,80(sp)
1c00b8c4:	00758793          	addi	a5,a1,7
1c00b8c8:	878d                	srai	a5,a5,0x3
1c00b8ca:	02f60733          	mul	a4,a2,a5
1c00b8ce:	97ba                	add	a5,a5,a4
1c00b8d0:	04b7c7b3          	p.min	a5,a5,a1
1c00b8d4:	e8f757e3          	ble	a5,a4,1c00b762 <mm_unroll_2x4+0x272>
1c00b8d8:	56f2                	lw	a3,60(sp)
1c00b8da:	00271313          	slli	t1,a4,0x2
1c00b8de:	00279813          	slli	a6,a5,0x2
1c00b8e2:	fff68613          	addi	a2,a3,-1
1c00b8e6:	038606b3          	mul	a3,a2,s8
1c00b8ea:	40680833          	sub	a6,a6,t1
1c00b8ee:	57c2                	lw	a5,48(sp)
1c00b8f0:	00259f13          	slli	t5,a1,0x2
1c00b8f4:	1871                	addi	a6,a6,-4
1c00b8f6:	00285813          	srli	a6,a6,0x2
1c00b8fa:	933e                	add	t1,t1,a5
1c00b8fc:	0805                	addi	a6,a6,1
1c00b8fe:	42b60733          	p.mac	a4,a2,a1
1c00b902:	4596                	lw	a1,68(sp)
1c00b904:	4636                	lw	a2,76(sp)
1c00b906:	01868eb3          	add	t4,a3,s8
1c00b90a:	0e8a                	slli	t4,t4,0x2
1c00b90c:	068a                	slli	a3,a3,0x2
1c00b90e:	00d58e33          	add	t3,a1,a3
1c00b912:	9eae                	add	t4,t4,a1
1c00b914:	00271893          	slli	a7,a4,0x2
1c00b918:	98b2                	add	a7,a7,a2
1c00b91a:	0001                	nop
1c00b91c:	01a8407b          	lp.setup	x0,a6,1c00b950 <mm_unroll_2x4+0x460>
1c00b920:	00000713          	li	a4,0
1c00b924:	03805463          	blez	s8,1c00b94c <mm_unroll_2x4+0x45c>
1c00b928:	41ce87b3          	sub	a5,t4,t3
1c00b92c:	17f1                	addi	a5,a5,-4
1c00b92e:	00000713          	li	a4,0
1c00b932:	8389                	srli	a5,a5,0x2
1c00b934:	861a                	mv	a2,t1
1c00b936:	86f2                	mv	a3,t3
1c00b938:	0785                	addi	a5,a5,1
1c00b93a:	0001                	nop
1c00b93c:	0067c0fb          	lp.setup	x1,a5,1c00b948 <mm_unroll_2x4+0x458>
1c00b940:	0046a50b          	p.lw	a0,4(a3!)
1c00b944:	21e6758b          	p.lw	a1,t5(a2!)
1c00b948:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00b94c:	00e8a22b          	p.sw	a4,4(a7!)
1c00b950:	0311                	addi	t1,t1,4
1c00b952:	bd01                	j	1c00b762 <mm_unroll_2x4+0x272>
1c00b954:	00000393          	li	t2,0
1c00b958:	829e                	mv	t0,t2
1c00b95a:	8f9e                	mv	t6,t2
1c00b95c:	8f1e                	mv	t5,t2
1c00b95e:	8e9e                	mv	t4,t2
1c00b960:	8e1e                	mv	t3,t2
1c00b962:	831e                	mv	t1,t2
1c00b964:	889e                	mv	a7,t2
1c00b966:	b721                	j	1c00b86e <mm_unroll_2x4+0x37e>
1c00b968:	00000393          	li	t2,0
1c00b96c:	829e                	mv	t0,t2
1c00b96e:	8f9e                	mv	t6,t2
1c00b970:	8f1e                	mv	t5,t2
1c00b972:	8e9e                	mv	t4,t2
1c00b974:	8e1e                	mv	t3,t2
1c00b976:	831e                	mv	t1,t2
1c00b978:	889e                	mv	a7,t2
1c00b97a:	b9cd                	j	1c00b66c <mm_unroll_2x4+0x17c>
1c00b97c:	57b2                	lw	a5,44(sp)
1c00b97e:	45b2                	lw	a1,12(sp)
1c00b980:	4ea2                	lw	t4,8(sp)
1c00b982:	52a2                	lw	t0,40(sp)
1c00b984:	5392                	lw	t2,36(sp)
1c00b986:	5462                	lw	s0,56(sp)
1c00b988:	4306                	lw	t1,64(sp)
1c00b98a:	00b78f33          	add	t5,a5,a1
1c00b98e:	8e66                	mv	t3,s9
1c00b990:	4f89                	li	t6,2
1c00b992:	01c38933          	add	s2,t2,t3
1c00b996:	045ed863          	ble	t0,t4,1c00b9e6 <mm_unroll_2x4+0x4f6>
1c00b99a:	40640833          	sub	a6,s0,t1
1c00b99e:	1871                	addi	a6,a6,-4
1c00b9a0:	00285813          	srli	a6,a6,0x2
1c00b9a4:	01c38933          	add	s2,t2,t3
1c00b9a8:	889a                	mv	a7,t1
1c00b9aa:	84fa                	mv	s1,t5
1c00b9ac:	0805                	addi	a6,a6,1
1c00b9ae:	0001                	nop
1c00b9b0:	01a8407b          	lp.setup	x0,a6,1c00b9e4 <mm_unroll_2x4+0x4f4>
1c00b9b4:	00000713          	li	a4,0
1c00b9b8:	03805463          	blez	s8,1c00b9e0 <mm_unroll_2x4+0x4f0>
1c00b9bc:	41c907b3          	sub	a5,s2,t3
1c00b9c0:	17f1                	addi	a5,a5,-4
1c00b9c2:	00000713          	li	a4,0
1c00b9c6:	8389                	srli	a5,a5,0x2
1c00b9c8:	8646                	mv	a2,a7
1c00b9ca:	86f2                	mv	a3,t3
1c00b9cc:	0785                	addi	a5,a5,1
1c00b9ce:	0001                	nop
1c00b9d0:	0067c0fb          	lp.setup	x1,a5,1c00b9dc <mm_unroll_2x4+0x4ec>
1c00b9d4:	0046a50b          	p.lw	a0,4(a3!)
1c00b9d8:	2156758b          	p.lw	a1,s5(a2!)
1c00b9dc:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00b9e0:	00e4a22b          	p.sw	a4,4(s1!)
1c00b9e4:	0891                	addi	a7,a7,4
1c00b9e6:	9f56                	add	t5,t5,s5
1c00b9e8:	8e4a                	mv	t3,s2
1c00b9ea:	ea1fa7e3          	p.beqimm	t6,1,1c00b898 <mm_unroll_2x4+0x3a8>
1c00b9ee:	4f85                	li	t6,1
1c00b9f0:	b74d                	j	1c00b992 <mm_unroll_2x4+0x4a2>
1c00b9f2:	4686                	lw	a3,64(sp)
1c00b9f4:	5702                	lw	a4,32(sp)
1c00b9f6:	5422                	lw	s0,40(sp)
1c00b9f8:	42c2                	lw	t0,16(sp)
1c00b9fa:	44f2                	lw	s1,28(sp)
1c00b9fc:	4966                	lw	s2,88(sp)
1c00b9fe:	49f6                	lw	s3,92(sp)
1c00ba00:	00e68f33          	add	t5,a3,a4
1c00ba04:	8e6a                	mv	t3,s10
1c00ba06:	4f89                	li	t6,2
1c00ba08:	83b6                	mv	t2,a3
1c00ba0a:	01cd8eb3          	add	t4,s11,t3
1c00ba0e:	0484d863          	ble	s0,s1,1c00ba5e <mm_unroll_2x4+0x56e>
1c00ba12:	01e90333          	add	t1,s2,t5
1c00ba16:	406f0833          	sub	a6,t5,t1
1c00ba1a:	1871                	addi	a6,a6,-4
1c00ba1c:	00285813          	srli	a6,a6,0x2
1c00ba20:	01cd8eb3          	add	t4,s11,t3
1c00ba24:	88ce                	mv	a7,s3
1c00ba26:	0805                	addi	a6,a6,1
1c00ba28:	01a8407b          	lp.setup	x0,a6,1c00ba5c <mm_unroll_2x4+0x56c>
1c00ba2c:	00000713          	li	a4,0
1c00ba30:	02505463          	blez	t0,1c00ba58 <mm_unroll_2x4+0x568>
1c00ba34:	41ce87b3          	sub	a5,t4,t3
1c00ba38:	17f1                	addi	a5,a5,-4
1c00ba3a:	00000713          	li	a4,0
1c00ba3e:	8389                	srli	a5,a5,0x2
1c00ba40:	8646                	mv	a2,a7
1c00ba42:	86f2                	mv	a3,t3
1c00ba44:	0785                	addi	a5,a5,1
1c00ba46:	0001                	nop
1c00ba48:	0067c0fb          	lp.setup	x1,a5,1c00ba54 <mm_unroll_2x4+0x564>
1c00ba4c:	0046a50b          	p.lw	a0,4(a3!)
1c00ba50:	0046258b          	p.lw	a1,4(a2!)
1c00ba54:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00ba58:	00e3222b          	p.sw	a4,4(t1!)
1c00ba5c:	98ee                	add	a7,a7,s11
1c00ba5e:	9f1e                	add	t5,t5,t2
1c00ba60:	8e76                	mv	t3,t4
1c00ba62:	c21face3          	p.beqimm	t6,1,1c00b69a <mm_unroll_2x4+0x1aa>
1c00ba66:	4f85                	li	t6,1
1c00ba68:	b74d                	j	1c00ba0a <mm_unroll_2x4+0x51a>
1c00ba6a:	443a                	lw	s0,140(sp)
1c00ba6c:	44aa                	lw	s1,136(sp)
1c00ba6e:	491a                	lw	s2,132(sp)
1c00ba70:	498a                	lw	s3,128(sp)
1c00ba72:	5a76                	lw	s4,124(sp)
1c00ba74:	5ae6                	lw	s5,120(sp)
1c00ba76:	5b56                	lw	s6,116(sp)
1c00ba78:	5bc6                	lw	s7,112(sp)
1c00ba7a:	5c36                	lw	s8,108(sp)
1c00ba7c:	5ca6                	lw	s9,104(sp)
1c00ba7e:	5d16                	lw	s10,100(sp)
1c00ba80:	5d86                	lw	s11,96(sp)
1c00ba82:	6149                	addi	sp,sp,144
1c00ba84:	a2cfe06f          	j	1c009cb0 <mm_unroll_1x8>
1c00ba88:	443a                	lw	s0,140(sp)
1c00ba8a:	44aa                	lw	s1,136(sp)
1c00ba8c:	491a                	lw	s2,132(sp)
1c00ba8e:	498a                	lw	s3,128(sp)
1c00ba90:	5a76                	lw	s4,124(sp)
1c00ba92:	5ae6                	lw	s5,120(sp)
1c00ba94:	5b56                	lw	s6,116(sp)
1c00ba96:	5bc6                	lw	s7,112(sp)
1c00ba98:	5c36                	lw	s8,108(sp)
1c00ba9a:	5ca6                	lw	s9,104(sp)
1c00ba9c:	5d16                	lw	s10,100(sp)
1c00ba9e:	5d86                	lw	s11,96(sp)
1c00baa0:	6149                	addi	sp,sp,144
1c00baa2:	d4aff06f          	j	1c00afec <mm_unroll_2x2>
1c00baa6:	4b05                	li	s6,1
1c00baa8:	bbb9                	j	1c00b806 <mm_unroll_2x4+0x316>
1c00baaa:	4c85                	li	s9,1
1c00baac:	b6a1                	j	1c00b5f4 <mm_unroll_2x4+0x104>
	...

1c00bab0 <mm_unroll_4x2>:
1c00bab0:	454c                	lw	a1,12(a0)
1c00bab2:	7131                	addi	sp,sp,-192
1c00bab4:	df22                	sw	s0,188(sp)
1c00bab6:	c205b633          	p.bclr	a2,a1,1,0
1c00baba:	dd26                	sw	s1,184(sp)
1c00babc:	db4a                	sw	s2,180(sp)
1c00babe:	d94e                	sw	s3,176(sp)
1c00bac0:	d752                	sw	s4,172(sp)
1c00bac2:	d556                	sw	s5,168(sp)
1c00bac4:	d35a                	sw	s6,164(sp)
1c00bac6:	d15e                	sw	s7,160(sp)
1c00bac8:	cf62                	sw	s8,156(sp)
1c00baca:	cd66                	sw	s9,152(sp)
1c00bacc:	cb6a                	sw	s10,148(sp)
1c00bace:	c96e                	sw	s11,144(sp)
1c00bad0:	deae                	sw	a1,124(sp)
1c00bad2:	477d                	li	a4,31
1c00bad4:	d4b2                	sw	a2,104(sp)
1c00bad6:	f14026f3          	csrr	a3,mhartid
1c00bada:	68c75563          	ble	a2,a4,1c00c164 <mm_unroll_4x2+0x6b4>
1c00bade:	4918                	lw	a4,16(a0)
1c00bae0:	87aa                	mv	a5,a0
1c00bae2:	ceba                	sw	a4,92(sp)
1c00bae4:	4876                	lw	a6,92(sp)
1c00bae6:	4705                	li	a4,1
1c00bae8:	69075d63          	ble	a6,a4,1c00c182 <mm_unroll_4x2+0x6d2>
1c00baec:	00760713          	addi	a4,a2,7
1c00baf0:	f456b6b3          	p.bclr	a3,a3,26,5
1c00baf4:	870d                	srai	a4,a4,0x3
1c00baf6:	02e68633          	mul	a2,a3,a4
1c00bafa:	00052803          	lw	a6,0(a0)
1c00bafe:	478c                	lw	a1,8(a5)
1c00bb00:	4bdc                	lw	a5,20(a5)
1c00bb02:	dcc2                	sw	a6,120(sp)
1c00bb04:	c336                	sw	a3,132(sp)
1c00bb06:	d83e                	sw	a5,48(sp)
1c00bb08:	57a6                	lw	a5,104(sp)
1c00bb0a:	4d14                	lw	a3,24(a0)
1c00bb0c:	4148                	lw	a0,4(a0)
1c00bb0e:	c0b2                	sw	a2,64(sp)
1c00bb10:	4806                	lw	a6,64(sp)
1c00bb12:	5676                	lw	a2,124(sp)
1c00bb14:	d6aa                	sw	a0,108(sp)
1c00bb16:	9742                	add	a4,a4,a6
1c00bb18:	8e1d                	sub	a2,a2,a5
1c00bb1a:	04f74733          	p.min	a4,a4,a5
1c00bb1e:	daae                	sw	a1,116(sp)
1c00bb20:	c732                	sw	a2,140(sp)
1c00bb22:	d8ba                	sw	a4,112(sp)
1c00bb24:	28069f63          	bnez	a3,1c00bdc2 <mm_unroll_4x2+0x312>
1c00bb28:	1ce85d63          	ble	a4,a6,1c00bd02 <mm_unroll_4x2+0x252>
1c00bb2c:	56c2                	lw	a3,48(sp)
1c00bb2e:	4676                	lw	a2,92(sp)
1c00bb30:	5542                	lw	a0,48(sp)
1c00bb32:	068a                	slli	a3,a3,0x2
1c00bb34:	d2b6                	sw	a3,100(sp)
1c00bb36:	00280713          	addi	a4,a6,2
1c00bb3a:	00180693          	addi	a3,a6,1
1c00bb3e:	00380793          	addi	a5,a6,3
1c00bb42:	0512                	slli	a0,a0,0x4
1c00bb44:	fc163833          	p.bclr	a6,a2,30,1
1c00bb48:	d0c2                	sw	a6,96(sp)
1c00bb4a:	dc2a                	sw	a0,56(sp)
1c00bb4c:	4876                	lw	a6,92(sp)
1c00bb4e:	4506                	lw	a0,64(sp)
1c00bb50:	00261b13          	slli	s6,a2,0x2
1c00bb54:	5642                	lw	a2,48(sp)
1c00bb56:	02a80833          	mul	a6,a6,a0
1c00bb5a:	45f6                	lw	a1,92(sp)
1c00bb5c:	c005b5b3          	p.bclr	a1,a1,0,0
1c00bb60:	caae                	sw	a1,84(sp)
1c00bb62:	058a                	slli	a1,a1,0x2
1c00bb64:	42a60633          	p.mac	a2,a2,a0
1c00bb68:	4576                	lw	a0,92(sp)
1c00bb6a:	ccc2                	sw	a6,88(sp)
1c00bb6c:	5866                	lw	a6,120(sp)
1c00bb6e:	0512                	slli	a0,a0,0x4
1c00bb70:	de2a                	sw	a0,60(sp)
1c00bb72:	5536                	lw	a0,108(sp)
1c00bb74:	95aa                	add	a1,a1,a0
1c00bb76:	060a                	slli	a2,a2,0x2
1c00bb78:	9642                	add	a2,a2,a6
1c00bb7a:	d232                	sw	a2,36(sp)
1c00bb7c:	c52e                	sw	a1,136(sp)
1c00bb7e:	5616                	lw	a2,100(sp)
1c00bb80:	00ab05b3          	add	a1,s6,a0
1c00bb84:	4506                	lw	a0,64(sp)
1c00bb86:	c12e                	sw	a1,128(sp)
1c00bb88:	55d6                	lw	a1,116(sp)
1c00bb8a:	42a60833          	p.mac	a6,a2,a0
1c00bb8e:	5656                	lw	a2,116(sp)
1c00bb90:	436505b3          	p.mac	a1,a0,s6
1c00bb94:	d042                	sw	a6,32(sp)
1c00bb96:	5556                	lw	a0,116(sp)
1c00bb98:	5856                	lw	a6,116(sp)
1c00bb9a:	43668633          	p.mac	a2,a3,s6
1c00bb9e:	c8ae                	sw	a1,80(sp)
1c00bba0:	55e6                	lw	a1,120(sp)
1c00bba2:	c6b2                	sw	a2,76(sp)
1c00bba4:	5616                	lw	a2,100(sp)
1c00bba6:	43670833          	p.mac	a6,a4,s6
1c00bbaa:	43678533          	p.mac	a0,a5,s6
1c00bbae:	c4c2                	sw	a6,72(sp)
1c00bbb0:	42d605b3          	p.mac	a1,a2,a3
1c00bbb4:	c2aa                	sw	a0,68(sp)
1c00bbb6:	da2e                	sw	a1,52(sp)
1c00bbb8:	56e6                	lw	a3,120(sp)
1c00bbba:	42e606b3          	p.mac	a3,a2,a4
1c00bbbe:	5766                	lw	a4,120(sp)
1c00bbc0:	42f60733          	p.mac	a4,a2,a5
1c00bbc4:	d636                	sw	a3,44(sp)
1c00bbc6:	d43a                	sw	a4,40(sp)
1c00bbc8:	47d6                	lw	a5,84(sp)
1c00bbca:	c3f5                	beqz	a5,1c00bcae <mm_unroll_4x2+0x1fe>
1c00bbcc:	4846                	lw	a6,80(sp)
1c00bbce:	4536                	lw	a0,76(sp)
1c00bbd0:	45a6                	lw	a1,72(sp)
1c00bbd2:	0811                	addi	a6,a6,4
1c00bbd4:	fff78693          	addi	a3,a5,-1
1c00bbd8:	ce42                	sw	a6,28(sp)
1c00bbda:	5836                	lw	a6,108(sp)
1c00bbdc:	4616                	lw	a2,68(sp)
1c00bbde:	8285                	srli	a3,a3,0x1
1c00bbe0:	0511                	addi	a0,a0,4
1c00bbe2:	0591                	addi	a1,a1,4
1c00bbe4:	0685                	addi	a3,a3,1
1c00bbe6:	0785                	addi	a5,a5,1
1c00bbe8:	4709                	li	a4,2
1c00bbea:	cc2a                	sw	a0,24(sp)
1c00bbec:	ca2e                	sw	a1,20(sp)
1c00bbee:	c642                	sw	a6,12(sp)
1c00bbf0:	c836                	sw	a3,16(sp)
1c00bbf2:	00460b93          	addi	s7,a2,4
1c00bbf6:	8c32                	mv	s8,a2
1c00bbf8:	4ca6                	lw	s9,72(sp)
1c00bbfa:	4d36                	lw	s10,76(sp)
1c00bbfc:	4dc6                	lw	s11,80(sp)
1c00bbfe:	5ae7e163          	bltu	a5,a4,1c00c1a0 <mm_unroll_4x2+0x6f0>
1c00bc02:	5842                	lw	a6,48(sp)
1c00bc04:	45005963          	blez	a6,1c00c056 <mm_unroll_4x2+0x5a6>
1c00bc08:	5802                	lw	a6,32(sp)
1c00bc0a:	5792                	lw	a5,36(sp)
1c00bc0c:	00000393          	li	t2,0
1c00bc10:	4532                	lw	a0,12(sp)
1c00bc12:	410786b3          	sub	a3,a5,a6
1c00bc16:	16f1                	addi	a3,a3,-4
1c00bc18:	829e                	mv	t0,t2
1c00bc1a:	8f9e                	mv	t6,t2
1c00bc1c:	8f1e                	mv	t5,t2
1c00bc1e:	8e9e                	mv	t4,t2
1c00bc20:	8e1e                	mv	t3,t2
1c00bc22:	831e                	mv	t1,t2
1c00bc24:	889e                	mv	a7,t2
1c00bc26:	5a22                	lw	s4,40(sp)
1c00bc28:	59b2                	lw	s3,44(sp)
1c00bc2a:	5952                	lw	s2,52(sp)
1c00bc2c:	8289                	srli	a3,a3,0x2
1c00bc2e:	00450a93          	addi	s5,a0,4
1c00bc32:	84aa                	mv	s1,a0
1c00bc34:	8442                	mv	s0,a6
1c00bc36:	0685                	addi	a3,a3,1
1c00bc38:	01c6c0fb          	lp.setup	x1,a3,1c00bc70 <mm_unroll_4x2+0x1c0>
1c00bc3c:	2164f70b          	p.lw	a4,s6(s1!)
1c00bc40:	216af78b          	p.lw	a5,s6(s5!)
1c00bc44:	0044280b          	p.lw	a6,4(s0!)
1c00bc48:	0049250b          	p.lw	a0,4(s2!)
1c00bc4c:	0049a58b          	p.lw	a1,4(s3!)
1c00bc50:	004a260b          	p.lw	a2,4(s4!)
1c00bc54:	88e878c3          	fmadd.s	a7,a6,a4,a7
1c00bc58:	30f87343          	fmadd.s	t1,a6,a5,t1
1c00bc5c:	e0a77e43          	fmadd.s	t3,a4,a0,t3
1c00bc60:	e8a7fec3          	fmadd.s	t4,a5,a0,t4
1c00bc64:	f0b77f43          	fmadd.s	t5,a4,a1,t5
1c00bc68:	f8b7ffc3          	fmadd.s	t6,a5,a1,t6
1c00bc6c:	28c772c3          	fmadd.s	t0,a4,a2,t0
1c00bc70:	38c7f3c3          	fmadd.s	t2,a5,a2,t2
1c00bc74:	45f2                	lw	a1,28(sp)
1c00bc76:	011da42b          	p.sw	a7,8(s11!)
1c00bc7a:	4662                	lw	a2,24(sp)
1c00bc7c:	0065a42b          	p.sw	t1,8(a1!)
1c00bc80:	01cd242b          	p.sw	t3,8(s10!)
1c00bc84:	46d2                	lw	a3,20(sp)
1c00bc86:	01d6242b          	p.sw	t4,8(a2!)
1c00bc8a:	4732                	lw	a4,12(sp)
1c00bc8c:	47c2                	lw	a5,16(sp)
1c00bc8e:	01eca42b          	p.sw	t5,8(s9!)
1c00bc92:	01f6a42b          	p.sw	t6,8(a3!)
1c00bc96:	005c242b          	p.sw	t0,8(s8!)
1c00bc9a:	0721                	addi	a4,a4,8
1c00bc9c:	17fd                	addi	a5,a5,-1
1c00bc9e:	ce2e                	sw	a1,28(sp)
1c00bca0:	cc32                	sw	a2,24(sp)
1c00bca2:	ca36                	sw	a3,20(sp)
1c00bca4:	007ba42b          	p.sw	t2,8(s7!)
1c00bca8:	c63a                	sw	a4,12(sp)
1c00bcaa:	c83e                	sw	a5,16(sp)
1c00bcac:	fbb9                	bnez	a5,1c00bc02 <mm_unroll_4x2+0x152>
1c00bcae:	5506                	lw	a0,96(sp)
1c00bcb0:	3a051d63          	bnez	a0,1c00c06a <mm_unroll_4x2+0x5ba>
1c00bcb4:	4686                	lw	a3,64(sp)
1c00bcb6:	4766                	lw	a4,88(sp)
1c00bcb8:	5862                	lw	a6,56(sp)
1c00bcba:	55f2                	lw	a1,60(sp)
1c00bcbc:	0691                	addi	a3,a3,4
1c00bcbe:	975a                	add	a4,a4,s6
1c00bcc0:	5782                	lw	a5,32(sp)
1c00bcc2:	4546                	lw	a0,80(sp)
1c00bcc4:	4636                	lw	a2,76(sp)
1c00bcc6:	c0b6                	sw	a3,64(sp)
1c00bcc8:	ccba                	sw	a4,88(sp)
1c00bcca:	46a6                	lw	a3,72(sp)
1c00bccc:	4716                	lw	a4,68(sp)
1c00bcce:	952e                	add	a0,a0,a1
1c00bcd0:	962e                	add	a2,a2,a1
1c00bcd2:	97c2                	add	a5,a5,a6
1c00bcd4:	96ae                	add	a3,a3,a1
1c00bcd6:	972e                	add	a4,a4,a1
1c00bcd8:	d03e                	sw	a5,32(sp)
1c00bcda:	55a2                	lw	a1,40(sp)
1c00bcdc:	57d2                	lw	a5,52(sp)
1c00bcde:	c8aa                	sw	a0,80(sp)
1c00bce0:	c6b2                	sw	a2,76(sp)
1c00bce2:	5532                	lw	a0,44(sp)
1c00bce4:	5612                	lw	a2,36(sp)
1c00bce6:	c4b6                	sw	a3,72(sp)
1c00bce8:	c2ba                	sw	a4,68(sp)
1c00bcea:	56c6                	lw	a3,112(sp)
1c00bcec:	4706                	lw	a4,64(sp)
1c00bcee:	97c2                	add	a5,a5,a6
1c00bcf0:	9542                	add	a0,a0,a6
1c00bcf2:	95c2                	add	a1,a1,a6
1c00bcf4:	9642                	add	a2,a2,a6
1c00bcf6:	da3e                	sw	a5,52(sp)
1c00bcf8:	d62a                	sw	a0,44(sp)
1c00bcfa:	d42e                	sw	a1,40(sp)
1c00bcfc:	d232                	sw	a2,36(sp)
1c00bcfe:	ecd745e3          	blt	a4,a3,1c00bbc8 <mm_unroll_4x2+0x118>
1c00bd02:	45ba                	lw	a1,140(sp)
1c00bd04:	0ab05163          	blez	a1,1c00bda6 <mm_unroll_4x2+0x2f6>
1c00bd08:	4676                	lw	a2,92(sp)
1c00bd0a:	469a                	lw	a3,132(sp)
1c00bd0c:	00760793          	addi	a5,a2,7
1c00bd10:	878d                	srai	a5,a5,0x3
1c00bd12:	02f68733          	mul	a4,a3,a5
1c00bd16:	97ba                	add	a5,a5,a4
1c00bd18:	04c7c7b3          	p.min	a5,a5,a2
1c00bd1c:	08f75563          	ble	a5,a4,1c00bda6 <mm_unroll_4x2+0x2f6>
1c00bd20:	5876                	lw	a6,124(sp)
1c00bd22:	55a6                	lw	a1,104(sp)
1c00bd24:	0905d163          	ble	a6,a1,1c00bda6 <mm_unroll_4x2+0x2f6>
1c00bd28:	8f3a                	mv	t5,a4
1c00bd2a:	42b60f33          	p.mac	t5,a2,a1
1c00bd2e:	5642                	lw	a2,48(sp)
1c00bd30:	52e6                	lw	t0,120(sp)
1c00bd32:	56b6                	lw	a3,108(sp)
1c00bd34:	00261e93          	slli	t4,a2,0x2
1c00bd38:	070a                	slli	a4,a4,0x2
1c00bd3a:	078a                	slli	a5,a5,0x2
1c00bd3c:	00e68e33          	add	t3,a3,a4
1c00bd40:	00f68fb3          	add	t6,a3,a5
1c00bd44:	4776                	lw	a4,92(sp)
1c00bd46:	43d582b3          	p.mac	t0,a1,t4
1c00bd4a:	57d6                	lw	a5,116(sp)
1c00bd4c:	0f0a                	slli	t5,t5,0x2
1c00bd4e:	00271393          	slli	t2,a4,0x2
1c00bd52:	9f3e                	add	t5,t5,a5
1c00bd54:	84c2                	mv	s1,a6
1c00bd56:	8432                	mv	s0,a2
1c00bd58:	892e                	mv	s2,a1
1c00bd5a:	8716                	mv	a4,t0
1c00bd5c:	837a                	mv	t1,t5
1c00bd5e:	41248833          	sub	a6,s1,s2
1c00bd62:	0001                	nop
1c00bd64:	01c8407b          	lp.setup	x0,a6,1c00bd9c <mm_unroll_4x2+0x2ec>
1c00bd68:	00000693          	li	a3,0
1c00bd6c:	00ee88b3          	add	a7,t4,a4
1c00bd70:	02805463          	blez	s0,1c00bd98 <mm_unroll_4x2+0x2e8>
1c00bd74:	00ee88b3          	add	a7,t4,a4
1c00bd78:	40e887b3          	sub	a5,a7,a4
1c00bd7c:	17f1                	addi	a5,a5,-4
1c00bd7e:	00000693          	li	a3,0
1c00bd82:	8389                	srli	a5,a5,0x2
1c00bd84:	8672                	mv	a2,t3
1c00bd86:	0785                	addi	a5,a5,1
1c00bd88:	0067c0fb          	lp.setup	x1,a5,1c00bd94 <mm_unroll_4x2+0x2e4>
1c00bd8c:	0047250b          	p.lw	a0,4(a4!)
1c00bd90:	2076758b          	p.lw	a1,t2(a2!)
1c00bd94:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00bd98:	00d363ab          	p.sw	a3,t2(t1!)
1c00bd9c:	8746                	mv	a4,a7
1c00bd9e:	0e11                	addi	t3,t3,4
1c00bda0:	0f11                	addi	t5,t5,4
1c00bda2:	fbcf9ce3          	bne	t6,t3,1c00bd5a <mm_unroll_4x2+0x2aa>
1c00bda6:	547a                	lw	s0,188(sp)
1c00bda8:	54ea                	lw	s1,184(sp)
1c00bdaa:	595a                	lw	s2,180(sp)
1c00bdac:	59ca                	lw	s3,176(sp)
1c00bdae:	5a3a                	lw	s4,172(sp)
1c00bdb0:	5aaa                	lw	s5,168(sp)
1c00bdb2:	5b1a                	lw	s6,164(sp)
1c00bdb4:	5b8a                	lw	s7,160(sp)
1c00bdb6:	4c7a                	lw	s8,156(sp)
1c00bdb8:	4cea                	lw	s9,152(sp)
1c00bdba:	4d5a                	lw	s10,148(sp)
1c00bdbc:	4dca                	lw	s11,144(sp)
1c00bdbe:	6129                	addi	sp,sp,192
1c00bdc0:	8082                	ret
1c00bdc2:	1ee85263          	ble	a4,a6,1c00bfa6 <mm_unroll_4x2+0x4f6>
1c00bdc6:	5542                	lw	a0,48(sp)
1c00bdc8:	45f6                	lw	a1,92(sp)
1c00bdca:	4676                	lw	a2,92(sp)
1c00bdcc:	050a                	slli	a0,a0,0x2
1c00bdce:	da2a                	sw	a0,52(sp)
1c00bdd0:	5542                	lw	a0,48(sp)
1c00bdd2:	058a                	slli	a1,a1,0x2
1c00bdd4:	d2ae                	sw	a1,100(sp)
1c00bdd6:	45f6                	lw	a1,92(sp)
1c00bdd8:	c0063633          	p.bclr	a2,a2,0,0
1c00bddc:	0512                	slli	a0,a0,0x4
1c00bdde:	ccb2                	sw	a2,88(sp)
1c00bde0:	de2a                	sw	a0,60(sp)
1c00bde2:	5642                	lw	a2,48(sp)
1c00bde4:	4506                	lw	a0,64(sp)
1c00bde6:	0592                	slli	a1,a1,0x4
1c00bde8:	c2ae                	sw	a1,68(sp)
1c00bdea:	45f6                	lw	a1,92(sp)
1c00bdec:	42a60633          	p.mac	a2,a2,a0
1c00bdf0:	00180693          	addi	a3,a6,1
1c00bdf4:	00280713          	addi	a4,a6,2
1c00bdf8:	00380793          	addi	a5,a6,3
1c00bdfc:	4876                	lw	a6,92(sp)
1c00bdfe:	fc183833          	p.bclr	a6,a6,30,1
1c00be02:	c142                	sw	a6,128(sp)
1c00be04:	5866                	lw	a6,120(sp)
1c00be06:	02a585b3          	mul	a1,a1,a0
1c00be0a:	060a                	slli	a2,a2,0x2
1c00be0c:	9642                	add	a2,a2,a6
1c00be0e:	d032                	sw	a2,32(sp)
1c00be10:	4606                	lw	a2,64(sp)
1c00be12:	5542                	lw	a0,48(sp)
1c00be14:	050e                	slli	a0,a0,0x3
1c00be16:	dc2a                	sw	a0,56(sp)
1c00be18:	d0ae                	sw	a1,96(sp)
1c00be1a:	55d2                	lw	a1,52(sp)
1c00be1c:	5516                	lw	a0,100(sp)
1c00be1e:	42c58833          	p.mac	a6,a1,a2
1c00be22:	55d6                	lw	a1,116(sp)
1c00be24:	ce42                	sw	a6,28(sp)
1c00be26:	5856                	lw	a6,116(sp)
1c00be28:	42a685b3          	p.mac	a1,a3,a0
1c00be2c:	42a60833          	p.mac	a6,a2,a0
1c00be30:	5656                	lw	a2,116(sp)
1c00be32:	caae                	sw	a1,84(sp)
1c00be34:	c4c2                	sw	a6,72(sp)
1c00be36:	5856                	lw	a6,116(sp)
1c00be38:	42a70633          	p.mac	a2,a4,a0
1c00be3c:	42a78833          	p.mac	a6,a5,a0
1c00be40:	c8b2                	sw	a2,80(sp)
1c00be42:	c6c2                	sw	a6,76(sp)
1c00be44:	5566                	lw	a0,120(sp)
1c00be46:	55d2                	lw	a1,52(sp)
1c00be48:	5666                	lw	a2,120(sp)
1c00be4a:	42d58533          	p.mac	a0,a1,a3
1c00be4e:	56e6                	lw	a3,120(sp)
1c00be50:	42e58633          	p.mac	a2,a1,a4
1c00be54:	5736                	lw	a4,108(sp)
1c00be56:	d62a                	sw	a0,44(sp)
1c00be58:	42f586b3          	p.mac	a3,a1,a5
1c00be5c:	47e6                	lw	a5,88(sp)
1c00be5e:	d432                	sw	a2,40(sp)
1c00be60:	42b78733          	p.mac	a4,a5,a1
1c00be64:	d236                	sw	a3,36(sp)
1c00be66:	c53a                	sw	a4,136(sp)
1c00be68:	46e6                	lw	a3,88(sp)
1c00be6a:	c2fd                	beqz	a3,1c00bf50 <mm_unroll_4x2+0x4a0>
1c00be6c:	5836                	lw	a6,108(sp)
1c00be6e:	4526                	lw	a0,72(sp)
1c00be70:	4646                	lw	a2,80(sp)
1c00be72:	8736                	mv	a4,a3
1c00be74:	16fd                	addi	a3,a3,-1
1c00be76:	45d6                	lw	a1,84(sp)
1c00be78:	8285                	srli	a3,a3,0x1
1c00be7a:	c642                	sw	a6,12(sp)
1c00be7c:	4836                	lw	a6,76(sp)
1c00be7e:	0511                	addi	a0,a0,4
1c00be80:	0611                	addi	a2,a2,4
1c00be82:	0685                	addi	a3,a3,1
1c00be84:	00170793          	addi	a5,a4,1
1c00be88:	cc2a                	sw	a0,24(sp)
1c00be8a:	4709                	li	a4,2
1c00be8c:	ca32                	sw	a2,20(sp)
1c00be8e:	c836                	sw	a3,16(sp)
1c00be90:	00458b93          	addi	s7,a1,4
1c00be94:	00480b13          	addi	s6,a6,4
1c00be98:	8c42                	mv	s8,a6
1c00be9a:	4cc6                	lw	s9,80(sp)
1c00be9c:	8d2e                	mv	s10,a1
1c00be9e:	4da6                	lw	s11,72(sp)
1c00bea0:	30e7e363          	bltu	a5,a4,1c00c1a6 <mm_unroll_4x2+0x6f6>
1c00bea4:	55c2                	lw	a1,48(sp)
1c00bea6:	18b05e63          	blez	a1,1c00c042 <mm_unroll_4x2+0x592>
1c00beaa:	4572                	lw	a0,28(sp)
1c00beac:	5802                	lw	a6,32(sp)
1c00beae:	00000393          	li	t2,0
1c00beb2:	4632                	lw	a2,12(sp)
1c00beb4:	40a806b3          	sub	a3,a6,a0
1c00beb8:	55d2                	lw	a1,52(sp)
1c00beba:	16f1                	addi	a3,a3,-4
1c00bebc:	829e                	mv	t0,t2
1c00bebe:	8f9e                	mv	t6,t2
1c00bec0:	8f1e                	mv	t5,t2
1c00bec2:	8e9e                	mv	t4,t2
1c00bec4:	8e1e                	mv	t3,t2
1c00bec6:	831e                	mv	t1,t2
1c00bec8:	889e                	mv	a7,t2
1c00beca:	5a12                	lw	s4,36(sp)
1c00becc:	59a2                	lw	s3,40(sp)
1c00bece:	5932                	lw	s2,44(sp)
1c00bed0:	8289                	srli	a3,a3,0x2
1c00bed2:	00c58ab3          	add	s5,a1,a2
1c00bed6:	84b2                	mv	s1,a2
1c00bed8:	842a                	mv	s0,a0
1c00beda:	0685                	addi	a3,a3,1
1c00bedc:	01c6c0fb          	lp.setup	x1,a3,1c00bf14 <mm_unroll_4x2+0x464>
1c00bee0:	0044a70b          	p.lw	a4,4(s1!)
1c00bee4:	004aa78b          	p.lw	a5,4(s5!)
1c00bee8:	0044280b          	p.lw	a6,4(s0!)
1c00beec:	0049250b          	p.lw	a0,4(s2!)
1c00bef0:	0049a58b          	p.lw	a1,4(s3!)
1c00bef4:	004a260b          	p.lw	a2,4(s4!)
1c00bef8:	88e878c3          	fmadd.s	a7,a6,a4,a7
1c00befc:	30f87343          	fmadd.s	t1,a6,a5,t1
1c00bf00:	e0a77e43          	fmadd.s	t3,a4,a0,t3
1c00bf04:	e8a7fec3          	fmadd.s	t4,a5,a0,t4
1c00bf08:	f0b77f43          	fmadd.s	t5,a4,a1,t5
1c00bf0c:	f8b7ffc3          	fmadd.s	t6,a5,a1,t6
1c00bf10:	28c772c3          	fmadd.s	t0,a4,a2,t0
1c00bf14:	38c7f3c3          	fmadd.s	t2,a5,a2,t2
1c00bf18:	46e2                	lw	a3,24(sp)
1c00bf1a:	011da42b          	p.sw	a7,8(s11!)
1c00bf1e:	4752                	lw	a4,20(sp)
1c00bf20:	0066a42b          	p.sw	t1,8(a3!)
1c00bf24:	01cd242b          	p.sw	t3,8(s10!)
1c00bf28:	01dba42b          	p.sw	t4,8(s7!)
1c00bf2c:	47b2                	lw	a5,12(sp)
1c00bf2e:	4542                	lw	a0,16(sp)
1c00bf30:	5862                	lw	a6,56(sp)
1c00bf32:	01eca42b          	p.sw	t5,8(s9!)
1c00bf36:	01f7242b          	p.sw	t6,8(a4!)
1c00bf3a:	005c242b          	p.sw	t0,8(s8!)
1c00bf3e:	97c2                	add	a5,a5,a6
1c00bf40:	157d                	addi	a0,a0,-1
1c00bf42:	cc36                	sw	a3,24(sp)
1c00bf44:	ca3a                	sw	a4,20(sp)
1c00bf46:	007b242b          	p.sw	t2,8(s6!)
1c00bf4a:	c63e                	sw	a5,12(sp)
1c00bf4c:	c82a                	sw	a0,16(sp)
1c00bf4e:	f939                	bnez	a0,1c00bea4 <mm_unroll_4x2+0x3f4>
1c00bf50:	460a                	lw	a2,128(sp)
1c00bf52:	18061863          	bnez	a2,1c00c0e2 <mm_unroll_4x2+0x632>
1c00bf56:	4506                	lw	a0,64(sp)
1c00bf58:	5586                	lw	a1,96(sp)
1c00bf5a:	5616                	lw	a2,100(sp)
1c00bf5c:	0511                	addi	a0,a0,4
1c00bf5e:	5772                	lw	a4,60(sp)
1c00bf60:	4816                	lw	a6,68(sp)
1c00bf62:	95b2                	add	a1,a1,a2
1c00bf64:	c0aa                	sw	a0,64(sp)
1c00bf66:	46f2                	lw	a3,28(sp)
1c00bf68:	47a6                	lw	a5,72(sp)
1c00bf6a:	4556                	lw	a0,84(sp)
1c00bf6c:	4636                	lw	a2,76(sp)
1c00bf6e:	d0ae                	sw	a1,96(sp)
1c00bf70:	45c6                	lw	a1,80(sp)
1c00bf72:	97c2                	add	a5,a5,a6
1c00bf74:	9542                	add	a0,a0,a6
1c00bf76:	96ba                	add	a3,a3,a4
1c00bf78:	95c2                	add	a1,a1,a6
1c00bf7a:	9642                	add	a2,a2,a6
1c00bf7c:	ce36                	sw	a3,28(sp)
1c00bf7e:	5812                	lw	a6,36(sp)
1c00bf80:	56b2                	lw	a3,44(sp)
1c00bf82:	c4be                	sw	a5,72(sp)
1c00bf84:	caaa                	sw	a0,84(sp)
1c00bf86:	57a2                	lw	a5,40(sp)
1c00bf88:	5502                	lw	a0,32(sp)
1c00bf8a:	c8ae                	sw	a1,80(sp)
1c00bf8c:	c6b2                	sw	a2,76(sp)
1c00bf8e:	55c6                	lw	a1,112(sp)
1c00bf90:	4606                	lw	a2,64(sp)
1c00bf92:	96ba                	add	a3,a3,a4
1c00bf94:	97ba                	add	a5,a5,a4
1c00bf96:	983a                	add	a6,a6,a4
1c00bf98:	953a                	add	a0,a0,a4
1c00bf9a:	d636                	sw	a3,44(sp)
1c00bf9c:	d43e                	sw	a5,40(sp)
1c00bf9e:	d242                	sw	a6,36(sp)
1c00bfa0:	d02a                	sw	a0,32(sp)
1c00bfa2:	ecb643e3          	blt	a2,a1,1c00be68 <mm_unroll_4x2+0x3b8>
1c00bfa6:	45ba                	lw	a1,140(sp)
1c00bfa8:	deb05fe3          	blez	a1,1c00bda6 <mm_unroll_4x2+0x2f6>
1c00bfac:	4676                	lw	a2,92(sp)
1c00bfae:	469a                	lw	a3,132(sp)
1c00bfb0:	00760793          	addi	a5,a2,7
1c00bfb4:	878d                	srai	a5,a5,0x3
1c00bfb6:	02f682b3          	mul	t0,a3,a5
1c00bfba:	9796                	add	a5,a5,t0
1c00bfbc:	04c7c3b3          	p.min	t2,a5,a2
1c00bfc0:	de72d3e3          	ble	t2,t0,1c00bda6 <mm_unroll_4x2+0x2f6>
1c00bfc4:	5776                	lw	a4,124(sp)
1c00bfc6:	57a6                	lw	a5,104(sp)
1c00bfc8:	dce7dfe3          	ble	a4,a5,1c00bda6 <mm_unroll_4x2+0x2f6>
1c00bfcc:	5842                	lw	a6,48(sp)
1c00bfce:	8e96                	mv	t4,t0
1c00bfd0:	5e36                	lw	t3,108(sp)
1c00bfd2:	5fe6                	lw	t6,120(sp)
1c00bfd4:	42f60eb3          	p.mac	t4,a2,a5
1c00bfd8:	00281f13          	slli	t5,a6,0x2
1c00bfdc:	5556                	lw	a0,116(sp)
1c00bfde:	00261413          	slli	s0,a2,0x2
1c00bfe2:	893a                	mv	s2,a4
1c00bfe4:	84c2                	mv	s1,a6
1c00bfe6:	89be                	mv	s3,a5
1c00bfe8:	43e28e33          	p.mac	t3,t0,t5
1c00bfec:	0e8a                	slli	t4,t4,0x2
1c00bfee:	9eaa                	add	t4,t4,a0
1c00bff0:	43e78fb3          	p.mac	t6,a5,t5
1c00bff4:	877e                	mv	a4,t6
1c00bff6:	8376                	mv	t1,t4
1c00bff8:	41390833          	sub	a6,s2,s3
1c00bffc:	01c8407b          	lp.setup	x0,a6,1c00c034 <mm_unroll_4x2+0x584>
1c00c000:	00000693          	li	a3,0
1c00c004:	00ef08b3          	add	a7,t5,a4
1c00c008:	02905463          	blez	s1,1c00c030 <mm_unroll_4x2+0x580>
1c00c00c:	00ef08b3          	add	a7,t5,a4
1c00c010:	40e887b3          	sub	a5,a7,a4
1c00c014:	17f1                	addi	a5,a5,-4
1c00c016:	00000693          	li	a3,0
1c00c01a:	8389                	srli	a5,a5,0x2
1c00c01c:	8672                	mv	a2,t3
1c00c01e:	0785                	addi	a5,a5,1
1c00c020:	0067c0fb          	lp.setup	x1,a5,1c00c02c <mm_unroll_4x2+0x57c>
1c00c024:	0047250b          	p.lw	a0,4(a4!)
1c00c028:	0046258b          	p.lw	a1,4(a2!)
1c00c02c:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00c030:	00d3642b          	p.sw	a3,s0(t1!)
1c00c034:	8746                	mv	a4,a7
1c00c036:	0285                	addi	t0,t0,1
1c00c038:	0e91                	addi	t4,t4,4
1c00c03a:	9e7a                	add	t3,t3,t5
1c00c03c:	fa539ce3          	bne	t2,t0,1c00bff4 <mm_unroll_4x2+0x544>
1c00c040:	b39d                	j	1c00bda6 <mm_unroll_4x2+0x2f6>
1c00c042:	00000393          	li	t2,0
1c00c046:	829e                	mv	t0,t2
1c00c048:	8f9e                	mv	t6,t2
1c00c04a:	8f1e                	mv	t5,t2
1c00c04c:	8e9e                	mv	t4,t2
1c00c04e:	8e1e                	mv	t3,t2
1c00c050:	831e                	mv	t1,t2
1c00c052:	889e                	mv	a7,t2
1c00c054:	b5d1                	j	1c00bf18 <mm_unroll_4x2+0x468>
1c00c056:	00000393          	li	t2,0
1c00c05a:	829e                	mv	t0,t2
1c00c05c:	8f9e                	mv	t6,t2
1c00c05e:	8f1e                	mv	t5,t2
1c00c060:	8e9e                	mv	t4,t2
1c00c062:	8e1e                	mv	t3,t2
1c00c064:	831e                	mv	t1,t2
1c00c066:	889e                	mv	a7,t2
1c00c068:	b131                	j	1c00bc74 <mm_unroll_4x2+0x1c4>
1c00c06a:	45f6                	lw	a1,92(sp)
1c00c06c:	4656                	lw	a2,84(sp)
1c00c06e:	c4b653e3          	ble	a1,a2,1c00bcb4 <mm_unroll_4x2+0x204>
1c00c072:	5e02                	lw	t3,32(sp)
1c00c074:	4fe6                	lw	t6,88(sp)
1c00c076:	52d6                	lw	t0,116(sp)
1c00c078:	5342                	lw	t1,48(sp)
1c00c07a:	5496                	lw	s1,100(sp)
1c00c07c:	490a                	lw	s2,128(sp)
1c00c07e:	4eaa                	lw	t4,136(sp)
1c00c080:	4f11                	li	t5,4
1c00c082:	83ae                	mv	t2,a1
1c00c084:	8432                	mv	s0,a2
1c00c086:	41d90833          	sub	a6,s2,t4
1c00c08a:	008f88b3          	add	a7,t6,s0
1c00c08e:	1871                	addi	a6,a6,-4
1c00c090:	088a                	slli	a7,a7,0x2
1c00c092:	00285813          	srli	a6,a6,0x2
1c00c096:	9896                	add	a7,a7,t0
1c00c098:	01c48a33          	add	s4,s1,t3
1c00c09c:	89f6                	mv	s3,t4
1c00c09e:	0805                	addi	a6,a6,1
1c00c0a0:	01a8407b          	lp.setup	x0,a6,1c00c0d4 <mm_unroll_4x2+0x624>
1c00c0a4:	00000713          	li	a4,0
1c00c0a8:	02605463          	blez	t1,1c00c0d0 <mm_unroll_4x2+0x620>
1c00c0ac:	41ca07b3          	sub	a5,s4,t3
1c00c0b0:	17f1                	addi	a5,a5,-4
1c00c0b2:	00000713          	li	a4,0
1c00c0b6:	8389                	srli	a5,a5,0x2
1c00c0b8:	864e                	mv	a2,s3
1c00c0ba:	86f2                	mv	a3,t3
1c00c0bc:	0785                	addi	a5,a5,1
1c00c0be:	0001                	nop
1c00c0c0:	0067c0fb          	lp.setup	x1,a5,1c00c0cc <mm_unroll_4x2+0x61c>
1c00c0c4:	0046a50b          	p.lw	a0,4(a3!)
1c00c0c8:	2166758b          	p.lw	a1,s6(a2!)
1c00c0cc:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00c0d0:	00e8a22b          	p.sw	a4,4(a7!)
1c00c0d4:	0991                	addi	s3,s3,4
1c00c0d6:	1f7d                	addi	t5,t5,-1
1c00c0d8:	9f9e                	add	t6,t6,t2
1c00c0da:	8e52                	mv	t3,s4
1c00c0dc:	fa0f15e3          	bnez	t5,1c00c086 <mm_unroll_4x2+0x5d6>
1c00c0e0:	bed1                	j	1c00bcb4 <mm_unroll_4x2+0x204>
1c00c0e2:	46f6                	lw	a3,92(sp)
1c00c0e4:	4766                	lw	a4,88(sp)
1c00c0e6:	e6d758e3          	ble	a3,a4,1c00bf56 <mm_unroll_4x2+0x4a6>
1c00c0ea:	5816                	lw	a6,100(sp)
1c00c0ec:	47a6                	lw	a5,72(sp)
1c00c0ee:	4e72                	lw	t3,28(sp)
1c00c0f0:	5f86                	lw	t6,96(sp)
1c00c0f2:	5ed2                	lw	t4,52(sp)
1c00c0f4:	5456                	lw	s0,116(sp)
1c00c0f6:	53c2                	lw	t2,48(sp)
1c00c0f8:	4a2a                	lw	s4,136(sp)
1c00c0fa:	010782b3          	add	t0,a5,a6
1c00c0fe:	4f11                	li	t5,4
1c00c100:	84b6                	mv	s1,a3
1c00c102:	893a                	mv	s2,a4
1c00c104:	89c2                	mv	s3,a6
1c00c106:	012f8333          	add	t1,t6,s2
1c00c10a:	030a                	slli	t1,t1,0x2
1c00c10c:	9322                	add	t1,t1,s0
1c00c10e:	40628833          	sub	a6,t0,t1
1c00c112:	1871                	addi	a6,a6,-4
1c00c114:	00285813          	srli	a6,a6,0x2
1c00c118:	01ce8ab3          	add	s5,t4,t3
1c00c11c:	88d2                	mv	a7,s4
1c00c11e:	0805                	addi	a6,a6,1
1c00c120:	01a8407b          	lp.setup	x0,a6,1c00c154 <mm_unroll_4x2+0x6a4>
1c00c124:	00000713          	li	a4,0
1c00c128:	02705463          	blez	t2,1c00c150 <mm_unroll_4x2+0x6a0>
1c00c12c:	41ca87b3          	sub	a5,s5,t3
1c00c130:	17f1                	addi	a5,a5,-4
1c00c132:	00000713          	li	a4,0
1c00c136:	8389                	srli	a5,a5,0x2
1c00c138:	8646                	mv	a2,a7
1c00c13a:	86f2                	mv	a3,t3
1c00c13c:	0785                	addi	a5,a5,1
1c00c13e:	0001                	nop
1c00c140:	0067c0fb          	lp.setup	x1,a5,1c00c14c <mm_unroll_4x2+0x69c>
1c00c144:	0046a50b          	p.lw	a0,4(a3!)
1c00c148:	0046258b          	p.lw	a1,4(a2!)
1c00c14c:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00c150:	00e3222b          	p.sw	a4,4(t1!)
1c00c154:	98f6                	add	a7,a7,t4
1c00c156:	1f7d                	addi	t5,t5,-1
1c00c158:	9fa6                	add	t6,t6,s1
1c00c15a:	92ce                	add	t0,t0,s3
1c00c15c:	8e56                	mv	t3,s5
1c00c15e:	fa0f14e3          	bnez	t5,1c00c106 <mm_unroll_4x2+0x656>
1c00c162:	bbd5                	j	1c00bf56 <mm_unroll_4x2+0x4a6>
1c00c164:	547a                	lw	s0,188(sp)
1c00c166:	54ea                	lw	s1,184(sp)
1c00c168:	595a                	lw	s2,180(sp)
1c00c16a:	59ca                	lw	s3,176(sp)
1c00c16c:	5a3a                	lw	s4,172(sp)
1c00c16e:	5aaa                	lw	s5,168(sp)
1c00c170:	5b1a                	lw	s6,164(sp)
1c00c172:	5b8a                	lw	s7,160(sp)
1c00c174:	4c7a                	lw	s8,156(sp)
1c00c176:	4cea                	lw	s9,152(sp)
1c00c178:	4d5a                	lw	s10,148(sp)
1c00c17a:	4dca                	lw	s11,144(sp)
1c00c17c:	6129                	addi	sp,sp,192
1c00c17e:	b33fd06f          	j	1c009cb0 <mm_unroll_1x8>
1c00c182:	547a                	lw	s0,188(sp)
1c00c184:	54ea                	lw	s1,184(sp)
1c00c186:	595a                	lw	s2,180(sp)
1c00c188:	59ca                	lw	s3,176(sp)
1c00c18a:	5a3a                	lw	s4,172(sp)
1c00c18c:	5aaa                	lw	s5,168(sp)
1c00c18e:	5b1a                	lw	s6,164(sp)
1c00c190:	5b8a                	lw	s7,160(sp)
1c00c192:	4c7a                	lw	s8,156(sp)
1c00c194:	4cea                	lw	s9,152(sp)
1c00c196:	4d5a                	lw	s10,148(sp)
1c00c198:	4dca                	lw	s11,144(sp)
1c00c19a:	6129                	addi	sp,sp,192
1c00c19c:	e51fe06f          	j	1c00afec <mm_unroll_2x2>
1c00c1a0:	4505                	li	a0,1
1c00c1a2:	c82a                	sw	a0,16(sp)
1c00c1a4:	bcb9                	j	1c00bc02 <mm_unroll_4x2+0x152>
1c00c1a6:	4505                	li	a0,1
1c00c1a8:	c82a                	sw	a0,16(sp)
1c00c1aa:	b9ed                	j	1c00bea4 <mm_unroll_4x2+0x3f4>
1c00c1ac:	0000                	unimp
	...

1c00c1b0 <mm_unroll_4x4>:
1c00c1b0:	00c52303          	lw	t1,12(a0)
1c00c1b4:	7115                	addi	sp,sp,-224
1c00c1b6:	cfa2                	sw	s0,220(sp)
1c00c1b8:	c20335b3          	p.bclr	a1,t1,1,0
1c00c1bc:	cda6                	sw	s1,216(sp)
1c00c1be:	cbca                	sw	s2,212(sp)
1c00c1c0:	c9ce                	sw	s3,208(sp)
1c00c1c2:	c7d2                	sw	s4,204(sp)
1c00c1c4:	c5d6                	sw	s5,200(sp)
1c00c1c6:	c3da                	sw	s6,196(sp)
1c00c1c8:	c1de                	sw	s7,192(sp)
1c00c1ca:	df62                	sw	s8,188(sp)
1c00c1cc:	dd66                	sw	s9,184(sp)
1c00c1ce:	db6a                	sw	s10,180(sp)
1c00c1d0:	d96e                	sw	s11,176(sp)
1c00c1d2:	c31a                	sw	t1,132(sp)
1c00c1d4:	473d                	li	a4,15
1c00c1d6:	d8ae                	sw	a1,112(sp)
1c00c1d8:	f14026f3          	csrr	a3,mhartid
1c00c1dc:	28b75763          	ble	a1,a4,1c00c46a <mm_unroll_4x4+0x2ba>
1c00c1e0:	477d                	li	a4,31
1c00c1e2:	26b75563          	ble	a1,a4,1c00c44c <mm_unroll_4x4+0x29c>
1c00c1e6:	4918                	lw	a4,16(a0)
1c00c1e8:	87aa                	mv	a5,a0
1c00c1ea:	d4ba                	sw	a4,104(sp)
1c00c1ec:	5826                	lw	a6,104(sp)
1c00c1ee:	470d                	li	a4,3
1c00c1f0:	4d075963          	ble	a6,a4,1c00c6c2 <mm_unroll_4x4+0x512>
1c00c1f4:	00758713          	addi	a4,a1,7
1c00c1f8:	f456b6b3          	p.bclr	a3,a3,26,5
1c00c1fc:	870d                	srai	a4,a4,0x3
1c00c1fe:	02e68333          	mul	t1,a3,a4
1c00c202:	88ae                	mv	a7,a1
1c00c204:	4790                	lw	a2,8(a5)
1c00c206:	43cc                	lw	a1,4(a5)
1c00c208:	4bdc                	lw	a5,20(a5)
1c00c20a:	d336                	sw	a3,164(sp)
1c00c20c:	4d14                	lw	a3,24(a0)
1c00c20e:	c4be                	sw	a5,72(sp)
1c00c210:	479a                	lw	a5,132(sp)
1c00c212:	4108                	lw	a0,0(a0)
1c00c214:	971a                	add	a4,a4,t1
1c00c216:	411787b3          	sub	a5,a5,a7
1c00c21a:	05174733          	p.min	a4,a4,a7
1c00c21e:	ce9a                	sw	t1,92(sp)
1c00c220:	cb2a                	sw	a0,148(sp)
1c00c222:	c12e                	sw	a1,128(sp)
1c00c224:	c732                	sw	a2,140(sp)
1c00c226:	d53e                	sw	a5,168(sp)
1c00c228:	c53a                	sw	a4,136(sp)
1c00c22a:	24069f63          	bnez	a3,1c00c488 <mm_unroll_4x4+0x2d8>
1c00c22e:	889a                	mv	a7,t1
1c00c230:	432a                	lw	t1,136(sp)
1c00c232:	4e68d963          	ble	t1,a7,1c00c724 <mm_unroll_4x4+0x574>
1c00c236:	4626                	lw	a2,72(sp)
1c00c238:	00281293          	slli	t0,a6,0x2
1c00c23c:	c2083533          	p.bclr	a0,a6,1,0
1c00c240:	5826                	lw	a6,104(sp)
1c00c242:	060a                	slli	a2,a2,0x2
1c00c244:	d6b2                	sw	a2,108(sp)
1c00c246:	caaa                	sw	a0,84(sp)
1c00c248:	00251593          	slli	a1,a0,0x2
1c00c24c:	4626                	lw	a2,72(sp)
1c00c24e:	4576                	lw	a0,92(sp)
1c00c250:	fa283833          	p.bclr	a6,a6,29,2
1c00c254:	d0c2                	sw	a6,96(sp)
1c00c256:	5826                	lw	a6,104(sp)
1c00c258:	42a60633          	p.mac	a2,a2,a0
1c00c25c:	00188693          	addi	a3,a7,1
1c00c260:	00288713          	addi	a4,a7,2
1c00c264:	00388793          	addi	a5,a7,3
1c00c268:	48a6                	lw	a7,72(sp)
1c00c26a:	5326                	lw	t1,104(sp)
1c00c26c:	c682                	sw	zero,76(sp)
1c00c26e:	0892                	slli	a7,a7,0x4
1c00c270:	c8c6                	sw	a7,80(sp)
1c00c272:	48da                	lw	a7,148(sp)
1c00c274:	02a80833          	mul	a6,a6,a0
1c00c278:	060a                	slli	a2,a2,0x2
1c00c27a:	9646                	add	a2,a2,a7
1c00c27c:	dc32                	sw	a2,56(sp)
1c00c27e:	5636                	lw	a2,108(sp)
1c00c280:	450a                	lw	a0,128(sp)
1c00c282:	0312                	slli	t1,t1,0x4
1c00c284:	d29a                	sw	t1,100(sp)
1c00c286:	95aa                	add	a1,a1,a0
1c00c288:	cf2e                	sw	a1,156(sp)
1c00c28a:	ccc2                	sw	a6,88(sp)
1c00c28c:	4876                	lw	a6,92(sp)
1c00c28e:	00a285b3          	add	a1,t0,a0
1c00c292:	435a                	lw	t1,148(sp)
1c00c294:	430608b3          	p.mac	a7,a2,a6
1c00c298:	455a                	lw	a0,148(sp)
1c00c29a:	cd2e                	sw	a1,152(sp)
1c00c29c:	45ba                	lw	a1,140(sp)
1c00c29e:	da46                	sw	a7,52(sp)
1c00c2a0:	48da                	lw	a7,148(sp)
1c00c2a2:	42e60333          	p.mac	t1,a2,a4
1c00c2a6:	42d608b3          	p.mac	a7,a2,a3
1c00c2aa:	c09a                	sw	t1,64(sp)
1c00c2ac:	42f60533          	p.mac	a0,a2,a5
1c00c2b0:	463a                	lw	a2,140(sp)
1c00c2b2:	c2c6                	sw	a7,68(sp)
1c00c2b4:	42568633          	p.mac	a2,a3,t0
1c00c2b8:	de2a                	sw	a0,60(sp)
1c00c2ba:	425805b3          	p.mac	a1,a6,t0
1c00c2be:	deb2                	sw	a2,124(sp)
1c00c2c0:	c92e                	sw	a1,144(sp)
1c00c2c2:	46ba                	lw	a3,140(sp)
1c00c2c4:	425706b3          	p.mac	a3,a4,t0
1c00c2c8:	473a                	lw	a4,140(sp)
1c00c2ca:	42578733          	p.mac	a4,a5,t0
1c00c2ce:	dcb6                	sw	a3,120(sp)
1c00c2d0:	daba                	sw	a4,116(sp)
1c00c2d2:	45d6                	lw	a1,84(sp)
1c00c2d4:	40058763          	beqz	a1,1c00c6e2 <mm_unroll_4x4+0x532>
1c00c2d8:	4836                	lw	a6,76(sp)
1c00c2da:	464a                	lw	a2,144(sp)
1c00c2dc:	58f6                	lw	a7,124(sp)
1c00c2de:	5366                	lw	t1,120(sp)
1c00c2e0:	5556                	lw	a0,116(sp)
1c00c2e2:	fff58693          	addi	a3,a1,-1
1c00c2e6:	00358793          	addi	a5,a1,3
1c00c2ea:	458a                	lw	a1,128(sp)
1c00c2ec:	8289                	srli	a3,a3,0x2
1c00c2ee:	9642                	add	a2,a2,a6
1c00c2f0:	98c2                	add	a7,a7,a6
1c00c2f2:	9342                	add	t1,t1,a6
1c00c2f4:	9542                	add	a0,a0,a6
1c00c2f6:	0685                	addi	a3,a3,1
1c00c2f8:	4711                	li	a4,4
1c00c2fa:	d232                	sw	a2,36(sp)
1c00c2fc:	d046                	sw	a7,32(sp)
1c00c2fe:	ce1a                	sw	t1,28(sp)
1c00c300:	cc2a                	sw	a0,24(sp)
1c00c302:	d82e                	sw	a1,48(sp)
1c00c304:	d636                	sw	a3,44(sp)
1c00c306:	6ee7ef63          	bltu	a5,a4,1c00ca04 <mm_unroll_4x4+0x854>
1c00c30a:	d416                	sw	t0,40(sp)
1c00c30c:	aa19                	j	1c00c422 <mm_unroll_4x4+0x272>
1c00c30e:	57e2                	lw	a5,56(sp)
1c00c310:	5852                	lw	a6,52(sp)
1c00c312:	00000b93          	li	s7,0
1c00c316:	5df2                	lw	s11,60(sp)
1c00c318:	41078e33          	sub	t3,a5,a6
1c00c31c:	1e71                	addi	t3,t3,-4
1c00c31e:	8b5e                	mv	s6,s7
1c00c320:	8ade                	mv	s5,s7
1c00c322:	8a5e                	mv	s4,s7
1c00c324:	89de                	mv	s3,s7
1c00c326:	895e                	mv	s2,s7
1c00c328:	84de                	mv	s1,s7
1c00c32a:	845e                	mv	s0,s7
1c00c32c:	83de                	mv	t2,s7
1c00c32e:	82de                	mv	t0,s7
1c00c330:	8fde                	mv	t6,s7
1c00c332:	8f5e                	mv	t5,s7
1c00c334:	8ede                	mv	t4,s7
1c00c336:	4d06                	lw	s10,64(sp)
1c00c338:	4c96                	lw	s9,68(sp)
1c00c33a:	57c2                	lw	a5,48(sp)
1c00c33c:	002e5e13          	srli	t3,t3,0x2
1c00c340:	ca02                	sw	zero,20(sp)
1c00c342:	c602                	sw	zero,12(sp)
1c00c344:	c802                	sw	zero,16(sp)
1c00c346:	8c42                	mv	s8,a6
1c00c348:	0e05                	addi	t3,t3,1
1c00c34a:	0001                	nop
1c00c34c:	035e40fb          	lp.setup	x1,t3,1c00c3b6 <mm_unroll_4x4+0x206>
1c00c350:	4388                	lw	a0,0(a5)
1c00c352:	43cc                	lw	a1,4(a5)
1c00c354:	4790                	lw	a2,8(a5)
1c00c356:	47d4                	lw	a3,12(a5)
1c00c358:	004c230b          	p.lw	t1,4(s8!)
1c00c35c:	004ca88b          	p.lw	a7,4(s9!)
1c00c360:	004d280b          	p.lw	a6,4(s10!)
1c00c364:	004da70b          	p.lw	a4,4(s11!)
1c00c368:	e8a37ec3          	fmadd.s	t4,t1,a0,t4
1c00c36c:	f8b37fc3          	fmadd.s	t6,t1,a1,t6
1c00c370:	38c373c3          	fmadd.s	t2,t1,a2,t2
1c00c374:	48d374c3          	fmadd.s	s1,t1,a3,s1
1c00c378:	991579c3          	fmadd.s	s3,a0,a7,s3
1c00c37c:	4332                	lw	t1,12(sp)
1c00c37e:	a915fac3          	fmadd.s	s5,a1,a7,s5
1c00c382:	f1057f43          	fmadd.s	t5,a0,a6,t5
1c00c386:	2905f2c3          	fmadd.s	t0,a1,a6,t0
1c00c38a:	a0e57a43          	fmadd.s	s4,a0,a4,s4
1c00c38e:	b0e5fb43          	fmadd.s	s6,a1,a4,s6
1c00c392:	4542                	lw	a0,16(sp)
1c00c394:	45d2                	lw	a1,20(sp)
1c00c396:	3116f343          	fmadd.s	t1,a3,a7,t1
1c00c39a:	50e67543          	fmadd.s	a0,a2,a4,a0
1c00c39e:	58e6f5c3          	fmadd.s	a1,a3,a4,a1
1c00c3a2:	b9167bc3          	fmadd.s	s7,a2,a7,s7
1c00c3a6:	41067443          	fmadd.s	s0,a2,a6,s0
1c00c3aa:	5622                	lw	a2,40(sp)
1c00c3ac:	c61a                	sw	t1,12(sp)
1c00c3ae:	c82a                	sw	a0,16(sp)
1c00c3b0:	ca2e                	sw	a1,20(sp)
1c00c3b2:	9106f943          	fmadd.s	s2,a3,a6,s2
1c00c3b6:	97b2                	add	a5,a5,a2
1c00c3b8:	5692                	lw	a3,36(sp)
1c00c3ba:	5702                	lw	a4,32(sp)
1c00c3bc:	47b2                	lw	a5,12(sp)
1c00c3be:	4872                	lw	a6,28(sp)
1c00c3c0:	01d6a023          	sw	t4,0(a3)
1c00c3c4:	01f6a223          	sw	t6,4(a3)
1c00c3c8:	0076a423          	sw	t2,8(a3)
1c00c3cc:	c6c4                	sw	s1,12(a3)
1c00c3ce:	48e2                	lw	a7,24(sp)
1c00c3d0:	55c2                	lw	a1,48(sp)
1c00c3d2:	5632                	lw	a2,44(sp)
1c00c3d4:	4342                	lw	t1,16(sp)
1c00c3d6:	4552                	lw	a0,20(sp)
1c00c3d8:	01372023          	sw	s3,0(a4)
1c00c3dc:	01572223          	sw	s5,4(a4)
1c00c3e0:	01772423          	sw	s7,8(a4)
1c00c3e4:	c75c                	sw	a5,12(a4)
1c00c3e6:	01e82023          	sw	t5,0(a6)
1c00c3ea:	00582223          	sw	t0,4(a6)
1c00c3ee:	00882423          	sw	s0,8(a6)
1c00c3f2:	01282623          	sw	s2,12(a6)
1c00c3f6:	0148a023          	sw	s4,0(a7)
1c00c3fa:	0168a223          	sw	s6,4(a7)
1c00c3fe:	0068a423          	sw	t1,8(a7)
1c00c402:	00a8a623          	sw	a0,12(a7)
1c00c406:	06c1                	addi	a3,a3,16
1c00c408:	0741                	addi	a4,a4,16
1c00c40a:	0841                	addi	a6,a6,16
1c00c40c:	08c1                	addi	a7,a7,16
1c00c40e:	05c1                	addi	a1,a1,16
1c00c410:	167d                	addi	a2,a2,-1
1c00c412:	d236                	sw	a3,36(sp)
1c00c414:	d03a                	sw	a4,32(sp)
1c00c416:	ce42                	sw	a6,28(sp)
1c00c418:	cc46                	sw	a7,24(sp)
1c00c41a:	d82e                	sw	a1,48(sp)
1c00c41c:	d632                	sw	a2,44(sp)
1c00c41e:	2c060163          	beqz	a2,1c00c6e0 <mm_unroll_4x4+0x530>
1c00c422:	46a6                	lw	a3,72(sp)
1c00c424:	eed045e3          	bgtz	a3,1c00c30e <mm_unroll_4x4+0x15e>
1c00c428:	00000b93          	li	s7,0
1c00c42c:	ca02                	sw	zero,20(sp)
1c00c42e:	c602                	sw	zero,12(sp)
1c00c430:	c802                	sw	zero,16(sp)
1c00c432:	8b5e                	mv	s6,s7
1c00c434:	8ade                	mv	s5,s7
1c00c436:	8a5e                	mv	s4,s7
1c00c438:	89de                	mv	s3,s7
1c00c43a:	895e                	mv	s2,s7
1c00c43c:	84de                	mv	s1,s7
1c00c43e:	845e                	mv	s0,s7
1c00c440:	83de                	mv	t2,s7
1c00c442:	82de                	mv	t0,s7
1c00c444:	8fde                	mv	t6,s7
1c00c446:	8f5e                	mv	t5,s7
1c00c448:	8ede                	mv	t4,s7
1c00c44a:	b7bd                	j	1c00c3b8 <mm_unroll_4x4+0x208>
1c00c44c:	447e                	lw	s0,220(sp)
1c00c44e:	44ee                	lw	s1,216(sp)
1c00c450:	495e                	lw	s2,212(sp)
1c00c452:	49ce                	lw	s3,208(sp)
1c00c454:	4a3e                	lw	s4,204(sp)
1c00c456:	4aae                	lw	s5,200(sp)
1c00c458:	4b1e                	lw	s6,196(sp)
1c00c45a:	4b8e                	lw	s7,192(sp)
1c00c45c:	5c7a                	lw	s8,188(sp)
1c00c45e:	5cea                	lw	s9,184(sp)
1c00c460:	5d5a                	lw	s10,180(sp)
1c00c462:	5dca                	lw	s11,176(sp)
1c00c464:	612d                	addi	sp,sp,224
1c00c466:	88aff06f          	j	1c00b4f0 <mm_unroll_2x4>
1c00c46a:	447e                	lw	s0,220(sp)
1c00c46c:	44ee                	lw	s1,216(sp)
1c00c46e:	495e                	lw	s2,212(sp)
1c00c470:	49ce                	lw	s3,208(sp)
1c00c472:	4a3e                	lw	s4,204(sp)
1c00c474:	4aae                	lw	s5,200(sp)
1c00c476:	4b1e                	lw	s6,196(sp)
1c00c478:	4b8e                	lw	s7,192(sp)
1c00c47a:	5c7a                	lw	s8,188(sp)
1c00c47c:	5cea                	lw	s9,184(sp)
1c00c47e:	5d5a                	lw	s10,180(sp)
1c00c480:	5dca                	lw	s11,176(sp)
1c00c482:	612d                	addi	sp,sp,224
1c00c484:	82dfd06f          	j	1c009cb0 <mm_unroll_1x8>
1c00c488:	3ae35e63          	ble	a4,t1,1c00c844 <mm_unroll_4x4+0x694>
1c00c48c:	88c2                	mv	a7,a6
1c00c48e:	8642                	mv	a2,a6
1c00c490:	088a                	slli	a7,a7,0x2
1c00c492:	0612                	slli	a2,a2,0x4
1c00c494:	dcc6                	sw	a7,120(sp)
1c00c496:	deb2                	sw	a2,124(sp)
1c00c498:	48f6                	lw	a7,92(sp)
1c00c49a:	4626                	lw	a2,72(sp)
1c00c49c:	8542                	mv	a0,a6
1c00c49e:	4826                	lw	a6,72(sp)
1c00c4a0:	43160633          	p.mac	a2,a2,a7
1c00c4a4:	00130693          	addi	a3,t1,1
1c00c4a8:	00230713          	addi	a4,t1,2
1c00c4ac:	00330793          	addi	a5,t1,3
1c00c4b0:	45a6                	lw	a1,72(sp)
1c00c4b2:	5326                	lw	t1,104(sp)
1c00c4b4:	080a                	slli	a6,a6,0x2
1c00c4b6:	de42                	sw	a6,60(sp)
1c00c4b8:	c2033333          	p.bclr	t1,t1,1,0
1c00c4bc:	fa253533          	p.bclr	a0,a0,29,2
1c00c4c0:	0592                	slli	a1,a1,0x4
1c00c4c2:	d29a                	sw	t1,100(sp)
1c00c4c4:	c92a                	sw	a0,144(sp)
1c00c4c6:	c2ae                	sw	a1,68(sp)
1c00c4c8:	455a                	lw	a0,148(sp)
1c00c4ca:	55f2                	lw	a1,60(sp)
1c00c4cc:	5326                	lw	t1,104(sp)
1c00c4ce:	060a                	slli	a2,a2,0x2
1c00c4d0:	962a                	add	a2,a2,a0
1c00c4d2:	03130333          	mul	t1,t1,a7
1c00c4d6:	5866                	lw	a6,120(sp)
1c00c4d8:	c6b2                	sw	a2,76(sp)
1c00c4da:	463a                	lw	a2,140(sp)
1c00c4dc:	d082                	sw	zero,96(sp)
1c00c4de:	43158533          	p.mac	a0,a1,a7
1c00c4e2:	da9a                	sw	t1,116(sp)
1c00c4e4:	435a                	lw	t1,148(sp)
1c00c4e6:	43088633          	p.mac	a2,a7,a6
1c00c4ea:	c0aa                	sw	a0,64(sp)
1c00c4ec:	48da                	lw	a7,148(sp)
1c00c4ee:	455a                	lw	a0,148(sp)
1c00c4f0:	42d588b3          	p.mac	a7,a1,a3
1c00c4f4:	d6b2                	sw	a2,108(sp)
1c00c4f6:	463a                	lw	a2,140(sp)
1c00c4f8:	42e58333          	p.mac	t1,a1,a4
1c00c4fc:	ccc6                	sw	a7,88(sp)
1c00c4fe:	42f58533          	p.mac	a0,a1,a5
1c00c502:	45ba                	lw	a1,140(sp)
1c00c504:	ca9a                	sw	t1,84(sp)
1c00c506:	430685b3          	p.mac	a1,a3,a6
1c00c50a:	46ba                	lw	a3,140(sp)
1c00c50c:	c8aa                	sw	a0,80(sp)
1c00c50e:	43070633          	p.mac	a2,a4,a6
1c00c512:	d12e                	sw	a1,160(sp)
1c00c514:	430786b3          	p.mac	a3,a5,a6
1c00c518:	cf32                	sw	a2,156(sp)
1c00c51a:	cd36                	sw	a3,152(sp)
1c00c51c:	470a                	lw	a4,128(sp)
1c00c51e:	5796                	lw	a5,100(sp)
1c00c520:	5872                	lw	a6,60(sp)
1c00c522:	43078733          	p.mac	a4,a5,a6
1c00c526:	d73a                	sw	a4,172(sp)
1c00c528:	5616                	lw	a2,100(sp)
1c00c52a:	2c060863          	beqz	a2,1c00c7fa <mm_unroll_4x4+0x64a>
1c00c52e:	5886                	lw	a7,96(sp)
1c00c530:	580a                	lw	a6,160(sp)
1c00c532:	437a                	lw	t1,156(sp)
1c00c534:	456a                	lw	a0,152(sp)
1c00c536:	fff60693          	addi	a3,a2,-1
1c00c53a:	00360793          	addi	a5,a2,3
1c00c53e:	458a                	lw	a1,128(sp)
1c00c540:	5636                	lw	a2,108(sp)
1c00c542:	8289                	srli	a3,a3,0x2
1c00c544:	9846                	add	a6,a6,a7
1c00c546:	9346                	add	t1,t1,a7
1c00c548:	9546                	add	a0,a0,a7
1c00c54a:	0685                	addi	a3,a3,1
1c00c54c:	4711                	li	a4,4
1c00c54e:	d642                	sw	a6,44(sp)
1c00c550:	d41a                	sw	t1,40(sp)
1c00c552:	d22a                	sw	a0,36(sp)
1c00c554:	da2e                	sw	a1,52(sp)
1c00c556:	d832                	sw	a2,48(sp)
1c00c558:	dc36                	sw	a3,56(sp)
1c00c55a:	12e7fe63          	bleu	a4,a5,1c00c696 <mm_unroll_4x4+0x4e6>
1c00c55e:	a145                	j	1c00c9fe <mm_unroll_4x4+0x84e>
1c00c560:	4786                	lw	a5,64(sp)
1c00c562:	4736                	lw	a4,76(sp)
1c00c564:	00000993          	li	s3,0
1c00c568:	5872                	lw	a6,60(sp)
1c00c56a:	58d2                	lw	a7,52(sp)
1c00c56c:	40f70333          	sub	t1,a4,a5
1c00c570:	1371                	addi	t1,t1,-4
1c00c572:	01180ab3          	add	s5,a6,a7
1c00c576:	894e                	mv	s2,s3
1c00c578:	84ce                	mv	s1,s3
1c00c57a:	844e                	mv	s0,s3
1c00c57c:	83ce                	mv	t2,s3
1c00c57e:	82ce                	mv	t0,s3
1c00c580:	8fce                	mv	t6,s3
1c00c582:	8f4e                	mv	t5,s3
1c00c584:	8ece                	mv	t4,s3
1c00c586:	8e4e                	mv	t3,s3
1c00c588:	4d46                	lw	s10,80(sp)
1c00c58a:	4cd6                	lw	s9,84(sp)
1c00c58c:	4c66                	lw	s8,88(sp)
1c00c58e:	01580a33          	add	s4,a6,s5
1c00c592:	00235313          	srli	t1,t1,0x2
1c00c596:	d002                	sw	zero,32(sp)
1c00c598:	ca02                	sw	zero,20(sp)
1c00c59a:	ce02                	sw	zero,28(sp)
1c00c59c:	c802                	sw	zero,16(sp)
1c00c59e:	cc02                	sw	zero,24(sp)
1c00c5a0:	c602                	sw	zero,12(sp)
1c00c5a2:	01480db3          	add	s11,a6,s4
1c00c5a6:	8bc6                	mv	s7,a7
1c00c5a8:	8b3e                	mv	s6,a5
1c00c5aa:	0305                	addi	t1,t1,1
1c00c5ac:	03d340fb          	lp.setup	x1,t1,1c00c626 <mm_unroll_4x4+0x476>
1c00c5b0:	004b288b          	p.lw	a7,4(s6!)
1c00c5b4:	004ba58b          	p.lw	a1,4(s7!)
1c00c5b8:	004aa60b          	p.lw	a2,4(s5!)
1c00c5bc:	004a268b          	p.lw	a3,4(s4!)
1c00c5c0:	004da70b          	p.lw	a4,4(s11!)
1c00c5c4:	004c280b          	p.lw	a6,4(s8!)
1c00c5c8:	e0b8fe43          	fmadd.s	t3,a7,a1,t3
1c00c5cc:	f0c8ff43          	fmadd.s	t5,a7,a2,t5
1c00c5d0:	28d8f2c3          	fmadd.s	t0,a7,a3,t0
1c00c5d4:	40e8f443          	fmadd.s	s0,a7,a4,s0
1c00c5d8:	48b2                	lw	a7,12(sp)
1c00c5da:	004ca50b          	p.lw	a0,4(s9!)
1c00c5de:	004d278b          	p.lw	a5,4(s10!)
1c00c5e2:	890678c3          	fmadd.s	a7,a2,a6,a7
1c00c5e6:	e8a5fec3          	fmadd.s	t4,a1,a0,t4
1c00c5ea:	f8a67fc3          	fmadd.s	t6,a2,a0,t6
1c00c5ee:	38a6f3c3          	fmadd.s	t2,a3,a0,t2
1c00c5f2:	48a774c3          	fmadd.s	s1,a4,a0,s1
1c00c5f6:	c646                	sw	a7,12(sp)
1c00c5f8:	48c2                	lw	a7,16(sp)
1c00c5fa:	4562                	lw	a0,24(sp)
1c00c5fc:	9105f943          	fmadd.s	s2,a1,a6,s2
1c00c600:	8906f8c3          	fmadd.s	a7,a3,a6,a7
1c00c604:	50f67543          	fmadd.s	a0,a2,a5,a0
1c00c608:	98f5f9c3          	fmadd.s	s3,a1,a5,s3
1c00c60c:	5602                	lw	a2,32(sp)
1c00c60e:	45f2                	lw	a1,28(sp)
1c00c610:	c846                	sw	a7,16(sp)
1c00c612:	48d2                	lw	a7,20(sp)
1c00c614:	58f6f5c3          	fmadd.s	a1,a3,a5,a1
1c00c618:	60f77643          	fmadd.s	a2,a4,a5,a2
1c00c61c:	890778c3          	fmadd.s	a7,a4,a6,a7
1c00c620:	cc2a                	sw	a0,24(sp)
1c00c622:	ce2e                	sw	a1,28(sp)
1c00c624:	d032                	sw	a2,32(sp)
1c00c626:	ca46                	sw	a7,20(sp)
1c00c628:	56c2                	lw	a3,48(sp)
1c00c62a:	5732                	lw	a4,44(sp)
1c00c62c:	47b2                	lw	a5,12(sp)
1c00c62e:	4842                	lw	a6,16(sp)
1c00c630:	01c6a023          	sw	t3,0(a3)
1c00c634:	01e6a223          	sw	t5,4(a3)
1c00c638:	0056a423          	sw	t0,8(a3)
1c00c63c:	c6c0                	sw	s0,12(a3)
1c00c63e:	5322                	lw	t1,40(sp)
1c00c640:	c35c                	sw	a5,4(a4)
1c00c642:	01072423          	sw	a6,8(a4)
1c00c646:	01272023          	sw	s2,0(a4)
1c00c64a:	01172623          	sw	a7,12(a4)
1c00c64e:	5512                	lw	a0,36(sp)
1c00c650:	5742                	lw	a4,48(sp)
1c00c652:	57b2                	lw	a5,44(sp)
1c00c654:	45e2                	lw	a1,24(sp)
1c00c656:	4672                	lw	a2,28(sp)
1c00c658:	5682                	lw	a3,32(sp)
1c00c65a:	01d32023          	sw	t4,0(t1)
1c00c65e:	01f32223          	sw	t6,4(t1)
1c00c662:	00732423          	sw	t2,8(t1)
1c00c666:	00932623          	sw	s1,12(t1)
1c00c66a:	01352023          	sw	s3,0(a0)
1c00c66e:	c14c                	sw	a1,4(a0)
1c00c670:	c510                	sw	a2,8(a0)
1c00c672:	c554                	sw	a3,12(a0)
1c00c674:	0341                	addi	t1,t1,16
1c00c676:	0741                	addi	a4,a4,16
1c00c678:	07c1                	addi	a5,a5,16
1c00c67a:	0541                	addi	a0,a0,16
1c00c67c:	d41a                	sw	t1,40(sp)
1c00c67e:	5852                	lw	a6,52(sp)
1c00c680:	d83a                	sw	a4,48(sp)
1c00c682:	d63e                	sw	a5,44(sp)
1c00c684:	d22a                	sw	a0,36(sp)
1c00c686:	4896                	lw	a7,68(sp)
1c00c688:	5362                	lw	t1,56(sp)
1c00c68a:	9846                	add	a6,a6,a7
1c00c68c:	137d                	addi	t1,t1,-1
1c00c68e:	da42                	sw	a6,52(sp)
1c00c690:	dc1a                	sw	t1,56(sp)
1c00c692:	16030463          	beqz	t1,1c00c7fa <mm_unroll_4x4+0x64a>
1c00c696:	4526                	lw	a0,72(sp)
1c00c698:	eca044e3          	bgtz	a0,1c00c560 <mm_unroll_4x4+0x3b0>
1c00c69c:	00000993          	li	s3,0
1c00c6a0:	ca02                	sw	zero,20(sp)
1c00c6a2:	d002                	sw	zero,32(sp)
1c00c6a4:	ce02                	sw	zero,28(sp)
1c00c6a6:	c802                	sw	zero,16(sp)
1c00c6a8:	cc02                	sw	zero,24(sp)
1c00c6aa:	c602                	sw	zero,12(sp)
1c00c6ac:	894e                	mv	s2,s3
1c00c6ae:	84ce                	mv	s1,s3
1c00c6b0:	844e                	mv	s0,s3
1c00c6b2:	83ce                	mv	t2,s3
1c00c6b4:	82ce                	mv	t0,s3
1c00c6b6:	8fce                	mv	t6,s3
1c00c6b8:	8f4e                	mv	t5,s3
1c00c6ba:	8ece                	mv	t4,s3
1c00c6bc:	8e4e                	mv	t3,s3
1c00c6be:	48d2                	lw	a7,20(sp)
1c00c6c0:	b7a5                	j	1c00c628 <mm_unroll_4x4+0x478>
1c00c6c2:	447e                	lw	s0,220(sp)
1c00c6c4:	44ee                	lw	s1,216(sp)
1c00c6c6:	495e                	lw	s2,212(sp)
1c00c6c8:	49ce                	lw	s3,208(sp)
1c00c6ca:	4a3e                	lw	s4,204(sp)
1c00c6cc:	4aae                	lw	s5,200(sp)
1c00c6ce:	4b1e                	lw	s6,196(sp)
1c00c6d0:	4b8e                	lw	s7,192(sp)
1c00c6d2:	5c7a                	lw	s8,188(sp)
1c00c6d4:	5cea                	lw	s9,184(sp)
1c00c6d6:	5d5a                	lw	s10,180(sp)
1c00c6d8:	5dca                	lw	s11,176(sp)
1c00c6da:	612d                	addi	sp,sp,224
1c00c6dc:	911fe06f          	j	1c00afec <mm_unroll_2x2>
1c00c6e0:	52a2                	lw	t0,40(sp)
1c00c6e2:	5706                	lw	a4,96(sp)
1c00c6e4:	2a071063          	bnez	a4,1c00c984 <mm_unroll_4x4+0x7d4>
1c00c6e8:	48f6                	lw	a7,92(sp)
1c00c6ea:	45c6                	lw	a1,80(sp)
1c00c6ec:	4366                	lw	t1,88(sp)
1c00c6ee:	5552                	lw	a0,52(sp)
1c00c6f0:	0891                	addi	a7,a7,4
1c00c6f2:	cec6                	sw	a7,92(sp)
1c00c6f4:	4616                	lw	a2,68(sp)
1c00c6f6:	4686                	lw	a3,64(sp)
1c00c6f8:	5772                	lw	a4,60(sp)
1c00c6fa:	57e2                	lw	a5,56(sp)
1c00c6fc:	4836                	lw	a6,76(sp)
1c00c6fe:	5896                	lw	a7,100(sp)
1c00c700:	9316                	add	t1,t1,t0
1c00c702:	952e                	add	a0,a0,a1
1c00c704:	cc9a                	sw	t1,88(sp)
1c00c706:	da2a                	sw	a0,52(sp)
1c00c708:	432a                	lw	t1,136(sp)
1c00c70a:	4576                	lw	a0,92(sp)
1c00c70c:	962e                	add	a2,a2,a1
1c00c70e:	96ae                	add	a3,a3,a1
1c00c710:	972e                	add	a4,a4,a1
1c00c712:	97ae                	add	a5,a5,a1
1c00c714:	9846                	add	a6,a6,a7
1c00c716:	c2b2                	sw	a2,68(sp)
1c00c718:	c0b6                	sw	a3,64(sp)
1c00c71a:	de3a                	sw	a4,60(sp)
1c00c71c:	dc3e                	sw	a5,56(sp)
1c00c71e:	c6c2                	sw	a6,76(sp)
1c00c720:	ba6549e3          	blt	a0,t1,1c00c2d2 <mm_unroll_4x4+0x122>
1c00c724:	56aa                	lw	a3,168(sp)
1c00c726:	0ad05c63          	blez	a3,1c00c7de <mm_unroll_4x4+0x62e>
1c00c72a:	5726                	lw	a4,104(sp)
1c00c72c:	581a                	lw	a6,164(sp)
1c00c72e:	489a                	lw	a7,132(sp)
1c00c730:	00770793          	addi	a5,a4,7
1c00c734:	878d                	srai	a5,a5,0x3
1c00c736:	02f80fb3          	mul	t6,a6,a5
1c00c73a:	5346                	lw	t1,112(sp)
1c00c73c:	97fe                	add	a5,a5,t6
1c00c73e:	04e7c7b3          	p.min	a5,a5,a4
1c00c742:	09135e63          	ble	a7,t1,1c00c7de <mm_unroll_4x4+0x62e>
1c00c746:	08ffdc63          	ble	a5,t6,1c00c7de <mm_unroll_4x4+0x62e>
1c00c74a:	4526                	lw	a0,72(sp)
1c00c74c:	465a                	lw	a2,148(sp)
1c00c74e:	89c6                	mv	s3,a7
1c00c750:	88aa                	mv	a7,a0
1c00c752:	426508b3          	p.mac	a7,a0,t1
1c00c756:	869a                	mv	a3,t1
1c00c758:	00251f13          	slli	t5,a0,0x2
1c00c75c:	83b2                	mv	t2,a2
1c00c75e:	458a                	lw	a1,128(sp)
1c00c760:	493a                	lw	s2,140(sp)
1c00c762:	002f9e93          	slli	t4,t6,0x2
1c00c766:	078a                	slli	a5,a5,0x2
1c00c768:	9eae                	add	t4,t4,a1
1c00c76a:	00f58e33          	add	t3,a1,a5
1c00c76e:	02e30333          	mul	t1,t1,a4
1c00c772:	088a                	slli	a7,a7,0x2
1c00c774:	00271293          	slli	t0,a4,0x2
1c00c778:	98b2                	add	a7,a7,a2
1c00c77a:	8a3a                	mv	s4,a4
1c00c77c:	842a                	mv	s0,a0
1c00c77e:	84b6                	mv	s1,a3
1c00c780:	42df03b3          	p.mac	t2,t5,a3
1c00c784:	41de0833          	sub	a6,t3,t4
1c00c788:	01f30ab3          	add	s5,t1,t6
1c00c78c:	1871                	addi	a6,a6,-4
1c00c78e:	0a8a                	slli	s5,s5,0x2
1c00c790:	00285813          	srli	a6,a6,0x2
1c00c794:	9aca                	add	s5,s5,s2
1c00c796:	8b76                	mv	s6,t4
1c00c798:	0805                	addi	a6,a6,1
1c00c79a:	0001                	nop
1c00c79c:	01a8407b          	lp.setup	x0,a6,1c00c7d0 <mm_unroll_4x4+0x620>
1c00c7a0:	00000713          	li	a4,0
1c00c7a4:	02805463          	blez	s0,1c00c7cc <mm_unroll_4x4+0x61c>
1c00c7a8:	407887b3          	sub	a5,a7,t2
1c00c7ac:	17f1                	addi	a5,a5,-4
1c00c7ae:	00000713          	li	a4,0
1c00c7b2:	8389                	srli	a5,a5,0x2
1c00c7b4:	865a                	mv	a2,s6
1c00c7b6:	869e                	mv	a3,t2
1c00c7b8:	0785                	addi	a5,a5,1
1c00c7ba:	0001                	nop
1c00c7bc:	0067c0fb          	lp.setup	x1,a5,1c00c7c8 <mm_unroll_4x4+0x618>
1c00c7c0:	0046a50b          	p.lw	a0,4(a3!)
1c00c7c4:	2056758b          	p.lw	a1,t0(a2!)
1c00c7c8:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00c7cc:	00eaa22b          	p.sw	a4,4(s5!)
1c00c7d0:	0b11                	addi	s6,s6,4
1c00c7d2:	0485                	addi	s1,s1,1
1c00c7d4:	9352                	add	t1,t1,s4
1c00c7d6:	93fa                	add	t2,t2,t5
1c00c7d8:	98fa                	add	a7,a7,t5
1c00c7da:	fa9995e3          	bne	s3,s1,1c00c784 <mm_unroll_4x4+0x5d4>
1c00c7de:	447e                	lw	s0,220(sp)
1c00c7e0:	44ee                	lw	s1,216(sp)
1c00c7e2:	495e                	lw	s2,212(sp)
1c00c7e4:	49ce                	lw	s3,208(sp)
1c00c7e6:	4a3e                	lw	s4,204(sp)
1c00c7e8:	4aae                	lw	s5,200(sp)
1c00c7ea:	4b1e                	lw	s6,196(sp)
1c00c7ec:	4b8e                	lw	s7,192(sp)
1c00c7ee:	5c7a                	lw	s8,188(sp)
1c00c7f0:	5cea                	lw	s9,184(sp)
1c00c7f2:	5d5a                	lw	s10,180(sp)
1c00c7f4:	5dca                	lw	s11,176(sp)
1c00c7f6:	612d                	addi	sp,sp,224
1c00c7f8:	8082                	ret
1c00c7fa:	45ca                	lw	a1,144(sp)
1c00c7fc:	10059363          	bnez	a1,1c00c902 <mm_unroll_4x4+0x752>
1c00c800:	4876                	lw	a6,92(sp)
1c00c802:	58d6                	lw	a7,116(sp)
1c00c804:	5366                	lw	t1,120(sp)
1c00c806:	4596                	lw	a1,68(sp)
1c00c808:	0811                	addi	a6,a6,4
1c00c80a:	989a                	add	a7,a7,t1
1c00c80c:	4506                	lw	a0,64(sp)
1c00c80e:	cec2                	sw	a6,92(sp)
1c00c810:	4766                	lw	a4,88(sp)
1c00c812:	47d6                	lw	a5,84(sp)
1c00c814:	4846                	lw	a6,80(sp)
1c00c816:	dac6                	sw	a7,116(sp)
1c00c818:	48b6                	lw	a7,76(sp)
1c00c81a:	5636                	lw	a2,108(sp)
1c00c81c:	56f6                	lw	a3,124(sp)
1c00c81e:	5306                	lw	t1,96(sp)
1c00c820:	952e                	add	a0,a0,a1
1c00c822:	972e                	add	a4,a4,a1
1c00c824:	97ae                	add	a5,a5,a1
1c00c826:	982e                	add	a6,a6,a1
1c00c828:	98ae                	add	a7,a7,a1
1c00c82a:	c0aa                	sw	a0,64(sp)
1c00c82c:	45f6                	lw	a1,92(sp)
1c00c82e:	452a                	lw	a0,136(sp)
1c00c830:	9636                	add	a2,a2,a3
1c00c832:	9336                	add	t1,t1,a3
1c00c834:	d6b2                	sw	a2,108(sp)
1c00c836:	ccba                	sw	a4,88(sp)
1c00c838:	cabe                	sw	a5,84(sp)
1c00c83a:	c8c2                	sw	a6,80(sp)
1c00c83c:	c6c6                	sw	a7,76(sp)
1c00c83e:	d09a                	sw	t1,96(sp)
1c00c840:	cea5c4e3          	blt	a1,a0,1c00c528 <mm_unroll_4x4+0x378>
1c00c844:	572a                	lw	a4,168(sp)
1c00c846:	f8e05ce3          	blez	a4,1c00c7de <mm_unroll_4x4+0x62e>
1c00c84a:	5826                	lw	a6,104(sp)
1c00c84c:	589a                	lw	a7,164(sp)
1c00c84e:	431a                	lw	t1,132(sp)
1c00c850:	00780793          	addi	a5,a6,7
1c00c854:	878d                	srai	a5,a5,0x3
1c00c856:	02f88fb3          	mul	t6,a7,a5
1c00c85a:	55c6                	lw	a1,112(sp)
1c00c85c:	97fe                	add	a5,a5,t6
1c00c85e:	0507c7b3          	p.min	a5,a5,a6
1c00c862:	f665dee3          	ble	t1,a1,1c00c7de <mm_unroll_4x4+0x62e>
1c00c866:	f6ffdce3          	ble	a5,t6,1c00c7de <mm_unroll_4x4+0x62e>
1c00c86a:	4626                	lw	a2,72(sp)
1c00c86c:	8a1a                	mv	s4,t1
1c00c86e:	03058f33          	mul	t5,a1,a6
1c00c872:	83b2                	mv	t2,a2
1c00c874:	435a                	lw	t1,148(sp)
1c00c876:	440a                	lw	s0,128(sp)
1c00c878:	00261e13          	slli	t3,a2,0x2
1c00c87c:	46ba                	lw	a3,140(sp)
1c00c87e:	475a                	lw	a4,148(sp)
1c00c880:	00281293          	slli	t0,a6,0x2
1c00c884:	89b6                	mv	s3,a3
1c00c886:	8ac2                	mv	s5,a6
1c00c888:	42b603b3          	p.mac	t2,a2,a1
1c00c88c:	01e78eb3          	add	t4,a5,t5
1c00c890:	0e8a                	slli	t4,t4,0x2
1c00c892:	9eb6                	add	t4,t4,a3
1c00c894:	84b2                	mv	s1,a2
1c00c896:	892e                	mv	s2,a1
1c00c898:	42be0333          	p.mac	t1,t3,a1
1c00c89c:	038a                	slli	t2,t2,0x2
1c00c89e:	93ba                	add	t2,t2,a4
1c00c8a0:	43cf8433          	p.mac	s0,t6,t3
1c00c8a4:	01ff08b3          	add	a7,t5,t6
1c00c8a8:	088a                	slli	a7,a7,0x2
1c00c8aa:	98ce                	add	a7,a7,s3
1c00c8ac:	411e8833          	sub	a6,t4,a7
1c00c8b0:	1871                	addi	a6,a6,-4
1c00c8b2:	00285813          	srli	a6,a6,0x2
1c00c8b6:	8b22                	mv	s6,s0
1c00c8b8:	0805                	addi	a6,a6,1
1c00c8ba:	0001                	nop
1c00c8bc:	01a8407b          	lp.setup	x0,a6,1c00c8f0 <mm_unroll_4x4+0x740>
1c00c8c0:	00000713          	li	a4,0
1c00c8c4:	02905463          	blez	s1,1c00c8ec <mm_unroll_4x4+0x73c>
1c00c8c8:	406387b3          	sub	a5,t2,t1
1c00c8cc:	17f1                	addi	a5,a5,-4
1c00c8ce:	00000713          	li	a4,0
1c00c8d2:	8389                	srli	a5,a5,0x2
1c00c8d4:	865a                	mv	a2,s6
1c00c8d6:	869a                	mv	a3,t1
1c00c8d8:	0785                	addi	a5,a5,1
1c00c8da:	0001                	nop
1c00c8dc:	0067c0fb          	lp.setup	x1,a5,1c00c8e8 <mm_unroll_4x4+0x738>
1c00c8e0:	0046a50b          	p.lw	a0,4(a3!)
1c00c8e4:	0046258b          	p.lw	a1,4(a2!)
1c00c8e8:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00c8ec:	00e8a22b          	p.sw	a4,4(a7!)
1c00c8f0:	9b72                	add	s6,s6,t3
1c00c8f2:	0905                	addi	s2,s2,1
1c00c8f4:	9f56                	add	t5,t5,s5
1c00c8f6:	9e96                	add	t4,t4,t0
1c00c8f8:	9372                	add	t1,t1,t3
1c00c8fa:	93f2                	add	t2,t2,t3
1c00c8fc:	fb2a14e3          	bne	s4,s2,1c00c8a4 <mm_unroll_4x4+0x6f4>
1c00c900:	bdf9                	j	1c00c7de <mm_unroll_4x4+0x62e>
1c00c902:	5626                	lw	a2,104(sp)
1c00c904:	5696                	lw	a3,100(sp)
1c00c906:	eec6dde3          	ble	a2,a3,1c00c800 <mm_unroll_4x4+0x650>
1c00c90a:	57e6                	lw	a5,120(sp)
1c00c90c:	5736                	lw	a4,108(sp)
1c00c90e:	4e06                	lw	t3,64(sp)
1c00c910:	5fd6                	lw	t6,116(sp)
1c00c912:	5ef2                	lw	t4,60(sp)
1c00c914:	443a                	lw	s0,140(sp)
1c00c916:	43a6                	lw	t2,72(sp)
1c00c918:	5a3a                	lw	s4,172(sp)
1c00c91a:	00f702b3          	add	t0,a4,a5
1c00c91e:	4f11                	li	t5,4
1c00c920:	84b2                	mv	s1,a2
1c00c922:	8936                	mv	s2,a3
1c00c924:	89be                	mv	s3,a5
1c00c926:	012f8333          	add	t1,t6,s2
1c00c92a:	030a                	slli	t1,t1,0x2
1c00c92c:	9322                	add	t1,t1,s0
1c00c92e:	40628833          	sub	a6,t0,t1
1c00c932:	1871                	addi	a6,a6,-4
1c00c934:	00285813          	srli	a6,a6,0x2
1c00c938:	01ce8ab3          	add	s5,t4,t3
1c00c93c:	88d2                	mv	a7,s4
1c00c93e:	0805                	addi	a6,a6,1
1c00c940:	01a8407b          	lp.setup	x0,a6,1c00c974 <mm_unroll_4x4+0x7c4>
1c00c944:	00000713          	li	a4,0
1c00c948:	02705463          	blez	t2,1c00c970 <mm_unroll_4x4+0x7c0>
1c00c94c:	41ca87b3          	sub	a5,s5,t3
1c00c950:	17f1                	addi	a5,a5,-4
1c00c952:	00000713          	li	a4,0
1c00c956:	8389                	srli	a5,a5,0x2
1c00c958:	8646                	mv	a2,a7
1c00c95a:	86f2                	mv	a3,t3
1c00c95c:	0785                	addi	a5,a5,1
1c00c95e:	0001                	nop
1c00c960:	0067c0fb          	lp.setup	x1,a5,1c00c96c <mm_unroll_4x4+0x7bc>
1c00c964:	0046a50b          	p.lw	a0,4(a3!)
1c00c968:	0046258b          	p.lw	a1,4(a2!)
1c00c96c:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00c970:	00e3222b          	p.sw	a4,4(t1!)
1c00c974:	98f6                	add	a7,a7,t4
1c00c976:	1f7d                	addi	t5,t5,-1
1c00c978:	9fa6                	add	t6,t6,s1
1c00c97a:	92ce                	add	t0,t0,s3
1c00c97c:	8e56                	mv	t3,s5
1c00c97e:	fa0f14e3          	bnez	t5,1c00c926 <mm_unroll_4x4+0x776>
1c00c982:	bdbd                	j	1c00c800 <mm_unroll_4x4+0x650>
1c00c984:	57a6                	lw	a5,104(sp)
1c00c986:	4856                	lw	a6,84(sp)
1c00c988:	d6f850e3          	ble	a5,a6,1c00c6e8 <mm_unroll_4x4+0x538>
1c00c98c:	5e52                	lw	t3,52(sp)
1c00c98e:	4fe6                	lw	t6,88(sp)
1c00c990:	443a                	lw	s0,140(sp)
1c00c992:	4326                	lw	t1,72(sp)
1c00c994:	5936                	lw	s2,108(sp)
1c00c996:	49ea                	lw	s3,152(sp)
1c00c998:	4efa                	lw	t4,156(sp)
1c00c99a:	4f11                	li	t5,4
1c00c99c:	83c2                	mv	t2,a6
1c00c99e:	84be                	mv	s1,a5
1c00c9a0:	41d98833          	sub	a6,s3,t4
1c00c9a4:	007f88b3          	add	a7,t6,t2
1c00c9a8:	1871                	addi	a6,a6,-4
1c00c9aa:	088a                	slli	a7,a7,0x2
1c00c9ac:	00285813          	srli	a6,a6,0x2
1c00c9b0:	98a2                	add	a7,a7,s0
1c00c9b2:	01c90ab3          	add	s5,s2,t3
1c00c9b6:	8a76                	mv	s4,t4
1c00c9b8:	0805                	addi	a6,a6,1
1c00c9ba:	0001                	nop
1c00c9bc:	01a8407b          	lp.setup	x0,a6,1c00c9f0 <mm_unroll_4x4+0x840>
1c00c9c0:	00000713          	li	a4,0
1c00c9c4:	02605463          	blez	t1,1c00c9ec <mm_unroll_4x4+0x83c>
1c00c9c8:	41ca87b3          	sub	a5,s5,t3
1c00c9cc:	17f1                	addi	a5,a5,-4
1c00c9ce:	00000713          	li	a4,0
1c00c9d2:	8389                	srli	a5,a5,0x2
1c00c9d4:	8652                	mv	a2,s4
1c00c9d6:	86f2                	mv	a3,t3
1c00c9d8:	0785                	addi	a5,a5,1
1c00c9da:	0001                	nop
1c00c9dc:	0067c0fb          	lp.setup	x1,a5,1c00c9e8 <mm_unroll_4x4+0x838>
1c00c9e0:	0046a50b          	p.lw	a0,4(a3!)
1c00c9e4:	2056758b          	p.lw	a1,t0(a2!)
1c00c9e8:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00c9ec:	00e8a22b          	p.sw	a4,4(a7!)
1c00c9f0:	0a11                	addi	s4,s4,4
1c00c9f2:	1f7d                	addi	t5,t5,-1
1c00c9f4:	9fa6                	add	t6,t6,s1
1c00c9f6:	8e56                	mv	t3,s5
1c00c9f8:	fa0f14e3          	bnez	t5,1c00c9a0 <mm_unroll_4x4+0x7f0>
1c00c9fc:	b1f5                	j	1c00c6e8 <mm_unroll_4x4+0x538>
1c00c9fe:	4685                	li	a3,1
1c00ca00:	dc36                	sw	a3,56(sp)
1c00ca02:	b951                	j	1c00c696 <mm_unroll_4x4+0x4e6>
1c00ca04:	4605                	li	a2,1
1c00ca06:	d632                	sw	a2,44(sp)
1c00ca08:	d416                	sw	t0,40(sp)
1c00ca0a:	bc21                	j	1c00c422 <mm_unroll_4x4+0x272>

1c00ca0c <mm_M_u2>:
1c00ca0c:	715d                	addi	sp,sp,-80
1c00ca0e:	da5a                	sw	s6,52(sp)
1c00ca10:	01052b03          	lw	s6,16(a0)
1c00ca14:	c0ce                	sw	s3,64(sp)
1c00ca16:	f14029f3          	csrr	s3,mhartid
1c00ca1a:	007b0713          	addi	a4,s6,7
1c00ca1e:	41f75793          	srai	a5,a4,0x1f
1c00ca22:	f837b7b3          	p.bclr	a5,a5,28,3
1c00ca26:	06e7a7db          	p.addn	a5,a5,a4,3
1c00ca2a:	f459b9b3          	p.bclr	s3,s3,26,5
1c00ca2e:	02f989b3          	mul	s3,s3,a5
1c00ca32:	4114                	lw	a3,0(a0)
1c00ca34:	d26a                	sw	s10,36(sp)
1c00ca36:	01452d03          	lw	s10,20(a0)
1c00ca3a:	c636                	sw	a3,12(sp)
1c00ca3c:	4154                	lw	a3,4(a0)
1c00ca3e:	4d18                	lw	a4,24(a0)
1c00ca40:	de52                	sw	s4,60(sp)
1c00ca42:	c836                	sw	a3,16(sp)
1c00ca44:	fc1d36b3          	p.bclr	a3,s10,30,1
1c00ca48:	97ce                	add	a5,a5,s3
1c00ca4a:	0567c7b3          	p.min	a5,a5,s6
1c00ca4e:	dc56                	sw	s5,56(sp)
1c00ca50:	c6a2                	sw	s0,76(sp)
1c00ca52:	c4a6                	sw	s1,72(sp)
1c00ca54:	c2ca                	sw	s2,68(sp)
1c00ca56:	d85e                	sw	s7,48(sp)
1c00ca58:	d662                	sw	s8,44(sp)
1c00ca5a:	d466                	sw	s9,40(sp)
1c00ca5c:	d06e                	sw	s11,32(sp)
1c00ca5e:	ca36                	sw	a3,20(sp)
1c00ca60:	c43e                	sw	a5,8(sp)
1c00ca62:	00852a83          	lw	s5,8(a0)
1c00ca66:	00c52a03          	lw	s4,12(a0)
1c00ca6a:	10070663          	beqz	a4,1c00cb76 <mm_M_u2+0x16a>
1c00ca6e:	0f405663          	blez	s4,1c00cb5a <mm_M_u2+0x14e>
1c00ca72:	c00d32b3          	p.bclr	t0,s10,0,0
1c00ca76:	0ef9d263          	ble	a5,s3,1c00cb5a <mm_M_u2+0x14e>
1c00ca7a:	4dc2                	lw	s11,16(sp)
1c00ca7c:	002d1393          	slli	t2,s10,0x2
1c00ca80:	4932                	lw	s2,12(sp)
1c00ca82:	42798db3          	p.mac	s11,s3,t2
1c00ca86:	00279b93          	slli	s7,a5,0x2
1c00ca8a:	002b1413          	slli	s0,s6,0x2
1c00ca8e:	9bd6                	add	s7,s7,s5
1c00ca90:	00490493          	addi	s1,s2,4
1c00ca94:	4c81                	li	s9,0
1c00ca96:	4c01                	li	s8,0
1c00ca98:	013c8f33          	add	t5,s9,s3
1c00ca9c:	0f0a                	slli	t5,t5,0x2
1c00ca9e:	9f56                	add	t5,t5,s5
1c00caa0:	41eb8e33          	sub	t3,s7,t5
1c00caa4:	1e71                	addi	t3,t3,-4
1c00caa6:	00000793          	li	a5,0
1c00caaa:	002e5e13          	srli	t3,t3,0x2
1c00caae:	8eee                	mv	t4,s11
1c00cab0:	0e05                	addi	t3,t3,1
1c00cab2:	0001                	nop
1c00cab4:	022e407b          	lp.setup	x0,t3,1c00caf8 <mm_M_u2+0xec>
1c00cab8:	02028e63          	beqz	t0,1c00caf4 <mm_M_u2+0xe8>
1c00cabc:	fff28693          	addi	a3,t0,-1
1c00cac0:	8285                	srli	a3,a3,0x1
1c00cac2:	00128713          	addi	a4,t0,1
1c00cac6:	4889                	li	a7,2
1c00cac8:	004e8613          	addi	a2,t4,4
1c00cacc:	85a6                	mv	a1,s1
1c00cace:	8576                	mv	a0,t4
1c00cad0:	884a                	mv	a6,s2
1c00cad2:	0685                	addi	a3,a3,1
1c00cad4:	1b176463          	bltu	a4,a7,1c00cc7c <mm_M_u2+0x270>
1c00cad8:	00c6c0fb          	lp.setup	x1,a3,1c00caf0 <mm_M_u2+0xe4>
1c00cadc:	00882f8b          	p.lw	t6,8(a6!)
1c00cae0:	0085230b          	p.lw	t1,8(a0!)
1c00cae4:	0085a88b          	p.lw	a7,8(a1!)
1c00cae8:	0086270b          	p.lw	a4,8(a2!)
1c00caec:	786ff7c3          	fmadd.s	a5,t6,t1,a5
1c00caf0:	78e8f7c3          	fmadd.s	a5,a7,a4,a5
1c00caf4:	00ff222b          	p.sw	a5,4(t5!)
1c00caf8:	9e9e                	add	t4,t4,t2
1c00cafa:	0c05                	addi	s8,s8,1
1c00cafc:	9cda                	add	s9,s9,s6
1c00cafe:	9ba2                	add	s7,s7,s0
1c00cb00:	991e                	add	s2,s2,t2
1c00cb02:	949e                	add	s1,s1,t2
1c00cb04:	f98a1ae3          	bne	s4,s8,1c00ca98 <mm_M_u2+0x8c>
1c00cb08:	47d2                	lw	a5,20(sp)
1c00cb0a:	cba1                	beqz	a5,1c00cb5a <mm_M_u2+0x14e>
1c00cb0c:	fffd0893          	addi	a7,s10,-1
1c00cb10:	433d08b3          	p.mac	a7,s10,s3
1c00cb14:	00299793          	slli	a5,s3,0x2
1c00cb18:	46a2                	lw	a3,8(sp)
1c00cb1a:	9abe                	add	s5,s5,a5
1c00cb1c:	4732                	lw	a4,12(sp)
1c00cb1e:	47c2                	lw	a5,16(sp)
1c00cb20:	ffc38813          	addi	a6,t2,-4
1c00cb24:	41368333          	sub	t1,a3,s3
1c00cb28:	983a                	add	a6,a6,a4
1c00cb2a:	030a                	slli	t1,t1,0x2
1c00cb2c:	088a                	slli	a7,a7,0x2
1c00cb2e:	98be                	add	a7,a7,a5
1c00cb30:	014a407b          	lp.setup	x0,s4,1c00cb58 <mm_M_u2+0x14c>
1c00cb34:	ffc30713          	addi	a4,t1,-4
1c00cb38:	8309                	srli	a4,a4,0x2
1c00cb3a:	8646                	mv	a2,a7
1c00cb3c:	86d6                	mv	a3,s5
1c00cb3e:	0705                	addi	a4,a4,1
1c00cb40:	009740fb          	lp.setup	x1,a4,1c00cb52 <mm_M_u2+0x146>
1c00cb44:	428c                	lw	a1,0(a3)
1c00cb46:	00082783          	lw	a5,0(a6)
1c00cb4a:	2076750b          	p.lw	a0,t2(a2!)
1c00cb4e:	58a7f7c3          	fmadd.s	a5,a5,a0,a1
1c00cb52:	00f6a22b          	p.sw	a5,4(a3!)
1c00cb56:	9aa2                	add	s5,s5,s0
1c00cb58:	981e                	add	a6,a6,t2
1c00cb5a:	4436                	lw	s0,76(sp)
1c00cb5c:	44a6                	lw	s1,72(sp)
1c00cb5e:	4916                	lw	s2,68(sp)
1c00cb60:	4986                	lw	s3,64(sp)
1c00cb62:	5a72                	lw	s4,60(sp)
1c00cb64:	5ae2                	lw	s5,56(sp)
1c00cb66:	5b52                	lw	s6,52(sp)
1c00cb68:	5bc2                	lw	s7,48(sp)
1c00cb6a:	5c32                	lw	s8,44(sp)
1c00cb6c:	5ca2                	lw	s9,40(sp)
1c00cb6e:	5d12                	lw	s10,36(sp)
1c00cb70:	5d82                	lw	s11,32(sp)
1c00cb72:	6161                	addi	sp,sp,80
1c00cb74:	8082                	ret
1c00cb76:	ff4052e3          	blez	s4,1c00cb5a <mm_M_u2+0x14e>
1c00cb7a:	873e                	mv	a4,a5
1c00cb7c:	c00d32b3          	p.bclr	t0,s10,0,0
1c00cb80:	fcf9dde3          	ble	a5,s3,1c00cb5a <mm_M_u2+0x14e>
1c00cb84:	00299793          	slli	a5,s3,0x2
1c00cb88:	cc3e                	sw	a5,24(sp)
1c00cb8a:	46c2                	lw	a3,16(sp)
1c00cb8c:	00271793          	slli	a5,a4,0x2
1c00cb90:	4932                	lw	s2,12(sp)
1c00cb92:	4762                	lw	a4,24(sp)
1c00cb94:	97b6                	add	a5,a5,a3
1c00cb96:	002d1393          	slli	t2,s10,0x2
1c00cb9a:	ce6a                	sw	s10,28(sp)
1c00cb9c:	00490493          	addi	s1,s2,4
1c00cba0:	00e68cb3          	add	s9,a3,a4
1c00cba4:	003b1d93          	slli	s11,s6,0x3
1c00cba8:	002b1413          	slli	s0,s6,0x2
1c00cbac:	4c01                	li	s8,0
1c00cbae:	4b81                	li	s7,0
1c00cbb0:	8d3e                	mv	s10,a5
1c00cbb2:	419d0eb3          	sub	t4,s10,s9
1c00cbb6:	013b8fb3          	add	t6,s7,s3
1c00cbba:	1ef1                	addi	t4,t4,-4
1c00cbbc:	0f8a                	slli	t6,t6,0x2
1c00cbbe:	002ede93          	srli	t4,t4,0x2
1c00cbc2:	9fd6                	add	t6,t6,s5
1c00cbc4:	8f66                	mv	t5,s9
1c00cbc6:	0e85                	addi	t4,t4,1
1c00cbc8:	026ec07b          	lp.setup	x0,t4,1c00cc14 <mm_M_u2+0x208>
1c00cbcc:	00000793          	li	a5,0
1c00cbd0:	04028063          	beqz	t0,1c00cc10 <mm_M_u2+0x204>
1c00cbd4:	fff28693          	addi	a3,t0,-1
1c00cbd8:	8285                	srli	a3,a3,0x1
1c00cbda:	00128713          	addi	a4,t0,1
1c00cbde:	4889                	li	a7,2
1c00cbe0:	01e405b3          	add	a1,s0,t5
1c00cbe4:	8626                	mv	a2,s1
1c00cbe6:	857a                	mv	a0,t5
1c00cbe8:	884a                	mv	a6,s2
1c00cbea:	00000793          	li	a5,0
1c00cbee:	0685                	addi	a3,a3,1
1c00cbf0:	09176463          	bltu	a4,a7,1c00cc78 <mm_M_u2+0x26c>
1c00cbf4:	00c6c0fb          	lp.setup	x1,a3,1c00cc0c <mm_M_u2+0x200>
1c00cbf8:	00882e0b          	p.lw	t3,8(a6!)
1c00cbfc:	21b5730b          	p.lw	t1,s11(a0!)
1c00cc00:	0086288b          	p.lw	a7,8(a2!)
1c00cc04:	21b5f70b          	p.lw	a4,s11(a1!)
1c00cc08:	786e77c3          	fmadd.s	a5,t3,t1,a5
1c00cc0c:	78e8f7c3          	fmadd.s	a5,a7,a4,a5
1c00cc10:	00ffa22b          	p.sw	a5,4(t6!)
1c00cc14:	0f11                	addi	t5,t5,4
1c00cc16:	0c05                	addi	s8,s8,1
1c00cc18:	9bda                	add	s7,s7,s6
1c00cc1a:	991e                	add	s2,s2,t2
1c00cc1c:	949e                	add	s1,s1,t2
1c00cc1e:	f98a1ae3          	bne	s4,s8,1c00cbb2 <mm_M_u2+0x1a6>
1c00cc22:	4752                	lw	a4,20(sp)
1c00cc24:	4d72                	lw	s10,28(sp)
1c00cc26:	db15                	beqz	a4,1c00cb5a <mm_M_u2+0x14e>
1c00cc28:	1d7d                	addi	s10,s10,-1
1c00cc2a:	88ce                	mv	a7,s3
1c00cc2c:	436d08b3          	p.mac	a7,s10,s6
1c00cc30:	46a2                	lw	a3,8(sp)
1c00cc32:	4762                	lw	a4,24(sp)
1c00cc34:	47b2                	lw	a5,12(sp)
1c00cc36:	41368333          	sub	t1,a3,s3
1c00cc3a:	46c2                	lw	a3,16(sp)
1c00cc3c:	ffc38813          	addi	a6,t2,-4
1c00cc40:	9aba                	add	s5,s5,a4
1c00cc42:	983e                	add	a6,a6,a5
1c00cc44:	030a                	slli	t1,t1,0x2
1c00cc46:	088a                	slli	a7,a7,0x2
1c00cc48:	98b6                	add	a7,a7,a3
1c00cc4a:	0001                	nop
1c00cc4c:	014a407b          	lp.setup	x0,s4,1c00cc74 <mm_M_u2+0x268>
1c00cc50:	ffc30713          	addi	a4,t1,-4
1c00cc54:	8309                	srli	a4,a4,0x2
1c00cc56:	8646                	mv	a2,a7
1c00cc58:	86d6                	mv	a3,s5
1c00cc5a:	0705                	addi	a4,a4,1
1c00cc5c:	009740fb          	lp.setup	x1,a4,1c00cc6e <mm_M_u2+0x262>
1c00cc60:	428c                	lw	a1,0(a3)
1c00cc62:	0046250b          	p.lw	a0,4(a2!)
1c00cc66:	00082783          	lw	a5,0(a6)
1c00cc6a:	58a7f7c3          	fmadd.s	a5,a5,a0,a1
1c00cc6e:	00f6a22b          	p.sw	a5,4(a3!)
1c00cc72:	9aa2                	add	s5,s5,s0
1c00cc74:	981e                	add	a6,a6,t2
1c00cc76:	b5d5                	j	1c00cb5a <mm_M_u2+0x14e>
1c00cc78:	4685                	li	a3,1
1c00cc7a:	bfad                	j	1c00cbf4 <mm_M_u2+0x1e8>
1c00cc7c:	4685                	li	a3,1
1c00cc7e:	bda9                	j	1c00cad8 <mm_M_u2+0xcc>

1c00cc80 <mm_M_unroll_2x1>:
1c00cc80:	7139                	addi	sp,sp,-64
1c00cc82:	d652                	sw	s4,44(sp)
1c00cc84:	01052a03          	lw	s4,16(a0)
1c00cc88:	d84e                	sw	s3,48(sp)
1c00cc8a:	f14029f3          	csrr	s3,mhartid
1c00cc8e:	007a0693          	addi	a3,s4,7
1c00cc92:	41f6d793          	srai	a5,a3,0x1f
1c00cc96:	f837b7b3          	p.bclr	a5,a5,28,3
1c00cc9a:	06d7a7db          	p.addn	a5,a5,a3,3
1c00cc9e:	f459b9b3          	p.bclr	s3,s3,26,5
1c00cca2:	02f989b3          	mul	s3,s3,a5
1c00cca6:	d25a                	sw	s6,36(sp)
1c00cca8:	00c52b03          	lw	s6,12(a0)
1c00ccac:	de22                	sw	s0,60(sp)
1c00ccae:	dc26                	sw	s1,56(sp)
1c00ccb0:	fc1b3633          	p.bclr	a2,s6,30,1
1c00ccb4:	da4a                	sw	s2,52(sp)
1c00ccb6:	d456                	sw	s5,40(sp)
1c00ccb8:	d05e                	sw	s7,32(sp)
1c00ccba:	ce62                	sw	s8,28(sp)
1c00ccbc:	97ce                	add	a5,a5,s3
1c00ccbe:	0547c7b3          	p.min	a5,a5,s4
1c00ccc2:	cc66                	sw	s9,24(sp)
1c00ccc4:	ca6a                	sw	s10,20(sp)
1c00ccc6:	c86e                	sw	s11,16(sp)
1c00ccc8:	4685                	li	a3,1
1c00ccca:	c632                	sw	a2,12(sp)
1c00cccc:	c03e                	sw	a5,0(sp)
1c00ccce:	2766db63          	ble	s6,a3,1c00cf44 <mm_M_unroll_2x1+0x2c4>
1c00ccd2:	4154                	lw	a3,4(a0)
1c00ccd4:	4510                	lw	a2,8(a0)
1c00ccd6:	4d1c                	lw	a5,24(a0)
1c00ccd8:	c236                	sw	a3,4(sp)
1c00ccda:	c432                	sw	a2,8(sp)
1c00ccdc:	00052b83          	lw	s7,0(a0)
1c00cce0:	01452f03          	lw	t5,20(a0)
1c00cce4:	12079963          	bnez	a5,1c00ce16 <mm_M_unroll_2x1+0x196>
1c00cce8:	4782                	lw	a5,0(sp)
1c00ccea:	10f9d863          	ble	a5,s3,1c00cdfa <mm_M_unroll_2x1+0x17a>
1c00ccee:	c00b3ab3          	p.bclr	s5,s6,0,0
1c00ccf2:	080a8863          	beqz	s5,1c00cd82 <mm_M_unroll_2x1+0x102>
1c00ccf6:	00299413          	slli	s0,s3,0x2
1c00ccfa:	00279c13          	slli	s8,a5,0x2
1c00ccfe:	00860933          	add	s2,a2,s0
1c00cd02:	9c36                	add	s8,s8,a3
1c00cd04:	9436                	add	s0,s0,a3
1c00cd06:	003a1d13          	slli	s10,s4,0x3
1c00cd0a:	002a1c93          	slli	s9,s4,0x2
1c00cd0e:	003f1493          	slli	s1,t5,0x3
1c00cd12:	002f1393          	slli	t2,t5,0x2
1c00cd16:	fffa8313          	addi	t1,s5,-1
1c00cd1a:	00135313          	srli	t1,t1,0x1
1c00cd1e:	001a8793          	addi	a5,s5,1
1c00cd22:	4709                	li	a4,2
1c00cd24:	012c8e33          	add	t3,s9,s2
1c00cd28:	88de                	mv	a7,s7
1c00cd2a:	8eca                	mv	t4,s2
1c00cd2c:	0305                	addi	t1,t1,1
1c00cd2e:	22e7ea63          	bltu	a5,a4,1c00cf62 <mm_M_unroll_2x1+0x2e2>
1c00cd32:	0001                	nop
1c00cd34:	0223407b          	lp.setup	x0,t1,1c00cd78 <mm_M_unroll_2x1+0xf8>
1c00cd38:	00000613          	li	a2,0
1c00cd3c:	86b2                	mv	a3,a2
1c00cd3e:	03e05963          	blez	t5,1c00cd70 <mm_M_unroll_2x1+0xf0>
1c00cd42:	00000613          	li	a2,0
1c00cd46:	ffc38793          	addi	a5,t2,-4
1c00cd4a:	8389                	srli	a5,a5,0x2
1c00cd4c:	86b2                	mv	a3,a2
1c00cd4e:	8846                	mv	a6,a7
1c00cd50:	8522                	mv	a0,s0
1c00cd52:	011385b3          	add	a1,t2,a7
1c00cd56:	0785                	addi	a5,a5,1
1c00cd58:	00a7c0fb          	lp.setup	x1,a5,1c00cd6c <mm_M_unroll_2x1+0xec>
1c00cd5c:	2195770b          	p.lw	a4,s9(a0!)
1c00cd60:	0048228b          	p.lw	t0,4(a6!)
1c00cd64:	0045af8b          	p.lw	t6,4(a1!)
1c00cd68:	68e2f6c3          	fmadd.s	a3,t0,a4,a3
1c00cd6c:	60eff643          	fmadd.s	a2,t6,a4,a2
1c00cd70:	00deed2b          	p.sw	a3,s10(t4!)
1c00cd74:	00ce6d2b          	p.sw	a2,s10(t3!)
1c00cd78:	98a6                	add	a7,a7,s1
1c00cd7a:	0411                	addi	s0,s0,4
1c00cd7c:	0911                	addi	s2,s2,4
1c00cd7e:	f88c1ce3          	bne	s8,s0,1c00cd16 <mm_M_unroll_2x1+0x96>
1c00cd82:	46b2                	lw	a3,12(sp)
1c00cd84:	cabd                	beqz	a3,1c00cdfa <mm_M_unroll_2x1+0x17a>
1c00cd86:	1b7d                	addi	s6,s6,-1
1c00cd88:	036f07b3          	mul	a5,t5,s6
1c00cd8c:	834e                	mv	t1,s3
1c00cd8e:	4702                	lw	a4,0(sp)
1c00cd90:	00299893          	slli	a7,s3,0x2
1c00cd94:	4612                	lw	a2,4(sp)
1c00cd96:	00271813          	slli	a6,a4,0x2
1c00cd9a:	41180833          	sub	a6,a6,a7
1c00cd9e:	46a2                	lw	a3,8(sp)
1c00cda0:	1871                	addi	a6,a6,-4
1c00cda2:	00285813          	srli	a6,a6,0x2
1c00cda6:	436a0333          	p.mac	t1,s4,s6
1c00cdaa:	00ff0733          	add	a4,t5,a5
1c00cdae:	070a                	slli	a4,a4,0x2
1c00cdb0:	078a                	slli	a5,a5,0x2
1c00cdb2:	00fb8e33          	add	t3,s7,a5
1c00cdb6:	98b2                	add	a7,a7,a2
1c00cdb8:	002a1e93          	slli	t4,s4,0x2
1c00cdbc:	0805                	addi	a6,a6,1
1c00cdbe:	9bba                	add	s7,s7,a4
1c00cdc0:	030a                	slli	t1,t1,0x2
1c00cdc2:	9336                	add	t1,t1,a3
1c00cdc4:	01a8407b          	lp.setup	x0,a6,1c00cdf8 <mm_M_unroll_2x1+0x178>
1c00cdc8:	00000713          	li	a4,0
1c00cdcc:	03e05463          	blez	t5,1c00cdf4 <mm_M_unroll_2x1+0x174>
1c00cdd0:	41cb87b3          	sub	a5,s7,t3
1c00cdd4:	17f1                	addi	a5,a5,-4
1c00cdd6:	00000713          	li	a4,0
1c00cdda:	8389                	srli	a5,a5,0x2
1c00cddc:	8646                	mv	a2,a7
1c00cdde:	86f2                	mv	a3,t3
1c00cde0:	0785                	addi	a5,a5,1
1c00cde2:	0001                	nop
1c00cde4:	0067c0fb          	lp.setup	x1,a5,1c00cdf0 <mm_M_unroll_2x1+0x170>
1c00cde8:	0046a50b          	p.lw	a0,4(a3!)
1c00cdec:	21d6758b          	p.lw	a1,t4(a2!)
1c00cdf0:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00cdf4:	00e3222b          	p.sw	a4,4(t1!)
1c00cdf8:	0891                	addi	a7,a7,4
1c00cdfa:	5472                	lw	s0,60(sp)
1c00cdfc:	54e2                	lw	s1,56(sp)
1c00cdfe:	5952                	lw	s2,52(sp)
1c00ce00:	59c2                	lw	s3,48(sp)
1c00ce02:	5a32                	lw	s4,44(sp)
1c00ce04:	5aa2                	lw	s5,40(sp)
1c00ce06:	5b12                	lw	s6,36(sp)
1c00ce08:	5b82                	lw	s7,32(sp)
1c00ce0a:	4c72                	lw	s8,28(sp)
1c00ce0c:	4ce2                	lw	s9,24(sp)
1c00ce0e:	4d52                	lw	s10,20(sp)
1c00ce10:	4dc2                	lw	s11,16(sp)
1c00ce12:	6121                	addi	sp,sp,64
1c00ce14:	8082                	ret
1c00ce16:	4682                	lw	a3,0(sp)
1c00ce18:	fed9d1e3          	ble	a3,s3,1c00cdfa <mm_M_unroll_2x1+0x17a>
1c00ce1c:	c00b3c33          	p.bclr	s8,s6,0,0
1c00ce20:	0a0c0263          	beqz	s8,1c00cec4 <mm_M_unroll_2x1+0x244>
1c00ce24:	847a                	mv	s0,t5
1c00ce26:	4392                	lw	t2,4(sp)
1c00ce28:	43e98433          	p.mac	s0,s3,t5
1c00ce2c:	002f1493          	slli	s1,t5,0x2
1c00ce30:	4712                	lw	a4,4(sp)
1c00ce32:	00299a93          	slli	s5,s3,0x2
1c00ce36:	00269c93          	slli	s9,a3,0x2
1c00ce3a:	9ab2                	add	s5,s5,a2
1c00ce3c:	9cb2                	add	s9,s9,a2
1c00ce3e:	003a1d93          	slli	s11,s4,0x3
1c00ce42:	002a1d13          	slli	s10,s4,0x2
1c00ce46:	003f1913          	slli	s2,t5,0x3
1c00ce4a:	429983b3          	p.mac	t2,s3,s1
1c00ce4e:	040a                	slli	s0,s0,0x2
1c00ce50:	943a                	add	s0,s0,a4
1c00ce52:	fffc0893          	addi	a7,s8,-1
1c00ce56:	0018d893          	srli	a7,a7,0x1
1c00ce5a:	001c0793          	addi	a5,s8,1
1c00ce5e:	4709                	li	a4,2
1c00ce60:	015d0333          	add	t1,s10,s5
1c00ce64:	885e                	mv	a6,s7
1c00ce66:	8e56                	mv	t3,s5
1c00ce68:	0885                	addi	a7,a7,1
1c00ce6a:	0ee7ee63          	bltu	a5,a4,1c00cf66 <mm_M_unroll_2x1+0x2e6>
1c00ce6e:	0001                	nop
1c00ce70:	0248c07b          	lp.setup	x0,a7,1c00ceb8 <mm_M_unroll_2x1+0x238>
1c00ce74:	00000613          	li	a2,0
1c00ce78:	86b2                	mv	a3,a2
1c00ce7a:	03e05b63          	blez	t5,1c00ceb0 <mm_M_unroll_2x1+0x230>
1c00ce7e:	00000613          	li	a2,0
1c00ce82:	407407b3          	sub	a5,s0,t2
1c00ce86:	17f1                	addi	a5,a5,-4
1c00ce88:	86b2                	mv	a3,a2
1c00ce8a:	8389                	srli	a5,a5,0x2
1c00ce8c:	01048eb3          	add	t4,s1,a6
1c00ce90:	8542                	mv	a0,a6
1c00ce92:	859e                	mv	a1,t2
1c00ce94:	0785                	addi	a5,a5,1
1c00ce96:	0001                	nop
1c00ce98:	00a7c0fb          	lp.setup	x1,a5,1c00ceac <mm_M_unroll_2x1+0x22c>
1c00ce9c:	0045a70b          	p.lw	a4,4(a1!)
1c00cea0:	0045228b          	p.lw	t0,4(a0!)
1c00cea4:	004eaf8b          	p.lw	t6,4(t4!)
1c00cea8:	68e2f6c3          	fmadd.s	a3,t0,a4,a3
1c00ceac:	60eff643          	fmadd.s	a2,t6,a4,a2
1c00ceb0:	00de6dab          	p.sw	a3,s11(t3!)
1c00ceb4:	00c36dab          	p.sw	a2,s11(t1!)
1c00ceb8:	984a                	add	a6,a6,s2
1c00ceba:	0a91                	addi	s5,s5,4
1c00cebc:	93a6                	add	t2,t2,s1
1c00cebe:	9426                	add	s0,s0,s1
1c00cec0:	f95c99e3          	bne	s9,s5,1c00ce52 <mm_M_unroll_2x1+0x1d2>
1c00cec4:	4732                	lw	a4,12(sp)
1c00cec6:	db15                	beqz	a4,1c00cdfa <mm_M_unroll_2x1+0x17a>
1c00cec8:	fffb0793          	addi	a5,s6,-1
1c00cecc:	02fa0a33          	mul	s4,s4,a5
1c00ced0:	4892                	lw	a7,4(sp)
1c00ced2:	002f1e13          	slli	t3,t5,0x2
1c00ced6:	4702                	lw	a4,0(sp)
1c00ced8:	4622                	lw	a2,8(sp)
1c00ceda:	02ff07b3          	mul	a5,t5,a5
1c00cede:	01470833          	add	a6,a4,s4
1c00cee2:	013a0eb3          	add	t4,s4,s3
1c00cee6:	0e8a                	slli	t4,t4,0x2
1c00cee8:	080a                	slli	a6,a6,0x2
1c00ceea:	41d80833          	sub	a6,a6,t4
1c00ceee:	1871                	addi	a6,a6,-4
1c00cef0:	00285813          	srli	a6,a6,0x2
1c00cef4:	9eb2                	add	t4,t4,a2
1c00cef6:	0805                	addi	a6,a6,1
1c00cef8:	43c988b3          	p.mac	a7,s3,t3
1c00cefc:	01e78733          	add	a4,a5,t5
1c00cf00:	070a                	slli	a4,a4,0x2
1c00cf02:	078a                	slli	a5,a5,0x2
1c00cf04:	00fb8333          	add	t1,s7,a5
1c00cf08:	9bba                	add	s7,s7,a4
1c00cf0a:	0001                	nop
1c00cf0c:	01a8407b          	lp.setup	x0,a6,1c00cf40 <mm_M_unroll_2x1+0x2c0>
1c00cf10:	00000713          	li	a4,0
1c00cf14:	03e05463          	blez	t5,1c00cf3c <mm_M_unroll_2x1+0x2bc>
1c00cf18:	406b87b3          	sub	a5,s7,t1
1c00cf1c:	17f1                	addi	a5,a5,-4
1c00cf1e:	00000713          	li	a4,0
1c00cf22:	8389                	srli	a5,a5,0x2
1c00cf24:	8646                	mv	a2,a7
1c00cf26:	869a                	mv	a3,t1
1c00cf28:	0785                	addi	a5,a5,1
1c00cf2a:	0001                	nop
1c00cf2c:	0067c0fb          	lp.setup	x1,a5,1c00cf38 <mm_M_unroll_2x1+0x2b8>
1c00cf30:	0046a50b          	p.lw	a0,4(a3!)
1c00cf34:	0046258b          	p.lw	a1,4(a2!)
1c00cf38:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00cf3c:	00eea22b          	p.sw	a4,4(t4!)
1c00cf40:	98f2                	add	a7,a7,t3
1c00cf42:	bd65                	j	1c00cdfa <mm_M_unroll_2x1+0x17a>
1c00cf44:	5472                	lw	s0,60(sp)
1c00cf46:	54e2                	lw	s1,56(sp)
1c00cf48:	5952                	lw	s2,52(sp)
1c00cf4a:	59c2                	lw	s3,48(sp)
1c00cf4c:	5a32                	lw	s4,44(sp)
1c00cf4e:	5aa2                	lw	s5,40(sp)
1c00cf50:	5b12                	lw	s6,36(sp)
1c00cf52:	5b82                	lw	s7,32(sp)
1c00cf54:	4c72                	lw	s8,28(sp)
1c00cf56:	4ce2                	lw	s9,24(sp)
1c00cf58:	4d52                	lw	s10,20(sp)
1c00cf5a:	4dc2                	lw	s11,16(sp)
1c00cf5c:	6121                	addi	sp,sp,64
1c00cf5e:	93afc06f          	j	1c009098 <mm_M>
1c00cf62:	4305                	li	t1,1
1c00cf64:	b3f9                	j	1c00cd32 <mm_M_unroll_2x1+0xb2>
1c00cf66:	4885                	li	a7,1
1c00cf68:	b719                	j	1c00ce6e <mm_M_unroll_2x1+0x1ee>
	...

1c00cf6c <mm_M_unroll_4x1>:
1c00cf6c:	4550                	lw	a2,12(a0)
1c00cf6e:	7159                	addi	sp,sp,-112
1c00cf70:	d6a2                	sw	s0,108(sp)
1c00cf72:	d4a6                	sw	s1,104(sp)
1c00cf74:	d2ca                	sw	s2,100(sp)
1c00cf76:	d0ce                	sw	s3,96(sp)
1c00cf78:	ced2                	sw	s4,92(sp)
1c00cf7a:	ccd6                	sw	s5,88(sp)
1c00cf7c:	cada                	sw	s6,84(sp)
1c00cf7e:	c8de                	sw	s7,80(sp)
1c00cf80:	c6e2                	sw	s8,76(sp)
1c00cf82:	c4e6                	sw	s9,72(sp)
1c00cf84:	c2ea                	sw	s10,68(sp)
1c00cf86:	c0ee                	sw	s11,64(sp)
1c00cf88:	cc32                	sw	a2,24(sp)
1c00cf8a:	468d                	li	a3,3
1c00cf8c:	f14027f3          	csrr	a5,mhartid
1c00cf90:	3ac6d763          	ble	a2,a3,1c00d33e <mm_M_unroll_4x1+0x3d2>
1c00cf94:	4914                	lw	a3,16(a0)
1c00cf96:	4762                	lw	a4,24(sp)
1c00cf98:	4d0c                	lw	a1,24(a0)
1c00cf9a:	d036                	sw	a3,32(sp)
1c00cf9c:	f457b6b3          	p.bclr	a3,a5,26,5
1c00cfa0:	5782                	lw	a5,32(sp)
1c00cfa2:	fa273733          	p.bclr	a4,a4,29,2
1c00cfa6:	dc3a                	sw	a4,56(sp)
1c00cfa8:	00778613          	addi	a2,a5,7
1c00cfac:	41f65793          	srai	a5,a2,0x1f
1c00cfb0:	f837b7b3          	p.bclr	a5,a5,28,3
1c00cfb4:	06c7a7db          	p.addn	a5,a5,a2,3
1c00cfb8:	02f686b3          	mul	a3,a3,a5
1c00cfbc:	4110                	lw	a2,0(a0)
1c00cfbe:	01452903          	lw	s2,20(a0)
1c00cfc2:	da32                	sw	a2,52(sp)
1c00cfc4:	4510                	lw	a2,8(a0)
1c00cfc6:	d432                	sw	a2,40(sp)
1c00cfc8:	ca36                	sw	a3,20(sp)
1c00cfca:	4154                	lw	a3,4(a0)
1c00cfcc:	4652                	lw	a2,20(sp)
1c00cfce:	d836                	sw	a3,48(sp)
1c00cfd0:	46e2                	lw	a3,24(sp)
1c00cfd2:	97b2                	add	a5,a5,a2
1c00cfd4:	de32                	sw	a2,60(sp)
1c00cfd6:	c206b6b3          	p.bclr	a3,a3,1,0
1c00cfda:	c636                	sw	a3,12(sp)
1c00cfdc:	5682                	lw	a3,32(sp)
1c00cfde:	04d7c7b3          	p.min	a5,a5,a3
1c00cfe2:	d23e                	sw	a5,36(sp)
1c00cfe4:	18059763          	bnez	a1,1c00d172 <mm_M_unroll_4x1+0x206>
1c00cfe8:	873e                	mv	a4,a5
1c00cfea:	16f65663          	ble	a5,a2,1c00d156 <mm_M_unroll_4x1+0x1ea>
1c00cfee:	47b2                	lw	a5,12(sp)
1c00cff0:	cff1                	beqz	a5,1c00d0cc <mm_M_unroll_4x1+0x160>
1c00cff2:	00261b93          	slli	s7,a2,0x2
1c00cff6:	5622                	lw	a2,40(sp)
1c00cff8:	56c2                	lw	a3,48(sp)
1c00cffa:	070a                	slli	a4,a4,0x2
1c00cffc:	965e                	add	a2,a2,s7
1c00cffe:	c432                	sw	a2,8(sp)
1c00d000:	5622                	lw	a2,40(sp)
1c00d002:	9bb6                	add	s7,s7,a3
1c00d004:	5682                	lw	a3,32(sp)
1c00d006:	9732                	add	a4,a4,a2
1c00d008:	c83a                	sw	a4,16(sp)
1c00d00a:	5752                	lw	a4,52(sp)
1c00d00c:	412007b3          	neg	a5,s2
1c00d010:	00291c93          	slli	s9,s2,0x2
1c00d014:	00279613          	slli	a2,a5,0x2
1c00d018:	9766                	add	a4,a4,s9
1c00d01a:	078e                	slli	a5,a5,0x3
1c00d01c:	00469d93          	slli	s11,a3,0x4
1c00d020:	00269d13          	slli	s10,a3,0x2
1c00d024:	00491c13          	slli	s8,s2,0x4
1c00d028:	ce3a                	sw	a4,28(sp)
1c00d02a:	c032                	sw	a2,0(sp)
1c00d02c:	c23e                	sw	a5,4(sp)
1c00d02e:	4722                	lw	a4,8(sp)
1c00d030:	47b2                	lw	a5,12(sp)
1c00d032:	4472                	lw	s0,28(sp)
1c00d034:	00ed0a33          	add	s4,s10,a4
1c00d038:	fff78493          	addi	s1,a5,-1
1c00d03c:	01aa09b3          	add	s3,s4,s10
1c00d040:	8089                	srli	s1,s1,0x2
1c00d042:	078d                	addi	a5,a5,3
1c00d044:	4711                	li	a4,4
1c00d046:	01a98ab3          	add	s5,s3,s10
1c00d04a:	4b22                	lw	s6,8(sp)
1c00d04c:	0485                	addi	s1,s1,1
1c00d04e:	30e7e663          	bltu	a5,a4,1c00d35a <mm_M_unroll_4x1+0x3ee>
1c00d052:	2b205263          	blez	s2,1c00d2f6 <mm_M_unroll_4x1+0x38a>
1c00d056:	4702                	lw	a4,0(sp)
1c00d058:	00000813          	li	a6,0
1c00d05c:	4792                	lw	a5,4(sp)
1c00d05e:	008706b3          	add	a3,a4,s0
1c00d062:	40d40733          	sub	a4,s0,a3
1c00d066:	1771                	addi	a4,a4,-4
1c00d068:	8542                	mv	a0,a6
1c00d06a:	85c2                	mv	a1,a6
1c00d06c:	8642                	mv	a2,a6
1c00d06e:	40f688b3          	sub	a7,a3,a5
1c00d072:	8309                	srli	a4,a4,0x2
1c00d074:	01988eb3          	add	t4,a7,s9
1c00d078:	8e22                	mv	t3,s0
1c00d07a:	835e                	mv	t1,s7
1c00d07c:	0705                	addi	a4,a4,1
1c00d07e:	0001                	nop
1c00d080:	012740fb          	lp.setup	x1,a4,1c00d0a4 <mm_M_unroll_4x1+0x138>
1c00d084:	21a3778b          	p.lw	a5,s10(t1!)
1c00d088:	0046a38b          	p.lw	t2,4(a3!)
1c00d08c:	004e228b          	p.lw	t0,4(t3!)
1c00d090:	0048af8b          	p.lw	t6,4(a7!)
1c00d094:	004eaf0b          	p.lw	t5,4(t4!)
1c00d098:	60f3f643          	fmadd.s	a2,t2,a5,a2
1c00d09c:	58f2f5c3          	fmadd.s	a1,t0,a5,a1
1c00d0a0:	50fff543          	fmadd.s	a0,t6,a5,a0
1c00d0a4:	80ff7843          	fmadd.s	a6,t5,a5,a6
1c00d0a8:	00cb6dab          	p.sw	a2,s11(s6!)
1c00d0ac:	00ba6dab          	p.sw	a1,s11(s4!)
1c00d0b0:	00a9edab          	p.sw	a0,s11(s3!)
1c00d0b4:	010aedab          	p.sw	a6,s11(s5!)
1c00d0b8:	9462                	add	s0,s0,s8
1c00d0ba:	14fd                	addi	s1,s1,-1
1c00d0bc:	f8d9                	bnez	s1,1c00d052 <mm_M_unroll_4x1+0xe6>
1c00d0be:	4622                	lw	a2,8(sp)
1c00d0c0:	46c2                	lw	a3,16(sp)
1c00d0c2:	0b91                	addi	s7,s7,4
1c00d0c4:	0611                	addi	a2,a2,4
1c00d0c6:	c432                	sw	a2,8(sp)
1c00d0c8:	f6c693e3          	bne	a3,a2,1c00d02e <mm_M_unroll_4x1+0xc2>
1c00d0cc:	5762                	lw	a4,56(sp)
1c00d0ce:	c741                	beqz	a4,1c00d156 <mm_M_unroll_4x1+0x1ea>
1c00d0d0:	47e2                	lw	a5,24(sp)
1c00d0d2:	4632                	lw	a2,12(sp)
1c00d0d4:	08f65163          	ble	a5,a2,1c00d156 <mm_M_unroll_4x1+0x1ea>
1c00d0d8:	4752                	lw	a4,20(sp)
1c00d0da:	5682                	lw	a3,32(sp)
1c00d0dc:	4f52                	lw	t5,20(sp)
1c00d0de:	00271e13          	slli	t3,a4,0x2
1c00d0e2:	52d2                	lw	t0,52(sp)
1c00d0e4:	4732                	lw	a4,12(sp)
1c00d0e6:	42c68f33          	p.mac	t5,a3,a2
1c00d0ea:	00291e93          	slli	t4,s2,0x2
1c00d0ee:	843e                	mv	s0,a5
1c00d0f0:	5792                	lw	a5,36(sp)
1c00d0f2:	5642                	lw	a2,48(sp)
1c00d0f4:	00269393          	slli	t2,a3,0x2
1c00d0f8:	56a2                	lw	a3,40(sp)
1c00d0fa:	00279f93          	slli	t6,a5,0x2
1c00d0fe:	9e32                	add	t3,t3,a2
1c00d100:	9fb2                	add	t6,t6,a2
1c00d102:	43d702b3          	p.mac	t0,a4,t4
1c00d106:	0f0a                	slli	t5,t5,0x2
1c00d108:	9f36                	add	t5,t5,a3
1c00d10a:	84ba                	mv	s1,a4
1c00d10c:	8716                	mv	a4,t0
1c00d10e:	837a                	mv	t1,t5
1c00d110:	40940833          	sub	a6,s0,s1
1c00d114:	01c8407b          	lp.setup	x0,a6,1c00d14c <mm_M_unroll_4x1+0x1e0>
1c00d118:	00000693          	li	a3,0
1c00d11c:	00ee88b3          	add	a7,t4,a4
1c00d120:	03205463          	blez	s2,1c00d148 <mm_M_unroll_4x1+0x1dc>
1c00d124:	00ee88b3          	add	a7,t4,a4
1c00d128:	40e887b3          	sub	a5,a7,a4
1c00d12c:	17f1                	addi	a5,a5,-4
1c00d12e:	00000693          	li	a3,0
1c00d132:	8389                	srli	a5,a5,0x2
1c00d134:	8672                	mv	a2,t3
1c00d136:	0785                	addi	a5,a5,1
1c00d138:	0067c0fb          	lp.setup	x1,a5,1c00d144 <mm_M_unroll_4x1+0x1d8>
1c00d13c:	0047250b          	p.lw	a0,4(a4!)
1c00d140:	2076758b          	p.lw	a1,t2(a2!)
1c00d144:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00d148:	00d363ab          	p.sw	a3,t2(t1!)
1c00d14c:	8746                	mv	a4,a7
1c00d14e:	0e11                	addi	t3,t3,4
1c00d150:	0f11                	addi	t5,t5,4
1c00d152:	fbcf9de3          	bne	t6,t3,1c00d10c <mm_M_unroll_4x1+0x1a0>
1c00d156:	5436                	lw	s0,108(sp)
1c00d158:	54a6                	lw	s1,104(sp)
1c00d15a:	5916                	lw	s2,100(sp)
1c00d15c:	5986                	lw	s3,96(sp)
1c00d15e:	4a76                	lw	s4,92(sp)
1c00d160:	4ae6                	lw	s5,88(sp)
1c00d162:	4b56                	lw	s6,84(sp)
1c00d164:	4bc6                	lw	s7,80(sp)
1c00d166:	4c36                	lw	s8,76(sp)
1c00d168:	4ca6                	lw	s9,72(sp)
1c00d16a:	4d16                	lw	s10,68(sp)
1c00d16c:	4d86                	lw	s11,64(sp)
1c00d16e:	6165                	addi	sp,sp,112
1c00d170:	8082                	ret
1c00d172:	fef652e3          	ble	a5,a2,1c00d156 <mm_M_unroll_4x1+0x1ea>
1c00d176:	4732                	lw	a4,12(sp)
1c00d178:	cb7d                	beqz	a4,1c00d26e <mm_M_unroll_4x1+0x302>
1c00d17a:	00261693          	slli	a3,a2,0x2
1c00d17e:	5612                	lw	a2,36(sp)
1c00d180:	8c4a                	mv	s8,s2
1c00d182:	5bc2                	lw	s7,48(sp)
1c00d184:	00261713          	slli	a4,a2,0x2
1c00d188:	5622                	lw	a2,40(sp)
1c00d18a:	00291c93          	slli	s9,s2,0x2
1c00d18e:	412007b3          	neg	a5,s2
1c00d192:	96b2                	add	a3,a3,a2
1c00d194:	9732                	add	a4,a4,a2
1c00d196:	4652                	lw	a2,20(sp)
1c00d198:	c436                	sw	a3,8(sp)
1c00d19a:	5682                	lw	a3,32(sp)
1c00d19c:	43260c33          	p.mac	s8,a2,s2
1c00d1a0:	ce3a                	sw	a4,28(sp)
1c00d1a2:	00269713          	slli	a4,a3,0x2
1c00d1a6:	c83a                	sw	a4,16(sp)
1c00d1a8:	5752                	lw	a4,52(sp)
1c00d1aa:	00469d93          	slli	s11,a3,0x4
1c00d1ae:	56c2                	lw	a3,48(sp)
1c00d1b0:	9766                	add	a4,a4,s9
1c00d1b2:	00491d13          	slli	s10,s2,0x4
1c00d1b6:	d63a                	sw	a4,44(sp)
1c00d1b8:	0c0a                	slli	s8,s8,0x2
1c00d1ba:	43960bb3          	p.mac	s7,a2,s9
1c00d1be:	00279613          	slli	a2,a5,0x2
1c00d1c2:	078e                	slli	a5,a5,0x3
1c00d1c4:	9c36                	add	s8,s8,a3
1c00d1c6:	c032                	sw	a2,0(sp)
1c00d1c8:	c23e                	sw	a5,4(sp)
1c00d1ca:	4742                	lw	a4,16(sp)
1c00d1cc:	47a2                	lw	a5,8(sp)
1c00d1ce:	4632                	lw	a2,12(sp)
1c00d1d0:	46c2                	lw	a3,16(sp)
1c00d1d2:	00f70a33          	add	s4,a4,a5
1c00d1d6:	fff60413          	addi	s0,a2,-1
1c00d1da:	00ea09b3          	add	s3,s4,a4
1c00d1de:	8009                	srli	s0,s0,0x2
1c00d1e0:	00360793          	addi	a5,a2,3
1c00d1e4:	4711                	li	a4,4
1c00d1e6:	00d98ab3          	add	s5,s3,a3
1c00d1ea:	54b2                	lw	s1,44(sp)
1c00d1ec:	4b22                	lw	s6,8(sp)
1c00d1ee:	0405                	addi	s0,s0,1
1c00d1f0:	16e7e763          	bltu	a5,a4,1c00d35e <mm_M_unroll_4x1+0x3f2>
1c00d1f4:	13205363          	blez	s2,1c00d31a <mm_M_unroll_4x1+0x3ae>
1c00d1f8:	00000513          	li	a0,0
1c00d1fc:	4602                	lw	a2,0(sp)
1c00d1fe:	4792                	lw	a5,4(sp)
1c00d200:	417c0733          	sub	a4,s8,s7
1c00d204:	009608b3          	add	a7,a2,s1
1c00d208:	1771                	addi	a4,a4,-4
1c00d20a:	85aa                	mv	a1,a0
1c00d20c:	862a                	mv	a2,a0
1c00d20e:	86aa                	mv	a3,a0
1c00d210:	40f88833          	sub	a6,a7,a5
1c00d214:	8309                	srli	a4,a4,0x2
1c00d216:	010c8eb3          	add	t4,s9,a6
1c00d21a:	8e26                	mv	t3,s1
1c00d21c:	835e                	mv	t1,s7
1c00d21e:	0705                	addi	a4,a4,1
1c00d220:	012740fb          	lp.setup	x1,a4,1c00d244 <mm_M_unroll_4x1+0x2d8>
1c00d224:	0043278b          	p.lw	a5,4(t1!)
1c00d228:	0048a38b          	p.lw	t2,4(a7!)
1c00d22c:	004e228b          	p.lw	t0,4(t3!)
1c00d230:	00482f8b          	p.lw	t6,4(a6!)
1c00d234:	004eaf0b          	p.lw	t5,4(t4!)
1c00d238:	68f3f6c3          	fmadd.s	a3,t2,a5,a3
1c00d23c:	60f2f643          	fmadd.s	a2,t0,a5,a2
1c00d240:	58fff5c3          	fmadd.s	a1,t6,a5,a1
1c00d244:	50ff7543          	fmadd.s	a0,t5,a5,a0
1c00d248:	00db6dab          	p.sw	a3,s11(s6!)
1c00d24c:	00ca6dab          	p.sw	a2,s11(s4!)
1c00d250:	00b9edab          	p.sw	a1,s11(s3!)
1c00d254:	00aaedab          	p.sw	a0,s11(s5!)
1c00d258:	94ea                	add	s1,s1,s10
1c00d25a:	147d                	addi	s0,s0,-1
1c00d25c:	fc41                	bnez	s0,1c00d1f4 <mm_M_unroll_4x1+0x288>
1c00d25e:	4622                	lw	a2,8(sp)
1c00d260:	46f2                	lw	a3,28(sp)
1c00d262:	9be6                	add	s7,s7,s9
1c00d264:	0611                	addi	a2,a2,4
1c00d266:	c432                	sw	a2,8(sp)
1c00d268:	9c66                	add	s8,s8,s9
1c00d26a:	f6c690e3          	bne	a3,a2,1c00d1ca <mm_M_unroll_4x1+0x25e>
1c00d26e:	57e2                	lw	a5,56(sp)
1c00d270:	ee0783e3          	beqz	a5,1c00d156 <mm_M_unroll_4x1+0x1ea>
1c00d274:	4662                	lw	a2,24(sp)
1c00d276:	46b2                	lw	a3,12(sp)
1c00d278:	ecc6dfe3          	ble	a2,a3,1c00d156 <mm_M_unroll_4x1+0x1ea>
1c00d27c:	5702                	lw	a4,32(sp)
1c00d27e:	4f52                	lw	t5,20(sp)
1c00d280:	8432                	mv	s0,a2
1c00d282:	5e42                	lw	t3,48(sp)
1c00d284:	4652                	lw	a2,20(sp)
1c00d286:	5fd2                	lw	t6,52(sp)
1c00d288:	42d70f33          	p.mac	t5,a4,a3
1c00d28c:	00291e93          	slli	t4,s2,0x2
1c00d290:	57a2                	lw	a5,40(sp)
1c00d292:	5992                	lw	s3,36(sp)
1c00d294:	53f2                	lw	t2,60(sp)
1c00d296:	00271293          	slli	t0,a4,0x2
1c00d29a:	84b6                	mv	s1,a3
1c00d29c:	43d60e33          	p.mac	t3,a2,t4
1c00d2a0:	0f0a                	slli	t5,t5,0x2
1c00d2a2:	9f3e                	add	t5,t5,a5
1c00d2a4:	43d68fb3          	p.mac	t6,a3,t4
1c00d2a8:	877e                	mv	a4,t6
1c00d2aa:	837a                	mv	t1,t5
1c00d2ac:	40940833          	sub	a6,s0,s1
1c00d2b0:	01c8407b          	lp.setup	x0,a6,1c00d2e8 <mm_M_unroll_4x1+0x37c>
1c00d2b4:	00000693          	li	a3,0
1c00d2b8:	00ee88b3          	add	a7,t4,a4
1c00d2bc:	03205463          	blez	s2,1c00d2e4 <mm_M_unroll_4x1+0x378>
1c00d2c0:	00ee88b3          	add	a7,t4,a4
1c00d2c4:	40e887b3          	sub	a5,a7,a4
1c00d2c8:	17f1                	addi	a5,a5,-4
1c00d2ca:	00000693          	li	a3,0
1c00d2ce:	8389                	srli	a5,a5,0x2
1c00d2d0:	8672                	mv	a2,t3
1c00d2d2:	0785                	addi	a5,a5,1
1c00d2d4:	0067c0fb          	lp.setup	x1,a5,1c00d2e0 <mm_M_unroll_4x1+0x374>
1c00d2d8:	0047250b          	p.lw	a0,4(a4!)
1c00d2dc:	0046258b          	p.lw	a1,4(a2!)
1c00d2e0:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00d2e4:	00d362ab          	p.sw	a3,t0(t1!)
1c00d2e8:	8746                	mv	a4,a7
1c00d2ea:	0385                	addi	t2,t2,1
1c00d2ec:	0f11                	addi	t5,t5,4
1c00d2ee:	9e76                	add	t3,t3,t4
1c00d2f0:	fa799ce3          	bne	s3,t2,1c00d2a8 <mm_M_unroll_4x1+0x33c>
1c00d2f4:	b58d                	j	1c00d156 <mm_M_unroll_4x1+0x1ea>
1c00d2f6:	00000813          	li	a6,0
1c00d2fa:	9462                	add	s0,s0,s8
1c00d2fc:	8642                	mv	a2,a6
1c00d2fe:	85c2                	mv	a1,a6
1c00d300:	8542                	mv	a0,a6
1c00d302:	00cb6dab          	p.sw	a2,s11(s6!)
1c00d306:	00ba6dab          	p.sw	a1,s11(s4!)
1c00d30a:	00a9edab          	p.sw	a0,s11(s3!)
1c00d30e:	010aedab          	p.sw	a6,s11(s5!)
1c00d312:	14fd                	addi	s1,s1,-1
1c00d314:	d2049fe3          	bnez	s1,1c00d052 <mm_M_unroll_4x1+0xe6>
1c00d318:	b35d                	j	1c00d0be <mm_M_unroll_4x1+0x152>
1c00d31a:	00000513          	li	a0,0
1c00d31e:	94ea                	add	s1,s1,s10
1c00d320:	86aa                	mv	a3,a0
1c00d322:	862a                	mv	a2,a0
1c00d324:	85aa                	mv	a1,a0
1c00d326:	00db6dab          	p.sw	a3,s11(s6!)
1c00d32a:	00ca6dab          	p.sw	a2,s11(s4!)
1c00d32e:	00b9edab          	p.sw	a1,s11(s3!)
1c00d332:	00aaedab          	p.sw	a0,s11(s5!)
1c00d336:	147d                	addi	s0,s0,-1
1c00d338:	ea041ee3          	bnez	s0,1c00d1f4 <mm_M_unroll_4x1+0x288>
1c00d33c:	b70d                	j	1c00d25e <mm_M_unroll_4x1+0x2f2>
1c00d33e:	5436                	lw	s0,108(sp)
1c00d340:	54a6                	lw	s1,104(sp)
1c00d342:	5916                	lw	s2,100(sp)
1c00d344:	5986                	lw	s3,96(sp)
1c00d346:	4a76                	lw	s4,92(sp)
1c00d348:	4ae6                	lw	s5,88(sp)
1c00d34a:	4b56                	lw	s6,84(sp)
1c00d34c:	4bc6                	lw	s7,80(sp)
1c00d34e:	4c36                	lw	s8,76(sp)
1c00d350:	4ca6                	lw	s9,72(sp)
1c00d352:	4d16                	lw	s10,68(sp)
1c00d354:	4d86                	lw	s11,64(sp)
1c00d356:	6165                	addi	sp,sp,112
1c00d358:	b225                	j	1c00cc80 <mm_M_unroll_2x1>
1c00d35a:	4485                	li	s1,1
1c00d35c:	b9dd                	j	1c00d052 <mm_M_unroll_4x1+0xe6>
1c00d35e:	4405                	li	s0,1
1c00d360:	bd51                	j	1c00d1f4 <mm_M_unroll_4x1+0x288>
	...

1c00d364 <mm_M_unroll_8x1>:
1c00d364:	454c                	lw	a1,12(a0)
1c00d366:	7171                	addi	sp,sp,-176
1c00d368:	d722                	sw	s0,172(sp)
1c00d36a:	d526                	sw	s1,168(sp)
1c00d36c:	d34a                	sw	s2,164(sp)
1c00d36e:	d14e                	sw	s3,160(sp)
1c00d370:	cf52                	sw	s4,156(sp)
1c00d372:	cd56                	sw	s5,152(sp)
1c00d374:	cb5a                	sw	s6,148(sp)
1c00d376:	c95e                	sw	s7,144(sp)
1c00d378:	c762                	sw	s8,140(sp)
1c00d37a:	c566                	sw	s9,136(sp)
1c00d37c:	c36a                	sw	s10,132(sp)
1c00d37e:	c16e                	sw	s11,128(sp)
1c00d380:	caae                	sw	a1,84(sp)
1c00d382:	469d                	li	a3,7
1c00d384:	f14027f3          	csrr	a5,mhartid
1c00d388:	50b6d163          	ble	a1,a3,1c00d88a <mm_M_unroll_8x1+0x526>
1c00d38c:	4910                	lw	a2,16(a0)
1c00d38e:	f457b6b3          	p.bclr	a3,a5,26,5
1c00d392:	872a                	mv	a4,a0
1c00d394:	d0b2                	sw	a2,96(sp)
1c00d396:	061d                	addi	a2,a2,7
1c00d398:	41f65793          	srai	a5,a2,0x1f
1c00d39c:	f837b7b3          	p.bclr	a5,a5,28,3
1c00d3a0:	06c7a7db          	p.addn	a5,a5,a2,3
1c00d3a4:	02f686b3          	mul	a3,a3,a5
1c00d3a8:	4d0c                	lw	a1,24(a0)
1c00d3aa:	4710                	lw	a2,8(a4)
1c00d3ac:	4b58                	lw	a4,20(a4)
1c00d3ae:	d4b2                	sw	a2,104(sp)
1c00d3b0:	dc3a                	sw	a4,56(sp)
1c00d3b2:	5606                	lw	a2,96(sp)
1c00d3b4:	4756                	lw	a4,84(sp)
1c00d3b6:	c8b6                	sw	a3,80(sp)
1c00d3b8:	4114                	lw	a3,0(a0)
1c00d3ba:	4148                	lw	a0,4(a0)
1c00d3bc:	f8373733          	p.bclr	a4,a4,28,3
1c00d3c0:	d8b6                	sw	a3,112(sp)
1c00d3c2:	d6aa                	sw	a0,108(sp)
1c00d3c4:	4546                	lw	a0,80(sp)
1c00d3c6:	46d6                	lw	a3,84(sp)
1c00d3c8:	daba                	sw	a4,116(sp)
1c00d3ca:	97aa                	add	a5,a5,a0
1c00d3cc:	c406b6b3          	p.bclr	a3,a3,2,0
1c00d3d0:	04c7c7b3          	p.min	a5,a5,a2
1c00d3d4:	c6b6                	sw	a3,76(sp)
1c00d3d6:	dcaa                	sw	a0,120(sp)
1c00d3d8:	d2be                	sw	a5,100(sp)
1c00d3da:	24059663          	bnez	a1,1c00d626 <mm_M_unroll_8x1+0x2c2>
1c00d3de:	863e                	mv	a2,a5
1c00d3e0:	22f55563          	ble	a5,a0,1c00d60a <mm_M_unroll_8x1+0x2a6>
1c00d3e4:	18068b63          	beqz	a3,1c00d57a <mm_M_unroll_8x1+0x216>
1c00d3e8:	00251713          	slli	a4,a0,0x2
1c00d3ec:	5562                	lw	a0,56(sp)
1c00d3ee:	00261693          	slli	a3,a2,0x2
1c00d3f2:	5626                	lw	a2,104(sp)
1c00d3f4:	40a007b3          	neg	a5,a0
1c00d3f8:	00251593          	slli	a1,a0,0x2
1c00d3fc:	5536                	lw	a0,108(sp)
1c00d3fe:	c42e                	sw	a1,8(sp)
1c00d400:	55a6                	lw	a1,104(sp)
1c00d402:	963a                	add	a2,a2,a4
1c00d404:	972a                	add	a4,a4,a0
1c00d406:	c4b2                	sw	a2,72(sp)
1c00d408:	d83a                	sw	a4,48(sp)
1c00d40a:	5606                	lw	a2,96(sp)
1c00d40c:	00d58733          	add	a4,a1,a3
1c00d410:	ccba                	sw	a4,88(sp)
1c00d412:	5686                	lw	a3,96(sp)
1c00d414:	5762                	lw	a4,56(sp)
1c00d416:	5546                	lw	a0,112(sp)
1c00d418:	45a2                	lw	a1,8(sp)
1c00d41a:	0616                	slli	a2,a2,0x5
1c00d41c:	dcb2                	sw	a2,120(sp)
1c00d41e:	068a                	slli	a3,a3,0x2
1c00d420:	00279613          	slli	a2,a5,0x2
1c00d424:	0716                	slli	a4,a4,0x5
1c00d426:	952e                	add	a0,a0,a1
1c00d428:	078e                	slli	a5,a5,0x3
1c00d42a:	da36                	sw	a3,52(sp)
1c00d42c:	de3a                	sw	a4,60(sp)
1c00d42e:	ceaa                	sw	a0,92(sp)
1c00d430:	c0b2                	sw	a2,64(sp)
1c00d432:	c2be                	sw	a5,68(sp)
1c00d434:	4626                	lw	a2,72(sp)
1c00d436:	55d2                	lw	a1,52(sp)
1c00d438:	56d2                	lw	a3,52(sp)
1c00d43a:	5752                	lw	a4,52(sp)
1c00d43c:	95b2                	add	a1,a1,a2
1c00d43e:	57d2                	lw	a5,52(sp)
1c00d440:	96ae                	add	a3,a3,a1
1c00d442:	5552                	lw	a0,52(sp)
1c00d444:	9736                	add	a4,a4,a3
1c00d446:	5652                	lw	a2,52(sp)
1c00d448:	97ba                	add	a5,a5,a4
1c00d44a:	953e                	add	a0,a0,a5
1c00d44c:	962a                	add	a2,a2,a0
1c00d44e:	d22e                	sw	a1,36(sp)
1c00d450:	ca2a                	sw	a0,20(sp)
1c00d452:	45b6                	lw	a1,76(sp)
1c00d454:	5552                	lw	a0,52(sp)
1c00d456:	ce3a                	sw	a4,28(sp)
1c00d458:	cc3e                	sw	a5,24(sp)
1c00d45a:	9532                	add	a0,a0,a2
1c00d45c:	fff58793          	addi	a5,a1,-1
1c00d460:	c832                	sw	a2,16(sp)
1c00d462:	00758713          	addi	a4,a1,7
1c00d466:	4626                	lw	a2,72(sp)
1c00d468:	45f6                	lw	a1,92(sp)
1c00d46a:	838d                	srli	a5,a5,0x3
1c00d46c:	0785                	addi	a5,a5,1
1c00d46e:	d036                	sw	a3,32(sp)
1c00d470:	d42a                	sw	a0,40(sp)
1c00d472:	46a1                	li	a3,8
1c00d474:	c22e                	sw	a1,4(sp)
1c00d476:	d632                	sw	a2,44(sp)
1c00d478:	c63e                	sw	a5,12(sp)
1c00d47a:	42d76763          	bltu	a4,a3,1c00d8a8 <mm_M_unroll_8x1+0x544>
1c00d47e:	56e2                	lw	a3,56(sp)
1c00d480:	3ed05163          	blez	a3,1c00d862 <mm_M_unroll_8x1+0x4fe>
1c00d484:	4792                	lw	a5,4(sp)
1c00d486:	4706                	lw	a4,64(sp)
1c00d488:	4516                	lw	a0,68(sp)
1c00d48a:	45a2                	lw	a1,8(sp)
1c00d48c:	00f706b3          	add	a3,a4,a5
1c00d490:	00000e93          	li	t4,0
1c00d494:	40a68433          	sub	s0,a3,a0
1c00d498:	00b403b3          	add	t2,s0,a1
1c00d49c:	40d78733          	sub	a4,a5,a3
1c00d4a0:	47a2                	lw	a5,8(sp)
1c00d4a2:	00b382b3          	add	t0,t2,a1
1c00d4a6:	1771                	addi	a4,a4,-4
1c00d4a8:	00b28fb3          	add	t6,t0,a1
1c00d4ac:	8309                	srli	a4,a4,0x2
1c00d4ae:	8e76                	mv	t3,t4
1c00d4b0:	8376                	mv	t1,t4
1c00d4b2:	88f6                	mv	a7,t4
1c00d4b4:	8876                	mv	a6,t4
1c00d4b6:	8576                	mv	a0,t4
1c00d4b8:	85f6                	mv	a1,t4
1c00d4ba:	8676                	mv	a2,t4
1c00d4bc:	4912                	lw	s2,4(sp)
1c00d4be:	54c2                	lw	s1,48(sp)
1c00d4c0:	00ff8f33          	add	t5,t6,a5
1c00d4c4:	0705                	addi	a4,a4,1
1c00d4c6:	00ff09b3          	add	s3,t5,a5
1c00d4ca:	8dba                	mv	s11,a4
1c00d4cc:	023dc0fb          	lp.setup	x1,s11,1c00d512 <mm_M_unroll_8x1+0x1ae>
1c00d4d0:	5752                	lw	a4,52(sp)
1c00d4d2:	0046ad0b          	p.lw	s10,4(a3!)
1c00d4d6:	00492c8b          	p.lw	s9,4(s2!)
1c00d4da:	20e4f78b          	p.lw	a5,a4(s1!)
1c00d4de:	00442c0b          	p.lw	s8,4(s0!)
1c00d4e2:	0043ab8b          	p.lw	s7,4(t2!)
1c00d4e6:	0042ab0b          	p.lw	s6,4(t0!)
1c00d4ea:	004faa8b          	p.lw	s5,4(t6!)
1c00d4ee:	004f2a0b          	p.lw	s4,4(t5!)
1c00d4f2:	0049a70b          	p.lw	a4,4(s3!)
1c00d4f6:	60fd7643          	fmadd.s	a2,s10,a5,a2
1c00d4fa:	58fcf5c3          	fmadd.s	a1,s9,a5,a1
1c00d4fe:	50fc7543          	fmadd.s	a0,s8,a5,a0
1c00d502:	80fbf843          	fmadd.s	a6,s7,a5,a6
1c00d506:	88fb78c3          	fmadd.s	a7,s6,a5,a7
1c00d50a:	30faf343          	fmadd.s	t1,s5,a5,t1
1c00d50e:	e0fa7e43          	fmadd.s	t3,s4,a5,t3
1c00d512:	e8f77ec3          	fmadd.s	t4,a4,a5,t4
1c00d516:	56e6                	lw	a3,120(sp)
1c00d518:	57b2                	lw	a5,44(sp)
1c00d51a:	5712                	lw	a4,36(sp)
1c00d51c:	00c7e6ab          	p.sw	a2,a3(a5!)
1c00d520:	d63e                	sw	a5,44(sp)
1c00d522:	5782                	lw	a5,32(sp)
1c00d524:	00b766ab          	p.sw	a1,a3(a4!)
1c00d528:	45e2                	lw	a1,24(sp)
1c00d52a:	00a7e6ab          	p.sw	a0,a3(a5!)
1c00d52e:	4572                	lw	a0,28(sp)
1c00d530:	4652                	lw	a2,20(sp)
1c00d532:	d23a                	sw	a4,36(sp)
1c00d534:	010566ab          	p.sw	a6,a3(a0!)
1c00d538:	0115e6ab          	p.sw	a7,a3(a1!)
1c00d53c:	006666ab          	p.sw	t1,a3(a2!)
1c00d540:	4742                	lw	a4,16(sp)
1c00d542:	d03e                	sw	a5,32(sp)
1c00d544:	ce2a                	sw	a0,28(sp)
1c00d546:	57a2                	lw	a5,40(sp)
1c00d548:	4512                	lw	a0,4(sp)
1c00d54a:	cc2e                	sw	a1,24(sp)
1c00d54c:	ca32                	sw	a2,20(sp)
1c00d54e:	55f2                	lw	a1,60(sp)
1c00d550:	4632                	lw	a2,12(sp)
1c00d552:	01c766ab          	p.sw	t3,a3(a4!)
1c00d556:	01d7e6ab          	p.sw	t4,a3(a5!)
1c00d55a:	952e                	add	a0,a0,a1
1c00d55c:	167d                	addi	a2,a2,-1
1c00d55e:	c83a                	sw	a4,16(sp)
1c00d560:	d43e                	sw	a5,40(sp)
1c00d562:	c22a                	sw	a0,4(sp)
1c00d564:	c632                	sw	a2,12(sp)
1c00d566:	fe01                	bnez	a2,1c00d47e <mm_M_unroll_8x1+0x11a>
1c00d568:	4726                	lw	a4,72(sp)
1c00d56a:	57c2                	lw	a5,48(sp)
1c00d56c:	4566                	lw	a0,88(sp)
1c00d56e:	0711                	addi	a4,a4,4
1c00d570:	0791                	addi	a5,a5,4
1c00d572:	c4ba                	sw	a4,72(sp)
1c00d574:	d83e                	sw	a5,48(sp)
1c00d576:	eae51fe3          	bne	a0,a4,1c00d434 <mm_M_unroll_8x1+0xd0>
1c00d57a:	5656                	lw	a2,116(sp)
1c00d57c:	c659                	beqz	a2,1c00d60a <mm_M_unroll_8x1+0x2a6>
1c00d57e:	46d6                	lw	a3,84(sp)
1c00d580:	4736                	lw	a4,76(sp)
1c00d582:	08d75463          	ble	a3,a4,1c00d60a <mm_M_unroll_8x1+0x2a6>
1c00d586:	5786                	lw	a5,96(sp)
1c00d588:	4f46                	lw	t5,80(sp)
1c00d58a:	5662                	lw	a2,56(sp)
1c00d58c:	00279393          	slli	t2,a5,0x2
1c00d590:	42e78f33          	p.mac	t5,a5,a4
1c00d594:	52c6                	lw	t0,112(sp)
1c00d596:	47b6                	lw	a5,76(sp)
1c00d598:	00261e93          	slli	t4,a2,0x2
1c00d59c:	4546                	lw	a0,80(sp)
1c00d59e:	5596                	lw	a1,100(sp)
1c00d5a0:	84b6                	mv	s1,a3
1c00d5a2:	5726                	lw	a4,104(sp)
1c00d5a4:	56b6                	lw	a3,108(sp)
1c00d5a6:	00251e13          	slli	t3,a0,0x2
1c00d5aa:	43d782b3          	p.mac	t0,a5,t4
1c00d5ae:	00259f93          	slli	t6,a1,0x2
1c00d5b2:	0f0a                	slli	t5,t5,0x2
1c00d5b4:	9e36                	add	t3,t3,a3
1c00d5b6:	9fb6                	add	t6,t6,a3
1c00d5b8:	9f3a                	add	t5,t5,a4
1c00d5ba:	8432                	mv	s0,a2
1c00d5bc:	893e                	mv	s2,a5
1c00d5be:	8716                	mv	a4,t0
1c00d5c0:	837a                	mv	t1,t5
1c00d5c2:	41248833          	sub	a6,s1,s2
1c00d5c6:	0001                	nop
1c00d5c8:	01c8407b          	lp.setup	x0,a6,1c00d600 <mm_M_unroll_8x1+0x29c>
1c00d5cc:	00000693          	li	a3,0
1c00d5d0:	00ee88b3          	add	a7,t4,a4
1c00d5d4:	02805463          	blez	s0,1c00d5fc <mm_M_unroll_8x1+0x298>
1c00d5d8:	00ee88b3          	add	a7,t4,a4
1c00d5dc:	40e887b3          	sub	a5,a7,a4
1c00d5e0:	17f1                	addi	a5,a5,-4
1c00d5e2:	00000693          	li	a3,0
1c00d5e6:	8389                	srli	a5,a5,0x2
1c00d5e8:	8672                	mv	a2,t3
1c00d5ea:	0785                	addi	a5,a5,1
1c00d5ec:	0067c0fb          	lp.setup	x1,a5,1c00d5f8 <mm_M_unroll_8x1+0x294>
1c00d5f0:	0047250b          	p.lw	a0,4(a4!)
1c00d5f4:	2076758b          	p.lw	a1,t2(a2!)
1c00d5f8:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00d5fc:	00d363ab          	p.sw	a3,t2(t1!)
1c00d600:	8746                	mv	a4,a7
1c00d602:	0e11                	addi	t3,t3,4
1c00d604:	0f11                	addi	t5,t5,4
1c00d606:	fbcf9ce3          	bne	t6,t3,1c00d5be <mm_M_unroll_8x1+0x25a>
1c00d60a:	543a                	lw	s0,172(sp)
1c00d60c:	54aa                	lw	s1,168(sp)
1c00d60e:	591a                	lw	s2,164(sp)
1c00d610:	598a                	lw	s3,160(sp)
1c00d612:	4a7a                	lw	s4,156(sp)
1c00d614:	4aea                	lw	s5,152(sp)
1c00d616:	4b5a                	lw	s6,148(sp)
1c00d618:	4bca                	lw	s7,144(sp)
1c00d61a:	4c3a                	lw	s8,140(sp)
1c00d61c:	4caa                	lw	s9,136(sp)
1c00d61e:	4d1a                	lw	s10,132(sp)
1c00d620:	4d8a                	lw	s11,128(sp)
1c00d622:	614d                	addi	sp,sp,176
1c00d624:	8082                	ret
1c00d626:	fef552e3          	ble	a5,a0,1c00d60a <mm_M_unroll_8x1+0x2a6>
1c00d62a:	1a068563          	beqz	a3,1c00d7d4 <mm_M_unroll_8x1+0x470>
1c00d62e:	85b2                	mv	a1,a2
1c00d630:	00251613          	slli	a2,a0,0x2
1c00d634:	5526                	lw	a0,104(sp)
1c00d636:	56e2                	lw	a3,56(sp)
1c00d638:	0596                	slli	a1,a1,0x5
1c00d63a:	962a                	add	a2,a2,a0
1c00d63c:	c4b2                	sw	a2,72(sp)
1c00d63e:	5606                	lw	a2,96(sp)
1c00d640:	068a                	slli	a3,a3,0x2
1c00d642:	c236                	sw	a3,4(sp)
1c00d644:	060a                	slli	a2,a2,0x2
1c00d646:	078a                	slli	a5,a5,0x2
1c00d648:	97aa                	add	a5,a5,a0
1c00d64a:	deae                	sw	a1,124(sp)
1c00d64c:	4546                	lw	a0,80(sp)
1c00d64e:	c0b2                	sw	a2,64(sp)
1c00d650:	4592                	lw	a1,4(sp)
1c00d652:	5636                	lw	a2,108(sp)
1c00d654:	ccbe                	sw	a5,88(sp)
1c00d656:	57e2                	lw	a5,56(sp)
1c00d658:	42b50633          	p.mac	a2,a0,a1
1c00d65c:	5762                	lw	a4,56(sp)
1c00d65e:	00371693          	slli	a3,a4,0x3
1c00d662:	40e00733          	neg	a4,a4
1c00d666:	42f507b3          	p.mac	a5,a0,a5
1c00d66a:	d832                	sw	a2,48(sp)
1c00d66c:	5636                	lw	a2,108(sp)
1c00d66e:	5546                	lw	a0,112(sp)
1c00d670:	078a                	slli	a5,a5,0x2
1c00d672:	97b2                	add	a5,a5,a2
1c00d674:	de3e                	sw	a5,60(sp)
1c00d676:	57e2                	lw	a5,56(sp)
1c00d678:	0796                	slli	a5,a5,0x5
1c00d67a:	c2be                	sw	a5,68(sp)
1c00d67c:	00d507b3          	add	a5,a0,a3
1c00d680:	cebe                	sw	a5,92(sp)
1c00d682:	00371793          	slli	a5,a4,0x3
1c00d686:	da3e                	sw	a5,52(sp)
1c00d688:	47a6                	lw	a5,72(sp)
1c00d68a:	4706                	lw	a4,64(sp)
1c00d68c:	4506                	lw	a0,64(sp)
1c00d68e:	4586                	lw	a1,64(sp)
1c00d690:	973e                	add	a4,a4,a5
1c00d692:	4606                	lw	a2,64(sp)
1c00d694:	953a                	add	a0,a0,a4
1c00d696:	4686                	lw	a3,64(sp)
1c00d698:	95aa                	add	a1,a1,a0
1c00d69a:	d02a                	sw	a0,32(sp)
1c00d69c:	4506                	lw	a0,64(sp)
1c00d69e:	962e                	add	a2,a2,a1
1c00d6a0:	d23a                	sw	a4,36(sp)
1c00d6a2:	ce2e                	sw	a1,28(sp)
1c00d6a4:	4736                	lw	a4,76(sp)
1c00d6a6:	4586                	lw	a1,64(sp)
1c00d6a8:	96b2                	add	a3,a3,a2
1c00d6aa:	9536                	add	a0,a0,a3
1c00d6ac:	fff70793          	addi	a5,a4,-1
1c00d6b0:	95aa                	add	a1,a1,a0
1c00d6b2:	cc32                	sw	a2,24(sp)
1c00d6b4:	c82a                	sw	a0,16(sp)
1c00d6b6:	4676                	lw	a2,92(sp)
1c00d6b8:	4526                	lw	a0,72(sp)
1c00d6ba:	838d                	srli	a5,a5,0x3
1c00d6bc:	0785                	addi	a5,a5,1
1c00d6be:	ca36                	sw	a3,20(sp)
1c00d6c0:	071d                	addi	a4,a4,7
1c00d6c2:	46a1                	li	a3,8
1c00d6c4:	d42e                	sw	a1,40(sp)
1c00d6c6:	c432                	sw	a2,8(sp)
1c00d6c8:	d62a                	sw	a0,44(sp)
1c00d6ca:	c63e                	sw	a5,12(sp)
1c00d6cc:	1ed76163          	bltu	a4,a3,1c00d8ae <mm_M_unroll_8x1+0x54a>
1c00d6d0:	5762                	lw	a4,56(sp)
1c00d6d2:	1ae05263          	blez	a4,1c00d876 <mm_M_unroll_8x1+0x512>
1c00d6d6:	5752                	lw	a4,52(sp)
1c00d6d8:	47a2                	lw	a5,8(sp)
1c00d6da:	4512                	lw	a0,4(sp)
1c00d6dc:	55f2                	lw	a1,60(sp)
1c00d6de:	00f70433          	add	s0,a4,a5
1c00d6e2:	5642                	lw	a2,48(sp)
1c00d6e4:	00000e13          	li	t3,0
1c00d6e8:	008503b3          	add	t2,a0,s0
1c00d6ec:	40e382b3          	sub	t0,t2,a4
1c00d6f0:	4792                	lw	a5,4(sp)
1c00d6f2:	40c58733          	sub	a4,a1,a2
1c00d6f6:	00550fb3          	add	t6,a0,t0
1c00d6fa:	01f50f33          	add	t5,a0,t6
1c00d6fe:	1771                	addi	a4,a4,-4
1c00d700:	8372                	mv	t1,t3
1c00d702:	88f2                	mv	a7,t3
1c00d704:	8872                	mv	a6,t3
1c00d706:	8572                	mv	a0,t3
1c00d708:	85f2                	mv	a1,t3
1c00d70a:	8672                	mv	a2,t3
1c00d70c:	86f2                	mv	a3,t3
1c00d70e:	4922                	lw	s2,8(sp)
1c00d710:	54c2                	lw	s1,48(sp)
1c00d712:	01e78eb3          	add	t4,a5,t5
1c00d716:	8309                	srli	a4,a4,0x2
1c00d718:	01d789b3          	add	s3,a5,t4
1c00d71c:	0705                	addi	a4,a4,1
1c00d71e:	0001                	nop
1c00d720:	022740fb          	lp.setup	x1,a4,1c00d764 <mm_M_unroll_8x1+0x400>
1c00d724:	0044a78b          	p.lw	a5,4(s1!)
1c00d728:	00442d8b          	p.lw	s11,4(s0!)
1c00d72c:	0043ad0b          	p.lw	s10,4(t2!)
1c00d730:	00492c8b          	p.lw	s9,4(s2!)
1c00d734:	0042ac0b          	p.lw	s8,4(t0!)
1c00d738:	004fab8b          	p.lw	s7,4(t6!)
1c00d73c:	004f2b0b          	p.lw	s6,4(t5!)
1c00d740:	004eaa8b          	p.lw	s5,4(t4!)
1c00d744:	0049aa0b          	p.lw	s4,4(s3!)
1c00d748:	68fdf6c3          	fmadd.s	a3,s11,a5,a3
1c00d74c:	60fd7643          	fmadd.s	a2,s10,a5,a2
1c00d750:	58fcf5c3          	fmadd.s	a1,s9,a5,a1
1c00d754:	50fc7543          	fmadd.s	a0,s8,a5,a0
1c00d758:	80fbf843          	fmadd.s	a6,s7,a5,a6
1c00d75c:	88fb78c3          	fmadd.s	a7,s6,a5,a7
1c00d760:	30faf343          	fmadd.s	t1,s5,a5,t1
1c00d764:	e0fa7e43          	fmadd.s	t3,s4,a5,t3
1c00d768:	57f6                	lw	a5,124(sp)
1c00d76a:	5732                	lw	a4,44(sp)
1c00d76c:	00d767ab          	p.sw	a3,a5(a4!)
1c00d770:	5692                	lw	a3,36(sp)
1c00d772:	d63a                	sw	a4,44(sp)
1c00d774:	5702                	lw	a4,32(sp)
1c00d776:	00c6e7ab          	p.sw	a2,a5(a3!)
1c00d77a:	4662                	lw	a2,24(sp)
1c00d77c:	00b767ab          	p.sw	a1,a5(a4!)
1c00d780:	45f2                	lw	a1,28(sp)
1c00d782:	d236                	sw	a3,36(sp)
1c00d784:	46d2                	lw	a3,20(sp)
1c00d786:	00a5e7ab          	p.sw	a0,a5(a1!)
1c00d78a:	010667ab          	p.sw	a6,a5(a2!)
1c00d78e:	0116e7ab          	p.sw	a7,a5(a3!)
1c00d792:	d03a                	sw	a4,32(sp)
1c00d794:	4742                	lw	a4,16(sp)
1c00d796:	5522                	lw	a0,40(sp)
1c00d798:	ce2e                	sw	a1,28(sp)
1c00d79a:	cc32                	sw	a2,24(sp)
1c00d79c:	45a2                	lw	a1,8(sp)
1c00d79e:	ca36                	sw	a3,20(sp)
1c00d7a0:	4616                	lw	a2,68(sp)
1c00d7a2:	46b2                	lw	a3,12(sp)
1c00d7a4:	006767ab          	p.sw	t1,a5(a4!)
1c00d7a8:	01c567ab          	p.sw	t3,a5(a0!)
1c00d7ac:	95b2                	add	a1,a1,a2
1c00d7ae:	16fd                	addi	a3,a3,-1
1c00d7b0:	c83a                	sw	a4,16(sp)
1c00d7b2:	d42a                	sw	a0,40(sp)
1c00d7b4:	c42e                	sw	a1,8(sp)
1c00d7b6:	c636                	sw	a3,12(sp)
1c00d7b8:	fe81                	bnez	a3,1c00d6d0 <mm_M_unroll_8x1+0x36c>
1c00d7ba:	47a6                	lw	a5,72(sp)
1c00d7bc:	5542                	lw	a0,48(sp)
1c00d7be:	4592                	lw	a1,4(sp)
1c00d7c0:	5672                	lw	a2,60(sp)
1c00d7c2:	46e6                	lw	a3,88(sp)
1c00d7c4:	0791                	addi	a5,a5,4
1c00d7c6:	952e                	add	a0,a0,a1
1c00d7c8:	962e                	add	a2,a2,a1
1c00d7ca:	c4be                	sw	a5,72(sp)
1c00d7cc:	d82a                	sw	a0,48(sp)
1c00d7ce:	de32                	sw	a2,60(sp)
1c00d7d0:	eaf69de3          	bne	a3,a5,1c00d68a <mm_M_unroll_8x1+0x326>
1c00d7d4:	5556                	lw	a0,116(sp)
1c00d7d6:	e2050ae3          	beqz	a0,1c00d60a <mm_M_unroll_8x1+0x2a6>
1c00d7da:	45d6                	lw	a1,84(sp)
1c00d7dc:	4636                	lw	a2,76(sp)
1c00d7de:	e2b656e3          	ble	a1,a2,1c00d60a <mm_M_unroll_8x1+0x2a6>
1c00d7e2:	5686                	lw	a3,96(sp)
1c00d7e4:	4f46                	lw	t5,80(sp)
1c00d7e6:	5762                	lw	a4,56(sp)
1c00d7e8:	5e36                	lw	t3,108(sp)
1c00d7ea:	4546                	lw	a0,80(sp)
1c00d7ec:	5fc6                	lw	t6,112(sp)
1c00d7ee:	42c68f33          	p.mac	t5,a3,a2
1c00d7f2:	00271e93          	slli	t4,a4,0x2
1c00d7f6:	57a6                	lw	a5,104(sp)
1c00d7f8:	5996                	lw	s3,100(sp)
1c00d7fa:	5466                	lw	s0,120(sp)
1c00d7fc:	00269293          	slli	t0,a3,0x2
1c00d800:	84ae                	mv	s1,a1
1c00d802:	83ba                	mv	t2,a4
1c00d804:	8932                	mv	s2,a2
1c00d806:	43d50e33          	p.mac	t3,a0,t4
1c00d80a:	0f0a                	slli	t5,t5,0x2
1c00d80c:	9f3e                	add	t5,t5,a5
1c00d80e:	43d60fb3          	p.mac	t6,a2,t4
1c00d812:	877e                	mv	a4,t6
1c00d814:	837a                	mv	t1,t5
1c00d816:	41248833          	sub	a6,s1,s2
1c00d81a:	0001                	nop
1c00d81c:	01c8407b          	lp.setup	x0,a6,1c00d854 <mm_M_unroll_8x1+0x4f0>
1c00d820:	00000693          	li	a3,0
1c00d824:	00ee88b3          	add	a7,t4,a4
1c00d828:	02705463          	blez	t2,1c00d850 <mm_M_unroll_8x1+0x4ec>
1c00d82c:	00ee88b3          	add	a7,t4,a4
1c00d830:	40e887b3          	sub	a5,a7,a4
1c00d834:	17f1                	addi	a5,a5,-4
1c00d836:	00000693          	li	a3,0
1c00d83a:	8389                	srli	a5,a5,0x2
1c00d83c:	8672                	mv	a2,t3
1c00d83e:	0785                	addi	a5,a5,1
1c00d840:	0067c0fb          	lp.setup	x1,a5,1c00d84c <mm_M_unroll_8x1+0x4e8>
1c00d844:	0047250b          	p.lw	a0,4(a4!)
1c00d848:	0046258b          	p.lw	a1,4(a2!)
1c00d84c:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00d850:	00d362ab          	p.sw	a3,t0(t1!)
1c00d854:	8746                	mv	a4,a7
1c00d856:	0405                	addi	s0,s0,1
1c00d858:	0f11                	addi	t5,t5,4
1c00d85a:	9e76                	add	t3,t3,t4
1c00d85c:	fa899be3          	bne	s3,s0,1c00d812 <mm_M_unroll_8x1+0x4ae>
1c00d860:	b36d                	j	1c00d60a <mm_M_unroll_8x1+0x2a6>
1c00d862:	00000e93          	li	t4,0
1c00d866:	8e76                	mv	t3,t4
1c00d868:	8376                	mv	t1,t4
1c00d86a:	88f6                	mv	a7,t4
1c00d86c:	8876                	mv	a6,t4
1c00d86e:	8576                	mv	a0,t4
1c00d870:	85f6                	mv	a1,t4
1c00d872:	8676                	mv	a2,t4
1c00d874:	b14d                	j	1c00d516 <mm_M_unroll_8x1+0x1b2>
1c00d876:	00000e13          	li	t3,0
1c00d87a:	8372                	mv	t1,t3
1c00d87c:	88f2                	mv	a7,t3
1c00d87e:	8872                	mv	a6,t3
1c00d880:	8572                	mv	a0,t3
1c00d882:	85f2                	mv	a1,t3
1c00d884:	8672                	mv	a2,t3
1c00d886:	86f2                	mv	a3,t3
1c00d888:	b5c5                	j	1c00d768 <mm_M_unroll_8x1+0x404>
1c00d88a:	543a                	lw	s0,172(sp)
1c00d88c:	54aa                	lw	s1,168(sp)
1c00d88e:	591a                	lw	s2,164(sp)
1c00d890:	598a                	lw	s3,160(sp)
1c00d892:	4a7a                	lw	s4,156(sp)
1c00d894:	4aea                	lw	s5,152(sp)
1c00d896:	4b5a                	lw	s6,148(sp)
1c00d898:	4bca                	lw	s7,144(sp)
1c00d89a:	4c3a                	lw	s8,140(sp)
1c00d89c:	4caa                	lw	s9,136(sp)
1c00d89e:	4d1a                	lw	s10,132(sp)
1c00d8a0:	4d8a                	lw	s11,128(sp)
1c00d8a2:	614d                	addi	sp,sp,176
1c00d8a4:	ec8ff06f          	j	1c00cf6c <mm_M_unroll_4x1>
1c00d8a8:	4685                	li	a3,1
1c00d8aa:	c636                	sw	a3,12(sp)
1c00d8ac:	bec9                	j	1c00d47e <mm_M_unroll_8x1+0x11a>
1c00d8ae:	4585                	li	a1,1
1c00d8b0:	c62e                	sw	a1,12(sp)
1c00d8b2:	bd39                	j	1c00d6d0 <mm_M_unroll_8x1+0x36c>
1c00d8b4:	0000                	unimp
	...

1c00d8b8 <mm_M_unroll_1x2>:
1c00d8b8:	7139                	addi	sp,sp,-64
1c00d8ba:	d05e                	sw	s7,32(sp)
1c00d8bc:	01052b83          	lw	s7,16(a0)
1c00d8c0:	d25a                	sw	s6,36(sp)
1c00d8c2:	de22                	sw	s0,60(sp)
1c00d8c4:	dc26                	sw	s1,56(sp)
1c00d8c6:	da4a                	sw	s2,52(sp)
1c00d8c8:	d84e                	sw	s3,48(sp)
1c00d8ca:	d652                	sw	s4,44(sp)
1c00d8cc:	d456                	sw	s5,40(sp)
1c00d8ce:	ce62                	sw	s8,28(sp)
1c00d8d0:	cc66                	sw	s9,24(sp)
1c00d8d2:	ca6a                	sw	s10,20(sp)
1c00d8d4:	c86e                	sw	s11,16(sp)
1c00d8d6:	473d                	li	a4,15
1c00d8d8:	c00bbb33          	p.bclr	s6,s7,0,0
1c00d8dc:	f14026f3          	csrr	a3,mhartid
1c00d8e0:	2f675863          	ble	s6,a4,1c00dbd0 <mm_M_unroll_1x2+0x318>
1c00d8e4:	007b0713          	addi	a4,s6,7
1c00d8e8:	f456b6b3          	p.bclr	a3,a3,26,5
1c00d8ec:	870d                	srai	a4,a4,0x3
1c00d8ee:	02e68933          	mul	s2,a3,a4
1c00d8f2:	4150                	lw	a2,4(a0)
1c00d8f4:	c236                	sw	a3,4(sp)
1c00d8f6:	4d14                	lw	a3,24(a0)
1c00d8f8:	416b87b3          	sub	a5,s7,s6
1c00d8fc:	c432                	sw	a2,8(sp)
1c00d8fe:	c63e                	sw	a5,12(sp)
1c00d900:	00052383          	lw	t2,0(a0)
1c00d904:	00852c83          	lw	s9,8(a0)
1c00d908:	00c52c03          	lw	s8,12(a0)
1c00d90c:	974a                	add	a4,a4,s2
1c00d90e:	01452e03          	lw	t3,20(a0)
1c00d912:	05674a33          	p.min	s4,a4,s6
1c00d916:	14069463          	bnez	a3,1c00da5e <mm_M_unroll_1x2+0x1a6>
1c00d91a:	07495f63          	ble	s4,s2,1c00d998 <mm_M_unroll_1x2+0xe0>
1c00d91e:	07805d63          	blez	s8,1c00d998 <mm_M_unroll_1x2+0xe0>
1c00d922:	00190413          	addi	s0,s2,1
1c00d926:	00291493          	slli	s1,s2,0x2
1c00d92a:	040a                	slli	s0,s0,0x2
1c00d92c:	009c8ab3          	add	s5,s9,s1
1c00d930:	9432                	add	s0,s0,a2
1c00d932:	94b2                	add	s1,s1,a2
1c00d934:	002b9d13          	slli	s10,s7,0x2
1c00d938:	002e1993          	slli	s3,t3,0x2
1c00d93c:	869e                	mv	a3,t2
1c00d93e:	004a8e93          	addi	t4,s5,4
1c00d942:	8356                	mv	t1,s5
1c00d944:	8862                	mv	a6,s8
1c00d946:	25c05963          	blez	t3,1c00db98 <mm_M_unroll_1x2+0x2e0>
1c00d94a:	00000593          	li	a1,0
1c00d94e:	00d988b3          	add	a7,s3,a3
1c00d952:	40d887b3          	sub	a5,a7,a3
1c00d956:	17f1                	addi	a5,a5,-4
1c00d958:	862e                	mv	a2,a1
1c00d95a:	8389                	srli	a5,a5,0x2
1c00d95c:	8f22                	mv	t5,s0
1c00d95e:	8526                	mv	a0,s1
1c00d960:	0785                	addi	a5,a5,1
1c00d962:	0001                	nop
1c00d964:	00a7c0fb          	lp.setup	x1,a5,1c00d978 <mm_M_unroll_1x2+0xc0>
1c00d968:	0046a70b          	p.lw	a4,4(a3!)
1c00d96c:	21a5728b          	p.lw	t0,s10(a0!)
1c00d970:	21af7f8b          	p.lw	t6,s10(t5!)
1c00d974:	60e2f643          	fmadd.s	a2,t0,a4,a2
1c00d978:	58eff5c3          	fmadd.s	a1,t6,a4,a1
1c00d97c:	00c36d2b          	p.sw	a2,s10(t1!)
1c00d980:	00beed2b          	p.sw	a1,s10(t4!)
1c00d984:	86c6                	mv	a3,a7
1c00d986:	187d                	addi	a6,a6,-1
1c00d988:	fa081fe3          	bnez	a6,1c00d946 <mm_M_unroll_1x2+0x8e>
1c00d98c:	0909                	addi	s2,s2,2
1c00d98e:	0aa1                	addi	s5,s5,8
1c00d990:	04a1                	addi	s1,s1,8
1c00d992:	0421                	addi	s0,s0,8
1c00d994:	fb4944e3          	blt	s2,s4,1c00d93c <mm_M_unroll_1x2+0x84>
1c00d998:	46b2                	lw	a3,12(sp)
1c00d99a:	0ad05463          	blez	a3,1c00da42 <mm_M_unroll_1x2+0x18a>
1c00d99e:	007c0713          	addi	a4,s8,7
1c00d9a2:	41f75793          	srai	a5,a4,0x1f
1c00d9a6:	f837b7b3          	p.bclr	a5,a5,28,3
1c00d9aa:	06e7a7db          	p.addn	a5,a5,a4,3
1c00d9ae:	4712                	lw	a4,4(sp)
1c00d9b0:	02f70333          	mul	t1,a4,a5
1c00d9b4:	979a                	add	a5,a5,t1
1c00d9b6:	0587cc33          	p.min	s8,a5,s8
1c00d9ba:	09835463          	ble	s8,t1,1c00da42 <mm_M_unroll_1x2+0x18a>
1c00d9be:	097b5263          	ble	s7,s6,1c00da42 <mm_M_unroll_1x2+0x18a>
1c00d9c2:	88f2                	mv	a7,t3
1c00d9c4:	426e08b3          	p.mac	a7,t3,t1
1c00d9c8:	002e1f93          	slli	t6,t3,0x2
1c00d9cc:	47a2                	lw	a5,8(sp)
1c00d9ce:	002b1f13          	slli	t5,s6,0x2
1c00d9d2:	002b9293          	slli	t0,s7,0x2
1c00d9d6:	9f3e                	add	t5,t5,a5
1c00d9d8:	00578433          	add	s0,a5,t0
1c00d9dc:	088a                	slli	a7,a7,0x2
1c00d9de:	989e                	add	a7,a7,t2
1c00d9e0:	03730eb3          	mul	t4,t1,s7
1c00d9e4:	426f83b3          	p.mac	t2,t6,t1
1c00d9e8:	41e40833          	sub	a6,s0,t5
1c00d9ec:	016e84b3          	add	s1,t4,s6
1c00d9f0:	1871                	addi	a6,a6,-4
1c00d9f2:	048a                	slli	s1,s1,0x2
1c00d9f4:	00285813          	srli	a6,a6,0x2
1c00d9f8:	94e6                	add	s1,s1,s9
1c00d9fa:	897a                	mv	s2,t5
1c00d9fc:	0805                	addi	a6,a6,1
1c00d9fe:	0001                	nop
1c00da00:	01a8407b          	lp.setup	x0,a6,1c00da34 <mm_M_unroll_1x2+0x17c>
1c00da04:	00000713          	li	a4,0
1c00da08:	03c05463          	blez	t3,1c00da30 <mm_M_unroll_1x2+0x178>
1c00da0c:	407887b3          	sub	a5,a7,t2
1c00da10:	17f1                	addi	a5,a5,-4
1c00da12:	00000713          	li	a4,0
1c00da16:	8389                	srli	a5,a5,0x2
1c00da18:	864a                	mv	a2,s2
1c00da1a:	869e                	mv	a3,t2
1c00da1c:	0785                	addi	a5,a5,1
1c00da1e:	0001                	nop
1c00da20:	0067c0fb          	lp.setup	x1,a5,1c00da2c <mm_M_unroll_1x2+0x174>
1c00da24:	0046a50b          	p.lw	a0,4(a3!)
1c00da28:	2056758b          	p.lw	a1,t0(a2!)
1c00da2c:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00da30:	00e4a22b          	p.sw	a4,4(s1!)
1c00da34:	0911                	addi	s2,s2,4
1c00da36:	0305                	addi	t1,t1,1
1c00da38:	9ede                	add	t4,t4,s7
1c00da3a:	93fe                	add	t2,t2,t6
1c00da3c:	98fe                	add	a7,a7,t6
1c00da3e:	fa6c15e3          	bne	s8,t1,1c00d9e8 <mm_M_unroll_1x2+0x130>
1c00da42:	5472                	lw	s0,60(sp)
1c00da44:	54e2                	lw	s1,56(sp)
1c00da46:	5952                	lw	s2,52(sp)
1c00da48:	59c2                	lw	s3,48(sp)
1c00da4a:	5a32                	lw	s4,44(sp)
1c00da4c:	5aa2                	lw	s5,40(sp)
1c00da4e:	5b12                	lw	s6,36(sp)
1c00da50:	5b82                	lw	s7,32(sp)
1c00da52:	4c72                	lw	s8,28(sp)
1c00da54:	4ce2                	lw	s9,24(sp)
1c00da56:	4d52                	lw	s10,20(sp)
1c00da58:	4dc2                	lw	s11,16(sp)
1c00da5a:	6121                	addi	sp,sp,64
1c00da5c:	8082                	ret
1c00da5e:	09495263          	ble	s4,s2,1c00dae2 <mm_M_unroll_1x2+0x22a>
1c00da62:	09805063          	blez	s8,1c00dae2 <mm_M_unroll_1x2+0x22a>
1c00da66:	8472                	mv	s0,t3
1c00da68:	43c90433          	p.mac	s0,s2,t3
1c00da6c:	002e1993          	slli	s3,t3,0x2
1c00da70:	84b2                	mv	s1,a2
1c00da72:	00291a93          	slli	s5,s2,0x2
1c00da76:	9ae6                	add	s5,s5,s9
1c00da78:	003e1d13          	slli	s10,t3,0x3
1c00da7c:	002b9d93          	slli	s11,s7,0x2
1c00da80:	432984b3          	p.mac	s1,s3,s2
1c00da84:	040a                	slli	s0,s0,0x2
1c00da86:	9432                	add	s0,s0,a2
1c00da88:	869e                	mv	a3,t2
1c00da8a:	004a8313          	addi	t1,s5,4
1c00da8e:	88d6                	mv	a7,s5
1c00da90:	8562                	mv	a0,s8
1c00da92:	13c05163          	blez	t3,1c00dbb4 <mm_M_unroll_1x2+0x2fc>
1c00da96:	00000593          	li	a1,0
1c00da9a:	01368833          	add	a6,a3,s3
1c00da9e:	40d807b3          	sub	a5,a6,a3
1c00daa2:	17f1                	addi	a5,a5,-4
1c00daa4:	862e                	mv	a2,a1
1c00daa6:	8389                	srli	a5,a5,0x2
1c00daa8:	8f22                	mv	t5,s0
1c00daaa:	8ea6                	mv	t4,s1
1c00daac:	0785                	addi	a5,a5,1
1c00daae:	0001                	nop
1c00dab0:	00a7c0fb          	lp.setup	x1,a5,1c00dac4 <mm_M_unroll_1x2+0x20c>
1c00dab4:	0046a70b          	p.lw	a4,4(a3!)
1c00dab8:	004ea28b          	p.lw	t0,4(t4!)
1c00dabc:	004f2f8b          	p.lw	t6,4(t5!)
1c00dac0:	60e2f643          	fmadd.s	a2,t0,a4,a2
1c00dac4:	58eff5c3          	fmadd.s	a1,t6,a4,a1
1c00dac8:	00c8edab          	p.sw	a2,s11(a7!)
1c00dacc:	00b36dab          	p.sw	a1,s11(t1!)
1c00dad0:	86c2                	mv	a3,a6
1c00dad2:	157d                	addi	a0,a0,-1
1c00dad4:	fd5d                	bnez	a0,1c00da92 <mm_M_unroll_1x2+0x1da>
1c00dad6:	0909                	addi	s2,s2,2
1c00dad8:	0aa1                	addi	s5,s5,8
1c00dada:	94ea                	add	s1,s1,s10
1c00dadc:	946a                	add	s0,s0,s10
1c00dade:	fb4945e3          	blt	s2,s4,1c00da88 <mm_M_unroll_1x2+0x1d0>
1c00dae2:	4632                	lw	a2,12(sp)
1c00dae4:	f4c05fe3          	blez	a2,1c00da42 <mm_M_unroll_1x2+0x18a>
1c00dae8:	007c0713          	addi	a4,s8,7
1c00daec:	41f75793          	srai	a5,a4,0x1f
1c00daf0:	4692                	lw	a3,4(sp)
1c00daf2:	f837b7b3          	p.bclr	a5,a5,28,3
1c00daf6:	06e7a7db          	p.addn	a5,a5,a4,3
1c00dafa:	02f68f33          	mul	t5,a3,a5
1c00dafe:	97fa                	add	a5,a5,t5
1c00db00:	0587cc33          	p.min	s8,a5,s8
1c00db04:	f38f5fe3          	ble	s8,t5,1c00da42 <mm_M_unroll_1x2+0x18a>
1c00db08:	f37b5de3          	ble	s7,s6,1c00da42 <mm_M_unroll_1x2+0x18a>
1c00db0c:	037f07b3          	mul	a5,t5,s7
1c00db10:	8772                	mv	a4,t3
1c00db12:	44a2                	lw	s1,8(sp)
1c00db14:	002e1e93          	slli	t4,t3,0x2
1c00db18:	831e                	mv	t1,t2
1c00db1a:	002b9413          	slli	s0,s7,0x2
1c00db1e:	43ee0733          	p.mac	a4,t3,t5
1c00db22:	017782b3          	add	t0,a5,s7
1c00db26:	00229f93          	slli	t6,t0,0x2
1c00db2a:	9fe6                	add	t6,t6,s9
1c00db2c:	43ee8333          	p.mac	t1,t4,t5
1c00db30:	070a                	slli	a4,a4,0x2
1c00db32:	93ba                	add	t2,t2,a4
1c00db34:	43db04b3          	p.mac	s1,s6,t4
1c00db38:	016788b3          	add	a7,a5,s6
1c00db3c:	088a                	slli	a7,a7,0x2
1c00db3e:	98e6                	add	a7,a7,s9
1c00db40:	411f8833          	sub	a6,t6,a7
1c00db44:	1871                	addi	a6,a6,-4
1c00db46:	00285813          	srli	a6,a6,0x2
1c00db4a:	8926                	mv	s2,s1
1c00db4c:	0805                	addi	a6,a6,1
1c00db4e:	0001                	nop
1c00db50:	01a8407b          	lp.setup	x0,a6,1c00db84 <mm_M_unroll_1x2+0x2cc>
1c00db54:	00000713          	li	a4,0
1c00db58:	03c05463          	blez	t3,1c00db80 <mm_M_unroll_1x2+0x2c8>
1c00db5c:	406387b3          	sub	a5,t2,t1
1c00db60:	17f1                	addi	a5,a5,-4
1c00db62:	00000713          	li	a4,0
1c00db66:	8389                	srli	a5,a5,0x2
1c00db68:	864a                	mv	a2,s2
1c00db6a:	869a                	mv	a3,t1
1c00db6c:	0785                	addi	a5,a5,1
1c00db6e:	0001                	nop
1c00db70:	0067c0fb          	lp.setup	x1,a5,1c00db7c <mm_M_unroll_1x2+0x2c4>
1c00db74:	0046a50b          	p.lw	a0,4(a3!)
1c00db78:	0046258b          	p.lw	a1,4(a2!)
1c00db7c:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00db80:	00e8a22b          	p.sw	a4,4(a7!)
1c00db84:	9976                	add	s2,s2,t4
1c00db86:	0f05                	addi	t5,t5,1
1c00db88:	8796                	mv	a5,t0
1c00db8a:	9fa2                	add	t6,t6,s0
1c00db8c:	9376                	add	t1,t1,t4
1c00db8e:	93f6                	add	t2,t2,t4
1c00db90:	92de                	add	t0,t0,s7
1c00db92:	fbec13e3          	bne	s8,t5,1c00db38 <mm_M_unroll_1x2+0x280>
1c00db96:	b575                	j	1c00da42 <mm_M_unroll_1x2+0x18a>
1c00db98:	00000593          	li	a1,0
1c00db9c:	00d988b3          	add	a7,s3,a3
1c00dba0:	86c6                	mv	a3,a7
1c00dba2:	862e                	mv	a2,a1
1c00dba4:	00c36d2b          	p.sw	a2,s10(t1!)
1c00dba8:	00beed2b          	p.sw	a1,s10(t4!)
1c00dbac:	187d                	addi	a6,a6,-1
1c00dbae:	d8081ce3          	bnez	a6,1c00d946 <mm_M_unroll_1x2+0x8e>
1c00dbb2:	bbe9                	j	1c00d98c <mm_M_unroll_1x2+0xd4>
1c00dbb4:	00000593          	li	a1,0
1c00dbb8:	01368833          	add	a6,a3,s3
1c00dbbc:	86c2                	mv	a3,a6
1c00dbbe:	862e                	mv	a2,a1
1c00dbc0:	00c8edab          	p.sw	a2,s11(a7!)
1c00dbc4:	00b36dab          	p.sw	a1,s11(t1!)
1c00dbc8:	157d                	addi	a0,a0,-1
1c00dbca:	ec0514e3          	bnez	a0,1c00da92 <mm_M_unroll_1x2+0x1da>
1c00dbce:	b721                	j	1c00dad6 <mm_M_unroll_1x2+0x21e>
1c00dbd0:	5472                	lw	s0,60(sp)
1c00dbd2:	54e2                	lw	s1,56(sp)
1c00dbd4:	5952                	lw	s2,52(sp)
1c00dbd6:	59c2                	lw	s3,48(sp)
1c00dbd8:	5a32                	lw	s4,44(sp)
1c00dbda:	5aa2                	lw	s5,40(sp)
1c00dbdc:	5b12                	lw	s6,36(sp)
1c00dbde:	5b82                	lw	s7,32(sp)
1c00dbe0:	4c72                	lw	s8,28(sp)
1c00dbe2:	4ce2                	lw	s9,24(sp)
1c00dbe4:	4d52                	lw	s10,20(sp)
1c00dbe6:	4dc2                	lw	s11,16(sp)
1c00dbe8:	6121                	addi	sp,sp,64
1c00dbea:	caefb06f          	j	1c009098 <mm_M>
	...

1c00dbf0 <mm_M_unroll_1x4>:
1c00dbf0:	7139                	addi	sp,sp,-64
1c00dbf2:	de22                	sw	s0,60(sp)
1c00dbf4:	4900                	lw	s0,16(a0)
1c00dbf6:	dc26                	sw	s1,56(sp)
1c00dbf8:	da4a                	sw	s2,52(sp)
1c00dbfa:	d84e                	sw	s3,48(sp)
1c00dbfc:	d652                	sw	s4,44(sp)
1c00dbfe:	d456                	sw	s5,40(sp)
1c00dc00:	d25a                	sw	s6,36(sp)
1c00dc02:	d05e                	sw	s7,32(sp)
1c00dc04:	ce62                	sw	s8,28(sp)
1c00dc06:	cc66                	sw	s9,24(sp)
1c00dc08:	ca6a                	sw	s10,20(sp)
1c00dc0a:	c86e                	sw	s11,16(sp)
1c00dc0c:	477d                	li	a4,31
1c00dc0e:	c20433b3          	p.bclr	t2,s0,1,0
1c00dc12:	f14026f3          	csrr	a3,mhartid
1c00dc16:	30775463          	ble	t2,a4,1c00df1e <mm_M_unroll_1x4+0x32e>
1c00dc1a:	00738713          	addi	a4,t2,7
1c00dc1e:	f456b6b3          	p.bclr	a3,a3,26,5
1c00dc22:	870d                	srai	a4,a4,0x3
1c00dc24:	02e68fb3          	mul	t6,a3,a4
1c00dc28:	4150                	lw	a2,4(a0)
1c00dc2a:	c236                	sw	a3,4(sp)
1c00dc2c:	4d14                	lw	a3,24(a0)
1c00dc2e:	407407b3          	sub	a5,s0,t2
1c00dc32:	c432                	sw	a2,8(sp)
1c00dc34:	c63e                	sw	a5,12(sp)
1c00dc36:	00052f03          	lw	t5,0(a0)
1c00dc3a:	00852903          	lw	s2,8(a0)
1c00dc3e:	4544                	lw	s1,12(a0)
1c00dc40:	977e                	add	a4,a4,t6
1c00dc42:	01452e83          	lw	t4,20(a0)
1c00dc46:	047742b3          	p.min	t0,a4,t2
1c00dc4a:	14069a63          	bnez	a3,1c00dd9e <mm_M_unroll_1x4+0x1ae>
1c00dc4e:	085fd663          	ble	t0,t6,1c00dcda <mm_M_unroll_1x4+0xea>
1c00dc52:	08905463          	blez	s1,1c00dcda <mm_M_unroll_1x4+0xea>
1c00dc56:	002f9b93          	slli	s7,t6,0x2
1c00dc5a:	01790cb3          	add	s9,s2,s7
1c00dc5e:	00241993          	slli	s3,s0,0x2
1c00dc62:	9bb2                	add	s7,s7,a2
1c00dc64:	002e9c13          	slli	s8,t4,0x2
1c00dc68:	867a                	mv	a2,t5
1c00dc6a:	8366                	mv	t1,s9
1c00dc6c:	8e26                	mv	t3,s1
1c00dc6e:	27d05563          	blez	t4,1c00ded8 <mm_M_unroll_1x4+0x2e8>
1c00dc72:	00000893          	li	a7,0
1c00dc76:	00cc0a33          	add	s4,s8,a2
1c00dc7a:	40ca06b3          	sub	a3,s4,a2
1c00dc7e:	16f1                	addi	a3,a3,-4
1c00dc80:	8846                	mv	a6,a7
1c00dc82:	8546                	mv	a0,a7
1c00dc84:	85c6                	mv	a1,a7
1c00dc86:	8289                	srli	a3,a3,0x2
1c00dc88:	87de                	mv	a5,s7
1c00dc8a:	0685                	addi	a3,a3,1
1c00dc8c:	0146c0fb          	lp.setup	x1,a3,1c00dcb4 <mm_M_unroll_1x4+0xc4>
1c00dc90:	0046270b          	p.lw	a4,4(a2!)
1c00dc94:	0007ad83          	lw	s11,0(a5)
1c00dc98:	0047ad03          	lw	s10,4(a5)
1c00dc9c:	0087ab03          	lw	s6,8(a5)
1c00dca0:	00c7aa83          	lw	s5,12(a5)
1c00dca4:	58edf5c3          	fmadd.s	a1,s11,a4,a1
1c00dca8:	50ed7543          	fmadd.s	a0,s10,a4,a0
1c00dcac:	80eb7843          	fmadd.s	a6,s6,a4,a6
1c00dcb0:	88eaf8c3          	fmadd.s	a7,s5,a4,a7
1c00dcb4:	97ce                	add	a5,a5,s3
1c00dcb6:	00b32023          	sw	a1,0(t1)
1c00dcba:	00a32223          	sw	a0,4(t1)
1c00dcbe:	01032423          	sw	a6,8(t1)
1c00dcc2:	01132623          	sw	a7,12(t1)
1c00dcc6:	8652                	mv	a2,s4
1c00dcc8:	934e                	add	t1,t1,s3
1c00dcca:	1e7d                	addi	t3,t3,-1
1c00dccc:	fa0e11e3          	bnez	t3,1c00dc6e <mm_M_unroll_1x4+0x7e>
1c00dcd0:	0f91                	addi	t6,t6,4
1c00dcd2:	0cc1                	addi	s9,s9,16
1c00dcd4:	0bc1                	addi	s7,s7,16
1c00dcd6:	f85fc9e3          	blt	t6,t0,1c00dc68 <mm_M_unroll_1x4+0x78>
1c00dcda:	46b2                	lw	a3,12(sp)
1c00dcdc:	0ad05363          	blez	a3,1c00dd82 <mm_M_unroll_1x4+0x192>
1c00dce0:	00748713          	addi	a4,s1,7
1c00dce4:	41f75793          	srai	a5,a4,0x1f
1c00dce8:	f837b7b3          	p.bclr	a5,a5,28,3
1c00dcec:	06e7a7db          	p.addn	a5,a5,a4,3
1c00dcf0:	4712                	lw	a4,4(sp)
1c00dcf2:	02f70333          	mul	t1,a4,a5
1c00dcf6:	979a                	add	a5,a5,t1
1c00dcf8:	0497c4b3          	p.min	s1,a5,s1
1c00dcfc:	08935363          	ble	s1,t1,1c00dd82 <mm_M_unroll_1x4+0x192>
1c00dd00:	0883d163          	ble	s0,t2,1c00dd82 <mm_M_unroll_1x4+0x192>
1c00dd04:	88f6                	mv	a7,t4
1c00dd06:	426e88b3          	p.mac	a7,t4,t1
1c00dd0a:	002e9293          	slli	t0,t4,0x2
1c00dd0e:	47a2                	lw	a5,8(sp)
1c00dd10:	00239f93          	slli	t6,t2,0x2
1c00dd14:	00241993          	slli	s3,s0,0x2
1c00dd18:	9fbe                	add	t6,t6,a5
1c00dd1a:	01378a33          	add	s4,a5,s3
1c00dd1e:	088a                	slli	a7,a7,0x2
1c00dd20:	98fa                	add	a7,a7,t5
1c00dd22:	02830e33          	mul	t3,t1,s0
1c00dd26:	42628f33          	p.mac	t5,t0,t1
1c00dd2a:	41fa0833          	sub	a6,s4,t6
1c00dd2e:	007e0ab3          	add	s5,t3,t2
1c00dd32:	1871                	addi	a6,a6,-4
1c00dd34:	0a8a                	slli	s5,s5,0x2
1c00dd36:	00285813          	srli	a6,a6,0x2
1c00dd3a:	9aca                	add	s5,s5,s2
1c00dd3c:	8b7e                	mv	s6,t6
1c00dd3e:	0805                	addi	a6,a6,1
1c00dd40:	01a8407b          	lp.setup	x0,a6,1c00dd74 <mm_M_unroll_1x4+0x184>
1c00dd44:	00000713          	li	a4,0
1c00dd48:	03d05463          	blez	t4,1c00dd70 <mm_M_unroll_1x4+0x180>
1c00dd4c:	41e887b3          	sub	a5,a7,t5
1c00dd50:	17f1                	addi	a5,a5,-4
1c00dd52:	00000713          	li	a4,0
1c00dd56:	8389                	srli	a5,a5,0x2
1c00dd58:	865a                	mv	a2,s6
1c00dd5a:	86fa                	mv	a3,t5
1c00dd5c:	0785                	addi	a5,a5,1
1c00dd5e:	0001                	nop
1c00dd60:	0067c0fb          	lp.setup	x1,a5,1c00dd6c <mm_M_unroll_1x4+0x17c>
1c00dd64:	0046a50b          	p.lw	a0,4(a3!)
1c00dd68:	2136758b          	p.lw	a1,s3(a2!)
1c00dd6c:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00dd70:	00eaa22b          	p.sw	a4,4(s5!)
1c00dd74:	0b11                	addi	s6,s6,4
1c00dd76:	0305                	addi	t1,t1,1
1c00dd78:	9e22                	add	t3,t3,s0
1c00dd7a:	9f16                	add	t5,t5,t0
1c00dd7c:	9896                	add	a7,a7,t0
1c00dd7e:	fa6496e3          	bne	s1,t1,1c00dd2a <mm_M_unroll_1x4+0x13a>
1c00dd82:	5472                	lw	s0,60(sp)
1c00dd84:	54e2                	lw	s1,56(sp)
1c00dd86:	5952                	lw	s2,52(sp)
1c00dd88:	59c2                	lw	s3,48(sp)
1c00dd8a:	5a32                	lw	s4,44(sp)
1c00dd8c:	5aa2                	lw	s5,40(sp)
1c00dd8e:	5b12                	lw	s6,36(sp)
1c00dd90:	5b82                	lw	s7,32(sp)
1c00dd92:	4c72                	lw	s8,28(sp)
1c00dd94:	4ce2                	lw	s9,24(sp)
1c00dd96:	4d52                	lw	s10,20(sp)
1c00dd98:	4dc2                	lw	s11,16(sp)
1c00dd9a:	6121                	addi	sp,sp,64
1c00dd9c:	8082                	ret
1c00dd9e:	085fd263          	ble	t0,t6,1c00de22 <mm_M_unroll_1x4+0x232>
1c00dda2:	08905063          	blez	s1,1c00de22 <mm_M_unroll_1x4+0x232>
1c00dda6:	8b76                	mv	s6,t4
1c00dda8:	43fe8b33          	p.mac	s6,t4,t6
1c00ddac:	002e9c13          	slli	s8,t4,0x2
1c00ddb0:	8bb2                	mv	s7,a2
1c00ddb2:	002f9c93          	slli	s9,t6,0x2
1c00ddb6:	9cca                	add	s9,s9,s2
1c00ddb8:	003e9d13          	slli	s10,t4,0x3
1c00ddbc:	00241d93          	slli	s11,s0,0x2
1c00ddc0:	438f8bb3          	p.mac	s7,t6,s8
1c00ddc4:	0b0a                	slli	s6,s6,0x2
1c00ddc6:	9b32                	add	s6,s6,a2
1c00ddc8:	86fa                	mv	a3,t5
1c00ddca:	004c8313          	addi	t1,s9,4
1c00ddce:	88e6                	mv	a7,s9
1c00ddd0:	8526                	mv	a0,s1
1c00ddd2:	13d05863          	blez	t4,1c00df02 <mm_M_unroll_1x4+0x312>
1c00ddd6:	00000593          	li	a1,0
1c00ddda:	01868833          	add	a6,a3,s8
1c00ddde:	40d807b3          	sub	a5,a6,a3
1c00dde2:	17f1                	addi	a5,a5,-4
1c00dde4:	862e                	mv	a2,a1
1c00dde6:	8389                	srli	a5,a5,0x2
1c00dde8:	89da                	mv	s3,s6
1c00ddea:	8e5e                	mv	t3,s7
1c00ddec:	0785                	addi	a5,a5,1
1c00ddee:	0001                	nop
1c00ddf0:	00a7c0fb          	lp.setup	x1,a5,1c00de04 <mm_M_unroll_1x4+0x214>
1c00ddf4:	0046a70b          	p.lw	a4,4(a3!)
1c00ddf8:	004e2a8b          	p.lw	s5,4(t3!)
1c00ddfc:	0049aa0b          	p.lw	s4,4(s3!)
1c00de00:	60eaf643          	fmadd.s	a2,s5,a4,a2
1c00de04:	58ea75c3          	fmadd.s	a1,s4,a4,a1
1c00de08:	00c8edab          	p.sw	a2,s11(a7!)
1c00de0c:	00b36dab          	p.sw	a1,s11(t1!)
1c00de10:	86c2                	mv	a3,a6
1c00de12:	157d                	addi	a0,a0,-1
1c00de14:	fd5d                	bnez	a0,1c00ddd2 <mm_M_unroll_1x4+0x1e2>
1c00de16:	0f89                	addi	t6,t6,2
1c00de18:	0ca1                	addi	s9,s9,8
1c00de1a:	9bea                	add	s7,s7,s10
1c00de1c:	9b6a                	add	s6,s6,s10
1c00de1e:	fa5fc5e3          	blt	t6,t0,1c00ddc8 <mm_M_unroll_1x4+0x1d8>
1c00de22:	4632                	lw	a2,12(sp)
1c00de24:	f4c05fe3          	blez	a2,1c00dd82 <mm_M_unroll_1x4+0x192>
1c00de28:	00748713          	addi	a4,s1,7
1c00de2c:	41f75793          	srai	a5,a4,0x1f
1c00de30:	4692                	lw	a3,4(sp)
1c00de32:	f837b7b3          	p.bclr	a5,a5,28,3
1c00de36:	06e7a7db          	p.addn	a5,a5,a4,3
1c00de3a:	02f68fb3          	mul	t6,a3,a5
1c00de3e:	97fe                	add	a5,a5,t6
1c00de40:	0497c9b3          	p.min	s3,a5,s1
1c00de44:	f33fdfe3          	ble	s3,t6,1c00dd82 <mm_M_unroll_1x4+0x192>
1c00de48:	f283dde3          	ble	s0,t2,1c00dd82 <mm_M_unroll_1x4+0x192>
1c00de4c:	028f87b3          	mul	a5,t6,s0
1c00de50:	8776                	mv	a4,t4
1c00de52:	4aa2                	lw	s5,8(sp)
1c00de54:	002e9e13          	slli	t3,t4,0x2
1c00de58:	837a                	mv	t1,t5
1c00de5a:	00241a13          	slli	s4,s0,0x2
1c00de5e:	43fe8733          	p.mac	a4,t4,t6
1c00de62:	008784b3          	add	s1,a5,s0
1c00de66:	00249293          	slli	t0,s1,0x2
1c00de6a:	92ca                	add	t0,t0,s2
1c00de6c:	43fe0333          	p.mac	t1,t3,t6
1c00de70:	070a                	slli	a4,a4,0x2
1c00de72:	9f3a                	add	t5,t5,a4
1c00de74:	43c38ab3          	p.mac	s5,t2,t3
1c00de78:	007788b3          	add	a7,a5,t2
1c00de7c:	088a                	slli	a7,a7,0x2
1c00de7e:	98ca                	add	a7,a7,s2
1c00de80:	41128833          	sub	a6,t0,a7
1c00de84:	1871                	addi	a6,a6,-4
1c00de86:	00285813          	srli	a6,a6,0x2
1c00de8a:	8b56                	mv	s6,s5
1c00de8c:	0805                	addi	a6,a6,1
1c00de8e:	0001                	nop
1c00de90:	01a8407b          	lp.setup	x0,a6,1c00dec4 <mm_M_unroll_1x4+0x2d4>
1c00de94:	00000713          	li	a4,0
1c00de98:	03d05463          	blez	t4,1c00dec0 <mm_M_unroll_1x4+0x2d0>
1c00de9c:	406f07b3          	sub	a5,t5,t1
1c00dea0:	17f1                	addi	a5,a5,-4
1c00dea2:	00000713          	li	a4,0
1c00dea6:	8389                	srli	a5,a5,0x2
1c00dea8:	865a                	mv	a2,s6
1c00deaa:	869a                	mv	a3,t1
1c00deac:	0785                	addi	a5,a5,1
1c00deae:	0001                	nop
1c00deb0:	0067c0fb          	lp.setup	x1,a5,1c00debc <mm_M_unroll_1x4+0x2cc>
1c00deb4:	0046a50b          	p.lw	a0,4(a3!)
1c00deb8:	0046258b          	p.lw	a1,4(a2!)
1c00debc:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00dec0:	00e8a22b          	p.sw	a4,4(a7!)
1c00dec4:	9b72                	add	s6,s6,t3
1c00dec6:	0f85                	addi	t6,t6,1
1c00dec8:	87a6                	mv	a5,s1
1c00deca:	92d2                	add	t0,t0,s4
1c00decc:	9372                	add	t1,t1,t3
1c00dece:	9f72                	add	t5,t5,t3
1c00ded0:	94a2                	add	s1,s1,s0
1c00ded2:	fbf993e3          	bne	s3,t6,1c00de78 <mm_M_unroll_1x4+0x288>
1c00ded6:	b575                	j	1c00dd82 <mm_M_unroll_1x4+0x192>
1c00ded8:	00000893          	li	a7,0
1c00dedc:	00cc0a33          	add	s4,s8,a2
1c00dee0:	8652                	mv	a2,s4
1c00dee2:	8846                	mv	a6,a7
1c00dee4:	8546                	mv	a0,a7
1c00dee6:	85c6                	mv	a1,a7
1c00dee8:	01032423          	sw	a6,8(t1)
1c00deec:	00a32223          	sw	a0,4(t1)
1c00def0:	00b32023          	sw	a1,0(t1)
1c00def4:	01132623          	sw	a7,12(t1)
1c00def8:	934e                	add	t1,t1,s3
1c00defa:	1e7d                	addi	t3,t3,-1
1c00defc:	d60e19e3          	bnez	t3,1c00dc6e <mm_M_unroll_1x4+0x7e>
1c00df00:	bbc1                	j	1c00dcd0 <mm_M_unroll_1x4+0xe0>
1c00df02:	00000593          	li	a1,0
1c00df06:	01868833          	add	a6,a3,s8
1c00df0a:	86c2                	mv	a3,a6
1c00df0c:	862e                	mv	a2,a1
1c00df0e:	00c8edab          	p.sw	a2,s11(a7!)
1c00df12:	00b36dab          	p.sw	a1,s11(t1!)
1c00df16:	157d                	addi	a0,a0,-1
1c00df18:	ea051de3          	bnez	a0,1c00ddd2 <mm_M_unroll_1x4+0x1e2>
1c00df1c:	bded                	j	1c00de16 <mm_M_unroll_1x4+0x226>
1c00df1e:	5472                	lw	s0,60(sp)
1c00df20:	54e2                	lw	s1,56(sp)
1c00df22:	5952                	lw	s2,52(sp)
1c00df24:	59c2                	lw	s3,48(sp)
1c00df26:	5a32                	lw	s4,44(sp)
1c00df28:	5aa2                	lw	s5,40(sp)
1c00df2a:	5b12                	lw	s6,36(sp)
1c00df2c:	5b82                	lw	s7,32(sp)
1c00df2e:	4c72                	lw	s8,28(sp)
1c00df30:	4ce2                	lw	s9,24(sp)
1c00df32:	4d52                	lw	s10,20(sp)
1c00df34:	4dc2                	lw	s11,16(sp)
1c00df36:	6121                	addi	sp,sp,64
1c00df38:	b241                	j	1c00d8b8 <mm_M_unroll_1x2>
	...

1c00df3c <mm_M_unroll_1x8>:
1c00df3c:	01052303          	lw	t1,16(a0)
1c00df40:	7135                	addi	sp,sp,-160
1c00df42:	cf22                	sw	s0,156(sp)
1c00df44:	c40335b3          	p.bclr	a1,t1,2,0
1c00df48:	cd26                	sw	s1,152(sp)
1c00df4a:	cb4a                	sw	s2,148(sp)
1c00df4c:	c94e                	sw	s3,144(sp)
1c00df4e:	c752                	sw	s4,140(sp)
1c00df50:	c556                	sw	s5,136(sp)
1c00df52:	c35a                	sw	s6,132(sp)
1c00df54:	c15e                	sw	s7,128(sp)
1c00df56:	dee2                	sw	s8,124(sp)
1c00df58:	dce6                	sw	s9,120(sp)
1c00df5a:	daea                	sw	s10,116(sp)
1c00df5c:	d8ee                	sw	s11,112(sp)
1c00df5e:	c89a                	sw	t1,80(sp)
1c00df60:	03f00713          	li	a4,63
1c00df64:	c6ae                	sw	a1,76(sp)
1c00df66:	f14026f3          	csrr	a3,mhartid
1c00df6a:	48b75c63          	ble	a1,a4,1c00e402 <mm_M_unroll_1x8+0x4c6>
1c00df6e:	4035d713          	srai	a4,a1,0x3
1c00df72:	f456b6b3          	p.bclr	a3,a3,26,5
1c00df76:	d2b6                	sw	a3,100(sp)
1c00df78:	02e686b3          	mul	a3,a3,a4
1c00df7c:	862e                	mv	a2,a1
1c00df7e:	40b30333          	sub	t1,t1,a1
1c00df82:	00052803          	lw	a6,0(a0)
1c00df86:	00452883          	lw	a7,4(a0)
1c00df8a:	00852983          	lw	s3,8(a0)
1c00df8e:	00c52e03          	lw	t3,12(a0)
1c00df92:	495c                	lw	a5,20(a0)
1c00df94:	ccc2                	sw	a6,88(sp)
1c00df96:	cec6                	sw	a7,92(sp)
1c00df98:	c2b6                	sw	a3,68(sp)
1c00df9a:	4596                	lw	a1,68(sp)
1c00df9c:	4d14                	lw	a3,24(a0)
1c00df9e:	d4ce                	sw	s3,104(sp)
1c00dfa0:	972e                	add	a4,a4,a1
1c00dfa2:	04c74733          	p.min	a4,a4,a2
1c00dfa6:	caf2                	sw	t3,84(sp)
1c00dfa8:	ce3e                	sw	a5,28(sp)
1c00dfaa:	d69a                	sw	t1,108(sp)
1c00dfac:	d0ba                	sw	a4,96(sp)
1c00dfae:	26068e63          	beqz	a3,1c00e22a <mm_M_unroll_1x8+0x2ee>
1c00dfb2:	18e5d963          	ble	a4,a1,1c00e144 <mm_M_unroll_1x8+0x208>
1c00dfb6:	00279713          	slli	a4,a5,0x2
1c00dfba:	00179613          	slli	a2,a5,0x1
1c00dfbe:	883e                	mv	a6,a5
1c00dfc0:	00f606b3          	add	a3,a2,a5
1c00dfc4:	de3a                	sw	a4,60(sp)
1c00dfc6:	078e                	slli	a5,a5,0x3
1c00dfc8:	01070533          	add	a0,a4,a6
1c00dfcc:	00169593          	slli	a1,a3,0x1
1c00dfd0:	41078733          	sub	a4,a5,a6
1c00dfd4:	17c05863          	blez	t3,1c00e144 <mm_M_unroll_1x8+0x208>
1c00dfd8:	4896                	lw	a7,68(sp)
1c00dfda:	8342                	mv	t1,a6
1c00dfdc:	4e46                	lw	t3,80(sp)
1c00dfde:	031807b3          	mul	a5,a6,a7
1c00dfe2:	00289813          	slli	a6,a7,0x2
1c00dfe6:	984e                	add	a6,a6,s3
1c00dfe8:	49f2                	lw	s3,28(sp)
1c00dfea:	0e0a                	slli	t3,t3,0x2
1c00dfec:	c0f2                	sw	t3,64(sp)
1c00dfee:	0996                	slli	s3,s3,0x5
1c00dff0:	ca4e                	sw	s3,20(sp)
1c00dff2:	4e76                	lw	t3,92(sp)
1c00dff4:	59f2                	lw	s3,60(sp)
1c00dff6:	00f308b3          	add	a7,t1,a5
1c00dffa:	4316                	lw	t1,68(sp)
1c00dffc:	96be                	add	a3,a3,a5
1c00dffe:	95be                	add	a1,a1,a5
1c00e000:	42698e33          	p.mac	t3,s3,t1
1c00e004:	963e                	add	a2,a2,a5
1c00e006:	953e                	add	a0,a0,a5
1c00e008:	c4c2                	sw	a6,72(sp)
1c00e00a:	00f98833          	add	a6,s3,a5
1c00e00e:	97ba                	add	a5,a5,a4
1c00e010:	00269713          	slli	a4,a3,0x2
1c00e014:	00259693          	slli	a3,a1,0x2
1c00e018:	45f6                	lw	a1,92(sp)
1c00e01a:	088a                	slli	a7,a7,0x2
1c00e01c:	060a                	slli	a2,a2,0x2
1c00e01e:	080a                	slli	a6,a6,0x2
1c00e020:	050a                	slli	a0,a0,0x2
1c00e022:	078a                	slli	a5,a5,0x2
1c00e024:	98ae                	add	a7,a7,a1
1c00e026:	962e                	add	a2,a2,a1
1c00e028:	972e                	add	a4,a4,a1
1c00e02a:	982e                	add	a6,a6,a1
1c00e02c:	952e                	add	a0,a0,a1
1c00e02e:	96ae                	add	a3,a3,a1
1c00e030:	97ae                	add	a5,a5,a1
1c00e032:	dc72                	sw	t3,56(sp)
1c00e034:	da46                	sw	a7,52(sp)
1c00e036:	d832                	sw	a2,48(sp)
1c00e038:	d63a                	sw	a4,44(sp)
1c00e03a:	d442                	sw	a6,40(sp)
1c00e03c:	d22a                	sw	a0,36(sp)
1c00e03e:	d036                	sw	a3,32(sp)
1c00e040:	cc3e                	sw	a5,24(sp)
1c00e042:	4626                	lw	a2,72(sp)
1c00e044:	4756                	lw	a4,84(sp)
1c00e046:	46e6                	lw	a3,88(sp)
1c00e048:	c432                	sw	a2,8(sp)
1c00e04a:	c63a                	sw	a4,12(sp)
1c00e04c:	48f2                	lw	a7,28(sp)
1c00e04e:	39105d63          	blez	a7,1c00e3e8 <mm_M_unroll_1x8+0x4ac>
1c00e052:	5e72                	lw	t3,60(sp)
1c00e054:	00000e93          	li	t4,0
1c00e058:	49e2                	lw	s3,24(sp)
1c00e05a:	9e36                	add	t3,t3,a3
1c00e05c:	40de0733          	sub	a4,t3,a3
1c00e060:	c872                	sw	t3,16(sp)
1c00e062:	1771                	addi	a4,a4,-4
1c00e064:	8e76                	mv	t3,t4
1c00e066:	8376                	mv	t1,t4
1c00e068:	88f6                	mv	a7,t4
1c00e06a:	8876                	mv	a6,t4
1c00e06c:	8576                	mv	a0,t4
1c00e06e:	85f6                	mv	a1,t4
1c00e070:	8676                	mv	a2,t4
1c00e072:	5902                	lw	s2,32(sp)
1c00e074:	5492                	lw	s1,36(sp)
1c00e076:	5422                	lw	s0,40(sp)
1c00e078:	53b2                	lw	t2,44(sp)
1c00e07a:	52c2                	lw	t0,48(sp)
1c00e07c:	5fd2                	lw	t6,52(sp)
1c00e07e:	5f62                	lw	t5,56(sp)
1c00e080:	8309                	srli	a4,a4,0x2
1c00e082:	0705                	addi	a4,a4,1
1c00e084:	022740fb          	lp.setup	x1,a4,1c00e0c8 <mm_M_unroll_1x8+0x18c>
1c00e088:	0046a78b          	p.lw	a5,4(a3!)
1c00e08c:	004f2d8b          	p.lw	s11,4(t5!)
1c00e090:	004fad0b          	p.lw	s10,4(t6!)
1c00e094:	0042ac8b          	p.lw	s9,4(t0!)
1c00e098:	0043ac0b          	p.lw	s8,4(t2!)
1c00e09c:	00442b8b          	p.lw	s7,4(s0!)
1c00e0a0:	0044ab0b          	p.lw	s6,4(s1!)
1c00e0a4:	00492a8b          	p.lw	s5,4(s2!)
1c00e0a8:	0049aa0b          	p.lw	s4,4(s3!)
1c00e0ac:	60fdf643          	fmadd.s	a2,s11,a5,a2
1c00e0b0:	58fd75c3          	fmadd.s	a1,s10,a5,a1
1c00e0b4:	50fcf543          	fmadd.s	a0,s9,a5,a0
1c00e0b8:	80fc7843          	fmadd.s	a6,s8,a5,a6
1c00e0bc:	88fbf8c3          	fmadd.s	a7,s7,a5,a7
1c00e0c0:	30fb7343          	fmadd.s	t1,s6,a5,t1
1c00e0c4:	e0fafe43          	fmadd.s	t3,s5,a5,t3
1c00e0c8:	e8fa7ec3          	fmadd.s	t4,s4,a5,t4
1c00e0cc:	46a2                	lw	a3,8(sp)
1c00e0ce:	4722                	lw	a4,8(sp)
1c00e0d0:	4786                	lw	a5,64(sp)
1c00e0d2:	0106a623          	sw	a6,12(a3)
1c00e0d6:	4832                	lw	a6,12(sp)
1c00e0d8:	973e                	add	a4,a4,a5
1c00e0da:	c290                	sw	a2,0(a3)
1c00e0dc:	187d                	addi	a6,a6,-1
1c00e0de:	c2cc                	sw	a1,4(a3)
1c00e0e0:	c688                	sw	a0,8(a3)
1c00e0e2:	0116a823          	sw	a7,16(a3)
1c00e0e6:	0066aa23          	sw	t1,20(a3)
1c00e0ea:	01c6ac23          	sw	t3,24(a3)
1c00e0ee:	01d6ae23          	sw	t4,28(a3)
1c00e0f2:	c43a                	sw	a4,8(sp)
1c00e0f4:	c642                	sw	a6,12(sp)
1c00e0f6:	46c2                	lw	a3,16(sp)
1c00e0f8:	f4081ae3          	bnez	a6,1c00e04c <mm_M_unroll_1x8+0x110>
1c00e0fc:	4e16                	lw	t3,68(sp)
1c00e0fe:	4652                	lw	a2,20(sp)
1c00e100:	4326                	lw	t1,72(sp)
1c00e102:	55e2                	lw	a1,56(sp)
1c00e104:	0e21                	addi	t3,t3,8
1c00e106:	c2f2                	sw	t3,68(sp)
1c00e108:	56d2                	lw	a3,52(sp)
1c00e10a:	5742                	lw	a4,48(sp)
1c00e10c:	57b2                	lw	a5,44(sp)
1c00e10e:	5822                	lw	a6,40(sp)
1c00e110:	5892                	lw	a7,36(sp)
1c00e112:	5982                	lw	s3,32(sp)
1c00e114:	4e62                	lw	t3,24(sp)
1c00e116:	02030313          	addi	t1,t1,32
1c00e11a:	95b2                	add	a1,a1,a2
1c00e11c:	c49a                	sw	t1,72(sp)
1c00e11e:	dc2e                	sw	a1,56(sp)
1c00e120:	5306                	lw	t1,96(sp)
1c00e122:	4596                	lw	a1,68(sp)
1c00e124:	96b2                	add	a3,a3,a2
1c00e126:	9732                	add	a4,a4,a2
1c00e128:	97b2                	add	a5,a5,a2
1c00e12a:	9832                	add	a6,a6,a2
1c00e12c:	98b2                	add	a7,a7,a2
1c00e12e:	99b2                	add	s3,s3,a2
1c00e130:	9e32                	add	t3,t3,a2
1c00e132:	da36                	sw	a3,52(sp)
1c00e134:	d83a                	sw	a4,48(sp)
1c00e136:	d63e                	sw	a5,44(sp)
1c00e138:	d442                	sw	a6,40(sp)
1c00e13a:	d246                	sw	a7,36(sp)
1c00e13c:	d04e                	sw	s3,32(sp)
1c00e13e:	cc72                	sw	t3,24(sp)
1c00e140:	f065c1e3          	blt	a1,t1,1c00e042 <mm_M_unroll_1x8+0x106>
1c00e144:	57b6                	lw	a5,108(sp)
1c00e146:	0cf05463          	blez	a5,1c00e20e <mm_M_unroll_1x8+0x2d2>
1c00e14a:	4856                	lw	a6,84(sp)
1c00e14c:	5896                	lw	a7,100(sp)
1c00e14e:	00780713          	addi	a4,a6,7
1c00e152:	41f75793          	srai	a5,a4,0x1f
1c00e156:	f837b7b3          	p.bclr	a5,a5,28,3
1c00e15a:	06e7a7db          	p.addn	a5,a5,a4,3
1c00e15e:	02f88eb3          	mul	t4,a7,a5
1c00e162:	97f6                	add	a5,a5,t4
1c00e164:	0507c2b3          	p.min	t0,a5,a6
1c00e168:	0a5ed363          	ble	t0,t4,1c00e20e <mm_M_unroll_1x8+0x2d2>
1c00e16c:	49c6                	lw	s3,80(sp)
1c00e16e:	4e36                	lw	t3,76(sp)
1c00e170:	093e5f63          	ble	s3,t3,1c00e20e <mm_M_unroll_1x8+0x2d2>
1c00e174:	4372                	lw	t1,28(sp)
1c00e176:	86f2                	mv	a3,t3
1c00e178:	033e87b3          	mul	a5,t4,s3
1c00e17c:	841a                	mv	s0,t1
1c00e17e:	00231e13          	slli	t3,t1,0x2
1c00e182:	44f6                	lw	s1,92(sp)
1c00e184:	55a6                	lw	a1,104(sp)
1c00e186:	4666                	lw	a2,88(sp)
1c00e188:	4972                	lw	s2,28(sp)
1c00e18a:	8a4e                	mv	s4,s3
1c00e18c:	00299393          	slli	t2,s3,0x2
1c00e190:	8ab6                	mv	s5,a3
1c00e192:	43d30433          	p.mac	s0,t1,t4
1c00e196:	4366                	lw	t1,88(sp)
1c00e198:	01378fb3          	add	t6,a5,s3
1c00e19c:	002f9f13          	slli	t5,t6,0x2
1c00e1a0:	9f2e                	add	t5,t5,a1
1c00e1a2:	89ae                	mv	s3,a1
1c00e1a4:	43de0333          	p.mac	t1,t3,t4
1c00e1a8:	040a                	slli	s0,s0,0x2
1c00e1aa:	9432                	add	s0,s0,a2
1c00e1ac:	43c684b3          	p.mac	s1,a3,t3
1c00e1b0:	015788b3          	add	a7,a5,s5
1c00e1b4:	088a                	slli	a7,a7,0x2
1c00e1b6:	98ce                	add	a7,a7,s3
1c00e1b8:	411f0833          	sub	a6,t5,a7
1c00e1bc:	1871                	addi	a6,a6,-4
1c00e1be:	00285813          	srli	a6,a6,0x2
1c00e1c2:	8b26                	mv	s6,s1
1c00e1c4:	0805                	addi	a6,a6,1
1c00e1c6:	0001                	nop
1c00e1c8:	01a8407b          	lp.setup	x0,a6,1c00e1fc <mm_M_unroll_1x8+0x2c0>
1c00e1cc:	00000713          	li	a4,0
1c00e1d0:	03205463          	blez	s2,1c00e1f8 <mm_M_unroll_1x8+0x2bc>
1c00e1d4:	406407b3          	sub	a5,s0,t1
1c00e1d8:	17f1                	addi	a5,a5,-4
1c00e1da:	00000713          	li	a4,0
1c00e1de:	8389                	srli	a5,a5,0x2
1c00e1e0:	865a                	mv	a2,s6
1c00e1e2:	869a                	mv	a3,t1
1c00e1e4:	0785                	addi	a5,a5,1
1c00e1e6:	0001                	nop
1c00e1e8:	0067c0fb          	lp.setup	x1,a5,1c00e1f4 <mm_M_unroll_1x8+0x2b8>
1c00e1ec:	0046a50b          	p.lw	a0,4(a3!)
1c00e1f0:	0046258b          	p.lw	a1,4(a2!)
1c00e1f4:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00e1f8:	00e8a22b          	p.sw	a4,4(a7!)
1c00e1fc:	9b72                	add	s6,s6,t3
1c00e1fe:	0e85                	addi	t4,t4,1
1c00e200:	87fe                	mv	a5,t6
1c00e202:	9f1e                	add	t5,t5,t2
1c00e204:	9372                	add	t1,t1,t3
1c00e206:	9472                	add	s0,s0,t3
1c00e208:	9fd2                	add	t6,t6,s4
1c00e20a:	fbd293e3          	bne	t0,t4,1c00e1b0 <mm_M_unroll_1x8+0x274>
1c00e20e:	447a                	lw	s0,156(sp)
1c00e210:	44ea                	lw	s1,152(sp)
1c00e212:	495a                	lw	s2,148(sp)
1c00e214:	49ca                	lw	s3,144(sp)
1c00e216:	4a3a                	lw	s4,140(sp)
1c00e218:	4aaa                	lw	s5,136(sp)
1c00e21a:	4b1a                	lw	s6,132(sp)
1c00e21c:	4b8a                	lw	s7,128(sp)
1c00e21e:	5c76                	lw	s8,124(sp)
1c00e220:	5ce6                	lw	s9,120(sp)
1c00e222:	5d56                	lw	s10,116(sp)
1c00e224:	5dc6                	lw	s11,112(sp)
1c00e226:	610d                	addi	sp,sp,160
1c00e228:	8082                	ret
1c00e22a:	86ba                	mv	a3,a4
1c00e22c:	0ee5d163          	ble	a4,a1,1c00e30e <mm_M_unroll_1x8+0x3d2>
1c00e230:	0dc05f63          	blez	t3,1c00e30e <mm_M_unroll_1x8+0x3d2>
1c00e234:	00259c93          	slli	s9,a1,0x2
1c00e238:	01998db3          	add	s11,s3,s9
1c00e23c:	49f2                	lw	s3,28(sp)
1c00e23e:	9cc6                	add	s9,s9,a7
1c00e240:	48c6                	lw	a7,80(sp)
1c00e242:	4fe6                	lw	t6,88(sp)
1c00e244:	84ee                	mv	s1,s11
1c00e246:	00289b13          	slli	s6,a7,0x2
1c00e24a:	00299d13          	slli	s10,s3,0x2
1c00e24e:	82f2                	mv	t0,t3
1c00e250:	83b6                	mv	t2,a3
1c00e252:	842e                	mv	s0,a1
1c00e254:	8dce                	mv	s11,s3
1c00e256:	867e                	mv	a2,t6
1c00e258:	8aa6                	mv	s5,s1
1c00e25a:	8b96                	mv	s7,t0
1c00e25c:	051bc07b          	lp.setup	x0,s7,1c00e2fe <mm_M_unroll_1x8+0x3c2>
1c00e260:	17b05863          	blez	s11,1c00e3d0 <mm_M_unroll_1x8+0x494>
1c00e264:	00000f13          	li	t5,0
1c00e268:	00cd0c33          	add	s8,s10,a2
1c00e26c:	40cc06b3          	sub	a3,s8,a2
1c00e270:	16f1                	addi	a3,a3,-4
1c00e272:	8efa                	mv	t4,t5
1c00e274:	8e7a                	mv	t3,t5
1c00e276:	837a                	mv	t1,t5
1c00e278:	88fa                	mv	a7,t5
1c00e27a:	887a                	mv	a6,t5
1c00e27c:	857a                	mv	a0,t5
1c00e27e:	85fa                	mv	a1,t5
1c00e280:	8289                	srli	a3,a3,0x2
1c00e282:	8922                	mv	s2,s0
1c00e284:	87e6                	mv	a5,s9
1c00e286:	841e                	mv	s0,t2
1c00e288:	0685                	addi	a3,a3,1
1c00e28a:	83ee                	mv	t2,s11
1c00e28c:	0236c0fb          	lp.setup	x1,a3,1c00e2d2 <mm_M_unroll_1x8+0x396>
1c00e290:	0046270b          	p.lw	a4,4(a2!)
1c00e294:	0087a983          	lw	s3,8(a5)
1c00e298:	0007ad83          	lw	s11,0(a5)
1c00e29c:	0047aa03          	lw	s4,4(a5)
1c00e2a0:	80e9f843          	fmadd.s	a6,s3,a4,a6
1c00e2a4:	00c7a983          	lw	s3,12(a5)
1c00e2a8:	58edf5c3          	fmadd.s	a1,s11,a4,a1
1c00e2ac:	50ea7543          	fmadd.s	a0,s4,a4,a0
1c00e2b0:	88e9f8c3          	fmadd.s	a7,s3,a4,a7
1c00e2b4:	0107a983          	lw	s3,16(a5)
1c00e2b8:	30e9f343          	fmadd.s	t1,s3,a4,t1
1c00e2bc:	0147a983          	lw	s3,20(a5)
1c00e2c0:	e0e9fe43          	fmadd.s	t3,s3,a4,t3
1c00e2c4:	0187a983          	lw	s3,24(a5)
1c00e2c8:	e8e9fec3          	fmadd.s	t4,s3,a4,t4
1c00e2cc:	01c7a983          	lw	s3,28(a5)
1c00e2d0:	97da                	add	a5,a5,s6
1c00e2d2:	f0e9ff43          	fmadd.s	t5,s3,a4,t5
1c00e2d6:	8d9e                	mv	s11,t2
1c00e2d8:	83a2                	mv	t2,s0
1c00e2da:	844a                	mv	s0,s2
1c00e2dc:	00baa023          	sw	a1,0(s5)
1c00e2e0:	00aaa223          	sw	a0,4(s5)
1c00e2e4:	010aa423          	sw	a6,8(s5)
1c00e2e8:	011aa623          	sw	a7,12(s5)
1c00e2ec:	006aa823          	sw	t1,16(s5)
1c00e2f0:	01caaa23          	sw	t3,20(s5)
1c00e2f4:	01daac23          	sw	t4,24(s5)
1c00e2f8:	01eaae23          	sw	t5,28(s5)
1c00e2fc:	8662                	mv	a2,s8
1c00e2fe:	9ada                	add	s5,s5,s6
1c00e300:	0421                	addi	s0,s0,8
1c00e302:	02048493          	addi	s1,s1,32
1c00e306:	020c8c93          	addi	s9,s9,32
1c00e30a:	f47446e3          	blt	s0,t2,1c00e256 <mm_M_unroll_1x8+0x31a>
1c00e30e:	5e36                	lw	t3,108(sp)
1c00e310:	efc05fe3          	blez	t3,1c00e20e <mm_M_unroll_1x8+0x2d2>
1c00e314:	4356                	lw	t1,84(sp)
1c00e316:	5596                	lw	a1,100(sp)
1c00e318:	4656                	lw	a2,84(sp)
1c00e31a:	00730713          	addi	a4,t1,7
1c00e31e:	41f75793          	srai	a5,a4,0x1f
1c00e322:	f837b7b3          	p.bclr	a5,a5,28,3
1c00e326:	06e7a7db          	p.addn	a5,a5,a4,3
1c00e32a:	02f58333          	mul	t1,a1,a5
1c00e32e:	979a                	add	a5,a5,t1
1c00e330:	04c7ce33          	p.min	t3,a5,a2
1c00e334:	edc35de3          	ble	t3,t1,1c00e20e <mm_M_unroll_1x8+0x2d2>
1c00e338:	46c6                	lw	a3,80(sp)
1c00e33a:	4736                	lw	a4,76(sp)
1c00e33c:	ecd759e3          	ble	a3,a4,1c00e20e <mm_M_unroll_1x8+0x2d2>
1c00e340:	47f2                	lw	a5,28(sp)
1c00e342:	49e6                	lw	s3,88(sp)
1c00e344:	02d30eb3          	mul	t4,t1,a3
1c00e348:	88be                	mv	a7,a5
1c00e34a:	844e                	mv	s0,s3
1c00e34c:	00279f93          	slli	t6,a5,0x2
1c00e350:	4876                	lw	a6,92(sp)
1c00e352:	5926                	lw	s2,104(sp)
1c00e354:	00271f13          	slli	t5,a4,0x2
1c00e358:	00269293          	slli	t0,a3,0x2
1c00e35c:	9f42                	add	t5,t5,a6
1c00e35e:	005803b3          	add	t2,a6,t0
1c00e362:	426788b3          	p.mac	a7,a5,t1
1c00e366:	84be                	mv	s1,a5
1c00e368:	8a3a                	mv	s4,a4
1c00e36a:	426f8433          	p.mac	s0,t6,t1
1c00e36e:	088a                	slli	a7,a7,0x2
1c00e370:	98ce                	add	a7,a7,s3
1c00e372:	89b6                	mv	s3,a3
1c00e374:	41e38833          	sub	a6,t2,t5
1c00e378:	014e8ab3          	add	s5,t4,s4
1c00e37c:	1871                	addi	a6,a6,-4
1c00e37e:	0a8a                	slli	s5,s5,0x2
1c00e380:	00285813          	srli	a6,a6,0x2
1c00e384:	9aca                	add	s5,s5,s2
1c00e386:	8b7a                	mv	s6,t5
1c00e388:	0805                	addi	a6,a6,1
1c00e38a:	0001                	nop
1c00e38c:	01a8407b          	lp.setup	x0,a6,1c00e3c0 <mm_M_unroll_1x8+0x484>
1c00e390:	00000713          	li	a4,0
1c00e394:	02905463          	blez	s1,1c00e3bc <mm_M_unroll_1x8+0x480>
1c00e398:	408887b3          	sub	a5,a7,s0
1c00e39c:	17f1                	addi	a5,a5,-4
1c00e39e:	00000713          	li	a4,0
1c00e3a2:	8389                	srli	a5,a5,0x2
1c00e3a4:	865a                	mv	a2,s6
1c00e3a6:	86a2                	mv	a3,s0
1c00e3a8:	0785                	addi	a5,a5,1
1c00e3aa:	0001                	nop
1c00e3ac:	0067c0fb          	lp.setup	x1,a5,1c00e3b8 <mm_M_unroll_1x8+0x47c>
1c00e3b0:	0046a50b          	p.lw	a0,4(a3!)
1c00e3b4:	2056758b          	p.lw	a1,t0(a2!)
1c00e3b8:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00e3bc:	00eaa22b          	p.sw	a4,4(s5!)
1c00e3c0:	0b11                	addi	s6,s6,4
1c00e3c2:	0305                	addi	t1,t1,1
1c00e3c4:	9ece                	add	t4,t4,s3
1c00e3c6:	947e                	add	s0,s0,t6
1c00e3c8:	98fe                	add	a7,a7,t6
1c00e3ca:	fa6e15e3          	bne	t3,t1,1c00e374 <mm_M_unroll_1x8+0x438>
1c00e3ce:	b581                	j	1c00e20e <mm_M_unroll_1x8+0x2d2>
1c00e3d0:	00000f13          	li	t5,0
1c00e3d4:	00cd0c33          	add	s8,s10,a2
1c00e3d8:	8efa                	mv	t4,t5
1c00e3da:	8e7a                	mv	t3,t5
1c00e3dc:	837a                	mv	t1,t5
1c00e3de:	88fa                	mv	a7,t5
1c00e3e0:	887a                	mv	a6,t5
1c00e3e2:	857a                	mv	a0,t5
1c00e3e4:	85fa                	mv	a1,t5
1c00e3e6:	bddd                	j	1c00e2dc <mm_M_unroll_1x8+0x3a0>
1c00e3e8:	59f2                	lw	s3,60(sp)
1c00e3ea:	00000e93          	li	t4,0
1c00e3ee:	96ce                	add	a3,a3,s3
1c00e3f0:	8e76                	mv	t3,t4
1c00e3f2:	8376                	mv	t1,t4
1c00e3f4:	88f6                	mv	a7,t4
1c00e3f6:	8876                	mv	a6,t4
1c00e3f8:	8576                	mv	a0,t4
1c00e3fa:	85f6                	mv	a1,t4
1c00e3fc:	8676                	mv	a2,t4
1c00e3fe:	c836                	sw	a3,16(sp)
1c00e400:	b1f1                	j	1c00e0cc <mm_M_unroll_1x8+0x190>
1c00e402:	447a                	lw	s0,156(sp)
1c00e404:	44ea                	lw	s1,152(sp)
1c00e406:	495a                	lw	s2,148(sp)
1c00e408:	49ca                	lw	s3,144(sp)
1c00e40a:	4a3a                	lw	s4,140(sp)
1c00e40c:	4aaa                	lw	s5,136(sp)
1c00e40e:	4b1a                	lw	s6,132(sp)
1c00e410:	4b8a                	lw	s7,128(sp)
1c00e412:	5c76                	lw	s8,124(sp)
1c00e414:	5ce6                	lw	s9,120(sp)
1c00e416:	5d56                	lw	s10,116(sp)
1c00e418:	5dc6                	lw	s11,112(sp)
1c00e41a:	610d                	addi	sp,sp,160
1c00e41c:	fd4ff06f          	j	1c00dbf0 <mm_M_unroll_1x4>
1c00e420:	0000                	unimp
	...

1c00e424 <mm_M_unroll_2x2>:
1c00e424:	490c                	lw	a1,16(a0)
1c00e426:	7119                	addi	sp,sp,-128
1c00e428:	dea2                	sw	s0,124(sp)
1c00e42a:	dca6                	sw	s1,120(sp)
1c00e42c:	daca                	sw	s2,116(sp)
1c00e42e:	d8ce                	sw	s3,112(sp)
1c00e430:	d6d2                	sw	s4,108(sp)
1c00e432:	d4d6                	sw	s5,104(sp)
1c00e434:	d2da                	sw	s6,100(sp)
1c00e436:	d0de                	sw	s7,96(sp)
1c00e438:	cee2                	sw	s8,92(sp)
1c00e43a:	cce6                	sw	s9,88(sp)
1c00e43c:	caea                	sw	s10,84(sp)
1c00e43e:	c8ee                	sw	s11,80(sp)
1c00e440:	ce2e                	sw	a1,28(sp)
1c00e442:	473d                	li	a4,15
1c00e444:	c005b6b3          	p.bclr	a3,a1,0,0
1c00e448:	f1402673          	csrr	a2,mhartid
1c00e44c:	48d75a63          	ble	a3,a4,1c00e8e0 <mm_M_unroll_2x2+0x4bc>
1c00e450:	00768713          	addi	a4,a3,7
1c00e454:	870d                	srai	a4,a4,0x3
1c00e456:	f4563633          	p.bclr	a2,a2,26,5
1c00e45a:	410c                	lw	a1,0(a0)
1c00e45c:	c2b2                	sw	a2,68(sp)
1c00e45e:	02e60633          	mul	a2,a2,a4
1c00e462:	cc2e                	sw	a1,24(sp)
1c00e464:	414c                	lw	a1,4(a0)
1c00e466:	47f2                	lw	a5,28(sp)
1c00e468:	01452283          	lw	t0,20(a0)
1c00e46c:	da2e                	sw	a1,52(sp)
1c00e46e:	450c                	lw	a1,8(a0)
1c00e470:	8f95                	sub	a5,a5,a3
1c00e472:	c4be                	sw	a5,72(sp)
1c00e474:	c0ae                	sw	a1,64(sp)
1c00e476:	454c                	lw	a1,12(a0)
1c00e478:	c632                	sw	a2,12(sp)
1c00e47a:	4d10                	lw	a2,24(a0)
1c00e47c:	d42e                	sw	a1,40(sp)
1c00e47e:	45b2                	lw	a1,12(sp)
1c00e480:	972e                	add	a4,a4,a1
1c00e482:	04d74733          	p.min	a4,a4,a3
1c00e486:	d03a                	sw	a4,32(sp)
1c00e488:	1c061163          	bnez	a2,1c00e64a <mm_M_unroll_2x2+0x226>
1c00e48c:	5782                	lw	a5,32(sp)
1c00e48e:	10f5d663          	ble	a5,a1,1c00e59a <mm_M_unroll_2x2+0x176>
1c00e492:	862e                	mv	a2,a1
1c00e494:	55a2                	lw	a1,40(sp)
1c00e496:	00261b93          	slli	s7,a2,0x2
1c00e49a:	00160b13          	addi	s6,a2,1
1c00e49e:	fff58793          	addi	a5,a1,-1
1c00e4a2:	5622                	lw	a2,40(sp)
1c00e4a4:	02f28733          	mul	a4,t0,a5
1c00e4a8:	46f2                	lw	a3,28(sp)
1c00e4aa:	fc163633          	p.bclr	a2,a2,30,1
1c00e4ae:	ca32                	sw	a2,20(sp)
1c00e4b0:	4672                	lw	a2,28(sp)
1c00e4b2:	c005b5b3          	p.bclr	a1,a1,0,0
1c00e4b6:	00269d13          	slli	s10,a3,0x2
1c00e4ba:	c82e                	sw	a1,16(sp)
1c00e4bc:	4686                	lw	a3,64(sp)
1c00e4be:	55d2                	lw	a1,52(sp)
1c00e4c0:	02c787b3          	mul	a5,a5,a2
1c00e4c4:	0b0a                	slli	s6,s6,0x2
1c00e4c6:	01768c33          	add	s8,a3,s7
1c00e4ca:	9b2e                	add	s6,s6,a1
1c00e4cc:	9bae                	add	s7,s7,a1
1c00e4ce:	45e2                	lw	a1,24(sp)
1c00e4d0:	00e286b3          	add	a3,t0,a4
1c00e4d4:	068a                	slli	a3,a3,0x2
1c00e4d6:	070a                	slli	a4,a4,0x2
1c00e4d8:	972e                	add	a4,a4,a1
1c00e4da:	96ae                	add	a3,a3,a1
1c00e4dc:	dc3a                	sw	a4,56(sp)
1c00e4de:	d836                	sw	a3,48(sp)
1c00e4e0:	4706                	lw	a4,64(sp)
1c00e4e2:	56d2                	lw	a3,52(sp)
1c00e4e4:	00361d93          	slli	s11,a2,0x3
1c00e4e8:	078a                	slli	a5,a5,0x2
1c00e4ea:	004d0613          	addi	a2,s10,4
1c00e4ee:	8e99                	sub	a3,a3,a4
1c00e4f0:	00329c93          	slli	s9,t0,0x3
1c00e4f4:	00229a93          	slli	s5,t0,0x2
1c00e4f8:	d232                	sw	a2,36(sp)
1c00e4fa:	d63e                	sw	a5,44(sp)
1c00e4fc:	de36                	sw	a3,60(sp)
1c00e4fe:	45c2                	lw	a1,16(sp)
1c00e500:	c1c9                	beqz	a1,1c00e582 <mm_M_unroll_2x2+0x15e>
1c00e502:	5612                	lw	a2,36(sp)
1c00e504:	fff58413          	addi	s0,a1,-1
1c00e508:	8005                	srli	s0,s0,0x1
1c00e50a:	00158793          	addi	a5,a1,1
1c00e50e:	4709                	li	a4,2
1c00e510:	018d09b3          	add	s3,s10,s8
1c00e514:	004c0913          	addi	s2,s8,4
1c00e518:	018604b3          	add	s1,a2,s8
1c00e51c:	43e2                	lw	t2,24(sp)
1c00e51e:	8a62                	mv	s4,s8
1c00e520:	0405                	addi	s0,s0,1
1c00e522:	3ee7e063          	bltu	a5,a4,1c00e902 <mm_M_unroll_2x2+0x4de>
1c00e526:	2e505163          	blez	t0,1c00e808 <mm_M_unroll_2x2+0x3e4>
1c00e52a:	00000893          	li	a7,0
1c00e52e:	ffca8793          	addi	a5,s5,-4
1c00e532:	8389                	srli	a5,a5,0x2
1c00e534:	8846                	mv	a6,a7
1c00e536:	8546                	mv	a0,a7
1c00e538:	85c6                	mv	a1,a7
1c00e53a:	8f5a                	mv	t5,s6
1c00e53c:	8e9e                	mv	t4,t2
1c00e53e:	8e5e                	mv	t3,s7
1c00e540:	007a8333          	add	t1,s5,t2
1c00e544:	0785                	addi	a5,a5,1
1c00e546:	0001                	nop
1c00e548:	0107c0fb          	lp.setup	x1,a5,1c00e568 <mm_M_unroll_2x2+0x144>
1c00e54c:	21ae7f8b          	p.lw	t6,s10(t3!)
1c00e550:	004ea60b          	p.lw	a2,4(t4!)
1c00e554:	0043268b          	p.lw	a3,4(t1!)
1c00e558:	21af770b          	p.lw	a4,s10(t5!)
1c00e55c:	58cff5c3          	fmadd.s	a1,t6,a2,a1
1c00e560:	50dff543          	fmadd.s	a0,t6,a3,a0
1c00e564:	80e67843          	fmadd.s	a6,a2,a4,a6
1c00e568:	88e6f8c3          	fmadd.s	a7,a3,a4,a7
1c00e56c:	00ba6dab          	p.sw	a1,s11(s4!)
1c00e570:	00a9edab          	p.sw	a0,s11(s3!)
1c00e574:	01096dab          	p.sw	a6,s11(s2!)
1c00e578:	0114edab          	p.sw	a7,s11(s1!)
1c00e57c:	93e6                	add	t2,t2,s9
1c00e57e:	147d                	addi	s0,s0,-1
1c00e580:	f05d                	bnez	s0,1c00e526 <mm_M_unroll_2x2+0x102>
1c00e582:	47d2                	lw	a5,20(sp)
1c00e584:	30079963          	bnez	a5,1c00e896 <mm_M_unroll_2x2+0x472>
1c00e588:	4732                	lw	a4,12(sp)
1c00e58a:	5782                	lw	a5,32(sp)
1c00e58c:	0c21                	addi	s8,s8,8
1c00e58e:	0709                	addi	a4,a4,2
1c00e590:	c63a                	sw	a4,12(sp)
1c00e592:	0ba1                	addi	s7,s7,8
1c00e594:	0b21                	addi	s6,s6,8
1c00e596:	f6f744e3          	blt	a4,a5,1c00e4fe <mm_M_unroll_2x2+0xda>
1c00e59a:	46a6                	lw	a3,72(sp)
1c00e59c:	08d05963          	blez	a3,1c00e62e <mm_M_unroll_2x2+0x20a>
1c00e5a0:	57a2                	lw	a5,40(sp)
1c00e5a2:	4596                	lw	a1,68(sp)
1c00e5a4:	5622                	lw	a2,40(sp)
1c00e5a6:	00778713          	addi	a4,a5,7
1c00e5aa:	41f75793          	srai	a5,a4,0x1f
1c00e5ae:	f837b7b3          	p.bclr	a5,a5,28,3
1c00e5b2:	06e7a7db          	p.addn	a5,a5,a4,3
1c00e5b6:	02f58733          	mul	a4,a1,a5
1c00e5ba:	00e78833          	add	a6,a5,a4
1c00e5be:	04c84833          	p.min	a6,a6,a2
1c00e5c2:	07075663          	ble	a6,a4,1c00e62e <mm_M_unroll_2x2+0x20a>
1c00e5c6:	46f2                	lw	a3,28(sp)
1c00e5c8:	00229e93          	slli	t4,t0,0x2
1c00e5cc:	55d2                	lw	a1,52(sp)
1c00e5ce:	fff68793          	addi	a5,a3,-1
1c00e5d2:	833e                	mv	t1,a5
1c00e5d4:	42e68333          	p.mac	t1,a3,a4
1c00e5d8:	00269f13          	slli	t5,a3,0x2
1c00e5dc:	46e2                	lw	a3,24(sp)
1c00e5de:	4606                	lw	a2,64(sp)
1c00e5e0:	078a                	slli	a5,a5,0x2
1c00e5e2:	40e80833          	sub	a6,a6,a4
1c00e5e6:	00f58e33          	add	t3,a1,a5
1c00e5ea:	43d706b3          	p.mac	a3,a4,t4
1c00e5ee:	030a                	slli	t1,t1,0x2
1c00e5f0:	9332                	add	t1,t1,a2
1c00e5f2:	8736                	mv	a4,a3
1c00e5f4:	01c8407b          	lp.setup	x0,a6,1c00e62c <mm_M_unroll_2x2+0x208>
1c00e5f8:	00000693          	li	a3,0
1c00e5fc:	00ee88b3          	add	a7,t4,a4
1c00e600:	02505463          	blez	t0,1c00e628 <mm_M_unroll_2x2+0x204>
1c00e604:	00ee88b3          	add	a7,t4,a4
1c00e608:	40e887b3          	sub	a5,a7,a4
1c00e60c:	17f1                	addi	a5,a5,-4
1c00e60e:	00000693          	li	a3,0
1c00e612:	8389                	srli	a5,a5,0x2
1c00e614:	8672                	mv	a2,t3
1c00e616:	0785                	addi	a5,a5,1
1c00e618:	0067c0fb          	lp.setup	x1,a5,1c00e624 <mm_M_unroll_2x2+0x200>
1c00e61c:	0047250b          	p.lw	a0,4(a4!)
1c00e620:	21e6758b          	p.lw	a1,t5(a2!)
1c00e624:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00e628:	00d36f2b          	p.sw	a3,t5(t1!)
1c00e62c:	8746                	mv	a4,a7
1c00e62e:	5476                	lw	s0,124(sp)
1c00e630:	54e6                	lw	s1,120(sp)
1c00e632:	5956                	lw	s2,116(sp)
1c00e634:	59c6                	lw	s3,112(sp)
1c00e636:	5a36                	lw	s4,108(sp)
1c00e638:	5aa6                	lw	s5,104(sp)
1c00e63a:	5b16                	lw	s6,100(sp)
1c00e63c:	5b86                	lw	s7,96(sp)
1c00e63e:	4c76                	lw	s8,92(sp)
1c00e640:	4ce6                	lw	s9,88(sp)
1c00e642:	4d56                	lw	s10,84(sp)
1c00e644:	4dc6                	lw	s11,80(sp)
1c00e646:	6109                	addi	sp,sp,128
1c00e648:	8082                	ret
1c00e64a:	12e5d263          	ble	a4,a1,1c00e76e <mm_M_unroll_2x2+0x34a>
1c00e64e:	5622                	lw	a2,40(sp)
1c00e650:	00259c93          	slli	s9,a1,0x2
1c00e654:	00158713          	addi	a4,a1,1
1c00e658:	45f2                	lw	a1,28(sp)
1c00e65a:	fff60793          	addi	a5,a2,-1
1c00e65e:	025786b3          	mul	a3,a5,t0
1c00e662:	058a                	slli	a1,a1,0x2
1c00e664:	d62e                	sw	a1,44(sp)
1c00e666:	55a2                	lw	a1,40(sp)
1c00e668:	c0063633          	p.bclr	a2,a2,0,0
1c00e66c:	c832                	sw	a2,16(sp)
1c00e66e:	fc15b5b3          	p.bclr	a1,a1,30,1
1c00e672:	d22e                	sw	a1,36(sp)
1c00e674:	45f2                	lw	a1,28(sp)
1c00e676:	4606                	lw	a2,64(sp)
1c00e678:	00229d13          	slli	s10,t0,0x2
1c00e67c:	00359d93          	slli	s11,a1,0x3
1c00e680:	45e2                	lw	a1,24(sp)
1c00e682:	9cb2                	add	s9,s9,a2
1c00e684:	00568633          	add	a2,a3,t0
1c00e688:	068a                	slli	a3,a3,0x2
1c00e68a:	96ae                	add	a3,a3,a1
1c00e68c:	060a                	slli	a2,a2,0x2
1c00e68e:	962e                	add	a2,a2,a1
1c00e690:	c6b6                	sw	a3,76(sp)
1c00e692:	45b2                	lw	a1,12(sp)
1c00e694:	56d2                	lw	a3,52(sp)
1c00e696:	de32                	sw	a2,60(sp)
1c00e698:	5632                	lw	a2,44(sp)
1c00e69a:	43a586b3          	p.mac	a3,a1,s10
1c00e69e:	8b96                	mv	s7,t0
1c00e6a0:	0611                	addi	a2,a2,4
1c00e6a2:	d832                	sw	a2,48(sp)
1c00e6a4:	5652                	lw	a2,52(sp)
1c00e6a6:	00329b13          	slli	s6,t0,0x3
1c00e6aa:	8c32                	mv	s8,a2
1c00e6ac:	ca36                	sw	a3,20(sp)
1c00e6ae:	46f2                	lw	a3,28(sp)
1c00e6b0:	42b28bb3          	p.mac	s7,t0,a1
1c00e6b4:	4ad2                	lw	s5,20(sp)
1c00e6b6:	02d787b3          	mul	a5,a5,a3
1c00e6ba:	0b8a                	slli	s7,s7,0x2
1c00e6bc:	9bb2                	add	s7,s7,a2
1c00e6be:	43a70c33          	p.mac	s8,a4,s10
1c00e6c2:	078a                	slli	a5,a5,0x2
1c00e6c4:	dc3e                	sw	a5,56(sp)
1c00e6c6:	4742                	lw	a4,16(sp)
1c00e6c8:	08e05363          	blez	a4,1c00e74e <mm_M_unroll_2x2+0x32a>
1c00e6cc:	55b2                	lw	a1,44(sp)
1c00e6ce:	5642                	lw	a2,48(sp)
1c00e6d0:	fff70413          	addi	s0,a4,-1
1c00e6d4:	8005                	srli	s0,s0,0x1
1c00e6d6:	00170793          	addi	a5,a4,1
1c00e6da:	4709                	li	a4,2
1c00e6dc:	019589b3          	add	s3,a1,s9
1c00e6e0:	004c8913          	addi	s2,s9,4
1c00e6e4:	019604b3          	add	s1,a2,s9
1c00e6e8:	43e2                	lw	t2,24(sp)
1c00e6ea:	8a66                	mv	s4,s9
1c00e6ec:	0405                	addi	s0,s0,1
1c00e6ee:	20e7c863          	blt	a5,a4,1c00e8fe <mm_M_unroll_2x2+0x4da>
1c00e6f2:	12505d63          	blez	t0,1c00e82c <mm_M_unroll_2x2+0x408>
1c00e6f6:	00000893          	li	a7,0
1c00e6fa:	415b87b3          	sub	a5,s7,s5
1c00e6fe:	17f1                	addi	a5,a5,-4
1c00e700:	8846                	mv	a6,a7
1c00e702:	8546                	mv	a0,a7
1c00e704:	85c6                	mv	a1,a7
1c00e706:	8389                	srli	a5,a5,0x2
1c00e708:	01a38f33          	add	t5,t2,s10
1c00e70c:	8ee2                	mv	t4,s8
1c00e70e:	8e1e                	mv	t3,t2
1c00e710:	8356                	mv	t1,s5
1c00e712:	0785                	addi	a5,a5,1
1c00e714:	0107c0fb          	lp.setup	x1,a5,1c00e734 <mm_M_unroll_2x2+0x310>
1c00e718:	00432f8b          	p.lw	t6,4(t1!)
1c00e71c:	004e260b          	p.lw	a2,4(t3!)
1c00e720:	004f268b          	p.lw	a3,4(t5!)
1c00e724:	004ea70b          	p.lw	a4,4(t4!)
1c00e728:	58cff5c3          	fmadd.s	a1,t6,a2,a1
1c00e72c:	50dff543          	fmadd.s	a0,t6,a3,a0
1c00e730:	80e67843          	fmadd.s	a6,a2,a4,a6
1c00e734:	88e6f8c3          	fmadd.s	a7,a3,a4,a7
1c00e738:	00ba6dab          	p.sw	a1,s11(s4!)
1c00e73c:	00a9edab          	p.sw	a0,s11(s3!)
1c00e740:	01096dab          	p.sw	a6,s11(s2!)
1c00e744:	0114edab          	p.sw	a7,s11(s1!)
1c00e748:	93da                	add	t2,t2,s6
1c00e74a:	147d                	addi	s0,s0,-1
1c00e74c:	f05d                	bnez	s0,1c00e6f2 <mm_M_unroll_2x2+0x2ce>
1c00e74e:	5712                	lw	a4,36(sp)
1c00e750:	10071063          	bnez	a4,1c00e850 <mm_M_unroll_2x2+0x42c>
1c00e754:	45b2                	lw	a1,12(sp)
1c00e756:	4652                	lw	a2,20(sp)
1c00e758:	5682                	lw	a3,32(sp)
1c00e75a:	0589                	addi	a1,a1,2
1c00e75c:	965a                	add	a2,a2,s6
1c00e75e:	c62e                	sw	a1,12(sp)
1c00e760:	ca32                	sw	a2,20(sp)
1c00e762:	0ca1                	addi	s9,s9,8
1c00e764:	9ada                	add	s5,s5,s6
1c00e766:	9c5a                	add	s8,s8,s6
1c00e768:	9bda                	add	s7,s7,s6
1c00e76a:	f4d5cee3          	blt	a1,a3,1c00e6c6 <mm_M_unroll_2x2+0x2a2>
1c00e76e:	46a6                	lw	a3,72(sp)
1c00e770:	ead05fe3          	blez	a3,1c00e62e <mm_M_unroll_2x2+0x20a>
1c00e774:	57a2                	lw	a5,40(sp)
1c00e776:	4596                	lw	a1,68(sp)
1c00e778:	5622                	lw	a2,40(sp)
1c00e77a:	00778713          	addi	a4,a5,7
1c00e77e:	41f75793          	srai	a5,a4,0x1f
1c00e782:	f837b7b3          	p.bclr	a5,a5,28,3
1c00e786:	06e7a7db          	p.addn	a5,a5,a4,3
1c00e78a:	02f58733          	mul	a4,a1,a5
1c00e78e:	00e78833          	add	a6,a5,a4
1c00e792:	04c84833          	p.min	a6,a6,a2
1c00e796:	e9075ce3          	ble	a6,a4,1c00e62e <mm_M_unroll_2x2+0x20a>
1c00e79a:	46f2                	lw	a3,28(sp)
1c00e79c:	45e2                	lw	a1,24(sp)
1c00e79e:	00229f13          	slli	t5,t0,0x2
1c00e7a2:	fff68793          	addi	a5,a3,-1
1c00e7a6:	833e                	mv	t1,a5
1c00e7a8:	42e68333          	p.mac	t1,a3,a4
1c00e7ac:	5652                	lw	a2,52(sp)
1c00e7ae:	40e80833          	sub	a6,a6,a4
1c00e7b2:	00269893          	slli	a7,a3,0x2
1c00e7b6:	02578e33          	mul	t3,a5,t0
1c00e7ba:	4786                	lw	a5,64(sp)
1c00e7bc:	030a                	slli	t1,t1,0x2
1c00e7be:	933e                	add	t1,t1,a5
1c00e7c0:	43e705b3          	p.mac	a1,a4,t5
1c00e7c4:	0e0a                	slli	t3,t3,0x2
1c00e7c6:	9e32                	add	t3,t3,a2
1c00e7c8:	872e                	mv	a4,a1
1c00e7ca:	0001                	nop
1c00e7cc:	01c8407b          	lp.setup	x0,a6,1c00e804 <mm_M_unroll_2x2+0x3e0>
1c00e7d0:	00000693          	li	a3,0
1c00e7d4:	00ef0eb3          	add	t4,t5,a4
1c00e7d8:	02505463          	blez	t0,1c00e800 <mm_M_unroll_2x2+0x3dc>
1c00e7dc:	00ef0eb3          	add	t4,t5,a4
1c00e7e0:	40ee87b3          	sub	a5,t4,a4
1c00e7e4:	17f1                	addi	a5,a5,-4
1c00e7e6:	00000693          	li	a3,0
1c00e7ea:	8389                	srli	a5,a5,0x2
1c00e7ec:	8672                	mv	a2,t3
1c00e7ee:	0785                	addi	a5,a5,1
1c00e7f0:	0067c0fb          	lp.setup	x1,a5,1c00e7fc <mm_M_unroll_2x2+0x3d8>
1c00e7f4:	0047250b          	p.lw	a0,4(a4!)
1c00e7f8:	0046258b          	p.lw	a1,4(a2!)
1c00e7fc:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00e800:	00d368ab          	p.sw	a3,a7(t1!)
1c00e804:	8776                	mv	a4,t4
1c00e806:	b525                	j	1c00e62e <mm_M_unroll_2x2+0x20a>
1c00e808:	00000893          	li	a7,0
1c00e80c:	93e6                	add	t2,t2,s9
1c00e80e:	85c6                	mv	a1,a7
1c00e810:	8546                	mv	a0,a7
1c00e812:	8846                	mv	a6,a7
1c00e814:	00ba6dab          	p.sw	a1,s11(s4!)
1c00e818:	00a9edab          	p.sw	a0,s11(s3!)
1c00e81c:	01096dab          	p.sw	a6,s11(s2!)
1c00e820:	0114edab          	p.sw	a7,s11(s1!)
1c00e824:	147d                	addi	s0,s0,-1
1c00e826:	d00410e3          	bnez	s0,1c00e526 <mm_M_unroll_2x2+0x102>
1c00e82a:	bba1                	j	1c00e582 <mm_M_unroll_2x2+0x15e>
1c00e82c:	00000893          	li	a7,0
1c00e830:	93da                	add	t2,t2,s6
1c00e832:	85c6                	mv	a1,a7
1c00e834:	8546                	mv	a0,a7
1c00e836:	8846                	mv	a6,a7
1c00e838:	00ba6dab          	p.sw	a1,s11(s4!)
1c00e83c:	00a9edab          	p.sw	a0,s11(s3!)
1c00e840:	01096dab          	p.sw	a6,s11(s2!)
1c00e844:	0114edab          	p.sw	a7,s11(s1!)
1c00e848:	147d                	addi	s0,s0,-1
1c00e84a:	ea0414e3          	bnez	s0,1c00e6f2 <mm_M_unroll_2x2+0x2ce>
1c00e84e:	b701                	j	1c00e74e <mm_M_unroll_2x2+0x32a>
1c00e850:	57e2                	lw	a5,56(sp)
1c00e852:	4852                	lw	a6,20(sp)
1c00e854:	5ef2                	lw	t4,60(sp)
1c00e856:	4e36                	lw	t3,76(sp)
1c00e858:	019788b3          	add	a7,a5,s9
1c00e85c:	4309                	li	t1,2
1c00e85e:	00000713          	li	a4,0
1c00e862:	02505363          	blez	t0,1c00e888 <mm_M_unroll_2x2+0x464>
1c00e866:	41ce87b3          	sub	a5,t4,t3
1c00e86a:	17f1                	addi	a5,a5,-4
1c00e86c:	00000713          	li	a4,0
1c00e870:	8389                	srli	a5,a5,0x2
1c00e872:	8642                	mv	a2,a6
1c00e874:	86f2                	mv	a3,t3
1c00e876:	0785                	addi	a5,a5,1
1c00e878:	0067c0fb          	lp.setup	x1,a5,1c00e884 <mm_M_unroll_2x2+0x460>
1c00e87c:	0046a50b          	p.lw	a0,4(a3!)
1c00e880:	0046258b          	p.lw	a1,4(a2!)
1c00e884:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00e888:	00e8a22b          	p.sw	a4,4(a7!)
1c00e88c:	986a                	add	a6,a6,s10
1c00e88e:	ec1323e3          	p.beqimm	t1,1,1c00e754 <mm_M_unroll_2x2+0x330>
1c00e892:	4305                	li	t1,1
1c00e894:	b7e9                	j	1c00e85e <mm_M_unroll_2x2+0x43a>
1c00e896:	5632                	lw	a2,44(sp)
1c00e898:	56f2                	lw	a3,60(sp)
1c00e89a:	5ec2                	lw	t4,48(sp)
1c00e89c:	5e62                	lw	t3,56(sp)
1c00e89e:	018605b3          	add	a1,a2,s8
1c00e8a2:	01868533          	add	a0,a3,s8
1c00e8a6:	4601                	li	a2,0
1c00e8a8:	00000713          	li	a4,0
1c00e8ac:	02505463          	blez	t0,1c00e8d4 <mm_M_unroll_2x2+0x4b0>
1c00e8b0:	41ce87b3          	sub	a5,t4,t3
1c00e8b4:	17f1                	addi	a5,a5,-4
1c00e8b6:	00000713          	li	a4,0
1c00e8ba:	8389                	srli	a5,a5,0x2
1c00e8bc:	00c50833          	add	a6,a0,a2
1c00e8c0:	86f2                	mv	a3,t3
1c00e8c2:	0785                	addi	a5,a5,1
1c00e8c4:	0067c0fb          	lp.setup	x1,a5,1c00e8d0 <mm_M_unroll_2x2+0x4ac>
1c00e8c8:	0046a30b          	p.lw	t1,4(a3!)
1c00e8cc:	21a8788b          	p.lw	a7,s10(a6!)
1c00e8d0:	71137743          	fmadd.s	a4,t1,a7,a4
1c00e8d4:	00e5a22b          	p.sw	a4,4(a1!)
1c00e8d8:	0611                	addi	a2,a2,4
1c00e8da:	fc8637e3          	p.bneimm	a2,8,1c00e8a8 <mm_M_unroll_2x2+0x484>
1c00e8de:	b16d                	j	1c00e588 <mm_M_unroll_2x2+0x164>
1c00e8e0:	5476                	lw	s0,124(sp)
1c00e8e2:	54e6                	lw	s1,120(sp)
1c00e8e4:	5956                	lw	s2,116(sp)
1c00e8e6:	59c6                	lw	s3,112(sp)
1c00e8e8:	5a36                	lw	s4,108(sp)
1c00e8ea:	5aa6                	lw	s5,104(sp)
1c00e8ec:	5b16                	lw	s6,100(sp)
1c00e8ee:	5b86                	lw	s7,96(sp)
1c00e8f0:	4c76                	lw	s8,92(sp)
1c00e8f2:	4ce6                	lw	s9,88(sp)
1c00e8f4:	4d56                	lw	s10,84(sp)
1c00e8f6:	4dc6                	lw	s11,80(sp)
1c00e8f8:	6109                	addi	sp,sp,128
1c00e8fa:	a6bfe06f          	j	1c00d364 <mm_M_unroll_8x1>
1c00e8fe:	4405                	li	s0,1
1c00e900:	bbcd                	j	1c00e6f2 <mm_M_unroll_2x2+0x2ce>
1c00e902:	4405                	li	s0,1
1c00e904:	b10d                	j	1c00e526 <mm_M_unroll_2x2+0x102>
	...

1c00e908 <mm_M_unroll_4x2>:
1c00e908:	01052303          	lw	t1,16(a0)
1c00e90c:	7171                	addi	sp,sp,-176
1c00e90e:	d722                	sw	s0,172(sp)
1c00e910:	d526                	sw	s1,168(sp)
1c00e912:	d34a                	sw	s2,164(sp)
1c00e914:	d14e                	sw	s3,160(sp)
1c00e916:	cf52                	sw	s4,156(sp)
1c00e918:	cd56                	sw	s5,152(sp)
1c00e91a:	cb5a                	sw	s6,148(sp)
1c00e91c:	c95e                	sw	s7,144(sp)
1c00e91e:	c762                	sw	s8,140(sp)
1c00e920:	c566                	sw	s9,136(sp)
1c00e922:	c36a                	sw	s10,132(sp)
1c00e924:	c16e                	sw	s11,128(sp)
1c00e926:	de1a                	sw	t1,60(sp)
1c00e928:	46bd                	li	a3,15
1c00e92a:	c0033733          	p.bclr	a4,t1,0,0
1c00e92e:	f1402673          	csrr	a2,mhartid
1c00e932:	60e6d463          	ble	a4,a3,1c00ef3a <mm_M_unroll_4x2+0x632>
1c00e936:	454c                	lw	a1,12(a0)
1c00e938:	468d                	li	a3,3
1c00e93a:	87aa                	mv	a5,a0
1c00e93c:	c0ae                	sw	a1,64(sp)
1c00e93e:	60b6dd63          	ble	a1,a3,1c00ef58 <mm_M_unroll_4x2+0x650>
1c00e942:	00770693          	addi	a3,a4,7
1c00e946:	868d                	srai	a3,a3,0x3
1c00e948:	f4563633          	p.bclr	a2,a2,26,5
1c00e94c:	d8b2                	sw	a2,112(sp)
1c00e94e:	02d60633          	mul	a2,a2,a3
1c00e952:	851a                	mv	a0,t1
1c00e954:	0007a803          	lw	a6,0(a5)
1c00e958:	0047a883          	lw	a7,4(a5)
1c00e95c:	0087a303          	lw	t1,8(a5)
1c00e960:	8d19                	sub	a0,a0,a4
1c00e962:	c205b5b3          	p.bclr	a1,a1,1,0
1c00e966:	c6c2                	sw	a6,76(sp)
1c00e968:	d4c6                	sw	a7,104(sp)
1c00e96a:	da9a                	sw	t1,116(sp)
1c00e96c:	da32                	sw	a2,52(sp)
1c00e96e:	4f90                	lw	a2,24(a5)
1c00e970:	4bdc                	lw	a5,20(a5)
1c00e972:	dcaa                	sw	a0,120(sp)
1c00e974:	dc2e                	sw	a1,56(sp)
1c00e976:	d23e                	sw	a5,36(sp)
1c00e978:	57d2                	lw	a5,52(sp)
1c00e97a:	96be                	add	a3,a3,a5
1c00e97c:	04e6c6b3          	p.min	a3,a3,a4
1c00e980:	c4b6                	sw	a3,72(sp)
1c00e982:	26061063          	bnez	a2,1c00ebe2 <mm_M_unroll_4x2+0x2da>
1c00e986:	85be                	mv	a1,a5
1c00e988:	1ad7d963          	ble	a3,a5,1c00eb3a <mm_M_unroll_4x2+0x232>
1c00e98c:	88be                	mv	a7,a5
1c00e98e:	5762                	lw	a4,56(sp)
1c00e990:	57f2                	lw	a5,60(sp)
1c00e992:	5872                	lw	a6,60(sp)
1c00e994:	00188613          	addi	a2,a7,1
1c00e998:	02f706b3          	mul	a3,a4,a5
1c00e99c:	95c2                	add	a1,a1,a6
1c00e99e:	00281c93          	slli	s9,a6,0x2
1c00e9a2:	00381713          	slli	a4,a6,0x3
1c00e9a6:	5812                	lw	a6,36(sp)
1c00e9a8:	00289513          	slli	a0,a7,0x2
1c00e9ac:	58f2                	lw	a7,60(sp)
1c00e9ae:	080a                	slli	a6,a6,0x2
1c00e9b0:	c242                	sw	a6,4(sp)
1c00e9b2:	5826                	lw	a6,104(sp)
1c00e9b4:	411686b3          	sub	a3,a3,a7
1c00e9b8:	58d6                	lw	a7,116(sp)
1c00e9ba:	058a                	slli	a1,a1,0x2
1c00e9bc:	060a                	slli	a2,a2,0x2
1c00e9be:	95c6                	add	a1,a1,a7
1c00e9c0:	9642                	add	a2,a2,a6
1c00e9c2:	068a                	slli	a3,a3,0x2
1c00e9c4:	ce2e                	sw	a1,28(sp)
1c00e9c6:	d432                	sw	a2,40(sp)
1c00e9c8:	d2b6                	sw	a3,100(sp)
1c00e9ca:	45b6                	lw	a1,76(sp)
1c00e9cc:	5662                	lw	a2,56(sp)
1c00e9ce:	4692                	lw	a3,4(sp)
1c00e9d0:	40f007b3          	neg	a5,a5
1c00e9d4:	00279313          	slli	t1,a5,0x2
1c00e9d8:	42d605b3          	p.mac	a1,a2,a3
1c00e9dc:	c89a                	sw	t1,80(sp)
1c00e9de:	4306                	lw	t1,64(sp)
1c00e9e0:	9542                	add	a0,a0,a6
1c00e9e2:	078e                	slli	a5,a5,0x3
1c00e9e4:	fa233333          	p.bclr	t1,t1,29,2
1c00e9e8:	d02a                	sw	a0,32(sp)
1c00e9ea:	cabe                	sw	a5,84(sp)
1c00e9ec:	5512                	lw	a0,36(sp)
1c00e9ee:	47c6                	lw	a5,80(sp)
1c00e9f0:	5892                	lw	a7,36(sp)
1c00e9f2:	c29a                	sw	t1,68(sp)
1c00e9f4:	5372                	lw	t1,60(sp)
1c00e9f6:	0512                	slli	a0,a0,0x4
1c00e9f8:	0791                	addi	a5,a5,4
1c00e9fa:	004c8813          	addi	a6,s9,4
1c00e9fe:	0711                	addi	a4,a4,4
1c00ea00:	088e                	slli	a7,a7,0x3
1c00ea02:	00431d13          	slli	s10,t1,0x4
1c00ea06:	d62a                	sw	a0,44(sp)
1c00ea08:	d6ae                	sw	a1,108(sp)
1c00ea0a:	ccbe                	sw	a5,88(sp)
1c00ea0c:	cec2                	sw	a6,92(sp)
1c00ea0e:	d0ba                	sw	a4,96(sp)
1c00ea10:	d846                	sw	a7,48(sp)
1c00ea12:	5362                	lw	t1,56(sp)
1c00ea14:	10030163          	beqz	t1,1c00eb16 <mm_M_unroll_4x2+0x20e>
1c00ea18:	45f2                	lw	a1,28(sp)
1c00ea1a:	4546                	lw	a0,80(sp)
1c00ea1c:	4656                	lw	a2,84(sp)
1c00ea1e:	fff30693          	addi	a3,t1,-1
1c00ea22:	00b50c33          	add	s8,a0,a1
1c00ea26:	00330793          	addi	a5,t1,3
1c00ea2a:	48e6                	lw	a7,88(sp)
1c00ea2c:	4376                	lw	t1,92(sp)
1c00ea2e:	5506                	lw	a0,96(sp)
1c00ea30:	40cc0bb3          	sub	s7,s8,a2
1c00ea34:	8289                	srli	a3,a3,0x2
1c00ea36:	019b8833          	add	a6,s7,s9
1c00ea3a:	98ae                	add	a7,a7,a1
1c00ea3c:	932e                	add	t1,t1,a1
1c00ea3e:	952e                	add	a0,a0,a1
1c00ea40:	0685                	addi	a3,a3,1
1c00ea42:	4711                	li	a4,4
1c00ea44:	ca42                	sw	a6,20(sp)
1c00ea46:	c846                	sw	a7,16(sp)
1c00ea48:	c61a                	sw	t1,12(sp)
1c00ea4a:	c42a                	sw	a0,8(sp)
1c00ea4c:	cc2e                	sw	a1,24(sp)
1c00ea4e:	c036                	sw	a3,0(sp)
1c00ea50:	4b36                	lw	s6,76(sp)
1c00ea52:	00458d93          	addi	s11,a1,4
1c00ea56:	52e7c063          	blt	a5,a4,1c00ef76 <mm_M_unroll_4x2+0x66e>
1c00ea5a:	5312                	lw	t1,36(sp)
1c00ea5c:	3e605c63          	blez	t1,1c00ee54 <mm_M_unroll_4x2+0x54c>
1c00ea60:	4312                	lw	t1,4(sp)
1c00ea62:	5542                	lw	a0,48(sp)
1c00ea64:	00000393          	li	t2,0
1c00ea68:	01630933          	add	s2,t1,s6
1c00ea6c:	006904b3          	add	s1,s2,t1
1c00ea70:	00648433          	add	s0,s1,t1
1c00ea74:	40a406b3          	sub	a3,s0,a0
1c00ea78:	416686b3          	sub	a3,a3,s6
1c00ea7c:	16f1                	addi	a3,a3,-4
1c00ea7e:	829e                	mv	t0,t2
1c00ea80:	8f9e                	mv	t6,t2
1c00ea82:	8f1e                	mv	t5,t2
1c00ea84:	8e9e                	mv	t4,t2
1c00ea86:	8e1e                	mv	t3,t2
1c00ea88:	831e                	mv	t1,t2
1c00ea8a:	889e                	mv	a7,t2
1c00ea8c:	5aa2                	lw	s5,40(sp)
1c00ea8e:	5982                	lw	s3,32(sp)
1c00ea90:	8289                	srli	a3,a3,0x2
1c00ea92:	8a5a                	mv	s4,s6
1c00ea94:	0685                	addi	a3,a3,1
1c00ea96:	0001                	nop
1c00ea98:	01c6c0fb          	lp.setup	x1,a3,1c00ead0 <mm_M_unroll_4x2+0x1c8>
1c00ea9c:	2199f70b          	p.lw	a4,s9(s3!)
1c00eaa0:	219af78b          	p.lw	a5,s9(s5!)
1c00eaa4:	004a280b          	p.lw	a6,4(s4!)
1c00eaa8:	0049250b          	p.lw	a0,4(s2!)
1c00eaac:	0044a58b          	p.lw	a1,4(s1!)
1c00eab0:	0044260b          	p.lw	a2,4(s0!)
1c00eab4:	88e878c3          	fmadd.s	a7,a6,a4,a7
1c00eab8:	30e57343          	fmadd.s	t1,a0,a4,t1
1c00eabc:	e0e5fe43          	fmadd.s	t3,a1,a4,t3
1c00eac0:	e8e67ec3          	fmadd.s	t4,a2,a4,t4
1c00eac4:	f0f87f43          	fmadd.s	t5,a6,a5,t5
1c00eac8:	f8f57fc3          	fmadd.s	t6,a0,a5,t6
1c00eacc:	28f5f2c3          	fmadd.s	t0,a1,a5,t0
1c00ead0:	38f673c3          	fmadd.s	t2,a2,a5,t2
1c00ead4:	45e2                	lw	a1,24(sp)
1c00ead6:	011c6d2b          	p.sw	a7,s10(s8!)
1c00eada:	4652                	lw	a2,20(sp)
1c00eadc:	0065ed2b          	p.sw	t1,s10(a1!)
1c00eae0:	46c2                	lw	a3,16(sp)
1c00eae2:	01cbed2b          	p.sw	t3,s10(s7!)
1c00eae6:	01d66d2b          	p.sw	t4,s10(a2!)
1c00eaea:	4732                	lw	a4,12(sp)
1c00eaec:	01e6ed2b          	p.sw	t5,s10(a3!)
1c00eaf0:	47a2                	lw	a5,8(sp)
1c00eaf2:	4882                	lw	a7,0(sp)
1c00eaf4:	01fded2b          	p.sw	t6,s10(s11!)
1c00eaf8:	00576d2b          	p.sw	t0,s10(a4!)
1c00eafc:	5832                	lw	a6,44(sp)
1c00eafe:	0077ed2b          	p.sw	t2,s10(a5!)
1c00eb02:	18fd                	addi	a7,a7,-1
1c00eb04:	cc2e                	sw	a1,24(sp)
1c00eb06:	ca32                	sw	a2,20(sp)
1c00eb08:	c836                	sw	a3,16(sp)
1c00eb0a:	c63a                	sw	a4,12(sp)
1c00eb0c:	c43e                	sw	a5,8(sp)
1c00eb0e:	c046                	sw	a7,0(sp)
1c00eb10:	9b42                	add	s6,s6,a6
1c00eb12:	f40894e3          	bnez	a7,1c00ea5a <mm_M_unroll_4x2+0x152>
1c00eb16:	4516                	lw	a0,68(sp)
1c00eb18:	34051863          	bnez	a0,1c00ee68 <mm_M_unroll_4x2+0x560>
1c00eb1c:	56d2                	lw	a3,52(sp)
1c00eb1e:	5702                	lw	a4,32(sp)
1c00eb20:	47f2                	lw	a5,28(sp)
1c00eb22:	5822                	lw	a6,40(sp)
1c00eb24:	48a6                	lw	a7,72(sp)
1c00eb26:	0689                	addi	a3,a3,2
1c00eb28:	0721                	addi	a4,a4,8
1c00eb2a:	07a1                	addi	a5,a5,8
1c00eb2c:	0821                	addi	a6,a6,8
1c00eb2e:	da36                	sw	a3,52(sp)
1c00eb30:	d03a                	sw	a4,32(sp)
1c00eb32:	ce3e                	sw	a5,28(sp)
1c00eb34:	d442                	sw	a6,40(sp)
1c00eb36:	ed16cee3          	blt	a3,a7,1c00ea12 <mm_M_unroll_4x2+0x10a>
1c00eb3a:	5666                	lw	a2,120(sp)
1c00eb3c:	08c05563          	blez	a2,1c00ebc6 <mm_M_unroll_4x2+0x2be>
1c00eb40:	4686                	lw	a3,64(sp)
1c00eb42:	5846                	lw	a6,112(sp)
1c00eb44:	00768793          	addi	a5,a3,7
1c00eb48:	878d                	srai	a5,a5,0x3
1c00eb4a:	02f80733          	mul	a4,a6,a5
1c00eb4e:	00e78833          	add	a6,a5,a4
1c00eb52:	04d84833          	p.min	a6,a6,a3
1c00eb56:	07075863          	ble	a6,a4,1c00ebc6 <mm_M_unroll_4x2+0x2be>
1c00eb5a:	58f2                	lw	a7,60(sp)
1c00eb5c:	5512                	lw	a0,36(sp)
1c00eb5e:	46b6                	lw	a3,76(sp)
1c00eb60:	fff88793          	addi	a5,a7,-1
1c00eb64:	833e                	mv	t1,a5
1c00eb66:	42e88333          	p.mac	t1,a7,a4
1c00eb6a:	00251e93          	slli	t4,a0,0x2
1c00eb6e:	55a6                	lw	a1,104(sp)
1c00eb70:	5656                	lw	a2,116(sp)
1c00eb72:	078a                	slli	a5,a5,0x2
1c00eb74:	40e80833          	sub	a6,a6,a4
1c00eb78:	00289f13          	slli	t5,a7,0x2
1c00eb7c:	00f58e33          	add	t3,a1,a5
1c00eb80:	8faa                	mv	t6,a0
1c00eb82:	43d706b3          	p.mac	a3,a4,t4
1c00eb86:	030a                	slli	t1,t1,0x2
1c00eb88:	9332                	add	t1,t1,a2
1c00eb8a:	8736                	mv	a4,a3
1c00eb8c:	01c8407b          	lp.setup	x0,a6,1c00ebc4 <mm_M_unroll_4x2+0x2bc>
1c00eb90:	00000693          	li	a3,0
1c00eb94:	00ee88b3          	add	a7,t4,a4
1c00eb98:	03f05463          	blez	t6,1c00ebc0 <mm_M_unroll_4x2+0x2b8>
1c00eb9c:	00ee88b3          	add	a7,t4,a4
1c00eba0:	40e887b3          	sub	a5,a7,a4
1c00eba4:	17f1                	addi	a5,a5,-4
1c00eba6:	00000693          	li	a3,0
1c00ebaa:	8389                	srli	a5,a5,0x2
1c00ebac:	8672                	mv	a2,t3
1c00ebae:	0785                	addi	a5,a5,1
1c00ebb0:	0067c0fb          	lp.setup	x1,a5,1c00ebbc <mm_M_unroll_4x2+0x2b4>
1c00ebb4:	0047250b          	p.lw	a0,4(a4!)
1c00ebb8:	21e6758b          	p.lw	a1,t5(a2!)
1c00ebbc:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00ebc0:	00d36f2b          	p.sw	a3,t5(t1!)
1c00ebc4:	8746                	mv	a4,a7
1c00ebc6:	543a                	lw	s0,172(sp)
1c00ebc8:	54aa                	lw	s1,168(sp)
1c00ebca:	591a                	lw	s2,164(sp)
1c00ebcc:	598a                	lw	s3,160(sp)
1c00ebce:	4a7a                	lw	s4,156(sp)
1c00ebd0:	4aea                	lw	s5,152(sp)
1c00ebd2:	4b5a                	lw	s6,148(sp)
1c00ebd4:	4bca                	lw	s7,144(sp)
1c00ebd6:	4c3a                	lw	s8,140(sp)
1c00ebd8:	4caa                	lw	s9,136(sp)
1c00ebda:	4d1a                	lw	s10,132(sp)
1c00ebdc:	4d8a                	lw	s11,128(sp)
1c00ebde:	614d                	addi	sp,sp,176
1c00ebe0:	8082                	ret
1c00ebe2:	1cd7d663          	ble	a3,a5,1c00edae <mm_M_unroll_4x2+0x4a6>
1c00ebe6:	5872                	lw	a6,60(sp)
1c00ebe8:	5312                	lw	t1,36(sp)
1c00ebea:	88be                	mv	a7,a5
1c00ebec:	00281593          	slli	a1,a6,0x2
1c00ebf0:	c2ae                	sw	a1,68(sp)
1c00ebf2:	5592                	lw	a1,36(sp)
1c00ebf4:	410007b3          	neg	a5,a6
1c00ebf8:	030a                	slli	t1,t1,0x2
1c00ebfa:	c41a                	sw	t1,8(sp)
1c00ebfc:	01180533          	add	a0,a6,a7
1c00ec00:	00279613          	slli	a2,a5,0x2
1c00ec04:	00188693          	addi	a3,a7,1
1c00ec08:	42b885b3          	p.mac	a1,a7,a1
1c00ec0c:	5312                	lw	t1,36(sp)
1c00ec0e:	4886                	lw	a7,64(sp)
1c00ec10:	cab2                	sw	a2,84(sp)
1c00ec12:	00381613          	slli	a2,a6,0x3
1c00ec16:	5862                	lw	a6,56(sp)
1c00ec18:	40000737          	lui	a4,0x40000
1c00ec1c:	fa28b8b3          	p.bclr	a7,a7,29,2
1c00ec20:	030e                	slli	t1,t1,0x3
1c00ec22:	177d                	addi	a4,a4,-1
1c00ec24:	9742                	add	a4,a4,a6
1c00ec26:	c8c6                	sw	a7,80(sp)
1c00ec28:	d81a                	sw	t1,48(sp)
1c00ec2a:	58d2                	lw	a7,52(sp)
1c00ec2c:	4322                	lw	t1,8(sp)
1c00ec2e:	5826                	lw	a6,104(sp)
1c00ec30:	050a                	slli	a0,a0,0x2
1c00ec32:	058a                	slli	a1,a1,0x2
1c00ec34:	42688833          	p.mac	a6,a7,t1
1c00ec38:	58a6                	lw	a7,104(sp)
1c00ec3a:	078e                	slli	a5,a5,0x3
1c00ec3c:	ccbe                	sw	a5,88(sp)
1c00ec3e:	95c6                	add	a1,a1,a7
1c00ec40:	d02e                	sw	a1,32(sp)
1c00ec42:	4796                	lw	a5,68(sp)
1c00ec44:	45d6                	lw	a1,84(sp)
1c00ec46:	5372                	lw	t1,60(sp)
1c00ec48:	0791                	addi	a5,a5,4
1c00ec4a:	cc42                	sw	a6,24(sp)
1c00ec4c:	5856                	lw	a6,116(sp)
1c00ec4e:	0591                	addi	a1,a1,4
1c00ec50:	0611                	addi	a2,a2,4
1c00ec52:	9542                	add	a0,a0,a6
1c00ec54:	4822                	lw	a6,8(sp)
1c00ec56:	ce2a                	sw	a0,28(sp)
1c00ec58:	5512                	lw	a0,36(sp)
1c00ec5a:	430688b3          	p.mac	a7,a3,a6
1c00ec5e:	00431c13          	slli	s8,t1,0x4
1c00ec62:	0512                	slli	a0,a0,0x4
1c00ec64:	d62a                	sw	a0,44(sp)
1c00ec66:	ceae                	sw	a1,92(sp)
1c00ec68:	d0be                	sw	a5,96(sp)
1c00ec6a:	d2b2                	sw	a2,100(sp)
1c00ec6c:	d446                	sw	a7,40(sp)
1c00ec6e:	4896                	lw	a7,68(sp)
1c00ec70:	03170733          	mul	a4,a4,a7
1c00ec74:	d6ba                	sw	a4,108(sp)
1c00ec76:	4336                	lw	t1,76(sp)
1c00ec78:	5562                	lw	a0,56(sp)
1c00ec7a:	42a80333          	p.mac	t1,a6,a0
1c00ec7e:	de9a                	sw	t1,124(sp)
1c00ec80:	5562                	lw	a0,56(sp)
1c00ec82:	10050063          	beqz	a0,1c00ed82 <mm_M_unroll_4x2+0x47a>
1c00ec86:	4672                	lw	a2,28(sp)
1c00ec88:	45d6                	lw	a1,84(sp)
1c00ec8a:	4766                	lw	a4,88(sp)
1c00ec8c:	fff50693          	addi	a3,a0,-1
1c00ec90:	00c58bb3          	add	s7,a1,a2
1c00ec94:	00350793          	addi	a5,a0,3
1c00ec98:	4816                	lw	a6,68(sp)
1c00ec9a:	5506                	lw	a0,96(sp)
1c00ec9c:	5596                	lw	a1,100(sp)
1c00ec9e:	48b6                	lw	a7,76(sp)
1c00eca0:	40eb8b33          	sub	s6,s7,a4
1c00eca4:	8289                	srli	a3,a3,0x2
1c00eca6:	4376                	lw	t1,92(sp)
1c00eca8:	985a                	add	a6,a6,s6
1c00ecaa:	9532                	add	a0,a0,a2
1c00ecac:	95b2                	add	a1,a1,a2
1c00ecae:	0685                	addi	a3,a3,1
1c00ecb0:	4711                	li	a4,4
1c00ecb2:	ca42                	sw	a6,20(sp)
1c00ecb4:	c046                	sw	a7,0(sp)
1c00ecb6:	c82a                	sw	a0,16(sp)
1c00ecb8:	c62e                	sw	a1,12(sp)
1c00ecba:	c236                	sw	a3,4(sp)
1c00ecbc:	00c30d33          	add	s10,t1,a2
1c00ecc0:	00460c93          	addi	s9,a2,4
1c00ecc4:	8db2                	mv	s11,a2
1c00ecc6:	2ae7cb63          	blt	a5,a4,1c00ef7c <mm_M_unroll_4x2+0x674>
1c00ecca:	5892                	lw	a7,36(sp)
1c00eccc:	17105a63          	blez	a7,1c00ee40 <mm_M_unroll_4x2+0x538>
1c00ecd0:	47e2                	lw	a5,24(sp)
1c00ecd2:	5702                	lw	a4,32(sp)
1c00ecd4:	00000393          	li	t2,0
1c00ecd8:	4802                	lw	a6,0(sp)
1c00ecda:	48a2                	lw	a7,8(sp)
1c00ecdc:	4522                	lw	a0,8(sp)
1c00ecde:	40f706b3          	sub	a3,a4,a5
1c00ece2:	011804b3          	add	s1,a6,a7
1c00ece6:	16f1                	addi	a3,a3,-4
1c00ece8:	829e                	mv	t0,t2
1c00ecea:	8f9e                	mv	t6,t2
1c00ecec:	8f1e                	mv	t5,t2
1c00ecee:	8e9e                	mv	t4,t2
1c00ecf0:	8e1e                	mv	t3,t2
1c00ecf2:	831e                	mv	t1,t2
1c00ecf4:	889e                	mv	a7,t2
1c00ecf6:	5a22                	lw	s4,40(sp)
1c00ecf8:	00a48433          	add	s0,s1,a0
1c00ecfc:	8289                	srli	a3,a3,0x2
1c00ecfe:	00a40ab3          	add	s5,s0,a0
1c00ed02:	89c2                	mv	s3,a6
1c00ed04:	893e                	mv	s2,a5
1c00ed06:	0685                	addi	a3,a3,1
1c00ed08:	01c6c0fb          	lp.setup	x1,a3,1c00ed40 <mm_M_unroll_4x2+0x438>
1c00ed0c:	0049270b          	p.lw	a4,4(s2!)
1c00ed10:	004a278b          	p.lw	a5,4(s4!)
1c00ed14:	0049a80b          	p.lw	a6,4(s3!)
1c00ed18:	0044a50b          	p.lw	a0,4(s1!)
1c00ed1c:	0044258b          	p.lw	a1,4(s0!)
1c00ed20:	004aa60b          	p.lw	a2,4(s5!)
1c00ed24:	88e878c3          	fmadd.s	a7,a6,a4,a7
1c00ed28:	30e57343          	fmadd.s	t1,a0,a4,t1
1c00ed2c:	e0e5fe43          	fmadd.s	t3,a1,a4,t3
1c00ed30:	e8e67ec3          	fmadd.s	t4,a2,a4,t4
1c00ed34:	f0f87f43          	fmadd.s	t5,a6,a5,t5
1c00ed38:	f8f57fc3          	fmadd.s	t6,a0,a5,t6
1c00ed3c:	28f5f2c3          	fmadd.s	t0,a1,a5,t0
1c00ed40:	38f673c3          	fmadd.s	t2,a2,a5,t2
1c00ed44:	011bec2b          	p.sw	a7,s8(s7!)
1c00ed48:	45d2                	lw	a1,20(sp)
1c00ed4a:	006dec2b          	p.sw	t1,s8(s11!)
1c00ed4e:	01cb6c2b          	p.sw	t3,s8(s6!)
1c00ed52:	01d5ec2b          	p.sw	t4,s8(a1!)
1c00ed56:	4642                	lw	a2,16(sp)
1c00ed58:	01ed6c2b          	p.sw	t5,s8(s10!)
1c00ed5c:	46b2                	lw	a3,12(sp)
1c00ed5e:	4702                	lw	a4,0(sp)
1c00ed60:	4812                	lw	a6,4(sp)
1c00ed62:	57b2                	lw	a5,44(sp)
1c00ed64:	01fcec2b          	p.sw	t6,s8(s9!)
1c00ed68:	00566c2b          	p.sw	t0,s8(a2!)
1c00ed6c:	0076ec2b          	p.sw	t2,s8(a3!)
1c00ed70:	973e                	add	a4,a4,a5
1c00ed72:	187d                	addi	a6,a6,-1
1c00ed74:	ca2e                	sw	a1,20(sp)
1c00ed76:	c832                	sw	a2,16(sp)
1c00ed78:	c636                	sw	a3,12(sp)
1c00ed7a:	c03a                	sw	a4,0(sp)
1c00ed7c:	c242                	sw	a6,4(sp)
1c00ed7e:	f40816e3          	bnez	a6,1c00ecca <mm_M_unroll_4x2+0x3c2>
1c00ed82:	4346                	lw	t1,80(sp)
1c00ed84:	14031663          	bnez	t1,1c00eed0 <mm_M_unroll_4x2+0x5c8>
1c00ed88:	5742                	lw	a4,48(sp)
1c00ed8a:	5652                	lw	a2,52(sp)
1c00ed8c:	46e2                	lw	a3,24(sp)
1c00ed8e:	47f2                	lw	a5,28(sp)
1c00ed90:	5822                	lw	a6,40(sp)
1c00ed92:	5882                	lw	a7,32(sp)
1c00ed94:	4326                	lw	t1,72(sp)
1c00ed96:	0609                	addi	a2,a2,2
1c00ed98:	96ba                	add	a3,a3,a4
1c00ed9a:	07a1                	addi	a5,a5,8
1c00ed9c:	983a                	add	a6,a6,a4
1c00ed9e:	98ba                	add	a7,a7,a4
1c00eda0:	da32                	sw	a2,52(sp)
1c00eda2:	cc36                	sw	a3,24(sp)
1c00eda4:	ce3e                	sw	a5,28(sp)
1c00eda6:	d442                	sw	a6,40(sp)
1c00eda8:	d046                	sw	a7,32(sp)
1c00edaa:	ec664be3          	blt	a2,t1,1c00ec80 <mm_M_unroll_4x2+0x378>
1c00edae:	56e6                	lw	a3,120(sp)
1c00edb0:	e0d05be3          	blez	a3,1c00ebc6 <mm_M_unroll_4x2+0x2be>
1c00edb4:	4706                	lw	a4,64(sp)
1c00edb6:	5846                	lw	a6,112(sp)
1c00edb8:	4306                	lw	t1,64(sp)
1c00edba:	00770793          	addi	a5,a4,7 # 40000007 <__l2_shared_end+0x23fe7afb>
1c00edbe:	878d                	srai	a5,a5,0x3
1c00edc0:	02f80733          	mul	a4,a6,a5
1c00edc4:	00e788b3          	add	a7,a5,a4
1c00edc8:	0468c8b3          	p.min	a7,a7,t1
1c00edcc:	df175de3          	ble	a7,a4,1c00ebc6 <mm_M_unroll_4x2+0x2be>
1c00edd0:	55f2                	lw	a1,60(sp)
1c00edd2:	5612                	lw	a2,36(sp)
1c00edd4:	56d6                	lw	a3,116(sp)
1c00edd6:	fff58793          	addi	a5,a1,-1
1c00edda:	833e                	mv	t1,a5
1c00eddc:	02c78e33          	mul	t3,a5,a2
1c00ede0:	47b6                	lw	a5,76(sp)
1c00ede2:	00261f13          	slli	t5,a2,0x2
1c00ede6:	5526                	lw	a0,104(sp)
1c00ede8:	40e888b3          	sub	a7,a7,a4
1c00edec:	00259813          	slli	a6,a1,0x2
1c00edf0:	8fb2                	mv	t6,a2
1c00edf2:	42e58333          	p.mac	t1,a1,a4
1c00edf6:	0e0a                	slli	t3,t3,0x2
1c00edf8:	9e2a                	add	t3,t3,a0
1c00edfa:	43e707b3          	p.mac	a5,a4,t5
1c00edfe:	030a                	slli	t1,t1,0x2
1c00ee00:	9336                	add	t1,t1,a3
1c00ee02:	873e                	mv	a4,a5
1c00ee04:	01c8c07b          	lp.setup	x0,a7,1c00ee3c <mm_M_unroll_4x2+0x534>
1c00ee08:	00000693          	li	a3,0
1c00ee0c:	00ef0eb3          	add	t4,t5,a4
1c00ee10:	03f05463          	blez	t6,1c00ee38 <mm_M_unroll_4x2+0x530>
1c00ee14:	00ef0eb3          	add	t4,t5,a4
1c00ee18:	40ee87b3          	sub	a5,t4,a4
1c00ee1c:	17f1                	addi	a5,a5,-4
1c00ee1e:	00000693          	li	a3,0
1c00ee22:	8389                	srli	a5,a5,0x2
1c00ee24:	8672                	mv	a2,t3
1c00ee26:	0785                	addi	a5,a5,1
1c00ee28:	0067c0fb          	lp.setup	x1,a5,1c00ee34 <mm_M_unroll_4x2+0x52c>
1c00ee2c:	0047250b          	p.lw	a0,4(a4!)
1c00ee30:	0046258b          	p.lw	a1,4(a2!)
1c00ee34:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00ee38:	00d3682b          	p.sw	a3,a6(t1!)
1c00ee3c:	8776                	mv	a4,t4
1c00ee3e:	b361                	j	1c00ebc6 <mm_M_unroll_4x2+0x2be>
1c00ee40:	00000393          	li	t2,0
1c00ee44:	829e                	mv	t0,t2
1c00ee46:	8f9e                	mv	t6,t2
1c00ee48:	8f1e                	mv	t5,t2
1c00ee4a:	8e9e                	mv	t4,t2
1c00ee4c:	8e1e                	mv	t3,t2
1c00ee4e:	831e                	mv	t1,t2
1c00ee50:	889e                	mv	a7,t2
1c00ee52:	bdcd                	j	1c00ed44 <mm_M_unroll_4x2+0x43c>
1c00ee54:	00000393          	li	t2,0
1c00ee58:	829e                	mv	t0,t2
1c00ee5a:	8f9e                	mv	t6,t2
1c00ee5c:	8f1e                	mv	t5,t2
1c00ee5e:	8e9e                	mv	t4,t2
1c00ee60:	8e1e                	mv	t3,t2
1c00ee62:	831e                	mv	t1,t2
1c00ee64:	889e                	mv	a7,t2
1c00ee66:	b1bd                	j	1c00ead4 <mm_M_unroll_4x2+0x1cc>
1c00ee68:	5596                	lw	a1,100(sp)
1c00ee6a:	4672                	lw	a2,28(sp)
1c00ee6c:	5e02                	lw	t3,32(sp)
1c00ee6e:	4386                	lw	t2,64(sp)
1c00ee70:	5f92                	lw	t6,36(sp)
1c00ee72:	5462                	lw	s0,56(sp)
1c00ee74:	4292                	lw	t0,4(sp)
1c00ee76:	54b6                	lw	s1,108(sp)
1c00ee78:	00c58f33          	add	t5,a1,a2
1c00ee7c:	4e81                	li	t4,0
1c00ee7e:	04745463          	ble	t2,s0,1c00eec6 <mm_M_unroll_4x2+0x5be>
1c00ee82:	01df0333          	add	t1,t5,t4
1c00ee86:	8726                	mv	a4,s1
1c00ee88:	40838833          	sub	a6,t2,s0
1c00ee8c:	01c8407b          	lp.setup	x0,a6,1c00eec4 <mm_M_unroll_4x2+0x5bc>
1c00ee90:	00000693          	li	a3,0
1c00ee94:	00e288b3          	add	a7,t0,a4
1c00ee98:	03f05463          	blez	t6,1c00eec0 <mm_M_unroll_4x2+0x5b8>
1c00ee9c:	00e288b3          	add	a7,t0,a4
1c00eea0:	40e887b3          	sub	a5,a7,a4
1c00eea4:	17f1                	addi	a5,a5,-4
1c00eea6:	00000693          	li	a3,0
1c00eeaa:	8389                	srli	a5,a5,0x2
1c00eeac:	8672                	mv	a2,t3
1c00eeae:	0785                	addi	a5,a5,1
1c00eeb0:	0067c0fb          	lp.setup	x1,a5,1c00eebc <mm_M_unroll_4x2+0x5b4>
1c00eeb4:	0047250b          	p.lw	a0,4(a4!)
1c00eeb8:	2196758b          	p.lw	a1,s9(a2!)
1c00eebc:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00eec0:	00d36cab          	p.sw	a3,s9(t1!)
1c00eec4:	8746                	mv	a4,a7
1c00eec6:	0e91                	addi	t4,t4,4
1c00eec8:	0e11                	addi	t3,t3,4
1c00eeca:	fa8ebae3          	p.bneimm	t4,8,1c00ee7e <mm_M_unroll_4x2+0x576>
1c00eece:	b1b9                	j	1c00eb1c <mm_M_unroll_4x2+0x214>
1c00eed0:	5536                	lw	a0,108(sp)
1c00eed2:	45f2                	lw	a1,28(sp)
1c00eed4:	4e62                	lw	t3,24(sp)
1c00eed6:	4916                	lw	s2,68(sp)
1c00eed8:	4386                	lw	t2,64(sp)
1c00eeda:	5292                	lw	t0,36(sp)
1c00eedc:	5462                	lw	s0,56(sp)
1c00eede:	4fa2                	lw	t6,8(sp)
1c00eee0:	54f6                	lw	s1,124(sp)
1c00eee2:	00b50eb3          	add	t4,a0,a1
1c00eee6:	4f09                	li	t5,2
1c00eee8:	04745363          	ble	t2,s0,1c00ef2e <mm_M_unroll_4x2+0x626>
1c00eeec:	8726                	mv	a4,s1
1c00eeee:	8376                	mv	t1,t4
1c00eef0:	40838833          	sub	a6,t2,s0
1c00eef4:	01c8407b          	lp.setup	x0,a6,1c00ef2c <mm_M_unroll_4x2+0x624>
1c00eef8:	00000693          	li	a3,0
1c00eefc:	00ef88b3          	add	a7,t6,a4
1c00ef00:	02505463          	blez	t0,1c00ef28 <mm_M_unroll_4x2+0x620>
1c00ef04:	00ef88b3          	add	a7,t6,a4
1c00ef08:	40e887b3          	sub	a5,a7,a4
1c00ef0c:	17f1                	addi	a5,a5,-4
1c00ef0e:	00000693          	li	a3,0
1c00ef12:	8389                	srli	a5,a5,0x2
1c00ef14:	8672                	mv	a2,t3
1c00ef16:	0785                	addi	a5,a5,1
1c00ef18:	0067c0fb          	lp.setup	x1,a5,1c00ef24 <mm_M_unroll_4x2+0x61c>
1c00ef1c:	0047250b          	p.lw	a0,4(a4!)
1c00ef20:	0046258b          	p.lw	a1,4(a2!)
1c00ef24:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00ef28:	00d3692b          	p.sw	a3,s2(t1!)
1c00ef2c:	8746                	mv	a4,a7
1c00ef2e:	0e91                	addi	t4,t4,4
1c00ef30:	9e7e                	add	t3,t3,t6
1c00ef32:	e41f2be3          	p.beqimm	t5,1,1c00ed88 <mm_M_unroll_4x2+0x480>
1c00ef36:	4f05                	li	t5,1
1c00ef38:	bf45                	j	1c00eee8 <mm_M_unroll_4x2+0x5e0>
1c00ef3a:	543a                	lw	s0,172(sp)
1c00ef3c:	54aa                	lw	s1,168(sp)
1c00ef3e:	591a                	lw	s2,164(sp)
1c00ef40:	598a                	lw	s3,160(sp)
1c00ef42:	4a7a                	lw	s4,156(sp)
1c00ef44:	4aea                	lw	s5,152(sp)
1c00ef46:	4b5a                	lw	s6,148(sp)
1c00ef48:	4bca                	lw	s7,144(sp)
1c00ef4a:	4c3a                	lw	s8,140(sp)
1c00ef4c:	4caa                	lw	s9,136(sp)
1c00ef4e:	4d1a                	lw	s10,132(sp)
1c00ef50:	4d8a                	lw	s11,128(sp)
1c00ef52:	614d                	addi	sp,sp,176
1c00ef54:	c10fe06f          	j	1c00d364 <mm_M_unroll_8x1>
1c00ef58:	543a                	lw	s0,172(sp)
1c00ef5a:	54aa                	lw	s1,168(sp)
1c00ef5c:	591a                	lw	s2,164(sp)
1c00ef5e:	598a                	lw	s3,160(sp)
1c00ef60:	4a7a                	lw	s4,156(sp)
1c00ef62:	4aea                	lw	s5,152(sp)
1c00ef64:	4b5a                	lw	s6,148(sp)
1c00ef66:	4bca                	lw	s7,144(sp)
1c00ef68:	4c3a                	lw	s8,140(sp)
1c00ef6a:	4caa                	lw	s9,136(sp)
1c00ef6c:	4d1a                	lw	s10,132(sp)
1c00ef6e:	4d8a                	lw	s11,128(sp)
1c00ef70:	614d                	addi	sp,sp,176
1c00ef72:	cb2ff06f          	j	1c00e424 <mm_M_unroll_2x2>
1c00ef76:	4585                	li	a1,1
1c00ef78:	c02e                	sw	a1,0(sp)
1c00ef7a:	b4c5                	j	1c00ea5a <mm_M_unroll_4x2+0x152>
1c00ef7c:	4605                	li	a2,1
1c00ef7e:	c232                	sw	a2,4(sp)
1c00ef80:	b3a9                	j	1c00ecca <mm_M_unroll_4x2+0x3c2>
	...

1c00ef84 <mm_M_unroll_2x4>:
1c00ef84:	490c                	lw	a1,16(a0)
1c00ef86:	7135                	addi	sp,sp,-160
1c00ef88:	cf22                	sw	s0,156(sp)
1c00ef8a:	c205b633          	p.bclr	a2,a1,1,0
1c00ef8e:	cd26                	sw	s1,152(sp)
1c00ef90:	cb4a                	sw	s2,148(sp)
1c00ef92:	c94e                	sw	s3,144(sp)
1c00ef94:	c752                	sw	s4,140(sp)
1c00ef96:	c556                	sw	s5,136(sp)
1c00ef98:	c35a                	sw	s6,132(sp)
1c00ef9a:	c15e                	sw	s7,128(sp)
1c00ef9c:	dee2                	sw	s8,124(sp)
1c00ef9e:	dce6                	sw	s9,120(sp)
1c00efa0:	daea                	sw	s10,116(sp)
1c00efa2:	d8ee                	sw	s11,112(sp)
1c00efa4:	d82e                	sw	a1,48(sp)
1c00efa6:	477d                	li	a4,31
1c00efa8:	dc32                	sw	a2,56(sp)
1c00efaa:	f14026f3          	csrr	a3,mhartid
1c00efae:	54c75f63          	ble	a2,a4,1c00f50c <mm_M_unroll_2x4+0x588>
1c00efb2:	4558                	lw	a4,12(a0)
1c00efb4:	87aa                	mv	a5,a0
1c00efb6:	c2ba                	sw	a4,68(sp)
1c00efb8:	4816                	lw	a6,68(sp)
1c00efba:	4705                	li	a4,1
1c00efbc:	57075763          	ble	a6,a4,1c00f52a <mm_M_unroll_2x4+0x5a6>
1c00efc0:	00760713          	addi	a4,a2,7
1c00efc4:	f456b6b3          	p.bclr	a3,a3,26,5
1c00efc8:	870d                	srai	a4,a4,0x3
1c00efca:	02e68633          	mul	a2,a3,a4
1c00efce:	00052803          	lw	a6,0(a0)
1c00efd2:	ccb6                	sw	a3,88(sp)
1c00efd4:	4d14                	lw	a3,24(a0)
1c00efd6:	4148                	lw	a0,4(a0)
1c00efd8:	478c                	lw	a1,8(a5)
1c00efda:	da42                	sw	a6,52(sp)
1c00efdc:	c4aa                	sw	a0,72(sp)
1c00efde:	0147ab03          	lw	s6,20(a5)
1c00efe2:	4816                	lw	a6,68(sp)
1c00efe4:	c632                	sw	a2,12(sp)
1c00efe6:	4532                	lw	a0,12(sp)
1c00efe8:	57e2                	lw	a5,56(sp)
1c00efea:	5642                	lw	a2,48(sp)
1c00efec:	972a                	add	a4,a4,a0
1c00efee:	c0083833          	p.bclr	a6,a6,0,0
1c00eff2:	8e1d                	sub	a2,a2,a5
1c00eff4:	04f74733          	p.min	a4,a4,a5
1c00eff8:	ceae                	sw	a1,92(sp)
1c00effa:	d0b2                	sw	a2,96(sp)
1c00effc:	d442                	sw	a6,40(sp)
1c00effe:	de3a                	sw	a4,60(sp)
1c00f000:	26068963          	beqz	a3,1c00f272 <mm_M_unroll_2x4+0x2ee>
1c00f004:	18e55d63          	ble	a4,a0,1c00f19e <mm_M_unroll_2x4+0x21a>
1c00f008:	882a                	mv	a6,a0
1c00f00a:	00380793          	addi	a5,a6,3
1c00f00e:	4816                	lw	a6,68(sp)
1c00f010:	4596                	lw	a1,68(sp)
1c00f012:	872a                	mv	a4,a0
1c00f014:	fc183833          	p.bclr	a6,a6,30,1
1c00f018:	fff58613          	addi	a2,a1,-1
1c00f01c:	c6c2                	sw	a6,76(sp)
1c00f01e:	004b1813          	slli	a6,s6,0x4
1c00f022:	02cb05b3          	mul	a1,s6,a2
1c00f026:	ce42                	sw	a6,28(sp)
1c00f028:	4876                	lw	a6,92(sp)
1c00f02a:	050a                	slli	a0,a0,0x2
1c00f02c:	002b1693          	slli	a3,s6,0x2
1c00f030:	9542                	add	a0,a0,a6
1c00f032:	5842                	lw	a6,48(sp)
1c00f034:	d62a                	sw	a0,44(sp)
1c00f036:	d036                	sw	a3,32(sp)
1c00f038:	00381d13          	slli	s10,a6,0x3
1c00f03c:	080a                	slli	a6,a6,0x2
1c00f03e:	c8c2                	sw	a6,80(sp)
1c00f040:	003b1813          	slli	a6,s6,0x3
1c00f044:	d242                	sw	a6,36(sp)
1c00f046:	5852                	lw	a6,52(sp)
1c00f048:	01658533          	add	a0,a1,s6
1c00f04c:	058a                	slli	a1,a1,0x2
1c00f04e:	95c2                	add	a1,a1,a6
1c00f050:	050a                	slli	a0,a0,0x2
1c00f052:	d4ae                	sw	a1,104(sp)
1c00f054:	55c2                	lw	a1,48(sp)
1c00f056:	9542                	add	a0,a0,a6
1c00f058:	4832                	lw	a6,12(sp)
1c00f05a:	02c58633          	mul	a2,a1,a2
1c00f05e:	85da                	mv	a1,s6
1c00f060:	d2aa                	sw	a0,100(sp)
1c00f062:	4526                	lw	a0,72(sp)
1c00f064:	00170693          	addi	a3,a4,1
1c00f068:	0709                	addi	a4,a4,2
1c00f06a:	cc5a                	sw	s6,24(sp)
1c00f06c:	430b05b3          	p.mac	a1,s6,a6
1c00f070:	060a                	slli	a2,a2,0x2
1c00f072:	cab2                	sw	a2,84(sp)
1c00f074:	0641                	addi	a2,a2,16
1c00f076:	d6b2                	sw	a2,108(sp)
1c00f078:	4626                	lw	a2,72(sp)
1c00f07a:	058a                	slli	a1,a1,0x2
1c00f07c:	95aa                	add	a1,a1,a0
1c00f07e:	c22e                	sw	a1,4(sp)
1c00f080:	5582                	lw	a1,32(sp)
1c00f082:	43058533          	p.mac	a0,a1,a6
1c00f086:	4826                	lw	a6,72(sp)
1c00f088:	c0aa                	sw	a0,64(sp)
1c00f08a:	8daa                	mv	s11,a0
1c00f08c:	4526                	lw	a0,72(sp)
1c00f08e:	42b68833          	p.mac	a6,a3,a1
1c00f092:	42b70533          	p.mac	a0,a4,a1
1c00f096:	ca42                	sw	a6,20(sp)
1c00f098:	42b78633          	p.mac	a2,a5,a1
1c00f09c:	c82a                	sw	a0,16(sp)
1c00f09e:	c432                	sw	a2,8(sp)
1c00f0a0:	5622                	lw	a2,40(sp)
1c00f0a2:	ce5d                	beqz	a2,1c00f160 <mm_M_unroll_2x4+0x1dc>
1c00f0a4:	5832                	lw	a6,44(sp)
1c00f0a6:	46c6                	lw	a3,80(sp)
1c00f0a8:	fff60c93          	addi	s9,a2,-1
1c00f0ac:	001cdc93          	srli	s9,s9,0x1
1c00f0b0:	00160793          	addi	a5,a2,1
1c00f0b4:	4709                	li	a4,2
1c00f0b6:	01068b33          	add	s6,a3,a6
1c00f0ba:	5c52                	lw	s8,52(sp)
1c00f0bc:	8bc2                	mv	s7,a6
1c00f0be:	0c85                	addi	s9,s9,1
1c00f0c0:	48e7c663          	blt	a5,a4,1c00f54c <mm_M_unroll_2x4+0x5c8>
1c00f0c4:	04dcc07b          	lp.setup	x0,s9,1c00f15e <mm_M_unroll_2x4+0x1da>
1c00f0c8:	45e2                	lw	a1,24(sp)
1c00f0ca:	38b05763          	blez	a1,1c00f458 <mm_M_unroll_2x4+0x4d4>
1c00f0ce:	4792                	lw	a5,4(sp)
1c00f0d0:	00000393          	li	t2,0
1c00f0d4:	5802                	lw	a6,32(sp)
1c00f0d6:	41b786b3          	sub	a3,a5,s11
1c00f0da:	16f1                	addi	a3,a3,-4
1c00f0dc:	829e                	mv	t0,t2
1c00f0de:	8f9e                	mv	t6,t2
1c00f0e0:	8f1e                	mv	t5,t2
1c00f0e2:	8e9e                	mv	t4,t2
1c00f0e4:	8e1e                	mv	t3,t2
1c00f0e6:	831e                	mv	t1,t2
1c00f0e8:	889e                	mv	a7,t2
1c00f0ea:	4a22                	lw	s4,8(sp)
1c00f0ec:	49c2                	lw	s3,16(sp)
1c00f0ee:	4952                	lw	s2,20(sp)
1c00f0f0:	8289                	srli	a3,a3,0x2
1c00f0f2:	01880ab3          	add	s5,a6,s8
1c00f0f6:	84e2                	mv	s1,s8
1c00f0f8:	846e                	mv	s0,s11
1c00f0fa:	0685                	addi	a3,a3,1
1c00f0fc:	01c6c0fb          	lp.setup	x1,a3,1c00f134 <mm_M_unroll_2x4+0x1b0>
1c00f100:	0044a70b          	p.lw	a4,4(s1!)
1c00f104:	004aa78b          	p.lw	a5,4(s5!)
1c00f108:	0044280b          	p.lw	a6,4(s0!)
1c00f10c:	0049250b          	p.lw	a0,4(s2!)
1c00f110:	0049a58b          	p.lw	a1,4(s3!)
1c00f114:	004a260b          	p.lw	a2,4(s4!)
1c00f118:	88e878c3          	fmadd.s	a7,a6,a4,a7
1c00f11c:	30f87343          	fmadd.s	t1,a6,a5,t1
1c00f120:	e0a77e43          	fmadd.s	t3,a4,a0,t3
1c00f124:	e8a7fec3          	fmadd.s	t4,a5,a0,t4
1c00f128:	f0b77f43          	fmadd.s	t5,a4,a1,t5
1c00f12c:	f8b7ffc3          	fmadd.s	t6,a5,a1,t6
1c00f130:	28c772c3          	fmadd.s	t0,a4,a2,t0
1c00f134:	38c7f3c3          	fmadd.s	t2,a5,a2,t2
1c00f138:	011ba023          	sw	a7,0(s7)
1c00f13c:	006b2023          	sw	t1,0(s6)
1c00f140:	01cba223          	sw	t3,4(s7)
1c00f144:	01db2223          	sw	t4,4(s6)
1c00f148:	01eba423          	sw	t5,8(s7)
1c00f14c:	01fb2423          	sw	t6,8(s6)
1c00f150:	5512                	lw	a0,36(sp)
1c00f152:	005ba623          	sw	t0,12(s7)
1c00f156:	007b2623          	sw	t2,12(s6)
1c00f15a:	9bea                	add	s7,s7,s10
1c00f15c:	9b6a                	add	s6,s6,s10
1c00f15e:	9c2a                	add	s8,s8,a0
1c00f160:	4636                	lw	a2,76(sp)
1c00f162:	34061963          	bnez	a2,1c00f4b4 <mm_M_unroll_2x4+0x530>
1c00f166:	4832                	lw	a6,12(sp)
1c00f168:	45f2                	lw	a1,28(sp)
1c00f16a:	4506                	lw	a0,64(sp)
1c00f16c:	0811                	addi	a6,a6,4
1c00f16e:	c642                	sw	a6,12(sp)
1c00f170:	46d2                	lw	a3,20(sp)
1c00f172:	4742                	lw	a4,16(sp)
1c00f174:	47a2                	lw	a5,8(sp)
1c00f176:	4812                	lw	a6,4(sp)
1c00f178:	5632                	lw	a2,44(sp)
1c00f17a:	952e                	add	a0,a0,a1
1c00f17c:	96ae                	add	a3,a3,a1
1c00f17e:	972e                	add	a4,a4,a1
1c00f180:	97ae                	add	a5,a5,a1
1c00f182:	982e                	add	a6,a6,a1
1c00f184:	c0aa                	sw	a0,64(sp)
1c00f186:	9dae                	add	s11,s11,a1
1c00f188:	5572                	lw	a0,60(sp)
1c00f18a:	45b2                	lw	a1,12(sp)
1c00f18c:	0641                	addi	a2,a2,16
1c00f18e:	d632                	sw	a2,44(sp)
1c00f190:	ca36                	sw	a3,20(sp)
1c00f192:	c83a                	sw	a4,16(sp)
1c00f194:	c43e                	sw	a5,8(sp)
1c00f196:	c242                	sw	a6,4(sp)
1c00f198:	f0a5c4e3          	blt	a1,a0,1c00f0a0 <mm_M_unroll_2x4+0x11c>
1c00f19c:	4b62                	lw	s6,24(sp)
1c00f19e:	5506                	lw	a0,96(sp)
1c00f1a0:	0aa05b63          	blez	a0,1c00f256 <mm_M_unroll_2x4+0x2d2>
1c00f1a4:	4596                	lw	a1,68(sp)
1c00f1a6:	4666                	lw	a2,88(sp)
1c00f1a8:	00758793          	addi	a5,a1,7
1c00f1ac:	878d                	srai	a5,a5,0x3
1c00f1ae:	02f60eb3          	mul	t4,a2,a5
1c00f1b2:	97f6                	add	a5,a5,t4
1c00f1b4:	04b7c2b3          	p.min	t0,a5,a1
1c00f1b8:	085edf63          	ble	t0,t4,1c00f256 <mm_M_unroll_2x4+0x2d2>
1c00f1bc:	56c2                	lw	a3,48(sp)
1c00f1be:	5762                	lw	a4,56(sp)
1c00f1c0:	08d75b63          	ble	a3,a4,1c00f256 <mm_M_unroll_2x4+0x2d2>
1c00f1c4:	02de87b3          	mul	a5,t4,a3
1c00f1c8:	845a                	mv	s0,s6
1c00f1ca:	5352                	lw	t1,52(sp)
1c00f1cc:	44a6                	lw	s1,72(sp)
1c00f1ce:	002b1e13          	slli	t3,s6,0x2
1c00f1d2:	4876                	lw	a6,92(sp)
1c00f1d4:	5552                	lw	a0,52(sp)
1c00f1d6:	00269393          	slli	t2,a3,0x2
1c00f1da:	8942                	mv	s2,a6
1c00f1dc:	89b6                	mv	s3,a3
1c00f1de:	43db0433          	p.mac	s0,s6,t4
1c00f1e2:	00d78fb3          	add	t6,a5,a3
1c00f1e6:	002f9f13          	slli	t5,t6,0x2
1c00f1ea:	9f42                	add	t5,t5,a6
1c00f1ec:	8a3a                	mv	s4,a4
1c00f1ee:	43de0333          	p.mac	t1,t3,t4
1c00f1f2:	040a                	slli	s0,s0,0x2
1c00f1f4:	942a                	add	s0,s0,a0
1c00f1f6:	43c704b3          	p.mac	s1,a4,t3
1c00f1fa:	014788b3          	add	a7,a5,s4
1c00f1fe:	088a                	slli	a7,a7,0x2
1c00f200:	98ca                	add	a7,a7,s2
1c00f202:	411f0833          	sub	a6,t5,a7
1c00f206:	1871                	addi	a6,a6,-4
1c00f208:	00285813          	srli	a6,a6,0x2
1c00f20c:	8aa6                	mv	s5,s1
1c00f20e:	0805                	addi	a6,a6,1
1c00f210:	01a8407b          	lp.setup	x0,a6,1c00f244 <mm_M_unroll_2x4+0x2c0>
1c00f214:	00000713          	li	a4,0
1c00f218:	03605463          	blez	s6,1c00f240 <mm_M_unroll_2x4+0x2bc>
1c00f21c:	406407b3          	sub	a5,s0,t1
1c00f220:	17f1                	addi	a5,a5,-4
1c00f222:	00000713          	li	a4,0
1c00f226:	8389                	srli	a5,a5,0x2
1c00f228:	8656                	mv	a2,s5
1c00f22a:	869a                	mv	a3,t1
1c00f22c:	0785                	addi	a5,a5,1
1c00f22e:	0001                	nop
1c00f230:	0067c0fb          	lp.setup	x1,a5,1c00f23c <mm_M_unroll_2x4+0x2b8>
1c00f234:	0046a50b          	p.lw	a0,4(a3!)
1c00f238:	0046258b          	p.lw	a1,4(a2!)
1c00f23c:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00f240:	00e8a22b          	p.sw	a4,4(a7!)
1c00f244:	9af2                	add	s5,s5,t3
1c00f246:	0e85                	addi	t4,t4,1
1c00f248:	87fe                	mv	a5,t6
1c00f24a:	9f1e                	add	t5,t5,t2
1c00f24c:	9372                	add	t1,t1,t3
1c00f24e:	9472                	add	s0,s0,t3
1c00f250:	9fce                	add	t6,t6,s3
1c00f252:	fbd294e3          	bne	t0,t4,1c00f1fa <mm_M_unroll_2x4+0x276>
1c00f256:	447a                	lw	s0,156(sp)
1c00f258:	44ea                	lw	s1,152(sp)
1c00f25a:	495a                	lw	s2,148(sp)
1c00f25c:	49ca                	lw	s3,144(sp)
1c00f25e:	4a3a                	lw	s4,140(sp)
1c00f260:	4aaa                	lw	s5,136(sp)
1c00f262:	4b1a                	lw	s6,132(sp)
1c00f264:	4b8a                	lw	s7,128(sp)
1c00f266:	5c76                	lw	s8,124(sp)
1c00f268:	5ce6                	lw	s9,120(sp)
1c00f26a:	5d56                	lw	s10,116(sp)
1c00f26c:	5dc6                	lw	s11,112(sp)
1c00f26e:	610d                	addi	sp,sp,160
1c00f270:	8082                	ret
1c00f272:	12e55063          	ble	a4,a0,1c00f392 <mm_M_unroll_2x4+0x40e>
1c00f276:	862e                	mv	a2,a1
1c00f278:	45a6                	lw	a1,72(sp)
1c00f27a:	4816                	lw	a6,68(sp)
1c00f27c:	00251693          	slli	a3,a0,0x2
1c00f280:	00d58d33          	add	s10,a1,a3
1c00f284:	96b2                	add	a3,a3,a2
1c00f286:	fff80793          	addi	a5,a6,-1
1c00f28a:	c436                	sw	a3,8(sp)
1c00f28c:	56c2                	lw	a3,48(sp)
1c00f28e:	02fb0733          	mul	a4,s6,a5
1c00f292:	fc183533          	p.bclr	a0,a6,30,1
1c00f296:	c82a                	sw	a0,16(sp)
1c00f298:	5552                	lw	a0,52(sp)
1c00f29a:	00369c93          	slli	s9,a3,0x3
1c00f29e:	00269a93          	slli	s5,a3,0x2
1c00f2a2:	003b1813          	slli	a6,s6,0x3
1c00f2a6:	c242                	sw	a6,4(sp)
1c00f2a8:	002b1d93          	slli	s11,s6,0x2
1c00f2ac:	02f687b3          	mul	a5,a3,a5
1c00f2b0:	00eb06b3          	add	a3,s6,a4
1c00f2b4:	068a                	slli	a3,a3,0x2
1c00f2b6:	070a                	slli	a4,a4,0x2
1c00f2b8:	972a                	add	a4,a4,a0
1c00f2ba:	96aa                	add	a3,a3,a0
1c00f2bc:	ce3a                	sw	a4,28(sp)
1c00f2be:	cc36                	sw	a3,24(sp)
1c00f2c0:	078a                	slli	a5,a5,0x2
1c00f2c2:	ca3e                	sw	a5,20(sp)
1c00f2c4:	55a2                	lw	a1,40(sp)
1c00f2c6:	c9c5                	beqz	a1,1c00f376 <mm_M_unroll_2x4+0x3f2>
1c00f2c8:	4622                	lw	a2,8(sp)
1c00f2ca:	fff58c13          	addi	s8,a1,-1
1c00f2ce:	001c5c13          	srli	s8,s8,0x1
1c00f2d2:	00158793          	addi	a5,a1,1
1c00f2d6:	4709                	li	a4,2
1c00f2d8:	00ca89b3          	add	s3,s5,a2
1c00f2dc:	5bd2                	lw	s7,52(sp)
1c00f2de:	8a32                	mv	s4,a2
1c00f2e0:	0c05                	addi	s8,s8,1
1c00f2e2:	26e7c363          	blt	a5,a4,1c00f548 <mm_M_unroll_2x4+0x5c4>
1c00f2e6:	0001                	nop
1c00f2e8:	046c407b          	lp.setup	x0,s8,1c00f374 <mm_M_unroll_2x4+0x3f0>
1c00f2ec:	15605c63          	blez	s6,1c00f444 <mm_M_unroll_2x4+0x4c0>
1c00f2f0:	00000413          	li	s0,0
1c00f2f4:	ffcd8613          	addi	a2,s11,-4
1c00f2f8:	8209                	srli	a2,a2,0x2
1c00f2fa:	83a2                	mv	t2,s0
1c00f2fc:	82a2                	mv	t0,s0
1c00f2fe:	8fa2                	mv	t6,s0
1c00f300:	8f22                	mv	t5,s0
1c00f302:	8ea2                	mv	t4,s0
1c00f304:	8e22                	mv	t3,s0
1c00f306:	8322                	mv	t1,s0
1c00f308:	895e                	mv	s2,s7
1c00f30a:	87ea                	mv	a5,s10
1c00f30c:	017d84b3          	add	s1,s11,s7
1c00f310:	0605                	addi	a2,a2,1
1c00f312:	0001                	nop
1c00f314:	01c640fb          	lp.setup	x1,a2,1c00f34c <mm_M_unroll_2x4+0x3c8>
1c00f318:	0049268b          	p.lw	a3,4(s2!)
1c00f31c:	0044a70b          	p.lw	a4,4(s1!)
1c00f320:	0007a883          	lw	a7,0(a5)
1c00f324:	0047a803          	lw	a6,4(a5)
1c00f328:	4788                	lw	a0,8(a5)
1c00f32a:	47cc                	lw	a1,12(a5)
1c00f32c:	30d8f343          	fmadd.s	t1,a7,a3,t1
1c00f330:	e0e8fe43          	fmadd.s	t3,a7,a4,t3
1c00f334:	e906fec3          	fmadd.s	t4,a3,a6,t4
1c00f338:	f1077f43          	fmadd.s	t5,a4,a6,t5
1c00f33c:	f8a6ffc3          	fmadd.s	t6,a3,a0,t6
1c00f340:	28a772c3          	fmadd.s	t0,a4,a0,t0
1c00f344:	38b6f3c3          	fmadd.s	t2,a3,a1,t2
1c00f348:	40b77443          	fmadd.s	s0,a4,a1,s0
1c00f34c:	97d6                	add	a5,a5,s5
1c00f34e:	006a2023          	sw	t1,0(s4)
1c00f352:	01c9a023          	sw	t3,0(s3)
1c00f356:	01da2223          	sw	t4,4(s4)
1c00f35a:	01e9a223          	sw	t5,4(s3)
1c00f35e:	01fa2423          	sw	t6,8(s4)
1c00f362:	0059a423          	sw	t0,8(s3)
1c00f366:	4592                	lw	a1,4(sp)
1c00f368:	007a2623          	sw	t2,12(s4)
1c00f36c:	0089a623          	sw	s0,12(s3)
1c00f370:	9a66                	add	s4,s4,s9
1c00f372:	99e6                	add	s3,s3,s9
1c00f374:	9bae                	add	s7,s7,a1
1c00f376:	4642                	lw	a2,16(sp)
1c00f378:	010d0313          	addi	t1,s10,16
1c00f37c:	ea65                	bnez	a2,1c00f46c <mm_M_unroll_2x4+0x4e8>
1c00f37e:	47b2                	lw	a5,12(sp)
1c00f380:	4822                	lw	a6,8(sp)
1c00f382:	5572                	lw	a0,60(sp)
1c00f384:	0791                	addi	a5,a5,4
1c00f386:	0841                	addi	a6,a6,16
1c00f388:	c63e                	sw	a5,12(sp)
1c00f38a:	c442                	sw	a6,8(sp)
1c00f38c:	8d1a                	mv	s10,t1
1c00f38e:	f2a7cbe3          	blt	a5,a0,1c00f2c4 <mm_M_unroll_2x4+0x340>
1c00f392:	5686                	lw	a3,96(sp)
1c00f394:	ecd051e3          	blez	a3,1c00f256 <mm_M_unroll_2x4+0x2d2>
1c00f398:	4716                	lw	a4,68(sp)
1c00f39a:	4866                	lw	a6,88(sp)
1c00f39c:	00770793          	addi	a5,a4,7
1c00f3a0:	878d                	srai	a5,a5,0x3
1c00f3a2:	02f80333          	mul	t1,a6,a5
1c00f3a6:	979a                	add	a5,a5,t1
1c00f3a8:	04e7ceb3          	p.min	t4,a5,a4
1c00f3ac:	ebd355e3          	ble	t4,t1,1c00f256 <mm_M_unroll_2x4+0x2d2>
1c00f3b0:	55c2                	lw	a1,48(sp)
1c00f3b2:	5662                	lw	a2,56(sp)
1c00f3b4:	eab651e3          	ble	a1,a2,1c00f256 <mm_M_unroll_2x4+0x2d2>
1c00f3b8:	5752                	lw	a4,52(sp)
1c00f3ba:	88da                	mv	a7,s6
1c00f3bc:	426b08b3          	p.mac	a7,s6,t1
1c00f3c0:	002b1f93          	slli	t6,s6,0x2
1c00f3c4:	843a                	mv	s0,a4
1c00f3c6:	46a6                	lw	a3,72(sp)
1c00f3c8:	44f6                	lw	s1,92(sp)
1c00f3ca:	00261f13          	slli	t5,a2,0x2
1c00f3ce:	00259293          	slli	t0,a1,0x2
1c00f3d2:	9f36                	add	t5,t5,a3
1c00f3d4:	005683b3          	add	t2,a3,t0
1c00f3d8:	892e                	mv	s2,a1
1c00f3da:	02b30e33          	mul	t3,t1,a1
1c00f3de:	088a                	slli	a7,a7,0x2
1c00f3e0:	98ba                	add	a7,a7,a4
1c00f3e2:	89b2                	mv	s3,a2
1c00f3e4:	426f8433          	p.mac	s0,t6,t1
1c00f3e8:	41e38833          	sub	a6,t2,t5
1c00f3ec:	013e0a33          	add	s4,t3,s3
1c00f3f0:	1871                	addi	a6,a6,-4
1c00f3f2:	0a0a                	slli	s4,s4,0x2
1c00f3f4:	00285813          	srli	a6,a6,0x2
1c00f3f8:	9a26                	add	s4,s4,s1
1c00f3fa:	8afa                	mv	s5,t5
1c00f3fc:	0805                	addi	a6,a6,1
1c00f3fe:	0001                	nop
1c00f400:	01a8407b          	lp.setup	x0,a6,1c00f434 <mm_M_unroll_2x4+0x4b0>
1c00f404:	00000713          	li	a4,0
1c00f408:	03605463          	blez	s6,1c00f430 <mm_M_unroll_2x4+0x4ac>
1c00f40c:	408887b3          	sub	a5,a7,s0
1c00f410:	17f1                	addi	a5,a5,-4
1c00f412:	00000713          	li	a4,0
1c00f416:	8389                	srli	a5,a5,0x2
1c00f418:	8656                	mv	a2,s5
1c00f41a:	86a2                	mv	a3,s0
1c00f41c:	0785                	addi	a5,a5,1
1c00f41e:	0001                	nop
1c00f420:	0067c0fb          	lp.setup	x1,a5,1c00f42c <mm_M_unroll_2x4+0x4a8>
1c00f424:	0046a50b          	p.lw	a0,4(a3!)
1c00f428:	2056758b          	p.lw	a1,t0(a2!)
1c00f42c:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00f430:	00ea222b          	p.sw	a4,4(s4!)
1c00f434:	0a91                	addi	s5,s5,4
1c00f436:	0305                	addi	t1,t1,1
1c00f438:	9e4a                	add	t3,t3,s2
1c00f43a:	947e                	add	s0,s0,t6
1c00f43c:	98fe                	add	a7,a7,t6
1c00f43e:	fa6e95e3          	bne	t4,t1,1c00f3e8 <mm_M_unroll_2x4+0x464>
1c00f442:	bd11                	j	1c00f256 <mm_M_unroll_2x4+0x2d2>
1c00f444:	00000413          	li	s0,0
1c00f448:	83a2                	mv	t2,s0
1c00f44a:	82a2                	mv	t0,s0
1c00f44c:	8fa2                	mv	t6,s0
1c00f44e:	8f22                	mv	t5,s0
1c00f450:	8ea2                	mv	t4,s0
1c00f452:	8e22                	mv	t3,s0
1c00f454:	8322                	mv	t1,s0
1c00f456:	bde5                	j	1c00f34e <mm_M_unroll_2x4+0x3ca>
1c00f458:	00000393          	li	t2,0
1c00f45c:	829e                	mv	t0,t2
1c00f45e:	8f9e                	mv	t6,t2
1c00f460:	8f1e                	mv	t5,t2
1c00f462:	8e9e                	mv	t4,t2
1c00f464:	8e1e                	mv	t3,t2
1c00f466:	831e                	mv	t1,t2
1c00f468:	889e                	mv	a7,t2
1c00f46a:	b1f9                	j	1c00f138 <mm_M_unroll_2x4+0x1b4>
1c00f46c:	46d2                	lw	a3,20(sp)
1c00f46e:	4722                	lw	a4,8(sp)
1c00f470:	4ee2                	lw	t4,24(sp)
1c00f472:	4e72                	lw	t3,28(sp)
1c00f474:	00e688b3          	add	a7,a3,a4
1c00f478:	4811                	li	a6,4
1c00f47a:	0001                	nop
1c00f47c:	01a8407b          	lp.setup	x0,a6,1c00f4b0 <mm_M_unroll_2x4+0x52c>
1c00f480:	00000713          	li	a4,0
1c00f484:	03605463          	blez	s6,1c00f4ac <mm_M_unroll_2x4+0x528>
1c00f488:	41ce87b3          	sub	a5,t4,t3
1c00f48c:	17f1                	addi	a5,a5,-4
1c00f48e:	00000713          	li	a4,0
1c00f492:	8389                	srli	a5,a5,0x2
1c00f494:	866a                	mv	a2,s10
1c00f496:	86f2                	mv	a3,t3
1c00f498:	0785                	addi	a5,a5,1
1c00f49a:	0001                	nop
1c00f49c:	0067c0fb          	lp.setup	x1,a5,1c00f4a8 <mm_M_unroll_2x4+0x524>
1c00f4a0:	0046a50b          	p.lw	a0,4(a3!)
1c00f4a4:	2156758b          	p.lw	a1,s5(a2!)
1c00f4a8:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00f4ac:	00e8a22b          	p.sw	a4,4(a7!)
1c00f4b0:	0d11                	addi	s10,s10,4
1c00f4b2:	b5f1                	j	1c00f37e <mm_M_unroll_2x4+0x3fa>
1c00f4b4:	4756                	lw	a4,84(sp)
1c00f4b6:	56b6                	lw	a3,108(sp)
1c00f4b8:	57b2                	lw	a5,44(sp)
1c00f4ba:	4886                	lw	a7,64(sp)
1c00f4bc:	40e68833          	sub	a6,a3,a4
1c00f4c0:	1871                	addi	a6,a6,-4
1c00f4c2:	5e82                	lw	t4,32(sp)
1c00f4c4:	4f62                	lw	t5,24(sp)
1c00f4c6:	5f96                	lw	t6,100(sp)
1c00f4c8:	5e26                	lw	t3,104(sp)
1c00f4ca:	00285813          	srli	a6,a6,0x2
1c00f4ce:	00f70333          	add	t1,a4,a5
1c00f4d2:	0805                	addi	a6,a6,1
1c00f4d4:	01a8407b          	lp.setup	x0,a6,1c00f508 <mm_M_unroll_2x4+0x584>
1c00f4d8:	00000713          	li	a4,0
1c00f4dc:	03e05463          	blez	t5,1c00f504 <mm_M_unroll_2x4+0x580>
1c00f4e0:	41cf87b3          	sub	a5,t6,t3
1c00f4e4:	17f1                	addi	a5,a5,-4
1c00f4e6:	00000713          	li	a4,0
1c00f4ea:	8389                	srli	a5,a5,0x2
1c00f4ec:	8646                	mv	a2,a7
1c00f4ee:	86f2                	mv	a3,t3
1c00f4f0:	0785                	addi	a5,a5,1
1c00f4f2:	0001                	nop
1c00f4f4:	0067c0fb          	lp.setup	x1,a5,1c00f500 <mm_M_unroll_2x4+0x57c>
1c00f4f8:	0046a50b          	p.lw	a0,4(a3!)
1c00f4fc:	0046258b          	p.lw	a1,4(a2!)
1c00f500:	70b57743          	fmadd.s	a4,a0,a1,a4
1c00f504:	00e3222b          	p.sw	a4,4(t1!)
1c00f508:	98f6                	add	a7,a7,t4
1c00f50a:	b9b1                	j	1c00f166 <mm_M_unroll_2x4+0x1e2>
1c00f50c:	447a                	lw	s0,156(sp)
1c00f50e:	44ea                	lw	s1,152(sp)
1c00f510:	495a                	lw	s2,148(sp)
1c00f512:	49ca                	lw	s3,144(sp)
1c00f514:	4a3a                	lw	s4,140(sp)
1c00f516:	4aaa                	lw	s5,136(sp)
1c00f518:	4b1a                	lw	s6,132(sp)
1c00f51a:	4b8a                	lw	s7,128(sp)
1c00f51c:	5c76                	lw	s8,124(sp)
1c00f51e:	5ce6                	lw	s9,120(sp)
1c00f520:	5d56                	lw	s10,116(sp)
1c00f522:	5dc6                	lw	s11,112(sp)
1c00f524:	610d                	addi	sp,sp,160
1c00f526:	e3ffd06f          	j	1c00d364 <mm_M_unroll_8x1>
1c00f52a:	447a                	lw	s0,156(sp)
1c00f52c:	44ea                	lw	s1,152(sp)
1c00f52e:	495a                	lw	s2,148(sp)
1c00f530:	49ca                	lw	s3,144(sp)
1c00f532:	4a3a                	lw	s4,140(sp)
1c00f534:	4aaa                	lw	s5,136(sp)
1c00f536:	4b1a                	lw	s6,132(sp)
1c00f538:	4b8a                	lw	s7,128(sp)
1c00f53a:	5c76                	lw	s8,124(sp)
1c00f53c:	5ce6                	lw	s9,120(sp)
1c00f53e:	5d56                	lw	s10,116(sp)
1c00f540:	5dc6                	lw	s11,112(sp)
1c00f542:	610d                	addi	sp,sp,160
1c00f544:	ee1fe06f          	j	1c00e424 <mm_M_unroll_2x2>
1c00f548:	4c05                	li	s8,1
1c00f54a:	bb71                	j	1c00f2e6 <mm_M_unroll_2x4+0x362>
1c00f54c:	4c85                	li	s9,1
1c00f54e:	be9d                	j	1c00f0c4 <mm_M_unroll_2x4+0x140>

1c00f550 <mm_M_unroll_4x4>:
1c00f550:	01052303          	lw	t1,16(a0)
1c00f554:	7155                	addi	sp,sp,-208
1c00f556:	c7a2                	sw	s0,204(sp)
1c00f558:	c20335b3          	p.bclr	a1,t1,1,0
1c00f55c:	c5a6                	sw	s1,200(sp)
1c00f55e:	c3ca                	sw	s2,196(sp)
1c00f560:	c1ce                	sw	s3,192(sp)
1c00f562:	df52                	sw	s4,188(sp)
1c00f564:	dd56                	sw	s5,184(sp)
1c00f566:	db5a                	sw	s6,180(sp)
1c00f568:	d95e                	sw	s7,176(sp)
1c00f56a:	d762                	sw	s8,172(sp)
1c00f56c:	d566                	sw	s9,168(sp)
1c00f56e:	d36a                	sw	s10,164(sp)
1c00f570:	d16e                	sw	s11,160(sp)
1c00f572:	d89a                	sw	t1,112(sp)
1c00f574:	473d                	li	a4,15
1c00f576:	d6ae                	sw	a1,108(sp)
1c00f578:	f14026f3          	csrr	a3,mhartid
1c00f57c:	26b75963          	ble	a1,a4,1c00f7ee <mm_M_unroll_4x4+0x29e>
1c00f580:	477d                	li	a4,31
1c00f582:	24b75763          	ble	a1,a4,1c00f7d0 <mm_M_unroll_4x4+0x280>
1c00f586:	4558                	lw	a4,12(a0)
1c00f588:	87aa                	mv	a5,a0
1c00f58a:	daba                	sw	a4,116(sp)
1c00f58c:	5856                	lw	a6,116(sp)
1c00f58e:	470d                	li	a4,3
1c00f590:	4b075563          	ble	a6,a4,1c00fa3a <mm_M_unroll_4x4+0x4ea>
1c00f594:	00758713          	addi	a4,a1,7
1c00f598:	f456b6b3          	p.bclr	a3,a3,26,5
1c00f59c:	870d                	srai	a4,a4,0x3
1c00f59e:	02e68333          	mul	t1,a3,a4
1c00f5a2:	88ae                	mv	a7,a1
1c00f5a4:	4790                	lw	a2,8(a5)
1c00f5a6:	43cc                	lw	a1,4(a5)
1c00f5a8:	4bdc                	lw	a5,20(a5)
1c00f5aa:	c736                	sw	a3,140(sp)
1c00f5ac:	4d14                	lw	a3,24(a0)
1c00f5ae:	c6be                	sw	a5,76(sp)
1c00f5b0:	57c6                	lw	a5,112(sp)
1c00f5b2:	4108                	lw	a0,0(a0)
1c00f5b4:	971a                	add	a4,a4,t1
1c00f5b6:	411787b3          	sub	a5,a5,a7
1c00f5ba:	05174733          	p.min	a4,a4,a7
1c00f5be:	ca9a                	sw	t1,84(sp)
1c00f5c0:	c12a                	sw	a0,128(sp)
1c00f5c2:	c32e                	sw	a1,132(sp)
1c00f5c4:	c932                	sw	a2,144(sp)
1c00f5c6:	cb3e                	sw	a5,148(sp)
1c00f5c8:	deba                	sw	a4,124(sp)
1c00f5ca:	24069163          	bnez	a3,1c00f80c <mm_M_unroll_4x4+0x2bc>
1c00f5ce:	889a                	mv	a7,t1
1c00f5d0:	5376                	lw	t1,124(sp)
1c00f5d2:	4a68d363          	ble	t1,a7,1c00fa78 <mm_M_unroll_4x4+0x528>
1c00f5d6:	47b6                	lw	a5,76(sp)
1c00f5d8:	55c6                	lw	a1,112(sp)
1c00f5da:	c2083533          	p.bclr	a0,a6,1,0
1c00f5de:	078a                	slli	a5,a5,0x2
1c00f5e0:	dc3e                	sw	a5,56(sp)
1c00f5e2:	400007b7          	lui	a5,0x40000
1c00f5e6:	17fd                	addi	a5,a5,-1
1c00f5e8:	00259f93          	slli	t6,a1,0x2
1c00f5ec:	97aa                	add	a5,a5,a0
1c00f5ee:	03f787b3          	mul	a5,a5,t6
1c00f5f2:	01158733          	add	a4,a1,a7
1c00f5f6:	5856                	lw	a6,116(sp)
1c00f5f8:	8332                	mv	t1,a2
1c00f5fa:	070a                	slli	a4,a4,0x2
1c00f5fc:	40b006b3          	neg	a3,a1
1c00f600:	971a                	add	a4,a4,t1
1c00f602:	c8aa                	sw	a0,80(sp)
1c00f604:	fa283833          	p.bclr	a6,a6,29,2
1c00f608:	c4ba                	sw	a4,72(sp)
1c00f60a:	00369713          	slli	a4,a3,0x3
1c00f60e:	ccc2                	sw	a6,88(sp)
1c00f610:	d2be                	sw	a5,100(sp)
1c00f612:	d0ba                	sw	a4,96(sp)
1c00f614:	47c6                	lw	a5,80(sp)
1c00f616:	470a                	lw	a4,128(sp)
1c00f618:	5862                	lw	a6,56(sp)
1c00f61a:	00289613          	slli	a2,a7,0x2
1c00f61e:	489a                	lw	a7,132(sp)
1c00f620:	43078733          	p.mac	a4,a5,a6
1c00f624:	00459513          	slli	a0,a1,0x4
1c00f628:	9646                	add	a2,a2,a7
1c00f62a:	00269593          	slli	a1,a3,0x2
1c00f62e:	de32                	sw	a2,60(sp)
1c00f630:	46b6                	lw	a3,76(sp)
1c00f632:	4636                	lw	a2,76(sp)
1c00f634:	d82a                	sw	a0,48(sp)
1c00f636:	068e                	slli	a3,a3,0x3
1c00f638:	0612                	slli	a2,a2,0x4
1c00f63a:	ceae                	sw	a1,92(sp)
1c00f63c:	c0b2                	sw	a2,64(sp)
1c00f63e:	c2b6                	sw	a3,68(sp)
1c00f640:	d4ba                	sw	a4,104(sp)
1c00f642:	d47e                	sw	t6,40(sp)
1c00f644:	4846                	lw	a6,80(sp)
1c00f646:	40080963          	beqz	a6,1c00fa58 <mm_M_unroll_4x4+0x508>
1c00f64a:	4326                	lw	t1,72(sp)
1c00f64c:	48f6                	lw	a7,92(sp)
1c00f64e:	5506                	lw	a0,96(sp)
1c00f650:	55a2                	lw	a1,40(sp)
1c00f652:	989a                	add	a7,a7,t1
1c00f654:	fff80793          	addi	a5,a6,-1
1c00f658:	460a                	lw	a2,128(sp)
1c00f65a:	40a88533          	sub	a0,a7,a0
1c00f65e:	8389                	srli	a5,a5,0x2
1c00f660:	95aa                	add	a1,a1,a0
1c00f662:	0785                	addi	a5,a5,1
1c00f664:	ce46                	sw	a7,28(sp)
1c00f666:	cc2a                	sw	a0,24(sp)
1c00f668:	00380693          	addi	a3,a6,3
1c00f66c:	4711                	li	a4,4
1c00f66e:	d02e                	sw	a1,32(sp)
1c00f670:	d632                	sw	a2,44(sp)
1c00f672:	d21a                	sw	t1,36(sp)
1c00f674:	da3e                	sw	a5,52(sp)
1c00f676:	12e6f863          	bleu	a4,a3,1c00f7a6 <mm_M_unroll_4x4+0x256>
1c00f67a:	a5a5                	j	1c00fce2 <mm_M_unroll_4x4+0x792>
1c00f67c:	58e2                	lw	a7,56(sp)
1c00f67e:	5332                	lw	t1,44(sp)
1c00f680:	4516                	lw	a0,68(sp)
1c00f682:	00000b93          	li	s7,0
1c00f686:	00688d33          	add	s10,a7,t1
1c00f68a:	011d0cb3          	add	s9,s10,a7
1c00f68e:	011c8c33          	add	s8,s9,a7
1c00f692:	40ac0e33          	sub	t3,s8,a0
1c00f696:	406e0e33          	sub	t3,t3,t1
1c00f69a:	1e71                	addi	t3,t3,-4
1c00f69c:	8b5e                	mv	s6,s7
1c00f69e:	8ade                	mv	s5,s7
1c00f6a0:	8a5e                	mv	s4,s7
1c00f6a2:	89de                	mv	s3,s7
1c00f6a4:	895e                	mv	s2,s7
1c00f6a6:	84de                	mv	s1,s7
1c00f6a8:	845e                	mv	s0,s7
1c00f6aa:	83de                	mv	t2,s7
1c00f6ac:	82de                	mv	t0,s7
1c00f6ae:	8fde                	mv	t6,s7
1c00f6b0:	8f5e                	mv	t5,s7
1c00f6b2:	8ede                	mv	t4,s7
1c00f6b4:	57f2                	lw	a5,60(sp)
1c00f6b6:	002e5e13          	srli	t3,t3,0x2
1c00f6ba:	ca02                	sw	zero,20(sp)
1c00f6bc:	c602                	sw	zero,12(sp)
1c00f6be:	c802                	sw	zero,16(sp)
1c00f6c0:	8d9a                	mv	s11,t1
1c00f6c2:	0e05                	addi	t3,t3,1
1c00f6c4:	038e40fb          	lp.setup	x1,t3,1c00f734 <mm_M_unroll_4x4+0x1e4>
1c00f6c8:	0007a303          	lw	t1,0(a5) # 40000000 <__l2_shared_end+0x23fe7af4>
1c00f6cc:	0047a883          	lw	a7,4(a5)
1c00f6d0:	0087a803          	lw	a6,8(a5)
1c00f6d4:	47d8                	lw	a4,12(a5)
1c00f6d6:	004da50b          	p.lw	a0,4(s11!)
1c00f6da:	004d258b          	p.lw	a1,4(s10!)
1c00f6de:	004ca60b          	p.lw	a2,4(s9!)
1c00f6e2:	004c268b          	p.lw	a3,4(s8!)
1c00f6e6:	e8a37ec3          	fmadd.s	t4,t1,a0,t4
1c00f6ea:	f8b37fc3          	fmadd.s	t6,t1,a1,t6
1c00f6ee:	38c373c3          	fmadd.s	t2,t1,a2,t2
1c00f6f2:	48d374c3          	fmadd.s	s1,t1,a3,s1
1c00f6f6:	991579c3          	fmadd.s	s3,a0,a7,s3
1c00f6fa:	4332                	lw	t1,12(sp)
1c00f6fc:	a915fac3          	fmadd.s	s5,a1,a7,s5
1c00f700:	f1057f43          	fmadd.s	t5,a0,a6,t5
1c00f704:	2905f2c3          	fmadd.s	t0,a1,a6,t0
1c00f708:	a0e57a43          	fmadd.s	s4,a0,a4,s4
1c00f70c:	b0e5fb43          	fmadd.s	s6,a1,a4,s6
1c00f710:	4542                	lw	a0,16(sp)
1c00f712:	45d2                	lw	a1,20(sp)
1c00f714:	3116f343          	fmadd.s	t1,a3,a7,t1
1c00f718:	50e67543          	fmadd.s	a0,a2,a4,a0
1c00f71c:	58e6f5c3          	fmadd.s	a1,a3,a4,a1
1c00f720:	b9167bc3          	fmadd.s	s7,a2,a7,s7
1c00f724:	41067443          	fmadd.s	s0,a2,a6,s0
1c00f728:	5622                	lw	a2,40(sp)
1c00f72a:	c61a                	sw	t1,12(sp)
1c00f72c:	c82a                	sw	a0,16(sp)
1c00f72e:	ca2e                	sw	a1,20(sp)
1c00f730:	9106f943          	fmadd.s	s2,a3,a6,s2
1c00f734:	97b2                	add	a5,a5,a2
1c00f736:	46f2                	lw	a3,28(sp)
1c00f738:	5712                	lw	a4,36(sp)
1c00f73a:	47e2                	lw	a5,24(sp)
1c00f73c:	01d6a023          	sw	t4,0(a3)
1c00f740:	5802                	lw	a6,32(sp)
1c00f742:	01f72023          	sw	t6,0(a4)
1c00f746:	0077a023          	sw	t2,0(a5)
1c00f74a:	00982023          	sw	s1,0(a6)
1c00f74e:	0136a223          	sw	s3,4(a3)
1c00f752:	48b2                	lw	a7,12(sp)
1c00f754:	01572223          	sw	s5,4(a4)
1c00f758:	0177a223          	sw	s7,4(a5)
1c00f75c:	01182223          	sw	a7,4(a6)
1c00f760:	01e6a423          	sw	t5,8(a3)
1c00f764:	00572423          	sw	t0,8(a4)
1c00f768:	55c2                	lw	a1,48(sp)
1c00f76a:	c780                	sw	s0,8(a5)
1c00f76c:	01282423          	sw	s2,8(a6)
1c00f770:	0146a623          	sw	s4,12(a3)
1c00f774:	4342                	lw	t1,16(sp)
1c00f776:	96ae                	add	a3,a3,a1
1c00f778:	01672623          	sw	s6,12(a4)
1c00f77c:	5632                	lw	a2,44(sp)
1c00f77e:	4552                	lw	a0,20(sp)
1c00f780:	ce36                	sw	a3,28(sp)
1c00f782:	972e                	add	a4,a4,a1
1c00f784:	4686                	lw	a3,64(sp)
1c00f786:	d23a                	sw	a4,36(sp)
1c00f788:	5752                	lw	a4,52(sp)
1c00f78a:	0067a623          	sw	t1,12(a5)
1c00f78e:	00a82623          	sw	a0,12(a6)
1c00f792:	97ae                	add	a5,a5,a1
1c00f794:	982e                	add	a6,a6,a1
1c00f796:	9636                	add	a2,a2,a3
1c00f798:	cc3e                	sw	a5,24(sp)
1c00f79a:	d042                	sw	a6,32(sp)
1c00f79c:	d632                	sw	a2,44(sp)
1c00f79e:	177d                	addi	a4,a4,-1
1c00f7a0:	da3a                	sw	a4,52(sp)
1c00f7a2:	2a070b63          	beqz	a4,1c00fa58 <mm_M_unroll_4x4+0x508>
1c00f7a6:	47b6                	lw	a5,76(sp)
1c00f7a8:	ecf04ae3          	bgtz	a5,1c00f67c <mm_M_unroll_4x4+0x12c>
1c00f7ac:	00000b93          	li	s7,0
1c00f7b0:	ca02                	sw	zero,20(sp)
1c00f7b2:	c602                	sw	zero,12(sp)
1c00f7b4:	c802                	sw	zero,16(sp)
1c00f7b6:	8b5e                	mv	s6,s7
1c00f7b8:	8ade                	mv	s5,s7
1c00f7ba:	8a5e                	mv	s4,s7
1c00f7bc:	89de                	mv	s3,s7
1c00f7be:	895e                	mv	s2,s7
1c00f7c0:	84de                	mv	s1,s7
1c00f7c2:	845e                	mv	s0,s7
1c00f7c4:	83de                	mv	t2,s7
1c00f7c6:	82de                	mv	t0,s7
1c00f7c8:	8fde                	mv	t6,s7
1c00f7ca:	8f5e                	mv	t5,s7
1c00f7cc:	8ede                	mv	t4,s7
1c00f7ce:	b7a5                	j	1c00f736 <mm_M_unroll_4x4+0x1e6>
1c00f7d0:	443e                	lw	s0,204(sp)
1c00f7d2:	44ae                	lw	s1,200(sp)
1c00f7d4:	491e                	lw	s2,196(sp)
1c00f7d6:	498e                	lw	s3,192(sp)
1c00f7d8:	5a7a                	lw	s4,188(sp)
1c00f7da:	5aea                	lw	s5,184(sp)
1c00f7dc:	5b5a                	lw	s6,180(sp)
1c00f7de:	5bca                	lw	s7,176(sp)
1c00f7e0:	5c3a                	lw	s8,172(sp)
1c00f7e2:	5caa                	lw	s9,168(sp)
1c00f7e4:	5d1a                	lw	s10,164(sp)
1c00f7e6:	5d8a                	lw	s11,160(sp)
1c00f7e8:	6169                	addi	sp,sp,208
1c00f7ea:	91eff06f          	j	1c00e908 <mm_M_unroll_4x2>
1c00f7ee:	443e                	lw	s0,204(sp)
1c00f7f0:	44ae                	lw	s1,200(sp)
1c00f7f2:	491e                	lw	s2,196(sp)
1c00f7f4:	498e                	lw	s3,192(sp)
1c00f7f6:	5a7a                	lw	s4,188(sp)
1c00f7f8:	5aea                	lw	s5,184(sp)
1c00f7fa:	5b5a                	lw	s6,180(sp)
1c00f7fc:	5bca                	lw	s7,176(sp)
1c00f7fe:	5c3a                	lw	s8,172(sp)
1c00f800:	5caa                	lw	s9,168(sp)
1c00f802:	5d1a                	lw	s10,164(sp)
1c00f804:	5d8a                	lw	s11,160(sp)
1c00f806:	6169                	addi	sp,sp,208
1c00f808:	b5dfd06f          	j	1c00d364 <mm_M_unroll_8x1>
1c00f80c:	34e35c63          	ble	a4,t1,1c00fb64 <mm_M_unroll_4x4+0x614>
1c00f810:	47b6                	lw	a5,76(sp)
1c00f812:	88c2                	mv	a7,a6
1c00f814:	c208b8b3          	p.bclr	a7,a7,1,0
1c00f818:	0792                	slli	a5,a5,0x4
1c00f81a:	d4c6                	sw	a7,104(sp)
1c00f81c:	c4be                	sw	a5,72(sp)
1c00f81e:	48d6                	lw	a7,84(sp)
1c00f820:	47b6                	lw	a5,76(sp)
1c00f822:	8542                	mv	a0,a6
1c00f824:	4836                	lw	a6,76(sp)
1c00f826:	431787b3          	p.mac	a5,a5,a7
1c00f82a:	00130613          	addi	a2,t1,1
1c00f82e:	00230693          	addi	a3,t1,2
1c00f832:	00330713          	addi	a4,t1,3
1c00f836:	00231593          	slli	a1,t1,0x2
1c00f83a:	5346                	lw	t1,112(sp)
1c00f83c:	080a                	slli	a6,a6,0x2
1c00f83e:	c0c2                	sw	a6,64(sp)
1c00f840:	030a                	slli	t1,t1,0x2
1c00f842:	fa253533          	p.bclr	a0,a0,29,2
1c00f846:	dc9a                	sw	t1,120(sp)
1c00f848:	c52a                	sw	a0,136(sp)
1c00f84a:	431a                	lw	t1,132(sp)
1c00f84c:	4506                	lw	a0,64(sp)
1c00f84e:	484a                	lw	a6,144(sp)
1c00f850:	078a                	slli	a5,a5,0x2
1c00f852:	42a88333          	p.mac	t1,a7,a0
1c00f856:	489a                	lw	a7,132(sp)
1c00f858:	95c2                	add	a1,a1,a6
1c00f85a:	d2ae                	sw	a1,100(sp)
1c00f85c:	97c6                	add	a5,a5,a7
1c00f85e:	459a                	lw	a1,132(sp)
1c00f860:	c8be                	sw	a5,80(sp)
1c00f862:	5826                	lw	a6,104(sp)
1c00f864:	478a                	lw	a5,128(sp)
1c00f866:	42a608b3          	p.mac	a7,a2,a0
1c00f86a:	461a                	lw	a2,132(sp)
1c00f86c:	c29a                	sw	t1,68(sp)
1c00f86e:	5346                	lw	t1,112(sp)
1c00f870:	0312                	slli	t1,t1,0x4
1c00f872:	da1a                	sw	t1,52(sp)
1c00f874:	42a685b3          	p.mac	a1,a3,a0
1c00f878:	56a6                	lw	a3,104(sp)
1c00f87a:	d0c6                	sw	a7,96(sp)
1c00f87c:	42a70633          	p.mac	a2,a4,a0
1c00f880:	5766                	lw	a4,120(sp)
1c00f882:	ceae                	sw	a1,92(sp)
1c00f884:	02e686b3          	mul	a3,a3,a4
1c00f888:	ccb2                	sw	a2,88(sp)
1c00f88a:	42a807b3          	p.mac	a5,a6,a0
1c00f88e:	cd36                	sw	a3,152(sp)
1c00f890:	cf3e                	sw	a5,156(sp)
1c00f892:	5526                	lw	a0,104(sp)
1c00f894:	28050d63          	beqz	a0,1c00fb2e <mm_M_unroll_4x4+0x5de>
1c00f898:	5616                	lw	a2,100(sp)
1c00f89a:	55e6                	lw	a1,120(sp)
1c00f89c:	5766                	lw	a4,120(sp)
1c00f89e:	58e6                	lw	a7,120(sp)
1c00f8a0:	95b2                	add	a1,a1,a2
1c00f8a2:	972e                	add	a4,a4,a1
1c00f8a4:	d23a                	sw	a4,36(sp)
1c00f8a6:	5812                	lw	a6,36(sp)
1c00f8a8:	fff50693          	addi	a3,a0,-1
1c00f8ac:	430a                	lw	t1,128(sp)
1c00f8ae:	8289                	srli	a3,a3,0x2
1c00f8b0:	9846                	add	a6,a6,a7
1c00f8b2:	0685                	addi	a3,a3,1
1c00f8b4:	d42e                	sw	a1,40(sp)
1c00f8b6:	00350793          	addi	a5,a0,3
1c00f8ba:	4711                	li	a4,4
1c00f8bc:	d642                	sw	a6,44(sp)
1c00f8be:	dc1a                	sw	t1,56(sp)
1c00f8c0:	d832                	sw	a2,48(sp)
1c00f8c2:	de36                	sw	a3,60(sp)
1c00f8c4:	14e7f663          	bleu	a4,a5,1c00fa10 <mm_M_unroll_4x4+0x4c0>
1c00f8c8:	a911                	j	1c00fcdc <mm_M_unroll_4x4+0x78c>
1c00f8ca:	4816                	lw	a6,68(sp)
1c00f8cc:	47c6                	lw	a5,80(sp)
1c00f8ce:	00000993          	li	s3,0
1c00f8d2:	4506                	lw	a0,64(sp)
1c00f8d4:	58e2                	lw	a7,56(sp)
1c00f8d6:	41078333          	sub	t1,a5,a6
1c00f8da:	1371                	addi	t1,t1,-4
1c00f8dc:	00a88ab3          	add	s5,a7,a0
1c00f8e0:	894e                	mv	s2,s3
1c00f8e2:	84ce                	mv	s1,s3
1c00f8e4:	844e                	mv	s0,s3
1c00f8e6:	83ce                	mv	t2,s3
1c00f8e8:	82ce                	mv	t0,s3
1c00f8ea:	8fce                	mv	t6,s3
1c00f8ec:	8f4e                	mv	t5,s3
1c00f8ee:	8ece                	mv	t4,s3
1c00f8f0:	8e4e                	mv	t3,s3
1c00f8f2:	4d66                	lw	s10,88(sp)
1c00f8f4:	4cf6                	lw	s9,92(sp)
1c00f8f6:	5c06                	lw	s8,96(sp)
1c00f8f8:	00aa8a33          	add	s4,s5,a0
1c00f8fc:	00235313          	srli	t1,t1,0x2
1c00f900:	d002                	sw	zero,32(sp)
1c00f902:	ca02                	sw	zero,20(sp)
1c00f904:	ce02                	sw	zero,28(sp)
1c00f906:	c802                	sw	zero,16(sp)
1c00f908:	cc02                	sw	zero,24(sp)
1c00f90a:	c602                	sw	zero,12(sp)
1c00f90c:	00aa0db3          	add	s11,s4,a0
1c00f910:	8bc6                	mv	s7,a7
1c00f912:	8b42                	mv	s6,a6
1c00f914:	0305                	addi	t1,t1,1
1c00f916:	0001                	nop
1c00f918:	03d340fb          	lp.setup	x1,t1,1c00f992 <mm_M_unroll_4x4+0x442>
1c00f91c:	004b288b          	p.lw	a7,4(s6!)
1c00f920:	004ba58b          	p.lw	a1,4(s7!)
1c00f924:	004aa60b          	p.lw	a2,4(s5!)
1c00f928:	004a268b          	p.lw	a3,4(s4!)
1c00f92c:	004da70b          	p.lw	a4,4(s11!)
1c00f930:	004c280b          	p.lw	a6,4(s8!)
1c00f934:	e0b8fe43          	fmadd.s	t3,a7,a1,t3
1c00f938:	f0c8ff43          	fmadd.s	t5,a7,a2,t5
1c00f93c:	28d8f2c3          	fmadd.s	t0,a7,a3,t0
1c00f940:	40e8f443          	fmadd.s	s0,a7,a4,s0
1c00f944:	48b2                	lw	a7,12(sp)
1c00f946:	004ca50b          	p.lw	a0,4(s9!)
1c00f94a:	004d278b          	p.lw	a5,4(s10!)
1c00f94e:	890678c3          	fmadd.s	a7,a2,a6,a7
1c00f952:	e8a5fec3          	fmadd.s	t4,a1,a0,t4
1c00f956:	f8a67fc3          	fmadd.s	t6,a2,a0,t6
1c00f95a:	38a6f3c3          	fmadd.s	t2,a3,a0,t2
1c00f95e:	48a774c3          	fmadd.s	s1,a4,a0,s1
1c00f962:	c646                	sw	a7,12(sp)
1c00f964:	48c2                	lw	a7,16(sp)
1c00f966:	4562                	lw	a0,24(sp)
1c00f968:	9105f943          	fmadd.s	s2,a1,a6,s2
1c00f96c:	8906f8c3          	fmadd.s	a7,a3,a6,a7
1c00f970:	50f67543          	fmadd.s	a0,a2,a5,a0
1c00f974:	98f5f9c3          	fmadd.s	s3,a1,a5,s3
1c00f978:	5602                	lw	a2,32(sp)
1c00f97a:	45f2                	lw	a1,28(sp)
1c00f97c:	c846                	sw	a7,16(sp)
1c00f97e:	48d2                	lw	a7,20(sp)
1c00f980:	58f6f5c3          	fmadd.s	a1,a3,a5,a1
1c00f984:	60f77643          	fmadd.s	a2,a4,a5,a2
1c00f988:	890778c3          	fmadd.s	a7,a4,a6,a7
1c00f98c:	cc2a                	sw	a0,24(sp)
1c00f98e:	ce2e                	sw	a1,28(sp)
1c00f990:	d032                	sw	a2,32(sp)
1c00f992:	ca46                	sw	a7,20(sp)
1c00f994:	56c2                	lw	a3,48(sp)
1c00f996:	5722                	lw	a4,40(sp)
1c00f998:	5792                	lw	a5,36(sp)
1c00f99a:	01c6a023          	sw	t3,0(a3)
1c00f99e:	5832                	lw	a6,44(sp)
1c00f9a0:	01e72023          	sw	t5,0(a4)
1c00f9a4:	0057a023          	sw	t0,0(a5)
1c00f9a8:	48b2                	lw	a7,12(sp)
1c00f9aa:	00882023          	sw	s0,0(a6)
1c00f9ae:	4342                	lw	t1,16(sp)
1c00f9b0:	0126a223          	sw	s2,4(a3)
1c00f9b4:	4552                	lw	a0,20(sp)
1c00f9b6:	01172223          	sw	a7,4(a4)
1c00f9ba:	0067a223          	sw	t1,4(a5)
1c00f9be:	00a82223          	sw	a0,4(a6)
1c00f9c2:	01d6a423          	sw	t4,8(a3)
1c00f9c6:	01f72423          	sw	t6,8(a4)
1c00f9ca:	0077a423          	sw	t2,8(a5)
1c00f9ce:	45e2                	lw	a1,24(sp)
1c00f9d0:	00982423          	sw	s1,8(a6)
1c00f9d4:	4672                	lw	a2,28(sp)
1c00f9d6:	0136a623          	sw	s3,12(a3)
1c00f9da:	5682                	lw	a3,32(sp)
1c00f9dc:	c74c                	sw	a1,12(a4)
1c00f9de:	c7d0                	sw	a2,12(a5)
1c00f9e0:	5742                	lw	a4,48(sp)
1c00f9e2:	57d2                	lw	a5,52(sp)
1c00f9e4:	00d82623          	sw	a3,12(a6)
1c00f9e8:	5822                	lw	a6,40(sp)
1c00f9ea:	5892                	lw	a7,36(sp)
1c00f9ec:	973e                	add	a4,a4,a5
1c00f9ee:	983e                	add	a6,a6,a5
1c00f9f0:	d83a                	sw	a4,48(sp)
1c00f9f2:	d442                	sw	a6,40(sp)
1c00f9f4:	98be                	add	a7,a7,a5
1c00f9f6:	d246                	sw	a7,36(sp)
1c00f9f8:	5332                	lw	t1,44(sp)
1c00f9fa:	5562                	lw	a0,56(sp)
1c00f9fc:	5672                	lw	a2,60(sp)
1c00f9fe:	45a6                	lw	a1,72(sp)
1c00fa00:	933e                	add	t1,t1,a5
1c00fa02:	167d                	addi	a2,a2,-1
1c00fa04:	952e                	add	a0,a0,a1
1c00fa06:	d61a                	sw	t1,44(sp)
1c00fa08:	dc2a                	sw	a0,56(sp)
1c00fa0a:	de32                	sw	a2,60(sp)
1c00fa0c:	12060163          	beqz	a2,1c00fb2e <mm_M_unroll_4x4+0x5de>
1c00fa10:	46b6                	lw	a3,76(sp)
1c00fa12:	ead04ce3          	bgtz	a3,1c00f8ca <mm_M_unroll_4x4+0x37a>
1c00fa16:	00000993          	li	s3,0
1c00fa1a:	d002                	sw	zero,32(sp)
1c00fa1c:	ca02                	sw	zero,20(sp)
1c00fa1e:	ce02                	sw	zero,28(sp)
1c00fa20:	c802                	sw	zero,16(sp)
1c00fa22:	cc02                	sw	zero,24(sp)
1c00fa24:	c602                	sw	zero,12(sp)
1c00fa26:	894e                	mv	s2,s3
1c00fa28:	84ce                	mv	s1,s3
1c00fa2a:	844e                	mv	s0,s3
1c00fa2c:	83ce                	mv	t2,s3
1c00fa2e:	82ce                	mv	t0,s3
1c00fa30:	8fce                	mv	t6,s3
1c00fa32:	8f4e                	mv	t5,s3
1c00fa34:	8ece                	mv	t4,s3
1c00fa36:	8e4e                	mv	t3,s3
1c00fa38:	bfb1                	j	1c00f994 <mm_M_unroll_4x4+0x444>
1c00fa3a:	443e                	lw	s0,204(sp)
1c00fa3c:	44ae                	lw	s1,200(sp)
1c00fa3e:	491e                	lw	s2,196(sp)
1c00fa40:	498e                	lw	s3,192(sp)
1c00fa42:	5a7a                	lw	s4,188(sp)
1c00fa44:	5aea                	lw	s5,184(sp)
1c00fa46:	5b5a                	lw	s6,180(sp)
1c00fa48:	5bca                	lw	s7,176(sp)
1c00fa4a:	5c3a                	lw	s8,172(sp)
1c00fa4c:	5caa                	lw	s9,168(sp)
1c00fa4e:	5d1a                	lw	s10,164(sp)
1c00fa50:	5d8a                	lw	s11,160(sp)
1c00fa52:	6169                	addi	sp,sp,208
1c00fa54:	9d1fe06f          	j	1c00e424 <mm_M_unroll_2x2>
1c00fa58:	5872                	lw	a6,60(sp)
1c00fa5a:	48e6                	lw	a7,88(sp)
1c00fa5c:	01080e93          	addi	t4,a6,16
1c00fa60:	20089963          	bnez	a7,1c00fc72 <mm_M_unroll_4x4+0x722>
1c00fa64:	46d6                	lw	a3,84(sp)
1c00fa66:	4726                	lw	a4,72(sp)
1c00fa68:	57f6                	lw	a5,124(sp)
1c00fa6a:	0691                	addi	a3,a3,4
1c00fa6c:	0741                	addi	a4,a4,16
1c00fa6e:	cab6                	sw	a3,84(sp)
1c00fa70:	de76                	sw	t4,60(sp)
1c00fa72:	c4ba                	sw	a4,72(sp)
1c00fa74:	bcf6c8e3          	blt	a3,a5,1c00f644 <mm_M_unroll_4x4+0xf4>
1c00fa78:	475a                	lw	a4,148(sp)
1c00fa7a:	08e05c63          	blez	a4,1c00fb12 <mm_M_unroll_4x4+0x5c2>
1c00fa7e:	5856                	lw	a6,116(sp)
1c00fa80:	48ba                	lw	a7,140(sp)
1c00fa82:	5346                	lw	t1,112(sp)
1c00fa84:	00780793          	addi	a5,a6,7
1c00fa88:	878d                	srai	a5,a5,0x3
1c00fa8a:	02f88fb3          	mul	t6,a7,a5
1c00fa8e:	55b6                	lw	a1,108(sp)
1c00fa90:	97fe                	add	a5,a5,t6
1c00fa92:	0507c2b3          	p.min	t0,a5,a6
1c00fa96:	0665de63          	ble	t1,a1,1c00fb12 <mm_M_unroll_4x4+0x5c2>
1c00fa9a:	065fdc63          	ble	t0,t6,1c00fb12 <mm_M_unroll_4x4+0x5c2>
1c00fa9e:	4636                	lw	a2,76(sp)
1c00faa0:	8f2e                	mv	t5,a1
1c00faa2:	438a                	lw	t2,128(sp)
1c00faa4:	43f30f33          	p.mac	t5,t1,t6
1c00faa8:	00261e93          	slli	t4,a2,0x2
1c00faac:	469a                	lw	a3,132(sp)
1c00faae:	474a                	lw	a4,144(sp)
1c00fab0:	00259e13          	slli	t3,a1,0x2
1c00fab4:	00231413          	slli	s0,t1,0x2
1c00fab8:	9e36                	add	t3,t3,a3
1c00faba:	008684b3          	add	s1,a3,s0
1c00fabe:	8932                	mv	s2,a2
1c00fac0:	43df83b3          	p.mac	t2,t6,t4
1c00fac4:	0f0a                	slli	t5,t5,0x2
1c00fac6:	9f3a                	add	t5,t5,a4
1c00fac8:	871e                	mv	a4,t2
1c00faca:	837a                	mv	t1,t5
1c00facc:	41f28833          	sub	a6,t0,t6
1c00fad0:	01c8407b          	lp.setup	x0,a6,1c00fb08 <mm_M_unroll_4x4+0x5b8>
1c00fad4:	00000693          	li	a3,0
1c00fad8:	00ee88b3          	add	a7,t4,a4
1c00fadc:	03205463          	blez	s2,1c00fb04 <mm_M_unroll_4x4+0x5b4>
1c00fae0:	00ee88b3          	add	a7,t4,a4
1c00fae4:	40e887b3          	sub	a5,a7,a4
1c00fae8:	17f1                	addi	a5,a5,-4
1c00faea:	00000693          	li	a3,0
1c00faee:	8389                	srli	a5,a5,0x2
1c00faf0:	8672                	mv	a2,t3
1c00faf2:	0785                	addi	a5,a5,1
1c00faf4:	0067c0fb          	lp.setup	x1,a5,1c00fb00 <mm_M_unroll_4x4+0x5b0>
1c00faf8:	0047250b          	p.lw	a0,4(a4!)
1c00fafc:	2086758b          	p.lw	a1,s0(a2!)
1c00fb00:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00fb04:	00d3642b          	p.sw	a3,s0(t1!)
1c00fb08:	8746                	mv	a4,a7
1c00fb0a:	0e11                	addi	t3,t3,4
1c00fb0c:	0f11                	addi	t5,t5,4
1c00fb0e:	fbc49de3          	bne	s1,t3,1c00fac8 <mm_M_unroll_4x4+0x578>
1c00fb12:	443e                	lw	s0,204(sp)
1c00fb14:	44ae                	lw	s1,200(sp)
1c00fb16:	491e                	lw	s2,196(sp)
1c00fb18:	498e                	lw	s3,192(sp)
1c00fb1a:	5a7a                	lw	s4,188(sp)
1c00fb1c:	5aea                	lw	s5,184(sp)
1c00fb1e:	5b5a                	lw	s6,180(sp)
1c00fb20:	5bca                	lw	s7,176(sp)
1c00fb22:	5c3a                	lw	s8,172(sp)
1c00fb24:	5caa                	lw	s9,168(sp)
1c00fb26:	5d1a                	lw	s10,164(sp)
1c00fb28:	5d8a                	lw	s11,160(sp)
1c00fb2a:	6169                	addi	sp,sp,208
1c00fb2c:	8082                	ret
1c00fb2e:	472a                	lw	a4,136(sp)
1c00fb30:	eb69                	bnez	a4,1c00fc02 <mm_M_unroll_4x4+0x6b2>
1c00fb32:	4626                	lw	a2,72(sp)
1c00fb34:	4556                	lw	a0,84(sp)
1c00fb36:	4596                	lw	a1,68(sp)
1c00fb38:	5696                	lw	a3,100(sp)
1c00fb3a:	5706                	lw	a4,96(sp)
1c00fb3c:	47f6                	lw	a5,92(sp)
1c00fb3e:	4866                	lw	a6,88(sp)
1c00fb40:	48c6                	lw	a7,80(sp)
1c00fb42:	5376                	lw	t1,124(sp)
1c00fb44:	0511                	addi	a0,a0,4
1c00fb46:	95b2                	add	a1,a1,a2
1c00fb48:	06c1                	addi	a3,a3,16
1c00fb4a:	9732                	add	a4,a4,a2
1c00fb4c:	97b2                	add	a5,a5,a2
1c00fb4e:	9832                	add	a6,a6,a2
1c00fb50:	98b2                	add	a7,a7,a2
1c00fb52:	caaa                	sw	a0,84(sp)
1c00fb54:	c2ae                	sw	a1,68(sp)
1c00fb56:	d2b6                	sw	a3,100(sp)
1c00fb58:	d0ba                	sw	a4,96(sp)
1c00fb5a:	cebe                	sw	a5,92(sp)
1c00fb5c:	ccc2                	sw	a6,88(sp)
1c00fb5e:	c8c6                	sw	a7,80(sp)
1c00fb60:	d26549e3          	blt	a0,t1,1c00f892 <mm_M_unroll_4x4+0x342>
1c00fb64:	45da                	lw	a1,148(sp)
1c00fb66:	fab056e3          	blez	a1,1c00fb12 <mm_M_unroll_4x4+0x5c2>
1c00fb6a:	5656                	lw	a2,116(sp)
1c00fb6c:	46ba                	lw	a3,140(sp)
1c00fb6e:	5746                	lw	a4,112(sp)
1c00fb70:	00760793          	addi	a5,a2,7
1c00fb74:	878d                	srai	a5,a5,0x3
1c00fb76:	02f683b3          	mul	t2,a3,a5
1c00fb7a:	979e                	add	a5,a5,t2
1c00fb7c:	04c7c2b3          	p.min	t0,a5,a2
1c00fb80:	57b6                	lw	a5,108(sp)
1c00fb82:	f8e7d8e3          	ble	a4,a5,1c00fb12 <mm_M_unroll_4x4+0x5c2>
1c00fb86:	f853d6e3          	ble	t0,t2,1c00fb12 <mm_M_unroll_4x4+0x5c2>
1c00fb8a:	4836                	lw	a6,76(sp)
1c00fb8c:	8fbe                	mv	t6,a5
1c00fb8e:	4e1a                	lw	t3,132(sp)
1c00fb90:	4f0a                	lw	t5,128(sp)
1c00fb92:	42770fb3          	p.mac	t6,a4,t2
1c00fb96:	00281e93          	slli	t4,a6,0x2
1c00fb9a:	48ca                	lw	a7,144(sp)
1c00fb9c:	00271413          	slli	s0,a4,0x2
1c00fba0:	89ba                	mv	s3,a4
1c00fba2:	84c2                	mv	s1,a6
1c00fba4:	893e                	mv	s2,a5
1c00fba6:	43d78e33          	p.mac	t3,a5,t4
1c00fbaa:	0f8a                	slli	t6,t6,0x2
1c00fbac:	9fc6                	add	t6,t6,a7
1c00fbae:	43d38f33          	p.mac	t5,t2,t4
1c00fbb2:	877a                	mv	a4,t5
1c00fbb4:	837e                	mv	t1,t6
1c00fbb6:	40728833          	sub	a6,t0,t2
1c00fbba:	0001                	nop
1c00fbbc:	01c8407b          	lp.setup	x0,a6,1c00fbf4 <mm_M_unroll_4x4+0x6a4>
1c00fbc0:	00000693          	li	a3,0
1c00fbc4:	00ee88b3          	add	a7,t4,a4
1c00fbc8:	02905463          	blez	s1,1c00fbf0 <mm_M_unroll_4x4+0x6a0>
1c00fbcc:	00ee88b3          	add	a7,t4,a4
1c00fbd0:	40e887b3          	sub	a5,a7,a4
1c00fbd4:	17f1                	addi	a5,a5,-4
1c00fbd6:	00000693          	li	a3,0
1c00fbda:	8389                	srli	a5,a5,0x2
1c00fbdc:	8672                	mv	a2,t3
1c00fbde:	0785                	addi	a5,a5,1
1c00fbe0:	0067c0fb          	lp.setup	x1,a5,1c00fbec <mm_M_unroll_4x4+0x69c>
1c00fbe4:	0047250b          	p.lw	a0,4(a4!)
1c00fbe8:	0046258b          	p.lw	a1,4(a2!)
1c00fbec:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00fbf0:	00d3642b          	p.sw	a3,s0(t1!)
1c00fbf4:	8746                	mv	a4,a7
1c00fbf6:	0905                	addi	s2,s2,1
1c00fbf8:	0f91                	addi	t6,t6,4
1c00fbfa:	9e76                	add	t3,t3,t4
1c00fbfc:	fb299be3          	bne	s3,s2,1c00fbb2 <mm_M_unroll_4x4+0x662>
1c00fc00:	bf09                	j	1c00fb12 <mm_M_unroll_4x4+0x5c2>
1c00fc02:	57d6                	lw	a5,116(sp)
1c00fc04:	5826                	lw	a6,104(sp)
1c00fc06:	f2f856e3          	ble	a5,a6,1c00fb32 <mm_M_unroll_4x4+0x5e2>
1c00fc0a:	48ea                	lw	a7,152(sp)
1c00fc0c:	5316                	lw	t1,100(sp)
1c00fc0e:	4e16                	lw	t3,68(sp)
1c00fc10:	42b6                	lw	t0,76(sp)
1c00fc12:	5966                	lw	s2,120(sp)
1c00fc14:	4f86                	lw	t6,64(sp)
1c00fc16:	44fa                	lw	s1,156(sp)
1c00fc18:	00688f33          	add	t5,a7,t1
1c00fc1c:	4e91                	li	t4,4
1c00fc1e:	83be                	mv	t2,a5
1c00fc20:	8442                	mv	s0,a6
1c00fc22:	8726                	mv	a4,s1
1c00fc24:	837a                	mv	t1,t5
1c00fc26:	40838833          	sub	a6,t2,s0
1c00fc2a:	0001                	nop
1c00fc2c:	01c8407b          	lp.setup	x0,a6,1c00fc64 <mm_M_unroll_4x4+0x714>
1c00fc30:	00000693          	li	a3,0
1c00fc34:	00ef88b3          	add	a7,t6,a4
1c00fc38:	02505463          	blez	t0,1c00fc60 <mm_M_unroll_4x4+0x710>
1c00fc3c:	00ef88b3          	add	a7,t6,a4
1c00fc40:	40e887b3          	sub	a5,a7,a4
1c00fc44:	17f1                	addi	a5,a5,-4
1c00fc46:	00000693          	li	a3,0
1c00fc4a:	8389                	srli	a5,a5,0x2
1c00fc4c:	8672                	mv	a2,t3
1c00fc4e:	0785                	addi	a5,a5,1
1c00fc50:	0067c0fb          	lp.setup	x1,a5,1c00fc5c <mm_M_unroll_4x4+0x70c>
1c00fc54:	0047250b          	p.lw	a0,4(a4!)
1c00fc58:	0046258b          	p.lw	a1,4(a2!)
1c00fc5c:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00fc60:	00d3692b          	p.sw	a3,s2(t1!)
1c00fc64:	8746                	mv	a4,a7
1c00fc66:	1efd                	addi	t4,t4,-1
1c00fc68:	0f11                	addi	t5,t5,4
1c00fc6a:	9e7e                	add	t3,t3,t6
1c00fc6c:	fa0e9be3          	bnez	t4,1c00fc22 <mm_M_unroll_4x4+0x6d2>
1c00fc70:	b5c9                	j	1c00fb32 <mm_M_unroll_4x4+0x5e2>
1c00fc72:	5356                	lw	t1,116(sp)
1c00fc74:	4546                	lw	a0,80(sp)
1c00fc76:	de6557e3          	ble	t1,a0,1c00fa64 <mm_M_unroll_4x4+0x514>
1c00fc7a:	5596                	lw	a1,100(sp)
1c00fc7c:	4626                	lw	a2,72(sp)
1c00fc7e:	42b6                	lw	t0,76(sp)
1c00fc80:	5fa2                	lw	t6,40(sp)
1c00fc82:	53e2                	lw	t2,56(sp)
1c00fc84:	5926                	lw	s2,104(sp)
1c00fc86:	00c58e33          	add	t3,a1,a2
1c00fc8a:	8f42                	mv	t5,a6
1c00fc8c:	842a                	mv	s0,a0
1c00fc8e:	849a                	mv	s1,t1
1c00fc90:	874a                	mv	a4,s2
1c00fc92:	8372                	mv	t1,t3
1c00fc94:	40848833          	sub	a6,s1,s0
1c00fc98:	01c8407b          	lp.setup	x0,a6,1c00fcd0 <mm_M_unroll_4x4+0x780>
1c00fc9c:	00000693          	li	a3,0
1c00fca0:	00e388b3          	add	a7,t2,a4
1c00fca4:	02505463          	blez	t0,1c00fccc <mm_M_unroll_4x4+0x77c>
1c00fca8:	00e388b3          	add	a7,t2,a4
1c00fcac:	40e887b3          	sub	a5,a7,a4
1c00fcb0:	17f1                	addi	a5,a5,-4
1c00fcb2:	00000693          	li	a3,0
1c00fcb6:	8389                	srli	a5,a5,0x2
1c00fcb8:	867a                	mv	a2,t5
1c00fcba:	0785                	addi	a5,a5,1
1c00fcbc:	0067c0fb          	lp.setup	x1,a5,1c00fcc8 <mm_M_unroll_4x4+0x778>
1c00fcc0:	0047250b          	p.lw	a0,4(a4!)
1c00fcc4:	21f6758b          	p.lw	a1,t6(a2!)
1c00fcc8:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c00fccc:	00d36fab          	p.sw	a3,t6(t1!)
1c00fcd0:	8746                	mv	a4,a7
1c00fcd2:	0f11                	addi	t5,t5,4
1c00fcd4:	0e11                	addi	t3,t3,4
1c00fcd6:	fbee9de3          	bne	t4,t5,1c00fc90 <mm_M_unroll_4x4+0x740>
1c00fcda:	b369                	j	1c00fa64 <mm_M_unroll_4x4+0x514>
1c00fcdc:	4505                	li	a0,1
1c00fcde:	de2a                	sw	a0,60(sp)
1c00fce0:	bb05                	j	1c00fa10 <mm_M_unroll_4x4+0x4c0>
1c00fce2:	4685                	li	a3,1
1c00fce4:	da36                	sw	a3,52(sp)
1c00fce6:	b4c1                	j	1c00f7a6 <mm_M_unroll_4x4+0x256>
1c00fce8:	0000                	unimp
	...

1c00fcec <mm_dw_u2>:
1c00fcec:	455c                	lw	a5,12(a0)
1c00fcee:	01852303          	lw	t1,24(a0)
1c00fcf2:	715d                	addi	sp,sp,-80
1c00fcf4:	00778693          	addi	a3,a5,7
1c00fcf8:	41f6d713          	srai	a4,a3,0x1f
1c00fcfc:	da5a                	sw	s6,52(sp)
1c00fcfe:	f8373733          	p.bclr	a4,a4,28,3
1c00fd02:	c6a2                	sw	s0,76(sp)
1c00fd04:	c4a6                	sw	s1,72(sp)
1c00fd06:	c2ca                	sw	s2,68(sp)
1c00fd08:	c0ce                	sw	s3,64(sp)
1c00fd0a:	de52                	sw	s4,60(sp)
1c00fd0c:	dc56                	sw	s5,56(sp)
1c00fd0e:	d85e                	sw	s7,48(sp)
1c00fd10:	d662                	sw	s8,44(sp)
1c00fd12:	d466                	sw	s9,40(sp)
1c00fd14:	d26a                	sw	s10,36(sp)
1c00fd16:	d06e                	sw	s11,32(sp)
1c00fd18:	4605                	li	a2,1
1c00fd1a:	00052b03          	lw	s6,0(a0)
1c00fd1e:	00452883          	lw	a7,4(a0)
1c00fd22:	00852383          	lw	t2,8(a0)
1c00fd26:	01052283          	lw	t0,16(a0)
1c00fd2a:	06d7275b          	p.addn	a4,a4,a3,3
1c00fd2e:	0a664563          	blt	a2,t1,1c00fdd8 <mm_dw_u2+0xec>
1c00fd32:	f1402473          	csrr	s0,mhartid
1c00fd36:	f4543433          	p.bclr	s0,s0,26,5
1c00fd3a:	02e40433          	mul	s0,s0,a4
1c00fd3e:	9722                	add	a4,a4,s0
1c00fd40:	04f74e33          	p.min	t3,a4,a5
1c00fd44:	04505d63          	blez	t0,1c00fd9e <mm_dw_u2+0xb2>
1c00fd48:	00229493          	slli	s1,t0,0x2
1c00fd4c:	429403b3          	p.mac	t2,s0,s1
1c00fd50:	00000813          	li	a6,0
1c00fd54:	00241e93          	slli	t4,s0,0x2
1c00fd58:	002e1713          	slli	a4,t3,0x2
1c00fd5c:	98f6                	add	a7,a7,t4
1c00fd5e:	00279f93          	slli	t6,a5,0x2
1c00fd62:	9eda                	add	t4,t4,s6
1c00fd64:	9b3a                	add	s6,s6,a4
1c00fd66:	03c45763          	ble	t3,s0,1c00fd94 <mm_dw_u2+0xa8>
1c00fd6a:	04133863          	p.bneimm	t1,1,1c00fdba <mm_dw_u2+0xce>
1c00fd6e:	41db0733          	sub	a4,s6,t4
1c00fd72:	1771                	addi	a4,a4,-4
1c00fd74:	8309                	srli	a4,a4,0x2
1c00fd76:	859e                	mv	a1,t2
1c00fd78:	8646                	mv	a2,a7
1c00fd7a:	86f6                	mv	a3,t4
1c00fd7c:	0705                	addi	a4,a4,1
1c00fd7e:	0001                	nop
1c00fd80:	008740fb          	lp.setup	x1,a4,1c00fd90 <mm_dw_u2+0xa4>
1c00fd84:	0046a78b          	p.lw	a5,4(a3!)
1c00fd88:	0046250b          	p.lw	a0,4(a2!)
1c00fd8c:	80a7f7c3          	fmadd.s	a5,a5,a0,a6
1c00fd90:	00f5e4ab          	p.sw	a5,s1(a1!)
1c00fd94:	98fe                	add	a7,a7,t6
1c00fd96:	0391                	addi	t2,t2,4
1c00fd98:	12fd                	addi	t0,t0,-1
1c00fd9a:	fc0296e3          	bnez	t0,1c00fd66 <mm_dw_u2+0x7a>
1c00fd9e:	4436                	lw	s0,76(sp)
1c00fda0:	44a6                	lw	s1,72(sp)
1c00fda2:	4916                	lw	s2,68(sp)
1c00fda4:	4986                	lw	s3,64(sp)
1c00fda6:	5a72                	lw	s4,60(sp)
1c00fda8:	5ae2                	lw	s5,56(sp)
1c00fdaa:	5b52                	lw	s6,52(sp)
1c00fdac:	5bc2                	lw	s7,48(sp)
1c00fdae:	5c32                	lw	s8,44(sp)
1c00fdb0:	5ca2                	lw	s9,40(sp)
1c00fdb2:	5d12                	lw	s10,36(sp)
1c00fdb4:	5d82                	lw	s11,32(sp)
1c00fdb6:	6161                	addi	sp,sp,80
1c00fdb8:	8082                	ret
1c00fdba:	871e                	mv	a4,t2
1c00fdbc:	408e07b3          	sub	a5,t3,s0
1c00fdc0:	0047c0fb          	lp.setup	x1,a5,1c00fdc8 <mm_dw_u2+0xdc>
1c00fdc4:	00000693          	li	a3,0
1c00fdc8:	00d764ab          	p.sw	a3,s1(a4!)
1c00fdcc:	98fe                	add	a7,a7,t6
1c00fdce:	0391                	addi	t2,t2,4
1c00fdd0:	12fd                	addi	t0,t0,-1
1c00fdd2:	f8029ae3          	bnez	t0,1c00fd66 <mm_dw_u2+0x7a>
1c00fdd6:	b7e1                	j	1c00fd9e <mm_dw_u2+0xb2>
1c00fdd8:	f14026f3          	csrr	a3,mhartid
1c00fddc:	f456b6b3          	p.bclr	a3,a3,26,5
1c00fde0:	02e686b3          	mul	a3,a3,a4
1c00fde4:	9736                	add	a4,a4,a3
1c00fde6:	c636                	sw	a3,12(sp)
1c00fde8:	04f74fb3          	p.min	t6,a4,a5
1c00fdec:	fa5059e3          	blez	t0,1c00fd9e <mm_dw_u2+0xb2>
1c00fdf0:	c00334b3          	p.bclr	s1,t1,0,0
1c00fdf4:	026787b3          	mul	a5,a5,t1
1c00fdf8:	fc133bb3          	p.bclr	s7,t1,30,1
1c00fdfc:	fbf6d1e3          	ble	t6,a3,1c00fd9e <mm_dw_u2+0xb2>
1c00fe00:	078a                	slli	a5,a5,0x2
1c00fe02:	cc3e                	sw	a5,24(sp)
1c00fe04:	400007b7          	lui	a5,0x40000
1c00fe08:	00168713          	addi	a4,a3,1
1c00fe0c:	17fd                	addi	a5,a5,-1
1c00fe0e:	00231a93          	slli	s5,t1,0x2
1c00fe12:	8f46                	mv	t5,a7
1c00fe14:	42e307b3          	p.mac	a5,t1,a4
1c00fe18:	00229d93          	slli	s11,t0,0x2
1c00fe1c:	4d01                	li	s10,0
1c00fe1e:	004b0c93          	addi	s9,s6,4
1c00fe22:	078a                	slli	a5,a5,0x2
1c00fe24:	97da                	add	a5,a5,s6
1c00fe26:	ca3e                	sw	a5,20(sp)
1c00fe28:	47b2                	lw	a5,12(sp)
1c00fe2a:	02d286b3          	mul	a3,t0,a3
1c00fe2e:	03570733          	mul	a4,a4,s5
1c00fe32:	068a                	slli	a3,a3,0x2
1c00fe34:	93b6                	add	t2,t2,a3
1c00fe36:	035787b3          	mul	a5,a5,s5
1c00fe3a:	1771                	addi	a4,a4,-4
1c00fe3c:	ce3a                	sw	a4,28(sp)
1c00fe3e:	c83e                	sw	a5,16(sp)
1c00fe40:	4772                	lw	a4,28(sp)
1c00fe42:	4ec2                	lw	t4,16(sp)
1c00fe44:	49d2                	lw	s3,20(sp)
1c00fe46:	4432                	lw	s0,12(sp)
1c00fe48:	01e70a33          	add	s4,a4,t5
1c00fe4c:	891e                	mv	s2,t2
1c00fe4e:	004f0c13          	addi	s8,t5,4
1c00fe52:	00000793          	li	a5,0
1c00fe56:	c0b9                	beqz	s1,1c00fe9c <mm_dw_u2+0x1b0>
1c00fe58:	fff48693          	addi	a3,s1,-1
1c00fe5c:	8285                	srli	a3,a3,0x1
1c00fe5e:	00148713          	addi	a4,s1,1
1c00fe62:	4889                	li	a7,2
1c00fe64:	016e8833          	add	a6,t4,s6
1c00fe68:	01ee8533          	add	a0,t4,t5
1c00fe6c:	01dc05b3          	add	a1,s8,t4
1c00fe70:	01dc8633          	add	a2,s9,t4
1c00fe74:	00000793          	li	a5,0
1c00fe78:	0685                	addi	a3,a3,1
1c00fe7a:	07176463          	bltu	a4,a7,1c00fee2 <mm_dw_u2+0x1f6>
1c00fe7e:	0001                	nop
1c00fe80:	00c6c0fb          	lp.setup	x1,a3,1c00fe98 <mm_dw_u2+0x1ac>
1c00fe84:	00882e0b          	p.lw	t3,8(a6!)
1c00fe88:	0085230b          	p.lw	t1,8(a0!)
1c00fe8c:	0086288b          	p.lw	a7,8(a2!)
1c00fe90:	0085a70b          	p.lw	a4,8(a1!)
1c00fe94:	786e77c3          	fmadd.s	a5,t3,t1,a5
1c00fe98:	78e8f7c3          	fmadd.s	a5,a7,a4,a5
1c00fe9c:	0405                	addi	s0,s0,1
1c00fe9e:	020b8463          	beqz	s7,1c00fec6 <mm_dw_u2+0x1da>
1c00fea2:	2159f70b          	p.lw	a4,s5(s3!)
1c00fea6:	215a768b          	p.lw	a3,s5(s4!)
1c00feaa:	9ed6                	add	t4,t4,s5
1c00feac:	78d777c3          	fmadd.s	a5,a4,a3,a5
1c00feb0:	00f96dab          	p.sw	a5,s11(s2!)
1c00feb4:	f88f9fe3          	bne	t6,s0,1c00fe52 <mm_dw_u2+0x166>
1c00feb8:	46e2                	lw	a3,24(sp)
1c00feba:	0d05                	addi	s10,s10,1
1c00febc:	0391                	addi	t2,t2,4
1c00febe:	9f36                	add	t5,t5,a3
1c00fec0:	f9a290e3          	bne	t0,s10,1c00fe40 <mm_dw_u2+0x154>
1c00fec4:	bde9                	j	1c00fd9e <mm_dw_u2+0xb2>
1c00fec6:	00f96dab          	p.sw	a5,s11(s2!)
1c00feca:	99d6                	add	s3,s3,s5
1c00fecc:	9a56                	add	s4,s4,s5
1c00fece:	9ed6                	add	t4,t4,s5
1c00fed0:	f88f91e3          	bne	t6,s0,1c00fe52 <mm_dw_u2+0x166>
1c00fed4:	46e2                	lw	a3,24(sp)
1c00fed6:	0d05                	addi	s10,s10,1
1c00fed8:	0391                	addi	t2,t2,4
1c00feda:	9f36                	add	t5,t5,a3
1c00fedc:	f7a292e3          	bne	t0,s10,1c00fe40 <mm_dw_u2+0x154>
1c00fee0:	bd7d                	j	1c00fd9e <mm_dw_u2+0xb2>
1c00fee2:	4685                	li	a3,1
1c00fee4:	bf69                	j	1c00fe7e <mm_dw_u2+0x192>
	...

1c00fee8 <mm_dw_u3>:
1c00fee8:	455c                	lw	a5,12(a0)
1c00feea:	01852883          	lw	a7,24(a0)
1c00feee:	7139                	addi	sp,sp,-64
1c00fef0:	00778693          	addi	a3,a5,7 # 40000007 <__l2_shared_end+0x23fe7afb>
1c00fef4:	41f6d713          	srai	a4,a3,0x1f
1c00fef8:	d84e                	sw	s3,48(sp)
1c00fefa:	d25a                	sw	s6,36(sp)
1c00fefc:	f8373733          	p.bclr	a4,a4,28,3
1c00ff00:	de22                	sw	s0,60(sp)
1c00ff02:	dc26                	sw	s1,56(sp)
1c00ff04:	da4a                	sw	s2,52(sp)
1c00ff06:	d652                	sw	s4,44(sp)
1c00ff08:	d456                	sw	s5,40(sp)
1c00ff0a:	d05e                	sw	s7,32(sp)
1c00ff0c:	ce62                	sw	s8,28(sp)
1c00ff0e:	cc66                	sw	s9,24(sp)
1c00ff10:	ca6a                	sw	s10,20(sp)
1c00ff12:	c86e                	sw	s11,16(sp)
1c00ff14:	4609                	li	a2,2
1c00ff16:	00052983          	lw	s3,0(a0)
1c00ff1a:	00452f03          	lw	t5,4(a0)
1c00ff1e:	00852b03          	lw	s6,8(a0)
1c00ff22:	01052e83          	lw	t4,16(a0)
1c00ff26:	06d7275b          	p.addn	a4,a4,a3,3
1c00ff2a:	0b164063          	blt	a2,a7,1c00ffca <mm_dw_u3+0xe2>
1c00ff2e:	f1402473          	csrr	s0,mhartid
1c00ff32:	f4543433          	p.bclr	s0,s0,26,5
1c00ff36:	02e40433          	mul	s0,s0,a4
1c00ff3a:	9722                	add	a4,a4,s0
1c00ff3c:	04f74fb3          	p.min	t6,a4,a5
1c00ff40:	07d05763          	blez	t4,1c00ffae <mm_dw_u3+0xc6>
1c00ff44:	031787b3          	mul	a5,a5,a7
1c00ff48:	07f45363          	ble	t6,s0,1c00ffae <mm_dw_u3+0xc6>
1c00ff4c:	00289813          	slli	a6,a7,0x2
1c00ff50:	002e9493          	slli	s1,t4,0x2
1c00ff54:	82f6                	mv	t0,t4
1c00ff56:	00279393          	slli	t2,a5,0x2
1c00ff5a:	00000e13          	li	t3,0
1c00ff5e:	028e8eb3          	mul	t4,t4,s0
1c00ff62:	030407b3          	mul	a5,s0,a6
1c00ff66:	0e8a                	slli	t4,t4,0x2
1c00ff68:	9eda                	add	t4,t4,s6
1c00ff6a:	9f3e                	add	t5,t5,a5
1c00ff6c:	99be                	add	s3,s3,a5
1c00ff6e:	0001                	nop
1c00ff70:	01e2c07b          	lp.setup	x0,t0,1c00ffac <mm_dw_u3+0xc4>
1c00ff74:	86fa                	mv	a3,t5
1c00ff76:	874e                	mv	a4,s3
1c00ff78:	85f6                	mv	a1,t4
1c00ff7a:	408f8633          	sub	a2,t6,s0
1c00ff7e:	0001                	nop
1c00ff80:	014640fb          	lp.setup	x1,a2,1c00ffa8 <mm_dw_u3+0xc0>
1c00ff84:	00000793          	li	a5,0
1c00ff88:	01105d63          	blez	a7,1c00ffa2 <mm_dw_u3+0xba>
1c00ff8c:	429c                	lw	a5,0(a3)
1c00ff8e:	4308                	lw	a0,0(a4)
1c00ff90:	e0a7f7c3          	fmadd.s	a5,a5,a0,t3
1c00ff94:	0028b763          	p.bneimm	a7,2,1c00ffa2 <mm_dw_u3+0xba>
1c00ff98:	00472303          	lw	t1,4(a4)
1c00ff9c:	42c8                	lw	a0,4(a3)
1c00ff9e:	78a377c3          	fmadd.s	a5,t1,a0,a5
1c00ffa2:	00f5e4ab          	p.sw	a5,s1(a1!)
1c00ffa6:	9742                	add	a4,a4,a6
1c00ffa8:	96c2                	add	a3,a3,a6
1c00ffaa:	0e91                	addi	t4,t4,4
1c00ffac:	9f1e                	add	t5,t5,t2
1c00ffae:	5472                	lw	s0,60(sp)
1c00ffb0:	54e2                	lw	s1,56(sp)
1c00ffb2:	5952                	lw	s2,52(sp)
1c00ffb4:	59c2                	lw	s3,48(sp)
1c00ffb6:	5a32                	lw	s4,44(sp)
1c00ffb8:	5aa2                	lw	s5,40(sp)
1c00ffba:	5b12                	lw	s6,36(sp)
1c00ffbc:	5b82                	lw	s7,32(sp)
1c00ffbe:	4c72                	lw	s8,28(sp)
1c00ffc0:	4ce2                	lw	s9,24(sp)
1c00ffc2:	4d52                	lw	s10,20(sp)
1c00ffc4:	4dc2                	lw	s11,16(sp)
1c00ffc6:	6121                	addi	sp,sp,64
1c00ffc8:	8082                	ret
1c00ffca:	f14026f3          	csrr	a3,mhartid
1c00ffce:	f456b6b3          	p.bclr	a3,a3,26,5
1c00ffd2:	02e686b3          	mul	a3,a3,a4
1c00ffd6:	9736                	add	a4,a4,a3
1c00ffd8:	04f74733          	p.min	a4,a4,a5
1c00ffdc:	c236                	sw	a3,4(sp)
1c00ffde:	c43a                	sw	a4,8(sp)
1c00ffe0:	fdd057e3          	blez	t4,1c00ffae <mm_dw_u3+0xc6>
1c00ffe4:	4a0d                	li	s4,3
1c00ffe6:	c018b333          	p.bclr	t1,a7,0,1
1c00ffea:	031787b3          	mul	a5,a5,a7
1c00ffee:	fc18bbb3          	p.bclr	s7,a7,30,1
1c00fff2:	0348ea33          	rem	s4,a7,s4
1c00fff6:	fae6dce3          	ble	a4,a3,1c00ffae <mm_dw_u3+0xc6>
1c00fffa:	00279493          	slli	s1,a5,0x2
1c00fffe:	4792                	lw	a5,4(sp)
1c010000:	8e36                	mv	t3,a3
1c010002:	0685                	addi	a3,a3,1
1c010004:	00289293          	slli	t0,a7,0x2
1c010008:	002e9d93          	slli	s11,t4,0x2
1c01000c:	4d01                	li	s10,0
1c01000e:	004f0a93          	addi	s5,t5,4
1c010012:	02fe8733          	mul	a4,t4,a5
1c010016:	400007b7          	lui	a5,0x40000
1c01001a:	17fd                	addi	a5,a5,-1
1c01001c:	431687b3          	p.mac	a5,a3,a7
1c010020:	070a                	slli	a4,a4,0x2
1c010022:	9b3a                	add	s6,s6,a4
1c010024:	025e0c33          	mul	s8,t3,t0
1c010028:	078a                	slli	a5,a5,0x2
1c01002a:	00ff0cb3          	add	s9,t5,a5
1c01002e:	97ce                	add	a5,a5,s3
1c010030:	c63e                	sw	a5,12(sp)
1c010032:	46a2                	lw	a3,8(sp)
1c010034:	4712                	lw	a4,4(sp)
1c010036:	43b2                	lw	t2,12(sp)
1c010038:	8e62                	mv	t3,s8
1c01003a:	8466                	mv	s0,s9
1c01003c:	895a                	mv	s2,s6
1c01003e:	40e68fb3          	sub	t6,a3,a4
1c010042:	0001                	nop
1c010044:	037fc0fb          	lp.setup	x1,t6,1c0100b2 <mm_dw_u3+0x1ca>
1c010048:	00000793          	li	a5,0
1c01004c:	02030e63          	beqz	t1,1c010088 <mm_dw_u3+0x1a0>
1c010050:	01c985b3          	add	a1,s3,t3
1c010054:	01cf08b3          	add	a7,t5,t3
1c010058:	01ca8533          	add	a0,s5,t3
1c01005c:	4801                	li	a6,0
1c01005e:	c072                	sw	t3,0(sp)
1c010060:	4190                	lw	a2,0(a1)
1c010062:	00c8a70b          	p.lw	a4,12(a7!)
1c010066:	41d4                	lw	a3,4(a1)
1c010068:	00052e03          	lw	t3,0(a0)
1c01006c:	78e677c3          	fmadd.s	a5,a2,a4,a5
1c010070:	4590                	lw	a2,8(a1)
1c010072:	4158                	lw	a4,4(a0)
1c010074:	080d                	addi	a6,a6,3
1c010076:	05b1                	addi	a1,a1,12
1c010078:	79c6f7c3          	fmadd.s	a5,a3,t3,a5
1c01007c:	0531                	addi	a0,a0,12
1c01007e:	78e677c3          	fmadd.s	a5,a2,a4,a5
1c010082:	fc686fe3          	bltu	a6,t1,1c010060 <mm_dw_u3+0x178>
1c010086:	4e02                	lw	t3,0(sp)
1c010088:	020a0163          	beqz	s4,1c0100aa <mm_dw_u3+0x1c2>
1c01008c:	0003a703          	lw	a4,0(t2)
1c010090:	4014                	lw	a3,0(s0)
1c010092:	10d77753          	fmul.s	a4,a4,a3
1c010096:	000b9863          	bnez	s7,1c0100a6 <mm_dw_u3+0x1be>
1c01009a:	ffc3a683          	lw	a3,-4(t2)
1c01009e:	ffc42603          	lw	a2,-4(s0)
1c0100a2:	78c6f7c3          	fmadd.s	a5,a3,a2,a5
1c0100a6:	00e7f7d3          	fadd.s	a5,a5,a4
1c0100aa:	00f96dab          	p.sw	a5,s11(s2!)
1c0100ae:	9396                	add	t2,t2,t0
1c0100b0:	9416                	add	s0,s0,t0
1c0100b2:	9e16                	add	t3,t3,t0
1c0100b4:	0d05                	addi	s10,s10,1
1c0100b6:	0b11                	addi	s6,s6,4
1c0100b8:	9ca6                	add	s9,s9,s1
1c0100ba:	9c26                	add	s8,s8,s1
1c0100bc:	409989b3          	sub	s3,s3,s1
1c0100c0:	f7ae99e3          	bne	t4,s10,1c010032 <mm_dw_u3+0x14a>
1c0100c4:	b5ed                	j	1c00ffae <mm_dw_u3+0xc6>
	...

1c0100c8 <mm_dw_unroll_1x2>:
1c0100c8:	00c52803          	lw	a6,12(a0)
1c0100cc:	490c                	lw	a1,16(a0)
1c0100ce:	7139                	addi	sp,sp,-64
1c0100d0:	00780713          	addi	a4,a6,7
1c0100d4:	41f75793          	srai	a5,a4,0x1f
1c0100d8:	d456                	sw	s5,40(sp)
1c0100da:	f837b7b3          	p.bclr	a5,a5,28,3
1c0100de:	de22                	sw	s0,60(sp)
1c0100e0:	dc26                	sw	s1,56(sp)
1c0100e2:	da4a                	sw	s2,52(sp)
1c0100e4:	d84e                	sw	s3,48(sp)
1c0100e6:	d652                	sw	s4,44(sp)
1c0100e8:	d25a                	sw	s6,36(sp)
1c0100ea:	d05e                	sw	s7,32(sp)
1c0100ec:	ce62                	sw	s8,28(sp)
1c0100ee:	cc66                	sw	s9,24(sp)
1c0100f0:	ca6a                	sw	s10,20(sp)
1c0100f2:	c86e                	sw	s11,16(sp)
1c0100f4:	4885                	li	a7,1
1c0100f6:	4114                	lw	a3,0(a0)
1c0100f8:	4150                	lw	a2,4(a0)
1c0100fa:	00852a83          	lw	s5,8(a0)
1c0100fe:	01852383          	lw	t2,24(a0)
1c010102:	06e7a7db          	p.addn	a5,a5,a4,3
1c010106:	14b8d963          	ble	a1,a7,1c010258 <mm_dw_unroll_1x2+0x190>
1c01010a:	f1402b73          	csrr	s6,mhartid
1c01010e:	f45b3b33          	p.bclr	s6,s6,26,5
1c010112:	02fb0b33          	mul	s6,s6,a5
1c010116:	c005bd33          	p.bclr	s10,a1,0,0
1c01011a:	97da                	add	a5,a5,s6
1c01011c:	0507c433          	p.min	s0,a5,a6
1c010120:	0c0d0c63          	beqz	s10,1c0101f8 <mm_dw_unroll_1x2+0x130>
1c010124:	02780bb3          	mul	s7,a6,t2
1c010128:	40d607b3          	sub	a5,a2,a3
1c01012c:	00239913          	slli	s2,t2,0x2
1c010130:	41278533          	sub	a0,a5,s2
1c010134:	fff58813          	addi	a6,a1,-1
1c010138:	fc15b733          	p.bclr	a4,a1,30,1
1c01013c:	c43a                	sw	a4,8(sp)
1c01013e:	8742                	mv	a4,a6
1c010140:	00259d93          	slli	s11,a1,0x2
1c010144:	41260633          	sub	a2,a2,s2
1c010148:	002b9993          	slli	s3,s7,0x2
1c01014c:	99be                	add	s3,s3,a5
1c01014e:	879e                	mv	a5,t2
1c010150:	427b07b3          	p.mac	a5,s6,t2
1c010154:	003b9c93          	slli	s9,s7,0x3
1c010158:	40700a33          	neg	s4,t2
1c01015c:	40d604b3          	sub	s1,a2,a3
1c010160:	412989b3          	sub	s3,s3,s2
1c010164:	0a0a                	slli	s4,s4,0x2
1c010166:	4c01                	li	s8,0
1c010168:	42bb0733          	p.mac	a4,s6,a1
1c01016c:	078a                	slli	a5,a5,0x2
1c01016e:	97b6                	add	a5,a5,a3
1c010170:	c23e                	sw	a5,4(sp)
1c010172:	030b8bb3          	mul	s7,s7,a6
1c010176:	00271793          	slli	a5,a4,0x2
1c01017a:	97d6                	add	a5,a5,s5
1c01017c:	c63e                	sw	a5,12(sp)
1c01017e:	43bb0ab3          	p.mac	s5,s6,s11
1c010182:	0b8a                	slli	s7,s7,0x2
1c010184:	9baa                	add	s7,s7,a0
1c010186:	068b5363          	ble	s0,s6,1c0101ec <mm_dw_unroll_1x2+0x124>
1c01018a:	4e12                	lw	t3,4(sp)
1c01018c:	004a8293          	addi	t0,s5,4
1c010190:	8fd6                	mv	t6,s5
1c010192:	8eda                	mv	t4,s6
1c010194:	41640f33          	sub	t5,s0,s6
1c010198:	027f407b          	lp.setup	x0,t5,1c0101e6 <mm_dw_unroll_1x2+0x11e>
1c01019c:	00000613          	li	a2,0
1c0101a0:	86b2                	mv	a3,a2
1c0101a2:	02705d63          	blez	t2,1c0101dc <mm_dw_unroll_1x2+0x114>
1c0101a6:	00000613          	li	a2,0
1c0101aa:	01ca05b3          	add	a1,s4,t3
1c0101ae:	40be07b3          	sub	a5,t3,a1
1c0101b2:	17f1                	addi	a5,a5,-4
1c0101b4:	86b2                	mv	a3,a2
1c0101b6:	8389                	srli	a5,a5,0x2
1c0101b8:	01c48833          	add	a6,s1,t3
1c0101bc:	013e0533          	add	a0,t3,s3
1c0101c0:	0785                	addi	a5,a5,1
1c0101c2:	0001                	nop
1c0101c4:	00a7c0fb          	lp.setup	x1,a5,1c0101d8 <mm_dw_unroll_1x2+0x110>
1c0101c8:	0045a70b          	p.lw	a4,4(a1!)
1c0101cc:	0048230b          	p.lw	t1,4(a6!)
1c0101d0:	0045288b          	p.lw	a7,4(a0!)
1c0101d4:	68e376c3          	fmadd.s	a3,t1,a4,a3
1c0101d8:	60e8f643          	fmadd.s	a2,a7,a4,a2
1c0101dc:	00dfedab          	p.sw	a3,s11(t6!)
1c0101e0:	00c2edab          	p.sw	a2,s11(t0!)
1c0101e4:	0e85                	addi	t4,t4,1
1c0101e6:	9e4a                	add	t3,t3,s2
1c0101e8:	47a2                	lw	a5,8(sp)
1c0101ea:	e78d                	bnez	a5,1c010214 <mm_dw_unroll_1x2+0x14c>
1c0101ec:	0c09                	addi	s8,s8,2
1c0101ee:	0aa1                	addi	s5,s5,8
1c0101f0:	94e6                	add	s1,s1,s9
1c0101f2:	99e6                	add	s3,s3,s9
1c0101f4:	f9ac69e3          	bltu	s8,s10,1c010186 <mm_dw_unroll_1x2+0xbe>
1c0101f8:	5472                	lw	s0,60(sp)
1c0101fa:	54e2                	lw	s1,56(sp)
1c0101fc:	5952                	lw	s2,52(sp)
1c0101fe:	59c2                	lw	s3,48(sp)
1c010200:	5a32                	lw	s4,44(sp)
1c010202:	5aa2                	lw	s5,40(sp)
1c010204:	5b12                	lw	s6,36(sp)
1c010206:	5b82                	lw	s7,32(sp)
1c010208:	4c72                	lw	s8,28(sp)
1c01020a:	4ce2                	lw	s9,24(sp)
1c01020c:	4d52                	lw	s10,20(sp)
1c01020e:	4dc2                	lw	s11,16(sp)
1c010210:	6121                	addi	sp,sp,64
1c010212:	8082                	ret
1c010214:	4512                	lw	a0,4(sp)
1c010216:	4832                	lw	a6,12(sp)
1c010218:	416e8eb3          	sub	t4,t4,s6
1c01021c:	01cec07b          	lp.setup	x0,t4,1c010254 <mm_dw_unroll_1x2+0x18c>
1c010220:	00000713          	li	a4,0
1c010224:	02705663          	blez	t2,1c010250 <mm_dw_unroll_1x2+0x188>
1c010228:	00aa06b3          	add	a3,s4,a0
1c01022c:	40d507b3          	sub	a5,a0,a3
1c010230:	17f1                	addi	a5,a5,-4
1c010232:	00000713          	li	a4,0
1c010236:	8389                	srli	a5,a5,0x2
1c010238:	00ab8633          	add	a2,s7,a0
1c01023c:	0785                	addi	a5,a5,1
1c01023e:	0001                	nop
1c010240:	0067c0fb          	lp.setup	x1,a5,1c01024c <mm_dw_unroll_1x2+0x184>
1c010244:	0046a88b          	p.lw	a7,4(a3!)
1c010248:	0046258b          	p.lw	a1,4(a2!)
1c01024c:	70b8f743          	fmadd.s	a4,a7,a1,a4
1c010250:	00e86dab          	p.sw	a4,s11(a6!)
1c010254:	954a                	add	a0,a0,s2
1c010256:	bf59                	j	1c0101ec <mm_dw_unroll_1x2+0x124>
1c010258:	f1402773          	csrr	a4,mhartid
1c01025c:	f4573733          	p.bclr	a4,a4,26,5
1c010260:	02f70733          	mul	a4,a4,a5
1c010264:	97ba                	add	a5,a5,a4
1c010266:	0507c7b3          	p.min	a5,a5,a6
1c01026a:	f815b7e3          	p.bneimm	a1,1,1c0101f8 <mm_dw_unroll_1x2+0x130>
1c01026e:	f8f755e3          	ble	a5,a4,1c0101f8 <mm_dw_unroll_1x2+0x130>
1c010272:	889e                	mv	a7,t2
1c010274:	427708b3          	p.mac	a7,a4,t2
1c010278:	00279813          	slli	a6,a5,0x2
1c01027c:	070a                	slli	a4,a4,0x2
1c01027e:	40e80833          	sub	a6,a6,a4
1c010282:	00239f13          	slli	t5,t2,0x2
1c010286:	1871                	addi	a6,a6,-4
1c010288:	40700e33          	neg	t3,t2
1c01028c:	41e60633          	sub	a2,a2,t5
1c010290:	00285813          	srli	a6,a6,0x2
1c010294:	00ea8333          	add	t1,s5,a4
1c010298:	088a                	slli	a7,a7,0x2
1c01029a:	98b6                	add	a7,a7,a3
1c01029c:	0e0a                	slli	t3,t3,0x2
1c01029e:	40d60eb3          	sub	t4,a2,a3
1c0102a2:	0805                	addi	a6,a6,1
1c0102a4:	01c8407b          	lp.setup	x0,a6,1c0102dc <mm_dw_unroll_1x2+0x214>
1c0102a8:	00000713          	li	a4,0
1c0102ac:	02705663          	blez	t2,1c0102d8 <mm_dw_unroll_1x2+0x210>
1c0102b0:	011e06b3          	add	a3,t3,a7
1c0102b4:	40d887b3          	sub	a5,a7,a3
1c0102b8:	17f1                	addi	a5,a5,-4
1c0102ba:	00000713          	li	a4,0
1c0102be:	8389                	srli	a5,a5,0x2
1c0102c0:	011e8633          	add	a2,t4,a7
1c0102c4:	0785                	addi	a5,a5,1
1c0102c6:	0001                	nop
1c0102c8:	0067c0fb          	lp.setup	x1,a5,1c0102d4 <mm_dw_unroll_1x2+0x20c>
1c0102cc:	0046a50b          	p.lw	a0,4(a3!)
1c0102d0:	0046258b          	p.lw	a1,4(a2!)
1c0102d4:	70b57743          	fmadd.s	a4,a0,a1,a4
1c0102d8:	00e3222b          	p.sw	a4,4(t1!)
1c0102dc:	98fa                	add	a7,a7,t5
1c0102de:	bf29                	j	1c0101f8 <mm_dw_unroll_1x2+0x130>
1c0102e0:	0000                	unimp
	...

1c0102e4 <mm_dw_unroll_1x4>:
1c0102e4:	4910                	lw	a2,16(a0)
1c0102e6:	7159                	addi	sp,sp,-112
1c0102e8:	d6a2                	sw	s0,108(sp)
1c0102ea:	d4a6                	sw	s1,104(sp)
1c0102ec:	d2ca                	sw	s2,100(sp)
1c0102ee:	d0ce                	sw	s3,96(sp)
1c0102f0:	ced2                	sw	s4,92(sp)
1c0102f2:	ccd6                	sw	s5,88(sp)
1c0102f4:	cada                	sw	s6,84(sp)
1c0102f6:	c8de                	sw	s7,80(sp)
1c0102f8:	c6e2                	sw	s8,76(sp)
1c0102fa:	c4e6                	sw	s9,72(sp)
1c0102fc:	c2ea                	sw	s10,68(sp)
1c0102fe:	c0ee                	sw	s11,64(sp)
1c010300:	d232                	sw	a2,36(sp)
1c010302:	478d                	li	a5,3
1c010304:	20c7d363          	ble	a2,a5,1c01050a <mm_dw_unroll_1x4+0x226>
1c010308:	4554                	lw	a3,12(a0)
1c01030a:	8832                	mv	a6,a2
1c01030c:	f1402673          	csrr	a2,mhartid
1c010310:	00768593          	addi	a1,a3,7
1c010314:	41f5d793          	srai	a5,a1,0x1f
1c010318:	f837b7b3          	p.bclr	a5,a5,28,3
1c01031c:	06b7a7db          	p.addn	a5,a5,a1,3
1c010320:	f4563633          	p.bclr	a2,a2,26,5
1c010324:	02f60633          	mul	a2,a2,a5
1c010328:	c2083833          	p.bclr	a6,a6,1,0
1c01032c:	ce42                	sw	a6,28(sp)
1c01032e:	872a                	mv	a4,a0
1c010330:	97b2                	add	a5,a5,a2
1c010332:	04d7c7b3          	p.min	a5,a5,a3
1c010336:	c632                	sw	a2,12(sp)
1c010338:	cc3e                	sw	a5,24(sp)
1c01033a:	12080363          	beqz	a6,1c010460 <mm_dw_unroll_1x4+0x17c>
1c01033e:	01852a83          	lw	s5,24(a0)
1c010342:	5892                	lw	a7,36(sp)
1c010344:	470c                	lw	a1,8(a4)
1c010346:	035686b3          	mul	a3,a3,s5
1c01034a:	fa28b7b3          	p.bclr	a5,a7,29,2
1c01034e:	d63e                	sw	a5,44(sp)
1c010350:	87c2                	mv	a5,a6
1c010352:	4832                	lw	a6,12(sp)
1c010354:	00289d13          	slli	s10,a7,0x2
1c010358:	4150                	lw	a2,4(a0)
1c01035a:	4108                	lw	a0,0(a0)
1c01035c:	002a9913          	slli	s2,s5,0x2
1c010360:	41500b33          	neg	s6,s5
1c010364:	431807b3          	p.mac	a5,a6,a7
1c010368:	d036                	sw	a3,32(sp)
1c01036a:	86d6                	mv	a3,s5
1c01036c:	5882                	lw	a7,32(sp)
1c01036e:	40a60733          	sub	a4,a2,a0
1c010372:	41270733          	sub	a4,a4,s2
1c010376:	0892                	slli	a7,a7,0x4
1c010378:	d846                	sw	a7,48(sp)
1c01037a:	48f2                	lw	a7,28(sp)
1c01037c:	0b0a                	slli	s6,s6,0x2
1c01037e:	435806b3          	p.mac	a3,a6,s5
1c010382:	078a                	slli	a5,a5,0x2
1c010384:	97ae                	add	a5,a5,a1
1c010386:	de3e                	sw	a5,60(sp)
1c010388:	5782                	lw	a5,32(sp)
1c01038a:	c802                	sw	zero,16(sp)
1c01038c:	00179813          	slli	a6,a5,0x1
1c010390:	00279c93          	slli	s9,a5,0x2
1c010394:	00379d93          	slli	s11,a5,0x3
1c010398:	068a                	slli	a3,a3,0x2
1c01039a:	96aa                	add	a3,a3,a0
1c01039c:	d436                	sw	a3,40(sp)
1c01039e:	46b2                	lw	a3,12(sp)
1c0103a0:	031787b3          	mul	a5,a5,a7
1c0103a4:	da42                	sw	a6,52(sp)
1c0103a6:	43a685b3          	p.mac	a1,a3,s10
1c0103aa:	078a                	slli	a5,a5,0x2
1c0103ac:	97ba                	add	a5,a5,a4
1c0103ae:	dc3e                	sw	a5,56(sp)
1c0103b0:	43268633          	p.mac	a2,a3,s2
1c0103b4:	c42e                	sw	a1,8(sp)
1c0103b6:	ca32                	sw	a2,20(sp)
1c0103b8:	4632                	lw	a2,12(sp)
1c0103ba:	46e2                	lw	a3,24(sp)
1c0103bc:	08d65363          	ble	a3,a2,1c010442 <mm_dw_unroll_1x4+0x15e>
1c0103c0:	5702                	lw	a4,32(sp)
1c0103c2:	57d2                	lw	a5,52(sp)
1c0103c4:	4822                	lw	a6,8(sp)
1c0103c6:	59a2                	lw	s3,40(sp)
1c0103c8:	4452                	lw	s0,20(sp)
1c0103ca:	00f70c33          	add	s8,a4,a5
1c0103ce:	00480493          	addi	s1,a6,4
1c0103d2:	0c0a                	slli	s8,s8,0x2
1c0103d4:	8bc2                	mv	s7,a6
1c0103d6:	40c68a33          	sub	s4,a3,a2
1c0103da:	0b505163          	blez	s5,1c01047c <mm_dw_unroll_1x4+0x198>
1c0103de:	00000513          	li	a0,0
1c0103e2:	013b0833          	add	a6,s6,s3
1c0103e6:	41098733          	sub	a4,s3,a6
1c0103ea:	1771                	addi	a4,a4,-4
1c0103ec:	85aa                	mv	a1,a0
1c0103ee:	862a                	mv	a2,a0
1c0103f0:	86aa                	mv	a3,a0
1c0103f2:	8309                	srli	a4,a4,0x2
1c0103f4:	008c8eb3          	add	t4,s9,s0
1c0103f8:	008d8e33          	add	t3,s11,s0
1c0103fc:	008c0333          	add	t1,s8,s0
1c010400:	88a2                	mv	a7,s0
1c010402:	0705                	addi	a4,a4,1
1c010404:	012740fb          	lp.setup	x1,a4,1c010428 <mm_dw_unroll_1x4+0x144>
1c010408:	0048278b          	p.lw	a5,4(a6!)
1c01040c:	0048a38b          	p.lw	t2,4(a7!)
1c010410:	004ea28b          	p.lw	t0,4(t4!)
1c010414:	004e2f8b          	p.lw	t6,4(t3!)
1c010418:	00432f0b          	p.lw	t5,4(t1!)
1c01041c:	68f3f6c3          	fmadd.s	a3,t2,a5,a3
1c010420:	60f2f643          	fmadd.s	a2,t0,a5,a2
1c010424:	58fff5c3          	fmadd.s	a1,t6,a5,a1
1c010428:	50ff7543          	fmadd.s	a0,t5,a5,a0
1c01042c:	00dbed2b          	p.sw	a3,s10(s7!)
1c010430:	c090                	sw	a2,0(s1)
1c010432:	c0cc                	sw	a1,4(s1)
1c010434:	c488                	sw	a0,8(s1)
1c010436:	944a                	add	s0,s0,s2
1c010438:	94ea                	add	s1,s1,s10
1c01043a:	99ca                	add	s3,s3,s2
1c01043c:	1a7d                	addi	s4,s4,-1
1c01043e:	f80a1ee3          	bnez	s4,1c0103da <mm_dw_unroll_1x4+0xf6>
1c010442:	5732                	lw	a4,44(sp)
1c010444:	ef29                	bnez	a4,1c01049e <mm_dw_unroll_1x4+0x1ba>
1c010446:	46c2                	lw	a3,16(sp)
1c010448:	4722                	lw	a4,8(sp)
1c01044a:	47d2                	lw	a5,20(sp)
1c01044c:	5842                	lw	a6,48(sp)
1c01044e:	48f2                	lw	a7,28(sp)
1c010450:	0691                	addi	a3,a3,4
1c010452:	0741                	addi	a4,a4,16
1c010454:	97c2                	add	a5,a5,a6
1c010456:	c836                	sw	a3,16(sp)
1c010458:	c43a                	sw	a4,8(sp)
1c01045a:	ca3e                	sw	a5,20(sp)
1c01045c:	f516eee3          	bltu	a3,a7,1c0103b8 <mm_dw_unroll_1x4+0xd4>
1c010460:	5436                	lw	s0,108(sp)
1c010462:	54a6                	lw	s1,104(sp)
1c010464:	5916                	lw	s2,100(sp)
1c010466:	5986                	lw	s3,96(sp)
1c010468:	4a76                	lw	s4,92(sp)
1c01046a:	4ae6                	lw	s5,88(sp)
1c01046c:	4b56                	lw	s6,84(sp)
1c01046e:	4bc6                	lw	s7,80(sp)
1c010470:	4c36                	lw	s8,76(sp)
1c010472:	4ca6                	lw	s9,72(sp)
1c010474:	4d16                	lw	s10,68(sp)
1c010476:	4d86                	lw	s11,64(sp)
1c010478:	6165                	addi	sp,sp,112
1c01047a:	8082                	ret
1c01047c:	00000513          	li	a0,0
1c010480:	944a                	add	s0,s0,s2
1c010482:	99ca                	add	s3,s3,s2
1c010484:	86aa                	mv	a3,a0
1c010486:	85aa                	mv	a1,a0
1c010488:	862a                	mv	a2,a0
1c01048a:	00dbed2b          	p.sw	a3,s10(s7!)
1c01048e:	c0cc                	sw	a1,4(s1)
1c010490:	c090                	sw	a2,0(s1)
1c010492:	c488                	sw	a0,8(s1)
1c010494:	94ea                	add	s1,s1,s10
1c010496:	1a7d                	addi	s4,s4,-1
1c010498:	f40a11e3          	bnez	s4,1c0103da <mm_dw_unroll_1x4+0xf6>
1c01049c:	b75d                	j	1c010442 <mm_dw_unroll_1x4+0x15e>
1c01049e:	47f2                	lw	a5,28(sp)
1c0104a0:	5812                	lw	a6,36(sp)
1c0104a2:	fb07d2e3          	ble	a6,a5,1c010446 <mm_dw_unroll_1x4+0x162>
1c0104a6:	48b2                	lw	a7,12(sp)
1c0104a8:	4662                	lw	a2,24(sp)
1c0104aa:	f8c8dee3          	ble	a2,a7,1c010446 <mm_dw_unroll_1x4+0x162>
1c0104ae:	5e62                	lw	t3,56(sp)
1c0104b0:	5f72                	lw	t5,60(sp)
1c0104b2:	5422                	lw	s0,40(sp)
1c0104b4:	8ebe                	mv	t4,a5
1c0104b6:	8fc2                	mv	t6,a6
1c0104b8:	82c6                	mv	t0,a7
1c0104ba:	83b2                	mv	t2,a2
1c0104bc:	8822                	mv	a6,s0
1c0104be:	837a                	mv	t1,t5
1c0104c0:	405388b3          	sub	a7,t2,t0
1c0104c4:	01c8c07b          	lp.setup	x0,a7,1c0104fc <mm_dw_unroll_1x4+0x218>
1c0104c8:	00000713          	li	a4,0
1c0104cc:	03505663          	blez	s5,1c0104f8 <mm_dw_unroll_1x4+0x214>
1c0104d0:	010b06b3          	add	a3,s6,a6
1c0104d4:	40d807b3          	sub	a5,a6,a3
1c0104d8:	17f1                	addi	a5,a5,-4
1c0104da:	00000713          	li	a4,0
1c0104de:	8389                	srli	a5,a5,0x2
1c0104e0:	010e0633          	add	a2,t3,a6
1c0104e4:	0785                	addi	a5,a5,1
1c0104e6:	0001                	nop
1c0104e8:	0067c0fb          	lp.setup	x1,a5,1c0104f4 <mm_dw_unroll_1x4+0x210>
1c0104ec:	0046a50b          	p.lw	a0,4(a3!)
1c0104f0:	0046258b          	p.lw	a1,4(a2!)
1c0104f4:	70b57743          	fmadd.s	a4,a0,a1,a4
1c0104f8:	00e36d2b          	p.sw	a4,s10(t1!)
1c0104fc:	984a                	add	a6,a6,s2
1c0104fe:	0e85                	addi	t4,t4,1
1c010500:	0f11                	addi	t5,t5,4
1c010502:	9e66                	add	t3,t3,s9
1c010504:	fbdf9ce3          	bne	t6,t4,1c0104bc <mm_dw_unroll_1x4+0x1d8>
1c010508:	bf3d                	j	1c010446 <mm_dw_unroll_1x4+0x162>
1c01050a:	5436                	lw	s0,108(sp)
1c01050c:	54a6                	lw	s1,104(sp)
1c01050e:	5916                	lw	s2,100(sp)
1c010510:	5986                	lw	s3,96(sp)
1c010512:	4a76                	lw	s4,92(sp)
1c010514:	4ae6                	lw	s5,88(sp)
1c010516:	4b56                	lw	s6,84(sp)
1c010518:	4bc6                	lw	s7,80(sp)
1c01051a:	4c36                	lw	s8,76(sp)
1c01051c:	4ca6                	lw	s9,72(sp)
1c01051e:	4d16                	lw	s10,68(sp)
1c010520:	4d86                	lw	s11,64(sp)
1c010522:	6165                	addi	sp,sp,112
1c010524:	b655                	j	1c0100c8 <mm_dw_unroll_1x2>
	...

1c010528 <mm_dw_unroll_1x2_u2>:
1c010528:	4914                	lw	a3,16(a0)
1c01052a:	7119                	addi	sp,sp,-128
1c01052c:	d2da                	sw	s6,100(sp)
1c01052e:	cce6                	sw	s9,88(sp)
1c010530:	dea2                	sw	s0,124(sp)
1c010532:	dca6                	sw	s1,120(sp)
1c010534:	daca                	sw	s2,116(sp)
1c010536:	d8ce                	sw	s3,112(sp)
1c010538:	d6d2                	sw	s4,108(sp)
1c01053a:	d4d6                	sw	s5,104(sp)
1c01053c:	d0de                	sw	s7,96(sp)
1c01053e:	cee2                	sw	s8,92(sp)
1c010540:	caea                	sw	s10,84(sp)
1c010542:	c8ee                	sw	s11,80(sp)
1c010544:	4605                	li	a2,1
1c010546:	00052c83          	lw	s9,0(a0)
1c01054a:	00452b03          	lw	s6,4(a0)
1c01054e:	4518                	lw	a4,8(a0)
1c010550:	454c                	lw	a1,12(a0)
1c010552:	01852803          	lw	a6,24(a0)
1c010556:	26d65a63          	ble	a3,a2,1c0107ca <mm_dw_unroll_1x2_u2+0x2a2>
1c01055a:	31065563          	ble	a6,a2,1c010864 <mm_dw_unroll_1x2_u2+0x33c>
1c01055e:	00758613          	addi	a2,a1,7
1c010562:	41f65793          	srai	a5,a2,0x1f
1c010566:	f837b7b3          	p.bclr	a5,a5,28,3
1c01056a:	06c7a7db          	p.addn	a5,a5,a2,3
1c01056e:	f1402673          	csrr	a2,mhartid
1c010572:	f4563633          	p.bclr	a2,a2,26,5
1c010576:	02f60633          	mul	a2,a2,a5
1c01057a:	c006b533          	p.bclr	a0,a3,0,0
1c01057e:	d82a                	sw	a0,48(sp)
1c010580:	97b2                	add	a5,a5,a2
1c010582:	d232                	sw	a2,36(sp)
1c010584:	04b7cbb3          	p.min	s7,a5,a1
1c010588:	18050363          	beqz	a0,1c01070e <mm_dw_unroll_1x2_u2+0x1e6>
1c01058c:	030587b3          	mul	a5,a1,a6
1c010590:	fff68513          	addi	a0,a3,-1
1c010594:	fc16b8b3          	p.bclr	a7,a3,30,1
1c010598:	c0c6                	sw	a7,64(sp)
1c01059a:	00160593          	addi	a1,a2,1
1c01059e:	00281493          	slli	s1,a6,0x2
1c0105a2:	c00833b3          	p.bclr	t2,a6,0,0
1c0105a6:	fc183c33          	p.bclr	s8,a6,30,1
1c0105aa:	40000637          	lui	a2,0x40000
1c0105ae:	167d                	addi	a2,a2,-1
1c0105b0:	02f50333          	mul	t1,a0,a5
1c0105b4:	00279e13          	slli	t3,a5,0x2
1c0105b8:	40f008b3          	neg	a7,a5
1c0105bc:	078e                	slli	a5,a5,0x3
1c0105be:	d03e                	sw	a5,32(sp)
1c0105c0:	004c8793          	addi	a5,s9,4
1c0105c4:	c23e                	sw	a5,4(sp)
1c0105c6:	5792                	lw	a5,36(sp)
1c0105c8:	088e                	slli	a7,a7,0x3
1c0105ca:	de46                	sw	a7,60(sp)
1c0105cc:	03058833          	mul	a6,a1,a6
1c0105d0:	00c308b3          	add	a7,t1,a2
1c0105d4:	030a                	slli	t1,t1,0x2
1c0105d6:	935a                	add	t1,t1,s6
1c0105d8:	00269d93          	slli	s11,a3,0x2
1c0105dc:	c872                	sw	t3,16(sp)
1c0105de:	01cb0d33          	add	s10,s6,t3
1c0105e2:	00430e13          	addi	t3,t1,4
1c0105e6:	d672                	sw	t3,44(sp)
1c0105e8:	ca02                	sw	zero,20(sp)
1c0105ea:	42d78533          	p.mac	a0,a5,a3
1c0105ee:	00c807b3          	add	a5,a6,a2
1c0105f2:	9846                	add	a6,a6,a7
1c0105f4:	5892                	lw	a7,36(sp)
1c0105f6:	078a                	slli	a5,a5,0x2
1c0105f8:	00281693          	slli	a3,a6,0x2
1c0105fc:	97e6                	add	a5,a5,s9
1c0105fe:	96da                	add	a3,a3,s6
1c010600:	ce02                	sw	zero,28(sp)
1c010602:	d41a                	sw	t1,40(sp)
1c010604:	029585b3          	mul	a1,a1,s1
1c010608:	050a                	slli	a0,a0,0x2
1c01060a:	953a                	add	a0,a0,a4
1c01060c:	dc3e                	sw	a5,56(sp)
1c01060e:	c4b6                	sw	a3,72(sp)
1c010610:	c6aa                	sw	a0,76(sp)
1c010612:	43b88733          	p.mac	a4,a7,s11
1c010616:	15f1                	addi	a1,a1,-4
1c010618:	c2ae                	sw	a1,68(sp)
1c01061a:	02988e33          	mul	t3,a7,s1
1c01061e:	cc3a                	sw	a4,24(sp)
1c010620:	da72                	sw	t3,52(sp)
1c010622:	5712                	lw	a4,36(sp)
1c010624:	0d775263          	ble	s7,a4,1c0106e8 <mm_dw_unroll_1x2_u2+0x1c0>
1c010628:	47e2                	lw	a5,24(sp)
1c01062a:	4896                	lw	a7,68(sp)
1c01062c:	52d2                	lw	t0,52(sp)
1c01062e:	5a62                	lw	s4,56(sp)
1c010630:	004b0e13          	addi	t3,s6,4
1c010634:	004d0513          	addi	a0,s10,4
1c010638:	00478a93          	addi	s5,a5,4 # 40000004 <__l2_shared_end+0x23fe7af8>
1c01063c:	01688933          	add	s2,a7,s6
1c010640:	89be                	mv	s3,a5
1c010642:	843a                	mv	s0,a4
1c010644:	c472                	sw	t3,8(sp)
1c010646:	c62a                	sw	a0,12(sp)
1c010648:	00000713          	li	a4,0
1c01064c:	86ba                	mv	a3,a4
1c01064e:	06038363          	beqz	t2,1c0106b4 <mm_dw_unroll_1x2_u2+0x18c>
1c010652:	4732                	lw	a4,12(sp)
1c010654:	4522                	lw	a0,8(sp)
1c010656:	fff38613          	addi	a2,t2,-1
1c01065a:	00570833          	add	a6,a4,t0
1c01065e:	4712                	lw	a4,4(sp)
1c010660:	005508b3          	add	a7,a0,t0
1c010664:	8205                	srli	a2,a2,0x1
1c010666:	00570533          	add	a0,a4,t0
1c01066a:	00000713          	li	a4,0
1c01066e:	00138793          	addi	a5,t2,1
1c010672:	4589                	li	a1,2
1c010674:	005c8eb3          	add	t4,s9,t0
1c010678:	005b0e33          	add	t3,s6,t0
1c01067c:	01a28333          	add	t1,t0,s10
1c010680:	86ba                	mv	a3,a4
1c010682:	0605                	addi	a2,a2,1
1c010684:	1eb7ef63          	bltu	a5,a1,1c010882 <mm_dw_unroll_1x2_u2+0x35a>
1c010688:	014640fb          	lp.setup	x1,a2,1c0106b0 <mm_dw_unroll_1x2_u2+0x188>
1c01068c:	008eaf0b          	p.lw	t5,8(t4!)
1c010690:	0083278b          	p.lw	a5,8(t1!)
1c010694:	008e2f8b          	p.lw	t6,8(t3!)
1c010698:	0085258b          	p.lw	a1,8(a0!)
1c01069c:	71e7f743          	fmadd.s	a4,a5,t5,a4
1c0106a0:	69eff6c3          	fmadd.s	a3,t6,t5,a3
1c0106a4:	0088278b          	p.lw	a5,8(a6!)
1c0106a8:	0088af0b          	p.lw	t5,8(a7!)
1c0106ac:	70b7f743          	fmadd.s	a4,a5,a1,a4
1c0106b0:	68bf76c3          	fmadd.s	a3,t5,a1,a3
1c0106b4:	0405                	addi	s0,s0,1
1c0106b6:	060c0a63          	beqz	s8,1c01072a <mm_dw_unroll_1x2_u2+0x202>
1c0106ba:	48d2                	lw	a7,20(sp)
1c0106bc:	4e42                	lw	t3,16(sp)
1c0106be:	209a778b          	p.lw	a5,s1(s4!)
1c0106c2:	012885b3          	add	a1,a7,s2
1c0106c6:	2099760b          	p.lw	a2,s1(s2!)
1c0106ca:	21c5f583          	p.lw	a1,t3(a1)
1c0106ce:	92a6                	add	t0,t0,s1
1c0106d0:	68c7f6c3          	fmadd.s	a3,a5,a2,a3
1c0106d4:	70b7f7c3          	fmadd.s	a5,a5,a1,a4
1c0106d8:	00d9edab          	p.sw	a3,s11(s3!)
1c0106dc:	00faedab          	p.sw	a5,s11(s5!)
1c0106e0:	f68b94e3          	bne	s7,s0,1c010648 <mm_dw_unroll_1x2_u2+0x120>
1c0106e4:	4606                	lw	a2,64(sp)
1c0106e6:	ee29                	bnez	a2,1c010740 <mm_dw_unroll_1x2_u2+0x218>
1c0106e8:	4652                	lw	a2,20(sp)
1c0106ea:	5772                	lw	a4,60(sp)
1c0106ec:	5882                	lw	a7,32(sp)
1c0106ee:	4572                	lw	a0,28(sp)
1c0106f0:	47c2                	lw	a5,16(sp)
1c0106f2:	4e62                	lw	t3,24(sp)
1c0106f4:	963a                	add	a2,a2,a4
1c0106f6:	ca32                	sw	a2,20(sp)
1c0106f8:	5642                	lw	a2,48(sp)
1c0106fa:	0509                	addi	a0,a0,2
1c0106fc:	97c6                	add	a5,a5,a7
1c0106fe:	0e21                	addi	t3,t3,8
1c010700:	ce2a                	sw	a0,28(sp)
1c010702:	c83e                	sw	a5,16(sp)
1c010704:	cc72                	sw	t3,24(sp)
1c010706:	9b46                	add	s6,s6,a7
1c010708:	9d46                	add	s10,s10,a7
1c01070a:	f0c56ce3          	bltu	a0,a2,1c010622 <mm_dw_unroll_1x2_u2+0xfa>
1c01070e:	5476                	lw	s0,124(sp)
1c010710:	54e6                	lw	s1,120(sp)
1c010712:	5956                	lw	s2,116(sp)
1c010714:	59c6                	lw	s3,112(sp)
1c010716:	5a36                	lw	s4,108(sp)
1c010718:	5aa6                	lw	s5,104(sp)
1c01071a:	5b16                	lw	s6,100(sp)
1c01071c:	5b86                	lw	s7,96(sp)
1c01071e:	4c76                	lw	s8,92(sp)
1c010720:	4ce6                	lw	s9,88(sp)
1c010722:	4d56                	lw	s10,84(sp)
1c010724:	4dc6                	lw	s11,80(sp)
1c010726:	6109                	addi	sp,sp,128
1c010728:	8082                	ret
1c01072a:	00d9edab          	p.sw	a3,s11(s3!)
1c01072e:	00eaedab          	p.sw	a4,s11(s5!)
1c010732:	9a26                	add	s4,s4,s1
1c010734:	9926                	add	s2,s2,s1
1c010736:	92a6                	add	t0,t0,s1
1c010738:	f08b98e3          	bne	s7,s0,1c010648 <mm_dw_unroll_1x2_u2+0x120>
1c01073c:	4606                	lw	a2,64(sp)
1c01073e:	d64d                	beqz	a2,1c0106e8 <mm_dw_unroll_1x2_u2+0x1c0>
1c010740:	5ed2                	lw	t4,52(sp)
1c010742:	4426                	lw	s0,72(sp)
1c010744:	52e2                	lw	t0,56(sp)
1c010746:	4fb6                	lw	t6,76(sp)
1c010748:	5f12                	lw	t5,36(sp)
1c01074a:	00000793          	li	a5,0
1c01074e:	04038763          	beqz	t2,1c01079c <mm_dw_unroll_1x2_u2+0x274>
1c010752:	5522                	lw	a0,40(sp)
1c010754:	57b2                	lw	a5,44(sp)
1c010756:	4e12                	lw	t3,4(sp)
1c010758:	fff38613          	addi	a2,t2,-1
1c01075c:	8205                	srli	a2,a2,0x1
1c01075e:	00138713          	addi	a4,t2,1
1c010762:	4689                	li	a3,2
1c010764:	01d50833          	add	a6,a0,t4
1c010768:	01dc88b3          	add	a7,s9,t4
1c01076c:	01d78533          	add	a0,a5,t4
1c010770:	01de05b3          	add	a1,t3,t4
1c010774:	00000793          	li	a5,0
1c010778:	0605                	addi	a2,a2,1
1c01077a:	10d76663          	bltu	a4,a3,1c010886 <mm_dw_unroll_1x2_u2+0x35e>
1c01077e:	0001                	nop
1c010780:	00c640fb          	lp.setup	x1,a2,1c010798 <mm_dw_unroll_1x2_u2+0x270>
1c010784:	0088ae0b          	p.lw	t3,8(a7!)
1c010788:	0088230b          	p.lw	t1,8(a6!)
1c01078c:	0085a68b          	p.lw	a3,8(a1!)
1c010790:	0085270b          	p.lw	a4,8(a0!)
1c010794:	786e77c3          	fmadd.s	a5,t3,t1,a5
1c010798:	78e6f7c3          	fmadd.s	a5,a3,a4,a5
1c01079c:	0f05                	addi	t5,t5,1
1c01079e:	000c0e63          	beqz	s8,1c0107ba <mm_dw_unroll_1x2_u2+0x292>
1c0107a2:	2092f70b          	p.lw	a4,s1(t0!)
1c0107a6:	2094768b          	p.lw	a3,s1(s0!)
1c0107aa:	9ea6                	add	t4,t4,s1
1c0107ac:	78d777c3          	fmadd.s	a5,a4,a3,a5
1c0107b0:	00ffedab          	p.sw	a5,s11(t6!)
1c0107b4:	f9eb9be3          	bne	s7,t5,1c01074a <mm_dw_unroll_1x2_u2+0x222>
1c0107b8:	bf05                	j	1c0106e8 <mm_dw_unroll_1x2_u2+0x1c0>
1c0107ba:	00ffedab          	p.sw	a5,s11(t6!)
1c0107be:	92a6                	add	t0,t0,s1
1c0107c0:	9426                	add	s0,s0,s1
1c0107c2:	9ea6                	add	t4,t4,s1
1c0107c4:	f9eb93e3          	bne	s7,t5,1c01074a <mm_dw_unroll_1x2_u2+0x222>
1c0107c8:	b705                	j	1c0106e8 <mm_dw_unroll_1x2_u2+0x1c0>
1c0107ca:	00758613          	addi	a2,a1,7
1c0107ce:	41f65793          	srai	a5,a2,0x1f
1c0107d2:	f837b7b3          	p.bclr	a5,a5,28,3
1c0107d6:	06c7a7db          	p.addn	a5,a5,a2,3
1c0107da:	f1402673          	csrr	a2,mhartid
1c0107de:	f4563633          	p.bclr	a2,a2,26,5
1c0107e2:	02f60633          	mul	a2,a2,a5
1c0107e6:	97b2                	add	a5,a5,a2
1c0107e8:	04b7c7b3          	p.min	a5,a5,a1
1c0107ec:	f216b1e3          	p.bneimm	a3,1,1c01070e <mm_dw_unroll_1x2_u2+0x1e6>
1c0107f0:	f0f65fe3          	ble	a5,a2,1c01070e <mm_dw_unroll_1x2_u2+0x1e6>
1c0107f4:	88c2                	mv	a7,a6
1c0107f6:	430608b3          	p.mac	a7,a2,a6
1c0107fa:	00279313          	slli	t1,a5,0x2
1c0107fe:	060a                	slli	a2,a2,0x2
1c010800:	40c30333          	sub	t1,t1,a2
1c010804:	00281f93          	slli	t6,a6,0x2
1c010808:	1371                	addi	t1,t1,-4
1c01080a:	41000eb3          	neg	t4,a6
1c01080e:	41fb0b33          	sub	s6,s6,t6
1c010812:	00235313          	srli	t1,t1,0x2
1c010816:	00c70e33          	add	t3,a4,a2
1c01081a:	088a                	slli	a7,a7,0x2
1c01081c:	98e6                	add	a7,a7,s9
1c01081e:	0e8a                	slli	t4,t4,0x2
1c010820:	419b0f33          	sub	t5,s6,s9
1c010824:	0305                	addi	t1,t1,1
1c010826:	0001                	nop
1c010828:	01c3407b          	lp.setup	x0,t1,1c010860 <mm_dw_unroll_1x2_u2+0x338>
1c01082c:	00000713          	li	a4,0
1c010830:	03005663          	blez	a6,1c01085c <mm_dw_unroll_1x2_u2+0x334>
1c010834:	011e86b3          	add	a3,t4,a7
1c010838:	40d887b3          	sub	a5,a7,a3
1c01083c:	17f1                	addi	a5,a5,-4
1c01083e:	00000713          	li	a4,0
1c010842:	8389                	srli	a5,a5,0x2
1c010844:	011f0633          	add	a2,t5,a7
1c010848:	0785                	addi	a5,a5,1
1c01084a:	0001                	nop
1c01084c:	0067c0fb          	lp.setup	x1,a5,1c010858 <mm_dw_unroll_1x2_u2+0x330>
1c010850:	0046a50b          	p.lw	a0,4(a3!)
1c010854:	0046258b          	p.lw	a1,4(a2!) # 40000004 <__l2_shared_end+0x23fe7af8>
1c010858:	70b57743          	fmadd.s	a4,a0,a1,a4
1c01085c:	00ee222b          	p.sw	a4,4(t3!)
1c010860:	98fe                	add	a7,a7,t6
1c010862:	b575                	j	1c01070e <mm_dw_unroll_1x2_u2+0x1e6>
1c010864:	5476                	lw	s0,124(sp)
1c010866:	54e6                	lw	s1,120(sp)
1c010868:	5956                	lw	s2,116(sp)
1c01086a:	59c6                	lw	s3,112(sp)
1c01086c:	5a36                	lw	s4,108(sp)
1c01086e:	5aa6                	lw	s5,104(sp)
1c010870:	5b16                	lw	s6,100(sp)
1c010872:	5b86                	lw	s7,96(sp)
1c010874:	4c76                	lw	s8,92(sp)
1c010876:	4ce6                	lw	s9,88(sp)
1c010878:	4d56                	lw	s10,84(sp)
1c01087a:	4dc6                	lw	s11,80(sp)
1c01087c:	6109                	addi	sp,sp,128
1c01087e:	84bff06f          	j	1c0100c8 <mm_dw_unroll_1x2>
1c010882:	4605                	li	a2,1
1c010884:	b511                	j	1c010688 <mm_dw_unroll_1x2_u2+0x160>
1c010886:	4605                	li	a2,1
1c010888:	bddd                	j	1c01077e <mm_dw_unroll_1x2_u2+0x256>
	...

1c01088c <mm_dw_unroll_1x4_u2>:
1c01088c:	490c                	lw	a1,16(a0)
1c01088e:	7171                	addi	sp,sp,-176
1c010890:	d722                	sw	s0,172(sp)
1c010892:	d526                	sw	s1,168(sp)
1c010894:	d34a                	sw	s2,164(sp)
1c010896:	d14e                	sw	s3,160(sp)
1c010898:	cf52                	sw	s4,156(sp)
1c01089a:	cd56                	sw	s5,152(sp)
1c01089c:	cb5a                	sw	s6,148(sp)
1c01089e:	c95e                	sw	s7,144(sp)
1c0108a0:	c762                	sw	s8,140(sp)
1c0108a2:	c566                	sw	s9,136(sp)
1c0108a4:	c36a                	sw	s10,132(sp)
1c0108a6:	c16e                	sw	s11,128(sp)
1c0108a8:	caae                	sw	a1,84(sp)
1c0108aa:	478d                	li	a5,3
1c0108ac:	38b7de63          	ble	a1,a5,1c010c48 <mm_dw_unroll_1x4_u2+0x3bc>
1c0108b0:	4d0c                	lw	a1,24(a0)
1c0108b2:	4785                	li	a5,1
1c0108b4:	86aa                	mv	a3,a0
1c0108b6:	3ab7d863          	ble	a1,a5,1c010c66 <mm_dw_unroll_1x4_u2+0x3da>
1c0108ba:	455c                	lw	a5,12(a0)
1c0108bc:	f1402673          	csrr	a2,mhartid
1c0108c0:	f4563633          	p.bclr	a2,a2,26,5
1c0108c4:	00778513          	addi	a0,a5,7
1c0108c8:	41f55713          	srai	a4,a0,0x1f
1c0108cc:	f8373733          	p.bclr	a4,a4,28,3
1c0108d0:	06a7275b          	p.addn	a4,a4,a0,3
1c0108d4:	02e60633          	mul	a2,a2,a4
1c0108d8:	c6b2                	sw	a2,76(sp)
1c0108da:	4836                	lw	a6,76(sp)
1c0108dc:	4656                	lw	a2,84(sp)
1c0108de:	9742                	add	a4,a4,a6
1c0108e0:	c2063633          	p.bclr	a2,a2,1,0
1c0108e4:	04f74733          	p.min	a4,a4,a5
1c0108e8:	d2b2                	sw	a2,100(sp)
1c0108ea:	cc3a                	sw	a4,24(sp)
1c0108ec:	24060563          	beqz	a2,1c010b36 <mm_dw_unroll_1x4_u2+0x2aa>
1c0108f0:	02b787b3          	mul	a5,a5,a1
1c0108f4:	4656                	lw	a2,84(sp)
1c0108f6:	00180713          	addi	a4,a6,1
1c0108fa:	0046a883          	lw	a7,4(a3)
1c0108fe:	00261b13          	slli	s6,a2,0x2
1c010902:	40000637          	lui	a2,0x40000
1c010906:	167d                	addi	a2,a2,-1
1c010908:	0006ae03          	lw	t3,0(a3)
1c01090c:	c446                	sw	a7,8(sp)
1c01090e:	00259b93          	slli	s7,a1,0x2
1c010912:	42b70633          	p.mac	a2,a4,a1
1c010916:	00279813          	slli	a6,a5,0x2
1c01091a:	d4c2                	sw	a6,104(sp)
1c01091c:	40f00833          	neg	a6,a5
1c010920:	0812                	slli	a6,a6,0x4
1c010922:	c005ba33          	p.bclr	s4,a1,0,0
1c010926:	fc15b5b3          	p.bclr	a1,a1,30,1
1c01092a:	da72                	sw	t3,52(sp)
1c01092c:	d02e                	sw	a1,32(sp)
1c01092e:	4e56                	lw	t3,84(sp)
1c010930:	dac2                	sw	a6,116(sp)
1c010932:	45a2                	lw	a1,8(sp)
1c010934:	5826                	lw	a6,104(sp)
1c010936:	fa2e3e33          	p.bclr	t3,t3,29,2
1c01093a:	d8f2                	sw	t3,112(sp)
1c01093c:	95c2                	add	a1,a1,a6
1c01093e:	4e22                	lw	t3,8(sp)
1c010940:	ca2e                	sw	a1,20(sp)
1c010942:	55d2                	lw	a1,52(sp)
1c010944:	00379893          	slli	a7,a5,0x3
1c010948:	060a                	slli	a2,a2,0x2
1c01094a:	962e                	add	a2,a2,a1
1c01094c:	c0c6                	sw	a7,64(sp)
1c01094e:	98f2                	add	a7,a7,t3
1c010950:	00179513          	slli	a0,a5,0x1
1c010954:	c846                	sw	a7,16(sp)
1c010956:	d0b2                	sw	a2,96(sp)
1c010958:	c2c2                	sw	a6,68(sp)
1c01095a:	5616                	lw	a2,100(sp)
1c01095c:	4836                	lw	a6,76(sp)
1c01095e:	48d6                	lw	a7,84(sp)
1c010960:	953e                	add	a0,a0,a5
1c010962:	050a                	slli	a0,a0,0x2
1c010964:	de2a                	sw	a0,60(sp)
1c010966:	00479513          	slli	a0,a5,0x4
1c01096a:	02c787b3          	mul	a5,a5,a2
1c01096e:	4694                	lw	a3,8(a3)
1c010970:	ce2a                	sw	a0,28(sp)
1c010972:	5572                	lw	a0,60(sp)
1c010974:	c482                	sw	zero,72(sp)
1c010976:	cc82                	sw	zero,88(sp)
1c010978:	9572                	add	a0,a0,t3
1c01097a:	00458e13          	addi	t3,a1,4
1c01097e:	c62a                	sw	a0,12(sp)
1c010980:	4522                	lw	a0,8(sp)
1c010982:	43180633          	p.mac	a2,a6,a7
1c010986:	078a                	slli	a5,a5,0x2
1c010988:	dc72                	sw	t3,56(sp)
1c01098a:	97aa                	add	a5,a5,a0
1c01098c:	8ada                	mv	s5,s6
1c01098e:	060a                	slli	a2,a2,0x2
1c010990:	03770733          	mul	a4,a4,s7
1c010994:	9636                	add	a2,a2,a3
1c010996:	deb2                	sw	a2,124(sp)
1c010998:	dcbe                	sw	a5,120(sp)
1c01099a:	436806b3          	p.mac	a3,a6,s6
1c01099e:	1771                	addi	a4,a4,-4
1c0109a0:	d6ba                	sw	a4,108(sp)
1c0109a2:	037805b3          	mul	a1,a6,s7
1c0109a6:	c8b6                	sw	a3,80(sp)
1c0109a8:	ceae                	sw	a1,92(sp)
1c0109aa:	4636                	lw	a2,76(sp)
1c0109ac:	4762                	lw	a4,24(sp)
1c0109ae:	12e65d63          	ble	a4,a2,1c010ae8 <mm_dw_unroll_1x4_u2+0x25c>
1c0109b2:	48a2                	lw	a7,8(sp)
1c0109b4:	c032                	sw	a2,0(sp)
1c0109b6:	45d2                	lw	a1,20(sp)
1c0109b8:	4642                	lw	a2,16(sp)
1c0109ba:	4732                	lw	a4,12(sp)
1c0109bc:	47c6                	lw	a5,80(sp)
1c0109be:	5836                	lw	a6,108(sp)
1c0109c0:	5e06                	lw	t3,96(sp)
1c0109c2:	44f6                	lw	s1,92(sp)
1c0109c4:	00488513          	addi	a0,a7,4
1c0109c8:	0591                	addi	a1,a1,4
1c0109ca:	0611                	addi	a2,a2,4
1c0109cc:	0711                	addi	a4,a4,4
1c0109ce:	00478913          	addi	s2,a5,4
1c0109d2:	011809b3          	add	s3,a6,a7
1c0109d6:	c272                	sw	t3,4(sp)
1c0109d8:	8b3e                	mv	s6,a5
1c0109da:	d22a                	sw	a0,36(sp)
1c0109dc:	d42e                	sw	a1,40(sp)
1c0109de:	d632                	sw	a2,44(sp)
1c0109e0:	d83a                	sw	a4,48(sp)
1c0109e2:	180a0c63          	beqz	s4,1c010b7a <mm_dw_unroll_1x4_u2+0x2ee>
1c0109e6:	47a2                	lw	a5,8(sp)
1c0109e8:	5652                	lw	a2,52(sp)
1c0109ea:	4852                	lw	a6,20(sp)
1c0109ec:	5592                	lw	a1,36(sp)
1c0109ee:	009783b3          	add	t2,a5,s1
1c0109f2:	57b2                	lw	a5,44(sp)
1c0109f4:	48c2                	lw	a7,16(sp)
1c0109f6:	4e32                	lw	t3,12(sp)
1c0109f8:	00960433          	add	s0,a2,s1
1c0109fc:	009802b3          	add	t0,a6,s1
1c010a00:	5622                	lw	a2,40(sp)
1c010a02:	5842                	lw	a6,48(sp)
1c010a04:	00958eb3          	add	t4,a1,s1
1c010a08:	00978333          	add	t1,a5,s1
1c010a0c:	55e2                	lw	a1,56(sp)
1c010a0e:	00000793          	li	a5,0
1c010a12:	fffa0513          	addi	a0,s4,-1
1c010a16:	8105                	srli	a0,a0,0x1
1c010a18:	001a0713          	addi	a4,s4,1
1c010a1c:	4c09                	li	s8,2
1c010a1e:	00988fb3          	add	t6,a7,s1
1c010a22:	009e0f33          	add	t5,t3,s1
1c010a26:	009808b3          	add	a7,a6,s1
1c010a2a:	00960e33          	add	t3,a2,s1
1c010a2e:	00958833          	add	a6,a1,s1
1c010a32:	863e                	mv	a2,a5
1c010a34:	85be                	mv	a1,a5
1c010a36:	86be                	mv	a3,a5
1c010a38:	0505                	addi	a0,a0,1
1c010a3a:	25876763          	bltu	a4,s8,1c010c88 <mm_dw_unroll_1x4_u2+0x3fc>
1c010a3e:	0001                	nop
1c010a40:	024540fb          	lp.setup	x1,a0,1c010a88 <mm_dw_unroll_1x4_u2+0x1fc>
1c010a44:	00842c0b          	p.lw	s8,8(s0!)
1c010a48:	0083ad8b          	p.lw	s11,8(t2!)
1c010a4c:	0082ad0b          	p.lw	s10,8(t0!)
1c010a50:	008fac8b          	p.lw	s9,8(t6!)
1c010a54:	008f270b          	p.lw	a4,8(t5!)
1c010a58:	698df6c3          	fmadd.s	a3,s11,s8,a3
1c010a5c:	618d7643          	fmadd.s	a2,s10,s8,a2
1c010a60:	598cf5c3          	fmadd.s	a1,s9,s8,a1
1c010a64:	798777c3          	fmadd.s	a5,a4,s8,a5
1c010a68:	008ead8b          	p.lw	s11,8(t4!)
1c010a6c:	0088270b          	p.lw	a4,8(a6!)
1c010a70:	008e2d0b          	p.lw	s10,8(t3!)
1c010a74:	00832c8b          	p.lw	s9,8(t1!)
1c010a78:	0088ac0b          	p.lw	s8,8(a7!)
1c010a7c:	68edf6c3          	fmadd.s	a3,s11,a4,a3
1c010a80:	60ed7643          	fmadd.s	a2,s10,a4,a2
1c010a84:	58ecf5c3          	fmadd.s	a1,s9,a4,a1
1c010a88:	78ec77c3          	fmadd.s	a5,s8,a4,a5
1c010a8c:	4702                	lw	a4,0(sp)
1c010a8e:	5802                	lw	a6,32(sp)
1c010a90:	0705                	addi	a4,a4,1
1c010a92:	c03a                	sw	a4,0(sp)
1c010a94:	0a080f63          	beqz	a6,1c010b52 <mm_dw_unroll_1x4_u2+0x2c6>
1c010a98:	4892                	lw	a7,4(sp)
1c010a9a:	4e26                	lw	t3,72(sp)
1c010a9c:	4816                	lw	a6,68(sp)
1c010a9e:	2178f50b          	p.lw	a0,s7(a7!)
1c010aa2:	013e0733          	add	a4,t3,s3
1c010aa6:	4e06                	lw	t3,64(sp)
1c010aa8:	2179f30b          	p.lw	t1,s7(s3!)
1c010aac:	c246                	sw	a7,4(sp)
1c010aae:	21077883          	p.lw	a7,a6(a4)
1c010ab2:	21c77803          	p.lw	a6,t3(a4)
1c010ab6:	5e72                	lw	t3,60(sp)
1c010ab8:	68a376c3          	fmadd.s	a3,t1,a0,a3
1c010abc:	58a875c3          	fmadd.s	a1,a6,a0,a1
1c010ac0:	21c77703          	p.lw	a4,t3(a4)
1c010ac4:	60a8f643          	fmadd.s	a2,a7,a0,a2
1c010ac8:	94de                	add	s1,s1,s7
1c010aca:	78a777c3          	fmadd.s	a5,a4,a0,a5
1c010ace:	00db6aab          	p.sw	a3,s5(s6!)
1c010ad2:	00b92223          	sw	a1,4(s2)
1c010ad6:	4562                	lw	a0,24(sp)
1c010ad8:	4582                	lw	a1,0(sp)
1c010ada:	00c92023          	sw	a2,0(s2)
1c010ade:	00f92423          	sw	a5,8(s2)
1c010ae2:	9956                	add	s2,s2,s5
1c010ae4:	eeb51fe3          	bne	a0,a1,1c0109e2 <mm_dw_unroll_1x4_u2+0x156>
1c010ae8:	5846                	lw	a6,112(sp)
1c010aea:	08081e63          	bnez	a6,1c010b86 <mm_dw_unroll_1x4_u2+0x2fa>
1c010aee:	4572                	lw	a0,28(sp)
1c010af0:	47e6                	lw	a5,88(sp)
1c010af2:	4826                	lw	a6,72(sp)
1c010af4:	58d6                	lw	a7,116(sp)
1c010af6:	4e16                	lw	t3,68(sp)
1c010af8:	0791                	addi	a5,a5,4
1c010afa:	9846                	add	a6,a6,a7
1c010afc:	9e2a                	add	t3,t3,a0
1c010afe:	4586                	lw	a1,64(sp)
1c010b00:	ccbe                	sw	a5,88(sp)
1c010b02:	5672                	lw	a2,60(sp)
1c010b04:	47a2                	lw	a5,8(sp)
1c010b06:	48c2                	lw	a7,16(sp)
1c010b08:	c4c2                	sw	a6,72(sp)
1c010b0a:	c2f2                	sw	t3,68(sp)
1c010b0c:	4852                	lw	a6,20(sp)
1c010b0e:	4e32                	lw	t3,12(sp)
1c010b10:	4746                	lw	a4,80(sp)
1c010b12:	95aa                	add	a1,a1,a0
1c010b14:	962a                	add	a2,a2,a0
1c010b16:	97aa                	add	a5,a5,a0
1c010b18:	982a                	add	a6,a6,a0
1c010b1a:	98aa                	add	a7,a7,a0
1c010b1c:	9e2a                	add	t3,t3,a0
1c010b1e:	c0ae                	sw	a1,64(sp)
1c010b20:	5516                	lw	a0,100(sp)
1c010b22:	45e6                	lw	a1,88(sp)
1c010b24:	0741                	addi	a4,a4,16
1c010b26:	de32                	sw	a2,60(sp)
1c010b28:	c8ba                	sw	a4,80(sp)
1c010b2a:	c43e                	sw	a5,8(sp)
1c010b2c:	ca42                	sw	a6,20(sp)
1c010b2e:	c846                	sw	a7,16(sp)
1c010b30:	c672                	sw	t3,12(sp)
1c010b32:	e6a5ece3          	bltu	a1,a0,1c0109aa <mm_dw_unroll_1x4_u2+0x11e>
1c010b36:	543a                	lw	s0,172(sp)
1c010b38:	54aa                	lw	s1,168(sp)
1c010b3a:	591a                	lw	s2,164(sp)
1c010b3c:	598a                	lw	s3,160(sp)
1c010b3e:	4a7a                	lw	s4,156(sp)
1c010b40:	4aea                	lw	s5,152(sp)
1c010b42:	4b5a                	lw	s6,148(sp)
1c010b44:	4bca                	lw	s7,144(sp)
1c010b46:	4c3a                	lw	s8,140(sp)
1c010b48:	4caa                	lw	s9,136(sp)
1c010b4a:	4d1a                	lw	s10,132(sp)
1c010b4c:	4d8a                	lw	s11,128(sp)
1c010b4e:	614d                	addi	sp,sp,176
1c010b50:	8082                	ret
1c010b52:	00db6aab          	p.sw	a3,s5(s6!)
1c010b56:	00c92023          	sw	a2,0(s2)
1c010b5a:	4612                	lw	a2,4(sp)
1c010b5c:	00f92423          	sw	a5,8(s2)
1c010b60:	4762                	lw	a4,24(sp)
1c010b62:	4782                	lw	a5,0(sp)
1c010b64:	965e                	add	a2,a2,s7
1c010b66:	00b92223          	sw	a1,4(s2)
1c010b6a:	c232                	sw	a2,4(sp)
1c010b6c:	9956                	add	s2,s2,s5
1c010b6e:	99de                	add	s3,s3,s7
1c010b70:	94de                	add	s1,s1,s7
1c010b72:	f6f70be3          	beq	a4,a5,1c010ae8 <mm_dw_unroll_1x4_u2+0x25c>
1c010b76:	e60a18e3          	bnez	s4,1c0109e6 <mm_dw_unroll_1x4_u2+0x15a>
1c010b7a:	00000793          	li	a5,0
1c010b7e:	85be                	mv	a1,a5
1c010b80:	863e                	mv	a2,a5
1c010b82:	86be                	mv	a3,a5
1c010b84:	b721                	j	1c010a8c <mm_dw_unroll_1x4_u2+0x200>
1c010b86:	5896                	lw	a7,100(sp)
1c010b88:	4e56                	lw	t3,84(sp)
1c010b8a:	f7c8d2e3          	ble	t3,a7,1c010aee <mm_dw_unroll_1x4_u2+0x262>
1c010b8e:	4536                	lw	a0,76(sp)
1c010b90:	45e2                	lw	a1,24(sp)
1c010b92:	f4b55ee3          	ble	a1,a0,1c010aee <mm_dw_unroll_1x4_u2+0x262>
1c010b96:	5466                	lw	s0,120(sp)
1c010b98:	59f6                	lw	s3,124(sp)
1c010b9a:	5482                	lw	s1,32(sp)
1c010b9c:	5cd2                	lw	s9,52(sp)
1c010b9e:	5db6                	lw	s11,108(sp)
1c010ba0:	5d62                	lw	s10,56(sp)
1c010ba2:	8946                	mv	s2,a7
1c010ba4:	8b2e                	mv	s6,a1
1c010ba6:	4ef6                	lw	t4,92(sp)
1c010ba8:	5286                	lw	t0,96(sp)
1c010baa:	4f36                	lw	t5,76(sp)
1c010bac:	008d83b3          	add	t2,s11,s0
1c010bb0:	8fce                	mv	t6,s3
1c010bb2:	00440c13          	addi	s8,s0,4
1c010bb6:	00000793          	li	a5,0
1c010bba:	040a0363          	beqz	s4,1c010c00 <mm_dw_unroll_1x4_u2+0x374>
1c010bbe:	fffa0693          	addi	a3,s4,-1
1c010bc2:	8285                	srli	a3,a3,0x1
1c010bc4:	001a0713          	addi	a4,s4,1
1c010bc8:	4889                	li	a7,2
1c010bca:	01dc8633          	add	a2,s9,t4
1c010bce:	01d40833          	add	a6,s0,t4
1c010bd2:	01dc0533          	add	a0,s8,t4
1c010bd6:	01dd05b3          	add	a1,s10,t4
1c010bda:	00000793          	li	a5,0
1c010bde:	0685                	addi	a3,a3,1
1c010be0:	0b176263          	bltu	a4,a7,1c010c84 <mm_dw_unroll_1x4_u2+0x3f8>
1c010be4:	00c6c0fb          	lp.setup	x1,a3,1c010bfc <mm_dw_unroll_1x4_u2+0x370>
1c010be8:	00862e0b          	p.lw	t3,8(a2!) # 40000008 <__l2_shared_end+0x23fe7afc>
1c010bec:	0088230b          	p.lw	t1,8(a6!)
1c010bf0:	0085a88b          	p.lw	a7,8(a1!)
1c010bf4:	0085270b          	p.lw	a4,8(a0!)
1c010bf8:	786e77c3          	fmadd.s	a5,t3,t1,a5
1c010bfc:	78e8f7c3          	fmadd.s	a5,a7,a4,a5
1c010c00:	0f05                	addi	t5,t5,1
1c010c02:	c485                	beqz	s1,1c010c2a <mm_dw_unroll_1x4_u2+0x39e>
1c010c04:	2172f70b          	p.lw	a4,s7(t0!)
1c010c08:	2173f68b          	p.lw	a3,s7(t2!)
1c010c0c:	9ede                	add	t4,t4,s7
1c010c0e:	78d777c3          	fmadd.s	a5,a4,a3,a5
1c010c12:	00ffeaab          	p.sw	a5,s5(t6!)
1c010c16:	fbeb10e3          	bne	s6,t5,1c010bb6 <mm_dw_unroll_1x4_u2+0x32a>
1c010c1a:	5626                	lw	a2,104(sp)
1c010c1c:	4756                	lw	a4,84(sp)
1c010c1e:	0905                	addi	s2,s2,1
1c010c20:	0991                	addi	s3,s3,4
1c010c22:	9432                	add	s0,s0,a2
1c010c24:	f92711e3          	bne	a4,s2,1c010ba6 <mm_dw_unroll_1x4_u2+0x31a>
1c010c28:	b5d9                	j	1c010aee <mm_dw_unroll_1x4_u2+0x262>
1c010c2a:	00ffeaab          	p.sw	a5,s5(t6!)
1c010c2e:	92de                	add	t0,t0,s7
1c010c30:	93de                	add	t2,t2,s7
1c010c32:	9ede                	add	t4,t4,s7
1c010c34:	f9eb11e3          	bne	s6,t5,1c010bb6 <mm_dw_unroll_1x4_u2+0x32a>
1c010c38:	5626                	lw	a2,104(sp)
1c010c3a:	4756                	lw	a4,84(sp)
1c010c3c:	0905                	addi	s2,s2,1
1c010c3e:	0991                	addi	s3,s3,4
1c010c40:	9432                	add	s0,s0,a2
1c010c42:	f72712e3          	bne	a4,s2,1c010ba6 <mm_dw_unroll_1x4_u2+0x31a>
1c010c46:	b565                	j	1c010aee <mm_dw_unroll_1x4_u2+0x262>
1c010c48:	543a                	lw	s0,172(sp)
1c010c4a:	54aa                	lw	s1,168(sp)
1c010c4c:	591a                	lw	s2,164(sp)
1c010c4e:	598a                	lw	s3,160(sp)
1c010c50:	4a7a                	lw	s4,156(sp)
1c010c52:	4aea                	lw	s5,152(sp)
1c010c54:	4b5a                	lw	s6,148(sp)
1c010c56:	4bca                	lw	s7,144(sp)
1c010c58:	4c3a                	lw	s8,140(sp)
1c010c5a:	4caa                	lw	s9,136(sp)
1c010c5c:	4d1a                	lw	s10,132(sp)
1c010c5e:	4d8a                	lw	s11,128(sp)
1c010c60:	614d                	addi	sp,sp,176
1c010c62:	c66ff06f          	j	1c0100c8 <mm_dw_unroll_1x2>
1c010c66:	543a                	lw	s0,172(sp)
1c010c68:	54aa                	lw	s1,168(sp)
1c010c6a:	591a                	lw	s2,164(sp)
1c010c6c:	598a                	lw	s3,160(sp)
1c010c6e:	4a7a                	lw	s4,156(sp)
1c010c70:	4aea                	lw	s5,152(sp)
1c010c72:	4b5a                	lw	s6,148(sp)
1c010c74:	4bca                	lw	s7,144(sp)
1c010c76:	4c3a                	lw	s8,140(sp)
1c010c78:	4caa                	lw	s9,136(sp)
1c010c7a:	4d1a                	lw	s10,132(sp)
1c010c7c:	4d8a                	lw	s11,128(sp)
1c010c7e:	614d                	addi	sp,sp,176
1c010c80:	e64ff06f          	j	1c0102e4 <mm_dw_unroll_1x4>
1c010c84:	4685                	li	a3,1
1c010c86:	bfb9                	j	1c010be4 <mm_dw_unroll_1x4_u2+0x358>
1c010c88:	4505                	li	a0,1
1c010c8a:	bb55                	j	1c010a3e <mm_dw_unroll_1x4_u2+0x1b2>
1c010c8c:	0000                	unimp
	...

1c010c90 <mm_dw_in_grad_u2>:
1c010c90:	4d18                	lw	a4,24(a0)
1c010c92:	4785                	li	a5,1
1c010c94:	16e7d363          	ble	a4,a5,1c010dfa <mm_dw_in_grad_u2+0x16a>
1c010c98:	715d                	addi	sp,sp,-80
1c010c9a:	d26a                	sw	s10,36(sp)
1c010c9c:	01452d03          	lw	s10,20(a0)
1c010ca0:	c2ca                	sw	s2,68(sp)
1c010ca2:	455c                	lw	a5,12(a0)
1c010ca4:	02ed4933          	div	s2,s10,a4
1c010ca8:	d662                	sw	s8,44(sp)
1c010caa:	ca3e                	sw	a5,20(sp)
1c010cac:	f1402c73          	csrr	s8,mhartid
1c010cb0:	f45c3c33          	p.bclr	s8,s8,26,5
1c010cb4:	c6a2                	sw	s0,76(sp)
1c010cb6:	c4a6                	sw	s1,72(sp)
1c010cb8:	c0ce                	sw	s3,64(sp)
1c010cba:	de52                	sw	s4,60(sp)
1c010cbc:	dc56                	sw	s5,56(sp)
1c010cbe:	da5a                	sw	s6,52(sp)
1c010cc0:	d85e                	sw	s7,48(sp)
1c010cc2:	d466                	sw	s9,40(sp)
1c010cc4:	d06e                	sw	s11,32(sp)
1c010cc6:	00790613          	addi	a2,s2,7
1c010cca:	41f65793          	srai	a5,a2,0x1f
1c010cce:	f837b7b3          	p.bclr	a5,a5,28,3
1c010cd2:	06c7a7db          	p.addn	a5,a5,a2,3
1c010cd6:	02fc0c33          	mul	s8,s8,a5
1c010cda:	97e2                	add	a5,a5,s8
1c010cdc:	0527c933          	p.min	s2,a5,s2
1c010ce0:	47d2                	lw	a5,20(sp)
1c010ce2:	0ef05163          	blez	a5,1c010dc4 <mm_dw_in_grad_u2+0x134>
1c010ce6:	01052c83          	lw	s9,16(a0)
1c010cea:	4110                	lw	a2,0(a0)
1c010cec:	414c                	lw	a1,4(a0)
1c010cee:	c00732b3          	p.bclr	t0,a4,0,0
1c010cf2:	4508                	lw	a0,8(a0)
1c010cf4:	fc173a33          	p.bclr	s4,a4,30,1
1c010cf8:	0d905663          	blez	s9,1c010dc4 <mm_dw_in_grad_u2+0x134>
1c010cfc:	0d2c5463          	ble	s2,s8,1c010dc4 <mm_dw_in_grad_u2+0x134>
1c010d00:	400007b7          	lui	a5,0x40000
1c010d04:	17fd                	addi	a5,a5,-1
1c010d06:	97ba                	add	a5,a5,a4
1c010d08:	42ec07b3          	p.mac	a5,s8,a4
1c010d0c:	40e00433          	neg	s0,a4
1c010d10:	00271993          	slli	s3,a4,0x2
1c010d14:	040a                	slli	s0,s0,0x2
1c010d16:	00440493          	addi	s1,s0,4
1c010d1a:	0d0a                	slli	s10,s10,0x2
1c010d1c:	002c9d93          	slli	s11,s9,0x2
1c010d20:	c802                	sw	zero,16(sp)
1c010d22:	0421                	addi	s0,s0,8
1c010d24:	038c86b3          	mul	a3,s9,s8
1c010d28:	078a                	slli	a5,a5,0x2
1c010d2a:	95be                	add	a1,a1,a5
1c010d2c:	963e                	add	a2,a2,a5
1c010d2e:	cc2e                	sw	a1,24(sp)
1c010d30:	c632                	sw	a2,12(sp)
1c010d32:	00269713          	slli	a4,a3,0x2
1c010d36:	972a                	add	a4,a4,a0
1c010d38:	ce3a                	sw	a4,28(sp)
1c010d3a:	4be2                	lw	s7,24(sp)
1c010d3c:	4b72                	lw	s6,28(sp)
1c010d3e:	4a81                	li	s5,0
1c010d40:	4eb2                	lw	t4,12(sp)
1c010d42:	8f5e                	mv	t5,s7
1c010d44:	83da                	mv	t2,s6
1c010d46:	8fe2                	mv	t6,s8
1c010d48:	00000793          	li	a5,0
1c010d4c:	04028463          	beqz	t0,1c010d94 <mm_dw_in_grad_u2+0x104>
1c010d50:	fff28693          	addi	a3,t0,-1
1c010d54:	8285                	srli	a3,a3,0x1
1c010d56:	00128713          	addi	a4,t0,1
1c010d5a:	4889                	li	a7,2
1c010d5c:	009e8833          	add	a6,t4,s1
1c010d60:	009f0533          	add	a0,t5,s1
1c010d64:	008f05b3          	add	a1,t5,s0
1c010d68:	008e8633          	add	a2,t4,s0
1c010d6c:	00000793          	li	a5,0
1c010d70:	0685                	addi	a3,a3,1
1c010d72:	09176663          	bltu	a4,a7,1c010dfe <mm_dw_in_grad_u2+0x16e>
1c010d76:	0001                	nop
1c010d78:	00c6c0fb          	lp.setup	x1,a3,1c010d90 <mm_dw_in_grad_u2+0x100>
1c010d7c:	00882e0b          	p.lw	t3,8(a6!)
1c010d80:	0085230b          	p.lw	t1,8(a0!)
1c010d84:	0086288b          	p.lw	a7,8(a2!)
1c010d88:	0085a70b          	p.lw	a4,8(a1!)
1c010d8c:	786e77c3          	fmadd.s	a5,t3,t1,a5
1c010d90:	78e8f7c3          	fmadd.s	a5,a7,a4,a5
1c010d94:	040a0663          	beqz	s4,1c010de0 <mm_dw_in_grad_u2+0x150>
1c010d98:	213ef70b          	p.lw	a4,s3(t4!)
1c010d9c:	213f768b          	p.lw	a3,s3(t5!)
1c010da0:	0f85                	addi	t6,t6,1
1c010da2:	78d777c3          	fmadd.s	a5,a4,a3,a5
1c010da6:	00f3edab          	p.sw	a5,s11(t2!)
1c010daa:	f9f91fe3          	bne	s2,t6,1c010d48 <mm_dw_in_grad_u2+0xb8>
1c010dae:	0a85                	addi	s5,s5,1
1c010db0:	0b11                	addi	s6,s6,4
1c010db2:	9bea                	add	s7,s7,s10
1c010db4:	f95c96e3          	bne	s9,s5,1c010d40 <mm_dw_in_grad_u2+0xb0>
1c010db8:	4742                	lw	a4,16(sp)
1c010dba:	47d2                	lw	a5,20(sp)
1c010dbc:	0705                	addi	a4,a4,1
1c010dbe:	c83a                	sw	a4,16(sp)
1c010dc0:	f6e79de3          	bne	a5,a4,1c010d3a <mm_dw_in_grad_u2+0xaa>
1c010dc4:	4436                	lw	s0,76(sp)
1c010dc6:	44a6                	lw	s1,72(sp)
1c010dc8:	4916                	lw	s2,68(sp)
1c010dca:	4986                	lw	s3,64(sp)
1c010dcc:	5a72                	lw	s4,60(sp)
1c010dce:	5ae2                	lw	s5,56(sp)
1c010dd0:	5b52                	lw	s6,52(sp)
1c010dd2:	5bc2                	lw	s7,48(sp)
1c010dd4:	5c32                	lw	s8,44(sp)
1c010dd6:	5ca2                	lw	s9,40(sp)
1c010dd8:	5d12                	lw	s10,36(sp)
1c010dda:	5d82                	lw	s11,32(sp)
1c010ddc:	6161                	addi	sp,sp,80
1c010dde:	8082                	ret
1c010de0:	00f3edab          	p.sw	a5,s11(t2!)
1c010de4:	0f85                	addi	t6,t6,1
1c010de6:	9ece                	add	t4,t4,s3
1c010de8:	9f4e                	add	t5,t5,s3
1c010dea:	f5f91fe3          	bne	s2,t6,1c010d48 <mm_dw_in_grad_u2+0xb8>
1c010dee:	0a85                	addi	s5,s5,1
1c010df0:	0b11                	addi	s6,s6,4
1c010df2:	9bea                	add	s7,s7,s10
1c010df4:	f55c96e3          	bne	s9,s5,1c010d40 <mm_dw_in_grad_u2+0xb0>
1c010df8:	b7c1                	j	1c010db8 <mm_dw_in_grad_u2+0x128>
1c010dfa:	cd2f806f          	j	1c0092cc <mm_dw_in_grad>
1c010dfe:	4685                	li	a3,1
1c010e00:	bf9d                	j	1c010d76 <mm_dw_in_grad_u2+0xe6>
	...

1c010e04 <mm_dw_in_grad_u3>:
1c010e04:	4d18                	lw	a4,24(a0)
1c010e06:	4789                	li	a5,2
1c010e08:	14e7de63          	ble	a4,a5,1c010f64 <mm_dw_in_grad_u3+0x160>
1c010e0c:	715d                	addi	sp,sp,-80
1c010e0e:	d466                	sw	s9,40(sp)
1c010e10:	01452c83          	lw	s9,20(a0)
1c010e14:	da5a                	sw	s6,52(sp)
1c010e16:	455c                	lw	a5,12(a0)
1c010e18:	02eccb33          	div	s6,s9,a4
1c010e1c:	d85e                	sw	s7,48(sp)
1c010e1e:	ca3e                	sw	a5,20(sp)
1c010e20:	f1402bf3          	csrr	s7,mhartid
1c010e24:	f45bbbb3          	p.bclr	s7,s7,26,5
1c010e28:	4e52                	lw	t3,20(sp)
1c010e2a:	c6a2                	sw	s0,76(sp)
1c010e2c:	c4a6                	sw	s1,72(sp)
1c010e2e:	c2ca                	sw	s2,68(sp)
1c010e30:	c0ce                	sw	s3,64(sp)
1c010e32:	de52                	sw	s4,60(sp)
1c010e34:	dc56                	sw	s5,56(sp)
1c010e36:	d662                	sw	s8,44(sp)
1c010e38:	d26a                	sw	s10,36(sp)
1c010e3a:	d06e                	sw	s11,32(sp)
1c010e3c:	007b0613          	addi	a2,s6,7
1c010e40:	41f65793          	srai	a5,a2,0x1f
1c010e44:	f837b7b3          	p.bclr	a5,a5,28,3
1c010e48:	06c7a7db          	p.addn	a5,a5,a2,3
1c010e4c:	02fb8bb3          	mul	s7,s7,a5
1c010e50:	97de                	add	a5,a5,s7
1c010e52:	0567cb33          	p.min	s6,a5,s6
1c010e56:	0fc05963          	blez	t3,1c010f48 <mm_dw_in_grad_u3+0x144>
1c010e5a:	01052c03          	lw	s8,16(a0)
1c010e5e:	440d                	li	s0,3
1c010e60:	00052d03          	lw	s10,0(a0)
1c010e64:	4150                	lw	a2,4(a0)
1c010e66:	450c                	lw	a1,8(a0)
1c010e68:	c0173333          	p.bclr	t1,a4,0,1
1c010e6c:	02876433          	rem	s0,a4,s0
1c010e70:	fc173933          	p.bclr	s2,a4,30,1
1c010e74:	0d805a63          	blez	s8,1c010f48 <mm_dw_in_grad_u3+0x144>
1c010e78:	0d6bd863          	ble	s6,s7,1c010f48 <mm_dw_in_grad_u3+0x144>
1c010e7c:	400007b7          	lui	a5,0x40000
1c010e80:	17fd                	addi	a5,a5,-1
1c010e82:	97ba                	add	a5,a5,a4
1c010e84:	40e004b3          	neg	s1,a4
1c010e88:	00271293          	slli	t0,a4,0x2
1c010e8c:	048a                	slli	s1,s1,0x2
1c010e8e:	00448393          	addi	t2,s1,4
1c010e92:	0c8a                	slli	s9,s9,0x2
1c010e94:	002c1d93          	slli	s11,s8,0x2
1c010e98:	c802                	sw	zero,16(sp)
1c010e9a:	04a1                	addi	s1,s1,8
1c010e9c:	42eb87b3          	p.mac	a5,s7,a4
1c010ea0:	037c06b3          	mul	a3,s8,s7
1c010ea4:	078a                	slli	a5,a5,0x2
1c010ea6:	963e                	add	a2,a2,a5
1c010ea8:	cc32                	sw	a2,24(sp)
1c010eaa:	9d3e                	add	s10,s10,a5
1c010eac:	00269713          	slli	a4,a3,0x2
1c010eb0:	972e                	add	a4,a4,a1
1c010eb2:	ce3a                	sw	a4,28(sp)
1c010eb4:	4ae2                	lw	s5,24(sp)
1c010eb6:	4a72                	lw	s4,28(sp)
1c010eb8:	4981                	li	s3,0
1c010eba:	8e56                	mv	t3,s5
1c010ebc:	8eea                	mv	t4,s10
1c010ebe:	8fd2                	mv	t6,s4
1c010ec0:	417b0f33          	sub	t5,s6,s7
1c010ec4:	036f40fb          	lp.setup	x1,t5,1c010f30 <mm_dw_in_grad_u3+0x12c>
1c010ec8:	00000793          	li	a5,0
1c010ecc:	02030e63          	beqz	t1,1c010f08 <mm_dw_in_grad_u3+0x104>
1c010ed0:	007e85b3          	add	a1,t4,t2
1c010ed4:	007e08b3          	add	a7,t3,t2
1c010ed8:	01c48533          	add	a0,s1,t3
1c010edc:	4801                	li	a6,0
1c010ede:	c672                	sw	t3,12(sp)
1c010ee0:	4190                	lw	a2,0(a1)
1c010ee2:	00c8a70b          	p.lw	a4,12(a7!)
1c010ee6:	41d4                	lw	a3,4(a1)
1c010ee8:	00052e03          	lw	t3,0(a0)
1c010eec:	78e677c3          	fmadd.s	a5,a2,a4,a5
1c010ef0:	4590                	lw	a2,8(a1)
1c010ef2:	4158                	lw	a4,4(a0)
1c010ef4:	080d                	addi	a6,a6,3
1c010ef6:	05b1                	addi	a1,a1,12
1c010ef8:	79c6f7c3          	fmadd.s	a5,a3,t3,a5
1c010efc:	0531                	addi	a0,a0,12
1c010efe:	78e677c3          	fmadd.s	a5,a2,a4,a5
1c010f02:	fc686fe3          	bltu	a6,t1,1c010ee0 <mm_dw_in_grad_u3+0xdc>
1c010f06:	4e32                	lw	t3,12(sp)
1c010f08:	c00d                	beqz	s0,1c010f2a <mm_dw_in_grad_u3+0x126>
1c010f0a:	000ea703          	lw	a4,0(t4)
1c010f0e:	000e2683          	lw	a3,0(t3)
1c010f12:	10d77753          	fmul.s	a4,a4,a3
1c010f16:	00091863          	bnez	s2,1c010f26 <mm_dw_in_grad_u3+0x122>
1c010f1a:	ffcea683          	lw	a3,-4(t4)
1c010f1e:	ffce2603          	lw	a2,-4(t3)
1c010f22:	78c6f7c3          	fmadd.s	a5,a3,a2,a5
1c010f26:	00e7f7d3          	fadd.s	a5,a5,a4
1c010f2a:	00ffedab          	p.sw	a5,s11(t6!)
1c010f2e:	9e96                	add	t4,t4,t0
1c010f30:	9e16                	add	t3,t3,t0
1c010f32:	0985                	addi	s3,s3,1
1c010f34:	0a11                	addi	s4,s4,4
1c010f36:	9ae6                	add	s5,s5,s9
1c010f38:	f93c11e3          	bne	s8,s3,1c010eba <mm_dw_in_grad_u3+0xb6>
1c010f3c:	4742                	lw	a4,16(sp)
1c010f3e:	47d2                	lw	a5,20(sp)
1c010f40:	0705                	addi	a4,a4,1
1c010f42:	c83a                	sw	a4,16(sp)
1c010f44:	f6e798e3          	bne	a5,a4,1c010eb4 <mm_dw_in_grad_u3+0xb0>
1c010f48:	4436                	lw	s0,76(sp)
1c010f4a:	44a6                	lw	s1,72(sp)
1c010f4c:	4916                	lw	s2,68(sp)
1c010f4e:	4986                	lw	s3,64(sp)
1c010f50:	5a72                	lw	s4,60(sp)
1c010f52:	5ae2                	lw	s5,56(sp)
1c010f54:	5b52                	lw	s6,52(sp)
1c010f56:	5bc2                	lw	s7,48(sp)
1c010f58:	5c32                	lw	s8,44(sp)
1c010f5a:	5ca2                	lw	s9,40(sp)
1c010f5c:	5d12                	lw	s10,36(sp)
1c010f5e:	5d82                	lw	s11,32(sp)
1c010f60:	6161                	addi	sp,sp,80
1c010f62:	8082                	ret
1c010f64:	b68f806f          	j	1c0092cc <mm_dw_in_grad>
1c010f68:	0000                	unimp
	...

1c010f6c <mm_dw_in_grad_unroll_1x2>:
1c010f6c:	715d                	addi	sp,sp,-80
1c010f6e:	d06e                	sw	s11,32(sp)
1c010f70:	01052d83          	lw	s11,16(a0)
1c010f74:	c6a2                	sw	s0,76(sp)
1c010f76:	c4a6                	sw	s1,72(sp)
1c010f78:	c2ca                	sw	s2,68(sp)
1c010f7a:	c0ce                	sw	s3,64(sp)
1c010f7c:	de52                	sw	s4,60(sp)
1c010f7e:	dc56                	sw	s5,56(sp)
1c010f80:	da5a                	sw	s6,52(sp)
1c010f82:	d85e                	sw	s7,48(sp)
1c010f84:	d662                	sw	s8,44(sp)
1c010f86:	d466                	sw	s9,40(sp)
1c010f88:	d26a                	sw	s10,36(sp)
1c010f8a:	4785                	li	a5,1
1c010f8c:	13b7d663          	ble	s11,a5,1c0110b8 <mm_dw_in_grad_unroll_1x2+0x14c>
1c010f90:	01452b03          	lw	s6,20(a0)
1c010f94:	01852283          	lw	t0,24(a0)
1c010f98:	455c                	lw	a5,12(a0)
1c010f9a:	f1402c73          	csrr	s8,mhartid
1c010f9e:	025b4bb3          	div	s7,s6,t0
1c010fa2:	c83e                	sw	a5,16(sp)
1c010fa4:	f45c3c33          	p.bclr	s8,s8,26,5
1c010fa8:	007b8693          	addi	a3,s7,7
1c010fac:	41f6d793          	srai	a5,a3,0x1f
1c010fb0:	f837b7b3          	p.bclr	a5,a5,28,3
1c010fb4:	06d7a7db          	p.addn	a5,a5,a3,3
1c010fb8:	02fc0c33          	mul	s8,s8,a5
1c010fbc:	97e2                	add	a5,a5,s8
1c010fbe:	0577cbb3          	p.min	s7,a5,s7
1c010fc2:	47c2                	lw	a5,16(sp)
1c010fc4:	0af05e63          	blez	a5,1c011080 <mm_dw_in_grad_unroll_1x2+0x114>
1c010fc8:	c00dbd33          	p.bclr	s10,s11,0,0
1c010fcc:	4114                	lw	a3,0(a0)
1c010fce:	4150                	lw	a2,4(a0)
1c010fd0:	450c                	lw	a1,8(a0)
1c010fd2:	0a0d0763          	beqz	s10,1c011080 <mm_dw_in_grad_unroll_1x2+0x114>
1c010fd6:	0b7c5563          	ble	s7,s8,1c011080 <mm_dw_in_grad_unroll_1x2+0x114>
1c010fda:	038d87b3          	mul	a5,s11,s8
1c010fde:	00229393          	slli	t2,t0,0x2
1c010fe2:	ffc68c93          	addi	s9,a3,-4
1c010fe6:	002b1713          	slli	a4,s6,0x2
1c010fea:	8f15                	sub	a4,a4,a3
1c010fec:	9732                	add	a4,a4,a2
1c010fee:	8e15                	sub	a2,a2,a3
1c010ff0:	0b0e                	slli	s6,s6,0x3
1c010ff2:	cc3a                	sw	a4,24(sp)
1c010ff4:	0d8a                	slli	s11,s11,0x2
1c010ff6:	427c0cb3          	p.mac	s9,s8,t2
1c010ffa:	078a                	slli	a5,a5,0x2
1c010ffc:	97ae                	add	a5,a5,a1
1c010ffe:	c602                	sw	zero,12(sp)
1c011000:	ce32                	sw	a2,28(sp)
1c011002:	ca3e                	sw	a5,20(sp)
1c011004:	4af2                	lw	s5,28(sp)
1c011006:	4a62                	lw	s4,24(sp)
1c011008:	4952                	lw	s2,20(sp)
1c01100a:	4981                	li	s3,0
1c01100c:	00490313          	addi	t1,s2,4
1c011010:	86e6                	mv	a3,s9
1c011012:	88ca                	mv	a7,s2
1c011014:	004a8e93          	addi	t4,s5,4
1c011018:	004a0e13          	addi	t3,s4,4
1c01101c:	418b8833          	sub	a6,s7,s8
1c011020:	06505e63          	blez	t0,1c01109c <mm_dw_in_grad_unroll_1x2+0x130>
1c011024:	00000593          	li	a1,0
1c011028:	00d38533          	add	a0,t2,a3
1c01102c:	40d507b3          	sub	a5,a0,a3
1c011030:	17f1                	addi	a5,a5,-4
1c011032:	862e                	mv	a2,a1
1c011034:	8389                	srli	a5,a5,0x2
1c011036:	00de8f33          	add	t5,t4,a3
1c01103a:	8faa                	mv	t6,a0
1c01103c:	96f2                	add	a3,a3,t3
1c01103e:	0785                	addi	a5,a5,1
1c011040:	00a7c0fb          	lp.setup	x1,a5,1c011054 <mm_dw_in_grad_unroll_1x2+0xe8>
1c011044:	ffcfa70b          	p.lw	a4,-4(t6!)
1c011048:	004f248b          	p.lw	s1,4(t5!)
1c01104c:	0046a40b          	p.lw	s0,4(a3!)
1c011050:	60e4f643          	fmadd.s	a2,s1,a4,a2
1c011054:	58e475c3          	fmadd.s	a1,s0,a4,a1
1c011058:	00c8edab          	p.sw	a2,s11(a7!)
1c01105c:	00b36dab          	p.sw	a1,s11(t1!)
1c011060:	86aa                	mv	a3,a0
1c011062:	187d                	addi	a6,a6,-1
1c011064:	fa081ee3          	bnez	a6,1c011020 <mm_dw_in_grad_unroll_1x2+0xb4>
1c011068:	0989                	addi	s3,s3,2
1c01106a:	0921                	addi	s2,s2,8
1c01106c:	9ada                	add	s5,s5,s6
1c01106e:	9a5a                	add	s4,s4,s6
1c011070:	f9a9eee3          	bltu	s3,s10,1c01100c <mm_dw_in_grad_unroll_1x2+0xa0>
1c011074:	4732                	lw	a4,12(sp)
1c011076:	47c2                	lw	a5,16(sp)
1c011078:	0705                	addi	a4,a4,1
1c01107a:	c63a                	sw	a4,12(sp)
1c01107c:	f8e794e3          	bne	a5,a4,1c011004 <mm_dw_in_grad_unroll_1x2+0x98>
1c011080:	4436                	lw	s0,76(sp)
1c011082:	44a6                	lw	s1,72(sp)
1c011084:	4916                	lw	s2,68(sp)
1c011086:	4986                	lw	s3,64(sp)
1c011088:	5a72                	lw	s4,60(sp)
1c01108a:	5ae2                	lw	s5,56(sp)
1c01108c:	5b52                	lw	s6,52(sp)
1c01108e:	5bc2                	lw	s7,48(sp)
1c011090:	5c32                	lw	s8,44(sp)
1c011092:	5ca2                	lw	s9,40(sp)
1c011094:	5d12                	lw	s10,36(sp)
1c011096:	5d82                	lw	s11,32(sp)
1c011098:	6161                	addi	sp,sp,80
1c01109a:	8082                	ret
1c01109c:	00000593          	li	a1,0
1c0110a0:	00d38533          	add	a0,t2,a3
1c0110a4:	86aa                	mv	a3,a0
1c0110a6:	862e                	mv	a2,a1
1c0110a8:	00c8edab          	p.sw	a2,s11(a7!)
1c0110ac:	00b36dab          	p.sw	a1,s11(t1!)
1c0110b0:	187d                	addi	a6,a6,-1
1c0110b2:	f60817e3          	bnez	a6,1c011020 <mm_dw_in_grad_unroll_1x2+0xb4>
1c0110b6:	bf4d                	j	1c011068 <mm_dw_in_grad_unroll_1x2+0xfc>
1c0110b8:	4436                	lw	s0,76(sp)
1c0110ba:	44a6                	lw	s1,72(sp)
1c0110bc:	4916                	lw	s2,68(sp)
1c0110be:	4986                	lw	s3,64(sp)
1c0110c0:	5a72                	lw	s4,60(sp)
1c0110c2:	5ae2                	lw	s5,56(sp)
1c0110c4:	5b52                	lw	s6,52(sp)
1c0110c6:	5bc2                	lw	s7,48(sp)
1c0110c8:	5c32                	lw	s8,44(sp)
1c0110ca:	5ca2                	lw	s9,40(sp)
1c0110cc:	5d12                	lw	s10,36(sp)
1c0110ce:	5d82                	lw	s11,32(sp)
1c0110d0:	6161                	addi	sp,sp,80
1c0110d2:	9faf806f          	j	1c0092cc <mm_dw_in_grad>
	...

1c0110d8 <mm_dw_in_grad_unroll_1x4>:
1c0110d8:	01052303          	lw	t1,16(a0)
1c0110dc:	7175                	addi	sp,sp,-144
1c0110de:	c722                	sw	s0,140(sp)
1c0110e0:	c526                	sw	s1,136(sp)
1c0110e2:	c34a                	sw	s2,132(sp)
1c0110e4:	c14e                	sw	s3,128(sp)
1c0110e6:	ded2                	sw	s4,124(sp)
1c0110e8:	dcd6                	sw	s5,120(sp)
1c0110ea:	dada                	sw	s6,116(sp)
1c0110ec:	d8de                	sw	s7,112(sp)
1c0110ee:	d6e2                	sw	s8,108(sp)
1c0110f0:	d4e6                	sw	s9,104(sp)
1c0110f2:	d2ea                	sw	s10,100(sp)
1c0110f4:	d0ee                	sw	s11,96(sp)
1c0110f6:	da1a                	sw	t1,52(sp)
1c0110f8:	478d                	li	a5,3
1c0110fa:	2667d663          	ble	t1,a5,1c011366 <mm_dw_in_grad_unroll_1x4+0x28e>
1c0110fe:	4954                	lw	a3,20(a0)
1c011100:	01852d83          	lw	s11,24(a0)
1c011104:	455c                	lw	a5,12(a0)
1c011106:	872a                	mv	a4,a0
1c011108:	03b6c5b3          	div	a1,a3,s11
1c01110c:	dc3e                	sw	a5,56(sp)
1c01110e:	f1402673          	csrr	a2,mhartid
1c011112:	f4563633          	p.bclr	a2,a2,26,5
1c011116:	5362                	lw	t1,56(sp)
1c011118:	00758513          	addi	a0,a1,7
1c01111c:	41f55793          	srai	a5,a0,0x1f
1c011120:	f837b7b3          	p.bclr	a5,a5,28,3
1c011124:	06a7a7db          	p.addn	a5,a5,a0,3
1c011128:	02f60633          	mul	a2,a2,a5
1c01112c:	97b2                	add	a5,a5,a2
1c01112e:	04b7c7b3          	p.min	a5,a5,a1
1c011132:	d032                	sw	a2,32(sp)
1c011134:	d43e                	sw	a5,40(sp)
1c011136:	16605c63          	blez	t1,1c0112ae <mm_dw_in_grad_unroll_1x4+0x1d6>
1c01113a:	55d2                	lw	a1,52(sp)
1c01113c:	00472803          	lw	a6,4(a4)
1c011140:	4708                	lw	a0,8(a4)
1c011142:	02b60633          	mul	a2,a2,a1
1c011146:	430c                	lw	a1,0(a4)
1c011148:	5752                	lw	a4,52(sp)
1c01114a:	00269313          	slli	t1,a3,0x2
1c01114e:	c69a                	sw	t1,76(sp)
1c011150:	c2073733          	p.bclr	a4,a4,1,0
1c011154:	d23a                	sw	a4,36(sp)
1c011156:	5312                	lw	t1,36(sp)
1c011158:	002d9c93          	slli	s9,s11,0x2
1c01115c:	00169793          	slli	a5,a3,0x1
1c011160:	8732                	mv	a4,a2
1c011162:	070a                	slli	a4,a4,0x2
1c011164:	961a                	add	a2,a2,t1
1c011166:	972a                	add	a4,a4,a0
1c011168:	4336                	lw	t1,76(sp)
1c01116a:	caba                	sw	a4,84(sp)
1c01116c:	5702                	lw	a4,32(sp)
1c01116e:	40b30e33          	sub	t3,t1,a1
1c011172:	ffc58313          	addi	t1,a1,-4
1c011176:	43970333          	p.mac	t1,a4,s9
1c01117a:	97b6                	add	a5,a5,a3
1c01117c:	00369893          	slli	a7,a3,0x3
1c011180:	078a                	slli	a5,a5,0x2
1c011182:	40b888b3          	sub	a7,a7,a1
1c011186:	8f8d                	sub	a5,a5,a1
1c011188:	40b805b3          	sub	a1,a6,a1
1c01118c:	c0ae                	sw	a1,64(sp)
1c01118e:	55d2                	lw	a1,52(sp)
1c011190:	97c2                	add	a5,a5,a6
1c011192:	c2be                	sw	a5,68(sp)
1c011194:	fa25b5b3          	p.bclr	a1,a1,29,2
1c011198:	de2e                	sw	a1,60(sp)
1c01119a:	d61a                	sw	t1,44(sp)
1c01119c:	4786                	lw	a5,64(sp)
1c01119e:	4336                	lw	t1,76(sp)
1c0111a0:	5592                	lw	a1,36(sp)
1c0111a2:	0692                	slli	a3,a3,0x4
1c0111a4:	c436                	sw	a3,8(sp)
1c0111a6:	42b307b3          	p.mac	a5,t1,a1
1c0111aa:	56d2                	lw	a3,52(sp)
1c0111ac:	060a                	slli	a2,a2,0x2
1c0111ae:	9e42                	add	t3,t3,a6
1c0111b0:	98c2                	add	a7,a7,a6
1c0111b2:	962a                	add	a2,a2,a0
1c0111b4:	c8f2                	sw	t3,80(sp)
1c0111b6:	c4c6                	sw	a7,72(sp)
1c0111b8:	00269d13          	slli	s10,a3,0x2
1c0111bc:	ceb2                	sw	a2,92(sp)
1c0111be:	d802                	sw	zero,48(sp)
1c0111c0:	ccbe                	sw	a5,88(sp)
1c0111c2:	5712                	lw	a4,36(sp)
1c0111c4:	cf69                	beqz	a4,1c01129e <mm_dw_in_grad_unroll_1x4+0x1c6>
1c0111c6:	5782                	lw	a5,32(sp)
1c0111c8:	5322                	lw	t1,40(sp)
1c0111ca:	0c67da63          	ble	t1,a5,1c01129e <mm_dw_in_grad_unroll_1x4+0x1c6>
1c0111ce:	4716                	lw	a4,68(sp)
1c0111d0:	47a6                	lw	a5,72(sp)
1c0111d2:	4346                	lw	t1,80(sp)
1c0111d4:	4586                	lw	a1,64(sp)
1c0111d6:	4c56                	lw	s8,84(sp)
1c0111d8:	c63a                	sw	a4,12(sp)
1c0111da:	c83e                	sw	a5,16(sp)
1c0111dc:	cc1a                	sw	t1,24(sp)
1c0111de:	ce2e                	sw	a1,28(sp)
1c0111e0:	ca02                	sw	zero,20(sp)
1c0111e2:	4672                	lw	a2,28(sp)
1c0111e4:	4762                	lw	a4,24(sp)
1c0111e6:	47c2                	lw	a5,16(sp)
1c0111e8:	00460b93          	addi	s7,a2,4
1c0111ec:	4332                	lw	t1,12(sp)
1c0111ee:	55a2                	lw	a1,40(sp)
1c0111f0:	5602                	lw	a2,32(sp)
1c0111f2:	56b2                	lw	a3,44(sp)
1c0111f4:	004c0413          	addi	s0,s8,4
1c0111f8:	89e2                	mv	s3,s8
1c0111fa:	00470b13          	addi	s6,a4,4
1c0111fe:	00478a93          	addi	s5,a5,4 # 40000004 <__l2_shared_end+0x23fe7af8>
1c011202:	00430a13          	addi	s4,t1,4
1c011206:	40c58933          	sub	s2,a1,a2
1c01120a:	0db05063          	blez	s11,1c0112ca <mm_dw_in_grad_unroll_1x4+0x1f2>
1c01120e:	00000813          	li	a6,0
1c011212:	00dc8f33          	add	t5,s9,a3
1c011216:	40df0733          	sub	a4,t5,a3
1c01121a:	1771                	addi	a4,a4,-4
1c01121c:	8542                	mv	a0,a6
1c01121e:	85c2                	mv	a1,a6
1c011220:	8642                	mv	a2,a6
1c011222:	8309                	srli	a4,a4,0x2
1c011224:	00db8e33          	add	t3,s7,a3
1c011228:	00db0333          	add	t1,s6,a3
1c01122c:	00da88b3          	add	a7,s5,a3
1c011230:	8efa                	mv	t4,t5
1c011232:	96d2                	add	a3,a3,s4
1c011234:	0705                	addi	a4,a4,1
1c011236:	0001                	nop
1c011238:	012740fb          	lp.setup	x1,a4,1c01125c <mm_dw_in_grad_unroll_1x4+0x184>
1c01123c:	ffcea78b          	p.lw	a5,-4(t4!)
1c011240:	004e248b          	p.lw	s1,4(t3!)
1c011244:	0043238b          	p.lw	t2,4(t1!)
1c011248:	0048a28b          	p.lw	t0,4(a7!)
1c01124c:	0046af8b          	p.lw	t6,4(a3!)
1c011250:	60f4f643          	fmadd.s	a2,s1,a5,a2
1c011254:	58f3f5c3          	fmadd.s	a1,t2,a5,a1
1c011258:	50f2f543          	fmadd.s	a0,t0,a5,a0
1c01125c:	80fff843          	fmadd.s	a6,t6,a5,a6
1c011260:	00c9ed2b          	p.sw	a2,s10(s3!)
1c011264:	c00c                	sw	a1,0(s0)
1c011266:	c048                	sw	a0,4(s0)
1c011268:	01042423          	sw	a6,8(s0)
1c01126c:	86fa                	mv	a3,t5
1c01126e:	946a                	add	s0,s0,s10
1c011270:	197d                	addi	s2,s2,-1
1c011272:	f8091ce3          	bnez	s2,1c01120a <mm_dw_in_grad_unroll_1x4+0x132>
1c011276:	4722                	lw	a4,8(sp)
1c011278:	46f2                	lw	a3,28(sp)
1c01127a:	4652                	lw	a2,20(sp)
1c01127c:	47e2                	lw	a5,24(sp)
1c01127e:	4342                	lw	t1,16(sp)
1c011280:	45b2                	lw	a1,12(sp)
1c011282:	96ba                	add	a3,a3,a4
1c011284:	ce36                	sw	a3,28(sp)
1c011286:	5692                	lw	a3,36(sp)
1c011288:	0611                	addi	a2,a2,4
1c01128a:	97ba                	add	a5,a5,a4
1c01128c:	933a                	add	t1,t1,a4
1c01128e:	95ba                	add	a1,a1,a4
1c011290:	ca32                	sw	a2,20(sp)
1c011292:	cc3e                	sw	a5,24(sp)
1c011294:	c81a                	sw	t1,16(sp)
1c011296:	c62e                	sw	a1,12(sp)
1c011298:	0c41                	addi	s8,s8,16
1c01129a:	f4d664e3          	bltu	a2,a3,1c0111e2 <mm_dw_in_grad_unroll_1x4+0x10a>
1c01129e:	56f2                	lw	a3,60(sp)
1c0112a0:	eaa1                	bnez	a3,1c0112f0 <mm_dw_in_grad_unroll_1x4+0x218>
1c0112a2:	5642                	lw	a2,48(sp)
1c0112a4:	56e2                	lw	a3,56(sp)
1c0112a6:	0605                	addi	a2,a2,1
1c0112a8:	d832                	sw	a2,48(sp)
1c0112aa:	f0c69ce3          	bne	a3,a2,1c0111c2 <mm_dw_in_grad_unroll_1x4+0xea>
1c0112ae:	443a                	lw	s0,140(sp)
1c0112b0:	44aa                	lw	s1,136(sp)
1c0112b2:	491a                	lw	s2,132(sp)
1c0112b4:	498a                	lw	s3,128(sp)
1c0112b6:	5a76                	lw	s4,124(sp)
1c0112b8:	5ae6                	lw	s5,120(sp)
1c0112ba:	5b56                	lw	s6,116(sp)
1c0112bc:	5bc6                	lw	s7,112(sp)
1c0112be:	5c36                	lw	s8,108(sp)
1c0112c0:	5ca6                	lw	s9,104(sp)
1c0112c2:	5d16                	lw	s10,100(sp)
1c0112c4:	5d86                	lw	s11,96(sp)
1c0112c6:	6149                	addi	sp,sp,144
1c0112c8:	8082                	ret
1c0112ca:	00000813          	li	a6,0
1c0112ce:	00dc8f33          	add	t5,s9,a3
1c0112d2:	86fa                	mv	a3,t5
1c0112d4:	8642                	mv	a2,a6
1c0112d6:	8542                	mv	a0,a6
1c0112d8:	85c2                	mv	a1,a6
1c0112da:	00c9ed2b          	p.sw	a2,s10(s3!)
1c0112de:	c048                	sw	a0,4(s0)
1c0112e0:	c00c                	sw	a1,0(s0)
1c0112e2:	01042423          	sw	a6,8(s0)
1c0112e6:	946a                	add	s0,s0,s10
1c0112e8:	197d                	addi	s2,s2,-1
1c0112ea:	f20910e3          	bnez	s2,1c01120a <mm_dw_in_grad_unroll_1x4+0x132>
1c0112ee:	b761                	j	1c011276 <mm_dw_in_grad_unroll_1x4+0x19e>
1c0112f0:	5752                	lw	a4,52(sp)
1c0112f2:	5792                	lw	a5,36(sp)
1c0112f4:	fae7d7e3          	ble	a4,a5,1c0112a2 <mm_dw_in_grad_unroll_1x4+0x1ca>
1c0112f8:	5302                	lw	t1,32(sp)
1c0112fa:	55a2                	lw	a1,40(sp)
1c0112fc:	fab353e3          	ble	a1,t1,1c0112a2 <mm_dw_in_grad_unroll_1x4+0x1ca>
1c011300:	4fe6                	lw	t6,88(sp)
1c011302:	4f76                	lw	t5,92(sp)
1c011304:	44b6                	lw	s1,76(sp)
1c011306:	5932                	lw	s2,44(sp)
1c011308:	8ebe                	mv	t4,a5
1c01130a:	82ba                	mv	t0,a4
1c01130c:	839a                	mv	t2,t1
1c01130e:	842e                	mv	s0,a1
1c011310:	874a                	mv	a4,s2
1c011312:	837a                	mv	t1,t5
1c011314:	004f8e13          	addi	t3,t6,4
1c011318:	407408b3          	sub	a7,s0,t2
1c01131c:	01e8c07b          	lp.setup	x0,a7,1c011358 <mm_dw_in_grad_unroll_1x4+0x280>
1c011320:	00000693          	li	a3,0
1c011324:	00ec8833          	add	a6,s9,a4
1c011328:	03b05663          	blez	s11,1c011354 <mm_dw_in_grad_unroll_1x4+0x27c>
1c01132c:	00ec8833          	add	a6,s9,a4
1c011330:	40e807b3          	sub	a5,a6,a4
1c011334:	17f1                	addi	a5,a5,-4
1c011336:	00000693          	li	a3,0
1c01133a:	8389                	srli	a5,a5,0x2
1c01133c:	8642                	mv	a2,a6
1c01133e:	9772                	add	a4,a4,t3
1c011340:	0785                	addi	a5,a5,1
1c011342:	0001                	nop
1c011344:	0067c0fb          	lp.setup	x1,a5,1c011350 <mm_dw_in_grad_unroll_1x4+0x278>
1c011348:	ffc6250b          	p.lw	a0,-4(a2!)
1c01134c:	0047258b          	p.lw	a1,4(a4!)
1c011350:	68b576c3          	fmadd.s	a3,a0,a1,a3
1c011354:	00d36d2b          	p.sw	a3,s10(t1!)
1c011358:	8742                	mv	a4,a6
1c01135a:	0e85                	addi	t4,t4,1
1c01135c:	0f11                	addi	t5,t5,4
1c01135e:	9fa6                	add	t6,t6,s1
1c011360:	fbd298e3          	bne	t0,t4,1c011310 <mm_dw_in_grad_unroll_1x4+0x238>
1c011364:	bf3d                	j	1c0112a2 <mm_dw_in_grad_unroll_1x4+0x1ca>
1c011366:	443a                	lw	s0,140(sp)
1c011368:	44aa                	lw	s1,136(sp)
1c01136a:	491a                	lw	s2,132(sp)
1c01136c:	498a                	lw	s3,128(sp)
1c01136e:	5a76                	lw	s4,124(sp)
1c011370:	5ae6                	lw	s5,120(sp)
1c011372:	5b56                	lw	s6,116(sp)
1c011374:	5bc6                	lw	s7,112(sp)
1c011376:	5c36                	lw	s8,108(sp)
1c011378:	5ca6                	lw	s9,104(sp)
1c01137a:	5d16                	lw	s10,100(sp)
1c01137c:	5d86                	lw	s11,96(sp)
1c01137e:	6149                	addi	sp,sp,144
1c011380:	b6f5                	j	1c010f6c <mm_dw_in_grad_unroll_1x2>
	...

1c011384 <mm_dw_in_grad_unroll_1x2_u2>:
1c011384:	7119                	addi	sp,sp,-128
1c011386:	c8ee                	sw	s11,80(sp)
1c011388:	01052d83          	lw	s11,16(a0)
1c01138c:	dea2                	sw	s0,124(sp)
1c01138e:	dca6                	sw	s1,120(sp)
1c011390:	daca                	sw	s2,116(sp)
1c011392:	d8ce                	sw	s3,112(sp)
1c011394:	d6d2                	sw	s4,108(sp)
1c011396:	d4d6                	sw	s5,104(sp)
1c011398:	d2da                	sw	s6,100(sp)
1c01139a:	d0de                	sw	s7,96(sp)
1c01139c:	cee2                	sw	s8,92(sp)
1c01139e:	cce6                	sw	s9,88(sp)
1c0113a0:	caea                	sw	s10,84(sp)
1c0113a2:	4785                	li	a5,1
1c0113a4:	21b7d263          	ble	s11,a5,1c0115a8 <mm_dw_in_grad_unroll_1x2_u2+0x224>
1c0113a8:	4d18                	lw	a4,24(a0)
1c0113aa:	86aa                	mv	a3,a0
1c0113ac:	1ee7d063          	ble	a4,a5,1c01158c <mm_dw_in_grad_unroll_1x2_u2+0x208>
1c0113b0:	4948                	lw	a0,20(a0)
1c0113b2:	46dc                	lw	a5,12(a3)
1c0113b4:	f1402673          	csrr	a2,mhartid
1c0113b8:	02e54a33          	div	s4,a0,a4
1c0113bc:	c0be                	sw	a5,64(sp)
1c0113be:	f4563633          	p.bclr	a2,a2,26,5
1c0113c2:	4e06                	lw	t3,64(sp)
1c0113c4:	007a0593          	addi	a1,s4,7
1c0113c8:	41f5d793          	srai	a5,a1,0x1f
1c0113cc:	f837b7b3          	p.bclr	a5,a5,28,3
1c0113d0:	06b7a7db          	p.addn	a5,a5,a1,3
1c0113d4:	02f60633          	mul	a2,a2,a5
1c0113d8:	97b2                	add	a5,a5,a2
1c0113da:	d632                	sw	a2,44(sp)
1c0113dc:	0547ca33          	p.min	s4,a5,s4
1c0113e0:	17c05e63          	blez	t3,1c01155c <mm_dw_in_grad_unroll_1x2_u2+0x1d8>
1c0113e4:	c00db633          	p.bclr	a2,s11,0,0
1c0113e8:	d832                	sw	a2,48(sp)
1c0113ea:	42cc                	lw	a1,4(a3)
1c0113ec:	4290                	lw	a2,0(a3)
1c0113ee:	0086a303          	lw	t1,8(a3)
1c0113f2:	56c2                	lw	a3,48(sp)
1c0113f4:	c00734b3          	p.bclr	s1,a4,0,0
1c0113f8:	fc173b33          	p.bclr	s6,a4,30,1
1c0113fc:	16068063          	beqz	a3,1c01155c <mm_dw_in_grad_unroll_1x2_u2+0x1d8>
1c011400:	57b2                	lw	a5,44(sp)
1c011402:	1547dd63          	ble	s4,a5,1c01155c <mm_dw_in_grad_unroll_1x2_u2+0x1d8>
1c011406:	8e3e                	mv	t3,a5
1c011408:	03cd88b3          	mul	a7,s11,t3
1c01140c:	40a00e33          	neg	t3,a0
1c011410:	0e0e                	slli	t3,t3,0x3
1c011412:	40e506b3          	sub	a3,a0,a4
1c011416:	dc72                	sw	t3,56(sp)
1c011418:	00351e13          	slli	t3,a0,0x3
1c01141c:	050a                	slli	a0,a0,0x2
1c01141e:	400007b7          	lui	a5,0x40000
1c011422:	c6aa                	sw	a0,76(sp)
1c011424:	5532                	lw	a0,44(sp)
1c011426:	17fd                	addi	a5,a5,-1
1c011428:	97ba                	add	a5,a5,a4
1c01142a:	42e507b3          	p.mac	a5,a0,a4
1c01142e:	00271a93          	slli	s5,a4,0x2
1c011432:	40e00833          	neg	a6,a4
1c011436:	00289713          	slli	a4,a7,0x2
1c01143a:	068a                	slli	a3,a3,0x2
1c01143c:	971a                	add	a4,a4,t1
1c01143e:	080a                	slli	a6,a6,0x2
1c011440:	d472                	sw	t3,40(sp)
1c011442:	c4ba                	sw	a4,72(sp)
1c011444:	00468e13          	addi	t3,a3,4
1c011448:	43550633          	p.mac	a2,a0,s5
1c01144c:	078a                	slli	a5,a5,0x2
1c01144e:	ff8a8713          	addi	a4,s5,-8
1c011452:	97ae                	add	a5,a5,a1
1c011454:	0d8a                	slli	s11,s11,0x2
1c011456:	de02                	sw	zero,60(sp)
1c011458:	c63a                	sw	a4,12(sp)
1c01145a:	00480b93          	addi	s7,a6,4
1c01145e:	c872                	sw	t3,16(sp)
1c011460:	00880c13          	addi	s8,a6,8
1c011464:	00868d13          	addi	s10,a3,8
1c011468:	c2be                	sw	a5,68(sp)
1c01146a:	da32                	sw	a2,52(sp)
1c01146c:	ffca8c93          	addi	s9,s5,-4
1c011470:	4616                	lw	a2,68(sp)
1c011472:	46a6                	lw	a3,72(sp)
1c011474:	4736                	lw	a4,76(sp)
1c011476:	d232                	sw	a2,36(sp)
1c011478:	ce36                	sw	a3,28(sp)
1c01147a:	ca3a                	sw	a4,20(sp)
1c01147c:	cc02                	sw	zero,24(sp)
1c01147e:	d002                	sw	zero,32(sp)
1c011480:	47f2                	lw	a5,28(sp)
1c011482:	5292                	lw	t0,36(sp)
1c011484:	53d2                	lw	t2,52(sp)
1c011486:	5432                	lw	s0,44(sp)
1c011488:	00478993          	addi	s3,a5,4 # 40000004 <__l2_shared_end+0x23fe7af8>
1c01148c:	893e                	mv	s2,a5
1c01148e:	00000693          	li	a3,0
1c011492:	8736                	mv	a4,a3
1c011494:	c0b5                	beqz	s1,1c0114f8 <mm_dw_in_grad_unroll_1x2_u2+0x174>
1c011496:	46c2                	lw	a3,16(sp)
1c011498:	4532                	lw	a0,12(sp)
1c01149a:	fff48613          	addi	a2,s1,-1
1c01149e:	00568333          	add	t1,a3,t0
1c0114a2:	00000693          	li	a3,0
1c0114a6:	8205                	srli	a2,a2,0x1
1c0114a8:	00148793          	addi	a5,s1,1
1c0114ac:	4589                	li	a1,2
1c0114ae:	00750eb3          	add	t4,a0,t2
1c0114b2:	005b8e33          	add	t3,s7,t0
1c0114b6:	005c08b3          	add	a7,s8,t0
1c0114ba:	005d0833          	add	a6,s10,t0
1c0114be:	007c8533          	add	a0,s9,t2
1c0114c2:	8736                	mv	a4,a3
1c0114c4:	0605                	addi	a2,a2,1
1c0114c6:	10b7e063          	bltu	a5,a1,1c0115c6 <mm_dw_in_grad_unroll_1x2_u2+0x242>
1c0114ca:	0001                	nop
1c0114cc:	014640fb          	lp.setup	x1,a2,1c0114f4 <mm_dw_in_grad_unroll_1x2_u2+0x170>
1c0114d0:	ff852f0b          	p.lw	t5,-8(a0!)
1c0114d4:	0083278b          	p.lw	a5,8(t1!)
1c0114d8:	008e2f8b          	p.lw	t6,8(t3!)
1c0114dc:	ff8ea58b          	p.lw	a1,-8(t4!)
1c0114e0:	71e7f743          	fmadd.s	a4,a5,t5,a4
1c0114e4:	69eff6c3          	fmadd.s	a3,t6,t5,a3
1c0114e8:	0088278b          	p.lw	a5,8(a6!)
1c0114ec:	0088af0b          	p.lw	t5,8(a7!)
1c0114f0:	70b7f743          	fmadd.s	a4,a5,a1,a4
1c0114f4:	68bf76c3          	fmadd.s	a3,t5,a1,a3
1c0114f8:	080b0063          	beqz	s6,1c011578 <mm_dw_in_grad_unroll_1x2_u2+0x1f4>
1c0114fc:	4e62                	lw	t3,24(sp)
1c0114fe:	4552                	lw	a0,20(sp)
1c011500:	2153f58b          	p.lw	a1,s5(t2!)
1c011504:	005e07b3          	add	a5,t3,t0
1c011508:	2152f60b          	p.lw	a2,s5(t0!)
1c01150c:	20a7f783          	p.lw	a5,a0(a5)
1c011510:	0405                	addi	s0,s0,1
1c011512:	68b676c3          	fmadd.s	a3,a2,a1,a3
1c011516:	70b7f743          	fmadd.s	a4,a5,a1,a4
1c01151a:	00d96dab          	p.sw	a3,s11(s2!)
1c01151e:	00e9edab          	p.sw	a4,s11(s3!)
1c011522:	f68a16e3          	bne	s4,s0,1c01148e <mm_dw_in_grad_unroll_1x2_u2+0x10a>
1c011526:	5602                	lw	a2,32(sp)
1c011528:	46e2                	lw	a3,24(sp)
1c01152a:	5762                	lw	a4,56(sp)
1c01152c:	0609                	addi	a2,a2,2
1c01152e:	d032                	sw	a2,32(sp)
1c011530:	47d2                	lw	a5,20(sp)
1c011532:	5e22                	lw	t3,40(sp)
1c011534:	4572                	lw	a0,28(sp)
1c011536:	5612                	lw	a2,36(sp)
1c011538:	96ba                	add	a3,a3,a4
1c01153a:	cc36                	sw	a3,24(sp)
1c01153c:	5702                	lw	a4,32(sp)
1c01153e:	56c2                	lw	a3,48(sp)
1c011540:	97f2                	add	a5,a5,t3
1c011542:	0521                	addi	a0,a0,8
1c011544:	9672                	add	a2,a2,t3
1c011546:	ca3e                	sw	a5,20(sp)
1c011548:	ce2a                	sw	a0,28(sp)
1c01154a:	d232                	sw	a2,36(sp)
1c01154c:	f2d76ae3          	bltu	a4,a3,1c011480 <mm_dw_in_grad_unroll_1x2_u2+0xfc>
1c011550:	5e72                	lw	t3,60(sp)
1c011552:	4506                	lw	a0,64(sp)
1c011554:	0e05                	addi	t3,t3,1
1c011556:	de72                	sw	t3,60(sp)
1c011558:	f1c51ce3          	bne	a0,t3,1c011470 <mm_dw_in_grad_unroll_1x2_u2+0xec>
1c01155c:	5476                	lw	s0,124(sp)
1c01155e:	54e6                	lw	s1,120(sp)
1c011560:	5956                	lw	s2,116(sp)
1c011562:	59c6                	lw	s3,112(sp)
1c011564:	5a36                	lw	s4,108(sp)
1c011566:	5aa6                	lw	s5,104(sp)
1c011568:	5b16                	lw	s6,100(sp)
1c01156a:	5b86                	lw	s7,96(sp)
1c01156c:	4c76                	lw	s8,92(sp)
1c01156e:	4ce6                	lw	s9,88(sp)
1c011570:	4d56                	lw	s10,84(sp)
1c011572:	4dc6                	lw	s11,80(sp)
1c011574:	6109                	addi	sp,sp,128
1c011576:	8082                	ret
1c011578:	00d96dab          	p.sw	a3,s11(s2!)
1c01157c:	00e9edab          	p.sw	a4,s11(s3!)
1c011580:	0405                	addi	s0,s0,1
1c011582:	93d6                	add	t2,t2,s5
1c011584:	92d6                	add	t0,t0,s5
1c011586:	f08a14e3          	bne	s4,s0,1c01148e <mm_dw_in_grad_unroll_1x2_u2+0x10a>
1c01158a:	bf71                	j	1c011526 <mm_dw_in_grad_unroll_1x2_u2+0x1a2>
1c01158c:	5476                	lw	s0,124(sp)
1c01158e:	54e6                	lw	s1,120(sp)
1c011590:	5956                	lw	s2,116(sp)
1c011592:	59c6                	lw	s3,112(sp)
1c011594:	5a36                	lw	s4,108(sp)
1c011596:	5aa6                	lw	s5,104(sp)
1c011598:	5b16                	lw	s6,100(sp)
1c01159a:	5b86                	lw	s7,96(sp)
1c01159c:	4c76                	lw	s8,92(sp)
1c01159e:	4ce6                	lw	s9,88(sp)
1c0115a0:	4d56                	lw	s10,84(sp)
1c0115a2:	4dc6                	lw	s11,80(sp)
1c0115a4:	6109                	addi	sp,sp,128
1c0115a6:	b2d9                	j	1c010f6c <mm_dw_in_grad_unroll_1x2>
1c0115a8:	5476                	lw	s0,124(sp)
1c0115aa:	54e6                	lw	s1,120(sp)
1c0115ac:	5956                	lw	s2,116(sp)
1c0115ae:	59c6                	lw	s3,112(sp)
1c0115b0:	5a36                	lw	s4,108(sp)
1c0115b2:	5aa6                	lw	s5,104(sp)
1c0115b4:	5b16                	lw	s6,100(sp)
1c0115b6:	5b86                	lw	s7,96(sp)
1c0115b8:	4c76                	lw	s8,92(sp)
1c0115ba:	4ce6                	lw	s9,88(sp)
1c0115bc:	4d56                	lw	s10,84(sp)
1c0115be:	4dc6                	lw	s11,80(sp)
1c0115c0:	6109                	addi	sp,sp,128
1c0115c2:	d0bf706f          	j	1c0092cc <mm_dw_in_grad>
1c0115c6:	4605                	li	a2,1
1c0115c8:	b709                	j	1c0114ca <mm_dw_in_grad_unroll_1x2_u2+0x146>
	...

1c0115cc <mm_dw_in_grad_unroll_1x4_u2>:
1c0115cc:	01052283          	lw	t0,16(a0)
1c0115d0:	7115                	addi	sp,sp,-224
1c0115d2:	cfa2                	sw	s0,220(sp)
1c0115d4:	cda6                	sw	s1,216(sp)
1c0115d6:	cbca                	sw	s2,212(sp)
1c0115d8:	c9ce                	sw	s3,208(sp)
1c0115da:	c7d2                	sw	s4,204(sp)
1c0115dc:	c5d6                	sw	s5,200(sp)
1c0115de:	c3da                	sw	s6,196(sp)
1c0115e0:	c1de                	sw	s7,192(sp)
1c0115e2:	df62                	sw	s8,188(sp)
1c0115e4:	dd66                	sw	s9,184(sp)
1c0115e6:	db6a                	sw	s10,180(sp)
1c0115e8:	d96e                	sw	s11,176(sp)
1c0115ea:	c116                	sw	t0,128(sp)
1c0115ec:	478d                	li	a5,3
1c0115ee:	3e57df63          	ble	t0,a5,1c0119ec <mm_dw_in_grad_unroll_1x4_u2+0x420>
1c0115f2:	01852303          	lw	t1,24(a0)
1c0115f6:	4785                	li	a5,1
1c0115f8:	de9a                	sw	t1,124(sp)
1c0115fa:	4067d863          	ble	t1,a5,1c011a0a <mm_dw_in_grad_unroll_1x4_u2+0x43e>
1c0115fe:	494c                	lw	a1,20(a0)
1c011600:	f14026f3          	csrr	a3,mhartid
1c011604:	f456b6b3          	p.bclr	a3,a3,26,5
1c011608:	0265c633          	div	a2,a1,t1
1c01160c:	c32e                	sw	a1,132(sp)
1c01160e:	00c52803          	lw	a6,12(a0)
1c011612:	c742                	sw	a6,140(sp)
1c011614:	00760593          	addi	a1,a2,7
1c011618:	41f5d793          	srai	a5,a1,0x1f
1c01161c:	f837b7b3          	p.bclr	a5,a5,28,3
1c011620:	06b7a7db          	p.addn	a5,a5,a1,3
1c011624:	02f686b3          	mul	a3,a3,a5
1c011628:	97b6                	add	a5,a5,a3
1c01162a:	04c7c7b3          	p.min	a5,a5,a2
1c01162e:	d2b6                	sw	a3,100(sp)
1c011630:	c23e                	sw	a5,4(sp)
1c011632:	2b005663          	blez	a6,1c0118de <mm_dw_in_grad_unroll_1x4_u2+0x312>
1c011636:	5e76                	lw	t3,124(sp)
1c011638:	4f9a                	lw	t6,132(sp)
1c01163a:	4150                	lw	a2,4(a0)
1c01163c:	03c68eb3          	mul	t4,a3,t3
1c011640:	c202b8b3          	p.bclr	a7,t0,1,0
1c011644:	001f9313          	slli	t1,t6,0x1
1c011648:	41fe06b3          	sub	a3,t3,t6
1c01164c:	d4c6                	sw	a7,104(sp)
1c01164e:	01f308b3          	add	a7,t1,t6
1c011652:	8796                	mv	a5,t0
1c011654:	d8b6                	sw	a3,112(sp)
1c011656:	00289293          	slli	t0,a7,0x2
1c01165a:	405606b3          	sub	a3,a2,t0
1c01165e:	5716                	lw	a4,100(sp)
1c011660:	cb16                	sw	t0,148(sp)
1c011662:	52c6                	lw	t0,112(sp)
1c011664:	40000837          	lui	a6,0x40000
1c011668:	187d                	addi	a6,a6,-1
1c01166a:	5ff6                	lw	t6,124(sp)
1c01166c:	9872                	add	a6,a6,t3
1c01166e:	02f70e33          	mul	t3,a4,a5
1c011672:	40500733          	neg	a4,t0
1c011676:	429a                	lw	t0,132(sp)
1c011678:	002f9b93          	slli	s7,t6,0x2
1c01167c:	01df8f33          	add	t5,t6,t4
1c011680:	41f007b3          	neg	a5,t6
1c011684:	40500fb3          	neg	t6,t0
1c011688:	0f92                	slli	t6,t6,0x4
1c01168a:	dafe                	sw	t6,116(sp)
1c01168c:	4f9a                	lw	t6,132(sp)
1c01168e:	0f0a                	slli	t5,t5,0x2
1c011690:	9f36                	add	t5,t5,a3
1c011692:	56a6                	lw	a3,104(sp)
1c011694:	0f92                	slli	t6,t6,0x4
1c011696:	d0fe                	sw	t6,96(sp)
1c011698:	5ff6                	lw	t6,124(sp)
1c01169a:	42d28833          	p.mac	a6,t0,a3
1c01169e:	52f6                	lw	t0,124(sp)
1c0116a0:	41f30333          	sub	t1,t1,t6
1c0116a4:	450c                	lw	a1,8(a0)
1c0116a6:	cf1a                	sw	t1,156(sp)
1c0116a8:	4108                	lw	a0,0(a0)
1c0116aa:	5316                	lw	t1,100(sp)
1c0116ac:	c002bb33          	p.bclr	s6,t0,0,0
1c0116b0:	fc12b2b3          	p.bclr	t0,t0,30,1
1c0116b4:	c416                	sw	t0,8(sp)
1c0116b6:	428a                	lw	t0,128(sp)
1c0116b8:	43730533          	p.mac	a0,t1,s7
1c0116bc:	9876                	add	a6,a6,t4
1c0116be:	fa22b2b3          	p.bclr	t0,t0,29,2
1c0116c2:	c916                	sw	t0,144(sp)
1c0116c4:	429a                	lw	t0,132(sp)
1c0116c6:	4e9a                	lw	t4,132(sp)
1c0116c8:	96f2                	add	a3,a3,t3
1c0116ca:	028a                	slli	t0,t0,0x2
1c0116cc:	c696                	sw	t0,76(sp)
1c0116ce:	428a                	lw	t0,128(sp)
1c0116d0:	0e0a                	slli	t3,t3,0x2
1c0116d2:	9e2e                	add	t3,t3,a1
1c0116d4:	068a                	slli	a3,a3,0x2
1c0116d6:	0e8e                	slli	t4,t4,0x3
1c0116d8:	41f888b3          	sub	a7,a7,t6
1c0116dc:	028a                	slli	t0,t0,0x2
1c0116de:	96ae                	add	a3,a3,a1
1c0116e0:	070a                	slli	a4,a4,0x2
1c0116e2:	d376                	sw	t4,164(sp)
1c0116e4:	d172                	sw	t3,160(sp)
1c0116e6:	cd46                	sw	a7,152(sp)
1c0116e8:	c616                	sw	t0,12(sp)
1c0116ea:	d6aa                	sw	a0,108(sp)
1c0116ec:	d736                	sw	a3,172(sp)
1c0116ee:	078a                	slli	a5,a5,0x2
1c0116f0:	00470693          	addi	a3,a4,4
1c0116f4:	080a                	slli	a6,a6,0x2
1c0116f6:	0721                	addi	a4,a4,8
1c0116f8:	9832                	add	a6,a6,a2
1c0116fa:	1f71                	addi	t5,t5,-4
1c0116fc:	00478613          	addi	a2,a5,4
1c011700:	ff8b8593          	addi	a1,s7,-8
1c011704:	07a1                	addi	a5,a5,8
1c011706:	dc3a                	sw	a4,56(sp)
1c011708:	ffcb8713          	addi	a4,s7,-4
1c01170c:	d542                	sw	a6,168(sp)
1c01170e:	c502                	sw	zero,136(sp)
1c011710:	dcfa                	sw	t5,120(sp)
1c011712:	d22e                	sw	a1,36(sp)
1c011714:	d432                	sw	a2,40(sp)
1c011716:	da36                	sw	a3,52(sp)
1c011718:	d63e                	sw	a5,44(sp)
1c01171a:	d83a                	sw	a4,48(sp)
1c01171c:	5726                	lw	a4,104(sp)
1c01171e:	1a070763          	beqz	a4,1c0118cc <mm_dw_in_grad_unroll_1x4_u2+0x300>
1c011722:	5796                	lw	a5,100(sp)
1c011724:	4812                	lw	a6,4(sp)
1c011726:	1b07d363          	ble	a6,a5,1c0118cc <mm_dw_in_grad_unroll_1x4_u2+0x300>
1c01172a:	489a                	lw	a7,132(sp)
1c01172c:	5e76                	lw	t3,124(sp)
1c01172e:	4eea                	lw	t4,152(sp)
1c011730:	4ffa                	lw	t6,156(sp)
1c011732:	528a                	lw	t0,160(sp)
1c011734:	435a                	lw	t1,148(sp)
1c011736:	559a                	lw	a1,164(sp)
1c011738:	4636                	lw	a2,76(sp)
1c01173a:	41c888b3          	sub	a7,a7,t3
1c01173e:	cec6                	sw	a7,92(sp)
1c011740:	caf6                	sw	t4,84(sp)
1c011742:	ccfe                	sw	t6,88(sp)
1c011744:	c496                	sw	t0,72(sp)
1c011746:	d01a                	sw	t1,32(sp)
1c011748:	de2e                	sw	a1,60(sp)
1c01174a:	c0b2                	sw	a2,64(sp)
1c01174c:	c282                	sw	zero,68(sp)
1c01174e:	c882                	sw	zero,80(sp)
1c011750:	5e46                	lw	t3,112(sp)
1c011752:	4ef6                	lw	t4,92(sp)
1c011754:	4fe6                	lw	t6,88(sp)
1c011756:	42d6                	lw	t0,84(sp)
1c011758:	01de07b3          	add	a5,t3,t4
1c01175c:	40ff8733          	sub	a4,t6,a5
1c011760:	4326                	lw	t1,72(sp)
1c011762:	40f287b3          	sub	a5,t0,a5
1c011766:	55e6                	lw	a1,120(sp)
1c011768:	5602                	lw	a2,32(sp)
1c01176a:	070a                	slli	a4,a4,0x2
1c01176c:	078a                	slli	a5,a5,0x2
1c01176e:	59b6                	lw	s3,108(sp)
1c011770:	5a16                	lw	s4,100(sp)
1c011772:	00470693          	addi	a3,a4,4
1c011776:	00478813          	addi	a6,a5,4
1c01177a:	0721                	addi	a4,a4,8
1c01177c:	07a1                	addi	a5,a5,8
1c01177e:	00430913          	addi	s2,t1,4
1c011782:	00c584b3          	add	s1,a1,a2
1c011786:	8a9a                	mv	s5,t1
1c011788:	c836                	sw	a3,16(sp)
1c01178a:	ca42                	sw	a6,20(sp)
1c01178c:	cc3a                	sw	a4,24(sp)
1c01178e:	ce3e                	sw	a5,28(sp)
1c011790:	180b0763          	beqz	s6,1c01191e <mm_dw_in_grad_unroll_1x4_u2+0x352>
1c011794:	5792                	lw	a5,36(sp)
1c011796:	5822                	lw	a6,40(sp)
1c011798:	58d2                	lw	a7,52(sp)
1c01179a:	01378433          	add	s0,a5,s3
1c01179e:	57c2                	lw	a5,48(sp)
1c0117a0:	4e42                	lw	t3,16(sp)
1c0117a2:	4ed2                	lw	t4,20(sp)
1c0117a4:	5332                	lw	t1,44(sp)
1c0117a6:	55e2                	lw	a1,56(sp)
1c0117a8:	4662                	lw	a2,24(sp)
1c0117aa:	46f2                	lw	a3,28(sp)
1c0117ac:	009803b3          	add	t2,a6,s1
1c0117b0:	01378833          	add	a6,a5,s3
1c0117b4:	00000793          	li	a5,0
1c0117b8:	fffb0513          	addi	a0,s6,-1
1c0117bc:	8105                	srli	a0,a0,0x1
1c0117be:	001b0713          	addi	a4,s6,1
1c0117c2:	4c09                	li	s8,2
1c0117c4:	009882b3          	add	t0,a7,s1
1c0117c8:	009e0fb3          	add	t6,t3,s1
1c0117cc:	009e8f33          	add	t5,t4,s1
1c0117d0:	00958e33          	add	t3,a1,s1
1c0117d4:	00930eb3          	add	t4,t1,s1
1c0117d8:	009688b3          	add	a7,a3,s1
1c0117dc:	00960333          	add	t1,a2,s1
1c0117e0:	85be                	mv	a1,a5
1c0117e2:	863e                	mv	a2,a5
1c0117e4:	86be                	mv	a3,a5
1c0117e6:	0505                	addi	a0,a0,1
1c0117e8:	25876263          	bltu	a4,s8,1c011a2c <mm_dw_in_grad_unroll_1x4_u2+0x460>
1c0117ec:	024540fb          	lp.setup	x1,a0,1c011834 <mm_dw_in_grad_unroll_1x4_u2+0x268>
1c0117f0:	ff882c0b          	p.lw	s8,-8(a6!) # 3ffffff8 <__l2_shared_end+0x23fe7aec>
1c0117f4:	0083ad8b          	p.lw	s11,8(t2!)
1c0117f8:	0082ad0b          	p.lw	s10,8(t0!)
1c0117fc:	008fac8b          	p.lw	s9,8(t6!)
1c011800:	008f270b          	p.lw	a4,8(t5!)
1c011804:	698df6c3          	fmadd.s	a3,s11,s8,a3
1c011808:	618d7643          	fmadd.s	a2,s10,s8,a2
1c01180c:	598cf5c3          	fmadd.s	a1,s9,s8,a1
1c011810:	798777c3          	fmadd.s	a5,a4,s8,a5
1c011814:	008ead8b          	p.lw	s11,8(t4!)
1c011818:	ff84270b          	p.lw	a4,-8(s0!)
1c01181c:	008e2d0b          	p.lw	s10,8(t3!)
1c011820:	00832c8b          	p.lw	s9,8(t1!)
1c011824:	0088ac0b          	p.lw	s8,8(a7!)
1c011828:	68edf6c3          	fmadd.s	a3,s11,a4,a3
1c01182c:	60ed7643          	fmadd.s	a2,s10,a4,a2
1c011830:	58ecf5c3          	fmadd.s	a1,s9,a4,a1
1c011834:	78ec77c3          	fmadd.s	a5,s8,a4,a5
1c011838:	4822                	lw	a6,8(sp)
1c01183a:	0c080063          	beqz	a6,1c0118fa <mm_dw_in_grad_unroll_1x4_u2+0x32e>
1c01183e:	4896                	lw	a7,68(sp)
1c011840:	4e06                	lw	t3,64(sp)
1c011842:	5ef2                	lw	t4,60(sp)
1c011844:	5f82                	lw	t6,32(sp)
1c011846:	2179f50b          	p.lw	a0,s7(s3!)
1c01184a:	00988733          	add	a4,a7,s1
1c01184e:	2174f30b          	p.lw	t1,s7(s1!)
1c011852:	21c77883          	p.lw	a7,t3(a4)
1c011856:	21d77803          	p.lw	a6,t4(a4)
1c01185a:	21f77703          	p.lw	a4,t6(a4)
1c01185e:	68a376c3          	fmadd.s	a3,t1,a0,a3
1c011862:	60a8f643          	fmadd.s	a2,a7,a0,a2
1c011866:	58a875c3          	fmadd.s	a1,a6,a0,a1
1c01186a:	78a777c3          	fmadd.s	a5,a4,a0,a5
1c01186e:	42b2                	lw	t0,12(sp)
1c011870:	4312                	lw	t1,4(sp)
1c011872:	0a05                	addi	s4,s4,1
1c011874:	00dae2ab          	p.sw	a3,t0(s5!)
1c011878:	00c92023          	sw	a2,0(s2)
1c01187c:	00b92223          	sw	a1,4(s2)
1c011880:	00f92423          	sw	a5,8(s2)
1c011884:	9916                	add	s2,s2,t0
1c011886:	f14315e3          	bne	t1,s4,1c011790 <mm_dw_in_grad_unroll_1x4_u2+0x1c4>
1c01188a:	4896                	lw	a7,68(sp)
1c01188c:	5e56                	lw	t3,116(sp)
1c01188e:	5f86                	lw	t6,96(sp)
1c011890:	46b6                	lw	a3,76(sp)
1c011892:	4846                	lw	a6,80(sp)
1c011894:	4e86                	lw	t4,64(sp)
1c011896:	52f2                	lw	t0,60(sp)
1c011898:	5302                	lw	t1,32(sp)
1c01189a:	45a6                	lw	a1,72(sp)
1c01189c:	4676                	lw	a2,92(sp)
1c01189e:	4766                	lw	a4,88(sp)
1c0118a0:	47d6                	lw	a5,84(sp)
1c0118a2:	98f2                	add	a7,a7,t3
1c0118a4:	c2c6                	sw	a7,68(sp)
1c0118a6:	58a6                	lw	a7,104(sp)
1c0118a8:	0811                	addi	a6,a6,4
1c0118aa:	9efe                	add	t4,t4,t6
1c0118ac:	92fe                	add	t0,t0,t6
1c0118ae:	937e                	add	t1,t1,t6
1c0118b0:	05c1                	addi	a1,a1,16
1c0118b2:	9636                	add	a2,a2,a3
1c0118b4:	9736                	add	a4,a4,a3
1c0118b6:	97b6                	add	a5,a5,a3
1c0118b8:	c8c2                	sw	a6,80(sp)
1c0118ba:	c0f6                	sw	t4,64(sp)
1c0118bc:	de16                	sw	t0,60(sp)
1c0118be:	d01a                	sw	t1,32(sp)
1c0118c0:	c4ae                	sw	a1,72(sp)
1c0118c2:	ceb2                	sw	a2,92(sp)
1c0118c4:	ccba                	sw	a4,88(sp)
1c0118c6:	cabe                	sw	a5,84(sp)
1c0118c8:	e91864e3          	bltu	a6,a7,1c011750 <mm_dw_in_grad_unroll_1x4_u2+0x184>
1c0118cc:	48ca                	lw	a7,144(sp)
1c0118ce:	04089e63          	bnez	a7,1c01192a <mm_dw_in_grad_unroll_1x4_u2+0x35e>
1c0118d2:	462a                	lw	a2,136(sp)
1c0118d4:	46ba                	lw	a3,140(sp)
1c0118d6:	0605                	addi	a2,a2,1
1c0118d8:	c532                	sw	a2,136(sp)
1c0118da:	e4c691e3          	bne	a3,a2,1c01171c <mm_dw_in_grad_unroll_1x4_u2+0x150>
1c0118de:	447e                	lw	s0,220(sp)
1c0118e0:	44ee                	lw	s1,216(sp)
1c0118e2:	495e                	lw	s2,212(sp)
1c0118e4:	49ce                	lw	s3,208(sp)
1c0118e6:	4a3e                	lw	s4,204(sp)
1c0118e8:	4aae                	lw	s5,200(sp)
1c0118ea:	4b1e                	lw	s6,196(sp)
1c0118ec:	4b8e                	lw	s7,192(sp)
1c0118ee:	5c7a                	lw	s8,188(sp)
1c0118f0:	5cea                	lw	s9,184(sp)
1c0118f2:	5d5a                	lw	s10,180(sp)
1c0118f4:	5dca                	lw	s11,176(sp)
1c0118f6:	612d                	addi	sp,sp,224
1c0118f8:	8082                	ret
1c0118fa:	4732                	lw	a4,12(sp)
1c0118fc:	0a05                	addi	s4,s4,1
1c0118fe:	99de                	add	s3,s3,s7
1c011900:	00dae72b          	p.sw	a3,a4(s5!)
1c011904:	00f92423          	sw	a5,8(s2)
1c011908:	4792                	lw	a5,4(sp)
1c01190a:	00c92023          	sw	a2,0(s2)
1c01190e:	00b92223          	sw	a1,4(s2)
1c011912:	94de                	add	s1,s1,s7
1c011914:	993a                	add	s2,s2,a4
1c011916:	f7478ae3          	beq	a5,s4,1c01188a <mm_dw_in_grad_unroll_1x4_u2+0x2be>
1c01191a:	e60b1de3          	bnez	s6,1c011794 <mm_dw_in_grad_unroll_1x4_u2+0x1c8>
1c01191e:	00000793          	li	a5,0
1c011922:	85be                	mv	a1,a5
1c011924:	863e                	mv	a2,a5
1c011926:	86be                	mv	a3,a5
1c011928:	bf01                	j	1c011838 <mm_dw_in_grad_unroll_1x4_u2+0x26c>
1c01192a:	4e0a                	lw	t3,128(sp)
1c01192c:	5ea6                	lw	t4,104(sp)
1c01192e:	fbced2e3          	ble	t3,t4,1c0118d2 <mm_dw_in_grad_unroll_1x4_u2+0x306>
1c011932:	5f96                	lw	t6,100(sp)
1c011934:	4292                	lw	t0,4(sp)
1c011936:	f85fdee3          	ble	t0,t6,1c0118d2 <mm_dw_in_grad_unroll_1x4_u2+0x306>
1c01193a:	542a                	lw	s0,168(sp)
1c01193c:	53ba                	lw	t2,172(sp)
1c01193e:	4922                	lw	s2,8(sp)
1c011940:	5d36                	lw	s10,108(sp)
1c011942:	4db2                	lw	s11,12(sp)
1c011944:	5a12                	lw	s4,36(sp)
1c011946:	5aa2                	lw	s5,40(sp)
1c011948:	5c32                	lw	s8,44(sp)
1c01194a:	5cc2                	lw	s9,48(sp)
1c01194c:	84f6                	mv	s1,t4
1c01194e:	8996                	mv	s3,t0
1c011950:	5f96                	lw	t6,100(sp)
1c011952:	8f22                	mv	t5,s0
1c011954:	8eea                	mv	t4,s10
1c011956:	829e                	mv	t0,t2
1c011958:	00000793          	li	a5,0
1c01195c:	040b0463          	beqz	s6,1c0119a4 <mm_dw_in_grad_unroll_1x4_u2+0x3d8>
1c011960:	fffb0693          	addi	a3,s6,-1
1c011964:	8285                	srli	a3,a3,0x1
1c011966:	001b0713          	addi	a4,s6,1
1c01196a:	4889                	li	a7,2
1c01196c:	01da0633          	add	a2,s4,t4
1c011970:	01ea8833          	add	a6,s5,t5
1c011974:	01ec0533          	add	a0,s8,t5
1c011978:	01dc85b3          	add	a1,s9,t4
1c01197c:	00000793          	li	a5,0
1c011980:	0685                	addi	a3,a3,1
1c011982:	0b176363          	bltu	a4,a7,1c011a28 <mm_dw_in_grad_unroll_1x4_u2+0x45c>
1c011986:	0001                	nop
1c011988:	00c6c0fb          	lp.setup	x1,a3,1c0119a0 <mm_dw_in_grad_unroll_1x4_u2+0x3d4>
1c01198c:	ff85ae0b          	p.lw	t3,-8(a1!)
1c011990:	0088230b          	p.lw	t1,8(a6!)
1c011994:	ff86288b          	p.lw	a7,-8(a2!)
1c011998:	0085270b          	p.lw	a4,8(a0!)
1c01199c:	786e77c3          	fmadd.s	a5,t3,t1,a5
1c0119a0:	78e8f7c3          	fmadd.s	a5,a7,a4,a5
1c0119a4:	02090563          	beqz	s2,1c0119ce <mm_dw_in_grad_unroll_1x4_u2+0x402>
1c0119a8:	217ef70b          	p.lw	a4,s7(t4!)
1c0119ac:	217f768b          	p.lw	a3,s7(t5!)
1c0119b0:	0f85                	addi	t6,t6,1
1c0119b2:	78d777c3          	fmadd.s	a5,a4,a3,a5
1c0119b6:	00f2edab          	p.sw	a5,s11(t0!)
1c0119ba:	f9f99fe3          	bne	s3,t6,1c011958 <mm_dw_in_grad_unroll_1x4_u2+0x38c>
1c0119be:	4336                	lw	t1,76(sp)
1c0119c0:	458a                	lw	a1,128(sp)
1c0119c2:	0485                	addi	s1,s1,1
1c0119c4:	0391                	addi	t2,t2,4
1c0119c6:	941a                	add	s0,s0,t1
1c0119c8:	f89594e3          	bne	a1,s1,1c011950 <mm_dw_in_grad_unroll_1x4_u2+0x384>
1c0119cc:	b719                	j	1c0118d2 <mm_dw_in_grad_unroll_1x4_u2+0x306>
1c0119ce:	00f2edab          	p.sw	a5,s11(t0!)
1c0119d2:	0f85                	addi	t6,t6,1
1c0119d4:	9ede                	add	t4,t4,s7
1c0119d6:	9f5e                	add	t5,t5,s7
1c0119d8:	f9f990e3          	bne	s3,t6,1c011958 <mm_dw_in_grad_unroll_1x4_u2+0x38c>
1c0119dc:	4336                	lw	t1,76(sp)
1c0119de:	458a                	lw	a1,128(sp)
1c0119e0:	0485                	addi	s1,s1,1
1c0119e2:	0391                	addi	t2,t2,4
1c0119e4:	941a                	add	s0,s0,t1
1c0119e6:	f69595e3          	bne	a1,s1,1c011950 <mm_dw_in_grad_unroll_1x4_u2+0x384>
1c0119ea:	b5e5                	j	1c0118d2 <mm_dw_in_grad_unroll_1x4_u2+0x306>
1c0119ec:	447e                	lw	s0,220(sp)
1c0119ee:	44ee                	lw	s1,216(sp)
1c0119f0:	495e                	lw	s2,212(sp)
1c0119f2:	49ce                	lw	s3,208(sp)
1c0119f4:	4a3e                	lw	s4,204(sp)
1c0119f6:	4aae                	lw	s5,200(sp)
1c0119f8:	4b1e                	lw	s6,196(sp)
1c0119fa:	4b8e                	lw	s7,192(sp)
1c0119fc:	5c7a                	lw	s8,188(sp)
1c0119fe:	5cea                	lw	s9,184(sp)
1c011a00:	5d5a                	lw	s10,180(sp)
1c011a02:	5dca                	lw	s11,176(sp)
1c011a04:	612d                	addi	sp,sp,224
1c011a06:	d66ff06f          	j	1c010f6c <mm_dw_in_grad_unroll_1x2>
1c011a0a:	447e                	lw	s0,220(sp)
1c011a0c:	44ee                	lw	s1,216(sp)
1c011a0e:	495e                	lw	s2,212(sp)
1c011a10:	49ce                	lw	s3,208(sp)
1c011a12:	4a3e                	lw	s4,204(sp)
1c011a14:	4aae                	lw	s5,200(sp)
1c011a16:	4b1e                	lw	s6,196(sp)
1c011a18:	4b8e                	lw	s7,192(sp)
1c011a1a:	5c7a                	lw	s8,188(sp)
1c011a1c:	5cea                	lw	s9,184(sp)
1c011a1e:	5d5a                	lw	s10,180(sp)
1c011a20:	5dca                	lw	s11,176(sp)
1c011a22:	612d                	addi	sp,sp,224
1c011a24:	eb4ff06f          	j	1c0110d8 <mm_dw_in_grad_unroll_1x4>
1c011a28:	4685                	li	a3,1
1c011a2a:	bfb1                	j	1c011986 <mm_dw_in_grad_unroll_1x4_u2+0x3ba>
1c011a2c:	4505                	li	a0,1
1c011a2e:	bb7d                	j	1c0117ec <mm_dw_in_grad_unroll_1x4_u2+0x220>
1c011a30:	0000                	unimp
	...

1c011a34 <pulp_im2col_fp32>:
1c011a34:	415c                	lw	a5,4(a0)
1c011a36:	00052883          	lw	a7,0(a0)
1c011a3a:	7159                	addi	sp,sp,-112
1c011a3c:	4b98                	lw	a4,16(a5)
1c011a3e:	47d0                	lw	a2,12(a5)
1c011a40:	02454783          	lbu	a5,36(a0)
1c011a44:	c63a                	sw	a4,12(sp)
1c011a46:	490c                	lw	a1,16(a0)
1c011a48:	4e32                	lw	t3,12(sp)
1c011a4a:	ced2                	sw	s4,92(sp)
1c011a4c:	0108aa03          	lw	s4,16(a7)
1c011a50:	c83e                	sw	a5,16(sp)
1c011a52:	00c8af03          	lw	t5,12(a7)
1c011a56:	ee85beb3          	p.bclr	t4,a1,23,8
1c011a5a:	4fc2                	lw	t6,16(sp)
1c011a5c:	01454703          	lbu	a4,20(a0)
1c011a60:	41ca0333          	sub	t1,s4,t3
1c011a64:	001e9693          	slli	a3,t4,0x1
1c011a68:	00d303b3          	add	t2,t1,a3
1c011a6c:	d6a2                	sw	s0,108(sp)
1c011a6e:	d4a6                	sw	s1,104(sp)
1c011a70:	d2ca                	sw	s2,100(sp)
1c011a72:	d0ce                	sw	s3,96(sp)
1c011a74:	ccd6                	sw	s5,88(sp)
1c011a76:	cada                	sw	s6,84(sp)
1c011a78:	c8de                	sw	s7,80(sp)
1c011a7a:	c6e2                	sw	s8,76(sp)
1c011a7c:	c4e6                	sw	s9,72(sp)
1c011a7e:	c2ea                	sw	s10,68(sp)
1c011a80:	c0ee                	sw	s11,64(sp)
1c011a82:	00c52803          	lw	a6,12(a0)
1c011a86:	02052e03          	lw	t3,32(a0)
1c011a8a:	40cf07b3          	sub	a5,t5,a2
1c011a8e:	93fe                	add	t2,t2,t6
1c011a90:	20071c63          	bnez	a4,1c011ca8 <pulp_im2col_fp32+0x274>
1c011a94:	03f3ffb3          	remu	t6,t2,t6
1c011a98:	0148a503          	lw	a0,20(a7)
1c011a9c:	f14022f3          	csrr	t0,mhartid
1c011aa0:	f452b2b3          	p.bclr	t0,t0,26,5
1c011aa4:	00750713          	addi	a4,a0,7
1c011aa8:	830d                	srli	a4,a4,0x3
1c011aaa:	02e28c33          	mul	s8,t0,a4
1c011aae:	9762                	add	a4,a4,s8
1c011ab0:	04a759b3          	p.minu	s3,a4,a0
1c011ab4:	320f9b63          	bnez	t6,1c011dea <pulp_im2col_fp32+0x3b6>
1c011ab8:	4fc2                	lw	t6,16(sp)
1c011aba:	ee8e3e33          	p.bclr	t3,t3,23,8
1c011abe:	00d78733          	add	a4,a5,a3
1c011ac2:	9772                	add	a4,a4,t3
1c011ac4:	03f3d3b3          	divu	t2,t2,t6
1c011ac8:	ca72                	sw	t3,20(sp)
1c011aca:	03c77e33          	remu	t3,a4,t3
1c011ace:	d61e                	sw	t2,44(sp)
1c011ad0:	340e1d63          	bnez	t3,1c011e2a <pulp_im2col_fp32+0x3f6>
1c011ad4:	4e52                	lw	t3,20(sp)
1c011ad6:	03c75733          	divu	a4,a4,t3
1c011ada:	d83a                	sw	a4,48(sp)
1c011adc:	10038563          	beqz	t2,1c011be6 <pulp_im2col_fp32+0x1b2>
1c011ae0:	0785                	addi	a5,a5,1
1c011ae2:	10070263          	beqz	a4,1c011be6 <pulp_im2col_fp32+0x1b2>
1c011ae6:	113c5063          	ble	s3,s8,1c011be6 <pulp_im2col_fp32+0x1b2>
1c011aea:	4732                	lw	a4,12(sp)
1c011aec:	cf6d                	beqz	a4,1c011be6 <pulp_im2col_fp32+0x1b2>
1c011aee:	96be                	add	a3,a3,a5
1c011af0:	034c07b3          	mul	a5,s8,s4
1c011af4:	41d68bb3          	sub	s7,a3,t4
1c011af8:	001e8293          	addi	t0,t4,1
1c011afc:	00261f93          	slli	t6,a2,0x2
1c011b00:	0ff5f593          	andi	a1,a1,255
1c011b04:	8b3a                	mv	s6,a4
1c011b06:	cc02                	sw	zero,24(sp)
1c011b08:	929a                	add	t0,t0,t1
1c011b0a:	de62                	sw	s8,60(sp)
1c011b0c:	02e60ab3          	mul	s5,a2,a4
1c011b10:	41d787b3          	sub	a5,a5,t4
1c011b14:	03e787b3          	mul	a5,a5,t5
1c011b18:	0a8a                	slli	s5,s5,0x2
1c011b1a:	03550533          	mul	a0,a0,s5
1c011b1e:	41d787b3          	sub	a5,a5,t4
1c011b22:	d23e                	sw	a5,36(sp)
1c011b24:	47c2                	lw	a5,16(sp)
1c011b26:	02d506b3          	mul	a3,a0,a3
1c011b2a:	d42a                	sw	a0,40(sp)
1c011b2c:	435c0833          	p.mac	a6,s8,s5
1c011b30:	da36                	sw	a3,52(sp)
1c011b32:	03e787b3          	mul	a5,a5,t5
1c011b36:	d042                	sw	a6,32(sp)
1c011b38:	034f0a33          	mul	s4,t5,s4
1c011b3c:	dc3e                	sw	a5,56(sp)
1c011b3e:	4732                	lw	a4,12(sp)
1c011b40:	5782                	lw	a5,32(sp)
1c011b42:	4e01                	li	t3,0
1c011b44:	40eb0933          	sub	s2,s6,a4
1c011b48:	ce3e                	sw	a5,28(sp)
1c011b4a:	4c01                	li	s8,0
1c011b4c:	5712                	lw	a4,36(sp)
1c011b4e:	4472                	lw	s0,28(sp)
1c011b50:	53f2                	lw	t2,60(sp)
1c011b52:	00ee04b3          	add	s1,t3,a4
1c011b56:	834a                	mv	t1,s2
1c011b58:	8d22                	mv	s10,s0
1c011b5a:	8726                	mv	a4,s1
1c011b5c:	412b0cb3          	sub	s9,s6,s2
1c011b60:	ce0d                	beqz	a2,1c011b9a <pulp_im2col_fp32+0x166>
1c011b62:	886a                	mv	a6,s10
1c011b64:	0a62e263          	bltu	t0,t1,1c011c08 <pulp_im2col_fp32+0x1d4>
1c011b68:	11d34563          	blt	t1,t4,1c011c72 <pulp_im2col_fp32+0x23e>
1c011b6c:	4681                	li	a3,0
1c011b6e:	8532                	mv	a0,a2
1c011b70:	014540fb          	lp.setup	x1,a0,1c011b98 <pulp_im2col_fp32+0x164>
1c011b74:	00e687b3          	add	a5,a3,a4
1c011b78:	00279d93          	slli	s11,a5,0x2
1c011b7c:	01c687b3          	add	a5,a3,t3
1c011b80:	c589                	beqz	a1,1c011b8a <pulp_im2col_fp32+0x156>
1c011b82:	09d7c063          	blt	a5,t4,1c011c02 <pulp_im2col_fp32+0x1ce>
1c011b86:	06fbee63          	bltu	s7,a5,1c011c02 <pulp_im2col_fp32+0x1ce>
1c011b8a:	0008a783          	lw	a5,0(a7)
1c011b8e:	21b7f783          	p.lw	a5,s11(a5)
1c011b92:	00f82023          	sw	a5,0(a6)
1c011b96:	0685                	addi	a3,a3,1
1c011b98:	0811                	addi	a6,a6,4
1c011b9a:	977a                	add	a4,a4,t5
1c011b9c:	9d7e                	add	s10,s10,t6
1c011b9e:	0305                	addi	t1,t1,1
1c011ba0:	1cfd                	addi	s9,s9,-1
1c011ba2:	fa0c9fe3          	bnez	s9,1c011b60 <pulp_im2col_fp32+0x12c>
1c011ba6:	0385                	addi	t2,t2,1
1c011ba8:	94d2                	add	s1,s1,s4
1c011baa:	9456                	add	s0,s0,s5
1c011bac:	fa7995e3          	bne	s3,t2,1c011b56 <pulp_im2col_fp32+0x122>
1c011bb0:	4752                	lw	a4,20(sp)
1c011bb2:	47f2                	lw	a5,28(sp)
1c011bb4:	0c05                	addi	s8,s8,1
1c011bb6:	9e3a                	add	t3,t3,a4
1c011bb8:	5722                	lw	a4,40(sp)
1c011bba:	97ba                	add	a5,a5,a4
1c011bbc:	ce3e                	sw	a5,28(sp)
1c011bbe:	57c2                	lw	a5,48(sp)
1c011bc0:	f8fc16e3          	bne	s8,a5,1c011b4c <pulp_im2col_fp32+0x118>
1c011bc4:	47e2                	lw	a5,24(sp)
1c011bc6:	5702                	lw	a4,32(sp)
1c011bc8:	4e42                	lw	t3,16(sp)
1c011bca:	0785                	addi	a5,a5,1
1c011bcc:	cc3e                	sw	a5,24(sp)
1c011bce:	57d2                	lw	a5,52(sp)
1c011bd0:	9b72                	add	s6,s6,t3
1c011bd2:	5e12                	lw	t3,36(sp)
1c011bd4:	973e                	add	a4,a4,a5
1c011bd6:	d03a                	sw	a4,32(sp)
1c011bd8:	5762                	lw	a4,56(sp)
1c011bda:	47e2                	lw	a5,24(sp)
1c011bdc:	9e3a                	add	t3,t3,a4
1c011bde:	d272                	sw	t3,36(sp)
1c011be0:	5e32                	lw	t3,44(sp)
1c011be2:	f5c79ee3          	bne	a5,t3,1c011b3e <pulp_im2col_fp32+0x10a>
1c011be6:	5436                	lw	s0,108(sp)
1c011be8:	54a6                	lw	s1,104(sp)
1c011bea:	5916                	lw	s2,100(sp)
1c011bec:	5986                	lw	s3,96(sp)
1c011bee:	4a76                	lw	s4,92(sp)
1c011bf0:	4ae6                	lw	s5,88(sp)
1c011bf2:	4b56                	lw	s6,84(sp)
1c011bf4:	4bc6                	lw	s7,80(sp)
1c011bf6:	4c36                	lw	s8,76(sp)
1c011bf8:	4ca6                	lw	s9,72(sp)
1c011bfa:	4d16                	lw	s10,68(sp)
1c011bfc:	4d86                	lw	s11,64(sp)
1c011bfe:	6165                	addi	sp,sp,112
1c011c00:	8082                	ret
1c011c02:	00082023          	sw	zero,0(a6)
1c011c06:	bf41                	j	1c011b96 <pulp_im2col_fp32+0x162>
1c011c08:	4501                	li	a0,0
1c011c0a:	86b2                	mv	a3,a2
1c011c0c:	03d34363          	blt	t1,t4,1c011c32 <pulp_im2col_fp32+0x1fe>
1c011c10:	0096c0fb          	lp.setup	x1,a3,1c011c22 <pulp_im2col_fp32+0x1ee>
1c011c14:	00e507b3          	add	a5,a0,a4
1c011c18:	078a                	slli	a5,a5,0x2
1c011c1a:	cd95                	beqz	a1,1c011c56 <pulp_im2col_fp32+0x222>
1c011c1c:	00082023          	sw	zero,0(a6)
1c011c20:	0505                	addi	a0,a0,1
1c011c22:	0811                	addi	a6,a6,4
1c011c24:	977a                	add	a4,a4,t5
1c011c26:	9d7e                	add	s10,s10,t6
1c011c28:	0305                	addi	t1,t1,1
1c011c2a:	1cfd                	addi	s9,s9,-1
1c011c2c:	f20c9ae3          	bnez	s9,1c011b60 <pulp_im2col_fp32+0x12c>
1c011c30:	bf9d                	j	1c011ba6 <pulp_im2col_fp32+0x172>
1c011c32:	0001                	nop
1c011c34:	0096c0fb          	lp.setup	x1,a3,1c011c46 <pulp_im2col_fp32+0x212>
1c011c38:	00e507b3          	add	a5,a0,a4
1c011c3c:	078a                	slli	a5,a5,0x2
1c011c3e:	c19d                	beqz	a1,1c011c64 <pulp_im2col_fp32+0x230>
1c011c40:	00082023          	sw	zero,0(a6)
1c011c44:	0505                	addi	a0,a0,1
1c011c46:	0811                	addi	a6,a6,4
1c011c48:	977a                	add	a4,a4,t5
1c011c4a:	9d7e                	add	s10,s10,t6
1c011c4c:	0305                	addi	t1,t1,1
1c011c4e:	1cfd                	addi	s9,s9,-1
1c011c50:	f00c98e3          	bnez	s9,1c011b60 <pulp_im2col_fp32+0x12c>
1c011c54:	bf89                	j	1c011ba6 <pulp_im2col_fp32+0x172>
1c011c56:	0008ad83          	lw	s11,0(a7)
1c011c5a:	20fdf783          	p.lw	a5,a5(s11)
1c011c5e:	00f82023          	sw	a5,0(a6)
1c011c62:	bf7d                	j	1c011c20 <pulp_im2col_fp32+0x1ec>
1c011c64:	0008ad83          	lw	s11,0(a7)
1c011c68:	20fdf783          	p.lw	a5,a5(s11)
1c011c6c:	00f82023          	sw	a5,0(a6)
1c011c70:	bfd1                	j	1c011c44 <pulp_im2col_fp32+0x210>
1c011c72:	4501                	li	a0,0
1c011c74:	86b2                	mv	a3,a2
1c011c76:	0001                	nop
1c011c78:	0096c0fb          	lp.setup	x1,a3,1c011c8a <pulp_im2col_fp32+0x256>
1c011c7c:	00e507b3          	add	a5,a0,a4
1c011c80:	078a                	slli	a5,a5,0x2
1c011c82:	cd81                	beqz	a1,1c011c9a <pulp_im2col_fp32+0x266>
1c011c84:	00082023          	sw	zero,0(a6)
1c011c88:	0505                	addi	a0,a0,1
1c011c8a:	0811                	addi	a6,a6,4
1c011c8c:	977a                	add	a4,a4,t5
1c011c8e:	9d7e                	add	s10,s10,t6
1c011c90:	0305                	addi	t1,t1,1
1c011c92:	1cfd                	addi	s9,s9,-1
1c011c94:	ec0c96e3          	bnez	s9,1c011b60 <pulp_im2col_fp32+0x12c>
1c011c98:	b739                	j	1c011ba6 <pulp_im2col_fp32+0x172>
1c011c9a:	0008ad83          	lw	s11,0(a7)
1c011c9e:	20fdf783          	p.lw	a5,a5(s11)
1c011ca2:	00f82023          	sw	a5,0(a6)
1c011ca6:	b7cd                	j	1c011c88 <pulp_im2col_fp32+0x254>
1c011ca8:	00852b83          	lw	s7,8(a0)
1c011cac:	014ba583          	lw	a1,20(s7)
1c011cb0:	16172063          	p.beqimm	a4,1,1c011e10 <pulp_im2col_fp32+0x3dc>
1c011cb4:	97b6                	add	a5,a5,a3
1c011cb6:	ee8e3e33          	p.bclr	t3,t3,23,8
1c011cba:	01c782b3          	add	t0,a5,t3
1c011cbe:	f20a04e3          	beqz	s4,1c011be6 <pulp_im2col_fp32+0x1b2>
1c011cc2:	f20f02e3          	beqz	t5,1c011be6 <pulp_im2col_fp32+0x1b2>
1c011cc6:	f35b50e3          	ble	s5,s6,1c011be6 <pulp_im2col_fp32+0x1b2>
1c011cca:	47b2                	lw	a5,12(sp)
1c011ccc:	df89                	beqz	a5,1c011be6 <pulp_im2col_fp32+0x1b2>
1c011cce:	8e3e                	mv	t3,a5
1c011cd0:	02cf07b3          	mul	a5,t5,a2
1c011cd4:	4985                	li	s3,1
1c011cd6:	41c989b3          	sub	s3,s3,t3
1c011cda:	8dce                	mv	s11,s3
1c011cdc:	0f05                	addi	t5,t5,1
1c011cde:	0a05                	addi	s4,s4,1
1c011ce0:	40cf0f33          	sub	t5,t5,a2
1c011ce4:	ce52                	sw	s4,28(sp)
1c011ce6:	4d01                	li	s10,0
1c011ce8:	4a05                	li	s4,1
1c011cea:	03c60c33          	mul	s8,a2,t3
1c011cee:	ca7a                	sw	t5,20(sp)
1c011cf0:	03c787b3          	mul	a5,a5,t3
1c011cf4:	43638db3          	p.mac	s11,t2,s6
1c011cf8:	02b787b3          	mul	a5,a5,a1
1c011cfc:	036c0eb3          	mul	t4,s8,s6
1c011d00:	cc3e                	sw	a5,24(sp)
1c011d02:	038585b3          	mul	a1,a1,s8
1c011d06:	c676                	sw	t4,12(sp)
1c011d08:	025d8db3          	mul	s11,s11,t0
1c011d0c:	c82e                	sw	a1,16(sp)
1c011d0e:	02538cb3          	mul	s9,t2,t0
1c011d12:	4705                	li	a4,1
1c011d14:	40c70fb3          	sub	t6,a4,a2
1c011d18:	85ea                	mv	a1,s10
1c011d1a:	44b2                	lw	s1,12(sp)
1c011d1c:	01bf8933          	add	s2,t6,s11
1c011d20:	845a                	mv	s0,s6
1c011d22:	88ce                	mv	a7,s3
1c011d24:	8e4a                	mv	t3,s2
1c011d26:	87a6                	mv	a5,s1
1c011d28:	413a0333          	sub	t1,s4,s3
1c011d2c:	0253407b          	lp.setup	x0,t1,1c011d76 <pulp_im2col_fp32+0x342>
1c011d30:	86be                	mv	a3,a5
1c011d32:	c221                	beqz	a2,1c011d72 <pulp_im2col_fp32+0x33e>
1c011d34:	0608cc63          	bltz	a7,1c011dac <pulp_im2col_fp32+0x378>
1c011d38:	0878f963          	bleu	t2,a7,1c011dca <pulp_im2col_fp32+0x396>
1c011d3c:	4681                	li	a3,0
1c011d3e:	8532                	mv	a0,a2
1c011d40:	00f68733          	add	a4,a3,a5
1c011d44:	972e                	add	a4,a4,a1
1c011d46:	070a                	slli	a4,a4,0x2
1c011d48:	01f68eb3          	add	t4,a3,t6
1c011d4c:	9742                	add	a4,a4,a6
1c011d4e:	040ec963          	bltz	t4,1c011da0 <pulp_im2col_fp32+0x36c>
1c011d52:	01c68f33          	add	t5,a3,t3
1c011d56:	0f0a                	slli	t5,t5,0x2
1c011d58:	045ef463          	bleu	t0,t4,1c011da0 <pulp_im2col_fp32+0x36c>
1c011d5c:	004bae83          	lw	t4,4(s7)
1c011d60:	0685                	addi	a3,a3,1
1c011d62:	21eeff03          	p.lw	t5,t5(t4)
1c011d66:	01e72023          	sw	t5,0(a4)
1c011d6a:	157d                	addi	a0,a0,-1
1c011d6c:	f971                	bnez	a0,1c011d40 <pulp_im2col_fp32+0x30c>
1c011d6e:	00f606b3          	add	a3,a2,a5
1c011d72:	87b6                	mv	a5,a3
1c011d74:	9e16                	add	t3,t3,t0
1c011d76:	0885                	addi	a7,a7,1
1c011d78:	0405                	addi	s0,s0,1
1c011d7a:	94e2                	add	s1,s1,s8
1c011d7c:	9966                	add	s2,s2,s9
1c011d7e:	fb5412e3          	bne	s0,s5,1c011d22 <pulp_im2col_fp32+0x2ee>
1c011d82:	4742                	lw	a4,16(sp)
1c011d84:	47d2                	lw	a5,20(sp)
1c011d86:	0f85                	addi	t6,t6,1
1c011d88:	95ba                	add	a1,a1,a4
1c011d8a:	f8ff98e3          	bne	t6,a5,1c011d1a <pulp_im2col_fp32+0x2e6>
1c011d8e:	4e62                	lw	t3,24(sp)
1c011d90:	4ef2                	lw	t4,28(sp)
1c011d92:	0a05                	addi	s4,s4,1
1c011d94:	0985                	addi	s3,s3,1
1c011d96:	9d72                	add	s10,s10,t3
1c011d98:	9d96                	add	s11,s11,t0
1c011d9a:	f7da1ce3          	bne	s4,t4,1c011d12 <pulp_im2col_fp32+0x2de>
1c011d9e:	b5a1                	j	1c011be6 <pulp_im2col_fp32+0x1b2>
1c011da0:	00072023          	sw	zero,0(a4)
1c011da4:	0685                	addi	a3,a3,1
1c011da6:	157d                	addi	a0,a0,-1
1c011da8:	fd41                	bnez	a0,1c011d40 <pulp_im2col_fp32+0x30c>
1c011daa:	b7d1                	j	1c011d6e <pulp_im2col_fp32+0x33a>
1c011dac:	00f606b3          	add	a3,a2,a5
1c011db0:	40f68533          	sub	a0,a3,a5
1c011db4:	009540fb          	lp.setup	x1,a0,1c011dc6 <pulp_im2col_fp32+0x392>
1c011db8:	00000e93          	li	t4,0
1c011dbc:	00b78733          	add	a4,a5,a1
1c011dc0:	070a                	slli	a4,a4,0x2
1c011dc2:	01d86723          	p.sw	t4,a4(a6)
1c011dc6:	0785                	addi	a5,a5,1
1c011dc8:	b76d                	j	1c011d72 <pulp_im2col_fp32+0x33e>
1c011dca:	00f606b3          	add	a3,a2,a5
1c011dce:	40f68533          	sub	a0,a3,a5
1c011dd2:	0001                	nop
1c011dd4:	009540fb          	lp.setup	x1,a0,1c011de6 <pulp_im2col_fp32+0x3b2>
1c011dd8:	00000e93          	li	t4,0
1c011ddc:	00b78733          	add	a4,a5,a1
1c011de0:	070a                	slli	a4,a4,0x2
1c011de2:	01d86723          	p.sw	t4,a4(a6)
1c011de6:	0785                	addi	a5,a5,1
1c011de8:	b769                	j	1c011d72 <pulp_im2col_fp32+0x33e>
1c011dea:	1c001537          	lui	a0,0x1c001
1c011dee:	b8450513          	addi	a0,a0,-1148 # 1c000b84 <__DTOR_END__+0x31c>
1c011df2:	5436                	lw	s0,108(sp)
1c011df4:	54a6                	lw	s1,104(sp)
1c011df6:	5916                	lw	s2,100(sp)
1c011df8:	5986                	lw	s3,96(sp)
1c011dfa:	4a76                	lw	s4,92(sp)
1c011dfc:	4ae6                	lw	s5,88(sp)
1c011dfe:	4b56                	lw	s6,84(sp)
1c011e00:	4bc6                	lw	s7,80(sp)
1c011e02:	4c36                	lw	s8,76(sp)
1c011e04:	4ca6                	lw	s9,72(sp)
1c011e06:	4d16                	lw	s10,68(sp)
1c011e08:	4d86                	lw	s11,64(sp)
1c011e0a:	6165                	addi	sp,sp,112
1c011e0c:	60b0006f          	j	1c012c16 <printf>
1c011e10:	00758713          	addi	a4,a1,7
1c011e14:	830d                	srli	a4,a4,0x3
1c011e16:	f1402b73          	csrr	s6,mhartid
1c011e1a:	f45b3b33          	p.bclr	s6,s6,26,5
1c011e1e:	02eb0b33          	mul	s6,s6,a4
1c011e22:	975a                	add	a4,a4,s6
1c011e24:	04b75ab3          	p.minu	s5,a4,a1
1c011e28:	b571                	j	1c011cb4 <pulp_im2col_fp32+0x280>
1c011e2a:	1c001537          	lui	a0,0x1c001
1c011e2e:	bc450513          	addi	a0,a0,-1084 # 1c000bc4 <__DTOR_END__+0x35c>
1c011e32:	b7c1                	j	1c011df2 <pulp_im2col_fp32+0x3be>

1c011e34 <pulp_conv2d_fp32_fw_cl>:
1c011e34:	00c5ae03          	lw	t3,12(a1)
1c011e38:	0105a303          	lw	t1,16(a1)
1c011e3c:	7115                	addi	sp,sp,-224
1c011e3e:	c1da                	sw	s6,192(sp)
1c011e40:	026e0b33          	mul	s6,t3,t1
1c011e44:	cda2                	sw	s0,216(sp)
1c011e46:	c3d6                	sw	s5,196(sp)
1c011e48:	df5e                	sw	s7,188(sp)
1c011e4a:	dd62                	sw	s8,184(sp)
1c011e4c:	db66                	sw	s9,180(sp)
1c011e4e:	d96a                	sw	s10,176(sp)
1c011e50:	00062c83          	lw	s9,0(a2)
1c011e54:	0005ad03          	lw	s10,0(a1)
1c011e58:	4540                	lw	s0,12(a0)
1c011e5a:	01052a83          	lw	s5,16(a0)
1c011e5e:	01452b83          	lw	s7,20(a0)
1c011e62:	01462c03          	lw	s8,20(a2)
1c011e66:	cba6                	sw	s1,212(sp)
1c011e68:	c5d2                	sw	s4,200(sp)
1c011e6a:	d76e                	sw	s11,172(sp)
1c011e6c:	d2be                	sw	a5,100(sp)
1c011e6e:	8dbe                	mv	s11,a5
1c011e70:	84b6                	mv	s1,a3
1c011e72:	00204a37          	lui	s4,0x204
1c011e76:	0ff00793          	li	a5,255
1c011e7a:	cf86                	sw	ra,220(sp)
1c011e7c:	c9ca                	sw	s2,208(sp)
1c011e7e:	c7ce                	sw	s3,204(sp)
1c011e80:	c0aa                	sw	a0,64(sp)
1c011e82:	c2ae                	sw	a1,68(sp)
1c011e84:	c4b2                	sw	a2,72(sp)
1c011e86:	c6c2                	sw	a6,76(sp)
1c011e88:	c8b6                	sw	a3,80(sp)
1c011e8a:	ca82                	sw	zero,84(sp)
1c011e8c:	d0ba                	sw	a4,96(sp)
1c011e8e:	cc82                	sw	zero,88(sp)
1c011e90:	ceda                	sw	s6,92(sp)
1c011e92:	d482                	sw	zero,104(sp)
1c011e94:	08fa2223          	sw	a5,132(s4) # 204084 <__l1_heap_size+0x1f5850>
1c011e98:	200a0993          	addi	s3,s4,512
1c011e9c:	00f9a023          	sw	a5,0(s3)
1c011ea0:	00f9a623          	sw	a5,12(s3)
1c011ea4:	220a0913          	addi	s2,s4,544
1c011ea8:	1ff00793          	li	a5,511
1c011eac:	00f92023          	sw	a5,0(s2)
1c011eb0:	00f92623          	sw	a5,12(s2)
1c011eb4:	1c0127b7          	lui	a5,0x1c012
1c011eb8:	a3478793          	addi	a5,a5,-1484 # 1c011a34 <pulp_im2col_fp32>
1c011ebc:	002046b7          	lui	a3,0x204
1c011ec0:	08f6a023          	sw	a5,128(a3) # 204080 <__l1_heap_size+0x1f584c>
1c011ec4:	002047b7          	lui	a5,0x204
1c011ec8:	0088                	addi	a0,sp,64
1c011eca:	08a7a023          	sw	a0,128(a5) # 204080 <__l1_heap_size+0x1f584c>
1c011ece:	cc1a                	sw	t1,24(sp)
1c011ed0:	ca3a                	sw	a4,20(sp)
1c011ed2:	c842                	sw	a6,16(sp)
1c011ed4:	c646                	sw	a7,12(sp)
1c011ed6:	ce72                	sw	t3,28(sp)
1c011ed8:	3eb1                	jal	1c011a34 <pulp_im2col_fp32>
1c011eda:	21ca0a13          	addi	s4,s4,540
1c011ede:	000a2783          	lw	a5,0(s4)
1c011ee2:	4e72                	lw	t3,28(sp)
1c011ee4:	4752                	lw	a4,20(sp)
1c011ee6:	0486                	slli	s1,s1,0x1
1c011ee8:	41c40433          	sub	s0,s0,t3
1c011eec:	943a                	add	s0,s0,a4
1c011eee:	9426                	add	s0,s0,s1
1c011ef0:	02e44733          	div	a4,s0,a4
1c011ef4:	4362                	lw	t1,24(sp)
1c011ef6:	4785                	li	a5,1
1c011ef8:	c33e                	sw	a5,132(sp)
1c011efa:	406a86b3          	sub	a3,s5,t1
1c011efe:	96ee                	add	a3,a3,s11
1c011f00:	96a6                	add	a3,a3,s1
1c011f02:	10fc                	addi	a5,sp,108
1c011f04:	d63e                	sw	a5,44(sp)
1c011f06:	4842                	lw	a6,16(sp)
1c011f08:	48b2                	lw	a7,12(sp)
1c011f0a:	d6ea                	sw	s10,108(sp)
1c011f0c:	d8c2                	sw	a6,112(sp)
1c011f0e:	dae6                	sw	s9,116(sp)
1c011f10:	dce2                	sw	s8,120(sp)
1c011f12:	da02                	sw	zero,52(sp)
1c011f14:	dc02                	sw	zero,56(sp)
1c011f16:	de46                	sw	a7,60(sp)
1c011f18:	02d70733          	mul	a4,a4,a3
1c011f1c:	03b747b3          	div	a5,a4,s11
1c011f20:	037b0733          	mul	a4,s6,s7
1c011f24:	debe                	sw	a5,124(sp)
1c011f26:	0ff00793          	li	a5,255
1c011f2a:	c13a                	sw	a4,128(sp)
1c011f2c:	00204737          	lui	a4,0x204
1c011f30:	08f72223          	sw	a5,132(a4) # 204084 <__l1_heap_size+0x1f5850>
1c011f34:	00f9a023          	sw	a5,0(s3)
1c011f38:	00f9a623          	sw	a5,12(s3)
1c011f3c:	1ff00793          	li	a5,511
1c011f40:	00f92023          	sw	a5,0(s2)
1c011f44:	00f92623          	sw	a5,12(s2)
1c011f48:	1c0127b7          	lui	a5,0x1c012
1c011f4c:	f8278793          	addi	a5,a5,-126 # 1c011f82 <mm_manager>
1c011f50:	08f72023          	sw	a5,128(a4)
1c011f54:	002047b7          	lui	a5,0x204
1c011f58:	1068                	addi	a0,sp,44
1c011f5a:	08a7a023          	sw	a0,128(a5) # 204080 <__l1_heap_size+0x1f584c>
1c011f5e:	2015                	jal	1c011f82 <mm_manager>
1c011f60:	000a2783          	lw	a5,0(s4)
1c011f64:	40fe                	lw	ra,220(sp)
1c011f66:	446e                	lw	s0,216(sp)
1c011f68:	44de                	lw	s1,212(sp)
1c011f6a:	494e                	lw	s2,208(sp)
1c011f6c:	49be                	lw	s3,204(sp)
1c011f6e:	4a2e                	lw	s4,200(sp)
1c011f70:	4a9e                	lw	s5,196(sp)
1c011f72:	4b0e                	lw	s6,192(sp)
1c011f74:	5bfa                	lw	s7,188(sp)
1c011f76:	5c6a                	lw	s8,184(sp)
1c011f78:	5cda                	lw	s9,180(sp)
1c011f7a:	5d4a                	lw	s10,176(sp)
1c011f7c:	5dba                	lw	s11,172(sp)
1c011f7e:	612d                	addi	sp,sp,224
1c011f80:	8082                	ret

1c011f82 <mm_manager>:
1c011f82:	4518                	lw	a4,8(a0)
1c011f84:	4110                	lw	a2,0(a0)
1c011f86:	4554                	lw	a3,12(a0)
1c011f88:	491c                	lw	a5,16(a0)
1c011f8a:	ef35                	bnez	a4,1c012006 <mm_manager+0x84>
1c011f8c:	10069063          	bnez	a3,1c01208c <mm_manager+0x10a>
1c011f90:	22078363          	beqz	a5,1c0121b6 <mm_manager+0x234>
1c011f94:	4c17af63          	p.beqimm	a5,1,1c012472 <mm_manager+0x4f0>
1c011f98:	5027a263          	p.beqimm	a5,2,1c01249c <mm_manager+0x51a>
1c011f9c:	5037a863          	p.beqimm	a5,3,1c0124ac <mm_manager+0x52a>
1c011fa0:	5047a963          	p.beqimm	a5,4,1c0124b2 <mm_manager+0x530>
1c011fa4:	5057aa63          	p.beqimm	a5,5,1c0124b8 <mm_manager+0x536>
1c011fa8:	5067ab63          	p.beqimm	a5,6,1c0124be <mm_manager+0x53c>
1c011fac:	4a77ad63          	p.beqimm	a5,7,1c012466 <mm_manager+0x4e4>
1c011fb0:	5287a663          	p.beqimm	a5,8,1c0124dc <mm_manager+0x55a>
1c011fb4:	5497a563          	p.beqimm	a5,9,1c0124fe <mm_manager+0x57c>
1c011fb8:	58a7a063          	p.beqimm	a5,10,1c012538 <mm_manager+0x5b6>
1c011fbc:	58b7af63          	p.beqimm	a5,11,1c01255a <mm_manager+0x5d8>
1c011fc0:	5ac7ae63          	p.beqimm	a5,12,1c01257c <mm_manager+0x5fa>
1c011fc4:	5cd7af63          	p.beqimm	a5,13,1c0125a2 <mm_manager+0x620>
1c011fc8:	60e7a663          	p.beqimm	a5,14,1c0125d4 <mm_manager+0x652>
1c011fcc:	64f7a663          	p.beqimm	a5,15,1c012618 <mm_manager+0x696>
1c011fd0:	4741                	li	a4,16
1c011fd2:	68e78163          	beq	a5,a4,1c012654 <mm_manager+0x6d2>
1c011fd6:	4745                	li	a4,17
1c011fd8:	6ae78663          	beq	a5,a4,1c012684 <mm_manager+0x702>
1c011fdc:	4749                	li	a4,18
1c011fde:	6ee78a63          	beq	a5,a4,1c0126d2 <mm_manager+0x750>
1c011fe2:	474d                	li	a4,19
1c011fe4:	70e78c63          	beq	a5,a4,1c0126fc <mm_manager+0x77a>
1c011fe8:	4751                	li	a4,20
1c011fea:	74e78163          	beq	a5,a4,1c01272c <mm_manager+0x7aa>
1c011fee:	4755                	li	a4,21
1c011ff0:	76e78f63          	beq	a5,a4,1c01276e <mm_manager+0x7ec>
1c011ff4:	4759                	li	a4,22
1c011ff6:	7ae78a63          	beq	a5,a4,1c0127aa <mm_manager+0x828>
1c011ffa:	475d                	li	a4,23
1c011ffc:	1ae79763          	bne	a5,a4,1c0121aa <mm_manager+0x228>
1c012000:	8532                	mv	a0,a2
1c012002:	d4efd06f          	j	1c00f550 <mm_M_unroll_4x4>
1c012006:	10172263          	p.beqimm	a4,1,1c01210a <mm_manager+0x188>
1c01200a:	1c272363          	p.beqimm	a4,2,1c0121d0 <mm_manager+0x24e>
1c01200e:	12373063          	p.bneimm	a4,3,1c01212e <mm_manager+0x1ac>
1c012012:	2a069a63          	bnez	a3,1c0122c6 <mm_manager+0x344>
1c012016:	44078063          	beqz	a5,1c012456 <mm_manager+0x4d4>
1c01201a:	4c17ac63          	p.beqimm	a5,1,1c0124f2 <mm_manager+0x570>
1c01201e:	5027a463          	p.beqimm	a5,2,1c012526 <mm_manager+0x5a4>
1c012022:	5237a963          	p.beqimm	a5,3,1c012554 <mm_manager+0x5d2>
1c012026:	5447a663          	p.beqimm	a5,4,1c012572 <mm_manager+0x5f0>
1c01202a:	5657a963          	p.beqimm	a5,5,1c01259c <mm_manager+0x61a>
1c01202e:	5867aa63          	p.beqimm	a5,6,1c0125c2 <mm_manager+0x640>
1c012032:	5c77a263          	p.beqimm	a5,7,1c0125f6 <mm_manager+0x674>
1c012036:	5e87ac63          	p.beqimm	a5,8,1c01262e <mm_manager+0x6ac>
1c01203a:	6297af63          	p.beqimm	a5,9,1c012678 <mm_manager+0x6f6>
1c01203e:	66a7ab63          	p.beqimm	a5,10,1c0126b4 <mm_manager+0x732>
1c012042:	6ab7a163          	p.beqimm	a5,11,1c0126e4 <mm_manager+0x762>
1c012046:	6cc7aa63          	p.beqimm	a5,12,1c01271a <mm_manager+0x798>
1c01204a:	70d7a363          	p.beqimm	a5,13,1c012750 <mm_manager+0x7ce>
1c01204e:	76e7a163          	p.beqimm	a5,14,1c0127b0 <mm_manager+0x82e>
1c012052:	72f7ad63          	p.beqimm	a5,15,1c01278c <mm_manager+0x80a>
1c012056:	4741                	li	a4,16
1c012058:	78e78163          	beq	a5,a4,1c0127da <mm_manager+0x858>
1c01205c:	4745                	li	a4,17
1c01205e:	7ee78a63          	beq	a5,a4,1c012852 <mm_manager+0x8d0>
1c012062:	4749                	li	a4,18
1c012064:	7ee78463          	beq	a5,a4,1c01284c <mm_manager+0x8ca>
1c012068:	474d                	li	a4,19
1c01206a:	04e784e3          	beq	a5,a4,1c0128b2 <mm_manager+0x930>
1c01206e:	4751                	li	a4,20
1c012070:	02e78ee3          	beq	a5,a4,1c0128ac <mm_manager+0x92a>
1c012074:	4755                	li	a4,21
1c012076:	08e782e3          	beq	a5,a4,1c0128fa <mm_manager+0x978>
1c01207a:	4759                	li	a4,22
1c01207c:	06e78ce3          	beq	a5,a4,1c0128f4 <mm_manager+0x972>
1c012080:	475d                	li	a4,23
1c012082:	12e79463          	bne	a5,a4,1c0121aa <mm_manager+0x228>
1c012086:	8532                	mv	a0,a2
1c012088:	cc8fd06f          	j	1c00f550 <mm_M_unroll_4x4>
1c01208c:	0a16a763          	p.beqimm	a3,1,1c01213a <mm_manager+0x1b8>
1c012090:	1226ba63          	p.bneimm	a3,2,1c0121c4 <mm_manager+0x242>
1c012094:	40078163          	beqz	a5,1c012496 <mm_manager+0x514>
1c012098:	4217a663          	p.beqimm	a5,1,1c0124c4 <mm_manager+0x542>
1c01209c:	3c27a863          	p.beqimm	a5,2,1c01246c <mm_manager+0x4ea>
1c0120a0:	4437a163          	p.beqimm	a5,3,1c0124e2 <mm_manager+0x560>
1c0120a4:	4647a063          	p.beqimm	a5,4,1c012504 <mm_manager+0x582>
1c0120a8:	4857ab63          	p.beqimm	a5,5,1c01253e <mm_manager+0x5bc>
1c0120ac:	4a67aa63          	p.beqimm	a5,6,1c012560 <mm_manager+0x5de>
1c0120b0:	4c77ab63          	p.beqimm	a5,7,1c012586 <mm_manager+0x604>
1c0120b4:	4e87ac63          	p.beqimm	a5,8,1c0125ac <mm_manager+0x62a>
1c0120b8:	5297a163          	p.beqimm	a5,9,1c0125da <mm_manager+0x658>
1c0120bc:	54a7ab63          	p.beqimm	a5,10,1c012612 <mm_manager+0x690>
1c0120c0:	5ab7a363          	p.beqimm	a5,11,1c012666 <mm_manager+0x6e4>
1c0120c4:	5cc7a963          	p.beqimm	a5,12,1c012696 <mm_manager+0x714>
1c0120c8:	60d7a263          	p.beqimm	a5,13,1c0126cc <mm_manager+0x74a>
1c0120cc:	62e7a563          	p.beqimm	a5,14,1c0126f6 <mm_manager+0x774>
1c0120d0:	66f7a763          	p.beqimm	a5,15,1c01273e <mm_manager+0x7bc>
1c0120d4:	4741                	li	a4,16
1c0120d6:	68e78963          	beq	a5,a4,1c012768 <mm_manager+0x7e6>
1c0120da:	4745                	li	a4,17
1c0120dc:	6ce78163          	beq	a5,a4,1c01279e <mm_manager+0x81c>
1c0120e0:	4749                	li	a4,18
1c0120e2:	6ee78f63          	beq	a5,a4,1c0127e0 <mm_manager+0x85e>
1c0120e6:	474d                	li	a4,19
1c0120e8:	74e78f63          	beq	a5,a4,1c012846 <mm_manager+0x8c4>
1c0120ec:	4751                	li	a4,20
1c0120ee:	74e78963          	beq	a5,a4,1c012840 <mm_manager+0x8be>
1c0120f2:	4755                	li	a4,21
1c0120f4:	7ae78963          	beq	a5,a4,1c0128a6 <mm_manager+0x924>
1c0120f8:	4759                	li	a4,22
1c0120fa:	7ae78363          	beq	a5,a4,1c0128a0 <mm_manager+0x91e>
1c0120fe:	475d                	li	a4,23
1c012100:	0ae79563          	bne	a5,a4,1c0121aa <mm_manager+0x228>
1c012104:	8532                	mv	a0,a2
1c012106:	c4afd06f          	j	1c00f550 <mm_M_unroll_4x4>
1c01210a:	4148                	lw	a0,4(a0)
1c01210c:	eac5                	bnez	a3,1c0121bc <mm_manager+0x23a>
1c01210e:	38078a63          	beqz	a5,1c0124a2 <mm_manager+0x520>
1c012112:	3417a563          	p.beqimm	a5,1,1c01245c <mm_manager+0x4da>
1c012116:	3a27af63          	p.beqimm	a5,2,1c0124d4 <mm_manager+0x552>
1c01211a:	3c37aa63          	p.beqimm	a5,3,1c0124ee <mm_manager+0x56c>
1c01211e:	3e47af63          	p.beqimm	a5,4,1c01251c <mm_manager+0x59a>
1c012122:	4257a163          	p.beqimm	a5,5,1c012544 <mm_manager+0x5c2>
1c012126:	0867b263          	p.bneimm	a5,6,1c0121aa <mm_manager+0x228>
1c01212a:	f62fe06f          	j	1c01088c <mm_dw_unroll_1x4_u2>
1c01212e:	1c001537          	lui	a0,0x1c001
1c012132:	c3850513          	addi	a0,a0,-968 # 1c000c38 <__DTOR_END__+0x3d0>
1c012136:	2130006f          	j	1c012b48 <puts>
1c01213a:	36078663          	beqz	a5,1c0124a6 <mm_manager+0x524>
1c01213e:	3217a163          	p.beqimm	a5,1,1c012460 <mm_manager+0x4de>
1c012142:	3827a663          	p.beqimm	a5,2,1c0124ce <mm_manager+0x54c>
1c012146:	3a37a163          	p.beqimm	a5,3,1c0124e8 <mm_manager+0x566>
1c01214a:	3c47a663          	p.beqimm	a5,4,1c012516 <mm_manager+0x594>
1c01214e:	3e57ad63          	p.beqimm	a5,5,1c012548 <mm_manager+0x5c6>
1c012152:	4067aa63          	p.beqimm	a5,6,1c012566 <mm_manager+0x5e4>
1c012156:	4277ab63          	p.beqimm	a5,7,1c01258c <mm_manager+0x60a>
1c01215a:	4487ac63          	p.beqimm	a5,8,1c0125b2 <mm_manager+0x630>
1c01215e:	4897a963          	p.beqimm	a5,9,1c0125f0 <mm_manager+0x66e>
1c012162:	4ca7a963          	p.beqimm	a5,10,1c012634 <mm_manager+0x6b2>
1c012166:	50b7a363          	p.beqimm	a5,11,1c01266c <mm_manager+0x6ea>
1c01216a:	52c7ac63          	p.beqimm	a5,12,1c0126a2 <mm_manager+0x720>
1c01216e:	56d7a563          	p.beqimm	a5,13,1c0126d8 <mm_manager+0x756>
1c012172:	5ae7a163          	p.beqimm	a5,14,1c012714 <mm_manager+0x792>
1c012176:	5af7ae63          	p.beqimm	a5,15,1c012732 <mm_manager+0x7b0>
1c01217a:	4741                	li	a4,16
1c01217c:	5ce78d63          	beq	a5,a4,1c012756 <mm_manager+0x7d4>
1c012180:	4745                	li	a4,17
1c012182:	62e78a63          	beq	a5,a4,1c0127b6 <mm_manager+0x834>
1c012186:	4749                	li	a4,18
1c012188:	64e78663          	beq	a5,a4,1c0127d4 <mm_manager+0x852>
1c01218c:	474d                	li	a4,19
1c01218e:	66e78b63          	beq	a5,a4,1c012804 <mm_manager+0x882>
1c012192:	4751                	li	a4,20
1c012194:	6ae78063          	beq	a5,a4,1c012834 <mm_manager+0x8b2>
1c012198:	4755                	li	a4,21
1c01219a:	70e78f63          	beq	a5,a4,1c0128b8 <mm_manager+0x936>
1c01219e:	4759                	li	a4,22
1c0121a0:	70e78f63          	beq	a5,a4,1c0128be <mm_manager+0x93c>
1c0121a4:	475d                	li	a4,23
1c0121a6:	68e78a63          	beq	a5,a4,1c01283a <mm_manager+0x8b8>
1c0121aa:	1c001537          	lui	a0,0x1c001
1c0121ae:	c0450513          	addi	a0,a0,-1020 # 1c000c04 <__DTOR_END__+0x39c>
1c0121b2:	1970006f          	j	1c012b48 <puts>
1c0121b6:	8532                	mv	a0,a2
1c0121b8:	cb9f606f          	j	1c008e70 <mm>
1c0121bc:	2a16ae63          	p.beqimm	a3,1,1c012478 <mm_manager+0x4f6>
1c0121c0:	1826a263          	p.beqimm	a3,2,1c012344 <mm_manager+0x3c2>
1c0121c4:	1c001537          	lui	a0,0x1c001
1c0121c8:	c2050513          	addi	a0,a0,-992 # 1c000c20 <__DTOR_END__+0x3b8>
1c0121cc:	17d0006f          	j	1c012b48 <puts>
1c0121d0:	eea5                	bnez	a3,1c012248 <mm_manager+0x2c6>
1c0121d2:	26078f63          	beqz	a5,1c012450 <mm_manager+0x4ce>
1c0121d6:	3217a163          	p.beqimm	a5,1,1c0124f8 <mm_manager+0x576>
1c0121da:	3427a363          	p.beqimm	a5,2,1c012520 <mm_manager+0x59e>
1c0121de:	3637a863          	p.beqimm	a5,3,1c01254e <mm_manager+0x5cc>
1c0121e2:	3847a563          	p.beqimm	a5,4,1c01256c <mm_manager+0x5ea>
1c0121e6:	3a57a863          	p.beqimm	a5,5,1c012596 <mm_manager+0x614>
1c0121ea:	3c67a963          	p.beqimm	a5,6,1c0125bc <mm_manager+0x63a>
1c0121ee:	4077a763          	p.beqimm	a5,7,1c0125fc <mm_manager+0x67a>
1c0121f2:	4487a463          	p.beqimm	a5,8,1c01263a <mm_manager+0x6b8>
1c0121f6:	4697ae63          	p.beqimm	a5,9,1c012672 <mm_manager+0x6f0>
1c0121fa:	4aa7aa63          	p.beqimm	a5,10,1c0126ae <mm_manager+0x72c>
1c0121fe:	4eb7a063          	p.beqimm	a5,11,1c0126de <mm_manager+0x75c>
1c012202:	52c7a263          	p.beqimm	a5,12,1c012726 <mm_manager+0x7a4>
1c012206:	52d7a963          	p.beqimm	a5,13,1c012738 <mm_manager+0x7b6>
1c01220a:	54e7ac63          	p.beqimm	a5,14,1c012762 <mm_manager+0x7e0>
1c01220e:	5af7a763          	p.beqimm	a5,15,1c0127bc <mm_manager+0x83a>
1c012212:	4741                	li	a4,16
1c012214:	5ae78763          	beq	a5,a4,1c0127c2 <mm_manager+0x840>
1c012218:	4745                	li	a4,17
1c01221a:	60e78463          	beq	a5,a4,1c012822 <mm_manager+0x8a0>
1c01221e:	4749                	li	a4,18
1c012220:	5ee78e63          	beq	a5,a4,1c01281c <mm_manager+0x89a>
1c012224:	474d                	li	a4,19
1c012226:	66e78a63          	beq	a5,a4,1c01289a <mm_manager+0x918>
1c01222a:	4751                	li	a4,20
1c01222c:	66e78463          	beq	a5,a4,1c012894 <mm_manager+0x912>
1c012230:	4755                	li	a4,21
1c012232:	6ce78a63          	beq	a5,a4,1c012906 <mm_manager+0x984>
1c012236:	4759                	li	a4,22
1c012238:	6ce78463          	beq	a5,a4,1c012900 <mm_manager+0x97e>
1c01223c:	475d                	li	a4,23
1c01223e:	f6e796e3          	bne	a5,a4,1c0121aa <mm_manager+0x228>
1c012242:	8532                	mv	a0,a2
1c012244:	b0cfd06f          	j	1c00f550 <mm_M_unroll_4x4>
1c012248:	1016ae63          	p.beqimm	a3,1,1c012364 <mm_manager+0x3e2>
1c01224c:	f626bce3          	p.bneimm	a3,2,1c0121c4 <mm_manager+0x242>
1c012250:	2a078d63          	beqz	a5,1c01250a <mm_manager+0x588>
1c012254:	3617aa63          	p.beqimm	a5,1,1c0125c8 <mm_manager+0x646>
1c012258:	3a27aa63          	p.beqimm	a5,2,1c01260c <mm_manager+0x68a>
1c01225c:	3e37a963          	p.beqimm	a5,3,1c01264e <mm_manager+0x6cc>
1c012260:	4247a563          	p.beqimm	a5,4,1c01268a <mm_manager+0x708>
1c012264:	4457a263          	p.beqimm	a5,5,1c0126a8 <mm_manager+0x726>
1c012268:	4867a163          	p.beqimm	a5,6,1c0126ea <mm_manager+0x768>
1c01226c:	4a77aa63          	p.beqimm	a5,7,1c012720 <mm_manager+0x79e>
1c012270:	5087a263          	p.beqimm	a5,8,1c012774 <mm_manager+0x7f2>
1c012274:	5097af63          	p.beqimm	a5,9,1c012792 <mm_manager+0x810>
1c012278:	56a7aa63          	p.beqimm	a5,10,1c0127ec <mm_manager+0x86a>
1c01227c:	56b7a563          	p.beqimm	a5,11,1c0127e6 <mm_manager+0x864>
1c012280:	58c7a863          	p.beqimm	a5,12,1c012810 <mm_manager+0x88e>
1c012284:	5ed7a663          	p.beqimm	a5,13,1c012870 <mm_manager+0x8ee>
1c012288:	60e7a363          	p.beqimm	a5,14,1c01288e <mm_manager+0x90c>
1c01228c:	5ef7ae63          	p.beqimm	a5,15,1c012888 <mm_manager+0x906>
1c012290:	4741                	li	a4,16
1c012292:	64e78e63          	beq	a5,a4,1c0128ee <mm_manager+0x96c>
1c012296:	4745                	li	a4,17
1c012298:	64e78863          	beq	a5,a4,1c0128e8 <mm_manager+0x966>
1c01229c:	4749                	li	a4,18
1c01229e:	64e78263          	beq	a5,a4,1c0128e2 <mm_manager+0x960>
1c0122a2:	474d                	li	a4,19
1c0122a4:	62e78c63          	beq	a5,a4,1c0128dc <mm_manager+0x95a>
1c0122a8:	4751                	li	a4,20
1c0122aa:	62e78663          	beq	a5,a4,1c0128d6 <mm_manager+0x954>
1c0122ae:	4755                	li	a4,21
1c0122b0:	62e78063          	beq	a5,a4,1c0128d0 <mm_manager+0x94e>
1c0122b4:	4759                	li	a4,22
1c0122b6:	6ce78763          	beq	a5,a4,1c012984 <mm_manager+0xa02>
1c0122ba:	475d                	li	a4,23
1c0122bc:	eee797e3          	bne	a5,a4,1c0121aa <mm_manager+0x228>
1c0122c0:	8532                	mv	a0,a2
1c0122c2:	a8efd06f          	j	1c00f550 <mm_M_unroll_4x4>
1c0122c6:	1016aa63          	p.beqimm	a3,1,1c0123da <mm_manager+0x458>
1c0122ca:	ee26bde3          	p.bneimm	a3,2,1c0121c4 <mm_manager+0x242>
1c0122ce:	24078163          	beqz	a5,1c012510 <mm_manager+0x58e>
1c0122d2:	2e17ae63          	p.beqimm	a5,1,1c0125ce <mm_manager+0x64c>
1c0122d6:	3227a863          	p.beqimm	a5,2,1c012606 <mm_manager+0x684>
1c0122da:	3637a763          	p.beqimm	a5,3,1c012648 <mm_manager+0x6c6>
1c0122de:	3a47a063          	p.beqimm	a5,4,1c01267e <mm_manager+0x6fc>
1c0122e2:	3c57ac63          	p.beqimm	a5,5,1c0126ba <mm_manager+0x738>
1c0122e6:	4067a563          	p.beqimm	a5,6,1c0126f0 <mm_manager+0x76e>
1c0122ea:	4277a263          	p.beqimm	a5,7,1c01270e <mm_manager+0x78c>
1c0122ee:	4887a663          	p.beqimm	a5,8,1c01277a <mm_manager+0x7f8>
1c0122f2:	4a97a363          	p.beqimm	a5,9,1c012798 <mm_manager+0x816>
1c0122f6:	4ca7ac63          	p.beqimm	a5,10,1c0127ce <mm_manager+0x84c>
1c0122fa:	4cb7a763          	p.beqimm	a5,11,1c0127c8 <mm_manager+0x846>
1c0122fe:	52c7a863          	p.beqimm	a5,12,1c01282e <mm_manager+0x8ac>
1c012302:	54d7ae63          	p.beqimm	a5,13,1c01285e <mm_manager+0x8dc>
1c012306:	56e7a263          	p.beqimm	a5,14,1c01286a <mm_manager+0x8e8>
1c01230a:	54f7ad63          	p.beqimm	a5,15,1c012864 <mm_manager+0x8e2>
1c01230e:	4741                	li	a4,16
1c012310:	62e78063          	beq	a5,a4,1c012930 <mm_manager+0x9ae>
1c012314:	4745                	li	a4,17
1c012316:	60e78a63          	beq	a5,a4,1c01292a <mm_manager+0x9a8>
1c01231a:	4749                	li	a4,18
1c01231c:	60e78463          	beq	a5,a4,1c012924 <mm_manager+0x9a2>
1c012320:	474d                	li	a4,19
1c012322:	5ee78e63          	beq	a5,a4,1c01291e <mm_manager+0x99c>
1c012326:	4751                	li	a4,20
1c012328:	5ee78563          	beq	a5,a4,1c012912 <mm_manager+0x990>
1c01232c:	4755                	li	a4,21
1c01232e:	5ce78f63          	beq	a5,a4,1c01290c <mm_manager+0x98a>
1c012332:	4759                	li	a4,22
1c012334:	62e78f63          	beq	a5,a4,1c012972 <mm_manager+0x9f0>
1c012338:	475d                	li	a4,23
1c01233a:	e6e798e3          	bne	a5,a4,1c0121aa <mm_manager+0x228>
1c01233e:	8532                	mv	a0,a2
1c012340:	a10fd06f          	j	1c00f550 <mm_M_unroll_4x4>
1c012344:	18078363          	beqz	a5,1c0124ca <mm_manager+0x548>
1c012348:	2217a863          	p.beqimm	a5,1,1c012578 <mm_manager+0x5f6>
1c01234c:	2427a363          	p.beqimm	a5,2,1c012592 <mm_manager+0x610>
1c012350:	2637a463          	p.beqimm	a5,3,1c0125b8 <mm_manager+0x636>
1c012354:	2a47a763          	p.beqimm	a5,4,1c012602 <mm_manager+0x680>
1c012358:	2e57a463          	p.beqimm	a5,5,1c012640 <mm_manager+0x6be>
1c01235c:	e467b7e3          	p.bneimm	a5,6,1c0121aa <mm_manager+0x228>
1c012360:	a6cff06f          	j	1c0115cc <mm_dw_in_grad_unroll_1x4_u2>
1c012364:	1c078763          	beqz	a5,1c012532 <mm_manager+0x5b0>
1c012368:	2617ac63          	p.beqimm	a5,1,1c0125e0 <mm_manager+0x65e>
1c01236c:	2a27a963          	p.beqimm	a5,2,1c01261e <mm_manager+0x69c>
1c012370:	2e37a863          	p.beqimm	a5,3,1c012660 <mm_manager+0x6de>
1c012374:	3047ae63          	p.beqimm	a5,4,1c012690 <mm_manager+0x70e>
1c012378:	3457a763          	p.beqimm	a5,5,1c0126c6 <mm_manager+0x744>
1c01237c:	3867a363          	p.beqimm	a5,6,1c012702 <mm_manager+0x780>
1c012380:	3c77a563          	p.beqimm	a5,7,1c01274a <mm_manager+0x7c8>
1c012384:	3e87ae63          	p.beqimm	a5,8,1c012780 <mm_manager+0x7fe>
1c012388:	3e97af63          	p.beqimm	a5,9,1c012786 <mm_manager+0x804>
1c01238c:	46a7a363          	p.beqimm	a5,10,1c0127f2 <mm_manager+0x870>
1c012390:	46b7ad63          	p.beqimm	a5,11,1c01280a <mm_manager+0x888>
1c012394:	48c7aa63          	p.beqimm	a5,12,1c012828 <mm_manager+0x8a6>
1c012398:	4cd7a063          	p.beqimm	a5,13,1c012858 <mm_manager+0x8d6>
1c01239c:	52e7a763          	p.beqimm	a5,14,1c0128ca <mm_manager+0x948>
1c0123a0:	52f7a263          	p.beqimm	a5,15,1c0128c4 <mm_manager+0x942>
1c0123a4:	4741                	li	a4,16
1c0123a6:	5ce78063          	beq	a5,a4,1c012966 <mm_manager+0x9e4>
1c0123aa:	4745                	li	a4,17
1c0123ac:	5ae78a63          	beq	a5,a4,1c012960 <mm_manager+0x9de>
1c0123b0:	4749                	li	a4,18
1c0123b2:	5ae78463          	beq	a5,a4,1c01295a <mm_manager+0x9d8>
1c0123b6:	474d                	li	a4,19
1c0123b8:	58e78e63          	beq	a5,a4,1c012954 <mm_manager+0x9d2>
1c0123bc:	4751                	li	a4,20
1c0123be:	58e78863          	beq	a5,a4,1c01294e <mm_manager+0x9cc>
1c0123c2:	4755                	li	a4,21
1c0123c4:	5ae78d63          	beq	a5,a4,1c01297e <mm_manager+0x9fc>
1c0123c8:	4759                	li	a4,22
1c0123ca:	5ce78063          	beq	a5,a4,1c01298a <mm_manager+0xa08>
1c0123ce:	475d                	li	a4,23
1c0123d0:	dce79de3          	bne	a5,a4,1c0121aa <mm_manager+0x228>
1c0123d4:	8532                	mv	a0,a2
1c0123d6:	97afd06f          	j	1c00f550 <mm_M_unroll_4x4>
1c0123da:	14078963          	beqz	a5,1c01252c <mm_manager+0x5aa>
1c0123de:	2017a463          	p.beqimm	a5,1,1c0125e6 <mm_manager+0x664>
1c0123e2:	2427a363          	p.beqimm	a5,2,1c012628 <mm_manager+0x6a6>
1c0123e6:	2637aa63          	p.beqimm	a5,3,1c01265a <mm_manager+0x6d8>
1c0123ea:	2a47a963          	p.beqimm	a5,4,1c01269c <mm_manager+0x71a>
1c0123ee:	2c57a963          	p.beqimm	a5,5,1c0126c0 <mm_manager+0x73e>
1c0123f2:	3067ab63          	p.beqimm	a5,6,1c012708 <mm_manager+0x786>
1c0123f6:	3477a763          	p.beqimm	a5,7,1c012744 <mm_manager+0x7c2>
1c0123fa:	3687a163          	p.beqimm	a5,8,1c01275c <mm_manager+0x7da>
1c0123fe:	3a97a363          	p.beqimm	a5,9,1c0127a4 <mm_manager+0x822>
1c012402:	3ea7ab63          	p.beqimm	a5,10,1c0127f8 <mm_manager+0x876>
1c012406:	3eb7ac63          	p.beqimm	a5,11,1c0127fe <mm_manager+0x87c>
1c01240a:	40c7a663          	p.beqimm	a5,12,1c012816 <mm_manager+0x894>
1c01240e:	46d7a463          	p.beqimm	a5,13,1c012876 <mm_manager+0x8f4>
1c012412:	46e7a863          	p.beqimm	a5,14,1c012882 <mm_manager+0x900>
1c012416:	46f7a363          	p.beqimm	a5,15,1c01287c <mm_manager+0x8fa>
1c01241a:	4741                	li	a4,16
1c01241c:	52e78663          	beq	a5,a4,1c012948 <mm_manager+0x9c6>
1c012420:	4745                	li	a4,17
1c012422:	52e78063          	beq	a5,a4,1c012942 <mm_manager+0x9c0>
1c012426:	4749                	li	a4,18
1c012428:	50e78a63          	beq	a5,a4,1c01293c <mm_manager+0x9ba>
1c01242c:	474d                	li	a4,19
1c01242e:	50e78463          	beq	a5,a4,1c012936 <mm_manager+0x9b4>
1c012432:	4751                	li	a4,20
1c012434:	4ee78263          	beq	a5,a4,1c012918 <mm_manager+0x996>
1c012438:	4755                	li	a4,21
1c01243a:	52e78f63          	beq	a5,a4,1c012978 <mm_manager+0x9f6>
1c01243e:	4759                	li	a4,22
1c012440:	52e78663          	beq	a5,a4,1c01296c <mm_manager+0x9ea>
1c012444:	475d                	li	a4,23
1c012446:	d6e792e3          	bne	a5,a4,1c0121aa <mm_manager+0x228>
1c01244a:	8532                	mv	a0,a2
1c01244c:	904fd06f          	j	1c00f550 <mm_M_unroll_4x4>
1c012450:	8532                	mv	a0,a2
1c012452:	a1ff606f          	j	1c008e70 <mm>
1c012456:	8532                	mv	a0,a2
1c012458:	a19f606f          	j	1c008e70 <mm>
1c01245c:	891fd06f          	j	1c00fcec <mm_dw_u2>
1c012460:	8532                	mv	a0,a2
1c012462:	c37f606f          	j	1c009098 <mm_M>
1c012466:	8532                	mv	a0,a2
1c012468:	83cf806f          	j	1c00a4a4 <mm_unroll_4x1>
1c01246c:	8532                	mv	a0,a2
1c01246e:	f57f606f          	j	1c0093c4 <mm_u2>
1c012472:	8532                	mv	a0,a2
1c012474:	c25f606f          	j	1c009098 <mm_M>
1c012478:	c3a5                	beqz	a5,1c0124d8 <mm_manager+0x556>
1c01247a:	1017a463          	p.beqimm	a5,1,1c012582 <mm_manager+0x600>
1c01247e:	1227a563          	p.beqimm	a5,2,1c0125a8 <mm_manager+0x626>
1c012482:	1637a563          	p.beqimm	a5,3,1c0125ec <mm_manager+0x66a>
1c012486:	1847af63          	p.beqimm	a5,4,1c012624 <mm_manager+0x6a2>
1c01248a:	1a57ad63          	p.beqimm	a5,5,1c012644 <mm_manager+0x6c2>
1c01248e:	d067bee3          	p.bneimm	a5,6,1c0121aa <mm_manager+0x228>
1c012492:	bfafe06f          	j	1c01088c <mm_dw_unroll_1x4_u2>
1c012496:	8532                	mv	a0,a2
1c012498:	9d9f606f          	j	1c008e70 <mm>
1c01249c:	8532                	mv	a0,a2
1c01249e:	f27f606f          	j	1c0093c4 <mm_u2>
1c0124a2:	d4ff606f          	j	1c0091f0 <mm_dw>
1c0124a6:	8532                	mv	a0,a2
1c0124a8:	9c9f606f          	j	1c008e70 <mm>
1c0124ac:	8532                	mv	a0,a2
1c0124ae:	99ef706f          	j	1c00964c <mm_unroll_1x2>
1c0124b2:	8532                	mv	a0,a2
1c0124b4:	c5cf706f          	j	1c009910 <mm_unroll_1x4>
1c0124b8:	8532                	mv	a0,a2
1c0124ba:	ff6f706f          	j	1c009cb0 <mm_unroll_1x8>
1c0124be:	8532                	mv	a0,a2
1c0124c0:	c89f706f          	j	1c00a148 <mm_unroll_2x1>
1c0124c4:	8532                	mv	a0,a2
1c0124c6:	bd3f606f          	j	1c009098 <mm_M>
1c0124ca:	e03f606f          	j	1c0092cc <mm_dw_in_grad>
1c0124ce:	8532                	mv	a0,a2
1c0124d0:	ef5f606f          	j	1c0093c4 <mm_u2>
1c0124d4:	a15fd06f          	j	1c00fee8 <mm_dw_u3>
1c0124d8:	d19f606f          	j	1c0091f0 <mm_dw>
1c0124dc:	8532                	mv	a0,a2
1c0124de:	c7ef806f          	j	1c00a95c <mm_unroll_8x1>
1c0124e2:	8532                	mv	a0,a2
1c0124e4:	968f706f          	j	1c00964c <mm_unroll_1x2>
1c0124e8:	8532                	mv	a0,a2
1c0124ea:	962f706f          	j	1c00964c <mm_unroll_1x2>
1c0124ee:	bdbfd06f          	j	1c0100c8 <mm_dw_unroll_1x2>
1c0124f2:	8532                	mv	a0,a2
1c0124f4:	ba5f606f          	j	1c009098 <mm_M>
1c0124f8:	8532                	mv	a0,a2
1c0124fa:	b9ff606f          	j	1c009098 <mm_M>
1c0124fe:	8532                	mv	a0,a2
1c012500:	aedf806f          	j	1c00afec <mm_unroll_2x2>
1c012504:	8532                	mv	a0,a2
1c012506:	c0af706f          	j	1c009910 <mm_unroll_1x4>
1c01250a:	8532                	mv	a0,a2
1c01250c:	965f606f          	j	1c008e70 <mm>
1c012510:	8532                	mv	a0,a2
1c012512:	95ff606f          	j	1c008e70 <mm>
1c012516:	8532                	mv	a0,a2
1c012518:	bf8f706f          	j	1c009910 <mm_unroll_1x4>
1c01251c:	dc9fd06f          	j	1c0102e4 <mm_dw_unroll_1x4>
1c012520:	8532                	mv	a0,a2
1c012522:	ea3f606f          	j	1c0093c4 <mm_u2>
1c012526:	8532                	mv	a0,a2
1c012528:	e9df606f          	j	1c0093c4 <mm_u2>
1c01252c:	8532                	mv	a0,a2
1c01252e:	943f606f          	j	1c008e70 <mm>
1c012532:	8532                	mv	a0,a2
1c012534:	93df606f          	j	1c008e70 <mm>
1c012538:	8532                	mv	a0,a2
1c01253a:	fb7f806f          	j	1c00b4f0 <mm_unroll_2x4>
1c01253e:	8532                	mv	a0,a2
1c012540:	f70f706f          	j	1c009cb0 <mm_unroll_1x8>
1c012544:	fe5fd06f          	j	1c010528 <mm_dw_unroll_1x2_u2>
1c012548:	8532                	mv	a0,a2
1c01254a:	f66f706f          	j	1c009cb0 <mm_unroll_1x8>
1c01254e:	8532                	mv	a0,a2
1c012550:	8fcf706f          	j	1c00964c <mm_unroll_1x2>
1c012554:	8532                	mv	a0,a2
1c012556:	8f6f706f          	j	1c00964c <mm_unroll_1x2>
1c01255a:	8532                	mv	a0,a2
1c01255c:	d54f906f          	j	1c00bab0 <mm_unroll_4x2>
1c012560:	8532                	mv	a0,a2
1c012562:	be7f706f          	j	1c00a148 <mm_unroll_2x1>
1c012566:	8532                	mv	a0,a2
1c012568:	be1f706f          	j	1c00a148 <mm_unroll_2x1>
1c01256c:	8532                	mv	a0,a2
1c01256e:	ba2f706f          	j	1c009910 <mm_unroll_1x4>
1c012572:	8532                	mv	a0,a2
1c012574:	b9cf706f          	j	1c009910 <mm_unroll_1x4>
1c012578:	f18fe06f          	j	1c010c90 <mm_dw_in_grad_u2>
1c01257c:	8532                	mv	a0,a2
1c01257e:	c33f906f          	j	1c00c1b0 <mm_unroll_4x4>
1c012582:	f6afd06f          	j	1c00fcec <mm_dw_u2>
1c012586:	8532                	mv	a0,a2
1c012588:	f1df706f          	j	1c00a4a4 <mm_unroll_4x1>
1c01258c:	8532                	mv	a0,a2
1c01258e:	f17f706f          	j	1c00a4a4 <mm_unroll_4x1>
1c012592:	873fe06f          	j	1c010e04 <mm_dw_in_grad_u3>
1c012596:	8532                	mv	a0,a2
1c012598:	f18f706f          	j	1c009cb0 <mm_unroll_1x8>
1c01259c:	8532                	mv	a0,a2
1c01259e:	f12f706f          	j	1c009cb0 <mm_unroll_1x8>
1c0125a2:	8532                	mv	a0,a2
1c0125a4:	c68fa06f          	j	1c00ca0c <mm_M_u2>
1c0125a8:	941fd06f          	j	1c00fee8 <mm_dw_u3>
1c0125ac:	8532                	mv	a0,a2
1c0125ae:	baef806f          	j	1c00a95c <mm_unroll_8x1>
1c0125b2:	8532                	mv	a0,a2
1c0125b4:	ba8f806f          	j	1c00a95c <mm_unroll_8x1>
1c0125b8:	9b5fe06f          	j	1c010f6c <mm_dw_in_grad_unroll_1x2>
1c0125bc:	8532                	mv	a0,a2
1c0125be:	b8bf706f          	j	1c00a148 <mm_unroll_2x1>
1c0125c2:	8532                	mv	a0,a2
1c0125c4:	b85f706f          	j	1c00a148 <mm_unroll_2x1>
1c0125c8:	8532                	mv	a0,a2
1c0125ca:	acff606f          	j	1c009098 <mm_M>
1c0125ce:	8532                	mv	a0,a2
1c0125d0:	ac9f606f          	j	1c009098 <mm_M>
1c0125d4:	8532                	mv	a0,a2
1c0125d6:	ae2fb06f          	j	1c00d8b8 <mm_M_unroll_1x2>
1c0125da:	8532                	mv	a0,a2
1c0125dc:	a11f806f          	j	1c00afec <mm_unroll_2x2>
1c0125e0:	8532                	mv	a0,a2
1c0125e2:	ab7f606f          	j	1c009098 <mm_M>
1c0125e6:	8532                	mv	a0,a2
1c0125e8:	ab1f606f          	j	1c009098 <mm_M>
1c0125ec:	addfd06f          	j	1c0100c8 <mm_dw_unroll_1x2>
1c0125f0:	8532                	mv	a0,a2
1c0125f2:	9fbf806f          	j	1c00afec <mm_unroll_2x2>
1c0125f6:	8532                	mv	a0,a2
1c0125f8:	eadf706f          	j	1c00a4a4 <mm_unroll_4x1>
1c0125fc:	8532                	mv	a0,a2
1c0125fe:	ea7f706f          	j	1c00a4a4 <mm_unroll_4x1>
1c012602:	ad7fe06f          	j	1c0110d8 <mm_dw_in_grad_unroll_1x4>
1c012606:	8532                	mv	a0,a2
1c012608:	dbdf606f          	j	1c0093c4 <mm_u2>
1c01260c:	8532                	mv	a0,a2
1c01260e:	db7f606f          	j	1c0093c4 <mm_u2>
1c012612:	8532                	mv	a0,a2
1c012614:	eddf806f          	j	1c00b4f0 <mm_unroll_2x4>
1c012618:	8532                	mv	a0,a2
1c01261a:	dd6fb06f          	j	1c00dbf0 <mm_M_unroll_1x4>
1c01261e:	8532                	mv	a0,a2
1c012620:	da5f606f          	j	1c0093c4 <mm_u2>
1c012624:	cc1fd06f          	j	1c0102e4 <mm_dw_unroll_1x4>
1c012628:	8532                	mv	a0,a2
1c01262a:	d9bf606f          	j	1c0093c4 <mm_u2>
1c01262e:	8532                	mv	a0,a2
1c012630:	b2cf806f          	j	1c00a95c <mm_unroll_8x1>
1c012634:	8532                	mv	a0,a2
1c012636:	ebbf806f          	j	1c00b4f0 <mm_unroll_2x4>
1c01263a:	8532                	mv	a0,a2
1c01263c:	b20f806f          	j	1c00a95c <mm_unroll_8x1>
1c012640:	d45fe06f          	j	1c011384 <mm_dw_in_grad_unroll_1x2_u2>
1c012644:	ee5fd06f          	j	1c010528 <mm_dw_unroll_1x2_u2>
1c012648:	8532                	mv	a0,a2
1c01264a:	802f706f          	j	1c00964c <mm_unroll_1x2>
1c01264e:	8532                	mv	a0,a2
1c012650:	ffdf606f          	j	1c00964c <mm_unroll_1x2>
1c012654:	8532                	mv	a0,a2
1c012656:	8e7fb06f          	j	1c00df3c <mm_M_unroll_1x8>
1c01265a:	8532                	mv	a0,a2
1c01265c:	ff1f606f          	j	1c00964c <mm_unroll_1x2>
1c012660:	8532                	mv	a0,a2
1c012662:	febf606f          	j	1c00964c <mm_unroll_1x2>
1c012666:	8532                	mv	a0,a2
1c012668:	c48f906f          	j	1c00bab0 <mm_unroll_4x2>
1c01266c:	8532                	mv	a0,a2
1c01266e:	c42f906f          	j	1c00bab0 <mm_unroll_4x2>
1c012672:	8532                	mv	a0,a2
1c012674:	979f806f          	j	1c00afec <mm_unroll_2x2>
1c012678:	8532                	mv	a0,a2
1c01267a:	973f806f          	j	1c00afec <mm_unroll_2x2>
1c01267e:	8532                	mv	a0,a2
1c012680:	a90f706f          	j	1c009910 <mm_unroll_1x4>
1c012684:	8532                	mv	a0,a2
1c012686:	dfafa06f          	j	1c00cc80 <mm_M_unroll_2x1>
1c01268a:	8532                	mv	a0,a2
1c01268c:	a84f706f          	j	1c009910 <mm_unroll_1x4>
1c012690:	8532                	mv	a0,a2
1c012692:	a7ef706f          	j	1c009910 <mm_unroll_1x4>
1c012696:	8532                	mv	a0,a2
1c012698:	b19f906f          	j	1c00c1b0 <mm_unroll_4x4>
1c01269c:	8532                	mv	a0,a2
1c01269e:	a72f706f          	j	1c009910 <mm_unroll_1x4>
1c0126a2:	8532                	mv	a0,a2
1c0126a4:	b0df906f          	j	1c00c1b0 <mm_unroll_4x4>
1c0126a8:	8532                	mv	a0,a2
1c0126aa:	e06f706f          	j	1c009cb0 <mm_unroll_1x8>
1c0126ae:	8532                	mv	a0,a2
1c0126b0:	e41f806f          	j	1c00b4f0 <mm_unroll_2x4>
1c0126b4:	8532                	mv	a0,a2
1c0126b6:	e3bf806f          	j	1c00b4f0 <mm_unroll_2x4>
1c0126ba:	8532                	mv	a0,a2
1c0126bc:	df4f706f          	j	1c009cb0 <mm_unroll_1x8>
1c0126c0:	8532                	mv	a0,a2
1c0126c2:	deef706f          	j	1c009cb0 <mm_unroll_1x8>
1c0126c6:	8532                	mv	a0,a2
1c0126c8:	de8f706f          	j	1c009cb0 <mm_unroll_1x8>
1c0126cc:	8532                	mv	a0,a2
1c0126ce:	b3efa06f          	j	1c00ca0c <mm_M_u2>
1c0126d2:	8532                	mv	a0,a2
1c0126d4:	899fa06f          	j	1c00cf6c <mm_M_unroll_4x1>
1c0126d8:	8532                	mv	a0,a2
1c0126da:	b32fa06f          	j	1c00ca0c <mm_M_u2>
1c0126de:	8532                	mv	a0,a2
1c0126e0:	bd0f906f          	j	1c00bab0 <mm_unroll_4x2>
1c0126e4:	8532                	mv	a0,a2
1c0126e6:	bcaf906f          	j	1c00bab0 <mm_unroll_4x2>
1c0126ea:	8532                	mv	a0,a2
1c0126ec:	a5df706f          	j	1c00a148 <mm_unroll_2x1>
1c0126f0:	8532                	mv	a0,a2
1c0126f2:	a57f706f          	j	1c00a148 <mm_unroll_2x1>
1c0126f6:	8532                	mv	a0,a2
1c0126f8:	9c0fb06f          	j	1c00d8b8 <mm_M_unroll_1x2>
1c0126fc:	8532                	mv	a0,a2
1c0126fe:	c67fa06f          	j	1c00d364 <mm_M_unroll_8x1>
1c012702:	8532                	mv	a0,a2
1c012704:	a45f706f          	j	1c00a148 <mm_unroll_2x1>
1c012708:	8532                	mv	a0,a2
1c01270a:	a3ff706f          	j	1c00a148 <mm_unroll_2x1>
1c01270e:	8532                	mv	a0,a2
1c012710:	d95f706f          	j	1c00a4a4 <mm_unroll_4x1>
1c012714:	8532                	mv	a0,a2
1c012716:	9a2fb06f          	j	1c00d8b8 <mm_M_unroll_1x2>
1c01271a:	8532                	mv	a0,a2
1c01271c:	a95f906f          	j	1c00c1b0 <mm_unroll_4x4>
1c012720:	8532                	mv	a0,a2
1c012722:	d83f706f          	j	1c00a4a4 <mm_unroll_4x1>
1c012726:	8532                	mv	a0,a2
1c012728:	a89f906f          	j	1c00c1b0 <mm_unroll_4x4>
1c01272c:	8532                	mv	a0,a2
1c01272e:	cf7fb06f          	j	1c00e424 <mm_M_unroll_2x2>
1c012732:	8532                	mv	a0,a2
1c012734:	cbcfb06f          	j	1c00dbf0 <mm_M_unroll_1x4>
1c012738:	8532                	mv	a0,a2
1c01273a:	ad2fa06f          	j	1c00ca0c <mm_M_u2>
1c01273e:	8532                	mv	a0,a2
1c012740:	cb0fb06f          	j	1c00dbf0 <mm_M_unroll_1x4>
1c012744:	8532                	mv	a0,a2
1c012746:	d5ff706f          	j	1c00a4a4 <mm_unroll_4x1>
1c01274a:	8532                	mv	a0,a2
1c01274c:	d59f706f          	j	1c00a4a4 <mm_unroll_4x1>
1c012750:	8532                	mv	a0,a2
1c012752:	abafa06f          	j	1c00ca0c <mm_M_u2>
1c012756:	8532                	mv	a0,a2
1c012758:	fe4fb06f          	j	1c00df3c <mm_M_unroll_1x8>
1c01275c:	8532                	mv	a0,a2
1c01275e:	9fef806f          	j	1c00a95c <mm_unroll_8x1>
1c012762:	8532                	mv	a0,a2
1c012764:	954fb06f          	j	1c00d8b8 <mm_M_unroll_1x2>
1c012768:	8532                	mv	a0,a2
1c01276a:	fd2fb06f          	j	1c00df3c <mm_M_unroll_1x8>
1c01276e:	8532                	mv	a0,a2
1c012770:	815fc06f          	j	1c00ef84 <mm_M_unroll_2x4>
1c012774:	8532                	mv	a0,a2
1c012776:	9e6f806f          	j	1c00a95c <mm_unroll_8x1>
1c01277a:	8532                	mv	a0,a2
1c01277c:	9e0f806f          	j	1c00a95c <mm_unroll_8x1>
1c012780:	8532                	mv	a0,a2
1c012782:	9daf806f          	j	1c00a95c <mm_unroll_8x1>
1c012786:	8532                	mv	a0,a2
1c012788:	865f806f          	j	1c00afec <mm_unroll_2x2>
1c01278c:	8532                	mv	a0,a2
1c01278e:	c62fb06f          	j	1c00dbf0 <mm_M_unroll_1x4>
1c012792:	8532                	mv	a0,a2
1c012794:	859f806f          	j	1c00afec <mm_unroll_2x2>
1c012798:	8532                	mv	a0,a2
1c01279a:	853f806f          	j	1c00afec <mm_unroll_2x2>
1c01279e:	8532                	mv	a0,a2
1c0127a0:	ce0fa06f          	j	1c00cc80 <mm_M_unroll_2x1>
1c0127a4:	8532                	mv	a0,a2
1c0127a6:	847f806f          	j	1c00afec <mm_unroll_2x2>
1c0127aa:	8532                	mv	a0,a2
1c0127ac:	95cfc06f          	j	1c00e908 <mm_M_unroll_4x2>
1c0127b0:	8532                	mv	a0,a2
1c0127b2:	906fb06f          	j	1c00d8b8 <mm_M_unroll_1x2>
1c0127b6:	8532                	mv	a0,a2
1c0127b8:	cc8fa06f          	j	1c00cc80 <mm_M_unroll_2x1>
1c0127bc:	8532                	mv	a0,a2
1c0127be:	c32fb06f          	j	1c00dbf0 <mm_M_unroll_1x4>
1c0127c2:	8532                	mv	a0,a2
1c0127c4:	f78fb06f          	j	1c00df3c <mm_M_unroll_1x8>
1c0127c8:	8532                	mv	a0,a2
1c0127ca:	ae6f906f          	j	1c00bab0 <mm_unroll_4x2>
1c0127ce:	8532                	mv	a0,a2
1c0127d0:	d21f806f          	j	1c00b4f0 <mm_unroll_2x4>
1c0127d4:	8532                	mv	a0,a2
1c0127d6:	f96fa06f          	j	1c00cf6c <mm_M_unroll_4x1>
1c0127da:	8532                	mv	a0,a2
1c0127dc:	f60fb06f          	j	1c00df3c <mm_M_unroll_1x8>
1c0127e0:	8532                	mv	a0,a2
1c0127e2:	f8afa06f          	j	1c00cf6c <mm_M_unroll_4x1>
1c0127e6:	8532                	mv	a0,a2
1c0127e8:	ac8f906f          	j	1c00bab0 <mm_unroll_4x2>
1c0127ec:	8532                	mv	a0,a2
1c0127ee:	d03f806f          	j	1c00b4f0 <mm_unroll_2x4>
1c0127f2:	8532                	mv	a0,a2
1c0127f4:	cfdf806f          	j	1c00b4f0 <mm_unroll_2x4>
1c0127f8:	8532                	mv	a0,a2
1c0127fa:	cf7f806f          	j	1c00b4f0 <mm_unroll_2x4>
1c0127fe:	8532                	mv	a0,a2
1c012800:	ab0f906f          	j	1c00bab0 <mm_unroll_4x2>
1c012804:	8532                	mv	a0,a2
1c012806:	b5ffa06f          	j	1c00d364 <mm_M_unroll_8x1>
1c01280a:	8532                	mv	a0,a2
1c01280c:	aa4f906f          	j	1c00bab0 <mm_unroll_4x2>
1c012810:	8532                	mv	a0,a2
1c012812:	99ff906f          	j	1c00c1b0 <mm_unroll_4x4>
1c012816:	8532                	mv	a0,a2
1c012818:	999f906f          	j	1c00c1b0 <mm_unroll_4x4>
1c01281c:	8532                	mv	a0,a2
1c01281e:	f4efa06f          	j	1c00cf6c <mm_M_unroll_4x1>
1c012822:	8532                	mv	a0,a2
1c012824:	c5cfa06f          	j	1c00cc80 <mm_M_unroll_2x1>
1c012828:	8532                	mv	a0,a2
1c01282a:	987f906f          	j	1c00c1b0 <mm_unroll_4x4>
1c01282e:	8532                	mv	a0,a2
1c012830:	981f906f          	j	1c00c1b0 <mm_unroll_4x4>
1c012834:	8532                	mv	a0,a2
1c012836:	beffb06f          	j	1c00e424 <mm_M_unroll_2x2>
1c01283a:	8532                	mv	a0,a2
1c01283c:	d15fc06f          	j	1c00f550 <mm_M_unroll_4x4>
1c012840:	8532                	mv	a0,a2
1c012842:	be3fb06f          	j	1c00e424 <mm_M_unroll_2x2>
1c012846:	8532                	mv	a0,a2
1c012848:	b1dfa06f          	j	1c00d364 <mm_M_unroll_8x1>
1c01284c:	8532                	mv	a0,a2
1c01284e:	f1efa06f          	j	1c00cf6c <mm_M_unroll_4x1>
1c012852:	8532                	mv	a0,a2
1c012854:	c2cfa06f          	j	1c00cc80 <mm_M_unroll_2x1>
1c012858:	8532                	mv	a0,a2
1c01285a:	9b2fa06f          	j	1c00ca0c <mm_M_u2>
1c01285e:	8532                	mv	a0,a2
1c012860:	9acfa06f          	j	1c00ca0c <mm_M_u2>
1c012864:	8532                	mv	a0,a2
1c012866:	b8afb06f          	j	1c00dbf0 <mm_M_unroll_1x4>
1c01286a:	8532                	mv	a0,a2
1c01286c:	84cfb06f          	j	1c00d8b8 <mm_M_unroll_1x2>
1c012870:	8532                	mv	a0,a2
1c012872:	99afa06f          	j	1c00ca0c <mm_M_u2>
1c012876:	8532                	mv	a0,a2
1c012878:	994fa06f          	j	1c00ca0c <mm_M_u2>
1c01287c:	8532                	mv	a0,a2
1c01287e:	b72fb06f          	j	1c00dbf0 <mm_M_unroll_1x4>
1c012882:	8532                	mv	a0,a2
1c012884:	834fb06f          	j	1c00d8b8 <mm_M_unroll_1x2>
1c012888:	8532                	mv	a0,a2
1c01288a:	b66fb06f          	j	1c00dbf0 <mm_M_unroll_1x4>
1c01288e:	8532                	mv	a0,a2
1c012890:	828fb06f          	j	1c00d8b8 <mm_M_unroll_1x2>
1c012894:	8532                	mv	a0,a2
1c012896:	b8ffb06f          	j	1c00e424 <mm_M_unroll_2x2>
1c01289a:	8532                	mv	a0,a2
1c01289c:	ac9fa06f          	j	1c00d364 <mm_M_unroll_8x1>
1c0128a0:	8532                	mv	a0,a2
1c0128a2:	866fc06f          	j	1c00e908 <mm_M_unroll_4x2>
1c0128a6:	8532                	mv	a0,a2
1c0128a8:	edcfc06f          	j	1c00ef84 <mm_M_unroll_2x4>
1c0128ac:	8532                	mv	a0,a2
1c0128ae:	b77fb06f          	j	1c00e424 <mm_M_unroll_2x2>
1c0128b2:	8532                	mv	a0,a2
1c0128b4:	ab1fa06f          	j	1c00d364 <mm_M_unroll_8x1>
1c0128b8:	8532                	mv	a0,a2
1c0128ba:	ecafc06f          	j	1c00ef84 <mm_M_unroll_2x4>
1c0128be:	8532                	mv	a0,a2
1c0128c0:	848fc06f          	j	1c00e908 <mm_M_unroll_4x2>
1c0128c4:	8532                	mv	a0,a2
1c0128c6:	b2afb06f          	j	1c00dbf0 <mm_M_unroll_1x4>
1c0128ca:	8532                	mv	a0,a2
1c0128cc:	fedfa06f          	j	1c00d8b8 <mm_M_unroll_1x2>
1c0128d0:	8532                	mv	a0,a2
1c0128d2:	eb2fc06f          	j	1c00ef84 <mm_M_unroll_2x4>
1c0128d6:	8532                	mv	a0,a2
1c0128d8:	b4dfb06f          	j	1c00e424 <mm_M_unroll_2x2>
1c0128dc:	8532                	mv	a0,a2
1c0128de:	a87fa06f          	j	1c00d364 <mm_M_unroll_8x1>
1c0128e2:	8532                	mv	a0,a2
1c0128e4:	e88fa06f          	j	1c00cf6c <mm_M_unroll_4x1>
1c0128e8:	8532                	mv	a0,a2
1c0128ea:	b96fa06f          	j	1c00cc80 <mm_M_unroll_2x1>
1c0128ee:	8532                	mv	a0,a2
1c0128f0:	e4cfb06f          	j	1c00df3c <mm_M_unroll_1x8>
1c0128f4:	8532                	mv	a0,a2
1c0128f6:	812fc06f          	j	1c00e908 <mm_M_unroll_4x2>
1c0128fa:	8532                	mv	a0,a2
1c0128fc:	e88fc06f          	j	1c00ef84 <mm_M_unroll_2x4>
1c012900:	8532                	mv	a0,a2
1c012902:	806fc06f          	j	1c00e908 <mm_M_unroll_4x2>
1c012906:	8532                	mv	a0,a2
1c012908:	e7cfc06f          	j	1c00ef84 <mm_M_unroll_2x4>
1c01290c:	8532                	mv	a0,a2
1c01290e:	e76fc06f          	j	1c00ef84 <mm_M_unroll_2x4>
1c012912:	8532                	mv	a0,a2
1c012914:	b11fb06f          	j	1c00e424 <mm_M_unroll_2x2>
1c012918:	8532                	mv	a0,a2
1c01291a:	b0bfb06f          	j	1c00e424 <mm_M_unroll_2x2>
1c01291e:	8532                	mv	a0,a2
1c012920:	a45fa06f          	j	1c00d364 <mm_M_unroll_8x1>
1c012924:	8532                	mv	a0,a2
1c012926:	e46fa06f          	j	1c00cf6c <mm_M_unroll_4x1>
1c01292a:	8532                	mv	a0,a2
1c01292c:	b54fa06f          	j	1c00cc80 <mm_M_unroll_2x1>
1c012930:	8532                	mv	a0,a2
1c012932:	e0afb06f          	j	1c00df3c <mm_M_unroll_1x8>
1c012936:	8532                	mv	a0,a2
1c012938:	a2dfa06f          	j	1c00d364 <mm_M_unroll_8x1>
1c01293c:	8532                	mv	a0,a2
1c01293e:	e2efa06f          	j	1c00cf6c <mm_M_unroll_4x1>
1c012942:	8532                	mv	a0,a2
1c012944:	b3cfa06f          	j	1c00cc80 <mm_M_unroll_2x1>
1c012948:	8532                	mv	a0,a2
1c01294a:	df2fb06f          	j	1c00df3c <mm_M_unroll_1x8>
1c01294e:	8532                	mv	a0,a2
1c012950:	ad5fb06f          	j	1c00e424 <mm_M_unroll_2x2>
1c012954:	8532                	mv	a0,a2
1c012956:	a0ffa06f          	j	1c00d364 <mm_M_unroll_8x1>
1c01295a:	8532                	mv	a0,a2
1c01295c:	e10fa06f          	j	1c00cf6c <mm_M_unroll_4x1>
1c012960:	8532                	mv	a0,a2
1c012962:	b1efa06f          	j	1c00cc80 <mm_M_unroll_2x1>
1c012966:	8532                	mv	a0,a2
1c012968:	dd4fb06f          	j	1c00df3c <mm_M_unroll_1x8>
1c01296c:	8532                	mv	a0,a2
1c01296e:	f9bfb06f          	j	1c00e908 <mm_M_unroll_4x2>
1c012972:	8532                	mv	a0,a2
1c012974:	f95fb06f          	j	1c00e908 <mm_M_unroll_4x2>
1c012978:	8532                	mv	a0,a2
1c01297a:	e0afc06f          	j	1c00ef84 <mm_M_unroll_2x4>
1c01297e:	8532                	mv	a0,a2
1c012980:	e04fc06f          	j	1c00ef84 <mm_M_unroll_2x4>
1c012984:	8532                	mv	a0,a2
1c012986:	f83fb06f          	j	1c00e908 <mm_M_unroll_4x2>
1c01298a:	8532                	mv	a0,a2
1c01298c:	f7dfb06f          	j	1c00e908 <mm_M_unroll_4x2>

1c012990 <pos_fll_set_freq>:
1c012990:	100596b3          	p.fl1	a3,a1
1c012994:	47f5                	li	a5,29
1c012996:	4705                	li	a4,1
1c012998:	82d7b7db          	p.subun	a5,a5,a3,1
1c01299c:	04f767b3          	p.max	a5,a4,a5
1c0129a0:	00400693          	li	a3,4
1c0129a4:	fff78813          	addi	a6,a5,-1
1c0129a8:	00f595b3          	sll	a1,a1,a5
1c0129ac:	00a68633          	add	a2,a3,a0
1c0129b0:	01071733          	sll	a4,a4,a6
1c0129b4:	00c64603          	lbu	a2,12(a2)
1c0129b8:	dc05b833          	p.bclr	a6,a1,14,0
1c0129bc:	c0f8275b          	p.addunr	a4,a6,a5
1c0129c0:	00251813          	slli	a6,a0,0x2
1c0129c4:	00e6e823          	p.sw	a4,a6(a3)
1c0129c8:	ce19                	beqz	a2,1c0129e6 <pos_fll_set_freq+0x56>
1c0129ca:	0512                	slli	a0,a0,0x4
1c0129cc:	0511                	addi	a0,a0,4
1c0129ce:	1a100637          	lui	a2,0x1a100
1c0129d2:	20a67683          	p.lw	a3,a0(a2)
1c0129d6:	81bd                	srli	a1,a1,0xf
1c0129d8:	de05a6b3          	p.insert	a3,a1,15,0
1c0129dc:	0785                	addi	a5,a5,1
1c0129de:	c7a7a6b3          	p.insert	a3,a5,3,26
1c0129e2:	00d66523          	p.sw	a3,a0(a2)
1c0129e6:	853a                	mv	a0,a4
1c0129e8:	8082                	ret

1c0129ea <pos_fll_init>:
1c0129ea:	00451593          	slli	a1,a0,0x4
1c0129ee:	832a                	mv	t1,a0
1c0129f0:	00458e13          	addi	t3,a1,4
1c0129f4:	1a100637          	lui	a2,0x1a100
1c0129f8:	21c67703          	p.lw	a4,t3(a2)
1c0129fc:	86ba                	mv	a3,a4
1c0129fe:	04074163          	bltz	a4,1c012a40 <pos_fll_init+0x56>
1c012a02:	00858793          	addi	a5,a1,8
1c012a06:	20f67883          	p.lw	a7,a5(a2)
1c012a0a:	4519                	li	a0,6
1c012a0c:	caa528b3          	p.insert	a7,a0,5,10
1c012a10:	05000513          	li	a0,80
1c012a14:	d70528b3          	p.insert	a7,a0,11,16
1c012a18:	1a100837          	lui	a6,0x1a100
1c012a1c:	011867a3          	p.sw	a7,a5(a6)
1c012a20:	05b1                	addi	a1,a1,12
1c012a22:	20b67603          	p.lw	a2,a1(a2)
1c012a26:	14c00793          	li	a5,332
1c012a2a:	d307a633          	p.insert	a2,a5,9,16
1c012a2e:	00c865a3          	p.sw	a2,a1(a6)
1c012a32:	4785                	li	a5,1
1c012a34:	c1e7a6b3          	p.insert	a3,a5,0,30
1c012a38:	c1f7a6b3          	p.insert	a3,a5,0,31
1c012a3c:	00d86e23          	p.sw	a3,t3(a6)
1c012a40:	00400793          	li	a5,4
1c012a44:	00231613          	slli	a2,t1,0x2
1c012a48:	963e                	add	a2,a2,a5
1c012a4a:	4208                	lw	a0,0(a2)
1c012a4c:	ed19                	bnez	a0,1c012a6a <pos_fll_init+0x80>
1c012a4e:	10075733          	p.exthz	a4,a4
1c012a52:	c7a696b3          	p.extractu	a3,a3,3,26
1c012a56:	073e                	slli	a4,a4,0xf
1c012a58:	16fd                	addi	a3,a3,-1
1c012a5a:	00d75533          	srl	a0,a4,a3
1c012a5e:	c208                	sw	a0,0(a2)
1c012a60:	979a                	add	a5,a5,t1
1c012a62:	4705                	li	a4,1
1c012a64:	00e78623          	sb	a4,12(a5)
1c012a68:	8082                	ret
1c012a6a:	46f5                	li	a3,29
1c012a6c:	10051733          	p.fl1	a4,a0
1c012a70:	4585                	li	a1,1
1c012a72:	82e6b75b          	p.subun	a4,a3,a4,1
1c012a76:	04e5e733          	p.max	a4,a1,a4
1c012a7a:	fff70893          	addi	a7,a4,-1
1c012a7e:	00e51833          	sll	a6,a0,a4
1c012a82:	006786b3          	add	a3,a5,t1
1c012a86:	011595b3          	sll	a1,a1,a7
1c012a8a:	00c6c683          	lbu	a3,12(a3)
1c012a8e:	dc0838b3          	p.bclr	a7,a6,14,0
1c012a92:	c0e8a5db          	p.addunr	a1,a7,a4
1c012a96:	c20c                	sw	a1,0(a2)
1c012a98:	d6e1                	beqz	a3,1c012a60 <pos_fll_init+0x76>
1c012a9a:	1a100637          	lui	a2,0x1a100
1c012a9e:	21c67683          	p.lw	a3,t3(a2)
1c012aa2:	00f85813          	srli	a6,a6,0xf
1c012aa6:	de0826b3          	p.insert	a3,a6,15,0
1c012aaa:	0705                	addi	a4,a4,1
1c012aac:	c7a726b3          	p.insert	a3,a4,3,26
1c012ab0:	00d66e23          	p.sw	a3,t3(a2)
1c012ab4:	979a                	add	a5,a5,t1
1c012ab6:	4705                	li	a4,1
1c012ab8:	00e78623          	sb	a4,12(a5)
1c012abc:	8082                	ret

1c012abe <pos_fll_constructor>:
1c012abe:	00400793          	li	a5,4
1c012ac2:	0007a023          	sw	zero,0(a5)
1c012ac6:	0007a223          	sw	zero,4(a5)
1c012aca:	0007a423          	sw	zero,8(a5)
1c012ace:	00079623          	sh	zero,12(a5)
1c012ad2:	00078723          	sb	zero,14(a5)
1c012ad6:	8082                	ret

1c012ad8 <pos_soc_init>:
1c012ad8:	1141                	addi	sp,sp,-16
1c012ada:	c606                	sw	ra,12(sp)
1c012adc:	c422                	sw	s0,8(sp)
1c012ade:	37c5                	jal	1c012abe <pos_fll_constructor>
1c012ae0:	4501                	li	a0,0
1c012ae2:	3721                	jal	1c0129ea <pos_fll_init>
1c012ae4:	00a02a23          	sw	a0,20(zero) # 14 <pos_freq_domains>
1c012ae8:	4505                	li	a0,1
1c012aea:	3701                	jal	1c0129ea <pos_fll_init>
1c012aec:	01400413          	li	s0,20
1c012af0:	c408                	sw	a0,8(s0)
1c012af2:	4509                	li	a0,2
1c012af4:	3ddd                	jal	1c0129ea <pos_fll_init>
1c012af6:	c048                	sw	a0,4(s0)
1c012af8:	40b2                	lw	ra,12(sp)
1c012afa:	4422                	lw	s0,8(sp)
1c012afc:	0141                	addi	sp,sp,16
1c012afe:	8082                	ret

1c012b00 <memset>:
1c012b00:	1141                	addi	sp,sp,-16
1c012b02:	c422                	sw	s0,8(sp)
1c012b04:	c606                	sw	ra,12(sp)
1c012b06:	842a                	mv	s0,a0
1c012b08:	c601                	beqz	a2,1c012b10 <memset+0x10>
1c012b0a:	0ff5f593          	andi	a1,a1,255
1c012b0e:	3fcd                	jal	1c012b00 <memset>
1c012b10:	8522                	mv	a0,s0
1c012b12:	40b2                	lw	ra,12(sp)
1c012b14:	4422                	lw	s0,8(sp)
1c012b16:	0141                	addi	sp,sp,16
1c012b18:	8082                	ret

1c012b1a <strchr>:
1c012b1a:	00054783          	lbu	a5,0(a0)
1c012b1e:	0ff5f593          	andi	a1,a1,255
1c012b22:	00b78c63          	beq	a5,a1,1c012b3a <strchr+0x20>
1c012b26:	cb99                	beqz	a5,1c012b3c <strchr+0x22>
1c012b28:	00150793          	addi	a5,a0,1
1c012b2c:	a011                	j	1c012b30 <strchr+0x16>
1c012b2e:	c719                	beqz	a4,1c012b3c <strchr+0x22>
1c012b30:	853e                	mv	a0,a5
1c012b32:	0017c70b          	p.lbu	a4,1(a5!)
1c012b36:	feb71ce3          	bne	a4,a1,1c012b2e <strchr+0x14>
1c012b3a:	8082                	ret
1c012b3c:	0015b593          	seqz	a1,a1
1c012b40:	40b005b3          	neg	a1,a1
1c012b44:	8d6d                	and	a0,a0,a1
1c012b46:	8082                	ret

1c012b48 <puts>:
1c012b48:	00054783          	lbu	a5,0(a0)
1c012b4c:	f14026f3          	csrr	a3,mhartid
1c012b50:	c395                	beqz	a5,1c012b74 <puts+0x2c>
1c012b52:	ca569733          	p.extractu	a4,a3,5,5
1c012b56:	f1402673          	csrr	a2,mhartid
1c012b5a:	071e                	slli	a4,a4,0x7
1c012b5c:	060e                	slli	a2,a2,0x3
1c012b5e:	1a10f5b7          	lui	a1,0x1a10f
1c012b62:	972e                	add	a4,a4,a1
1c012b64:	ee863633          	p.bclr	a2,a2,23,8
1c012b68:	0505                	addi	a0,a0,1
1c012b6a:	9732                	add	a4,a4,a2
1c012b6c:	c31c                	sw	a5,0(a4)
1c012b6e:	0015478b          	p.lbu	a5,1(a0!)
1c012b72:	ffed                	bnez	a5,1c012b6c <puts+0x24>
1c012b74:	ca5696b3          	p.extractu	a3,a3,5,5
1c012b78:	1a10f737          	lui	a4,0x1a10f
1c012b7c:	f14027f3          	csrr	a5,mhartid
1c012b80:	069e                	slli	a3,a3,0x7
1c012b82:	078e                	slli	a5,a5,0x3
1c012b84:	96ba                	add	a3,a3,a4
1c012b86:	ee87b7b3          	p.bclr	a5,a5,23,8
1c012b8a:	4729                	li	a4,10
1c012b8c:	00e6e7a3          	p.sw	a4,a5(a3)
1c012b90:	4501                	li	a0,0
1c012b92:	8082                	ret

1c012b94 <pos_libc_fputc_locked>:
1c012b94:	f14027f3          	csrr	a5,mhartid
1c012b98:	1a10f6b7          	lui	a3,0x1a10f
1c012b9c:	f1402773          	csrr	a4,mhartid
1c012ba0:	ca5797b3          	p.extractu	a5,a5,5,5
1c012ba4:	070e                	slli	a4,a4,0x3
1c012ba6:	079e                	slli	a5,a5,0x7
1c012ba8:	0ff57513          	andi	a0,a0,255
1c012bac:	ee873733          	p.bclr	a4,a4,23,8
1c012bb0:	97b6                	add	a5,a5,a3
1c012bb2:	00a767a3          	p.sw	a0,a5(a4)
1c012bb6:	4501                	li	a0,0
1c012bb8:	8082                	ret

1c012bba <putchar>:
1c012bba:	f14027f3          	csrr	a5,mhartid
1c012bbe:	1a10f6b7          	lui	a3,0x1a10f
1c012bc2:	f1402773          	csrr	a4,mhartid
1c012bc6:	ca5797b3          	p.extractu	a5,a5,5,5
1c012bca:	070e                	slli	a4,a4,0x3
1c012bcc:	079e                	slli	a5,a5,0x7
1c012bce:	0ff57513          	andi	a0,a0,255
1c012bd2:	ee873733          	p.bclr	a4,a4,23,8
1c012bd6:	97b6                	add	a5,a5,a3
1c012bd8:	00a767a3          	p.sw	a0,a5(a4)
1c012bdc:	4501                	li	a0,0
1c012bde:	8082                	ret

1c012be0 <pos_libc_prf_locked>:
1c012be0:	a095                	j	1c012c44 <pos_libc_prf>

1c012be2 <exit>:
1c012be2:	1141                	addi	sp,sp,-16
1c012be4:	c422                	sw	s0,8(sp)
1c012be6:	c606                	sw	ra,12(sp)
1c012be8:	842a                	mv	s0,a0
1c012bea:	3bf010ef          	jal	ra,1c0147a8 <pos_init_stop>
1c012bee:	c1f44533          	p.bset	a0,s0,0,31
1c012bf2:	1a1047b7          	lui	a5,0x1a104
1c012bf6:	0aa7a023          	sw	a0,160(a5) # 1a1040a0 <__l1_end+0xa1028d4>
1c012bfa:	1a10a7b7          	lui	a5,0x1a10a
1c012bfe:	577d                	li	a4,-1
1c012c00:	80e7a423          	sw	a4,-2040(a5) # 1a109808 <__l1_end+0xa10803c>
1c012c04:	10500073          	wfi
1c012c08:	10500073          	wfi
1c012c0c:	bfe5                	j	1c012c04 <exit+0x22>

1c012c0e <pos_io_start>:
1c012c0e:	4501                	li	a0,0
1c012c10:	8082                	ret

1c012c12 <pos_io_stop>:
1c012c12:	4501                	li	a0,0
1c012c14:	8082                	ret

1c012c16 <printf>:
1c012c16:	7139                	addi	sp,sp,-64
1c012c18:	02410313          	addi	t1,sp,36
1c012c1c:	d432                	sw	a2,40(sp)
1c012c1e:	862a                	mv	a2,a0
1c012c20:	1c013537          	lui	a0,0x1c013
1c012c24:	d22e                	sw	a1,36(sp)
1c012c26:	d636                	sw	a3,44(sp)
1c012c28:	4589                	li	a1,2
1c012c2a:	869a                	mv	a3,t1
1c012c2c:	b9450513          	addi	a0,a0,-1132 # 1c012b94 <pos_libc_fputc_locked>
1c012c30:	ce06                	sw	ra,28(sp)
1c012c32:	d83a                	sw	a4,48(sp)
1c012c34:	da3e                	sw	a5,52(sp)
1c012c36:	dc42                	sw	a6,56(sp)
1c012c38:	de46                	sw	a7,60(sp)
1c012c3a:	c61a                	sw	t1,12(sp)
1c012c3c:	3755                	jal	1c012be0 <pos_libc_prf_locked>
1c012c3e:	40f2                	lw	ra,28(sp)
1c012c40:	6121                	addi	sp,sp,64
1c012c42:	8082                	ret

1c012c44 <pos_libc_prf>:
1c012c44:	7175                	addi	sp,sp,-144
1c012c46:	1c0017b7          	lui	a5,0x1c001
1c012c4a:	1858                	addi	a4,sp,52
1c012c4c:	c522                	sw	s0,136(sp)
1c012c4e:	c14a                	sw	s2,128(sp)
1c012c50:	dece                	sw	s3,124(sp)
1c012c52:	d8da                	sw	s6,112(sp)
1c012c54:	d6de                	sw	s7,108(sp)
1c012c56:	d4e2                	sw	s8,104(sp)
1c012c58:	d0ea                	sw	s10,96(sp)
1c012c5a:	c706                	sw	ra,140(sp)
1c012c5c:	c326                	sw	s1,132(sp)
1c012c5e:	dcd2                	sw	s4,120(sp)
1c012c60:	dad6                	sw	s5,116(sp)
1c012c62:	d2e6                	sw	s9,100(sp)
1c012c64:	ceee                	sw	s11,92(sp)
1c012c66:	8baa                	mv	s7,a0
1c012c68:	8d2e                	mv	s10,a1
1c012c6a:	8932                	mv	s2,a2
1c012c6c:	89b6                	mv	s3,a3
1c012c6e:	4401                	li	s0,0
1c012c70:	1c001b37          	lui	s6,0x1c001
1c012c74:	c5878c13          	addi	s8,a5,-936 # 1c000c58 <__DTOR_END__+0x3f0>
1c012c78:	c43a                	sw	a4,8(sp)
1c012c7a:	a039                	j	1c012c88 <pos_libc_prf+0x44>
1c012c7c:	85ea                	mv	a1,s10
1c012c7e:	9b82                	jalr	s7
1c012c80:	07f52c63          	p.beqimm	a0,-1,1c012cf8 <pos_libc_prf+0xb4>
1c012c84:	0405                	addi	s0,s0,1
1c012c86:	896e                	mv	s2,s11
1c012c88:	00094503          	lbu	a0,0(s2)
1c012c8c:	00190d93          	addi	s11,s2,1
1c012c90:	c52d                	beqz	a0,1c012cfa <pos_libc_prf+0xb6>
1c012c92:	02500713          	li	a4,37
1c012c96:	fee513e3          	bne	a0,a4,1c012c7c <pos_libc_prf+0x38>
1c012c9a:	ce02                	sw	zero,28(sp)
1c012c9c:	cc02                	sw	zero,24(sp)
1c012c9e:	ca02                	sw	zero,20(sp)
1c012ca0:	c802                	sw	zero,16(sp)
1c012ca2:	c602                	sw	zero,12(sp)
1c012ca4:	03000493          	li	s1,48
1c012ca8:	000dca03          	lbu	s4,0(s11)
1c012cac:	001d8a93          	addi	s5,s11,1
1c012cb0:	f00b0513          	addi	a0,s6,-256 # 1c000f00 <__DTOR_END__+0x698>
1c012cb4:	85d2                	mv	a1,s4
1c012cb6:	8956                	mv	s2,s5
1c012cb8:	8cd2                	mv	s9,s4
1c012cba:	3585                	jal	1c012b1a <strchr>
1c012cbc:	c159                	beqz	a0,1c012d42 <pos_libc_prf+0xfe>
1c012cbe:	0744e063          	bltu	s1,s4,1c012d1e <pos_libc_prf+0xda>
1c012cc2:	0a0a                	slli	s4,s4,0x2
1c012cc4:	214c7a03          	p.lw	s4,s4(s8)
1c012cc8:	8a02                	jr	s4
1c012cca:	000dc503          	lbu	a0,0(s11)
1c012cce:	4725                	li	a4,9
1c012cd0:	fd050693          	addi	a3,a0,-48
1c012cd4:	6ed76ee3          	bltu	a4,a3,1c013bd0 <pos_libc_prf+0xf8c>
1c012cd8:	001d8c93          	addi	s9,s11,1
1c012cdc:	4aa5                	li	s5,9
1c012cde:	a809                	j	1c012cf0 <pos_libc_prf+0xac>
1c012ce0:	8de6                	mv	s11,s9
1c012ce2:	001cc50b          	p.lbu	a0,1(s9!)
1c012ce6:	19fd                	addi	s3,s3,-1
1c012ce8:	fd050793          	addi	a5,a0,-48
1c012cec:	6efae3e3          	bltu	s5,a5,1c013bd2 <pos_libc_prf+0xf8e>
1c012cf0:	85ea                	mv	a1,s10
1c012cf2:	9b82                	jalr	s7
1c012cf4:	fff536e3          	p.bneimm	a0,-1,1c012ce0 <pos_libc_prf+0x9c>
1c012cf8:	547d                	li	s0,-1
1c012cfa:	8522                	mv	a0,s0
1c012cfc:	40ba                	lw	ra,140(sp)
1c012cfe:	442a                	lw	s0,136(sp)
1c012d00:	449a                	lw	s1,132(sp)
1c012d02:	490a                	lw	s2,128(sp)
1c012d04:	59f6                	lw	s3,124(sp)
1c012d06:	5a66                	lw	s4,120(sp)
1c012d08:	5ad6                	lw	s5,116(sp)
1c012d0a:	5b46                	lw	s6,112(sp)
1c012d0c:	5bb6                	lw	s7,108(sp)
1c012d0e:	5c26                	lw	s8,104(sp)
1c012d10:	5c96                	lw	s9,100(sp)
1c012d12:	5d06                	lw	s10,96(sp)
1c012d14:	4df6                	lw	s11,92(sp)
1c012d16:	6149                	addi	sp,sp,144
1c012d18:	8082                	ret
1c012d1a:	4785                	li	a5,1
1c012d1c:	c83e                	sw	a5,16(sp)
1c012d1e:	8dd6                	mv	s11,s5
1c012d20:	b761                	j	1c012ca8 <pos_libc_prf+0x64>
1c012d22:	4785                	li	a5,1
1c012d24:	ca3e                	sw	a5,20(sp)
1c012d26:	8dd6                	mv	s11,s5
1c012d28:	b741                	j	1c012ca8 <pos_libc_prf+0x64>
1c012d2a:	4785                	li	a5,1
1c012d2c:	c63e                	sw	a5,12(sp)
1c012d2e:	8dd6                	mv	s11,s5
1c012d30:	bfa5                	j	1c012ca8 <pos_libc_prf+0x64>
1c012d32:	4705                	li	a4,1
1c012d34:	cc3a                	sw	a4,24(sp)
1c012d36:	8dd6                	mv	s11,s5
1c012d38:	bf85                	j	1c012ca8 <pos_libc_prf+0x64>
1c012d3a:	4705                	li	a4,1
1c012d3c:	ce3a                	sw	a4,28(sp)
1c012d3e:	8dd6                	mv	s11,s5
1c012d40:	b7a5                	j	1c012ca8 <pos_libc_prf+0x64>
1c012d42:	02a00613          	li	a2,42
1c012d46:	0cca0863          	beq	s4,a2,1c012e16 <pos_libc_prf+0x1d2>
1c012d4a:	fd0a0613          	addi	a2,s4,-48
1c012d4e:	45a5                	li	a1,9
1c012d50:	d002                	sw	zero,32(sp)
1c012d52:	4701                	li	a4,0
1c012d54:	04c5fd63          	bleu	a2,a1,1c012dae <pos_libc_prf+0x16a>
1c012d58:	02e00793          	li	a5,46
1c012d5c:	06fc8b63          	beq	s9,a5,1c012dd2 <pos_libc_prf+0x18e>
1c012d60:	8de6                	mv	s11,s9
1c012d62:	54fd                	li	s1,-1
1c012d64:	1c0016b7          	lui	a3,0x1c001
1c012d68:	f0868513          	addi	a0,a3,-248 # 1c000f08 <__DTOR_END__+0x6a0>
1c012d6c:	85e6                	mv	a1,s9
1c012d6e:	3375                	jal	1c012b1a <strchr>
1c012d70:	4681                	li	a3,0
1c012d72:	cd19                	beqz	a0,1c012d90 <pos_libc_prf+0x14c>
1c012d74:	06c00693          	li	a3,108
1c012d78:	00190613          	addi	a2,s2,1
1c012d7c:	00094c83          	lbu	s9,0(s2)
1c012d80:	62dd8463          	beq	s11,a3,1c0133a8 <pos_libc_prf+0x764>
1c012d84:	06800693          	li	a3,104
1c012d88:	62dd8463          	beq	s11,a3,1c0133b0 <pos_libc_prf+0x76c>
1c012d8c:	8932                	mv	s2,a2
1c012d8e:	86ee                	mv	a3,s11
1c012d90:	07800793          	li	a5,120
1c012d94:	4f97e163          	bltu	a5,s9,1c013276 <pos_libc_prf+0x632>
1c012d98:	1c001737          	lui	a4,0x1c001
1c012d9c:	002c9793          	slli	a5,s9,0x2
1c012da0:	d1c70713          	addi	a4,a4,-740 # 1c000d1c <__DTOR_END__+0x4b4>
1c012da4:	20f77783          	p.lw	a5,a5(a4)
1c012da8:	8782                	jr	a5
1c012daa:	8dd6                	mv	s11,s5
1c012dac:	0a85                	addi	s5,s5,1
1c012dae:	00271793          	slli	a5,a4,0x2
1c012db2:	000acc83          	lbu	s9,0(s5)
1c012db6:	97ba                	add	a5,a5,a4
1c012db8:	0786                	slli	a5,a5,0x1
1c012dba:	97d2                	add	a5,a5,s4
1c012dbc:	fd0c8693          	addi	a3,s9,-48
1c012dc0:	fd078713          	addi	a4,a5,-48
1c012dc4:	8a66                	mv	s4,s9
1c012dc6:	fed5f2e3          	bleu	a3,a1,1c012daa <pos_libc_prf+0x166>
1c012dca:	d03a                	sw	a4,32(sp)
1c012dcc:	002d8913          	addi	s2,s11,2
1c012dd0:	b761                	j	1c012d58 <pos_libc_prf+0x114>
1c012dd2:	00094c83          	lbu	s9,0(s2)
1c012dd6:	02a00793          	li	a5,42
1c012dda:	00190613          	addi	a2,s2,1
1c012dde:	04fc8763          	beq	s9,a5,1c012e2c <pos_libc_prf+0x1e8>
1c012de2:	fd0c8513          	addi	a0,s9,-48
1c012de6:	4825                	li	a6,9
1c012de8:	8de6                	mv	s11,s9
1c012dea:	86b2                	mv	a3,a2
1c012dec:	4481                	li	s1,0
1c012dee:	45a5                	li	a1,9
1c012df0:	8932                	mv	s2,a2
1c012df2:	f6a869e3          	bltu	a6,a0,1c012d64 <pos_libc_prf+0x120>
1c012df6:	00249793          	slli	a5,s1,0x2
1c012dfa:	97a6                	add	a5,a5,s1
1c012dfc:	0786                	slli	a5,a5,0x1
1c012dfe:	97e6                	add	a5,a5,s9
1c012e00:	0016cc8b          	p.lbu	s9,1(a3!)
1c012e04:	fd078493          	addi	s1,a5,-48
1c012e08:	fd0c8793          	addi	a5,s9,-48
1c012e0c:	8de6                	mv	s11,s9
1c012e0e:	fef5f4e3          	bleu	a5,a1,1c012df6 <pos_libc_prf+0x1b2>
1c012e12:	8936                	mv	s2,a3
1c012e14:	bf81                	j	1c012d64 <pos_libc_prf+0x120>
1c012e16:	0009a703          	lw	a4,0(s3)
1c012e1a:	0991                	addi	s3,s3,4
1c012e1c:	d03a                	sw	a4,32(sp)
1c012e1e:	00074f63          	bltz	a4,1c012e3c <pos_libc_prf+0x1f8>
1c012e22:	000acc83          	lbu	s9,0(s5)
1c012e26:	002d8913          	addi	s2,s11,2
1c012e2a:	b73d                	j	1c012d58 <pos_libc_prf+0x114>
1c012e2c:	00194c83          	lbu	s9,1(s2)
1c012e30:	0009a483          	lw	s1,0(s3)
1c012e34:	0909                	addi	s2,s2,2
1c012e36:	8de6                	mv	s11,s9
1c012e38:	0991                	addi	s3,s3,4
1c012e3a:	b72d                	j	1c012d64 <pos_libc_prf+0x120>
1c012e3c:	40e00733          	neg	a4,a4
1c012e40:	4785                	li	a5,1
1c012e42:	d03a                	sw	a4,32(sp)
1c012e44:	c83e                	sw	a5,16(sp)
1c012e46:	bff1                	j	1c012e22 <pos_libc_prf+0x1de>
1c012e48:	00498713          	addi	a4,s3,4
1c012e4c:	d63a                	sw	a4,44(sp)
1c012e4e:	0009ad83          	lw	s11,0(s3)
1c012e52:	3a04c8e3          	bltz	s1,1c013a02 <pos_libc_prf+0xdbe>
1c012e56:	5a0482e3          	beqz	s1,1c013bfa <pos_libc_prf+0xfb6>
1c012e5a:	000dc703          	lbu	a4,0(s11)
1c012e5e:	58070de3          	beqz	a4,1c013bf8 <pos_libc_prf+0xfb4>
1c012e62:	001d8713          	addi	a4,s11,1
1c012e66:	4981                	li	s3,0
1c012e68:	a029                	j	1c012e72 <pos_libc_prf+0x22e>
1c012e6a:	0017478b          	p.lbu	a5,1(a4!)
1c012e6e:	380782e3          	beqz	a5,1c0139f2 <pos_libc_prf+0xdae>
1c012e72:	0985                	addi	s3,s3,1
1c012e74:	fe99cbe3          	blt	s3,s1,1c012e6a <pos_libc_prf+0x226>
1c012e78:	5702                	lw	a4,32(sp)
1c012e7a:	4481                	li	s1,0
1c012e7c:	c602                	sw	zero,12(sp)
1c012e7e:	41370733          	sub	a4,a4,s3
1c012e82:	d03a                	sw	a4,32(sp)
1c012e84:	aa99                	j	1c012fda <pos_libc_prf+0x396>
1c012e86:	00498793          	addi	a5,s3,4
1c012e8a:	d63e                	sw	a5,44(sp)
1c012e8c:	0009a703          	lw	a4,0(s3)
1c012e90:	77e1                	lui	a5,0xffff8
1c012e92:	8307c793          	xori	a5,a5,-2000
1c012e96:	03610513          	addi	a0,sp,54
1c012e9a:	02f11a23          	sh	a5,52(sp)
1c012e9e:	4601                	li	a2,0
1c012ea0:	87aa                	mv	a5,a0
1c012ea2:	4825                	li	a6,9
1c012ea4:	01c61593          	slli	a1,a2,0x1c
1c012ea8:	f64736b3          	p.bclr	a3,a4,27,4
1c012eac:	8311                	srli	a4,a4,0x4
1c012eae:	8f4d                	or	a4,a4,a1
1c012eb0:	8211                	srli	a2,a2,0x4
1c012eb2:	3cd87f63          	bleu	a3,a6,1c013290 <pos_libc_prf+0x64c>
1c012eb6:	05768693          	addi	a3,a3,87
1c012eba:	00d78023          	sb	a3,0(a5) # ffff8000 <pulp__FC+0xffff8001>
1c012ebe:	00c765b3          	or	a1,a4,a2
1c012ec2:	00178693          	addi	a3,a5,1
1c012ec6:	3c058f63          	beqz	a1,1c0132a4 <pos_libc_prf+0x660>
1c012eca:	87b6                	mv	a5,a3
1c012ecc:	bfe1                	j	1c012ea4 <pos_libc_prf+0x260>
1c012ece:	04c00713          	li	a4,76
1c012ed2:	0049a78b          	p.lw	a5,4(s3!)
1c012ed6:	00e69463          	bne	a3,a4,1c012ede <pos_libc_prf+0x29a>
1c012eda:	6d20106f          	j	1c0145ac <pos_libc_prf+0x1968>
1c012ede:	44d755e3          	ble	a3,a4,1c013b28 <pos_libc_prf+0xee4>
1c012ee2:	06800713          	li	a4,104
1c012ee6:	42e68663          	beq	a3,a4,1c013312 <pos_libc_prf+0x6ce>
1c012eea:	c380                	sw	s0,0(a5)
1c012eec:	bb71                	j	1c012c88 <pos_libc_prf+0x44>
1c012eee:	06c00793          	li	a5,108
1c012ef2:	00f68a63          	beq	a3,a5,1c012f06 <pos_libc_prf+0x2c2>
1c012ef6:	07a00793          	li	a5,122
1c012efa:	00f68663          	beq	a3,a5,1c012f06 <pos_libc_prf+0x2c2>
1c012efe:	04c00793          	li	a5,76
1c012f02:	42f68ce3          	beq	a3,a5,1c013b3a <pos_libc_prf+0xef6>
1c012f06:	0009a783          	lw	a5,0(s3)
1c012f0a:	0991                	addi	s3,s3,4
1c012f0c:	d64e                	sw	s3,44(sp)
1c012f0e:	41f7da13          	srai	s4,a5,0x1f
1c012f12:	3e0a4ae3          	bltz	s4,1c013b06 <pos_libc_prf+0xec2>
1c012f16:	4752                	lw	a4,20(sp)
1c012f18:	3e0701e3          	beqz	a4,1c013afa <pos_libc_prf+0xeb6>
1c012f1c:	02b00713          	li	a4,43
1c012f20:	03510693          	addi	a3,sp,53
1c012f24:	02e10a23          	sb	a4,52(sp)
1c012f28:	c636                	sw	a3,12(sp)
1c012f2a:	8752                	mv	a4,s4
1c012f2c:	8cb6                	mv	s9,a3
1c012f2e:	89be                	mv	s3,a5
1c012f30:	8dba                	mv	s11,a4
1c012f32:	a011                	j	1c012f36 <pos_libc_prf+0x2f2>
1c012f34:	8cb6                	mv	s9,a3
1c012f36:	4629                	li	a2,10
1c012f38:	4681                	li	a3,0
1c012f3a:	854e                	mv	a0,s3
1c012f3c:	85ee                	mv	a1,s11
1c012f3e:	c9af50ef          	jal	ra,1c0083d8 <__umoddi3>
1c012f42:	8aaa                	mv	s5,a0
1c012f44:	85ee                	mv	a1,s11
1c012f46:	854e                	mv	a0,s3
1c012f48:	4681                	li	a3,0
1c012f4a:	4629                	li	a2,10
1c012f4c:	950f50ef          	jal	ra,1c00809c <__udivdi3>
1c012f50:	030a8a93          	addi	s5,s5,48
1c012f54:	015c8023          	sb	s5,0(s9)
1c012f58:	00a5e7b3          	or	a5,a1,a0
1c012f5c:	89aa                	mv	s3,a0
1c012f5e:	8dae                	mv	s11,a1
1c012f60:	001c8693          	addi	a3,s9,1
1c012f64:	fbe1                	bnez	a5,1c012f34 <pos_libc_prf+0x2f0>
1c012f66:	4732                	lw	a4,12(sp)
1c012f68:	00068023          	sb	zero,0(a3)
1c012f6c:	01977d63          	bleu	s9,a4,1c012f86 <pos_libc_prf+0x342>
1c012f70:	87e6                	mv	a5,s9
1c012f72:	00074583          	lbu	a1,0(a4)
1c012f76:	0007c603          	lbu	a2,0(a5)
1c012f7a:	feb78fab          	p.sb	a1,-1(a5!)
1c012f7e:	00c700ab          	p.sb	a2,1(a4!)
1c012f82:	fef768e3          	bltu	a4,a5,1c012f72 <pos_libc_prf+0x32e>
1c012f86:	47a2                	lw	a5,8(sp)
1c012f88:	4752                	lw	a4,20(sp)
1c012f8a:	40f689b3          	sub	s3,a3,a5
1c012f8e:	22071fe3          	bnez	a4,1c0139cc <pos_libc_prf+0xd88>
1c012f92:	47e2                	lw	a5,24(sp)
1c012f94:	22079ce3          	bnez	a5,1c0139cc <pos_libc_prf+0xd88>
1c012f98:	01fa5a13          	srli	s4,s4,0x1f
1c012f9c:	c652                	sw	s4,12(sp)
1c012f9e:	4004cf63          	bltz	s1,1c0133bc <pos_libc_prf+0x778>
1c012fa2:	47b2                	lw	a5,12(sp)
1c012fa4:	413484b3          	sub	s1,s1,s3
1c012fa8:	5702                	lw	a4,32(sp)
1c012faa:	94be                	add	s1,s1,a5
1c012fac:	0404e4b3          	p.max	s1,s1,zero
1c012fb0:	009987b3          	add	a5,s3,s1
1c012fb4:	8f1d                	sub	a4,a4,a5
1c012fb6:	d03a                	sw	a4,32(sp)
1c012fb8:	03410d93          	addi	s11,sp,52
1c012fbc:	a839                	j	1c012fda <pos_libc_prf+0x396>
1c012fbe:	0009a783          	lw	a5,0(s3)
1c012fc2:	5702                	lw	a4,32(sp)
1c012fc4:	0991                	addi	s3,s3,4
1c012fc6:	177d                	addi	a4,a4,-1
1c012fc8:	d64e                	sw	s3,44(sp)
1c012fca:	d03a                	sw	a4,32(sp)
1c012fcc:	02f10a23          	sb	a5,52(sp)
1c012fd0:	4481                	li	s1,0
1c012fd2:	c602                	sw	zero,12(sp)
1c012fd4:	4985                	li	s3,1
1c012fd6:	03410d93          	addi	s11,sp,52
1c012fda:	4a01                	li	s4,0
1c012fdc:	d402                	sw	zero,40(sp)
1c012fde:	d202                	sw	zero,36(sp)
1c012fe0:	47c2                	lw	a5,16(sp)
1c012fe2:	e39d                	bnez	a5,1c013008 <pos_libc_prf+0x3c4>
1c012fe4:	5702                	lw	a4,32(sp)
1c012fe6:	02e05163          	blez	a4,1c013008 <pos_libc_prf+0x3c4>
1c012fea:	fff70c93          	addi	s9,a4,-1
1c012fee:	85ea                	mv	a1,s10
1c012ff0:	02000513          	li	a0,32
1c012ff4:	9b82                	jalr	s7
1c012ff6:	d1f521e3          	p.beqimm	a0,-1,1c012cf8 <pos_libc_prf+0xb4>
1c012ffa:	1cfd                	addi	s9,s9,-1
1c012ffc:	fffcb9e3          	p.bneimm	s9,-1,1c012fee <pos_libc_prf+0x3aa>
1c013000:	5782                	lw	a5,32(sp)
1c013002:	577d                	li	a4,-1
1c013004:	d03a                	sw	a4,32(sp)
1c013006:	943e                	add	s0,s0,a5
1c013008:	47b2                	lw	a5,12(sp)
1c01300a:	00fd8cb3          	add	s9,s11,a5
1c01300e:	319d8663          	beq	s11,s9,1c01331a <pos_libc_prf+0x6d6>
1c013012:	001dc50b          	p.lbu	a0,1(s11!)
1c013016:	85ea                	mv	a1,s10
1c013018:	9b82                	jalr	s7
1c01301a:	fff53ae3          	p.bneimm	a0,-1,1c01300e <pos_libc_prf+0x3ca>
1c01301e:	547d                	li	s0,-1
1c013020:	b9e9                	j	1c012cfa <pos_libc_prf+0xb6>
1c013022:	00798793          	addi	a5,s3,7
1c013026:	c407b7b3          	p.bclr	a5,a5,2,0
1c01302a:	43cc                	lw	a1,4(a5)
1c01302c:	4394                	lw	a3,0(a5)
1c01302e:	07a1                	addi	a5,a5,8
1c013030:	d63e                	sw	a5,44(sp)
1c013032:	00b59613          	slli	a2,a1,0xb
1c013036:	0156d793          	srli	a5,a3,0x15
1c01303a:	0145d813          	srli	a6,a1,0x14
1c01303e:	8e5d                	or	a2,a2,a5
1c013040:	e8b83833          	p.bclr	a6,a6,20,11
1c013044:	06ae                	slli	a3,a3,0xb
1c013046:	c1f63633          	p.bclr	a2,a2,0,31
1c01304a:	2005cbe3          	bltz	a1,1c013a60 <pos_libc_prf+0xe1c>
1c01304e:	4752                	lw	a4,20(sp)
1c013050:	1e070be3          	beqz	a4,1c013a46 <pos_libc_prf+0xe02>
1c013054:	02b00793          	li	a5,43
1c013058:	02f10a23          	sb	a5,52(sp)
1c01305c:	03510e13          	addi	t3,sp,53
1c013060:	7ff00793          	li	a5,2047
1c013064:	1af805e3          	beq	a6,a5,1c013a0e <pos_libc_prf+0xdca>
1c013068:	04600793          	li	a5,70
1c01306c:	0dfcf293          	andi	t0,s9,223
1c013070:	44fc86e3          	beq	s9,a5,1c013cbc <pos_libc_prf+0x1078>
1c013074:	41f85593          	srai	a1,a6,0x1f
1c013078:	00d867b3          	or	a5,a6,a3
1c01307c:	8dd1                	or	a1,a1,a2
1c01307e:	8fcd                	or	a5,a5,a1
1c013080:	36078163          	beqz	a5,1c0133e2 <pos_libc_prf+0x79e>
1c013084:	02081363          	bnez	a6,1c0130aa <pos_libc_prf+0x466>
1c013088:	01f6d793          	srli	a5,a3,0x1f
1c01308c:	0606                	slli	a2,a2,0x1
1c01308e:	8e5d                	or	a2,a2,a5
1c013090:	0686                	slli	a3,a3,0x1
1c013092:	00065463          	bgez	a2,1c01309a <pos_libc_prf+0x456>
1c013096:	60a0106f          	j	1c0146a0 <pos_libc_prf+0x1a5c>
1c01309a:	01f6d793          	srli	a5,a3,0x1f
1c01309e:	0606                	slli	a2,a2,0x1
1c0130a0:	8e5d                	or	a2,a2,a5
1c0130a2:	187d                	addi	a6,a6,-1
1c0130a4:	0686                	slli	a3,a3,0x1
1c0130a6:	fe065ae3          	bgez	a2,1c01309a <pos_libc_prf+0x456>
1c0130aa:	c0280813          	addi	a6,a6,-1022 # 1a0ffc02 <__l1_end+0xa0fe436>
1c0130ae:	57f9                	li	a5,-2
1c0130b0:	c1f64633          	p.bset	a2,a2,0,31
1c0130b4:	4301                	li	t1,0
1c0130b6:	06f85863          	ble	a5,a6,1c013126 <pos_libc_prf+0x4e2>
1c0130ba:	333335b7          	lui	a1,0x33333
1c0130be:	80000f37          	lui	t5,0x80000
1c0130c2:	4301                	li	t1,0
1c0130c4:	33258593          	addi	a1,a1,818 # 33333332 <__l2_shared_end+0x1731ae26>
1c0130c8:	ffff4f13          	not	t5,t5
1c0130cc:	5ff9                	li	t6,-2
1c0130ce:	a011                	j	1c0130d2 <pos_libc_prf+0x48e>
1c0130d0:	883e                	mv	a6,a5
1c0130d2:	01f61513          	slli	a0,a2,0x1f
1c0130d6:	0016d713          	srli	a4,a3,0x1
1c0130da:	fc16b7b3          	p.bclr	a5,a3,30,1
1c0130de:	8f49                	or	a4,a4,a0
1c0130e0:	00e786b3          	add	a3,a5,a4
1c0130e4:	00f6b7b3          	sltu	a5,a3,a5
1c0130e8:	8205                	srli	a2,a2,0x1
1c0130ea:	963e                	add	a2,a2,a5
1c0130ec:	00180793          	addi	a5,a6,1
1c0130f0:	fec5e0e3          	bltu	a1,a2,1c0130d0 <pos_libc_prf+0x48c>
1c0130f4:	00269513          	slli	a0,a3,0x2
1c0130f8:	01e6d893          	srli	a7,a3,0x1e
1c0130fc:	00261713          	slli	a4,a2,0x2
1c013100:	96aa                	add	a3,a3,a0
1c013102:	00e8e733          	or	a4,a7,a4
1c013106:	00a6b533          	sltu	a0,a3,a0
1c01310a:	963a                	add	a2,a2,a4
1c01310c:	962a                	add	a2,a2,a0
1c01310e:	0809                	addi	a6,a6,2
1c013110:	137d                	addi	t1,t1,-1
1c013112:	01e61863          	bne	a2,t5,1c013122 <pos_libc_prf+0x4de>
1c013116:	01f6d613          	srli	a2,a3,0x1f
1c01311a:	fc164633          	p.bset	a2,a2,30,1
1c01311e:	0686                	slli	a3,a3,0x1
1c013120:	883e                	mv	a6,a5
1c013122:	fbf848e3          	blt	a6,t6,1c0130d2 <pos_libc_prf+0x48e>
1c013126:	80000537          	lui	a0,0x80000
1c01312a:	4895                	li	a7,5
1c01312c:	fff54513          	not	a0,a0
1c013130:	2b005d63          	blez	a6,1c0133ea <pos_libc_prf+0x7a6>
1c013134:	00268593          	addi	a1,a3,2
1c013138:	00d5b6b3          	sltu	a3,a1,a3
1c01313c:	9636                	add	a2,a2,a3
1c01313e:	031657b3          	divu	a5,a2,a7
1c013142:	0035d693          	srli	a3,a1,0x3
1c013146:	187d                	addi	a6,a6,-1
1c013148:	0305                	addi	t1,t1,1
1c01314a:	00279713          	slli	a4,a5,0x2
1c01314e:	973e                	add	a4,a4,a5
1c013150:	8e19                	sub	a2,a2,a4
1c013152:	01d61713          	slli	a4,a2,0x1d
1c013156:	8ed9                	or	a3,a3,a4
1c013158:	0316d633          	divu	a2,a3,a7
1c01315c:	00361693          	slli	a3,a2,0x3
1c013160:	00561713          	slli	a4,a2,0x5
1c013164:	9736                	add	a4,a4,a3
1c013166:	40e58733          	sub	a4,a1,a4
1c01316a:	03175733          	divu	a4,a4,a7
1c01316e:	8275                	srli	a2,a2,0x1d
1c013170:	97b2                	add	a5,a5,a2
1c013172:	96ba                	add	a3,a3,a4
1c013174:	00e6b733          	sltu	a4,a3,a4
1c013178:	00f70633          	add	a2,a4,a5
1c01317c:	01f6d793          	srli	a5,a3,0x1f
1c013180:	0606                	slli	a2,a2,0x1
1c013182:	8e5d                	or	a2,a2,a5
1c013184:	0686                	slli	a3,a3,0x1
1c013186:	187d                	addi	a6,a6,-1
1c013188:	fec57ae3          	bleu	a2,a0,1c01317c <pos_libc_prf+0x538>
1c01318c:	b755                	j	1c013130 <pos_libc_prf+0x4ec>
1c01318e:	06c00793          	li	a5,108
1c013192:	02f685e3          	beq	a3,a5,1c0139bc <pos_libc_prf+0xd78>
1c013196:	07a00793          	li	a5,122
1c01319a:	02f681e3          	beq	a3,a5,1c0139bc <pos_libc_prf+0xd78>
1c01319e:	04c00793          	li	a5,76
1c0131a2:	1ef681e3          	beq	a3,a5,1c013b84 <pos_libc_prf+0xf40>
1c0131a6:	0009ad83          	lw	s11,0(s3)
1c0131aa:	0991                	addi	s3,s3,4
1c0131ac:	d64e                	sw	s3,44(sp)
1c0131ae:	4981                	li	s3,0
1c0131b0:	06f00793          	li	a5,111
1c0131b4:	28fc8be3          	beq	s9,a5,1c013c4a <pos_libc_prf+0x1006>
1c0131b8:	07500793          	li	a5,117
1c0131bc:	03410a13          	addi	s4,sp,52
1c0131c0:	04fc8763          	beq	s9,a5,1c01320e <pos_libc_prf+0x5ca>
1c0131c4:	47b2                	lw	a5,12(sp)
1c0131c6:	180786e3          	beqz	a5,1c013b52 <pos_libc_prf+0xf0e>
1c0131ca:	77e1                	lui	a5,0xffff8
1c0131cc:	8307c793          	xori	a5,a5,-2000
1c0131d0:	02f11a23          	sh	a5,52(sp)
1c0131d4:	4509                	li	a0,2
1c0131d6:	03610693          	addi	a3,sp,54
1c0131da:	87b6                	mv	a5,a3
1c0131dc:	4825                	li	a6,9
1c0131de:	01c99613          	slli	a2,s3,0x1c
1c0131e2:	f64db733          	p.bclr	a4,s11,27,4
1c0131e6:	004ddd93          	srli	s11,s11,0x4
1c0131ea:	01b66db3          	or	s11,a2,s11
1c0131ee:	0049d993          	srli	s3,s3,0x4
1c0131f2:	0ce87e63          	bleu	a4,a6,1c0132ce <pos_libc_prf+0x68a>
1c0131f6:	05770713          	addi	a4,a4,87
1c0131fa:	00e78023          	sb	a4,0(a5) # ffff8000 <pulp__FC+0xffff8001>
1c0131fe:	013de733          	or	a4,s11,s3
1c013202:	00178613          	addi	a2,a5,1
1c013206:	cf69                	beqz	a4,1c0132e0 <pos_libc_prf+0x69c>
1c013208:	87b2                	mv	a5,a2
1c01320a:	bfd1                	j	1c0131de <pos_libc_prf+0x59a>
1c01320c:	8a3e                	mv	s4,a5
1c01320e:	4629                	li	a2,10
1c013210:	4681                	li	a3,0
1c013212:	856e                	mv	a0,s11
1c013214:	85ce                	mv	a1,s3
1c013216:	9c2f50ef          	jal	ra,1c0083d8 <__umoddi3>
1c01321a:	8aaa                	mv	s5,a0
1c01321c:	85ce                	mv	a1,s3
1c01321e:	856e                	mv	a0,s11
1c013220:	4629                	li	a2,10
1c013222:	4681                	li	a3,0
1c013224:	e79f40ef          	jal	ra,1c00809c <__udivdi3>
1c013228:	030a8a93          	addi	s5,s5,48
1c01322c:	015a0023          	sb	s5,0(s4)
1c013230:	00a5e733          	or	a4,a1,a0
1c013234:	8daa                	mv	s11,a0
1c013236:	89ae                	mv	s3,a1
1c013238:	001a0793          	addi	a5,s4,1
1c01323c:	fb61                	bnez	a4,1c01320c <pos_libc_prf+0x5c8>
1c01323e:	4722                	lw	a4,8(sp)
1c013240:	00078023          	sb	zero,0(a5)
1c013244:	40e789b3          	sub	s3,a5,a4
1c013248:	01477d63          	bleu	s4,a4,1c013262 <pos_libc_prf+0x61e>
1c01324c:	185c                	addi	a5,sp,52
1c01324e:	0007c683          	lbu	a3,0(a5)
1c013252:	000a4703          	lbu	a4,0(s4)
1c013256:	feda0fab          	p.sb	a3,-1(s4!)
1c01325a:	00e780ab          	p.sb	a4,1(a5!)
1c01325e:	ff47e8e3          	bltu	a5,s4,1c01324e <pos_libc_prf+0x60a>
1c013262:	c602                	sw	zero,12(sp)
1c013264:	bb2d                	j	1c012f9e <pos_libc_prf+0x35a>
1c013266:	85ea                	mv	a1,s10
1c013268:	02500513          	li	a0,37
1c01326c:	9b82                	jalr	s7
1c01326e:	a9f525e3          	p.beqimm	a0,-1,1c012cf8 <pos_libc_prf+0xb4>
1c013272:	0405                	addi	s0,s0,1
1c013274:	bc11                	j	1c012c88 <pos_libc_prf+0x44>
1c013276:	85ea                	mv	a1,s10
1c013278:	02500513          	li	a0,37
1c01327c:	9b82                	jalr	s7
1c01327e:	a7f52de3          	p.beqimm	a0,-1,1c012cf8 <pos_libc_prf+0xb4>
1c013282:	85ea                	mv	a1,s10
1c013284:	8566                	mv	a0,s9
1c013286:	9b82                	jalr	s7
1c013288:	a7f528e3          	p.beqimm	a0,-1,1c012cf8 <pos_libc_prf+0xb4>
1c01328c:	0409                	addi	s0,s0,2
1c01328e:	baed                	j	1c012c88 <pos_libc_prf+0x44>
1c013290:	03068693          	addi	a3,a3,48
1c013294:	00d78023          	sb	a3,0(a5)
1c013298:	00c765b3          	or	a1,a4,a2
1c01329c:	00178693          	addi	a3,a5,1
1c0132a0:	c20595e3          	bnez	a1,1c012eca <pos_libc_prf+0x286>
1c0132a4:	00068023          	sb	zero,0(a3)
1c0132a8:	8e89                	sub	a3,a3,a0
1c0132aa:	00f57d63          	bleu	a5,a0,1c0132c4 <pos_libc_prf+0x680>
1c0132ae:	872a                	mv	a4,a0
1c0132b0:	00074583          	lbu	a1,0(a4)
1c0132b4:	0007c603          	lbu	a2,0(a5)
1c0132b8:	feb78fab          	p.sb	a1,-1(a5!)
1c0132bc:	00c700ab          	p.sb	a2,1(a4!)
1c0132c0:	fef768e3          	bltu	a4,a5,1c0132b0 <pos_libc_prf+0x66c>
1c0132c4:	00268993          	addi	s3,a3,2
1c0132c8:	4689                	li	a3,2
1c0132ca:	c636                	sw	a3,12(sp)
1c0132cc:	b9c9                	j	1c012f9e <pos_libc_prf+0x35a>
1c0132ce:	03070713          	addi	a4,a4,48
1c0132d2:	00e78023          	sb	a4,0(a5)
1c0132d6:	013de733          	or	a4,s11,s3
1c0132da:	00178613          	addi	a2,a5,1
1c0132de:	f70d                	bnez	a4,1c013208 <pos_libc_prf+0x5c4>
1c0132e0:	00060023          	sb	zero,0(a2) # 1a100000 <__l1_end+0xa0fe834>
1c0132e4:	8e15                	sub	a2,a2,a3
1c0132e6:	00f6fc63          	bleu	a5,a3,1c0132fe <pos_libc_prf+0x6ba>
1c0132ea:	0006c583          	lbu	a1,0(a3)
1c0132ee:	0007c703          	lbu	a4,0(a5)
1c0132f2:	feb78fab          	p.sb	a1,-1(a5!)
1c0132f6:	00e680ab          	p.sb	a4,1(a3!)
1c0132fa:	fef6e8e3          	bltu	a3,a5,1c0132ea <pos_libc_prf+0x6a6>
1c0132fe:	05800793          	li	a5,88
1c013302:	10fc80e3          	beq	s9,a5,1c013c02 <pos_libc_prf+0xfbe>
1c013306:	46b2                	lw	a3,12(sp)
1c013308:	00a609b3          	add	s3,a2,a0
1c01330c:	0686                	slli	a3,a3,0x1
1c01330e:	c636                	sw	a3,12(sp)
1c013310:	b179                	j	1c012f9e <pos_libc_prf+0x35a>
1c013312:	8722                	mv	a4,s0
1c013314:	00e79023          	sh	a4,0(a5)
1c013318:	ba85                	j	1c012c88 <pos_libc_prf+0x44>
1c01331a:	8ca6                	mv	s9,s1
1c01331c:	a039                	j	1c01332a <pos_libc_prf+0x6e6>
1c01331e:	85ea                	mv	a1,s10
1c013320:	03000513          	li	a0,48
1c013324:	9b82                	jalr	s7
1c013326:	9df529e3          	p.beqimm	a0,-1,1c012cf8 <pos_libc_prf+0xb4>
1c01332a:	1cfd                	addi	s9,s9,-1
1c01332c:	fffcb9e3          	p.bneimm	s9,-1,1c01331e <pos_libc_prf+0x6da>
1c013330:	46b2                	lw	a3,12(sp)
1c013332:	5712                	lw	a4,36(sp)
1c013334:	40d989b3          	sub	s3,s3,a3
1c013338:	c84e                	sw	s3,16(sp)
1c01333a:	980718e3          	bnez	a4,1c012cca <pos_libc_prf+0x86>
1c01333e:	56a2                	lw	a3,40(sp)
1c013340:	e2b9                	bnez	a3,1c013386 <pos_libc_prf+0x742>
1c013342:	680a0b63          	beqz	s4,1c0139d8 <pos_libc_prf+0xd94>
1c013346:	000dc503          	lbu	a0,0(s11)
1c01334a:	4ca5                	li	s9,9
1c01334c:	02e00a93          	li	s5,46
1c013350:	a809                	j	1c013362 <pos_libc_prf+0x71e>
1c013352:	85ea                	mv	a1,s10
1c013354:	9b82                	jalr	s7
1c013356:	9bf521e3          	p.beqimm	a0,-1,1c012cf8 <pos_libc_prf+0xb4>
1c01335a:	0d85                	addi	s11,s11,1
1c01335c:	000dc503          	lbu	a0,0(s11)
1c013360:	19fd                	addi	s3,s3,-1
1c013362:	fd050793          	addi	a5,a0,-48 # 7fffffd0 <pulp__FC+0x7fffffd1>
1c013366:	fefcf6e3          	bleu	a5,s9,1c013352 <pos_libc_prf+0x70e>
1c01336a:	ff5504e3          	beq	a0,s5,1c013352 <pos_libc_prf+0x70e>
1c01336e:	8cd2                	mv	s9,s4
1c013370:	67905263          	blez	s9,1c0139d4 <pos_libc_prf+0xd90>
1c013374:	85ea                	mv	a1,s10
1c013376:	03000513          	li	a0,48
1c01337a:	9b82                	jalr	s7
1c01337c:	1cfd                	addi	s9,s9,-1
1c01337e:	fff539e3          	p.bneimm	a0,-1,1c013370 <pos_libc_prf+0x72c>
1c013382:	547d                	li	s0,-1
1c013384:	ba9d                	j	1c012cfa <pos_libc_prf+0xb6>
1c013386:	02e00c93          	li	s9,46
1c01338a:	a031                	j	1c013396 <pos_libc_prf+0x752>
1c01338c:	19fd                	addi	s3,s3,-1
1c01338e:	019a9463          	bne	s5,s9,1c013396 <pos_libc_prf+0x752>
1c013392:	1f00106f          	j	1c014582 <pos_libc_prf+0x193e>
1c013396:	001dca8b          	p.lbu	s5,1(s11!)
1c01339a:	85ea                	mv	a1,s10
1c01339c:	8556                	mv	a0,s5
1c01339e:	9b82                	jalr	s7
1c0133a0:	fff536e3          	p.bneimm	a0,-1,1c01338c <pos_libc_prf+0x748>
1c0133a4:	547d                	li	s0,-1
1c0133a6:	ba91                	j	1c012cfa <pos_libc_prf+0xb6>
1c0133a8:	5fbc8763          	beq	s9,s11,1c013996 <pos_libc_prf+0xd52>
1c0133ac:	8932                	mv	s2,a2
1c0133ae:	b2cd                	j	1c012d90 <pos_libc_prf+0x14c>
1c0133b0:	5fbc8a63          	beq	s9,s11,1c0139a4 <pos_libc_prf+0xd60>
1c0133b4:	8932                	mv	s2,a2
1c0133b6:	06800693          	li	a3,104
1c0133ba:	bad9                	j	1c012d90 <pos_libc_prf+0x14c>
1c0133bc:	4a01                	li	s4,0
1c0133be:	d402                	sw	zero,40(sp)
1c0133c0:	d202                	sw	zero,36(sp)
1c0133c2:	47f2                	lw	a5,28(sp)
1c0133c4:	5c078063          	beqz	a5,1c013984 <pos_libc_prf+0xd40>
1c0133c8:	5782                	lw	a5,32(sp)
1c0133ca:	5702                	lw	a4,32(sp)
1c0133cc:	03410d93          	addi	s11,sp,52
1c0133d0:	413784b3          	sub	s1,a5,s3
1c0133d4:	0404e4b3          	p.max	s1,s1,zero
1c0133d8:	009987b3          	add	a5,s3,s1
1c0133dc:	8f1d                	sub	a4,a4,a5
1c0133de:	d03a                	sw	a4,32(sp)
1c0133e0:	b101                	j	1c012fe0 <pos_libc_prf+0x39c>
1c0133e2:	4301                	li	t1,0
1c0133e4:	4801                	li	a6,0
1c0133e6:	4681                	li	a3,0
1c0133e8:	4601                	li	a2,0
1c0133ea:	4591                	li	a1,4
1c0133ec:	410585b3          	sub	a1,a1,a6
1c0133f0:	0125c0fb          	lp.setup	x1,a1,1c013414 <pos_libc_prf+0x7d0>
1c0133f4:	01f61713          	slli	a4,a2,0x1f
1c0133f8:	0016d793          	srli	a5,a3,0x1
1c0133fc:	8fd9                	or	a5,a5,a4
1c0133fe:	fc16b6b3          	p.bclr	a3,a3,30,1
1c013402:	00f688b3          	add	a7,a3,a5
1c013406:	8205                	srli	a2,a2,0x1
1c013408:	00d8b6b3          	sltu	a3,a7,a3
1c01340c:	00c687b3          	add	a5,a3,a2
1c013410:	863e                	mv	a2,a5
1c013412:	86c6                	mv	a3,a7
1c013414:	0805                	addi	a6,a6,1
1c013416:	0004d363          	bgez	s1,1c01341c <pos_libc_prf+0x7d8>
1c01341a:	4499                	li	s1,6
1c01341c:	04700693          	li	a3,71
1c013420:	64d28863          	beq	t0,a3,1c013a70 <pos_libc_prf+0xe2c>
1c013424:	06600693          	li	a3,102
1c013428:	4281                	li	t0,0
1c01342a:	66dc8163          	beq	s9,a3,1c013a8c <pos_libc_prf+0xe48>
1c01342e:	00148e93          	addi	t4,s1,1
1c013432:	4741                	li	a4,16
1c013434:	04eeceb3          	p.min	t4,t4,a4
1c013438:	1efd                	addi	t4,t4,-1
1c01343a:	4681                	li	a3,0
1c01343c:	08000637          	lui	a2,0x8000
1c013440:	4f15                	li	t5,5
1c013442:	0e85                	addi	t4,t4,1
1c013444:	02eec0fb          	lp.setup	x1,t4,1c0134a0 <pos_libc_prf+0x85c>
1c013448:	00268593          	addi	a1,a3,2
1c01344c:	00d5b733          	sltu	a4,a1,a3
1c013450:	9732                	add	a4,a4,a2
1c013452:	03e75533          	divu	a0,a4,t5
1c013456:	0035d613          	srli	a2,a1,0x3
1c01345a:	00251693          	slli	a3,a0,0x2
1c01345e:	96aa                	add	a3,a3,a0
1c013460:	8f15                	sub	a4,a4,a3
1c013462:	0776                	slli	a4,a4,0x1d
1c013464:	8e59                	or	a2,a2,a4
1c013466:	03e65633          	divu	a2,a2,t5
1c01346a:	00361693          	slli	a3,a2,0x3
1c01346e:	00561713          	slli	a4,a2,0x5
1c013472:	9736                	add	a4,a4,a3
1c013474:	40e58733          	sub	a4,a1,a4
1c013478:	03e75733          	divu	a4,a4,t5
1c01347c:	8275                	srli	a2,a2,0x1d
1c01347e:	962a                	add	a2,a2,a0
1c013480:	96ba                	add	a3,a3,a4
1c013482:	00e6b733          	sltu	a4,a3,a4
1c013486:	963a                	add	a2,a2,a4
1c013488:	01f61513          	slli	a0,a2,0x1f
1c01348c:	0016d713          	srli	a4,a3,0x1
1c013490:	fc16b5b3          	p.bclr	a1,a3,30,1
1c013494:	8f49                	or	a4,a4,a0
1c013496:	00e586b3          	add	a3,a1,a4
1c01349a:	8205                	srli	a2,a2,0x1
1c01349c:	00b6b733          	sltu	a4,a3,a1
1c0134a0:	963a                	add	a2,a2,a4
1c0134a2:	96c6                	add	a3,a3,a7
1c0134a4:	963e                	add	a2,a2,a5
1c0134a6:	0116b533          	sltu	a0,a3,a7
1c0134aa:	9532                	add	a0,a0,a2
1c0134ac:	f6053633          	p.bclr	a2,a0,27,0
1c0134b0:	87b6                	mv	a5,a3
1c0134b2:	c22d                	beqz	a2,1c013514 <pos_libc_prf+0x8d0>
1c0134b4:	00268593          	addi	a1,a3,2
1c0134b8:	00d5b6b3          	sltu	a3,a1,a3
1c0134bc:	9536                	add	a0,a0,a3
1c0134be:	4e95                	li	t4,5
1c0134c0:	03d558b3          	divu	a7,a0,t4
1c0134c4:	0035d793          	srli	a5,a1,0x3
1c0134c8:	0305                	addi	t1,t1,1
1c0134ca:	00289693          	slli	a3,a7,0x2
1c0134ce:	96c6                	add	a3,a3,a7
1c0134d0:	8d15                	sub	a0,a0,a3
1c0134d2:	01d51693          	slli	a3,a0,0x1d
1c0134d6:	8fd5                	or	a5,a5,a3
1c0134d8:	03d7d6b3          	divu	a3,a5,t4
1c0134dc:	00369613          	slli	a2,a3,0x3
1c0134e0:	00569793          	slli	a5,a3,0x5
1c0134e4:	97b2                	add	a5,a5,a2
1c0134e6:	40f587b3          	sub	a5,a1,a5
1c0134ea:	03d7d7b3          	divu	a5,a5,t4
1c0134ee:	82f5                	srli	a3,a3,0x1d
1c0134f0:	96c6                	add	a3,a3,a7
1c0134f2:	963e                	add	a2,a2,a5
1c0134f4:	00f637b3          	sltu	a5,a2,a5
1c0134f8:	96be                	add	a3,a3,a5
1c0134fa:	01f69593          	slli	a1,a3,0x1f
1c0134fe:	00165793          	srli	a5,a2,0x1
1c013502:	8fcd                	or	a5,a5,a1
1c013504:	fc163633          	p.bclr	a2,a2,30,1
1c013508:	97b2                	add	a5,a5,a2
1c01350a:	8285                	srli	a3,a3,0x1
1c01350c:	00c7b633          	sltu	a2,a5,a2
1c013510:	00d60533          	add	a0,a2,a3
1c013514:	06600613          	li	a2,102
1c013518:	001e0693          	addi	a3,t3,1
1c01351c:	7ccc8163          	beq	s9,a2,1c013cde <pos_libc_prf+0x109a>
1c013520:	00279613          	slli	a2,a5,0x2
1c013524:	01e7d593          	srli	a1,a5,0x1e
1c013528:	00251813          	slli	a6,a0,0x2
1c01352c:	97b2                	add	a5,a5,a2
1c01352e:	0105e833          	or	a6,a1,a6
1c013532:	9542                	add	a0,a0,a6
1c013534:	00c7b5b3          	sltu	a1,a5,a2
1c013538:	95aa                	add	a1,a1,a0
1c01353a:	01f7d613          	srli	a2,a5,0x1f
1c01353e:	0586                	slli	a1,a1,0x1
1c013540:	8dd1                	or	a1,a1,a2
1c013542:	01c5d613          	srli	a2,a1,0x1c
1c013546:	8532                	mv	a0,a2
1c013548:	03060613          	addi	a2,a2,48 # 8000030 <__l1_heap_size+0x7ff17fc>
1c01354c:	00ce0023          	sb	a2,0(t3)
1c013550:	00179613          	slli	a2,a5,0x1
1c013554:	47b2                	lw	a5,12(sp)
1c013556:	00a03533          	snez	a0,a0
1c01355a:	c7c5b5b3          	p.bclr	a1,a1,3,28
1c01355e:	40a30333          	sub	t1,t1,a0
1c013562:	76079363          	bnez	a5,1c013cc8 <pos_libc_prf+0x1084>
1c013566:	8a26                	mv	s4,s1
1c013568:	3a905d63          	blez	s1,1c013922 <pos_libc_prf+0xcde>
1c01356c:	02e00693          	li	a3,46
1c013570:	002e0793          	addi	a5,t3,2
1c013574:	00de00a3          	sb	a3,1(t3)
1c013578:	00261693          	slli	a3,a2,0x2
1c01357c:	01e65513          	srli	a0,a2,0x1e
1c013580:	00259813          	slli	a6,a1,0x2
1c013584:	9636                	add	a2,a2,a3
1c013586:	01056833          	or	a6,a0,a6
1c01358a:	95c2                	add	a1,a1,a6
1c01358c:	00d63533          	sltu	a0,a2,a3
1c013590:	952e                	add	a0,a0,a1
1c013592:	01f65693          	srli	a3,a2,0x1f
1c013596:	0506                	slli	a0,a0,0x1
1c013598:	8d55                	or	a0,a0,a3
1c01359a:	01c55813          	srli	a6,a0,0x1c
1c01359e:	100006b7          	lui	a3,0x10000
1c0135a2:	03080813          	addi	a6,a6,48
1c0135a6:	fff68593          	addi	a1,a3,-1 # fffffff <__l1_heap_size+0xfff17cb>
1c0135aa:	01078023          	sb	a6,0(a5)
1c0135ae:	00161893          	slli	a7,a2,0x1
1c0135b2:	8d6d                	and	a0,a0,a1
1c0135b4:	00178693          	addi	a3,a5,1
1c0135b8:	4a01                	li	s4,0
1c0135ba:	3614a463          	p.beqimm	s1,1,1c013922 <pos_libc_prf+0xcde>
1c0135be:	00361693          	slli	a3,a2,0x3
1c0135c2:	00251813          	slli	a6,a0,0x2
1c0135c6:	01e8de13          	srli	t3,a7,0x1e
1c0135ca:	01168633          	add	a2,a3,a7
1c0135ce:	010e6833          	or	a6,t3,a6
1c0135d2:	9542                	add	a0,a0,a6
1c0135d4:	00d636b3          	sltu	a3,a2,a3
1c0135d8:	96aa                	add	a3,a3,a0
1c0135da:	0686                	slli	a3,a3,0x1
1c0135dc:	01f65513          	srli	a0,a2,0x1f
1c0135e0:	8ec9                	or	a3,a3,a0
1c0135e2:	01c6d513          	srli	a0,a3,0x1c
1c0135e6:	03050513          	addi	a0,a0,48
1c0135ea:	00a780a3          	sb	a0,1(a5)
1c0135ee:	ffe48a13          	addi	s4,s1,-2
1c0135f2:	00161513          	slli	a0,a2,0x1
1c0135f6:	8eed                	and	a3,a3,a1
1c0135f8:	01404463          	bgtz	s4,1c013600 <pos_libc_prf+0x9bc>
1c0135fc:	0f40106f          	j	1c0146f0 <pos_libc_prf+0x1aac>
1c013600:	060e                	slli	a2,a2,0x3
1c013602:	01e55893          	srli	a7,a0,0x1e
1c013606:	00269813          	slli	a6,a3,0x2
1c01360a:	9532                	add	a0,a0,a2
1c01360c:	0108e833          	or	a6,a7,a6
1c013610:	96c2                	add	a3,a3,a6
1c013612:	00c53633          	sltu	a2,a0,a2
1c013616:	9636                	add	a2,a2,a3
1c013618:	0606                	slli	a2,a2,0x1
1c01361a:	01f55693          	srli	a3,a0,0x1f
1c01361e:	8e55                	or	a2,a2,a3
1c013620:	01c65693          	srli	a3,a2,0x1c
1c013624:	03068693          	addi	a3,a3,48
1c013628:	00d78123          	sb	a3,2(a5)
1c01362c:	ffd48a13          	addi	s4,s1,-3
1c013630:	00151813          	slli	a6,a0,0x1
1c013634:	8e6d                	and	a2,a2,a1
1c013636:	00378693          	addi	a3,a5,3
1c01363a:	2f405463          	blez	s4,1c013922 <pos_libc_prf+0xcde>
1c01363e:	00351693          	slli	a3,a0,0x3
1c013642:	01e85893          	srli	a7,a6,0x1e
1c013646:	00261513          	slli	a0,a2,0x2
1c01364a:	9836                	add	a6,a6,a3
1c01364c:	00a8e533          	or	a0,a7,a0
1c013650:	962a                	add	a2,a2,a0
1c013652:	00d836b3          	sltu	a3,a6,a3
1c013656:	96b2                	add	a3,a3,a2
1c013658:	0686                	slli	a3,a3,0x1
1c01365a:	01f85613          	srli	a2,a6,0x1f
1c01365e:	8ed1                	or	a3,a3,a2
1c013660:	01c6d613          	srli	a2,a3,0x1c
1c013664:	03060613          	addi	a2,a2,48
1c013668:	00c781a3          	sb	a2,3(a5)
1c01366c:	ffc48a13          	addi	s4,s1,-4
1c013670:	00181513          	slli	a0,a6,0x1
1c013674:	8eed                	and	a3,a3,a1
1c013676:	01404463          	bgtz	s4,1c01367e <pos_libc_prf+0xa3a>
1c01367a:	06e0106f          	j	1c0146e8 <pos_libc_prf+0x1aa4>
1c01367e:	00381613          	slli	a2,a6,0x3
1c013682:	01e55893          	srli	a7,a0,0x1e
1c013686:	00269813          	slli	a6,a3,0x2
1c01368a:	9532                	add	a0,a0,a2
1c01368c:	0108e833          	or	a6,a7,a6
1c013690:	96c2                	add	a3,a3,a6
1c013692:	00c53633          	sltu	a2,a0,a2
1c013696:	9636                	add	a2,a2,a3
1c013698:	0606                	slli	a2,a2,0x1
1c01369a:	01f55693          	srli	a3,a0,0x1f
1c01369e:	8e55                	or	a2,a2,a3
1c0136a0:	01c65693          	srli	a3,a2,0x1c
1c0136a4:	03068693          	addi	a3,a3,48
1c0136a8:	00d78223          	sb	a3,4(a5)
1c0136ac:	ffb48a13          	addi	s4,s1,-5
1c0136b0:	00151813          	slli	a6,a0,0x1
1c0136b4:	8e6d                	and	a2,a2,a1
1c0136b6:	00578693          	addi	a3,a5,5
1c0136ba:	27405463          	blez	s4,1c013922 <pos_libc_prf+0xcde>
1c0136be:	00351693          	slli	a3,a0,0x3
1c0136c2:	01e85893          	srli	a7,a6,0x1e
1c0136c6:	00261513          	slli	a0,a2,0x2
1c0136ca:	9836                	add	a6,a6,a3
1c0136cc:	00a8e533          	or	a0,a7,a0
1c0136d0:	962a                	add	a2,a2,a0
1c0136d2:	00d836b3          	sltu	a3,a6,a3
1c0136d6:	96b2                	add	a3,a3,a2
1c0136d8:	0686                	slli	a3,a3,0x1
1c0136da:	01f85613          	srli	a2,a6,0x1f
1c0136de:	8ed1                	or	a3,a3,a2
1c0136e0:	01c6d613          	srli	a2,a3,0x1c
1c0136e4:	03060613          	addi	a2,a2,48
1c0136e8:	00c782a3          	sb	a2,5(a5)
1c0136ec:	ffa48a13          	addi	s4,s1,-6
1c0136f0:	00181513          	slli	a0,a6,0x1
1c0136f4:	8eed                	and	a3,a3,a1
1c0136f6:	7f4055e3          	blez	s4,1c0146e0 <pos_libc_prf+0x1a9c>
1c0136fa:	00381613          	slli	a2,a6,0x3
1c0136fe:	01e55893          	srli	a7,a0,0x1e
1c013702:	00269813          	slli	a6,a3,0x2
1c013706:	9532                	add	a0,a0,a2
1c013708:	0108e833          	or	a6,a7,a6
1c01370c:	96c2                	add	a3,a3,a6
1c01370e:	00c53633          	sltu	a2,a0,a2
1c013712:	9636                	add	a2,a2,a3
1c013714:	0606                	slli	a2,a2,0x1
1c013716:	01f55693          	srli	a3,a0,0x1f
1c01371a:	8e55                	or	a2,a2,a3
1c01371c:	01c65693          	srli	a3,a2,0x1c
1c013720:	03068693          	addi	a3,a3,48
1c013724:	00d78323          	sb	a3,6(a5)
1c013728:	ff948a13          	addi	s4,s1,-7
1c01372c:	00151813          	slli	a6,a0,0x1
1c013730:	8e6d                	and	a2,a2,a1
1c013732:	00778693          	addi	a3,a5,7
1c013736:	1f405663          	blez	s4,1c013922 <pos_libc_prf+0xcde>
1c01373a:	00351693          	slli	a3,a0,0x3
1c01373e:	01e85893          	srli	a7,a6,0x1e
1c013742:	00261513          	slli	a0,a2,0x2
1c013746:	9836                	add	a6,a6,a3
1c013748:	00a8e533          	or	a0,a7,a0
1c01374c:	962a                	add	a2,a2,a0
1c01374e:	00d836b3          	sltu	a3,a6,a3
1c013752:	96b2                	add	a3,a3,a2
1c013754:	0686                	slli	a3,a3,0x1
1c013756:	01f85613          	srli	a2,a6,0x1f
1c01375a:	8ed1                	or	a3,a3,a2
1c01375c:	01c6d613          	srli	a2,a3,0x1c
1c013760:	03060613          	addi	a2,a2,48
1c013764:	00c783a3          	sb	a2,7(a5)
1c013768:	ff848a13          	addi	s4,s1,-8
1c01376c:	00181513          	slli	a0,a6,0x1
1c013770:	8eed                	and	a3,a3,a1
1c013772:	774053e3          	blez	s4,1c0146d8 <pos_libc_prf+0x1a94>
1c013776:	00381613          	slli	a2,a6,0x3
1c01377a:	01e55893          	srli	a7,a0,0x1e
1c01377e:	00269813          	slli	a6,a3,0x2
1c013782:	9532                	add	a0,a0,a2
1c013784:	0108e833          	or	a6,a7,a6
1c013788:	96c2                	add	a3,a3,a6
1c01378a:	00c53633          	sltu	a2,a0,a2
1c01378e:	9636                	add	a2,a2,a3
1c013790:	0606                	slli	a2,a2,0x1
1c013792:	01f55693          	srli	a3,a0,0x1f
1c013796:	8e55                	or	a2,a2,a3
1c013798:	01c65693          	srli	a3,a2,0x1c
1c01379c:	03068693          	addi	a3,a3,48
1c0137a0:	00d78423          	sb	a3,8(a5)
1c0137a4:	ff748a13          	addi	s4,s1,-9
1c0137a8:	00151813          	slli	a6,a0,0x1
1c0137ac:	8e6d                	and	a2,a2,a1
1c0137ae:	00978693          	addi	a3,a5,9
1c0137b2:	17405863          	blez	s4,1c013922 <pos_libc_prf+0xcde>
1c0137b6:	00351693          	slli	a3,a0,0x3
1c0137ba:	01e85893          	srli	a7,a6,0x1e
1c0137be:	00261513          	slli	a0,a2,0x2
1c0137c2:	9836                	add	a6,a6,a3
1c0137c4:	00a8e533          	or	a0,a7,a0
1c0137c8:	9532                	add	a0,a0,a2
1c0137ca:	00d836b3          	sltu	a3,a6,a3
1c0137ce:	96aa                	add	a3,a3,a0
1c0137d0:	01f85613          	srli	a2,a6,0x1f
1c0137d4:	0686                	slli	a3,a3,0x1
1c0137d6:	8ed1                	or	a3,a3,a2
1c0137d8:	01c6d613          	srli	a2,a3,0x1c
1c0137dc:	03060613          	addi	a2,a2,48
1c0137e0:	00c784a3          	sb	a2,9(a5)
1c0137e4:	ff648a13          	addi	s4,s1,-10
1c0137e8:	00181613          	slli	a2,a6,0x1
1c0137ec:	8eed                	and	a3,a3,a1
1c0137ee:	6f4051e3          	blez	s4,1c0146d0 <pos_libc_prf+0x1a8c>
1c0137f2:	00261593          	slli	a1,a2,0x2
1c0137f6:	01e65513          	srli	a0,a2,0x1e
1c0137fa:	00269813          	slli	a6,a3,0x2
1c0137fe:	962e                	add	a2,a2,a1
1c013800:	01056833          	or	a6,a0,a6
1c013804:	96c2                	add	a3,a3,a6
1c013806:	00b63533          	sltu	a0,a2,a1
1c01380a:	9536                	add	a0,a0,a3
1c01380c:	0506                	slli	a0,a0,0x1
1c01380e:	01f65693          	srli	a3,a2,0x1f
1c013812:	8d55                	or	a0,a0,a3
1c013814:	01c55813          	srli	a6,a0,0x1c
1c013818:	100006b7          	lui	a3,0x10000
1c01381c:	03080813          	addi	a6,a6,48
1c013820:	fff68593          	addi	a1,a3,-1 # fffffff <__l1_heap_size+0xfff17cb>
1c013824:	01078523          	sb	a6,10(a5)
1c013828:	ff548a13          	addi	s4,s1,-11
1c01382c:	00161893          	slli	a7,a2,0x1
1c013830:	8d6d                	and	a0,a0,a1
1c013832:	00b78693          	addi	a3,a5,11
1c013836:	0f405663          	blez	s4,1c013922 <pos_libc_prf+0xcde>
1c01383a:	00361693          	slli	a3,a2,0x3
1c01383e:	00251813          	slli	a6,a0,0x2
1c013842:	01e8de13          	srli	t3,a7,0x1e
1c013846:	01168633          	add	a2,a3,a7
1c01384a:	010e6833          	or	a6,t3,a6
1c01384e:	9542                	add	a0,a0,a6
1c013850:	00d636b3          	sltu	a3,a2,a3
1c013854:	96aa                	add	a3,a3,a0
1c013856:	0686                	slli	a3,a3,0x1
1c013858:	01f65513          	srli	a0,a2,0x1f
1c01385c:	8ec9                	or	a3,a3,a0
1c01385e:	01c6d513          	srli	a0,a3,0x1c
1c013862:	03050513          	addi	a0,a0,48
1c013866:	00a785a3          	sb	a0,11(a5)
1c01386a:	ff448a13          	addi	s4,s1,-12
1c01386e:	00161513          	slli	a0,a2,0x1
1c013872:	8eed                	and	a3,a3,a1
1c013874:	65405ae3          	blez	s4,1c0146c8 <pos_libc_prf+0x1a84>
1c013878:	01e55893          	srli	a7,a0,0x1e
1c01387c:	060e                	slli	a2,a2,0x3
1c01387e:	00269813          	slli	a6,a3,0x2
1c013882:	9532                	add	a0,a0,a2
1c013884:	0108e833          	or	a6,a7,a6
1c013888:	96c2                	add	a3,a3,a6
1c01388a:	00c53633          	sltu	a2,a0,a2
1c01388e:	9636                	add	a2,a2,a3
1c013890:	0606                	slli	a2,a2,0x1
1c013892:	01f55693          	srli	a3,a0,0x1f
1c013896:	8e55                	or	a2,a2,a3
1c013898:	01c65693          	srli	a3,a2,0x1c
1c01389c:	03068693          	addi	a3,a3,48
1c0138a0:	00d78623          	sb	a3,12(a5)
1c0138a4:	ff348a13          	addi	s4,s1,-13
1c0138a8:	00151893          	slli	a7,a0,0x1
1c0138ac:	00d78693          	addi	a3,a5,13
1c0138b0:	07405963          	blez	s4,1c013922 <pos_libc_prf+0xcde>
1c0138b4:	8e6d                	and	a2,a2,a1
1c0138b6:	00351693          	slli	a3,a0,0x3
1c0138ba:	00261813          	slli	a6,a2,0x2
1c0138be:	01e8de13          	srli	t3,a7,0x1e
1c0138c2:	01168533          	add	a0,a3,a7
1c0138c6:	010e6833          	or	a6,t3,a6
1c0138ca:	9642                	add	a2,a2,a6
1c0138cc:	00d536b3          	sltu	a3,a0,a3
1c0138d0:	96b2                	add	a3,a3,a2
1c0138d2:	0686                	slli	a3,a3,0x1
1c0138d4:	01f55613          	srli	a2,a0,0x1f
1c0138d8:	8ed1                	or	a3,a3,a2
1c0138da:	01c6d613          	srli	a2,a3,0x1c
1c0138de:	03060613          	addi	a2,a2,48
1c0138e2:	00c786a3          	sb	a2,13(a5)
1c0138e6:	ff248a13          	addi	s4,s1,-14
1c0138ea:	00151613          	slli	a2,a0,0x1
1c0138ee:	8eed                	and	a3,a3,a1
1c0138f0:	5d4058e3          	blez	s4,1c0146c0 <pos_libc_prf+0x1a7c>
1c0138f4:	00351593          	slli	a1,a0,0x3
1c0138f8:	01e65813          	srli	a6,a2,0x1e
1c0138fc:	00269513          	slli	a0,a3,0x2
1c013900:	00a86533          	or	a0,a6,a0
1c013904:	962e                	add	a2,a2,a1
1c013906:	96aa                	add	a3,a3,a0
1c013908:	00b63633          	sltu	a2,a2,a1
1c01390c:	9636                	add	a2,a2,a3
1c01390e:	0606                	slli	a2,a2,0x1
1c013910:	8271                	srli	a2,a2,0x1c
1c013912:	03060613          	addi	a2,a2,48
1c013916:	00f78693          	addi	a3,a5,15
1c01391a:	00c78723          	sb	a2,14(a5)
1c01391e:	ff148a13          	addi	s4,s1,-15
1c013922:	24028d63          	beqz	t0,1c013b7c <pos_libc_prf+0xf38>
1c013926:	4581                	li	a1,0
1c013928:	d402                	sw	zero,40(sp)
1c01392a:	d202                	sw	zero,36(sp)
1c01392c:	03000513          	li	a0,48
1c013930:	a011                	j	1c013934 <pos_libc_prf+0xcf0>
1c013932:	86be                	mv	a3,a5
1c013934:	fff68793          	addi	a5,a3,-1
1c013938:	0007c603          	lbu	a2,0(a5)
1c01393c:	fea60be3          	beq	a2,a0,1c013932 <pos_libc_prf+0xcee>
1c013940:	02e00513          	li	a0,46
1c013944:	4a01                	li	s4,0
1c013946:	00a61363          	bne	a2,a0,1c01394c <pos_libc_prf+0xd08>
1c01394a:	86be                	mv	a3,a5
1c01394c:	0dfcf793          	andi	a5,s9,223
1c013950:	04500613          	li	a2,69
1c013954:	14c78b63          	beq	a5,a2,1c013aaa <pos_libc_prf+0xe66>
1c013958:	4722                	lw	a4,8(sp)
1c01395a:	00068023          	sb	zero,0(a3)
1c01395e:	8e99                	sub	a3,a3,a4
1c013960:	47d2                	lw	a5,20(sp)
1c013962:	eba1                	bnez	a5,1c0139b2 <pos_libc_prf+0xd6e>
1c013964:	4762                	lw	a4,24(sp)
1c013966:	e731                	bnez	a4,1c0139b2 <pos_libc_prf+0xd6e>
1c013968:	03414783          	lbu	a5,52(sp)
1c01396c:	02d00713          	li	a4,45
1c013970:	c602                	sw	zero,12(sp)
1c013972:	04e78063          	beq	a5,a4,1c0139b2 <pos_libc_prf+0xd6e>
1c013976:	fd078793          	addi	a5,a5,-48
1c01397a:	4725                	li	a4,9
1c01397c:	00b689b3          	add	s3,a3,a1
1c013980:	a4f771e3          	bleu	a5,a4,1c0133c2 <pos_libc_prf+0x77e>
1c013984:	5702                	lw	a4,32(sp)
1c013986:	03410d93          	addi	s11,sp,52
1c01398a:	4481                	li	s1,0
1c01398c:	41370733          	sub	a4,a4,s3
1c013990:	d03a                	sw	a4,32(sp)
1c013992:	e4eff06f          	j	1c012fe0 <pos_libc_prf+0x39c>
1c013996:	00194c83          	lbu	s9,1(s2)
1c01399a:	04c00693          	li	a3,76
1c01399e:	0909                	addi	s2,s2,2
1c0139a0:	bf0ff06f          	j	1c012d90 <pos_libc_prf+0x14c>
1c0139a4:	00194c83          	lbu	s9,1(s2)
1c0139a8:	04800693          	li	a3,72
1c0139ac:	0909                	addi	s2,s2,2
1c0139ae:	be2ff06f          	j	1c012d90 <pos_libc_prf+0x14c>
1c0139b2:	4705                	li	a4,1
1c0139b4:	03514783          	lbu	a5,53(sp)
1c0139b8:	c63a                	sw	a4,12(sp)
1c0139ba:	bf75                	j	1c013976 <pos_libc_prf+0xd32>
1c0139bc:	00498793          	addi	a5,s3,4
1c0139c0:	d63e                	sw	a5,44(sp)
1c0139c2:	0009ad83          	lw	s11,0(s3)
1c0139c6:	4981                	li	s3,0
1c0139c8:	fe8ff06f          	j	1c0131b0 <pos_libc_prf+0x56c>
1c0139cc:	4705                	li	a4,1
1c0139ce:	c63a                	sw	a4,12(sp)
1c0139d0:	dceff06f          	j	1c012f9e <pos_libc_prf+0x35a>
1c0139d4:	414989b3          	sub	s3,s3,s4
1c0139d8:	8cce                	mv	s9,s3
1c0139da:	1d905263          	blez	s9,1c013b9e <pos_libc_prf+0xf5a>
1c0139de:	001dc50b          	p.lbu	a0,1(s11!)
1c0139e2:	85ea                	mv	a1,s10
1c0139e4:	1cfd                	addi	s9,s9,-1
1c0139e6:	9b82                	jalr	s7
1c0139e8:	fff539e3          	p.bneimm	a0,-1,1c0139da <pos_libc_prf+0xd96>
1c0139ec:	547d                	li	s0,-1
1c0139ee:	b0cff06f          	j	1c012cfa <pos_libc_prf+0xb6>
1c0139f2:	5782                	lw	a5,32(sp)
1c0139f4:	4481                	li	s1,0
1c0139f6:	c602                	sw	zero,12(sp)
1c0139f8:	413787b3          	sub	a5,a5,s3
1c0139fc:	d03e                	sw	a5,32(sp)
1c0139fe:	ddcff06f          	j	1c012fda <pos_libc_prf+0x396>
1c013a02:	800004b7          	lui	s1,0x80000
1c013a06:	fff4c493          	not	s1,s1
1c013a0a:	c50ff06f          	j	1c012e5a <pos_libc_prf+0x216>
1c013a0e:	8e55                	or	a2,a2,a3
1c013a10:	fbfc8713          	addi	a4,s9,-65
1c013a14:	003e0693          	addi	a3,t3,3
1c013a18:	47e5                	li	a5,25
1c013a1a:	20061c63          	bnez	a2,1c013c32 <pos_libc_prf+0xfee>
1c013a1e:	42e7e1e3          	bltu	a5,a4,1c014640 <pos_libc_prf+0x19fc>
1c013a22:	6795                	lui	a5,0x5
1c013a24:	e4978793          	addi	a5,a5,-439 # 4e49 <_l1_preload_size+0x3681>
1c013a28:	00fe1023          	sh	a5,0(t3)
1c013a2c:	04600793          	li	a5,70
1c013a30:	00fe0123          	sb	a5,2(t3)
1c013a34:	4722                	lw	a4,8(sp)
1c013a36:	000e01a3          	sb	zero,3(t3)
1c013a3a:	4581                	li	a1,0
1c013a3c:	8e99                	sub	a3,a3,a4
1c013a3e:	4a01                	li	s4,0
1c013a40:	d402                	sw	zero,40(sp)
1c013a42:	d202                	sw	zero,36(sp)
1c013a44:	bf31                	j	1c013960 <pos_libc_prf+0xd1c>
1c013a46:	47e2                	lw	a5,24(sp)
1c013a48:	03410e13          	addi	t3,sp,52
1c013a4c:	e0078a63          	beqz	a5,1c013060 <pos_libc_prf+0x41c>
1c013a50:	02000793          	li	a5,32
1c013a54:	02f10a23          	sb	a5,52(sp)
1c013a58:	03510e13          	addi	t3,sp,53
1c013a5c:	e04ff06f          	j	1c013060 <pos_libc_prf+0x41c>
1c013a60:	02d00793          	li	a5,45
1c013a64:	02f10a23          	sb	a5,52(sp)
1c013a68:	03510e13          	addi	t3,sp,53
1c013a6c:	df4ff06f          	j	1c013060 <pos_libc_prf+0x41c>
1c013a70:	56f5                	li	a3,-3
1c013a72:	0ed34463          	blt	t1,a3,1c013b5a <pos_libc_prf+0xf16>
1c013a76:	0e64c263          	blt	s1,t1,1c013b5a <pos_libc_prf+0xf16>
1c013a7a:	46b2                	lw	a3,12(sp)
1c013a7c:	406484b3          	sub	s1,s1,t1
1c013a80:	4281                	li	t0,0
1c013a82:	e689                	bnez	a3,1c013a8c <pos_libc_prf+0xe48>
1c013a84:	009022b3          	sgtz	t0,s1
1c013a88:	0ff2f293          	andi	t0,t0,255
1c013a8c:	009306b3          	add	a3,t1,s1
1c013a90:	3c06c3e3          	bltz	a3,1c014656 <pos_libc_prf+0x1a12>
1c013a94:	4741                	li	a4,16
1c013a96:	04e6ceb3          	p.min	t4,a3,a4
1c013a9a:	1efd                	addi	t4,t4,-1
1c013a9c:	06600c93          	li	s9,102
1c013aa0:	98069de3          	bnez	a3,1c01343a <pos_libc_prf+0x7f6>
1c013aa4:	08000637          	lui	a2,0x8000
1c013aa8:	baed                	j	1c0134a2 <pos_libc_prf+0x85e>
1c013aaa:	8636                	mv	a2,a3
1c013aac:	0196012b          	p.sb	s9,2(a2!) # 8000002 <__l1_heap_size+0x7ff17ce>
1c013ab0:	2e0346e3          	bltz	t1,1c01459c <pos_libc_prf+0x1958>
1c013ab4:	02b00793          	li	a5,43
1c013ab8:	00f680a3          	sb	a5,1(a3)
1c013abc:	06300793          	li	a5,99
1c013ac0:	0067de63          	ble	t1,a5,1c013adc <pos_libc_prf+0xe98>
1c013ac4:	06400713          	li	a4,100
1c013ac8:	02e347b3          	div	a5,t1,a4
1c013acc:	00368613          	addi	a2,a3,3
1c013ad0:	02e36333          	rem	t1,t1,a4
1c013ad4:	03078793          	addi	a5,a5,48
1c013ad8:	00f68123          	sb	a5,2(a3)
1c013adc:	47a9                	li	a5,10
1c013ade:	86b2                	mv	a3,a2
1c013ae0:	02f34733          	div	a4,t1,a5
1c013ae4:	02f367b3          	rem	a5,t1,a5
1c013ae8:	03070713          	addi	a4,a4,48
1c013aec:	00e6812b          	p.sb	a4,2(a3!)
1c013af0:	03078793          	addi	a5,a5,48
1c013af4:	00f600a3          	sb	a5,1(a2)
1c013af8:	b585                	j	1c013958 <pos_libc_prf+0xd14>
1c013afa:	4762                	lw	a4,24(sp)
1c013afc:	cb65                	beqz	a4,1c013bec <pos_libc_prf+0xfa8>
1c013afe:	02000713          	li	a4,32
1c013b02:	c1eff06f          	j	1c012f20 <pos_libc_prf+0x2dc>
1c013b06:	40f007b3          	neg	a5,a5
1c013b0a:	00f036b3          	snez	a3,a5
1c013b0e:	41400733          	neg	a4,s4
1c013b12:	02d00613          	li	a2,45
1c013b16:	8f15                	sub	a4,a4,a3
1c013b18:	03510693          	addi	a3,sp,53
1c013b1c:	02c10a23          	sb	a2,52(sp)
1c013b20:	c636                	sw	a3,12(sp)
1c013b22:	8cb6                	mv	s9,a3
1c013b24:	c0aff06f          	j	1c012f2e <pos_libc_prf+0x2ea>
1c013b28:	04800713          	li	a4,72
1c013b2c:	bae69f63          	bne	a3,a4,1c012eea <pos_libc_prf+0x2a6>
1c013b30:	8722                	mv	a4,s0
1c013b32:	00e78023          	sb	a4,0(a5)
1c013b36:	952ff06f          	j	1c012c88 <pos_libc_prf+0x44>
1c013b3a:	00798713          	addi	a4,s3,7
1c013b3e:	c4073733          	p.bclr	a4,a4,2,0
1c013b42:	00870793          	addi	a5,a4,8
1c013b46:	d63e                	sw	a5,44(sp)
1c013b48:	00472a03          	lw	s4,4(a4)
1c013b4c:	431c                	lw	a5,0(a4)
1c013b4e:	bc4ff06f          	j	1c012f12 <pos_libc_prf+0x2ce>
1c013b52:	4501                	li	a0,0
1c013b54:	1854                	addi	a3,sp,52
1c013b56:	e84ff06f          	j	1c0131da <pos_libc_prf+0x596>
1c013b5a:	ffec8713          	addi	a4,s9,-2
1c013b5e:	0ff77c93          	andi	s9,a4,255
1c013b62:	2c0486e3          	beqz	s1,1c01462e <pos_libc_prf+0x19ea>
1c013b66:	4732                	lw	a4,12(sp)
1c013b68:	fff48693          	addi	a3,s1,-1 # 7fffffff <pulp__FC+0x80000000>
1c013b6c:	200713e3          	bnez	a4,1c014572 <pos_libc_prf+0x192e>
1c013b70:	00d022b3          	sgtz	t0,a3
1c013b74:	0ff2f293          	andi	t0,t0,255
1c013b78:	84b6                	mv	s1,a3
1c013b7a:	b855                	j	1c01342e <pos_libc_prf+0x7ea>
1c013b7c:	85d2                	mv	a1,s4
1c013b7e:	d402                	sw	zero,40(sp)
1c013b80:	d202                	sw	zero,36(sp)
1c013b82:	b3e9                	j	1c01394c <pos_libc_prf+0xd08>
1c013b84:	00798793          	addi	a5,s3,7
1c013b88:	c407b7b3          	p.bclr	a5,a5,2,0
1c013b8c:	00878713          	addi	a4,a5,8
1c013b90:	d63a                	sw	a4,44(sp)
1c013b92:	0007ad83          	lw	s11,0(a5)
1c013b96:	0047a983          	lw	s3,4(a5)
1c013b9a:	e16ff06f          	j	1c0131b0 <pos_libc_prf+0x56c>
1c013b9e:	47b2                	lw	a5,12(sp)
1c013ba0:	46c2                	lw	a3,16(sp)
1c013ba2:	5702                	lw	a4,32(sp)
1c013ba4:	943e                	add	s0,s0,a5
1c013ba6:	9426                	add	s0,s0,s1
1c013ba8:	9436                	add	s0,s0,a3
1c013baa:	59b2                	lw	s3,44(sp)
1c013bac:	8ce05e63          	blez	a4,1c012c88 <pos_libc_prf+0x44>
1c013bb0:	943a                	add	s0,s0,a4
1c013bb2:	fff70a93          	addi	s5,a4,-1
1c013bb6:	a021                	j	1c013bbe <pos_libc_prf+0xf7a>
1c013bb8:	1afd                	addi	s5,s5,-1
1c013bba:	2dfaa0e3          	p.beqimm	s5,-1,1c01467a <pos_libc_prf+0x1a36>
1c013bbe:	85ea                	mv	a1,s10
1c013bc0:	02000513          	li	a0,32
1c013bc4:	9b82                	jalr	s7
1c013bc6:	fff539e3          	p.bneimm	a0,-1,1c013bb8 <pos_libc_prf+0xf74>
1c013bca:	547d                	li	s0,-1
1c013bcc:	92eff06f          	j	1c012cfa <pos_libc_prf+0xb6>
1c013bd0:	49c2                	lw	s3,16(sp)
1c013bd2:	5c92                	lw	s9,36(sp)
1c013bd4:	1f9052e3          	blez	s9,1c0145b8 <pos_libc_prf+0x1974>
1c013bd8:	85ea                	mv	a1,s10
1c013bda:	03000513          	li	a0,48
1c013bde:	9b82                	jalr	s7
1c013be0:	1cfd                	addi	s9,s9,-1
1c013be2:	fff539e3          	p.bneimm	a0,-1,1c013bd4 <pos_libc_prf+0xf90>
1c013be6:	547d                	li	s0,-1
1c013be8:	912ff06f          	j	1c012cfa <pos_libc_prf+0xb6>
1c013bec:	1854                	addi	a3,sp,52
1c013bee:	8752                	mv	a4,s4
1c013bf0:	c636                	sw	a3,12(sp)
1c013bf2:	8cb6                	mv	s9,a3
1c013bf4:	b3aff06f          	j	1c012f2e <pos_libc_prf+0x2ea>
1c013bf8:	4481                	li	s1,0
1c013bfa:	c602                	sw	zero,12(sp)
1c013bfc:	4981                	li	s3,0
1c013bfe:	bdcff06f          	j	1c012fda <pos_libc_prf+0x396>
1c013c02:	1858                	addi	a4,sp,52
1c013c04:	45e5                	li	a1,25
1c013c06:	0017478b          	p.lbu	a5,1(a4!)
1c013c0a:	f9f78693          	addi	a3,a5,-97
1c013c0e:	0ff6f693          	andi	a3,a3,255
1c013c12:	00d5ed63          	bltu	a1,a3,1c013c2c <pos_libc_prf+0xfe8>
1c013c16:	1781                	addi	a5,a5,-32
1c013c18:	fef70fa3          	sb	a5,-1(a4)
1c013c1c:	0017478b          	p.lbu	a5,1(a4!)
1c013c20:	f9f78693          	addi	a3,a5,-97
1c013c24:	0ff6f693          	andi	a3,a3,255
1c013c28:	fed5f7e3          	bleu	a3,a1,1c013c16 <pos_libc_prf+0xfd2>
1c013c2c:	ffe9                	bnez	a5,1c013c06 <pos_libc_prf+0xfc2>
1c013c2e:	ed8ff06f          	j	1c013306 <pos_libc_prf+0x6c2>
1c013c32:	22e7e9e3          	bltu	a5,a4,1c014664 <pos_libc_prf+0x1a20>
1c013c36:	6791                	lui	a5,0x4
1c013c38:	14e78793          	addi	a5,a5,334 # 414e <_l1_preload_size+0x2986>
1c013c3c:	00fe1023          	sh	a5,0(t3)
1c013c40:	04e00793          	li	a5,78
1c013c44:	00fe0123          	sb	a5,2(t3)
1c013c48:	b3f5                	j	1c013a34 <pos_libc_prf+0xdf0>
1c013c4a:	46b2                	lw	a3,12(sp)
1c013c4c:	10068ee3          	beqz	a3,1c014568 <pos_libc_prf+0x1924>
1c013c50:	03000793          	li	a5,48
1c013c54:	02f10a23          	sb	a5,52(sp)
1c013c58:	013de7b3          	or	a5,s11,s3
1c013c5c:	160783e3          	beqz	a5,1c0145c2 <pos_libc_prf+0x197e>
1c013c60:	03510613          	addi	a2,sp,53
1c013c64:	4585                	li	a1,1
1c013c66:	8732                	mv	a4,a2
1c013c68:	87ba                	mv	a5,a4
1c013c6a:	a011                	j	1c013c6e <pos_libc_prf+0x102a>
1c013c6c:	87b6                	mv	a5,a3
1c013c6e:	01d99513          	slli	a0,s3,0x1d
1c013c72:	f83db6b3          	p.bclr	a3,s11,28,3
1c013c76:	003ddd93          	srli	s11,s11,0x3
1c013c7a:	03068693          	addi	a3,a3,48
1c013c7e:	01b56db3          	or	s11,a0,s11
1c013c82:	0039d993          	srli	s3,s3,0x3
1c013c86:	00d78023          	sb	a3,0(a5)
1c013c8a:	013de533          	or	a0,s11,s3
1c013c8e:	00178693          	addi	a3,a5,1
1c013c92:	fd69                	bnez	a0,1c013c6c <pos_libc_prf+0x1028>
1c013c94:	00068023          	sb	zero,0(a3)
1c013c98:	8e91                	sub	a3,a3,a2
1c013c9a:	00f77c63          	bleu	a5,a4,1c013cb2 <pos_libc_prf+0x106e>
1c013c9e:	00074503          	lbu	a0,0(a4)
1c013ca2:	0007c603          	lbu	a2,0(a5)
1c013ca6:	fea78fab          	p.sb	a0,-1(a5!)
1c013caa:	00c700ab          	p.sb	a2,1(a4!)
1c013cae:	fef768e3          	bltu	a4,a5,1c013c9e <pos_libc_prf+0x105a>
1c013cb2:	00b689b3          	add	s3,a3,a1
1c013cb6:	c602                	sw	zero,12(sp)
1c013cb8:	ae6ff06f          	j	1c012f9e <pos_libc_prf+0x35a>
1c013cbc:	04600293          	li	t0,70
1c013cc0:	06600c93          	li	s9,102
1c013cc4:	bb0ff06f          	j	1c013074 <pos_libc_prf+0x430>
1c013cc8:	02e00793          	li	a5,46
1c013ccc:	00fe00a3          	sb	a5,1(t3)
1c013cd0:	002e0793          	addi	a5,t3,2
1c013cd4:	8a9042e3          	bgtz	s1,1c013578 <pos_libc_prf+0x934>
1c013cd8:	8a26                	mv	s4,s1
1c013cda:	86be                	mv	a3,a5
1c013cdc:	b199                	j	1c013922 <pos_libc_prf+0xcde>
1c013cde:	0e6058e3          	blez	t1,1c0145ce <pos_libc_prf+0x198a>
1c013ce2:	00279613          	slli	a2,a5,0x2
1c013ce6:	01e7de93          	srli	t4,a5,0x1e
1c013cea:	00251893          	slli	a7,a0,0x2
1c013cee:	00f605b3          	add	a1,a2,a5
1c013cf2:	011ee8b3          	or	a7,t4,a7
1c013cf6:	98aa                	add	a7,a7,a0
1c013cf8:	00c5b533          	sltu	a0,a1,a2
1c013cfc:	9546                	add	a0,a0,a7
1c013cfe:	01f5d793          	srli	a5,a1,0x1f
1c013d02:	0506                	slli	a0,a0,0x1
1c013d04:	8d5d                	or	a0,a0,a5
1c013d06:	01c55793          	srli	a5,a0,0x1c
1c013d0a:	03078793          	addi	a5,a5,48
1c013d0e:	10000637          	lui	a2,0x10000
1c013d12:	167d                	addi	a2,a2,-1
1c013d14:	00fe0023          	sb	a5,0(t3)
1c013d18:	8d71                	and	a0,a0,a2
1c013d1a:	00159793          	slli	a5,a1,0x1
1c013d1e:	241325e3          	p.beqimm	t1,1,1c014768 <pos_libc_prf+0x1b24>
1c013d22:	01e7de93          	srli	t4,a5,0x1e
1c013d26:	00251893          	slli	a7,a0,0x2
1c013d2a:	00359693          	slli	a3,a1,0x3
1c013d2e:	00f685b3          	add	a1,a3,a5
1c013d32:	011ee8b3          	or	a7,t4,a7
1c013d36:	98aa                	add	a7,a7,a0
1c013d38:	00d5b533          	sltu	a0,a1,a3
1c013d3c:	9546                	add	a0,a0,a7
1c013d3e:	01f5d793          	srli	a5,a1,0x1f
1c013d42:	0506                	slli	a0,a0,0x1
1c013d44:	8d5d                	or	a0,a0,a5
1c013d46:	01c55793          	srli	a5,a0,0x1c
1c013d4a:	03078793          	addi	a5,a5,48
1c013d4e:	00fe00a3          	sb	a5,1(t3)
1c013d52:	8d71                	and	a0,a0,a2
1c013d54:	00159793          	slli	a5,a1,0x1
1c013d58:	202323e3          	p.beqimm	t1,2,1c01475e <pos_libc_prf+0x1b1a>
1c013d5c:	01e7de93          	srli	t4,a5,0x1e
1c013d60:	00251893          	slli	a7,a0,0x2
1c013d64:	058e                	slli	a1,a1,0x3
1c013d66:	00f586b3          	add	a3,a1,a5
1c013d6a:	011ee8b3          	or	a7,t4,a7
1c013d6e:	98aa                	add	a7,a7,a0
1c013d70:	00b6b533          	sltu	a0,a3,a1
1c013d74:	9546                	add	a0,a0,a7
1c013d76:	01f6d793          	srli	a5,a3,0x1f
1c013d7a:	0506                	slli	a0,a0,0x1
1c013d7c:	8d5d                	or	a0,a0,a5
1c013d7e:	01c55793          	srli	a5,a0,0x1c
1c013d82:	03078793          	addi	a5,a5,48
1c013d86:	00fe0123          	sb	a5,2(t3)
1c013d8a:	ffd30793          	addi	a5,t1,-3
1c013d8e:	d23e                	sw	a5,36(sp)
1c013d90:	5712                	lw	a4,36(sp)
1c013d92:	00169793          	slli	a5,a3,0x1
1c013d96:	8d71                	and	a0,a0,a2
1c013d98:	1ae05fe3          	blez	a4,1c014756 <pos_libc_prf+0x1b12>
1c013d9c:	068e                	slli	a3,a3,0x3
1c013d9e:	01e7de93          	srli	t4,a5,0x1e
1c013da2:	00251893          	slli	a7,a0,0x2
1c013da6:	00f685b3          	add	a1,a3,a5
1c013daa:	011ee8b3          	or	a7,t4,a7
1c013dae:	98aa                	add	a7,a7,a0
1c013db0:	00d5b533          	sltu	a0,a1,a3
1c013db4:	9546                	add	a0,a0,a7
1c013db6:	01f5d793          	srli	a5,a1,0x1f
1c013dba:	0506                	slli	a0,a0,0x1
1c013dbc:	8d5d                	or	a0,a0,a5
1c013dbe:	01c55793          	srli	a5,a0,0x1c
1c013dc2:	03078793          	addi	a5,a5,48
1c013dc6:	00fe01a3          	sb	a5,3(t3)
1c013dca:	ffc30793          	addi	a5,t1,-4
1c013dce:	d23e                	sw	a5,36(sp)
1c013dd0:	5692                	lw	a3,36(sp)
1c013dd2:	00159793          	slli	a5,a1,0x1
1c013dd6:	8d71                	and	a0,a0,a2
1c013dd8:	16d05be3          	blez	a3,1c01474e <pos_libc_prf+0x1b0a>
1c013ddc:	01e7de93          	srli	t4,a5,0x1e
1c013de0:	00251893          	slli	a7,a0,0x2
1c013de4:	058e                	slli	a1,a1,0x3
1c013de6:	00f586b3          	add	a3,a1,a5
1c013dea:	011ee8b3          	or	a7,t4,a7
1c013dee:	98aa                	add	a7,a7,a0
1c013df0:	00b6b533          	sltu	a0,a3,a1
1c013df4:	9546                	add	a0,a0,a7
1c013df6:	01f6d793          	srli	a5,a3,0x1f
1c013dfa:	0506                	slli	a0,a0,0x1
1c013dfc:	8d5d                	or	a0,a0,a5
1c013dfe:	01c55793          	srli	a5,a0,0x1c
1c013e02:	03078793          	addi	a5,a5,48
1c013e06:	ffb30713          	addi	a4,t1,-5
1c013e0a:	00fe0223          	sb	a5,4(t3)
1c013e0e:	d23a                	sw	a4,36(sp)
1c013e10:	00169793          	slli	a5,a3,0x1
1c013e14:	8d71                	and	a0,a0,a2
1c013e16:	12e058e3          	blez	a4,1c014746 <pos_libc_prf+0x1b02>
1c013e1a:	068e                	slli	a3,a3,0x3
1c013e1c:	01e7de93          	srli	t4,a5,0x1e
1c013e20:	00251893          	slli	a7,a0,0x2
1c013e24:	00f685b3          	add	a1,a3,a5
1c013e28:	011ee8b3          	or	a7,t4,a7
1c013e2c:	98aa                	add	a7,a7,a0
1c013e2e:	00d5b533          	sltu	a0,a1,a3
1c013e32:	9546                	add	a0,a0,a7
1c013e34:	01f5d793          	srli	a5,a1,0x1f
1c013e38:	0506                	slli	a0,a0,0x1
1c013e3a:	8d5d                	or	a0,a0,a5
1c013e3c:	01c55793          	srli	a5,a0,0x1c
1c013e40:	03078793          	addi	a5,a5,48
1c013e44:	00fe02a3          	sb	a5,5(t3)
1c013e48:	ffa30793          	addi	a5,t1,-6
1c013e4c:	d23e                	sw	a5,36(sp)
1c013e4e:	5692                	lw	a3,36(sp)
1c013e50:	00159793          	slli	a5,a1,0x1
1c013e54:	8d71                	and	a0,a0,a2
1c013e56:	0ed054e3          	blez	a3,1c01473e <pos_libc_prf+0x1afa>
1c013e5a:	01e7de93          	srli	t4,a5,0x1e
1c013e5e:	00251893          	slli	a7,a0,0x2
1c013e62:	058e                	slli	a1,a1,0x3
1c013e64:	00f586b3          	add	a3,a1,a5
1c013e68:	011ee8b3          	or	a7,t4,a7
1c013e6c:	98aa                	add	a7,a7,a0
1c013e6e:	00b6b533          	sltu	a0,a3,a1
1c013e72:	9546                	add	a0,a0,a7
1c013e74:	01f6d793          	srli	a5,a3,0x1f
1c013e78:	0506                	slli	a0,a0,0x1
1c013e7a:	8d5d                	or	a0,a0,a5
1c013e7c:	01c55793          	srli	a5,a0,0x1c
1c013e80:	03078793          	addi	a5,a5,48
1c013e84:	ff930713          	addi	a4,t1,-7
1c013e88:	00fe0323          	sb	a5,6(t3)
1c013e8c:	d23a                	sw	a4,36(sp)
1c013e8e:	00169793          	slli	a5,a3,0x1
1c013e92:	8d71                	and	a0,a0,a2
1c013e94:	0ae051e3          	blez	a4,1c014736 <pos_libc_prf+0x1af2>
1c013e98:	068e                	slli	a3,a3,0x3
1c013e9a:	01e7d893          	srli	a7,a5,0x1e
1c013e9e:	00251593          	slli	a1,a0,0x2
1c013ea2:	97b6                	add	a5,a5,a3
1c013ea4:	00b8e5b3          	or	a1,a7,a1
1c013ea8:	95aa                	add	a1,a1,a0
1c013eaa:	00d7b533          	sltu	a0,a5,a3
1c013eae:	952e                	add	a0,a0,a1
1c013eb0:	01f7d693          	srli	a3,a5,0x1f
1c013eb4:	0506                	slli	a0,a0,0x1
1c013eb6:	8d55                	or	a0,a0,a3
1c013eb8:	01c55693          	srli	a3,a0,0x1c
1c013ebc:	03068693          	addi	a3,a3,48
1c013ec0:	00de03a3          	sb	a3,7(t3)
1c013ec4:	ff830693          	addi	a3,t1,-8
1c013ec8:	d236                	sw	a3,36(sp)
1c013eca:	5712                	lw	a4,36(sp)
1c013ecc:	0786                	slli	a5,a5,0x1
1c013ece:	86be                	mv	a3,a5
1c013ed0:	8d71                	and	a0,a0,a2
1c013ed2:	04e05ee3          	blez	a4,1c01472e <pos_libc_prf+0x1aea>
1c013ed6:	078a                	slli	a5,a5,0x2
1c013ed8:	00251613          	slli	a2,a0,0x2
1c013edc:	01e6d593          	srli	a1,a3,0x1e
1c013ee0:	8e4d                	or	a2,a2,a1
1c013ee2:	96be                	add	a3,a3,a5
1c013ee4:	962a                	add	a2,a2,a0
1c013ee6:	00f6b533          	sltu	a0,a3,a5
1c013eea:	9532                	add	a0,a0,a2
1c013eec:	01f6d793          	srli	a5,a3,0x1f
1c013ef0:	0506                	slli	a0,a0,0x1
1c013ef2:	8d5d                	or	a0,a0,a5
1c013ef4:	01c55793          	srli	a5,a0,0x1c
1c013ef8:	03078793          	addi	a5,a5,48
1c013efc:	00fe0423          	sb	a5,8(t3)
1c013f00:	ff730793          	addi	a5,t1,-9
1c013f04:	d23e                	sw	a5,36(sp)
1c013f06:	5712                	lw	a4,36(sp)
1c013f08:	10000637          	lui	a2,0x10000
1c013f0c:	167d                	addi	a2,a2,-1
1c013f0e:	00169793          	slli	a5,a3,0x1
1c013f12:	8d71                	and	a0,a0,a2
1c013f14:	00e059e3          	blez	a4,1c014726 <pos_libc_prf+0x1ae2>
1c013f18:	01e7de93          	srli	t4,a5,0x1e
1c013f1c:	00251893          	slli	a7,a0,0x2
1c013f20:	068e                	slli	a3,a3,0x3
1c013f22:	00f685b3          	add	a1,a3,a5
1c013f26:	011ee8b3          	or	a7,t4,a7
1c013f2a:	00d5b6b3          	sltu	a3,a1,a3
1c013f2e:	9546                	add	a0,a0,a7
1c013f30:	9536                	add	a0,a0,a3
1c013f32:	01f5d793          	srli	a5,a1,0x1f
1c013f36:	0506                	slli	a0,a0,0x1
1c013f38:	8d5d                	or	a0,a0,a5
1c013f3a:	01c55793          	srli	a5,a0,0x1c
1c013f3e:	03078793          	addi	a5,a5,48
1c013f42:	00fe04a3          	sb	a5,9(t3)
1c013f46:	ff630793          	addi	a5,t1,-10
1c013f4a:	d23e                	sw	a5,36(sp)
1c013f4c:	5692                	lw	a3,36(sp)
1c013f4e:	00159793          	slli	a5,a1,0x1
1c013f52:	8d71                	and	a0,a0,a2
1c013f54:	7cd05563          	blez	a3,1c01471e <pos_libc_prf+0x1ada>
1c013f58:	01e7de93          	srli	t4,a5,0x1e
1c013f5c:	00251893          	slli	a7,a0,0x2
1c013f60:	058e                	slli	a1,a1,0x3
1c013f62:	00f586b3          	add	a3,a1,a5
1c013f66:	011ee8b3          	or	a7,t4,a7
1c013f6a:	98aa                	add	a7,a7,a0
1c013f6c:	00b6b533          	sltu	a0,a3,a1
1c013f70:	9546                	add	a0,a0,a7
1c013f72:	01f6d793          	srli	a5,a3,0x1f
1c013f76:	0506                	slli	a0,a0,0x1
1c013f78:	8d5d                	or	a0,a0,a5
1c013f7a:	01c55793          	srli	a5,a0,0x1c
1c013f7e:	03078793          	addi	a5,a5,48
1c013f82:	ff530713          	addi	a4,t1,-11
1c013f86:	00fe0523          	sb	a5,10(t3)
1c013f8a:	d23a                	sw	a4,36(sp)
1c013f8c:	00169793          	slli	a5,a3,0x1
1c013f90:	8d71                	and	a0,a0,a2
1c013f92:	78e05263          	blez	a4,1c014716 <pos_libc_prf+0x1ad2>
1c013f96:	068e                	slli	a3,a3,0x3
1c013f98:	01e7de93          	srli	t4,a5,0x1e
1c013f9c:	00251893          	slli	a7,a0,0x2
1c013fa0:	00f685b3          	add	a1,a3,a5
1c013fa4:	011ee8b3          	or	a7,t4,a7
1c013fa8:	98aa                	add	a7,a7,a0
1c013faa:	00d5b533          	sltu	a0,a1,a3
1c013fae:	9546                	add	a0,a0,a7
1c013fb0:	01f5d793          	srli	a5,a1,0x1f
1c013fb4:	0506                	slli	a0,a0,0x1
1c013fb6:	8d5d                	or	a0,a0,a5
1c013fb8:	01c55793          	srli	a5,a0,0x1c
1c013fbc:	03078793          	addi	a5,a5,48
1c013fc0:	00fe05a3          	sb	a5,11(t3)
1c013fc4:	ff430793          	addi	a5,t1,-12
1c013fc8:	d23e                	sw	a5,36(sp)
1c013fca:	5692                	lw	a3,36(sp)
1c013fcc:	00159793          	slli	a5,a1,0x1
1c013fd0:	8d71                	and	a0,a0,a2
1c013fd2:	72d05f63          	blez	a3,1c014710 <pos_libc_prf+0x1acc>
1c013fd6:	01e7d893          	srli	a7,a5,0x1e
1c013fda:	00251813          	slli	a6,a0,0x2
1c013fde:	058e                	slli	a1,a1,0x3
1c013fe0:	00f586b3          	add	a3,a1,a5
1c013fe4:	0108e833          	or	a6,a7,a6
1c013fe8:	982a                	add	a6,a6,a0
1c013fea:	00b6b533          	sltu	a0,a3,a1
1c013fee:	9542                	add	a0,a0,a6
1c013ff0:	01f6d793          	srli	a5,a3,0x1f
1c013ff4:	0506                	slli	a0,a0,0x1
1c013ff6:	8d5d                	or	a0,a0,a5
1c013ff8:	01c55793          	srli	a5,a0,0x1c
1c013ffc:	03078793          	addi	a5,a5,48
1c014000:	ff330713          	addi	a4,t1,-13
1c014004:	00fe0623          	sb	a5,12(t3)
1c014008:	d23a                	sw	a4,36(sp)
1c01400a:	00169793          	slli	a5,a3,0x1
1c01400e:	8d71                	and	a0,a0,a2
1c014010:	6ee05c63          	blez	a4,1c014708 <pos_libc_prf+0x1ac4>
1c014014:	068e                	slli	a3,a3,0x3
1c014016:	01e7d893          	srli	a7,a5,0x1e
1c01401a:	00251813          	slli	a6,a0,0x2
1c01401e:	00f685b3          	add	a1,a3,a5
1c014022:	0108e833          	or	a6,a7,a6
1c014026:	982a                	add	a6,a6,a0
1c014028:	00d5b533          	sltu	a0,a1,a3
1c01402c:	9542                	add	a0,a0,a6
1c01402e:	01f5d793          	srli	a5,a1,0x1f
1c014032:	0506                	slli	a0,a0,0x1
1c014034:	8d5d                	or	a0,a0,a5
1c014036:	01c55793          	srli	a5,a0,0x1c
1c01403a:	03078793          	addi	a5,a5,48
1c01403e:	00fe06a3          	sb	a5,13(t3)
1c014042:	ff230793          	addi	a5,t1,-14
1c014046:	d23e                	sw	a5,36(sp)
1c014048:	5692                	lw	a3,36(sp)
1c01404a:	00159793          	slli	a5,a1,0x1
1c01404e:	8d71                	and	a0,a0,a2
1c014050:	6ad05863          	blez	a3,1c014700 <pos_libc_prf+0x1abc>
1c014054:	01e7d893          	srli	a7,a5,0x1e
1c014058:	00251813          	slli	a6,a0,0x2
1c01405c:	058e                	slli	a1,a1,0x3
1c01405e:	00f586b3          	add	a3,a1,a5
1c014062:	0108e833          	or	a6,a7,a6
1c014066:	982a                	add	a6,a6,a0
1c014068:	00b6b533          	sltu	a0,a3,a1
1c01406c:	9542                	add	a0,a0,a6
1c01406e:	01f6d793          	srli	a5,a3,0x1f
1c014072:	0506                	slli	a0,a0,0x1
1c014074:	8d5d                	or	a0,a0,a5
1c014076:	01c55793          	srli	a5,a0,0x1c
1c01407a:	03078793          	addi	a5,a5,48
1c01407e:	ff130713          	addi	a4,t1,-15
1c014082:	00fe0723          	sb	a5,14(t3)
1c014086:	d23a                	sw	a4,36(sp)
1c014088:	00169793          	slli	a5,a3,0x1
1c01408c:	8d71                	and	a0,a0,a2
1c01408e:	66e05563          	blez	a4,1c0146f8 <pos_libc_prf+0x1ab4>
1c014092:	01e7d813          	srli	a6,a5,0x1e
1c014096:	068e                	slli	a3,a3,0x3
1c014098:	00251593          	slli	a1,a0,0x2
1c01409c:	97b6                	add	a5,a5,a3
1c01409e:	00b865b3          	or	a1,a6,a1
1c0140a2:	95aa                	add	a1,a1,a0
1c0140a4:	00d7b533          	sltu	a0,a5,a3
1c0140a8:	952e                	add	a0,a0,a1
1c0140aa:	01f7d693          	srli	a3,a5,0x1f
1c0140ae:	0506                	slli	a0,a0,0x1
1c0140b0:	8d55                	or	a0,a0,a3
1c0140b2:	01c55593          	srli	a1,a0,0x1c
1c0140b6:	03058593          	addi	a1,a1,48
1c0140ba:	1341                	addi	t1,t1,-16
1c0140bc:	010e0693          	addi	a3,t3,16
1c0140c0:	0786                	slli	a5,a5,0x1
1c0140c2:	8d71                	and	a0,a0,a2
1c0140c4:	00be07a3          	sb	a1,15(t3)
1c0140c8:	d21a                	sw	t1,36(sp)
1c0140ca:	4801                	li	a6,0
1c0140cc:	4732                	lw	a4,12(sp)
1c0140ce:	5a070963          	beqz	a4,1c014680 <pos_libc_prf+0x1a3c>
1c0140d2:	02e00593          	li	a1,46
1c0140d6:	00b68023          	sb	a1,0(a3)
1c0140da:	d402                	sw	zero,40(sp)
1c0140dc:	85a6                	mv	a1,s1
1c0140de:	00168613          	addi	a2,a3,1
1c0140e2:	4301                	li	t1,0
1c0140e4:	5ab05363          	blez	a1,1c01468a <pos_libc_prf+0x1a46>
1c0140e8:	5a080163          	beqz	a6,1c01468a <pos_libc_prf+0x1a46>
1c0140ec:	00279693          	slli	a3,a5,0x2
1c0140f0:	01e7de13          	srli	t3,a5,0x1e
1c0140f4:	00251893          	slli	a7,a0,0x2
1c0140f8:	97b6                	add	a5,a5,a3
1c0140fa:	011e68b3          	or	a7,t3,a7
1c0140fe:	9546                	add	a0,a0,a7
1c014100:	00d7be33          	sltu	t3,a5,a3
1c014104:	9e2a                	add	t3,t3,a0
1c014106:	01f7d693          	srli	a3,a5,0x1f
1c01410a:	0e06                	slli	t3,t3,0x1
1c01410c:	01c6ee33          	or	t3,a3,t3
1c014110:	01ce5693          	srli	a3,t3,0x1c
1c014114:	03068693          	addi	a3,a3,48
1c014118:	00d60023          	sb	a3,0(a2) # 10000000 <__l1_heap_size+0xfff17cc>
1c01411c:	fff58493          	addi	s1,a1,-1
1c014120:	00160693          	addi	a3,a2,1
1c014124:	00179893          	slli	a7,a5,0x1
1c014128:	42048663          	beqz	s1,1c014554 <pos_libc_prf+0x1910>
1c01412c:	42182463          	p.beqimm	a6,1,1c014554 <pos_libc_prf+0x1910>
1c014130:	10000537          	lui	a0,0x10000
1c014134:	157d                	addi	a0,a0,-1
1c014136:	00ae7e33          	and	t3,t3,a0
1c01413a:	078e                	slli	a5,a5,0x3
1c01413c:	01e8de93          	srli	t4,a7,0x1e
1c014140:	002e1693          	slli	a3,t3,0x2
1c014144:	98be                	add	a7,a7,a5
1c014146:	00dee6b3          	or	a3,t4,a3
1c01414a:	9e36                	add	t3,t3,a3
1c01414c:	00f8b7b3          	sltu	a5,a7,a5
1c014150:	97f2                	add	a5,a5,t3
1c014152:	01f8d693          	srli	a3,a7,0x1f
1c014156:	0786                	slli	a5,a5,0x1
1c014158:	8fd5                	or	a5,a5,a3
1c01415a:	01c7d693          	srli	a3,a5,0x1c
1c01415e:	03068693          	addi	a3,a3,48
1c014162:	00d600a3          	sb	a3,1(a2)
1c014166:	ffe58493          	addi	s1,a1,-2
1c01416a:	00260693          	addi	a3,a2,2
1c01416e:	00189e13          	slli	t3,a7,0x1
1c014172:	8fe9                	and	a5,a5,a0
1c014174:	3e048063          	beqz	s1,1c014554 <pos_libc_prf+0x1910>
1c014178:	3c282e63          	p.beqimm	a6,2,1c014554 <pos_libc_prf+0x1910>
1c01417c:	088e                	slli	a7,a7,0x3
1c01417e:	01ee5e93          	srli	t4,t3,0x1e
1c014182:	00279693          	slli	a3,a5,0x2
1c014186:	9e46                	add	t3,t3,a7
1c014188:	00dee6b3          	or	a3,t4,a3
1c01418c:	011e38b3          	sltu	a7,t3,a7
1c014190:	97b6                	add	a5,a5,a3
1c014192:	97c6                	add	a5,a5,a7
1c014194:	01fe5693          	srli	a3,t3,0x1f
1c014198:	0786                	slli	a5,a5,0x1
1c01419a:	8fd5                	or	a5,a5,a3
1c01419c:	01c7d693          	srli	a3,a5,0x1c
1c0141a0:	03068693          	addi	a3,a3,48
1c0141a4:	00d60123          	sb	a3,2(a2)
1c0141a8:	ffd58493          	addi	s1,a1,-3
1c0141ac:	00360693          	addi	a3,a2,3
1c0141b0:	001e1893          	slli	a7,t3,0x1
1c0141b4:	8fe9                	and	a5,a5,a0
1c0141b6:	38905f63          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c0141ba:	38382d63          	p.beqimm	a6,3,1c014554 <pos_libc_prf+0x1910>
1c0141be:	0e0e                	slli	t3,t3,0x3
1c0141c0:	01e8de93          	srli	t4,a7,0x1e
1c0141c4:	00279693          	slli	a3,a5,0x2
1c0141c8:	98f2                	add	a7,a7,t3
1c0141ca:	00dee6b3          	or	a3,t4,a3
1c0141ce:	01c8be33          	sltu	t3,a7,t3
1c0141d2:	97b6                	add	a5,a5,a3
1c0141d4:	97f2                	add	a5,a5,t3
1c0141d6:	01f8d693          	srli	a3,a7,0x1f
1c0141da:	0786                	slli	a5,a5,0x1
1c0141dc:	8fd5                	or	a5,a5,a3
1c0141de:	01c7d693          	srli	a3,a5,0x1c
1c0141e2:	03068693          	addi	a3,a3,48
1c0141e6:	00d601a3          	sb	a3,3(a2)
1c0141ea:	ffc58493          	addi	s1,a1,-4
1c0141ee:	00460693          	addi	a3,a2,4
1c0141f2:	00189e13          	slli	t3,a7,0x1
1c0141f6:	8d7d                	and	a0,a0,a5
1c0141f8:	34905e63          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c0141fc:	34482c63          	p.beqimm	a6,4,1c014554 <pos_libc_prf+0x1910>
1c014200:	00389693          	slli	a3,a7,0x3
1c014204:	01ee5793          	srli	a5,t3,0x1e
1c014208:	00251e93          	slli	t4,a0,0x2
1c01420c:	01c688b3          	add	a7,a3,t3
1c014210:	01d7eeb3          	or	t4,a5,t4
1c014214:	9576                	add	a0,a0,t4
1c014216:	00d8b7b3          	sltu	a5,a7,a3
1c01421a:	97aa                	add	a5,a5,a0
1c01421c:	01f8d693          	srli	a3,a7,0x1f
1c014220:	0786                	slli	a5,a5,0x1
1c014222:	8fd5                	or	a5,a5,a3
1c014224:	01c7d693          	srli	a3,a5,0x1c
1c014228:	03068693          	addi	a3,a3,48
1c01422c:	00d60223          	sb	a3,4(a2)
1c014230:	ffb58493          	addi	s1,a1,-5
1c014234:	00560693          	addi	a3,a2,5
1c014238:	00189e13          	slli	t3,a7,0x1
1c01423c:	30905c63          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c014240:	30582a63          	p.beqimm	a6,5,1c014554 <pos_libc_prf+0x1910>
1c014244:	10000537          	lui	a0,0x10000
1c014248:	157d                	addi	a0,a0,-1
1c01424a:	8fe9                	and	a5,a5,a0
1c01424c:	01ee5693          	srli	a3,t3,0x1e
1c014250:	002e1893          	slli	a7,t3,0x2
1c014254:	00279e93          	slli	t4,a5,0x2
1c014258:	9e46                	add	t3,t3,a7
1c01425a:	01d6eeb3          	or	t4,a3,t4
1c01425e:	97f6                	add	a5,a5,t4
1c014260:	011e38b3          	sltu	a7,t3,a7
1c014264:	98be                	add	a7,a7,a5
1c014266:	0886                	slli	a7,a7,0x1
1c014268:	01fe5793          	srli	a5,t3,0x1f
1c01426c:	0117e8b3          	or	a7,a5,a7
1c014270:	01c8d793          	srli	a5,a7,0x1c
1c014274:	03078793          	addi	a5,a5,48
1c014278:	00f602a3          	sb	a5,5(a2)
1c01427c:	ffa58493          	addi	s1,a1,-6
1c014280:	00660693          	addi	a3,a2,6
1c014284:	001e1793          	slli	a5,t3,0x1
1c014288:	00a8f8b3          	and	a7,a7,a0
1c01428c:	2c905463          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c014290:	2c682263          	p.beqimm	a6,6,1c014554 <pos_libc_prf+0x1910>
1c014294:	003e1693          	slli	a3,t3,0x3
1c014298:	01e7de93          	srli	t4,a5,0x1e
1c01429c:	00289e13          	slli	t3,a7,0x2
1c0142a0:	97b6                	add	a5,a5,a3
1c0142a2:	01ceee33          	or	t3,t4,t3
1c0142a6:	98f2                	add	a7,a7,t3
1c0142a8:	00d7b6b3          	sltu	a3,a5,a3
1c0142ac:	96c6                	add	a3,a3,a7
1c0142ae:	00169e13          	slli	t3,a3,0x1
1c0142b2:	01f7d893          	srli	a7,a5,0x1f
1c0142b6:	01c8ee33          	or	t3,a7,t3
1c0142ba:	01ce5693          	srli	a3,t3,0x1c
1c0142be:	03068693          	addi	a3,a3,48
1c0142c2:	00d60323          	sb	a3,6(a2)
1c0142c6:	ff958493          	addi	s1,a1,-7
1c0142ca:	00760693          	addi	a3,a2,7
1c0142ce:	00179893          	slli	a7,a5,0x1
1c0142d2:	00ae7e33          	and	t3,t3,a0
1c0142d6:	26905f63          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c0142da:	26782d63          	p.beqimm	a6,7,1c014554 <pos_libc_prf+0x1910>
1c0142de:	078e                	slli	a5,a5,0x3
1c0142e0:	01e8de93          	srli	t4,a7,0x1e
1c0142e4:	002e1693          	slli	a3,t3,0x2
1c0142e8:	98be                	add	a7,a7,a5
1c0142ea:	00dee6b3          	or	a3,t4,a3
1c0142ee:	9e36                	add	t3,t3,a3
1c0142f0:	00f8b7b3          	sltu	a5,a7,a5
1c0142f4:	97f2                	add	a5,a5,t3
1c0142f6:	01f8d693          	srli	a3,a7,0x1f
1c0142fa:	00179e13          	slli	t3,a5,0x1
1c0142fe:	01c6ee33          	or	t3,a3,t3
1c014302:	01ce5793          	srli	a5,t3,0x1c
1c014306:	03078793          	addi	a5,a5,48
1c01430a:	00f603a3          	sb	a5,7(a2)
1c01430e:	ff858493          	addi	s1,a1,-8
1c014312:	00860693          	addi	a3,a2,8
1c014316:	00189793          	slli	a5,a7,0x1
1c01431a:	00ae7e33          	and	t3,t3,a0
1c01431e:	22905b63          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c014322:	22882963          	p.beqimm	a6,8,1c014554 <pos_libc_prf+0x1910>
1c014326:	00389693          	slli	a3,a7,0x3
1c01432a:	01e7de93          	srli	t4,a5,0x1e
1c01432e:	002e1893          	slli	a7,t3,0x2
1c014332:	97b6                	add	a5,a5,a3
1c014334:	011ee8b3          	or	a7,t4,a7
1c014338:	9e46                	add	t3,t3,a7
1c01433a:	00d7b6b3          	sltu	a3,a5,a3
1c01433e:	96f2                	add	a3,a3,t3
1c014340:	01f7d893          	srli	a7,a5,0x1f
1c014344:	00169e13          	slli	t3,a3,0x1
1c014348:	01c8ee33          	or	t3,a7,t3
1c01434c:	01ce5693          	srli	a3,t3,0x1c
1c014350:	03068693          	addi	a3,a3,48
1c014354:	00d60423          	sb	a3,8(a2)
1c014358:	ff758493          	addi	s1,a1,-9
1c01435c:	00960693          	addi	a3,a2,9
1c014360:	00179893          	slli	a7,a5,0x1
1c014364:	00ae7e33          	and	t3,t3,a0
1c014368:	1e905663          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c01436c:	1e982463          	p.beqimm	a6,9,1c014554 <pos_libc_prf+0x1910>
1c014370:	078e                	slli	a5,a5,0x3
1c014372:	01e8de93          	srli	t4,a7,0x1e
1c014376:	002e1693          	slli	a3,t3,0x2
1c01437a:	98be                	add	a7,a7,a5
1c01437c:	00dee6b3          	or	a3,t4,a3
1c014380:	00f8b7b3          	sltu	a5,a7,a5
1c014384:	9e36                	add	t3,t3,a3
1c014386:	97f2                	add	a5,a5,t3
1c014388:	01f8d693          	srli	a3,a7,0x1f
1c01438c:	0786                	slli	a5,a5,0x1
1c01438e:	8fd5                	or	a5,a5,a3
1c014390:	01c7d693          	srli	a3,a5,0x1c
1c014394:	03068693          	addi	a3,a3,48
1c014398:	00d604a3          	sb	a3,9(a2)
1c01439c:	ff658493          	addi	s1,a1,-10
1c0143a0:	00a60693          	addi	a3,a2,10
1c0143a4:	0886                	slli	a7,a7,0x1
1c0143a6:	8fe9                	and	a5,a5,a0
1c0143a8:	1a905663          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c0143ac:	1aa82463          	p.beqimm	a6,10,1c014554 <pos_libc_prf+0x1910>
1c0143b0:	00289513          	slli	a0,a7,0x2
1c0143b4:	01e8d693          	srli	a3,a7,0x1e
1c0143b8:	00279e93          	slli	t4,a5,0x2
1c0143bc:	01150e33          	add	t3,a0,a7
1c0143c0:	01d6eeb3          	or	t4,a3,t4
1c0143c4:	00ae38b3          	sltu	a7,t3,a0
1c0143c8:	97f6                	add	a5,a5,t4
1c0143ca:	98be                	add	a7,a7,a5
1c0143cc:	0886                	slli	a7,a7,0x1
1c0143ce:	01fe5793          	srli	a5,t3,0x1f
1c0143d2:	0117e8b3          	or	a7,a5,a7
1c0143d6:	01c8d793          	srli	a5,a7,0x1c
1c0143da:	03078793          	addi	a5,a5,48
1c0143de:	10000537          	lui	a0,0x10000
1c0143e2:	157d                	addi	a0,a0,-1
1c0143e4:	00f60523          	sb	a5,10(a2)
1c0143e8:	ff558493          	addi	s1,a1,-11
1c0143ec:	00b60693          	addi	a3,a2,11
1c0143f0:	001e1793          	slli	a5,t3,0x1
1c0143f4:	00a8f8b3          	and	a7,a7,a0
1c0143f8:	14905e63          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c0143fc:	14b82c63          	p.beqimm	a6,11,1c014554 <pos_libc_prf+0x1910>
1c014400:	003e1693          	slli	a3,t3,0x3
1c014404:	01e7de93          	srli	t4,a5,0x1e
1c014408:	00289e13          	slli	t3,a7,0x2
1c01440c:	97b6                	add	a5,a5,a3
1c01440e:	01ceee33          	or	t3,t4,t3
1c014412:	98f2                	add	a7,a7,t3
1c014414:	00d7b6b3          	sltu	a3,a5,a3
1c014418:	96c6                	add	a3,a3,a7
1c01441a:	00169e13          	slli	t3,a3,0x1
1c01441e:	01f7d893          	srli	a7,a5,0x1f
1c014422:	01c8ee33          	or	t3,a7,t3
1c014426:	01ce5693          	srli	a3,t3,0x1c
1c01442a:	03068693          	addi	a3,a3,48
1c01442e:	00d605a3          	sb	a3,11(a2)
1c014432:	ff458493          	addi	s1,a1,-12
1c014436:	00c60693          	addi	a3,a2,12
1c01443a:	00179893          	slli	a7,a5,0x1
1c01443e:	10905b63          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c014442:	10c82963          	p.beqimm	a6,12,1c014554 <pos_libc_prf+0x1910>
1c014446:	00ae7e33          	and	t3,t3,a0
1c01444a:	078e                	slli	a5,a5,0x3
1c01444c:	01e8de93          	srli	t4,a7,0x1e
1c014450:	002e1693          	slli	a3,t3,0x2
1c014454:	98be                	add	a7,a7,a5
1c014456:	00dee6b3          	or	a3,t4,a3
1c01445a:	9e36                	add	t3,t3,a3
1c01445c:	00f8b7b3          	sltu	a5,a7,a5
1c014460:	97f2                	add	a5,a5,t3
1c014462:	01f8d693          	srli	a3,a7,0x1f
1c014466:	00179e13          	slli	t3,a5,0x1
1c01446a:	01c6ee33          	or	t3,a3,t3
1c01446e:	01ce5793          	srli	a5,t3,0x1c
1c014472:	03078793          	addi	a5,a5,48
1c014476:	00f60623          	sb	a5,12(a2)
1c01447a:	ff358493          	addi	s1,a1,-13
1c01447e:	00d60693          	addi	a3,a2,13
1c014482:	00189793          	slli	a5,a7,0x1
1c014486:	00ae7e33          	and	t3,t3,a0
1c01448a:	0c905563          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c01448e:	0cd82363          	p.beqimm	a6,13,1c014554 <pos_libc_prf+0x1910>
1c014492:	00389693          	slli	a3,a7,0x3
1c014496:	01e7de93          	srli	t4,a5,0x1e
1c01449a:	002e1893          	slli	a7,t3,0x2
1c01449e:	97b6                	add	a5,a5,a3
1c0144a0:	011ee8b3          	or	a7,t4,a7
1c0144a4:	9e46                	add	t3,t3,a7
1c0144a6:	00d7b6b3          	sltu	a3,a5,a3
1c0144aa:	96f2                	add	a3,a3,t3
1c0144ac:	01f7d893          	srli	a7,a5,0x1f
1c0144b0:	00169e13          	slli	t3,a3,0x1
1c0144b4:	01c8ee33          	or	t3,a7,t3
1c0144b8:	01ce5693          	srli	a3,t3,0x1c
1c0144bc:	03068693          	addi	a3,a3,48
1c0144c0:	00d606a3          	sb	a3,13(a2)
1c0144c4:	ff258493          	addi	s1,a1,-14
1c0144c8:	00e60693          	addi	a3,a2,14
1c0144cc:	00179893          	slli	a7,a5,0x1
1c0144d0:	00ae7e33          	and	t3,t3,a0
1c0144d4:	08905063          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c0144d8:	06e82e63          	p.beqimm	a6,14,1c014554 <pos_libc_prf+0x1910>
1c0144dc:	078e                	slli	a5,a5,0x3
1c0144de:	01e8de93          	srli	t4,a7,0x1e
1c0144e2:	002e1693          	slli	a3,t3,0x2
1c0144e6:	98be                	add	a7,a7,a5
1c0144e8:	00dee6b3          	or	a3,t4,a3
1c0144ec:	9e36                	add	t3,t3,a3
1c0144ee:	00f8b7b3          	sltu	a5,a7,a5
1c0144f2:	97f2                	add	a5,a5,t3
1c0144f4:	01f8d693          	srli	a3,a7,0x1f
1c0144f8:	00179e13          	slli	t3,a5,0x1
1c0144fc:	01c6ee33          	or	t3,a3,t3
1c014500:	01ce5793          	srli	a5,t3,0x1c
1c014504:	03078793          	addi	a5,a5,48
1c014508:	00f60723          	sb	a5,14(a2)
1c01450c:	ff158493          	addi	s1,a1,-15
1c014510:	00f60693          	addi	a3,a2,15
1c014514:	00189793          	slli	a5,a7,0x1
1c014518:	00ae7533          	and	a0,t3,a0
1c01451c:	02905c63          	blez	s1,1c014554 <pos_libc_prf+0x1910>
1c014520:	48c1                	li	a7,16
1c014522:	03181963          	bne	a6,a7,1c014554 <pos_libc_prf+0x1910>
1c014526:	00279813          	slli	a6,a5,0x2
1c01452a:	01e7d893          	srli	a7,a5,0x1e
1c01452e:	00251693          	slli	a3,a0,0x2
1c014532:	00d8e6b3          	or	a3,a7,a3
1c014536:	97c2                	add	a5,a5,a6
1c014538:	9536                	add	a0,a0,a3
1c01453a:	0107b7b3          	sltu	a5,a5,a6
1c01453e:	97aa                	add	a5,a5,a0
1c014540:	0786                	slli	a5,a5,0x1
1c014542:	83f1                	srli	a5,a5,0x1c
1c014544:	03078793          	addi	a5,a5,48
1c014548:	01060693          	addi	a3,a2,16
1c01454c:	00f607a3          	sb	a5,15(a2)
1c014550:	ff058493          	addi	s1,a1,-16
1c014554:	57a2                	lw	a5,40(sp)
1c014556:	5712                	lw	a4,36(sp)
1c014558:	00e785b3          	add	a1,a5,a4
1c01455c:	bc029863          	bnez	t0,1c01392c <pos_libc_prf+0xce8>
1c014560:	95a6                	add	a1,a1,s1
1c014562:	8a26                	mv	s4,s1
1c014564:	bf4ff06f          	j	1c013958 <pos_libc_prf+0xd14>
1c014568:	1850                	addi	a2,sp,52
1c01456a:	4581                	li	a1,0
1c01456c:	8732                	mv	a4,a2
1c01456e:	efaff06f          	j	1c013c68 <pos_libc_prf+0x1024>
1c014572:	4741                	li	a4,16
1c014574:	04e4ceb3          	p.min	t4,s1,a4
1c014578:	1efd                	addi	t4,t4,-1
1c01457a:	84b6                	mv	s1,a3
1c01457c:	4281                	li	t0,0
1c01457e:	ebdfe06f          	j	1c01343a <pos_libc_prf+0x7f6>
1c014582:	5ca2                	lw	s9,40(sp)
1c014584:	0b905063          	blez	s9,1c014624 <pos_libc_prf+0x19e0>
1c014588:	85ea                	mv	a1,s10
1c01458a:	03000513          	li	a0,48
1c01458e:	9b82                	jalr	s7
1c014590:	1cfd                	addi	s9,s9,-1
1c014592:	fff539e3          	p.bneimm	a0,-1,1c014584 <pos_libc_prf+0x1940>
1c014596:	547d                	li	s0,-1
1c014598:	f62fe06f          	j	1c012cfa <pos_libc_prf+0xb6>
1c01459c:	02d00793          	li	a5,45
1c0145a0:	40600333          	neg	t1,t1
1c0145a4:	00f680a3          	sb	a5,1(a3)
1c0145a8:	d14ff06f          	j	1c013abc <pos_libc_prf+0xe78>
1c0145ac:	41f45713          	srai	a4,s0,0x1f
1c0145b0:	c380                	sw	s0,0(a5)
1c0145b2:	c3d8                	sw	a4,4(a5)
1c0145b4:	ed4fe06f          	j	1c012c88 <pos_libc_prf+0x44>
1c0145b8:	5792                	lw	a5,36(sp)
1c0145ba:	40f989b3          	sub	s3,s3,a5
1c0145be:	d81fe06f          	j	1c01333e <pos_libc_prf+0x6fa>
1c0145c2:	02010aa3          	sb	zero,53(sp)
1c0145c6:	c602                	sw	zero,12(sp)
1c0145c8:	4985                	li	s3,1
1c0145ca:	9d5fe06f          	j	1c012f9e <pos_libc_prf+0x35a>
1c0145ce:	4732                	lw	a4,12(sp)
1c0145d0:	03000613          	li	a2,48
1c0145d4:	00ce0023          	sb	a2,0(t3)
1c0145d8:	eb0d                	bnez	a4,1c01460a <pos_libc_prf+0x19c6>
1c0145da:	04905263          	blez	s1,1c01461e <pos_libc_prf+0x19da>
1c0145de:	02e00693          	li	a3,46
1c0145e2:	00de00a3          	sb	a3,1(t3)
1c0145e6:	4841                	li	a6,16
1c0145e8:	002e0693          	addi	a3,t3,2
1c0145ec:	0c030463          	beqz	t1,1c0146b4 <pos_libc_prf+0x1a70>
1c0145f0:	40600633          	neg	a2,t1
1c0145f4:	04964633          	p.min	a2,a2,s1
1c0145f8:	d432                	sw	a2,40(sp)
1c0145fa:	40c485b3          	sub	a1,s1,a2
1c0145fe:	d202                	sw	zero,36(sp)
1c014600:	8636                	mv	a2,a3
1c014602:	4841                	li	a6,16
1c014604:	aeb042e3          	bgtz	a1,1c0140e8 <pos_libc_prf+0x14a4>
1c014608:	a049                	j	1c01468a <pos_libc_prf+0x1a46>
1c01460a:	02e00693          	li	a3,46
1c01460e:	00de00a3          	sb	a3,1(t3)
1c014612:	002e0693          	addi	a3,t3,2
1c014616:	06030d63          	beqz	t1,1c014690 <pos_libc_prf+0x1a4c>
1c01461a:	fc904be3          	bgtz	s1,1c0145f0 <pos_libc_prf+0x19ac>
1c01461e:	d202                	sw	zero,36(sp)
1c014620:	d402                	sw	zero,40(sp)
1c014622:	bf0d                	j	1c014554 <pos_libc_prf+0x1910>
1c014624:	5722                	lw	a4,40(sp)
1c014626:	40e989b3          	sub	s3,s3,a4
1c01462a:	d19fe06f          	j	1c013342 <pos_libc_prf+0x6fe>
1c01462e:	46b2                	lw	a3,12(sp)
1c014630:	4281                	li	t0,0
1c014632:	e299                	bnez	a3,1c014638 <pos_libc_prf+0x19f4>
1c014634:	dfbfe06f          	j	1c01342e <pos_libc_prf+0x7ea>
1c014638:	4e81                	li	t4,0
1c01463a:	4281                	li	t0,0
1c01463c:	dfffe06f          	j	1c01343a <pos_libc_prf+0x7f6>
1c014640:	679d                	lui	a5,0x7
1c014642:	e6978793          	addi	a5,a5,-407 # 6e69 <_l1_preload_size+0x56a1>
1c014646:	00fe1023          	sh	a5,0(t3)
1c01464a:	06600793          	li	a5,102
1c01464e:	00fe0123          	sb	a5,2(t3)
1c014652:	be2ff06f          	j	1c013a34 <pos_libc_prf+0xdf0>
1c014656:	06600c93          	li	s9,102
1c01465a:	4681                	li	a3,0
1c01465c:	08000637          	lui	a2,0x8000
1c014660:	e43fe06f          	j	1c0134a2 <pos_libc_prf+0x85e>
1c014664:	6799                	lui	a5,0x6
1c014666:	16e78793          	addi	a5,a5,366 # 616e <_l1_preload_size+0x49a6>
1c01466a:	00fe1023          	sh	a5,0(t3)
1c01466e:	06e00793          	li	a5,110
1c014672:	00fe0123          	sb	a5,2(t3)
1c014676:	bbeff06f          	j	1c013a34 <pos_libc_prf+0xdf0>
1c01467a:	59b2                	lw	s3,44(sp)
1c01467c:	e0cfe06f          	j	1c012c88 <pos_libc_prf+0x44>
1c014680:	02904463          	bgtz	s1,1c0146a8 <pos_libc_prf+0x1a64>
1c014684:	4301                	li	t1,0
1c014686:	d402                	sw	zero,40(sp)
1c014688:	b5f1                	j	1c014554 <pos_libc_prf+0x1910>
1c01468a:	84ae                	mv	s1,a1
1c01468c:	86b2                	mv	a3,a2
1c01468e:	b5d9                	j	1c014554 <pos_libc_prf+0x1910>
1c014690:	f89057e3          	blez	s1,1c01461e <pos_libc_prf+0x19da>
1c014694:	8636                	mv	a2,a3
1c014696:	d202                	sw	zero,36(sp)
1c014698:	d402                	sw	zero,40(sp)
1c01469a:	85a6                	mv	a1,s1
1c01469c:	4841                	li	a6,16
1c01469e:	b4b9                	j	1c0140ec <pos_libc_prf+0x14a8>
1c0146a0:	c0200813          	li	a6,-1022
1c0146a4:	a17fe06f          	j	1c0130ba <pos_libc_prf+0x476>
1c0146a8:	5312                	lw	t1,36(sp)
1c0146aa:	02e00613          	li	a2,46
1c0146ae:	00c68023          	sb	a2,0(a3)
1c0146b2:	0685                	addi	a3,a3,1
1c0146b4:	d21a                	sw	t1,36(sp)
1c0146b6:	8636                	mv	a2,a3
1c0146b8:	85a6                	mv	a1,s1
1c0146ba:	d402                	sw	zero,40(sp)
1c0146bc:	4301                	li	t1,0
1c0146be:	b42d                	j	1c0140e8 <pos_libc_prf+0x14a4>
1c0146c0:	00e78693          	addi	a3,a5,14
1c0146c4:	a5eff06f          	j	1c013922 <pos_libc_prf+0xcde>
1c0146c8:	00c78693          	addi	a3,a5,12
1c0146cc:	a56ff06f          	j	1c013922 <pos_libc_prf+0xcde>
1c0146d0:	00a78693          	addi	a3,a5,10
1c0146d4:	a4eff06f          	j	1c013922 <pos_libc_prf+0xcde>
1c0146d8:	00878693          	addi	a3,a5,8
1c0146dc:	a46ff06f          	j	1c013922 <pos_libc_prf+0xcde>
1c0146e0:	00678693          	addi	a3,a5,6
1c0146e4:	a3eff06f          	j	1c013922 <pos_libc_prf+0xcde>
1c0146e8:	00478693          	addi	a3,a5,4
1c0146ec:	a36ff06f          	j	1c013922 <pos_libc_prf+0xcde>
1c0146f0:	00278693          	addi	a3,a5,2
1c0146f4:	a2eff06f          	j	1c013922 <pos_libc_prf+0xcde>
1c0146f8:	00fe0693          	addi	a3,t3,15
1c0146fc:	4805                	li	a6,1
1c0146fe:	b2f9                	j	1c0140cc <pos_libc_prf+0x1488>
1c014700:	00ee0693          	addi	a3,t3,14
1c014704:	4809                	li	a6,2
1c014706:	b2d9                	j	1c0140cc <pos_libc_prf+0x1488>
1c014708:	00de0693          	addi	a3,t3,13
1c01470c:	480d                	li	a6,3
1c01470e:	ba7d                	j	1c0140cc <pos_libc_prf+0x1488>
1c014710:	00ce0693          	addi	a3,t3,12
1c014714:	ba65                	j	1c0140cc <pos_libc_prf+0x1488>
1c014716:	00be0693          	addi	a3,t3,11
1c01471a:	4815                	li	a6,5
1c01471c:	ba45                	j	1c0140cc <pos_libc_prf+0x1488>
1c01471e:	00ae0693          	addi	a3,t3,10
1c014722:	4819                	li	a6,6
1c014724:	b265                	j	1c0140cc <pos_libc_prf+0x1488>
1c014726:	009e0693          	addi	a3,t3,9
1c01472a:	481d                	li	a6,7
1c01472c:	b245                	j	1c0140cc <pos_libc_prf+0x1488>
1c01472e:	008e0693          	addi	a3,t3,8
1c014732:	4821                	li	a6,8
1c014734:	ba61                	j	1c0140cc <pos_libc_prf+0x1488>
1c014736:	007e0693          	addi	a3,t3,7
1c01473a:	4825                	li	a6,9
1c01473c:	ba41                	j	1c0140cc <pos_libc_prf+0x1488>
1c01473e:	006e0693          	addi	a3,t3,6
1c014742:	4829                	li	a6,10
1c014744:	b261                	j	1c0140cc <pos_libc_prf+0x1488>
1c014746:	005e0693          	addi	a3,t3,5
1c01474a:	482d                	li	a6,11
1c01474c:	b241                	j	1c0140cc <pos_libc_prf+0x1488>
1c01474e:	004e0693          	addi	a3,t3,4
1c014752:	4831                	li	a6,12
1c014754:	baa5                	j	1c0140cc <pos_libc_prf+0x1488>
1c014756:	003e0693          	addi	a3,t3,3
1c01475a:	4835                	li	a6,13
1c01475c:	ba85                	j	1c0140cc <pos_libc_prf+0x1488>
1c01475e:	002e0693          	addi	a3,t3,2
1c014762:	d202                	sw	zero,36(sp)
1c014764:	4839                	li	a6,14
1c014766:	b29d                	j	1c0140cc <pos_libc_prf+0x1488>
1c014768:	d202                	sw	zero,36(sp)
1c01476a:	483d                	li	a6,15
1c01476c:	b285                	j	1c0140cc <pos_libc_prf+0x1488>
	...

1c014770 <pos_init_start>:
1c014770:	1141                	addi	sp,sp,-16
1c014772:	c422                	sw	s0,8(sp)
1c014774:	1c001437          	lui	s0,0x1c001
1c014778:	c606                	sw	ra,12(sp)
1c01477a:	84c40413          	addi	s0,s0,-1972 # 1c00084c <ctor_list>
1c01477e:	2c79                	jal	1c014a1c <pos_irq_init>
1c014780:	b58fe0ef          	jal	ra,1c012ad8 <pos_soc_init>
1c014784:	24e9                	jal	1c014a4e <pos_soc_event_init>
1c014786:	2a65                	jal	1c01493e <pos_allocs_init>
1c014788:	284d                	jal	1c01483a <pos_sched_init>
1c01478a:	405c                	lw	a5,4(s0)
1c01478c:	c791                	beqz	a5,1c014798 <pos_init_start+0x28>
1c01478e:	0421                	addi	s0,s0,8
1c014790:	9782                	jalr	a5
1c014792:	0044278b          	p.lw	a5,4(s0!)
1c014796:	ffed                	bnez	a5,1c014790 <pos_init_start+0x20>
1c014798:	c76fe0ef          	jal	ra,1c012c0e <pos_io_start>
1c01479c:	300467f3          	csrrsi	a5,mstatus,8
1c0147a0:	40b2                	lw	ra,12(sp)
1c0147a2:	4422                	lw	s0,8(sp)
1c0147a4:	0141                	addi	sp,sp,16
1c0147a6:	8082                	ret

1c0147a8 <pos_init_stop>:
1c0147a8:	1141                	addi	sp,sp,-16
1c0147aa:	c422                	sw	s0,8(sp)
1c0147ac:	1c001437          	lui	s0,0x1c001
1c0147b0:	c606                	sw	ra,12(sp)
1c0147b2:	86040413          	addi	s0,s0,-1952 # 1c000860 <dtor_list>
1c0147b6:	c5cfe0ef          	jal	ra,1c012c12 <pos_io_stop>
1c0147ba:	405c                	lw	a5,4(s0)
1c0147bc:	c791                	beqz	a5,1c0147c8 <pos_init_stop+0x20>
1c0147be:	0421                	addi	s0,s0,8
1c0147c0:	9782                	jalr	a5
1c0147c2:	0044278b          	p.lw	a5,4(s0!)
1c0147c6:	ffed                	bnez	a5,1c0147c0 <pos_init_stop+0x18>
1c0147c8:	40b2                	lw	ra,12(sp)
1c0147ca:	4422                	lw	s0,8(sp)
1c0147cc:	0141                	addi	sp,sp,16
1c0147ce:	8082                	ret

1c0147d0 <pos_cbsys_add>:
1c0147d0:	00259793          	slli	a5,a1,0x2
1c0147d4:	1c0015b7          	lui	a1,0x1c001
1c0147d8:	76058593          	addi	a1,a1,1888 # 1c001760 <stack>
1c0147dc:	95be                	add	a1,a1,a5
1c0147de:	419c                	lw	a5,0(a1)
1c0147e0:	c110                	sw	a2,0(a0)
1c0147e2:	c154                	sw	a3,4(a0)
1c0147e4:	c51c                	sw	a5,8(a0)
1c0147e6:	c188                	sw	a0,0(a1)
1c0147e8:	8082                	ret

1c0147ea <pi_open_from_conf>:
1c0147ea:	c14c                	sw	a1,4(a0)
1c0147ec:	8082                	ret

1c0147ee <pos_task_handle_blocking>:
1c0147ee:	4785                	li	a5,1
1c0147f0:	00f50a23          	sb	a5,20(a0) # 10000014 <__l1_heap_size+0xfff17e0>
1c0147f4:	8082                	ret

1c0147f6 <pos_task_handle>:
1c0147f6:	1141                	addi	sp,sp,-16
1c0147f8:	c422                	sw	s0,8(sp)
1c0147fa:	02000413          	li	s0,32
1c0147fe:	401c                	lw	a5,0(s0)
1c014800:	c606                	sw	ra,12(sp)
1c014802:	c385                	beqz	a5,1c014822 <pos_task_handle+0x2c>
1c014804:	4394                	lw	a3,0(a5)
1c014806:	43d8                	lw	a4,4(a5)
1c014808:	4788                	lw	a0,8(a5)
1c01480a:	c014                	sw	a3,0(s0)
1c01480c:	300467f3          	csrrsi	a5,mstatus,8
1c014810:	9702                	jalr	a4
1c014812:	300477f3          	csrrci	a5,mstatus,8
1c014816:	401c                	lw	a5,0(s0)
1c014818:	f7f5                	bnez	a5,1c014804 <pos_task_handle+0xe>
1c01481a:	40b2                	lw	ra,12(sp)
1c01481c:	4422                	lw	s0,8(sp)
1c01481e:	0141                	addi	sp,sp,16
1c014820:	8082                	ret
1c014822:	10500073          	wfi
1c014826:	300467f3          	csrrsi	a5,mstatus,8
1c01482a:	300477f3          	csrrci	a5,mstatus,8
1c01482e:	401c                	lw	a5,0(s0)
1c014830:	fbf1                	bnez	a5,1c014804 <pos_task_handle+0xe>
1c014832:	40b2                	lw	ra,12(sp)
1c014834:	4422                	lw	s0,8(sp)
1c014836:	0141                	addi	sp,sp,16
1c014838:	8082                	ret

1c01483a <pos_sched_init>:
1c01483a:	02002023          	sw	zero,32(zero) # 20 <pos_sched_first>
1c01483e:	8082                	ret

1c014840 <pos_alloc_init>:
1c014840:	00758793          	addi	a5,a1,7
1c014844:	c407b7b3          	p.bclr	a5,a5,2,0
1c014848:	40b785b3          	sub	a1,a5,a1
1c01484c:	c11c                	sw	a5,0(a0)
1c01484e:	8e0d                	sub	a2,a2,a1
1c014850:	00c05763          	blez	a2,1c01485e <pos_alloc_init+0x1e>
1c014854:	c4063633          	p.bclr	a2,a2,2,0
1c014858:	c390                	sw	a2,0(a5)
1c01485a:	0007a223          	sw	zero,4(a5)
1c01485e:	8082                	ret

1c014860 <pos_alloc>:
1c014860:	4110                	lw	a2,0(a0)
1c014862:	059d                	addi	a1,a1,7
1c014864:	c405b5b3          	p.bclr	a1,a1,2,0
1c014868:	ca31                	beqz	a2,1c0148bc <pos_alloc+0x5c>
1c01486a:	4218                	lw	a4,0(a2)
1c01486c:	425c                	lw	a5,4(a2)
1c01486e:	02b75963          	ble	a1,a4,1c0148a0 <pos_alloc+0x40>
1c014872:	cb81                	beqz	a5,1c014882 <pos_alloc+0x22>
1c014874:	4398                	lw	a4,0(a5)
1c014876:	43d4                	lw	a3,4(a5)
1c014878:	00b75763          	ble	a1,a4,1c014886 <pos_alloc+0x26>
1c01487c:	863e                	mv	a2,a5
1c01487e:	87b6                	mv	a5,a3
1c014880:	fbf5                	bnez	a5,1c014874 <pos_alloc+0x14>
1c014882:	853e                	mv	a0,a5
1c014884:	8082                	ret
1c014886:	00b70a63          	beq	a4,a1,1c01489a <pos_alloc+0x3a>
1c01488a:	00b78533          	add	a0,a5,a1
1c01488e:	8f0d                	sub	a4,a4,a1
1c014890:	c118                	sw	a4,0(a0)
1c014892:	c154                	sw	a3,4(a0)
1c014894:	c248                	sw	a0,4(a2)
1c014896:	853e                	mv	a0,a5
1c014898:	8082                	ret
1c01489a:	c254                	sw	a3,4(a2)
1c01489c:	853e                	mv	a0,a5
1c01489e:	8082                	ret
1c0148a0:	00e58b63          	beq	a1,a4,1c0148b6 <pos_alloc+0x56>
1c0148a4:	00b606b3          	add	a3,a2,a1
1c0148a8:	8f0d                	sub	a4,a4,a1
1c0148aa:	c2dc                	sw	a5,4(a3)
1c0148ac:	c298                	sw	a4,0(a3)
1c0148ae:	87b2                	mv	a5,a2
1c0148b0:	c114                	sw	a3,0(a0)
1c0148b2:	853e                	mv	a0,a5
1c0148b4:	8082                	ret
1c0148b6:	c11c                	sw	a5,0(a0)
1c0148b8:	87b2                	mv	a5,a2
1c0148ba:	b7e1                	j	1c014882 <pos_alloc+0x22>
1c0148bc:	4781                	li	a5,0
1c0148be:	b7d1                	j	1c014882 <pos_alloc+0x22>

1c0148c0 <pos_free>:
1c0148c0:	4118                	lw	a4,0(a0)
1c0148c2:	061d                	addi	a2,a2,7
1c0148c4:	c4063633          	p.bclr	a2,a2,2,0
1c0148c8:	c729                	beqz	a4,1c014912 <pos_free+0x52>
1c0148ca:	00b76663          	bltu	a4,a1,1c0148d6 <pos_free+0x16>
1c0148ce:	a0a1                	j	1c014916 <pos_free+0x56>
1c0148d0:	00b7f563          	bleu	a1,a5,1c0148da <pos_free+0x1a>
1c0148d4:	873e                	mv	a4,a5
1c0148d6:	435c                	lw	a5,4(a4)
1c0148d8:	ffe5                	bnez	a5,1c0148d0 <pos_free+0x10>
1c0148da:	00c586b3          	add	a3,a1,a2
1c0148de:	02d78463          	beq	a5,a3,1c014906 <pos_free+0x46>
1c0148e2:	c190                	sw	a2,0(a1)
1c0148e4:	c1dc                	sw	a5,4(a1)
1c0148e6:	cf11                	beqz	a4,1c014902 <pos_free+0x42>
1c0148e8:	431c                	lw	a5,0(a4)
1c0148ea:	00f706b3          	add	a3,a4,a5
1c0148ee:	00d58463          	beq	a1,a3,1c0148f6 <pos_free+0x36>
1c0148f2:	c34c                	sw	a1,4(a4)
1c0148f4:	8082                	ret
1c0148f6:	4194                	lw	a3,0(a1)
1c0148f8:	41d0                	lw	a2,4(a1)
1c0148fa:	97b6                	add	a5,a5,a3
1c0148fc:	c31c                	sw	a5,0(a4)
1c0148fe:	c350                	sw	a2,4(a4)
1c014900:	8082                	ret
1c014902:	c10c                	sw	a1,0(a0)
1c014904:	8082                	ret
1c014906:	4394                	lw	a3,0(a5)
1c014908:	43dc                	lw	a5,4(a5)
1c01490a:	9636                	add	a2,a2,a3
1c01490c:	c190                	sw	a2,0(a1)
1c01490e:	c1dc                	sw	a5,4(a1)
1c014910:	bfd9                	j	1c0148e6 <pos_free+0x26>
1c014912:	4781                	li	a5,0
1c014914:	b7d9                	j	1c0148da <pos_free+0x1a>
1c014916:	87ba                	mv	a5,a4
1c014918:	4701                	li	a4,0
1c01491a:	b7c1                	j	1c0148da <pos_free+0x1a>

1c01491c <pos_alloc_init_l1>:
1c01491c:	1c0017b7          	lui	a5,0x1c001
1c014920:	7c47a703          	lw	a4,1988(a5) # 1c0017c4 <pos_alloc_l1>
1c014924:	100017b7          	lui	a5,0x10001
1c014928:	01651593          	slli	a1,a0,0x16
1c01492c:	663d                	lui	a2,0xf
1c01492e:	050a                	slli	a0,a0,0x2
1c014930:	7cc78793          	addi	a5,a5,1996 # 100017cc <__l1_end>
1c014934:	83460613          	addi	a2,a2,-1996 # e834 <__l1_heap_size>
1c014938:	95be                	add	a1,a1,a5
1c01493a:	953a                	add	a0,a0,a4
1c01493c:	b711                	j	1c014840 <pos_alloc_init>

1c01493e <pos_allocs_init>:
1c01493e:	1141                	addi	sp,sp,-16
1c014940:	1c0015b7          	lui	a1,0x1c001
1c014944:	c606                	sw	ra,12(sp)
1c014946:	c422                	sw	s0,8(sp)
1c014948:	7f058793          	addi	a5,a1,2032 # 1c0017f0 <__l2_priv0_end>
1c01494c:	1c008637          	lui	a2,0x1c008
1c014950:	04c7c363          	blt	a5,a2,1c014996 <pos_allocs_init+0x58>
1c014954:	4581                	li	a1,0
1c014956:	4601                	li	a2,0
1c014958:	1c001437          	lui	s0,0x1c001
1c01495c:	7c840513          	addi	a0,s0,1992 # 1c0017c8 <pos_alloc_l2>
1c014960:	35c5                	jal	1c014840 <pos_alloc_init>
1c014962:	1c0185b7          	lui	a1,0x1c018
1c014966:	50c58793          	addi	a5,a1,1292 # 1c01850c <__l2_shared_end>
1c01496a:	1c080637          	lui	a2,0x1c080
1c01496e:	1c001537          	lui	a0,0x1c001
1c014972:	8e1d                	sub	a2,a2,a5
1c014974:	50c58593          	addi	a1,a1,1292
1c014978:	7cc50513          	addi	a0,a0,1996 # 1c0017cc <pos_alloc_l2+0x4>
1c01497c:	35d1                	jal	1c014840 <pos_alloc_init>
1c01497e:	7c840513          	addi	a0,s0,1992
1c014982:	4591                	li	a1,4
1c014984:	3df1                	jal	1c014860 <pos_alloc>
1c014986:	40b2                	lw	ra,12(sp)
1c014988:	4422                	lw	s0,8(sp)
1c01498a:	1c0017b7          	lui	a5,0x1c001
1c01498e:	7ca7a223          	sw	a0,1988(a5) # 1c0017c4 <pos_alloc_l1>
1c014992:	0141                	addi	sp,sp,16
1c014994:	8082                	ret
1c014996:	8e1d                	sub	a2,a2,a5
1c014998:	7f058593          	addi	a1,a1,2032
1c01499c:	bf75                	j	1c014958 <pos_allocs_init+0x1a>

1c01499e <pi_cl_l1_malloc>:
1c01499e:	4781                	li	a5,0
1c0149a0:	c509                	beqz	a0,1c0149aa <pi_cl_l1_malloc+0xc>
1c0149a2:	451c                	lw	a5,8(a0)
1c0149a4:	01c7c783          	lbu	a5,28(a5)
1c0149a8:	078a                	slli	a5,a5,0x2
1c0149aa:	1c001737          	lui	a4,0x1c001
1c0149ae:	7c472503          	lw	a0,1988(a4) # 1c0017c4 <pos_alloc_l1>
1c0149b2:	953e                	add	a0,a0,a5
1c0149b4:	b575                	j	1c014860 <pos_alloc>

1c0149b6 <pi_cl_l1_free>:
1c0149b6:	4781                	li	a5,0
1c0149b8:	c509                	beqz	a0,1c0149c2 <pi_cl_l1_free+0xc>
1c0149ba:	451c                	lw	a5,8(a0)
1c0149bc:	01c7c783          	lbu	a5,28(a5)
1c0149c0:	078a                	slli	a5,a5,0x2
1c0149c2:	1c001737          	lui	a4,0x1c001
1c0149c6:	7c472503          	lw	a0,1988(a4) # 1c0017c4 <pos_alloc_l1>
1c0149ca:	953e                	add	a0,a0,a5
1c0149cc:	bdd5                	j	1c0148c0 <pos_free>

1c0149ce <pos_irq_set_handler>:
1c0149ce:	f14027f3          	csrr	a5,mhartid
1c0149d2:	477d                	li	a4,31
1c0149d4:	ca5797b3          	p.extractu	a5,a5,5,5
1c0149d8:	02e78d63          	beq	a5,a4,1c014a12 <pos_irq_set_handler+0x44>
1c0149dc:	002007b7          	lui	a5,0x200
1c0149e0:	43b8                	lw	a4,64(a5)
1c0149e2:	050a                	slli	a0,a0,0x2
1c0149e4:	8d89                	sub	a1,a1,a0
1c0149e6:	8d99                	sub	a1,a1,a4
1c0149e8:	c14586b3          	p.extract	a3,a1,0,20
1c0149ec:	06f00793          	li	a5,111
1c0149f0:	c1f6a7b3          	p.insert	a5,a3,0,31
1c0149f4:	d21586b3          	p.extract	a3,a1,9,1
1c0149f8:	d356a7b3          	p.insert	a5,a3,9,21
1c0149fc:	c0b586b3          	p.extract	a3,a1,0,11
1c014a00:	c146a7b3          	p.insert	a5,a3,0,20
1c014a04:	cec585b3          	p.extract	a1,a1,7,12
1c014a08:	cec5a7b3          	p.insert	a5,a1,7,12
1c014a0c:	00f56723          	p.sw	a5,a4(a0)
1c014a10:	8082                	ret
1c014a12:	30502773          	csrr	a4,mtvec
1c014a16:	c0073733          	p.bclr	a4,a4,0,0
1c014a1a:	b7e1                	j	1c0149e2 <pos_irq_set_handler+0x14>

1c014a1c <pos_irq_init>:
1c014a1c:	f1402773          	csrr	a4,mhartid
1c014a20:	1a10a7b7          	lui	a5,0x1a10a
1c014a24:	56fd                	li	a3,-1
1c014a26:	80d7a423          	sw	a3,-2040(a5) # 1a109808 <__l1_end+0xa10803c>
1c014a2a:	ca571733          	p.extractu	a4,a4,5,5
1c014a2e:	1c0087b7          	lui	a5,0x1c008
1c014a32:	46fd                	li	a3,31
1c014a34:	00078793          	mv	a5,a5
1c014a38:	00d70663          	beq	a4,a3,1c014a44 <pos_irq_init+0x28>
1c014a3c:	00200737          	lui	a4,0x200
1c014a40:	c33c                	sw	a5,64(a4)
1c014a42:	8082                	ret
1c014a44:	c007c7b3          	p.bset	a5,a5,0,0
1c014a48:	30579073          	csrw	mtvec,a5
1c014a4c:	8082                	ret

1c014a4e <pos_soc_event_init>:
1c014a4e:	1a1067b7          	lui	a5,0x1a106
1c014a52:	1141                	addi	sp,sp,-16
1c014a54:	c606                	sw	ra,12(sp)
1c014a56:	577d                	li	a4,-1
1c014a58:	00478693          	addi	a3,a5,4 # 1a106004 <__l1_end+0xa104838>
1c014a5c:	c298                	sw	a4,0(a3)
1c014a5e:	00878693          	addi	a3,a5,8
1c014a62:	c298                	sw	a4,0(a3)
1c014a64:	00c78693          	addi	a3,a5,12
1c014a68:	c298                	sw	a4,0(a3)
1c014a6a:	01078693          	addi	a3,a5,16
1c014a6e:	c298                	sw	a4,0(a3)
1c014a70:	01478693          	addi	a3,a5,20
1c014a74:	c298                	sw	a4,0(a3)
1c014a76:	01878693          	addi	a3,a5,24
1c014a7a:	c298                	sw	a4,0(a3)
1c014a7c:	01c78693          	addi	a3,a5,28
1c014a80:	c298                	sw	a4,0(a3)
1c014a82:	02078793          	addi	a5,a5,32
1c014a86:	1c0155b7          	lui	a1,0x1c015
1c014a8a:	c398                	sw	a4,0(a5)
1c014a8c:	0bc58593          	addi	a1,a1,188 # 1c0150bc <pos_soc_event_handler_asm>
1c014a90:	4569                	li	a0,26
1c014a92:	3f35                	jal	1c0149ce <pos_irq_set_handler>
1c014a94:	40b2                	lw	ra,12(sp)
1c014a96:	1a10a7b7          	lui	a5,0x1a10a
1c014a9a:	04000737          	lui	a4,0x4000
1c014a9e:	80e7a223          	sw	a4,-2044(a5) # 1a109804 <__l1_end+0xa108038>
1c014aa2:	0141                	addi	sp,sp,16
1c014aa4:	8082                	ret

1c014aa6 <pos_time_poweroff>:
1c014aa6:	1a10b7b7          	lui	a5,0x1a10b
1c014aaa:	0791                	addi	a5,a5,4
1c014aac:	0087a783          	lw	a5,8(a5) # 1a10b008 <__l1_end+0xa10983c>
1c014ab0:	1c001737          	lui	a4,0x1c001
1c014ab4:	76f72c23          	sw	a5,1912(a4) # 1c001778 <pos_time_timer_count>
1c014ab8:	4501                	li	a0,0
1c014aba:	8082                	ret

1c014abc <pos_time_poweron>:
1c014abc:	1c0017b7          	lui	a5,0x1c001
1c014ac0:	7787a703          	lw	a4,1912(a5) # 1c001778 <pos_time_timer_count>
1c014ac4:	1a10b7b7          	lui	a5,0x1a10b
1c014ac8:	0791                	addi	a5,a5,4
1c014aca:	00e7a423          	sw	a4,8(a5) # 1a10b008 <__l1_end+0xa10983c>
1c014ace:	4501                	li	a0,0
1c014ad0:	8082                	ret

1c014ad2 <pos_time_timer_handler>:
1c014ad2:	1c0018b7          	lui	a7,0x1c001
1c014ad6:	77888893          	addi	a7,a7,1912 # 1c001778 <pos_time_timer_count>
1c014ada:	0048a783          	lw	a5,4(a7)
1c014ade:	1a10b637          	lui	a2,0x1a10b
1c014ae2:	0611                	addi	a2,a2,4
1c014ae4:	00862603          	lw	a2,8(a2) # 1a10b008 <__l1_end+0xa10983c>
1c014ae8:	cba9                	beqz	a5,1c014b3a <pos_time_timer_handler+0x68>
1c014aea:	5fd8                	lw	a4,60(a5)
1c014aec:	800005b7          	lui	a1,0x80000
1c014af0:	ffe5c593          	xori	a1,a1,-2
1c014af4:	40e60733          	sub	a4,a2,a4
1c014af8:	0ae5e763          	bltu	a1,a4,1c014ba6 <pos_time_timer_handler+0xd4>
1c014afc:	02002803          	lw	a6,32(zero) # 20 <pos_sched_first>
1c014b00:	02402503          	lw	a0,36(zero) # 24 <pos_sched_last>
1c014b04:	4398                	lw	a4,0(a5)
1c014b06:	0007a023          	sw	zero,0(a5)
1c014b0a:	4301                	li	t1,0
1c014b0c:	02080063          	beqz	a6,1c014b2c <pos_time_timer_handler+0x5a>
1c014b10:	c11c                	sw	a5,0(a0)
1c014b12:	c305                	beqz	a4,1c014b32 <pos_time_timer_handler+0x60>
1c014b14:	5f54                	lw	a3,60(a4)
1c014b16:	853e                	mv	a0,a5
1c014b18:	40d606b3          	sub	a3,a2,a3
1c014b1c:	04d5e763          	bltu	a1,a3,1c014b6a <pos_time_timer_handler+0x98>
1c014b20:	87ba                	mv	a5,a4
1c014b22:	4398                	lw	a4,0(a5)
1c014b24:	0007a023          	sw	zero,0(a5)
1c014b28:	fe0814e3          	bnez	a6,1c014b10 <pos_time_timer_handler+0x3e>
1c014b2c:	883e                	mv	a6,a5
1c014b2e:	4305                	li	t1,1
1c014b30:	f375                	bnez	a4,1c014b14 <pos_time_timer_handler+0x42>
1c014b32:	06031c63          	bnez	t1,1c014baa <pos_time_timer_handler+0xd8>
1c014b36:	02f02223          	sw	a5,36(zero) # 24 <pos_sched_last>
1c014b3a:	1a10b7b7          	lui	a5,0x1a10b
1c014b3e:	08100713          	li	a4,129
1c014b42:	0791                	addi	a5,a5,4
1c014b44:	0008a223          	sw	zero,4(a7)
1c014b48:	00e7a023          	sw	a4,0(a5) # 1a10b000 <__l1_end+0xa109834>
1c014b4c:	f14027f3          	csrr	a5,mhartid
1c014b50:	477d                	li	a4,31
1c014b52:	ca5797b3          	p.extractu	a5,a5,5,5
1c014b56:	04e78063          	beq	a5,a4,1c014b96 <pos_time_timer_handler+0xc4>
1c014b5a:	6785                	lui	a5,0x1
1c014b5c:	80078793          	addi	a5,a5,-2048 # 800 <pos_soc_event_callback+0x3b8>
1c014b60:	00204737          	lui	a4,0x204
1c014b64:	02f72423          	sw	a5,40(a4) # 204028 <__l1_heap_size+0x1f57f4>
1c014b68:	8082                	ret
1c014b6a:	04031363          	bnez	t1,1c014bb0 <pos_time_timer_handler+0xde>
1c014b6e:	02f02223          	sw	a5,36(zero) # 24 <pos_sched_last>
1c014b72:	1a10b7b7          	lui	a5,0x1a10b
1c014b76:	0791                	addi	a5,a5,4
1c014b78:	00e8a223          	sw	a4,4(a7)
1c014b7c:	0087a683          	lw	a3,8(a5) # 1a10b008 <__l1_end+0xa10983c>
1c014b80:	5f58                	lw	a4,60(a4)
1c014b82:	40c70633          	sub	a2,a4,a2
1c014b86:	9636                	add	a2,a2,a3
1c014b88:	00c7a823          	sw	a2,16(a5)
1c014b8c:	08500713          	li	a4,133
1c014b90:	00e7a023          	sw	a4,0(a5)
1c014b94:	8082                	ret
1c014b96:	6785                	lui	a5,0x1
1c014b98:	1a10a737          	lui	a4,0x1a10a
1c014b9c:	80078793          	addi	a5,a5,-2048 # 800 <pos_soc_event_callback+0x3b8>
1c014ba0:	80f72a23          	sw	a5,-2028(a4) # 1a109814 <__l1_end+0xa108048>
1c014ba4:	8082                	ret
1c014ba6:	873e                	mv	a4,a5
1c014ba8:	b7e9                	j	1c014b72 <pos_time_timer_handler+0xa0>
1c014baa:	03002023          	sw	a6,32(zero) # 20 <pos_sched_first>
1c014bae:	b761                	j	1c014b36 <pos_time_timer_handler+0x64>
1c014bb0:	03002023          	sw	a6,32(zero) # 20 <pos_sched_first>
1c014bb4:	bf6d                	j	1c014b6e <pos_time_timer_handler+0x9c>

1c014bb6 <pos_time_init>:
1c014bb6:	1141                	addi	sp,sp,-16
1c014bb8:	c422                	sw	s0,8(sp)
1c014bba:	1a10b7b7          	lui	a5,0x1a10b
1c014bbe:	1c001437          	lui	s0,0x1c001
1c014bc2:	c606                	sw	ra,12(sp)
1c014bc4:	77840413          	addi	s0,s0,1912 # 1c001778 <pos_time_timer_count>
1c014bc8:	08300713          	li	a4,131
1c014bcc:	0791                	addi	a5,a5,4
1c014bce:	00042223          	sw	zero,4(s0)
1c014bd2:	00e7a023          	sw	a4,0(a5) # 1a10b000 <__l1_end+0xa109834>
1c014bd6:	1c0155b7          	lui	a1,0x1c015
1c014bda:	09858593          	addi	a1,a1,152 # 1c015098 <pos_time_timer_handler_asm>
1c014bde:	452d                	li	a0,11
1c014be0:	33fd                	jal	1c0149ce <pos_irq_set_handler>
1c014be2:	6785                	lui	a5,0x1
1c014be4:	1a10a737          	lui	a4,0x1a10a
1c014be8:	80078793          	addi	a5,a5,-2048 # 800 <pos_soc_event_callback+0x3b8>
1c014bec:	1c015637          	lui	a2,0x1c015
1c014bf0:	00840513          	addi	a0,s0,8
1c014bf4:	4681                	li	a3,0
1c014bf6:	aa660613          	addi	a2,a2,-1370 # 1c014aa6 <pos_time_poweroff>
1c014bfa:	4589                	li	a1,2
1c014bfc:	80f72223          	sw	a5,-2044(a4) # 1a109804 <__l1_end+0xa108038>
1c014c00:	bd1ff0ef          	jal	ra,1c0147d0 <pos_cbsys_add>
1c014c04:	01840513          	addi	a0,s0,24
1c014c08:	4422                	lw	s0,8(sp)
1c014c0a:	40b2                	lw	ra,12(sp)
1c014c0c:	1c015637          	lui	a2,0x1c015
1c014c10:	4681                	li	a3,0
1c014c12:	abc60613          	addi	a2,a2,-1348 # 1c014abc <pos_time_poweron>
1c014c16:	458d                	li	a1,3
1c014c18:	0141                	addi	sp,sp,16
1c014c1a:	bb7ff06f          	j	1c0147d0 <pos_cbsys_add>

1c014c1e <pos_uart_init>:
1c014c1e:	1c0177b7          	lui	a5,0x1c017
1c014c22:	d007a623          	sw	zero,-756(a5) # 1c016d0c <pos_uart>
1c014c26:	8082                	ret

1c014c28 <pi_cluster_conf_init>:
1c014c28:	00052223          	sw	zero,4(a0)
1c014c2c:	00052a23          	sw	zero,20(a0)
1c014c30:	8082                	ret
	...

1c014c34 <pi_cluster_open>:
1c014c34:	7179                	addi	sp,sp,-48
1c014c36:	c85a                	sw	s6,16(sp)
1c014c38:	00452b03          	lw	s6,4(a0)
1c014c3c:	ca56                	sw	s5,20(sp)
1c014c3e:	1c0017b7          	lui	a5,0x1c001
1c014c42:	004b2a83          	lw	s5,4(s6)
1c014c46:	d422                	sw	s0,40(sp)
1c014c48:	c462                	sw	s8,8(sp)
1c014c4a:	7d078413          	addi	s0,a5,2000 # 1c0017d0 <pos_cluster>
1c014c4e:	005a9c13          	slli	s8,s5,0x5
1c014c52:	d606                	sw	ra,44(sp)
1c014c54:	d226                	sw	s1,36(sp)
1c014c56:	d04a                	sw	s2,32(sp)
1c014c58:	ce4e                	sw	s3,28(sp)
1c014c5a:	cc52                	sw	s4,24(sp)
1c014c5c:	c65e                	sw	s7,12(sp)
1c014c5e:	018404b3          	add	s1,s0,s8
1c014c62:	c504                	sw	s1,8(a0)
1c014c64:	1c0155b7          	lui	a1,0x1c015
1c014c68:	0ffafb93          	andi	s7,s5,255
1c014c6c:	00858593          	addi	a1,a1,8 # 1c015008 <pos_task_remote_enqueue>
1c014c70:	4505                	li	a0,1
1c014c72:	01748e23          	sb	s7,28(s1)
1c014c76:	d59ff0ef          	jal	ra,1c0149ce <pos_irq_set_handler>
1c014c7a:	040a8793          	addi	a5,s5,64
1c014c7e:	1a10a737          	lui	a4,0x1a10a
1c014c82:	01679993          	slli	s3,a5,0x16
1c014c86:	00400913          	li	s2,4
1c014c8a:	4789                	li	a5,2
1c014c8c:	80f72223          	sw	a5,-2044(a4) # 1a109804 <__l1_end+0xa108038>
1c014c90:	e6c93933          	p.bclr	s2,s2,19,12
1c014c94:	0004a023          	sw	zero,0(s1)
1c014c98:	994e                	add	s2,s2,s3
1c014c9a:	10001a37          	lui	s4,0x10001
1c014c9e:	00092023          	sw	zero,0(s2)
1c014ca2:	7c4a0a13          	addi	s4,s4,1988 # 100017c4 <pos_cluster_fc_task_lock>
1c014ca6:	4509                	li	a0,2
1c014ca8:	00092623          	sw	zero,12(s2)
1c014cac:	00092223          	sw	zero,4(s2)
1c014cb0:	000a2023          	sw	zero,0(s4)
1c014cb4:	d37fd0ef          	jal	ra,1c0129ea <pos_fll_init>
1c014cb8:	01400793          	li	a5,20
1c014cbc:	478c                	lw	a1,8(a5)
1c014cbe:	edf1                	bnez	a1,1c014d9a <pi_cluster_open+0x166>
1c014cc0:	c788                	sw	a0,8(a5)
1c014cc2:	002004b7          	lui	s1,0x200
1c014cc6:	94ce                	add	s1,s1,s3
1c014cc8:	4785                	li	a5,1
1c014cca:	02f4a023          	sw	a5,32(s1) # 200020 <__l1_heap_size+0x1f17ec>
1c014cce:	00400693          	li	a3,4
1c014cd2:	6785                	lui	a5,0x1
1c014cd4:	e6c6b6b3          	p.bclr	a3,a3,19,12
1c014cd8:	7c878793          	addi	a5,a5,1992 # 17c8 <_l1_preload_size>
1c014cdc:	96ce                	add	a3,a3,s3
1c014cde:	02f05763          	blez	a5,1c014d0c <pi_cluster_open+0xd8>
1c014ce2:	17fd                	addi	a5,a5,-1
1c014ce4:	1c017737          	lui	a4,0x1c017
1c014ce8:	6605                	lui	a2,0x1
1c014cea:	d4470713          	addi	a4,a4,-700 # 1c016d44 <_l1_preload_start_inL2>
1c014cee:	8389                	srli	a5,a5,0x2
1c014cf0:	7c460613          	addi	a2,a2,1988 # 17c4 <pos_soc_event_callback+0x137c>
1c014cf4:	55f5                	li	a1,-3
1c014cf6:	8e99                	sub	a3,a3,a4
1c014cf8:	0785                	addi	a5,a5,1
1c014cfa:	0ab64463          	blt	a2,a1,1c014da2 <pi_cluster_open+0x16e>
1c014cfe:	0001                	nop
1c014d00:	0057c0fb          	lp.setup	x1,a5,1c014d0a <pi_cluster_open+0xd6>
1c014d04:	4310                	lw	a2,0(a4)
1c014d06:	00c766a3          	p.sw	a2,a3(a4)
1c014d0a:	0711                	addi	a4,a4,4
1c014d0c:	018407b3          	add	a5,s0,s8
1c014d10:	01778e23          	sb	s7,28(a5)
1c014d14:	014b4683          	lbu	a3,20(s6)
1c014d18:	00201437          	lui	s0,0x201
1c014d1c:	e0440713          	addi	a4,s0,-508 # 200e04 <__l1_heap_size+0x1f25d0>
1c014d20:	974e                	add	a4,a4,s3
1c014d22:	cf98                	sw	a4,24(a5)
1c014d24:	cb98                	sw	a4,16(a5)
1c014d26:	0127a223          	sw	s2,4(a5)
1c014d2a:	00d78ea3          	sb	a3,29(a5)
1c014d2e:	8556                	mv	a0,s5
1c014d30:	0007a423          	sw	zero,8(a5)
1c014d34:	0007aa23          	sw	zero,20(a5)
1c014d38:	00078f23          	sb	zero,30(a5)
1c014d3c:	00da0223          	sb	a3,4(s4)
1c014d40:	bddff0ef          	jal	ra,1c01491c <pos_alloc_init_l1>
1c014d44:	40040793          	addi	a5,s0,1024
1c014d48:	97ce                	add	a5,a5,s3
1c014d4a:	577d                	li	a4,-1
1c014d4c:	00e7a023          	sw	a4,0(a5)
1c014d50:	1c0087b7          	lui	a5,0x1c008
1c014d54:	08078793          	addi	a5,a5,128 # 1c008080 <_start>
1c014d58:	04f4a023          	sw	a5,64(s1)
1c014d5c:	04f4a223          	sw	a5,68(s1)
1c014d60:	04f4a423          	sw	a5,72(s1)
1c014d64:	04f4a623          	sw	a5,76(s1)
1c014d68:	04f4a823          	sw	a5,80(s1)
1c014d6c:	04f4aa23          	sw	a5,84(s1)
1c014d70:	04f4ac23          	sw	a5,88(s1)
1c014d74:	04f4ae23          	sw	a5,92(s1)
1c014d78:	0ff00793          	li	a5,255
1c014d7c:	00f4a423          	sw	a5,8(s1)
1c014d80:	50b2                	lw	ra,44(sp)
1c014d82:	5422                	lw	s0,40(sp)
1c014d84:	5492                	lw	s1,36(sp)
1c014d86:	5902                	lw	s2,32(sp)
1c014d88:	49f2                	lw	s3,28(sp)
1c014d8a:	4a62                	lw	s4,24(sp)
1c014d8c:	4ad2                	lw	s5,20(sp)
1c014d8e:	4b42                	lw	s6,16(sp)
1c014d90:	4bb2                	lw	s7,12(sp)
1c014d92:	4c22                	lw	s8,8(sp)
1c014d94:	4501                	li	a0,0
1c014d96:	6145                	addi	sp,sp,48
1c014d98:	8082                	ret
1c014d9a:	4509                	li	a0,2
1c014d9c:	bf5fd0ef          	jal	ra,1c012990 <pos_fll_set_freq>
1c014da0:	b70d                	j	1c014cc2 <pi_cluster_open+0x8e>
1c014da2:	4785                	li	a5,1
1c014da4:	bfa9                	j	1c014cfe <pi_cluster_open+0xca>
	...

1c014da8 <pi_cluster_close>:
1c014da8:	4501                	li	a0,0
1c014daa:	8082                	ret

1c014dac <pos_cluster_task_set_stack>:
1c014dac:	1101                	addi	sp,sp,-32
1c014dae:	c84a                	sw	s2,16(sp)
1c014db0:	00852903          	lw	s2,8(a0)
1c014db4:	ca26                	sw	s1,20(sp)
1c014db6:	c64e                	sw	s3,12(sp)
1c014db8:	4785                	li	a5,1
1c014dba:	ce06                	sw	ra,28(sp)
1c014dbc:	cc22                	sw	s0,24(sp)
1c014dbe:	00f90f23          	sb	a5,30(s2)
1c014dc2:	45dc                	lw	a5,12(a1)
1c014dc4:	89aa                	mv	s3,a0
1c014dc6:	84ae                	mv	s1,a1
1c014dc8:	efa1                	bnez	a5,1c014e20 <pos_cluster_task_set_stack+0x74>
1c014dca:	6785                	lui	a5,0x1
1c014dcc:	80078793          	addi	a5,a5,-2048 # 800 <pos_soc_event_callback+0x3b8>
1c014dd0:	40000713          	li	a4,1024
1c014dd4:	c5dc                	sw	a5,12(a1)
1c014dd6:	c998                	sw	a4,16(a1)
1c014dd8:	40000693          	li	a3,1024
1c014ddc:	48d8                	lw	a4,20(s1)
1c014dde:	00892503          	lw	a0,8(s2)
1c014de2:	843e                	mv	s0,a5
1c014de4:	177d                	addi	a4,a4,-1
1c014de6:	42d70433          	p.mac	s0,a4,a3
1c014dea:	c909                	beqz	a0,1c014dfc <pos_cluster_task_set_stack+0x50>
1c014dec:	00c92603          	lw	a2,12(s2)
1c014df0:	00860f63          	beq	a2,s0,1c014e0e <pos_cluster_task_set_stack+0x62>
1c014df4:	85aa                	mv	a1,a0
1c014df6:	854e                	mv	a0,s3
1c014df8:	bbfff0ef          	jal	ra,1c0149b6 <pi_cl_l1_free>
1c014dfc:	00892623          	sw	s0,12(s2)
1c014e00:	85a2                	mv	a1,s0
1c014e02:	854e                	mv	a0,s3
1c014e04:	b9bff0ef          	jal	ra,1c01499e <pi_cl_l1_malloc>
1c014e08:	00a92423          	sw	a0,8(s2)
1c014e0c:	cd19                	beqz	a0,1c014e2a <pos_cluster_task_set_stack+0x7e>
1c014e0e:	c488                	sw	a0,8(s1)
1c014e10:	4501                	li	a0,0
1c014e12:	40f2                	lw	ra,28(sp)
1c014e14:	4462                	lw	s0,24(sp)
1c014e16:	44d2                	lw	s1,20(sp)
1c014e18:	4942                	lw	s2,16(sp)
1c014e1a:	49b2                	lw	s3,12(sp)
1c014e1c:	6105                	addi	sp,sp,32
1c014e1e:	8082                	ret
1c014e20:	4994                	lw	a3,16(a1)
1c014e22:	fecd                	bnez	a3,1c014ddc <pos_cluster_task_set_stack+0x30>
1c014e24:	c99c                	sw	a5,16(a1)
1c014e26:	86be                	mv	a3,a5
1c014e28:	bf55                	j	1c014ddc <pos_cluster_task_set_stack+0x30>
1c014e2a:	557d                	li	a0,-1
1c014e2c:	b7dd                	j	1c014e12 <pos_cluster_task_set_stack+0x66>

1c014e2e <pi_cluster_send_task_to_cl>:
1c014e2e:	715d                	addi	sp,sp,-80
1c014e30:	c2a6                	sw	s1,68(sp)
1c014e32:	4504                	lw	s1,8(a0)
1c014e34:	c4a2                	sw	s0,72(sp)
1c014e36:	c0ca                	sw	s2,64(sp)
1c014e38:	c686                	sw	ra,76(sp)
1c014e3a:	1c0147b7          	lui	a5,0x1c014
1c014e3e:	7ee78793          	addi	a5,a5,2030 # 1c0147ee <pos_task_handle_blocking>
1c014e42:	01d4c703          	lbu	a4,29(s1)
1c014e46:	c23e                	sw	a5,4(sp)
1c014e48:	00010793          	mv	a5,sp
1c014e4c:	00010a23          	sb	zero,20(sp)
1c014e50:	c43e                	sw	a5,8(sp)
1c014e52:	842e                	mv	s0,a1
1c014e54:	0044a903          	lw	s2,4(s1)
1c014e58:	459c                	lw	a5,8(a1)
1c014e5a:	04173c63          	p.bneimm	a4,1,1c014eb2 <pi_cluster_send_task_to_cl+0x84>
1c014e5e:	c3b5                	beqz	a5,1c014ec2 <pi_cluster_send_task_to_cl+0x94>
1c014e60:	00010793          	mv	a5,sp
1c014e64:	02042023          	sw	zero,32(s0)
1c014e68:	cc1c                	sw	a5,24(s0)
1c014e6a:	30047773          	csrrci	a4,mstatus,8
1c014e6e:	409c                	lw	a5,0(s1)
1c014e70:	c391                	beqz	a5,1c014e74 <pi_cluster_send_task_to_cl+0x46>
1c014e72:	d380                	sw	s0,32(a5)
1c014e74:	c080                	sw	s0,0(s1)
1c014e76:	00092783          	lw	a5,0(s2)
1c014e7a:	cb8d                	beqz	a5,1c014eac <pi_cluster_send_task_to_cl+0x7e>
1c014e7c:	4c9c                	lw	a5,24(s1)
1c014e7e:	0007a023          	sw	zero,0(a5)
1c014e82:	30071073          	csrw	mstatus,a4
1c014e86:	30047473          	csrrci	s0,mstatus,8
1c014e8a:	01410783          	lb	a5,20(sp)
1c014e8e:	e791                	bnez	a5,1c014e9a <pi_cluster_send_task_to_cl+0x6c>
1c014e90:	967ff0ef          	jal	ra,1c0147f6 <pos_task_handle>
1c014e94:	01410783          	lb	a5,20(sp)
1c014e98:	dfe5                	beqz	a5,1c014e90 <pi_cluster_send_task_to_cl+0x62>
1c014e9a:	30041073          	csrw	mstatus,s0
1c014e9e:	4501                	li	a0,0
1c014ea0:	40b6                	lw	ra,76(sp)
1c014ea2:	4426                	lw	s0,72(sp)
1c014ea4:	4496                	lw	s1,68(sp)
1c014ea6:	4906                	lw	s2,64(sp)
1c014ea8:	6161                	addi	sp,sp,80
1c014eaa:	8082                	ret
1c014eac:	00892023          	sw	s0,0(s2)
1c014eb0:	b7f1                	j	1c014e7c <pi_cluster_send_task_to_cl+0x4e>
1c014eb2:	cf99                	beqz	a5,1c014ed0 <pi_cluster_send_task_to_cl+0xa2>
1c014eb4:	4858                	lw	a4,20(s0)
1c014eb6:	4785                	li	a5,1
1c014eb8:	00e797b3          	sll	a5,a5,a4
1c014ebc:	17fd                	addi	a5,a5,-1
1c014ebe:	d05c                	sw	a5,36(s0)
1c014ec0:	b745                	j	1c014e60 <pi_cluster_send_task_to_cl+0x32>
1c014ec2:	01e4c783          	lbu	a5,30(s1)
1c014ec6:	ffc9                	bnez	a5,1c014e60 <pi_cluster_send_task_to_cl+0x32>
1c014ec8:	35d5                	jal	1c014dac <pos_cluster_task_set_stack>
1c014eca:	d959                	beqz	a0,1c014e60 <pi_cluster_send_task_to_cl+0x32>
1c014ecc:	557d                	li	a0,-1
1c014ece:	bfc9                	j	1c014ea0 <pi_cluster_send_task_to_cl+0x72>
1c014ed0:	3df1                	jal	1c014dac <pos_cluster_task_set_stack>
1c014ed2:	d16d                	beqz	a0,1c014eb4 <pi_cluster_send_task_to_cl+0x86>
1c014ed4:	bfe5                	j	1c014ecc <pi_cluster_send_task_to_cl+0x9e>

1c014ed6 <pos_init_entry>:
1c014ed6:	7a101073          	csrw	pcmr,zero
1c014eda:	f1402573          	csrr	a0,mhartid
1c014ede:	01f57593          	andi	a1,a0,31
1c014ee2:	8115                	srli	a0,a0,0x5
1c014ee4:	467d                	li	a2,31
1c014ee6:	00c50463          	beq	a0,a2,1c014eee <label0>
1c014eea:	24a0006f          	j	1c015134 <__l2_priv1_end>

1c014eee <label0>:
1c014eee:	fffed297          	auipc	t0,0xfffed
1c014ef2:	8b228293          	addi	t0,t0,-1870 # 1c0017a0 <_edata>
1c014ef6:	fffed317          	auipc	t1,0xfffed
1c014efa:	8fa30313          	addi	t1,t1,-1798 # 1c0017f0 <__l2_priv0_end>
1c014efe:	0002a023          	sw	zero,0(t0)
1c014f02:	0291                	addi	t0,t0,4
1c014f04:	fe62ede3          	bltu	t0,t1,1c014efe <label0+0x10>
1c014f08:	fffed117          	auipc	sp,0xfffed
1c014f0c:	85810113          	addi	sp,sp,-1960 # 1c001760 <stack>
1c014f10:	00000297          	auipc	t0,0x0
1c014f14:	86028293          	addi	t0,t0,-1952 # 1c014770 <pos_init_start>
1c014f18:	000280e7          	jalr	t0
1c014f1c:	00000513          	li	a0,0
1c014f20:	00000593          	li	a1,0
1c014f24:	ffff4397          	auipc	t2,0xffff4
1c014f28:	9d638393          	addi	t2,t2,-1578 # 1c0088fa <main>
1c014f2c:	000380e7          	jalr	t2
1c014f30:	842a                	mv	s0,a0
1c014f32:	8522                	mv	a0,s0
1c014f34:	ffffe297          	auipc	t0,0xffffe
1c014f38:	cae28293          	addi	t0,t0,-850 # 1c012be2 <exit>
1c014f3c:	000280e7          	jalr	t0

1c014f40 <pos_irq_call_external_c_function>:
1c014f40:	7119                	addi	sp,sp,-128
1c014f42:	c006                	sw	ra,0(sp)
1c014f44:	c20e                	sw	gp,4(sp)
1c014f46:	c412                	sw	tp,8(sp)
1c014f48:	c616                	sw	t0,12(sp)
1c014f4a:	c81a                	sw	t1,16(sp)
1c014f4c:	ca1e                	sw	t2,20(sp)
1c014f4e:	d236                	sw	a3,36(sp)
1c014f50:	d43a                	sw	a4,40(sp)
1c014f52:	d63e                	sw	a5,44(sp)
1c014f54:	d842                	sw	a6,48(sp)
1c014f56:	da46                	sw	a7,52(sp)
1c014f58:	dc72                	sw	t3,56(sp)
1c014f5a:	de76                	sw	t4,60(sp)
1c014f5c:	c0fa                	sw	t5,64(sp)
1c014f5e:	c6fe                	sw	t6,76(sp)
1c014f60:	000600e7          	jalr	a2
1c014f64:	4082                	lw	ra,0(sp)
1c014f66:	4192                	lw	gp,4(sp)
1c014f68:	4222                	lw	tp,8(sp)
1c014f6a:	42b2                	lw	t0,12(sp)
1c014f6c:	4342                	lw	t1,16(sp)
1c014f6e:	43d2                	lw	t2,20(sp)
1c014f70:	5692                	lw	a3,36(sp)
1c014f72:	5722                	lw	a4,40(sp)
1c014f74:	57b2                	lw	a5,44(sp)
1c014f76:	5842                	lw	a6,48(sp)
1c014f78:	58d2                	lw	a7,52(sp)
1c014f7a:	5e62                	lw	t3,56(sp)
1c014f7c:	5ef2                	lw	t4,60(sp)
1c014f7e:	4f06                	lw	t5,64(sp)
1c014f80:	4fb6                	lw	t6,76(sp)
1c014f82:	6109                	addi	sp,sp,128
1c014f84:	8482                	jr	s1

1c014f86 <pos_irq_call_external_c_function_full>:
1c014f86:	7119                	addi	sp,sp,-128
1c014f88:	c006                	sw	ra,0(sp)
1c014f8a:	c20e                	sw	gp,4(sp)
1c014f8c:	c412                	sw	tp,8(sp)
1c014f8e:	c616                	sw	t0,12(sp)
1c014f90:	c81a                	sw	t1,16(sp)
1c014f92:	ca1e                	sw	t2,20(sp)
1c014f94:	cc2a                	sw	a0,24(sp)
1c014f96:	ce2e                	sw	a1,28(sp)
1c014f98:	d236                	sw	a3,36(sp)
1c014f9a:	d43a                	sw	a4,40(sp)
1c014f9c:	d63e                	sw	a5,44(sp)
1c014f9e:	d842                	sw	a6,48(sp)
1c014fa0:	da46                	sw	a7,52(sp)
1c014fa2:	dc72                	sw	t3,56(sp)
1c014fa4:	de76                	sw	t4,60(sp)
1c014fa6:	c0fa                	sw	t5,64(sp)
1c014fa8:	c6fe                	sw	t6,76(sp)
1c014faa:	000600e7          	jalr	a2
1c014fae:	4082                	lw	ra,0(sp)
1c014fb0:	4192                	lw	gp,4(sp)
1c014fb2:	4222                	lw	tp,8(sp)
1c014fb4:	42b2                	lw	t0,12(sp)
1c014fb6:	4342                	lw	t1,16(sp)
1c014fb8:	43d2                	lw	t2,20(sp)
1c014fba:	4562                	lw	a0,24(sp)
1c014fbc:	45f2                	lw	a1,28(sp)
1c014fbe:	5692                	lw	a3,36(sp)
1c014fc0:	5722                	lw	a4,40(sp)
1c014fc2:	57b2                	lw	a5,44(sp)
1c014fc4:	5842                	lw	a6,48(sp)
1c014fc6:	58d2                	lw	a7,52(sp)
1c014fc8:	5e62                	lw	t3,56(sp)
1c014fca:	5ef2                	lw	t4,60(sp)
1c014fcc:	4f06                	lw	t5,64(sp)
1c014fce:	4fb6                	lw	t6,76(sp)
1c014fd0:	6109                	addi	sp,sp,128
1c014fd2:	8482                	jr	s1

1c014fd4 <pos_task_push_asm>:
1c014fd4:	e3feb297          	auipc	t0,0xe3feb
1c014fd8:	04c28293          	addi	t0,t0,76 # 20 <pos_sched_first>
1c014fdc:	0002a603          	lw	a2,0(t0)
1c014fe0:	0005a023          	sw	zero,0(a1)
1c014fe4:	ca09                	beqz	a2,1c014ff6 <__rt_no_first>
1c014fe6:	e3feb297          	auipc	t0,0xe3feb
1c014fea:	03e28293          	addi	t0,t0,62 # 24 <pos_sched_last>
1c014fee:	0002a603          	lw	a2,0(t0)
1c014ff2:	c20c                	sw	a1,0(a2)
1c014ff4:	a029                	j	1c014ffe <__rt_common>

1c014ff6 <__rt_no_first>:
1c014ff6:	e3feb297          	auipc	t0,0xe3feb
1c014ffa:	02b2a523          	sw	a1,42(t0) # 20 <pos_sched_first>

1c014ffe <__rt_common>:
1c014ffe:	e3feb297          	auipc	t0,0xe3feb
1c015002:	02b2a323          	sw	a1,38(t0) # 24 <pos_sched_last>

1c015006 <enqueue_end>:
1c015006:	8482                	jr	s1

1c015008 <pos_task_remote_enqueue>:
1c015008:	fe812e23          	sw	s0,-4(sp)
1c01500c:	fe912c23          	sw	s1,-8(sp)
1c015010:	fea12a23          	sw	a0,-12(sp)
1c015014:	feb12823          	sw	a1,-16(sp)
1c015018:	fec12623          	sw	a2,-20(sp)
1c01501c:	fe512423          	sw	t0,-24(sp)
1c015020:	4405                	li	s0,1
1c015022:	fffec497          	auipc	s1,0xfffec
1c015026:	7ae48493          	addi	s1,s1,1966 # 1c0017d0 <pos_cluster>

1c01502a <pos_task_remote_enqueue_event_loop_cluster>:
1c01502a:	48cc                	lw	a1,20(s1)
1c01502c:	40d0                	lw	a2,4(s1)
1c01502e:	02058a63          	beqz	a1,1c015062 <pos_task_remote_enqueue_event_loop_cluster_continue>
1c015032:	4208                	lw	a0,0(a2)
1c015034:	e119                	bnez	a0,1c01503a <pos_task_remote_enqueue_no_queue_update>
1c015036:	0004a023          	sw	zero,0(s1)

1c01503a <pos_task_remote_enqueue_no_queue_update>:
1c01503a:	4890                	lw	a2,16(s1)
1c01503c:	0004aa23          	sw	zero,20(s1)
1c015040:	00062023          	sw	zero,0(a2)
1c015044:	00000497          	auipc	s1,0x0
1c015048:	01e48493          	addi	s1,s1,30 # 1c015062 <pos_task_remote_enqueue_event_loop_cluster_continue>
1c01504c:	0035f513          	andi	a0,a1,3
1c015050:	00051363          	bnez	a0,1c015056 <pos_task_callback>
1c015054:	b741                	j	1c014fd4 <pos_task_push_asm>

1c015056 <pos_task_callback>:
1c015056:	5571                	li	a0,-4
1c015058:	8de9                	and	a1,a1,a0
1c01505a:	41d0                	lw	a2,4(a1)
1c01505c:	4588                	lw	a0,8(a1)
1c01505e:	ee3ff06f          	j	1c014f40 <pos_irq_call_external_c_function>

1c015062 <pos_task_remote_enqueue_event_loop_cluster_continue>:
1c015062:	147d                	addi	s0,s0,-1
1c015064:	02804063          	bgtz	s0,1c015084 <pos_task_remote_enqueue_event_loop_next_cluster>
1c015068:	ffc12403          	lw	s0,-4(sp)
1c01506c:	ff812483          	lw	s1,-8(sp)
1c015070:	ff412503          	lw	a0,-12(sp)
1c015074:	ff012583          	lw	a1,-16(sp)
1c015078:	fec12603          	lw	a2,-20(sp)
1c01507c:	fe812283          	lw	t0,-24(sp)
1c015080:	30200073          	mret

1c015084 <pos_task_remote_enqueue_event_loop_next_cluster>:
1c015084:	fffec497          	auipc	s1,0xfffec
1c015088:	74c48493          	addi	s1,s1,1868 # 1c0017d0 <pos_cluster>
1c01508c:	02c00593          	li	a1,44
1c015090:	02b405b3          	mul	a1,s0,a1
1c015094:	94ae                	add	s1,s1,a1
1c015096:	bf51                	j	1c01502a <pos_task_remote_enqueue_event_loop_cluster>

1c015098 <pos_time_timer_handler_asm>:
1c015098:	1161                	addi	sp,sp,-8
1c01509a:	c032                	sw	a2,0(sp)
1c01509c:	c226                	sw	s1,4(sp)
1c01509e:	00000617          	auipc	a2,0x0
1c0150a2:	a3460613          	addi	a2,a2,-1484 # 1c014ad2 <pos_time_timer_handler>
1c0150a6:	00000497          	auipc	s1,0x0
1c0150aa:	00c48493          	addi	s1,s1,12 # 1c0150b2 <pos_time_timer_handler_asm_ret>
1c0150ae:	ed9ff06f          	j	1c014f86 <pos_irq_call_external_c_function_full>

1c0150b2 <pos_time_timer_handler_asm_ret>:
1c0150b2:	4492                	lw	s1,4(sp)
1c0150b4:	4602                	lw	a2,0(sp)
1c0150b6:	0121                	addi	sp,sp,8
1c0150b8:	30200073          	mret

1c0150bc <pos_soc_event_handler_asm>:
1c0150bc:	7119                	addi	sp,sp,-128
1c0150be:	c022                	sw	s0,0(sp)
1c0150c0:	c226                	sw	s1,4(sp)
1c0150c2:	c42a                	sw	a0,8(sp)
1c0150c4:	c62e                	sw	a1,12(sp)
1c0150c6:	c832                	sw	a2,16(sp)
1c0150c8:	ca16                	sw	t0,20(sp)
1c0150ca:	1a10a437          	lui	s0,0x1a10a
1c0150ce:	80040413          	addi	s0,s0,-2048 # 1a109800 <__l1_end+0xa108034>
1c0150d2:	5048                	lw	a0,36(s0)
1c0150d4:	00251593          	slli	a1,a0,0x2
1c0150d8:	e3feb297          	auipc	t0,0xe3feb
1c0150dc:	37028293          	addi	t0,t0,880 # 448 <pos_soc_event_callback>
1c0150e0:	2055f603          	p.lw	a2,t0(a1)
1c0150e4:	e3feb297          	auipc	t0,0xe3feb
1c0150e8:	f6428293          	addi	t0,t0,-156 # 48 <pos_soc_event_callback_arg>
1c0150ec:	2055f583          	p.lw	a1,t0(a1)
1c0150f0:	00000497          	auipc	s1,0x0
1c0150f4:	03048493          	addi	s1,s1,48 # 1c015120 <pos_soc_event_handler_end_asm>
1c0150f8:	e49ff06f          	j	1c014f40 <pos_irq_call_external_c_function>

1c0150fc <pos_soc_event_no_udma_asm>:
1c0150fc:	00555593          	srli	a1,a0,0x5
1c015100:	058a                	slli	a1,a1,0x2
1c015102:	e3feb297          	auipc	t0,0xe3feb
1c015106:	f2628293          	addi	t0,t0,-218 # 28 <pos_soc_event_status>
1c01510a:	2055f603          	p.lw	a2,t0(a1)
1c01510e:	897d                	andi	a0,a0,31
1c015110:	80a64633          	p.bsetr	a2,a2,a0
1c015114:	e3feb297          	auipc	t0,0xe3feb
1c015118:	f1428293          	addi	t0,t0,-236 # 28 <pos_soc_event_status>
1c01511c:	00c5e2a3          	p.sw	a2,t0(a1)

1c015120 <pos_soc_event_handler_end_asm>:
1c015120:	4402                	lw	s0,0(sp)
1c015122:	4492                	lw	s1,4(sp)
1c015124:	4522                	lw	a0,8(sp)
1c015126:	45b2                	lw	a1,12(sp)
1c015128:	4642                	lw	a2,16(sp)
1c01512a:	42d2                	lw	t0,20(sp)
1c01512c:	6109                	addi	sp,sp,128
1c01512e:	30200073          	mret

1c015132 <_endtext>:
	...

Disassembly of section .l2_data:

1c015134 <__cluster_text_start>:
1c015134:	f1402573          	csrr	a0,mhartid
1c015138:	01f57593          	andi	a1,a0,31
1c01513c:	8115                	srli	a0,a0,0x5
1c01513e:	000702b7          	lui	t0,0x70
1c015142:	00204337          	lui	t1,0x204
1c015146:	00532023          	sw	t0,0(t1) # 204000 <__l1_heap_size+0x1f57cc>
1c01514a:	4381                	li	t2,0
1c01514c:	0e759463          	bne	a1,t2,1c015234 <pos_slave_start>
1c015150:	20000293          	li	t0,512
1c015154:	00204337          	lui	t1,0x204
1c015158:	00532a23          	sw	t0,20(t1) # 204014 <__l1_heap_size+0x1f57e0>
1c01515c:	e3feb417          	auipc	s0,0xe3feb
1c015160:	ea840413          	addi	s0,s0,-344 # 4 <pos_fll_freq>
1c015164:	002049b7          	lui	s3,0x204
1c015168:	4a09                	li	s4,2
1c01516a:	00000a97          	auipc	s5,0x0
1c01516e:	038a8a93          	addi	s5,s5,56 # 1c0151a2 <pos_master_event>
1c015172:	fffecb97          	auipc	s7,0xfffec
1c015176:	65eb8b93          	addi	s7,s7,1630 # 1c0017d0 <pos_cluster>
1c01517a:	02c00393          	li	t2,44
1c01517e:	02a383b3          	mul	t2,t2,a0
1c015182:	9b9e                	add	s7,s7,t2
1c015184:	0bd1                	addi	s7,s7,20
1c015186:	1a10acb7          	lui	s9,0x1a10a
1c01518a:	810c8c93          	addi	s9,s9,-2032 # 1a109810 <__l1_end+0xa108044>
1c01518e:	4c09                	li	s8,2
1c015190:	00000d17          	auipc	s10,0x0
1c015194:	0e2d0d13          	addi	s10,s10,226 # 1c015272 <pos_set_slave_stack>
1c015198:	001d6d13          	ori	s10,s10,1
1c01519c:	30045073          	csrwi	mstatus,8
1c0151a0:	a829                	j	1c0151ba <pos_master_loop>

1c0151a2 <pos_master_event>:
1c0151a2:	018b2b03          	lw	s6,24(s6)
1c0151a6:	000b0a63          	beqz	s6,1c0151ba <pos_master_loop>

1c0151aa <pos_push_event_to_fc_retry>:
1c0151aa:	000ba283          	lw	t0,0(s7)
1c0151ae:	06029c63          	bnez	t0,1c015226 <pos_push_event_to_fc_wait>
1c0151b2:	016ba023          	sw	s6,0(s7)
1c0151b6:	018ca023          	sw	s8,0(s9)

1c0151ba <pos_master_loop>:
1c0151ba:	00042b03          	lw	s6,0(s0)
1c0151be:	040b0d63          	beqz	s6,1c015218 <pos_master_sleep>

1c0151c2 <pos_master_loop_update_next>:
1c0151c2:	020b2e83          	lw	t4,32(s6)
1c0151c6:	01d42023          	sw	t4,0(s0)
1c0151ca:	020b2f03          	lw	t5,32(s6)
1c0151ce:	ffee9ae3          	bne	t4,t5,1c0151c2 <pos_master_loop_update_next>

1c0151d2 <pos_master_loop_exec_task>:
1c0151d2:	004b2503          	lw	a0,4(s6)
1c0151d6:	000b2283          	lw	t0,0(s6)
1c0151da:	008b2103          	lw	sp,8(s6)
1c0151de:	00cb2303          	lw	t1,12(s6)
1c0151e2:	010b2383          	lw	t2,16(s6)
1c0151e6:	024b2f03          	lw	t5,36(s6)
1c0151ea:	014b2f83          	lw	t6,20(s6)
1c0151ee:	80d6                	mv	ra,s5
1c0151f0:	911a                	add	sp,sp,t1
1c0151f2:	e3febe17          	auipc	t3,0xe3feb
1c0151f6:	e3fe2323          	sw	t6,-474(t3) # 18 <pos_cluster_nb_active_pe>

1c0151fa <pos_no_stack_check>:
1c0151fa:	09e9a223          	sw	t5,132(s3) # 204084 <__l1_heap_size+0x1f5850>
1c0151fe:	21e9a023          	sw	t5,512(s3)
1c015202:	21e9a623          	sw	t5,524(s3)

1c015206 <pos_master_no_slave_barrier>:
1c015206:	000f2863          	p.beqimm	t5,0,1c015216 <pos_master_loop_no_slave>
1c01520a:	09a9a023          	sw	s10,128(s3)
1c01520e:	0879a023          	sw	t2,128(s3)
1c015212:	0829a023          	sw	sp,128(s3)

1c015216 <pos_master_loop_no_slave>:
1c015216:	8282                	jr	t0

1c015218 <pos_master_sleep>:
1c015218:	0149a423          	sw	s4,8(s3)
1c01521c:	03c9e003          	p.elw	zero,60(s3)
1c015220:	0149a223          	sw	s4,4(s3)
1c015224:	bf59                	j	1c0151ba <pos_master_loop>

1c015226 <pos_push_event_to_fc_wait>:
1c015226:	0149a423          	sw	s4,8(s3)
1c01522a:	03c9e003          	p.elw	zero,60(s3)
1c01522e:	0149a223          	sw	s4,4(s3)
1c015232:	bfa5                	j	1c0151aa <pos_push_event_to_fc_retry>

1c015234 <pos_slave_start>:
1c015234:	00204937          	lui	s2,0x204
1c015238:	f14029f3          	csrr	s3,mhartid
1c01523c:	01f9f993          	andi	s3,s3,31
1c015240:	00000a17          	auipc	s4,0x0
1c015244:	012a0a13          	addi	s4,s4,18 # 1c015252 <pos_fork_return>
1c015248:	00000a97          	auipc	s5,0x0
1c01524c:	00ea8a93          	addi	s5,s5,14 # 1c015256 <pos_wait_for_dispatch>
1c015250:	a019                	j	1c015256 <pos_wait_for_dispatch>

1c015252 <pos_fork_return>:
1c015252:	21c96283          	p.elw	t0,540(s2) # 20421c <__l1_heap_size+0x1f59e8>

1c015256 <pos_wait_for_dispatch>:
1c015256:	08096283          	p.elw	t0,128(s2)
1c01525a:	08096503          	p.elw	a0,128(s2)
1c01525e:	0012f313          	andi	t1,t0,1
1c015262:	00031563          	bnez	t1,1c01526c <pos_other_entry>

1c015266 <pos_fork_entry>:
1c015266:	000a00b3          	add	ra,s4,zero
1c01526a:	8282                	jr	t0

1c01526c <pos_other_entry>:
1c01526c:	000a80b3          	add	ra,s5,zero
1c015270:	8282                	jr	t0

1c015272 <pos_set_slave_stack>:
1c015272:	08096283          	p.elw	t0,128(s2)
1c015276:	00098f13          	mv	t5,s3
1c01527a:	02af0eb3          	mul	t4,t5,a0
1c01527e:	005e8133          	add	sp,t4,t0
1c015282:	8082                	ret

1c015284 <L1_memocc_bytes>:
1c015284:	0000 0000                                   ....

1c015288 <L2_memocc_bytes>:
1c015288:	0000 0000                                   ....

1c01528c <INPUT>:
1c01528c:	126f 3a83 c2e4 3a81 a883 3a7b 8d12 3a6e     o..:...:..{:..n:
1c01529c:	3373 3a5c 61fa 3a84 126f 3a83 479a 3a7e     s3\:.a.:o..:.G~:
1c0152ac:	2c28 3a71 d28a 3a5e 509c 3a88 0111 3a87     (,q:..^:.P.:...:
1c0152bc:	126f 3a83 096d 3a79 afce 3a66 de55 3a8e     o..:m.y:..f:U..:
1c0152cc:	8eca 3a8d a028 3a89 126f 3a83 cb3f 3a73     ...:(..:o..:?.s:
1c0152dc:	0b24 3a98 bb99 3a96 ccf7 3a92 3f3e 3a8c     $..:...:...:>?.:
1c0152ec:	126f 3a83 61fa 3a84 126f 3a83 479a 3a7e     o..:.a.:o..:.G~:
1c0152fc:	2c28 3a71 d28a 3a5e 509c 3a88 0111 3a87     (,q:..^:.P.:...:
1c01530c:	126f 3a83 096d 3a79 afce 3a66 de55 3a8e     o..:m.y:..f:U..:
1c01531c:	8eca 3a8d a028 3a89 126f 3a83 cb3f 3a73     ...:(..:o..:?.s:
1c01532c:	0b24 3a98 bb99 3a96 ccf7 3a92 3f3e 3a8c     $..:...:...:>?.:
1c01533c:	126f 3a83 d70b 3aa3 877f 3aa2 98dd 3a9e     o..:...:...:...:
1c01534c:	0b24 3a98 de55 3a8e 509c 3a88 0111 3a87     $..:U..:.P.:...:
1c01535c:	126f 3a83 096d 3a79 afce 3a66 de55 3a8e     o..:m.y:..f:U..:
1c01536c:	8eca 3a8d a028 3a89 126f 3a83 cb3f 3a73     ...:(..:o..:?.s:
1c01537c:	0b24 3a98 bb99 3a96 ccf7 3a92 3f3e 3a8c     $..:...:...:>?.:
1c01538c:	126f 3a83 d70b 3aa3 877f 3aa2 98dd 3a9e     o..:...:...:...:
1c01539c:	0b24 3a98 de55 3a8e 4208 3ab2 f27c 3ab0     $..:U..:.B.:|..:
1c0153ac:	03da 3aad 7621 3aa6 4952 3a9d de55 3a8e     ...:!v.:RI.:U..:
1c0153bc:	8eca 3a8d a028 3a89 126f 3a83 cb3f 3a73     ...:(..:o..:?.s:
1c0153cc:	0b24 3a98 bb99 3a96 ccf7 3a92 3f3e 3a8c     $..:...:...:>?.:
1c0153dc:	126f 3a83 d70b 3aa3 877f 3aa2 98dd 3a9e     o..:...:...:...:
1c0153ec:	0b24 3a98 de55 3a8e 4208 3ab2 f27c 3ab0     $..:U..:.B.:|..:
1c0153fc:	03da 3aad 7621 3aa6 4952 3a9d 4c1b 3ac3     ...:!v.:RI.:.L.:
1c01540c:	fc90 3ac1 0dee 3abe 8035 3ab7 5366 3aae     ...:...:5..:fS.:
1c01541c:	0b24 3a98 bb99 3a96 ccf7 3a92 3f3e 3a8c     $..:...:...:>?.:
1c01542c:	126f 3a83 d70b 3aa3 877f 3aa2 98dd 3a9e     o..:...:...:...:
1c01543c:	0b24 3a98 de55 3a8e 4208 3ab2 f27c 3ab0     $..:U..:.B.:|..:
1c01544c:	03da 3aad 7621 3aa6 4952 3a9d 4c1b 3ac3     ...:!v.:RI.:.L.:
1c01545c:	fc90 3ac1 0dee 3abe 8035 3ab7 5366 3aae     ...:...:5..:fS.:
1c01546c:	f545 3ad6 a5ba 3ad5 b718 3ad1 295f 3acb     E..:...:...:_).:
1c01547c:	fc90 3ac1 d70b 3aa3 877f 3aa2 98dd 3a9e     ...:...:...:...:
1c01548c:	0b24 3a98 de55 3a8e 4208 3ab2 f27c 3ab0     $..:U..:.B.:|..:
1c01549c:	03da 3aad 7621 3aa6 4952 3a9d 4c1b 3ac3     ...:!v.:RI.:.L.:
1c0154ac:	fc90 3ac1 0dee 3abe 8035 3ab7 5366 3aae     ...:...:5..:fS.:
1c0154bc:	f545 3ad6 a5ba 3ad5 b718 3ad1 295f 3acb     E..:...:...:_).:
1c0154cc:	fc90 3ac1 3d86 3aed edfa 3aeb ff58 3ae7     ...:.=.:...:X..:
1c0154dc:	71a0 3ae1 44d0 3ad8 4208 3ab2 f27c 3ab0     .q.:.D.:.B.:|..:
1c0154ec:	03da 3aad 7621 3aa6 4952 3a9d 4c1b 3ac3     ...:!v.:RI.:.L.:
1c0154fc:	fc90 3ac1 0dee 3abe 8035 3ab7 5366 3aae     ...:...:5..:fS.:
1c01550c:	f545 3ad6 a5ba 3ad5 b718 3ad1 295f 3acb     E..:...:...:_).:
1c01551c:	fc90 3ac1 3d86 3aed edfa 3aeb ff58 3ae7     ...:.=.:...:X..:
1c01552c:	71a0 3ae1 44d0 3ad8 126f 3b03 6aa9 3b02     .q.:.D.:o..;.j.;
1c01553c:	7358 3b00 58f8 3afa 2c28 3af1 4c1b 3ac3     Xs.;.X.:(,.:.L.:
1c01554c:	fc90 3ac1 0dee 3abe 8035 3ab7 5366 3aae     ...:...:5..:fS.:
1c01555c:	f545 3ad6 a5ba 3ad5 b718 3ad1 295f 3acb     E..:...:...:_).:
1c01556c:	fc90 3ac1 3d86 3aed edfa 3aeb ff58 3ae7     ...:.=.:...:X..:
1c01557c:	71a0 3ae1 44d0 3ad8 126f 3b03 6aa9 3b02     .q.:.D.:o..;.j.;
1c01558c:	7358 3b00 58f8 3afa 2c28 3af1 d5a6 3b10     Xs.;.X.:(,.:...;
1c01559c:	2de0 3b10 368f 3b0e efb3 3b0a 594b 3b06     .-.;.6.;...;KY.;

1c0155ac <OUTPUT>:
1c0155ac:	407a 3d59 fe19 3d53 7d41 3d4b f639 3d6c     z@Y=..S=A}K=9.l=
1c0155bc:	b3d9 3d67 3303 3d5f f53b 3d81 a814 3d7e     ..g=.3_=;..=..~=
1c0155cc:	273b 3d76 f7e7 3d7d d30e 3d77 d8b5 3d6d     ;'v=..}=..w=..m=
1c0155dc:	5051 3d8a 3de2 3d87 40b6 3d82 8f69 3d97     QP.=.=.=.@.=i..=
1c0155ec:	7cfa 3d94 7fd2 3d8f 57ad 3d91 d3fe 3d8d     .|.=...=.W.=...=
1c0155fc:	1a15 3d88 2584 3d9e a1d9 3d9a e7ee 3d94     ...=.%.=...=...=
1c01560c:	299a 3dad a5eb 3da9 ec02 3da3 b364 3da3     .).=...=...=d..=
1c01561c:	be76 3d9f 47d0 3d99 faba 3db1 05cc 3dae     v..=.G.=...=...=
1c01562c:	8f26 3da7 c3c9 3dc2 cede 3dbe 5836 3db8     &..=...=...=6X.=
1c01563c:	0f1c 3db6 a8f0 3db1 7589 3daa cff0 3dc5     ...=...=.u.=...=
1c01564c:	69c2 3dc1 365c 3dba 5dfc 3dd8 f7cd 3dd3     .i.=\6.=.].=...=
1c01565c:	c468 3dcc 6ad3 3dc8 9369 3dc3 a345 3dbb     h..=.j.=i..=E..=
1c01566c:	a523 3dd9 cdb9 3dd4 dd93 3dcc f82e 3ded     #..=...=...=...=
1c01567c:	20c1 3de9 309b 3de1 c68b 3dda 7de4 3dd5     . .=.0.=...=.}.=
1c01568c:	d100 3dcc 7a58 3ded 31ae 3de8 84cb 3ddf     ...=Xz.=.1.=...=
1c01569c:	c92e 3e01 49b5 3dfe 9cd1 3df5 2243 3ded     ...>.I.=...=C".=
1c0156ac:	685a 3de7 feba 3ddd a7c6 3e00 95a5 3dfb     Zh.=...=...>...=
1c0156bc:	2c04 3df2 9647 3e0c b952 3e09 0482 3e05     .,.=G..>R..>...>

1c0156cc <WEIGHTS>:
1c0156cc:	0000 0000 cccd 3dcc cccd 3e4c cccd 3dcc     .......=..L>...=
1c0156dc:	cccd 3e4c 999a 3e99 cccd 3e4c 999a 3e99     ..L>...>..L>...>
1c0156ec:	cccd 3ecc cccd 3dcc cccd 3e4c 999a 3e99     ...>...=..L>...>
1c0156fc:	cccd 3e4c 999a 3e99 cccd 3ecc 999a 3e99     ..L>...>...>...>
1c01570c:	cccd 3ecc 0000 3f00 cccd 3e4c 999a 3e99     ...>...?..L>...>
1c01571c:	cccd 3ecc 999a 3e99 cccd 3ecc 0000 3f00     ...>...>...>...?
1c01572c:	cccd 3ecc 0000 3f00 999a 3f19 999a 3e99     ...>...?...?...>
1c01573c:	cccd 3ecc 0000 3f00 cccd 3ecc 0000 3f00     ...>...?...>...?
1c01574c:	999a 3f19 0000 3f00 999a 3f19 3333 3f33     ...?...?...?333?
1c01575c:	cccd 3ecc 0000 3f00 999a 3f19 0000 3f00     ...>...?...?...?
1c01576c:	999a 3f19 3333 3f33 999a 3f19 3333 3f33     ...?333?...?333?
1c01577c:	cccd 3f4c 0000 3f00 999a 3f19 3333 3f33     ..L?...?...?333?
1c01578c:	999a 3f19 3333 3f33 cccd 3f4c 3333 3f33     ...?333?..L?333?
1c01579c:	cccd 3f4c 6666 3f66 999a 3f19 3333 3f33     ..L?fff?...?333?
1c0157ac:	cccd 3f4c 3333 3f33 cccd 3f4c 6666 3f66     ..L?333?..L?fff?
1c0157bc:	cccd 3f4c 6666 3f66 0000 3f80 3333 3f33     ..L?fff?...?333?
1c0157cc:	cccd 3f4c 6666 3f66 cccd 3f4c 6666 3f66     ..L?fff?..L?fff?
1c0157dc:	0000 3f80 6666 3f66 0000 3f80 cccd 3f8c     ...?fff?...?...?
1c0157ec:	cccd 3dcc cccd 3e4c 999a 3e99 cccd 3e4c     ...=..L>...>..L>
1c0157fc:	999a 3e99 cccd 3ecc 999a 3e99 cccd 3ecc     ...>...>...>...>
1c01580c:	0000 3f00 cccd 3e4c 999a 3e99 cccd 3ecc     ...?..L>...>...>
1c01581c:	999a 3e99 cccd 3ecc 0000 3f00 cccd 3ecc     ...>...>...?...>
1c01582c:	0000 3f00 999a 3f19 999a 3e99 cccd 3ecc     ...?...?...>...>
1c01583c:	0000 3f00 cccd 3ecc 0000 3f00 999a 3f19     ...?...>...?...?
1c01584c:	0000 3f00 999a 3f19 3333 3f33 cccd 3ecc     ...?...?333?...>
1c01585c:	0000 3f00 999a 3f19 0000 3f00 999a 3f19     ...?...?...?...?
1c01586c:	3333 3f33 999a 3f19 3333 3f33 cccd 3f4c     333?...?333?..L?
1c01587c:	0000 3f00 999a 3f19 3333 3f33 999a 3f19     ...?...?333?...?
1c01588c:	3333 3f33 cccd 3f4c 3333 3f33 cccd 3f4c     333?..L?333?..L?
1c01589c:	6666 3f66 999a 3f19 3333 3f33 cccd 3f4c     fff?...?333?..L?
1c0158ac:	3333 3f33 cccd 3f4c 6666 3f66 cccd 3f4c     333?..L?fff?..L?
1c0158bc:	6666 3f66 0000 3f80 3333 3f33 cccd 3f4c     fff?...?333?..L?
1c0158cc:	6666 3f66 cccd 3f4c 6666 3f66 0000 3f80     fff?..L?fff?...?
1c0158dc:	6666 3f66 0000 3f80 cccd 3f8c cccd 3f4c     fff?...?...?..L?
1c0158ec:	6666 3f66 0000 3f80 6666 3f66 0000 3f80     fff?...?fff?...?
1c0158fc:	cccd 3f8c 0000 3f80 cccd 3f8c 999a 3f99     ...?...?...?...?
1c01590c:	cccd 3e4c 999a 3e99 cccd 3ecc 999a 3e99     ..L>...>...>...>
1c01591c:	cccd 3ecc 0000 3f00 cccd 3ecc 0000 3f00     ...>...?...>...?
1c01592c:	999a 3f19 999a 3e99 cccd 3ecc 0000 3f00     ...?...>...>...?
1c01593c:	cccd 3ecc 0000 3f00 999a 3f19 0000 3f00     ...>...?...?...?
1c01594c:	999a 3f19 3333 3f33 cccd 3ecc 0000 3f00     ...?333?...>...?
1c01595c:	999a 3f19 0000 3f00 999a 3f19 3333 3f33     ...?...?...?333?
1c01596c:	999a 3f19 3333 3f33 cccd 3f4c 0000 3f00     ...?333?..L?...?
1c01597c:	999a 3f19 3333 3f33 999a 3f19 3333 3f33     ...?333?...?333?
1c01598c:	cccd 3f4c 3333 3f33 cccd 3f4c 6666 3f66     ..L?333?..L?fff?
1c01599c:	999a 3f19 3333 3f33 cccd 3f4c 3333 3f33     ...?333?..L?333?
1c0159ac:	cccd 3f4c 6666 3f66 cccd 3f4c 6666 3f66     ..L?fff?..L?fff?
1c0159bc:	0000 3f80 3333 3f33 cccd 3f4c 6666 3f66     ...?333?..L?fff?
1c0159cc:	cccd 3f4c 6666 3f66 0000 3f80 6666 3f66     ..L?fff?...?fff?
1c0159dc:	0000 3f80 cccd 3f8c cccd 3f4c 6666 3f66     ...?...?..L?fff?
1c0159ec:	0000 3f80 6666 3f66 0000 3f80 cccd 3f8c     ...?fff?...?...?
1c0159fc:	0000 3f80 cccd 3f8c 999a 3f99 6666 3f66     ...?...?...?fff?
1c015a0c:	0000 3f80 cccd 3f8c 0000 3f80 cccd 3f8c     ...?...?...?...?
1c015a1c:	999a 3f99 cccd 3f8c 999a 3f99 6666 3fa6     ...?...?...?ff.?
1c015a2c:	999a 3e99 cccd 3ecc 0000 3f00 cccd 3ecc     ...>...>...?...>
1c015a3c:	0000 3f00 999a 3f19 0000 3f00 999a 3f19     ...?...?...?...?
1c015a4c:	3333 3f33 cccd 3ecc 0000 3f00 999a 3f19     333?...>...?...?
1c015a5c:	0000 3f00 999a 3f19 3333 3f33 999a 3f19     ...?...?333?...?
1c015a6c:	3333 3f33 cccd 3f4c 0000 3f00 999a 3f19     333?..L?...?...?
1c015a7c:	3333 3f33 999a 3f19 3333 3f33 cccd 3f4c     333?...?333?..L?
1c015a8c:	3333 3f33 cccd 3f4c 6666 3f66 999a 3f19     333?..L?fff?...?
1c015a9c:	3333 3f33 cccd 3f4c 3333 3f33 cccd 3f4c     333?..L?333?..L?
1c015aac:	6666 3f66 cccd 3f4c 6666 3f66 0000 3f80     fff?..L?fff?...?
1c015abc:	3333 3f33 cccd 3f4c 6666 3f66 cccd 3f4c     333?..L?fff?..L?
1c015acc:	6666 3f66 0000 3f80 6666 3f66 0000 3f80     fff?...?fff?...?
1c015adc:	cccd 3f8c cccd 3f4c 6666 3f66 0000 3f80     ...?..L?fff?...?
1c015aec:	6666 3f66 0000 3f80 cccd 3f8c 0000 3f80     fff?...?...?...?
1c015afc:	cccd 3f8c 999a 3f99 6666 3f66 0000 3f80     ...?...?fff?...?
1c015b0c:	cccd 3f8c 0000 3f80 cccd 3f8c 999a 3f99     ...?...?...?...?
1c015b1c:	cccd 3f8c 999a 3f99 6666 3fa6 0000 3f80     ...?...?ff.?...?
1c015b2c:	cccd 3f8c 999a 3f99 cccd 3f8c 999a 3f99     ...?...?...?...?
1c015b3c:	6666 3fa6 999a 3f99 6666 3fa6 3333 3fb3     ff.?...?ff.?33.?
1c015b4c:	cccd 3ecc 0000 3f00 999a 3f19 0000 3f00     ...>...?...?...?
1c015b5c:	999a 3f19 3333 3f33 999a 3f19 3333 3f33     ...?333?...?333?
1c015b6c:	cccd 3f4c 0000 3f00 999a 3f19 3333 3f33     ..L?...?...?333?
1c015b7c:	999a 3f19 3333 3f33 cccd 3f4c 3333 3f33     ...?333?..L?333?
1c015b8c:	cccd 3f4c 6666 3f66 999a 3f19 3333 3f33     ..L?fff?...?333?
1c015b9c:	cccd 3f4c 3333 3f33 cccd 3f4c 6666 3f66     ..L?333?..L?fff?
1c015bac:	cccd 3f4c 6666 3f66 0000 3f80 3333 3f33     ..L?fff?...?333?
1c015bbc:	cccd 3f4c 6666 3f66 cccd 3f4c 6666 3f66     ..L?fff?..L?fff?
1c015bcc:	0000 3f80 6666 3f66 0000 3f80 cccd 3f8c     ...?fff?...?...?
1c015bdc:	cccd 3f4c 6666 3f66 0000 3f80 6666 3f66     ..L?fff?...?fff?
1c015bec:	0000 3f80 cccd 3f8c 0000 3f80 cccd 3f8c     ...?...?...?...?
1c015bfc:	999a 3f99 6666 3f66 0000 3f80 cccd 3f8c     ...?fff?...?...?
1c015c0c:	0000 3f80 cccd 3f8c 999a 3f99 cccd 3f8c     ...?...?...?...?
1c015c1c:	999a 3f99 6666 3fa6 0000 3f80 cccd 3f8c     ...?ff.?...?...?
1c015c2c:	999a 3f99 cccd 3f8c 999a 3f99 6666 3fa6     ...?...?...?ff.?
1c015c3c:	999a 3f99 6666 3fa6 3333 3fb3 cccd 3f8c     ...?ff.?33.?...?
1c015c4c:	999a 3f99 6666 3fa6 999a 3f99 6666 3fa6     ...?ff.?...?ff.?
1c015c5c:	3333 3fb3 6666 3fa6 3333 3fb3 0000 3fc0     33.?ff.?33.?...?
1c015c6c:	0000 3f00 999a 3f19 3333 3f33 999a 3f19     ...?...?333?...?
1c015c7c:	3333 3f33 cccd 3f4c 3333 3f33 cccd 3f4c     333?..L?333?..L?
1c015c8c:	6666 3f66 999a 3f19 3333 3f33 cccd 3f4c     fff?...?333?..L?
1c015c9c:	3333 3f33 cccd 3f4c 6666 3f66 cccd 3f4c     333?..L?fff?..L?
1c015cac:	6666 3f66 0000 3f80 3333 3f33 cccd 3f4c     fff?...?333?..L?
1c015cbc:	6666 3f66 cccd 3f4c 6666 3f66 0000 3f80     fff?..L?fff?...?
1c015ccc:	6666 3f66 0000 3f80 cccd 3f8c cccd 3f4c     fff?...?...?..L?
1c015cdc:	6666 3f66 0000 3f80 6666 3f66 0000 3f80     fff?...?fff?...?
1c015cec:	cccd 3f8c 0000 3f80 cccd 3f8c 999a 3f99     ...?...?...?...?
1c015cfc:	6666 3f66 0000 3f80 cccd 3f8c 0000 3f80     fff?...?...?...?
1c015d0c:	cccd 3f8c 999a 3f99 cccd 3f8c 999a 3f99     ...?...?...?...?
1c015d1c:	6666 3fa6 0000 3f80 cccd 3f8c 999a 3f99     ff.?...?...?...?
1c015d2c:	cccd 3f8c 999a 3f99 6666 3fa6 999a 3f99     ...?...?ff.?...?
1c015d3c:	6666 3fa6 3333 3fb3 cccd 3f8c 999a 3f99     ff.?33.?...?...?
1c015d4c:	6666 3fa6 999a 3f99 6666 3fa6 3333 3fb3     ff.?...?ff.?33.?
1c015d5c:	6666 3fa6 3333 3fb3 0000 3fc0 999a 3f99     ff.?33.?...?...?
1c015d6c:	6666 3fa6 3333 3fb3 6666 3fa6 3333 3fb3     ff.?33.?ff.?33.?
1c015d7c:	0000 3fc0 3333 3fb3 0000 3fc0 cccd 3fcc     ...?33.?...?...?
1c015d8c:	999a 3f19 3333 3f33 cccd 3f4c 3333 3f33     ...?333?..L?333?
1c015d9c:	cccd 3f4c 6666 3f66 cccd 3f4c 6666 3f66     ..L?fff?..L?fff?
1c015dac:	0000 3f80 3333 3f33 cccd 3f4c 6666 3f66     ...?333?..L?fff?
1c015dbc:	cccd 3f4c 6666 3f66 0000 3f80 6666 3f66     ..L?fff?...?fff?
1c015dcc:	0000 3f80 cccd 3f8c cccd 3f4c 6666 3f66     ...?...?..L?fff?
1c015ddc:	0000 3f80 6666 3f66 0000 3f80 cccd 3f8c     ...?fff?...?...?
1c015dec:	0000 3f80 cccd 3f8c 999a 3f99 6666 3f66     ...?...?...?fff?
1c015dfc:	0000 3f80 cccd 3f8c 0000 3f80 cccd 3f8c     ...?...?...?...?
1c015e0c:	999a 3f99 cccd 3f8c 999a 3f99 6666 3fa6     ...?...?...?ff.?
1c015e1c:	0000 3f80 cccd 3f8c 999a 3f99 cccd 3f8c     ...?...?...?...?
1c015e2c:	999a 3f99 6666 3fa6 999a 3f99 6666 3fa6     ...?ff.?...?ff.?
1c015e3c:	3333 3fb3 cccd 3f8c 999a 3f99 6666 3fa6     33.?...?...?ff.?
1c015e4c:	999a 3f99 6666 3fa6 3333 3fb3 6666 3fa6     ...?ff.?33.?ff.?
1c015e5c:	3333 3fb3 0000 3fc0 999a 3f99 6666 3fa6     33.?...?...?ff.?
1c015e6c:	3333 3fb3 6666 3fa6 3333 3fb3 0000 3fc0     33.?ff.?33.?...?
1c015e7c:	3333 3fb3 0000 3fc0 cccd 3fcc 6666 3fa6     33.?...?...?ff.?
1c015e8c:	3333 3fb3 0000 3fc0 3333 3fb3 0000 3fc0     33.?...?33.?...?
1c015e9c:	cccd 3fcc 0000 3fc0 cccd 3fcc 999a 3fd9     ...?...?...?...?
1c015eac:	3333 3f33 cccd 3f4c 6666 3f66 cccd 3f4c     333?..L?fff?..L?
1c015ebc:	6666 3f66 0000 3f80 6666 3f66 0000 3f80     fff?...?fff?...?
1c015ecc:	cccd 3f8c cccd 3f4c 6666 3f66 0000 3f80     ...?..L?fff?...?
1c015edc:	6666 3f66 0000 3f80 cccd 3f8c 0000 3f80     fff?...?...?...?
1c015eec:	cccd 3f8c 999a 3f99 6666 3f66 0000 3f80     ...?...?fff?...?
1c015efc:	cccd 3f8c 0000 3f80 cccd 3f8c 999a 3f99     ...?...?...?...?
1c015f0c:	cccd 3f8c 999a 3f99 6666 3fa6 0000 3f80     ...?...?ff.?...?
1c015f1c:	cccd 3f8c 999a 3f99 cccd 3f8c 999a 3f99     ...?...?...?...?
1c015f2c:	6666 3fa6 999a 3f99 6666 3fa6 3333 3fb3     ff.?...?ff.?33.?
1c015f3c:	cccd 3f8c 999a 3f99 6666 3fa6 999a 3f99     ...?...?ff.?...?
1c015f4c:	6666 3fa6 3333 3fb3 6666 3fa6 3333 3fb3     ff.?33.?ff.?33.?
1c015f5c:	0000 3fc0 999a 3f99 6666 3fa6 3333 3fb3     ...?...?ff.?33.?
1c015f6c:	6666 3fa6 3333 3fb3 0000 3fc0 3333 3fb3     ff.?33.?...?33.?
1c015f7c:	0000 3fc0 cccd 3fcc 6666 3fa6 3333 3fb3     ...?...?ff.?33.?
1c015f8c:	0000 3fc0 3333 3fb3 0000 3fc0 cccd 3fcc     ...?33.?...?...?
1c015f9c:	0000 3fc0 cccd 3fcc 999a 3fd9 3333 3fb3     ...?...?...?33.?
1c015fac:	0000 3fc0 cccd 3fcc 0000 3fc0 cccd 3fcc     ...?...?...?...?
1c015fbc:	999a 3fd9 cccd 3fcc 999a 3fd9 6666 3fe6     ...?...?...?ff.?

1c015fcc <OUTPUT_GRAD>:
1c015fcc:	7c6b bcd7 c737 bcd7 4027 bcd8 6419 bcd6     k|..7...'@...d..
1c015fdc:	aee5 bcd6 27d5 bcd7 1da5 bcd5 6871 bcd5     .....'......qh..
1c015fec:	e160 bcd5 723b bcd5 c99c bcd5 5785 bcd6     `...;r.......W..
1c015ffc:	2ff7 bcd4 8759 bcd4 1541 bcd5 b72d bcd2     ./..Y...A...-...
1c01600c:	0e8f bcd3 9c77 bcd3 6809 bcd3 cc00 bcd3     ....w....h......
1c01601c:	6ee1 bcd4 fbd5 bcd1 5fcc bcd2 02ad bcd3     .n......._......
1c01602c:	50b6 bcd0 b4ad bcd0 578e bcd1 5dd9 bcd1     .P.......W...]..
1c01603c:	ce65 bcd1 863e bcd2 c7b3 bccf 383e bcd0     e...>.......>8..
1c01604c:	f018 bcd0 ea3f bccd 5acb bcce 12a5 bccf     ....?....Z......
1c01605c:	53a7 bccf d0c9 bccf 9d9c bcd0 9390 bccd     .S..............
1c01606c:	10b2 bcce dd84 bcce 83c7 bccb 00e9 bccc     ................
1c01607c:	cdbc bccc 4977 bccd d32d bccd b4f8 bcce     ....wI..-.......
1c01608c:	5f6e bccb e925 bccb caf0 bccc 1d50 bcc9     n_..%.......P...
1c01609c:	a707 bcc9 88d3 bcca 3f46 bccb d592 bccb     ........F?......
1c0160ac:	cc55 bccc 2b4c bcc9 c197 bcc9 b85c bcca     U...L+......\...
1c0160bc:	b6d9 bcc6 4d25 bcc7 43e9 bcc8 3515 bcc9     ....%M...C...5..
1c0160cc:	d7f6 bcc9 e3b3 bcca f729 bcc6 9a0a bcc7     ........).......
1c0160dc:	a5c7 bcc8 5062 bcc4 f344 bcc4 ff00 bcc5     ....bP..D.......

1c0160ec <WEIGHT_GRAD>:
1c0160ec:	4a5f b977 dd7c b96f 7e5a b963 b118 b97e     _Jw.|.o.Z~c...~.
1c0160fc:	4436 b977 e512 b96a 8508 b985 ce97 b981     6Dw...j.........
1c01610c:	3e0b b977 b118 b97e 4436 b977 e512 b96a     .>w...~.6Dw...j.
1c01611c:	8508 b985 ce97 b981 3e0b b977 2aa5 b98e     .........>w..*..
1c01612c:	7434 b98a 44a2 b984 8508 b985 ce97 b981     4t...D..........
1c01613c:	3e0b b977 2aa5 b98e 7434 b98a 44a2 b984     .>w..*..4t...D..
1c01614c:	4963 b999 92f2 b995 635f b98f 2aa5 b98e     cI......_c...*..
1c01615c:	7434 b98a 44a2 b984 4963 b999 92f2 b995     4t...D..cI......
1c01616c:	635f b98f e140 b9a6 2ace b9a3 fb3e b99c     _c..@....*..>...
1c01617c:	4963 b999 92f2 b995 635f b98f e140 b9a6     cI......_c..@...
1c01618c:	2ace b9a3 fb3e b99c f23e b9b6 3bcd b9b3     .*..>...>....;..
1c01619c:	0c3c b9ad e140 b9a6 2ace b9a3 fb3e b99c     <...@....*..>...
1c0161ac:	f23e b9b6 3bcd b9b3 0c3c b9ad 7c5c b9c9     >....;..<...\|..
1c0161bc:	c5ea b9c5 965b b9bf f23e b9b6 3bcd b9b3     ....[...>....;..
1c0161cc:	0c3c b9ad 7c5c b9c9 c5ea b9c5 965b b9bf     <...\|......[...
1c0161dc:	7f9c b9de c92a b9da 9998 b9d4 7c5c b9c9     ....*.......\|..
1c0161ec:	c5ea b9c5 965b b9bf 7f9c b9de c92a b9da     ....[.......*...
1c0161fc:	9998 b9d4 fbfa b9f5 4588 b9f2 15f8 b9ec     .........E......
1c01620c:	cdff b974 73ea b96d 344c b961 20f2 b97c     ..t..sm.L4a.. |.
1c01621c:	c6de b974 873f b968 2cb8 b984 7fae b980     ..t.?.h..,......
1c01622c:	bfbb b974 20f2 b97c c6de b974 873f b968     ..t.. |...t.?.h.
1c01623c:	2cb8 b984 7fae b980 bfbb b974 bbbe b98c     .,........t.....
1c01624c:	0eb3 b989 eee3 b982 2cb8 b984 7fae b980     .........,......
1c01625c:	bfbb b974 bbbe b98c 0eb3 b989 eee3 b982     ..t.............
1c01626c:	bd88 b997 107d b994 f0ae b98d bbbe b98c     ....}...........
1c01627c:	0eb3 b989 eee3 b982 bd88 b997 107d b994     ............}...
1c01628c:	f0ae b98d 321a b9a5 850e b9a1 653f b99b     .....2......?e..
1c01629c:	bd88 b997 107d b994 f0ae b98d 321a b9a5     ....}........2..
1c0162ac:	850e b9a1 653f b99b 1970 b9b5 6c66 b9b1     ....?e..p...fl..
1c0162bc:	4c96 b9ab 321a b9a5 850e b9a1 653f b99b     .L...2......?e..
1c0162cc:	1970 b9b5 6c66 b9b1 4c96 b9ab 738c b9c7     p...fl...L...s..
1c0162dc:	c682 b9c3 a6b2 b9bd 1970 b9b5 6c66 b9b1     ........p...fl..
1c0162ec:	4c96 b9ab 738c b9c7 c682 b9c3 a6b2 b9bd     .L...s..........
1c0162fc:	406e b9dc 9364 b9d8 7394 b9d2 738c b9c7     n@..d....s...s..
1c01630c:	c682 b9c3 a6b2 b9bd 406e b9dc 9364 b9d8     ........n@..d...
1c01631c:	7394 b9d2 8017 b9f3 d30c b9ef b33c b9e9     .s..........<...
1c01632c:	51a0 b972 0a59 b96b ea3c b95e 90cd b979     .Qr.Y.k.<.^...y.
1c01633c:	4986 b972 296a b966 d468 b982 618b b97e     .Ir.j)f.h....a~.
1c01634c:	416d b972 90cd b979 4986 b972 296a b966     mAr...y..Ir.j)f.
1c01635c:	d468 b982 618b b97e 416d b972 4cd6 b98b     h....a~.mAr..L..
1c01636c:	a932 b987 9924 b981 d468 b982 618b b97e     2...$...h....a~.
1c01637c:	416d b972 4cd6 b98b a932 b987 9924 b981     mAr..L..2...$...
1c01638c:	31ae b996 8e0b b992 7dfd b98c 4cd6 b98b     .1.......}...L..
1c01639c:	a932 b987 9924 b981 31ae b996 8e0b b992     2...$....1......
1c0163ac:	7dfd b98c 82f2 b9a3 df50 b99f cf41 b999     .}......P...A...
1c0163bc:	31ae b996 8e0b b992 7dfd b98c 82f2 b9a3     .1.......}......
1c0163cc:	df50 b99f cf41 b999 40a2 b9b3 9cfe b9af     P...A....@......
1c0163dc:	8cef b9a9 82f2 b9a3 df50 b99f cf41 b999     ........P...A...
1c0163ec:	40a2 b9b3 9cfe b9af 8cef b9a9 6abb b9c5     .@...........j..
1c0163fc:	c718 b9c1 b70b b9bb 40a2 b9b3 9cfe b9af     .........@......
1c01640c:	8cef b9a9 6abb b9c5 c718 b9c1 b70b b9bb     .....j..........
1c01641c:	0140 b9da 5d9e b9d6 4d8f b9d0 6abb b9c5     @....]...M...j..
1c01642c:	c718 b9c1 b70b b9bb 0140 b9da 5d9e b9d6     ........@....]..
1c01643c:	4d8f b9d0 0430 b9f1 608e b9ed 507f b9e7     .M..0....`...P..
1c01644c:	d53f b96f a0c6 b968 a02e b95c 00a9 b977     ?.o...h...\...w.
1c01645c:	cc30 b96f cb95 b963 7c19 b981 c3ba b97b     0.o...c..|....{.
1c01646c:	c31f b96f 00a9 b977 cc30 b96f cb95 b963     ..o...w.0.o...c.
1c01647c:	7c19 b981 c3ba b97b c31f b96f ddee b989     .|....{...o.....
1c01648c:	43b2 b986 4365 b980 7c19 b981 c3ba b97b     .C..eC...|....{.
1c01649c:	c31f b96f ddee b989 43b2 b986 4365 b980     ..o......C..eC..
1c0164ac:	a5d5 b994 0b98 b991 0b4c b98b ddee b989     ........L.......
1c0164bc:	43b2 b986 4365 b980 a5d5 b994 0b98 b991     .C..eC..........
1c0164cc:	0b4c b98b d3cb b9a1 398f b99e 3942 b998     L........9..B9..
1c0164dc:	a5d5 b994 0b98 b991 0b4c b98b d3cb b9a1     ........L.......
1c0164ec:	398f b99e 3942 b998 67d3 b9b1 cd97 b9ad     .9..B9...g......
1c0164fc:	cd49 b9a7 d3cb b9a1 398f b99e 3942 b998     I........9..B9..
1c01650c:	67d3 b9b1 cd97 b9ad cd49 b9a7 61ec b9c3     .g......I....a..
1c01651c:	c7ae b9bf c762 b9b9 67d3 b9b1 cd97 b9ad     ....b....g......
1c01652c:	cd49 b9a7 61ec b9c3 c7ae b9bf c762 b9b9     I....a......b...
1c01653c:	c213 b9d7 27d8 b9d4 278b b9ce 61ec b9c3     .....'...'...a..
1c01654c:	c7ae b9bf c762 b9b9 c213 b9d7 27d8 b9d4     ....b........'..
1c01655c:	278b b9ce 884b b9ee ee10 b9ea edc3 b9e4     .'..K...........
1c01656c:	58e0 b96d 3735 b966 561e b95a 7082 b974     .Xm.57f..VZ..pt.
1c01657c:	4ed8 b96d 6dc2 b961 23c9 b980 25e8 b979     .Nm..ma..#...%y.
1c01658c:	44d0 b96d 7082 b974 4ed8 b96d 6dc2 b961     .Dm..pt..Nm..ma.
1c01659c:	23c9 b980 25e8 b979 44d0 b96d 6f07 b988     .#...%y..Dm..o..
1c0165ac:	de32 b984 db4c b97d 23c9 b980 25e8 b979     2...L.}..#...%y.
1c0165bc:	44d0 b96d 6f07 b988 de32 b984 db4c b97d     .Dm..o..2...L.}.
1c0165cc:	19fc b993 8925 b98f 989a b989 6f07 b988     ....%........o..
1c0165dc:	de32 b984 db4c b97d 19fc b993 8925 b98f     2...L.}.....%...
1c0165ec:	989a b989 24a4 b9a0 93cf b99c a344 b996     .....$......D...
1c0165fc:	19fc b993 8925 b98f 989a b989 24a4 b9a0     ....%........$..
1c01660c:	93cf b99c a344 b996 8f04 b9af fe30 b9ab     ....D.......0...
1c01661c:	0da3 b9a6 24a4 b9a0 93cf b99c a344 b996     .....$......D...
1c01662c:	8f04 b9af fe30 b9ab 0da3 b9a6 591a b9c1     ....0........Y..
1c01663c:	c845 b9bd d7ba b9b7 8f04 b9af fe30 b9ab     E...........0...
1c01664c:	0da3 b9a6 591a b9c1 c845 b9bd d7ba b9b7     .....Y..E.......
1c01665c:	82e6 b9d5 f211 b9d1 0185 b9cc 591a b9c1     .............Y..
1c01666c:	c845 b9bd d7ba b9b7 82e6 b9d5 f211 b9d1     E...............
1c01667c:	0185 b9cc 0c67 b9ec 7b93 b9e8 8b07 b9e2     ....g....{......
1c01668c:	dc7f b96a cda3 b963 0c0e b958 e05e b971     ..j...c...X.^.q.
1c01669c:	d181 b96a 0fee b95f 96f3 b97d 8816 b976     ..j..._...}...v.
1c0166ac:	c682 b96a e05e b971 d181 b96a 0fee b95f     ..j.^.q...j..._.
1c0166bc:	96f3 b97d 8816 b976 c682 b96a 001f b987     ..}...v...j.....
1c0166cc:	78b1 b983 2fcf b97b 96f3 b97d 8816 b976     .x.../{...}...v.
1c0166dc:	c682 b96a 001f b987 78b1 b983 2fcf b97b     ..j......x.../{.
1c0166ec:	8e20 b991 06b2 b98e 25e8 b988 001f b987      ........%......
1c0166fc:	78b1 b983 2fcf b97b 8e20 b991 06b2 b98e     .x.../{. .......
1c01670c:	25e8 b988 757e b99e ee10 b99a 0d45 b995     .%..~u......E...
1c01671c:	8e20 b991 06b2 b98e 25e8 b988 757e b99e      ........%..~u..
1c01672c:	ee10 b99a 0d45 b995 b636 b9ad 2ec8 b9aa     ....E...6.......
1c01673c:	4dfe b9a4 757e b99e ee10 b99a 0d45 b995     .M..~u......E...
1c01674c:	b636 b9ad 2ec8 b9aa 4dfe b9a4 5049 b9bf     6........M..IP..
1c01675c:	c8db b9bb e812 b9b5 b636 b9ad 2ec8 b9aa     ........6.......
1c01676c:	4dfe b9a4 5049 b9bf c8db b9bb e812 b9b5     .M..IP..........
1c01677c:	43b9 b9d3 bc4a b9cf db80 b9c9 5049 b9bf     .C..J.......IP..
1c01678c:	c8db b9bb e812 b9b5 43b9 b9d3 bc4a b9cf     .........C..J...
1c01679c:	db80 b9c9 9083 b9e9 0915 b9e6 284b b9e0     ............K(..
1c0167ac:	6020 b968 6411 b961 c1ff b955 5038 b96f      `h..da...U.8Po.
1c0167bc:	542a b968 b219 b95c e654 b97a ea44 b973     *Th...\.T.z.D.s.
1c0167cc:	4833 b968 5038 b96f 542a b968 b219 b95c     3Hh.8Po.*Th...\.
1c0167dc:	e654 b97a ea44 b973 4833 b968 9138 b985     T.z.D.s.3Hh.8...
1c0167ec:	1331 b982 844f b978 e654 b97a ea44 b973     1...O.x.T.z.D.s.
1c0167fc:	4833 b968 9138 b985 1331 b982 844f b978     3Hh.8...1...O.x.
1c01680c:	0246 b990 843f b98c b337 b986 9138 b985     F...?...7...8...
1c01681c:	1331 b982 844f b978 0246 b990 843f b98c     1...O.x.F...?...
1c01682c:	b337 b986 c658 b99c 484f b999 7747 b993     7...X...OH..Gw..
1c01683c:	0246 b990 843f b98c b337 b986 c658 b99c     F...?...7...X...
1c01684c:	484f b999 7747 b993 dd69 b9ab 5f60 b9a8     OH..Gw..i...`_..
1c01685c:	8e58 b9a2 c658 b99c 484f b999 7747 b993     X...X...OH..Gw..
1c01686c:	dd69 b9ab 5f60 b9a8 8e58 b9a2 477a b9bd     i...`_..X...zG..
1c01687c:	c971 b9b9 f869 b9b3 dd69 b9ab 5f60 b9a8     q...i...i...`_..
1c01688c:	8e58 b9a2 477a b9bd c971 b9b9 f869 b9b3     X...zG..q...i...
1c01689c:	048c b9d1 8684 b9cd b57b b9c7 477a b9bd     ........{...zG..
1c0168ac:	c971 b9b9 f869 b9b3 048c b9d1 8684 b9cd     q...i...........
1c0168bc:	b57b b9c7 149e b9e7 9697 b9e3 c58e b9dd     {...............
1c0168cc:	e3c0 b965 fa7e b95e 77f0 b953 c014 b96c     ..e.~.^..wS...l.
1c0168dc:	d6d2 b965 5444 b95a 35b3 b978 4c72 b971     ..e.DTZ..5x.rLq.
1c0168ec:	c9e4 b965 c014 b96c d6d2 b965 5444 b95a     ..e...l...e.DTZ.
1c0168fc:	35b3 b978 4c72 b971 c9e4 b965 2251 b984     .5x.rLq...e.Q"..
1c01690c:	adb0 b980 d8d2 b975 35b3 b978 4c72 b971     ......u..5x.rLq.
1c01691c:	c9e4 b965 2251 b984 adb0 b980 d8d2 b975     ..e.Q"........u.
1c01692c:	766d b98e 01cc b98b 4085 b985 2251 b984     mv.......@..Q"..
1c01693c:	adb0 b980 d8d2 b975 766d b98e 01cc b98b     ......u.mv......
1c01694c:	4085 b985 1731 b99b a28f b997 e148 b991     .@..1.......H...
1c01695c:	766d b98e 01cc b98b 4085 b985 1731 b99b     mv.......@..1...
1c01696c:	a28f b997 e148 b991 0499 b9aa 8ff8 b9a6     ....H...........
1c01697c:	ceb2 b9a0 1731 b99b a28f b997 e148 b991     ....1.......H...
1c01698c:	0499 b9aa 8ff8 b9a6 ceb2 b9a0 3ea9 b9bb     .............>..
1c01699c:	ca08 b9b7 08c1 b9b2 0499 b9aa 8ff8 b9a6     ................
1c0169ac:	ceb2 b9a0 3ea9 b9bb ca08 b9b7 08c1 b9b2     .....>..........
1c0169bc:	c55e b9ce 50be b9cb 8f77 b9c5 3ea9 b9bb     ^....P..w....>..
1c0169cc:	ca08 b9b7 08c1 b9b2 c55e b9ce 50be b9cb     ........^....P..
1c0169dc:	8f77 b9c5 98b9 b9e4 2419 b9e1 62d2 b9db     w........$...b..

1c0169ec <INPUT_GRAD>:
1c0169ec:	b36d bd8f b94d be24 64bc be8b 1af6 be4f     m...M.$..d....O.
1c0169fc:	85da bde4 dd97 be23 d40c beb8 6f20 bf1a     ......#..... o..
1c016a0c:	1aa9 bee3 79e8 be78 e10b be89 8c71 bf19     .....yx.....q...
1c016a1c:	f974 bf7d 2956 bf39 2626 bec9 60b0 be4b     t.}.V)9.&&...`K.
1c016a2c:	348c bee0 e152 bf37 1ec0 bf05 c0d9 be8f     .4..R.7.........
1c016a3c:	aa88 bdde 6b32 be73 3dce bec6 a8c6 be8e     ....2ks..=......
1c016a4c:	432a be19 5ec7 bdb9 7088 be4e bada beaa     *C...^...pN.....
1c016a5c:	f183 be78 37eb be07 5e3a be4d 6092 bee2     ..x..7..:^M..`..
1c016a6c:	a534 bf39 6340 bf06 1c96 be91 ddb6 bea8     4.9.@c..........
1c016a7c:	9206 bf38 4a6e bf96 4667 bf58 51cd bee8     ..8.nJ..gFX..Q..
1c016a8c:	845b be74 b20c bf04 d1ab bf56 c62f bf19     [.t.......V./...
1c016a9c:	7200 bea4 cdf7 be03 343e be8e 0874 bee5     .r......>4..t...
1c016aac:	3714 bea3 db2e be2d 0a20 bde3 27c4 be78     .7....-. ....'x.
1c016abc:	10f6 beca 6408 be91 2ce8 be1c dede be76     .....d...,....v.
1c016acc:	f68c bf05 db4a bf58 392b bf1b fc3a bea5     ....J.X.+9..:...
1c016adc:	da62 bec7 979c bf57 9822 bfad 637a bf77     b.....W."...zcw.
1c016aec:	beba bf03 d402 be8e 49d2 bf19 c204 bf75     .........I....u.
1c016afc:	6d9e bf2e 2325 beb9 46aa be18 b2e2 bea2     .m..%#...F......
1c016b0c:	e98e bf01 c560 beb7 7334 be42 5abd be06     ....`...4sB..Z..
1c016b1c:	ef80 be90 6714 bee9 4f4e bea6 21e8 be31     .....g..NO...!1.
1c016b2c:	2fc0 be90 bccf bf1a 115d bf78 0f17 bf30     ./......].x...0.
1c016b3c:	dbde beba d70e bee6 9d30 bf76 e5d6 bfc4     ........0.v.....
1c016b4c:	4046 bf8b 548b bf13 65d7 bea3 e198 bf2d     F@...T...e....-.
1c016b5c:	592f bf8a 150c bf43 d44c becd bf5d be2c     /Y....C.L...].,.
1c016b6c:	3188 beb7 4ee1 bf11 53ae becc 0b38 be57     .1...N...S..8.W.
1c016b7c:	306a be1b cb1e bea5 5e98 bf04 3a96 bebb     j0.......^...:..
1c016b8c:	16e6 be46 f012 bea4 8312 bf2f a3b8 bf8b     ..F......./.....
1c016b9c:	e503 bf44 bb7f becf e9dc bf02 d162 bf8a     ..D.........b...
1c016bac:	338a bfdc cece bf9a ea5e bf22 f7ac beb7     .3......^.".....
1c016bbc:	795e bf42 d15c bf99 bc7c bf57 8572 bee2     ^yB.\...|.W.r...
1c016bcc:	3812 be41 b02c becb b435 bf20 e1fb bee0     .8A.,...5. .....
1c016bdc:	a33d be6b 0617 be30 a6bc beba 09a7 bf14     =.k...0.........
1c016bec:	25dc bed0 0be2 be5b b064 beb9 4954 bf44     .%....[.d...TID.
1c016bfc:	3ec3 bf9b baec bf59 9b22 bee4 6832 bf12     .>....Y."...2h..
1c016c0c:	542d bf9a 813d bff3 5d58 bfaa 8031 bf32     -T..=...X]..1.2.
1c016c1c:	8982 becc 1125 bf57 4989 bfa9 63e9 bf6c     ....%.W..I...cl.
1c016c2c:	3698 bef7 b0c5 be55 2ed0 bee0 1988 bf30     .6....U.......0.
1c016c3c:	7048 bef5 1da1 be80 dbc4 be44 825a becf     Hp........D.Z...
1c016c4c:	b4b5 bf23 1122 bee5 00e0 be70 70b5 bece     ..#.".....p..p..
1c016c5c:	0f97 bf59 d9cc bfaa 90d9 bf6e 7ac6 bef9     ..Y.......n..z..
1c016c6c:	e688 bf21 d6f8 bfa9 6779 c005 ebe0 bfb9     ..!.....yg......
1c016c7c:	1604 bf42 1b56 bee1 a8ec bf6b c1b4 bfb8     ..B.V.....k.....
1c016c8c:	85ac bf80 f3e0 bf05 2978 be6a ad75 bef4     ........x)j.u...
1c016c9c:	7edc bf3f ff4b bf04 69a4 be8a b170 be59     .~?.K....i..p.Y.
1c016cac:	5df8 bee4 5fc3 bf33 fc68 bef9 7af0 be82     .]..._3.h....z..
1c016cbc:	3108 bee3 d5da bf6d 74d7 bfba b362 bf81     .1....m..t..b...
1c016ccc:	2d34 bf07 64dd bf31 59c3 bfb9 0e52 c011     4-...d1..Y..R...
1c016cdc:	7a6a bfc9 abd7 bf51 ad2b bef5 2058 bf80     jz....Q.+...X ..
1c016cec:	39e2 bfc8 d964 bf8a 4c72 bf10 a22c be7e     .9..d...rL..,.~.
1c016cfc:	960d bf04 e430 bf4e 4672 bf0f b5a6 be94     ....0.N.rF......

1c016d0c <pos_uart>:
	...
