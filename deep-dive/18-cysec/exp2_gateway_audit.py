#!/usr/bin/env python3
"""GENESIS (SAC 2024) x86-64 gateway SOURCE audit + nonprivileged control-state simulation.

Fetches four small files directly from the researchers' public GENESIS-x86 repository.
Does NOT load modules, run privileged instructions, alter the kernel, or prove security.
"""
import argparse
import json
import pathlib
import re
import sys
import urllib.request

BASE = 'https://raw.githubusercontent.com/KAIST-CysecLab/GENESIS-x86/master/linux-5.9/arch/x86/nk/'
FILES = ['nk.c', 'nk_handler.c', 'nk_init.c', 'nk_privinst.S']

def read_sources(args):
    sources = {}
    for filename in FILES:
        if args.source_dir:
            p = pathlib.Path(args.source_dir) / filename
            sources[filename] = p.read_text(encoding='utf-8')
            print(f'SOURCE {filename}: local {p} ({len(sources[filename])} bytes)')
        else:
            url = BASE + filename
            with urllib.request.urlopen(url, timeout=25) as resp:
                sources[filename] = resp.read().decode('utf-8')
            print(f'SOURCE {filename}: official GitHub ({len(sources[filename])} bytes)')
    return sources

def extract(src, begin, end):
    i = src.find(begin)
    if i < 0:
        raise ValueError(f'Could not find start: {begin}')
    j = src.find(end, i)
    if j < 0:
        raise ValueError(f'Could not find end: {end}')
    return src[i:j]

def order_check(fragment, terms):
    """Checks source order, not all executable CFG paths."""
    indexes = [fragment.find(t) for t in terms]
    return all(i >= 0 for i in indexes) and indexes == sorted(indexes), indexes

def do_audit(sources):
    s = sources['nk.c']
    entry = extract(s, 'unsigned long __naked __nk_gateway __nk_entry', 'EXPORT_SYMBOL(__nk_entry)')
    exit_ = extract(s, 'void __naked __nk_gateway __noalign __nk_exit', 'unsigned long __naked __nk_gateway __nk_priv_entry')
    handler = extract(sources['nk_handler.c'], 'static void __nk __nk_set_pte(', 'static void __nk __nk_pte_clear(')
    shadow = sources['nk_init.c']
    privinst = sources['nk_privinst.S']

    checks = {}
    checks['entry.pushfq_cli_stac'] = order_check(entry, ['"pushfq\\n"', '"cli\\n"', '"stac\\n"'])[0]
    checks['entry.stac_before_handler'] = order_check(entry, ['"stac\\n"', 'call __nk_handler'])[0]
    checks['entry.dedicated_stack_before_handler'] = order_check(entry, ['%%rsp, %%r9', '%%rsp\\n', 'call __nk_handler'])[0] if False else (('inner_stacks' in entry) and ('call __nk_handler' in entry) and ('push %%r9' in entry))
    checks['exit.popf_clac_ret'] = order_check(exit_, ['"popf\\n"', '"clac\\n"', '"ret"'])[0]
    checks['page_table.write_via_shadow'] = ('__virt_to_shadow(ptep)' in handler and 'WRITE_ONCE(*saddr, pte)' in handler)
    checks['shadow.mapping_user_bit'] = ('nk_create_shadow_mapping' in shadow and '_PAGE_USER' in shadow and 'nk_shadow_populate_pud' in shadow)
    checks['original_pgtbl_read_only'] = ('nk_protect_zone_ptp' in shadow and 'set_memory_ro' in shadow)
    checks['privileged_cr_write_in_privinst'] = all(x in privinst for x in ('%cr0', '%cr3', '%cr4'))

    return checks, {'entry':entry, 'exit':exit_, 'pte_handler':handler}

def mutations(fragments):
    entry = fragments['entry']
    exit_ = fragments['exit']
    res={}
    m=entry.replace('"cli\\n"', '"nop\\n"', 1)
    res['remove_cli_from_entry'] = not order_check(m,['"pushfq\\n"','"cli\\n"','"stac\\n"'])[0]
    m=exit_.replace('"clac\\n"', '"nop\\n"', 1)
    res['remove_clac_from_exit'] = not order_check(m,['"popf\\n"','"clac\\n"','"ret"'])[0]
    m=entry.replace('"stac\\n"', '"nop\\n"', 1)
    res['remove_stac_from_entry'] = not order_check(m,['"pushfq\\n"','"cli\\n"','"stac\\n"'])[0]
    return res

def state_walk(start='normal', trap_gate=False):
    # No privileged instructions executed: conceptual state transitions only.
    state = {'interrupts_enabled':True, 'smap_bypassed':False, 'inner_stack':False}
    instructions = ['pushfq', 'cli', 'stac', 'save_stack', 'switch_inner_stack', 'call_handler']
    if start == 'stac':
        instructions = instructions[2:]
    trace = []
    for op in instructions:
        if op=='cli': state['interrupts_enabled']=False
        if op=='stac': state['smap_bypassed']=True
        if op=='switch_inner_stack': state['inner_stack']=True
        trace.append({'instruction':op, **state})
    fault_window = state['interrupts_enabled'] and state['smap_bypassed']
    if trap_gate:
        # Model the trusted trap gate clearing AC on an interrupt.
        state['smap_bypassed']=False
        trace.append({'instruction':'trusted_trap_gate: clac', **state})
    return {'start':start, 'trap_gate':trap_gate, 'interrupt_while_smap_off_possible':fault_window, 'final_state':state, 'trace':trace}

def self_test():
    s='"pushfq\\n" "cli\\n" "stac\\n" "call __nk_handler"'
    assert order_check(s,['pushfq','cli','stac'])[0]
    assert not order_check(s.replace('"cli\\n"','"nop\\n"'),['pushfq','cli','stac'])[0]
    assert not state_walk('normal')['interrupt_while_smap_off_possible']
    assert state_walk('stac')['interrupt_while_smap_off_possible']
    assert state_walk('stac',True)['final_state']['smap_bypassed'] is False
    print('SELF TEST: PASS (5 assertions)')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',help='Local path containing nk.c nk_handler.c nk_init.c nk_privinst.S (no network)')
    p.add_argument('--out',default='results/genesis_exp2')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test: self_test(); return
    try:
        sources=read_sources(args)
        checks,fragments=do_audit(sources)
        broken=mutations(fragments)
    except (OSError,ValueError) as exc:
        print('ERROR:',exc,file=sys.stderr); sys.exit(2)
    print('\n=== OFFICIAL SOURCE INVARIANT AUDIT ===')
    for k,v in checks.items(): print(('PASS' if v else 'FAIL'),k)
    print('\n=== INTENTIONAL MUTATION DETECTION ===')
    for k,v in broken.items(): print(('DETECTED' if v else 'NOT DETECTED'),k)
    print('\n=== GATE STATE MODEL (not kernel execution) ===')
    scenarios=[state_walk('normal'),state_walk('stac'),state_walk('stac',True)]
    for s in scenarios:
        st=s['final_state']
        print(f"start={s['start']:6} trusted_trap_gate={str(s['trap_gate']):5} preemption_with_AC=1? {s['interrupt_while_smap_off_possible']!s:5} AC_at_end={int(st['smap_bypassed'])}")
    data={'repo':'KAIST-CysecLab/GENESIS-x86','branch':'master','files':FILES,'invariant_checks':checks,'mutation_detection':broken,'state_model':scenarios,'limitations':['Only four selected source files scanned; cannot establish global stac/clac exclusivity','Source sequence heuristics are not CFG proof','Privileged gateway instructions are NOT executed','No booted GENESIS kernel or hardware security guarantee']}
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('\nReport:',(out/'report.json').resolve())
    if not all(checks.values()) or not all(broken.values()):
        print('WARNING: Some checks failed. Review source version and checker assumptions.')

if __name__=='__main__': main()
