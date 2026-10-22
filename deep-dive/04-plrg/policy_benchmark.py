from engine import Function, Const, Call, Div, Return, BranchPositive
from policy_engine import Analyzer


def generate_program(n):
    assert n >= 2
    p = {
        'transform': Function('x', {
            'entry': ([], BranchPositive('x', 'positive', 'zero')),
            'positive': ([], Return('x')),
            'zero': ([Const('r', 0)], Return('r')),
        }),
        'wrapper': Function('x', {
            'entry': ([Call('r', 'transform', 'x', 'wrapper:transform')], Return('r')),
        }),
    }
    statements = []
    for i in range(n):
        statements.extend([
            Const(f'x{i}', 10 if i == 0 else -10),
            Call(f'r{i}', 'wrapper', f'x{i}', f'main:call_{i}'),
        ])
    statements.append(Div('z', 100, 'r0', 'main:divide'))
    p['main'] = Function(None, {'entry': (statements, Return('z'))})
    return p


POLICIES = {
    'uniform-0': {},
    'uniform-1': {'wrapper': 1, 'transform': 1},
    'uniform-2': {'wrapper': 2, 'transform': 2},
    'only-transform-2': {'transform': 2},
    'wrapper-1+transform-2': {'wrapper': 1, 'transform': 2},
    'wrapper-2+transform-1': {'wrapper': 2, 'transform': 1},
}


def main():
    print(f"{'calls':>5} {'policy':>24} {'ctx':>5} {'runs':>5} {'steps':>6} {'FP':>5}")
    for n in [2, 4, 8, 16]:
        program = generate_program(n)
        for name, policy in POLICIES.items():
            r = Analyzer(policy, program).run()
            print(f"{n:>5} {name:>24} {r.contexts:>5} {r.reanalyses:>5} "
                  f"{r.transfer_count:>6} {str('main:divide' in r.warnings):>5}")
        print()


if __name__ == '__main__':
    main()
