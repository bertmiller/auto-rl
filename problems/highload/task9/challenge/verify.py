import random
import time
import sys

def generate_expression(rng, num_count=1000):
    """Generate a random arithmetic expression with ~num_count numbers."""
    ops = ['+', '-', '*']
    tokens = []
    paren_depth = 0

    for i in range(num_count):
        if i > 0:
            op = rng.choice(ops)
            tokens.append(f' {op} ')

        opens = 0
        if rng.random() < 0.15 and paren_depth < 5:
            opens = rng.randint(1, 2)
            tokens.append('(' * opens)
            paren_depth += opens

        num = rng.randint(-32768, 32767)
        if num < 0:
            tokens.append(f'({num})')
        else:
            tokens.append(str(num))

        if paren_depth > 0 and (rng.random() < 0.15 or i == num_count - 1):
            closes = rng.randint(1, min(paren_depth, 2))
            tokens.append(')' * closes)
            paren_depth -= closes

    if paren_depth > 0:
        tokens.append(')' * paren_depth)

    return ''.join(tokens)

def tokenize(expr):
    """Tokenize an arithmetic expression."""
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
        elif expr[i] in '()+*':
            tokens.append(expr[i])
            i += 1
        elif expr[i] == '-':
            if not tokens or tokens[-1] in ('(', '+', '-', '*'):
                j = i + 1
                while j < len(expr) and expr[j].isdigit():
                    j += 1
                if j > i + 1:
                    tokens.append(expr[i:j])
                    i = j
                else:
                    tokens.append('-')
                    i += 1
            else:
                tokens.append('-')
                i += 1
        elif expr[i].isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(expr[i:j])
            i = j
        else:
            i += 1
    return tokens

def reference_eval(expr):
    """Evaluate expression using recursive descent parser."""
    tokens = tokenize(expr)
    pos = [0]

    def parse_expr():
        left = parse_term()
        while pos[0] < len(tokens) and tokens[pos[0]] in ('+', '-'):
            op = tokens[pos[0]]
            pos[0] += 1
            right = parse_term()
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left

    def parse_term():
        left = parse_factor()
        while pos[0] < len(tokens) and tokens[pos[0]] == '*':
            pos[0] += 1
            right = parse_factor()
            left = left * right
        return left

    def parse_factor():
        if tokens[pos[0]] == '(':
            pos[0] += 1
            val = parse_expr()
            pos[0] += 1  # skip ')'
            return val
        elif tokens[pos[0]] == '-' and (pos[0] == 0 or tokens[pos[0]-1] in ('(', '+', '-', '*')):
            pos[0] += 1
            return -parse_factor()
        else:
            val = int(tokens[pos[0]])
            pos[0] += 1
            return val

    return parse_expr()

def generate_data(n_expr=1000, nums_per_expr=2000, seed=42):
    """Generate n random arithmetic expressions."""
    rng = random.Random(seed)
    expressions = []
    for _ in range(n_expr):
        expr = generate_expression(rng, nums_per_expr)
        expressions.append(expr)
    return expressions

def main():
    expressions = generate_data()

    # Compute expected answers with reference implementation
    expected = []
    for expr in expressions:
        expected.append(reference_eval(expr))

    import solution

    # Timed run
    t0 = time.perf_counter()
    result = solution.solve(expressions)
    t1 = time.perf_counter()

    # Verify
    if not isinstance(result, list):
        print(f"ERROR: result is not a list, got {type(result)}", file=sys.stderr)
        sys.exit(1)
    if len(result) != len(expected):
        print(f"ERROR: expected {len(expected)} results, got {len(result)}", file=sys.stderr)
        sys.exit(1)
    for i, (got, exp) in enumerate(zip(result, expected)):
        if got != exp:
            print(f"ERROR: expression {i}: expected {exp}, got {got}", file=sys.stderr)
            print(f"  Expression: {expressions[i][:200]}...", file=sys.stderr)
            sys.exit(1)

    elapsed = t1 - t0
    print(f"time={elapsed:.6f}")

if __name__ == "__main__":
    main()
