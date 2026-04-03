def solve(expressions: list[str]) -> list[int]:
    """Evaluate arithmetic expressions."""
    results = []
    for expr in expressions:
        result = evaluate(expr)
        results.append(result)
    return results

def evaluate(expr):
    """Parse and evaluate an arithmetic expression using recursive descent."""
    tokens = tokenize(expr)
    pos = [0]

    def parse_expr():
        """Parse addition and subtraction (lowest precedence)."""
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
        """Parse multiplication (higher precedence)."""
        left = parse_factor()
        while pos[0] < len(tokens) and tokens[pos[0]] == '*':
            pos[0] += 1
            right = parse_factor()
            left = left * right
        return left

    def parse_factor():
        """Parse unary minus, numbers, and parenthesized expressions."""
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

def tokenize(expr):
    """Tokenize an arithmetic expression into numbers and operators."""
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
                # Unary minus: collect the number
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
