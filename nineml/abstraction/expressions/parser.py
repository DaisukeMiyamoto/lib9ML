from __future__ import division
from itertools import chain, izip
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr as sympy_parse, standard_transformations, convert_xor)
from sympy.parsing.sympy_tokenize import NAME, OP
from sympy.functions import Piecewise
import operator
import re
from nineml.exceptions import NineMLMathParseError
from .base import builtin_constants, builtin_functions

# # Inline randoms are deprecated in favour of RandomVariable elements,
# # but included here to get Brunel model to work
# inline_random_distributions = set(('random.uniform', 'random.binomial',
#                                    'random.poisson', 'random.exponential'))


class Parser(object):
    # Escape all objects in sympy namespace that aren't defined in NineML
    # by predefining them as symbol names to avoid naming conflicts when
    # sympifying RHS strings.
    _to_escape = set(s for s in dir(sympy)
                     if s not in chain(builtin_constants, builtin_functions))
    _valid_funcs = set((sympy.And, sympy.Or, sympy.Not)) | builtin_functions
    _func_to_op_map = {sympy.Function('pow'): operator.pow}
    _escape_random_re = re.compile(r'(?<!\w)random\.(\w+)(?!\w)')
    _unescape_random_re = re.compile(r'(?<!\w)random_(\w+)_(?!\w)')
    _ternary_split_re = re.compile(r'[\?:]')
    _match_first_re = re.compile(r'((?:-)?(?:\w+|[\d\.]+) *)$')
    _match_second_re = re.compile(r' *(?:-)?[\w\d\.]+')
    _sympy_transforms = list(standard_transformations) + [convert_xor]
    inline_randoms_dict = {
        'random_uniform_': sympy.Function('random_uniform_'),
        'random_binomial_': sympy.Function('random_binomial_'),
        'random_poisson_': sympy.Function('random_poisson_'),
        'random_exponential_': sympy.Function('random_exponential_')}

    def __init__(self):
        self.escaped_names = set()

    def parse(self, expr):
        if not isinstance(expr, (int, float)):
            if isinstance(expr, sympy.Basic):
                self._check_valid_funcs(expr)
            elif isinstance(expr, basestring):
                    # Check to see whether expression contains a ternary op.
                    if '?' in expr:
                        return Piecewise(*self._split_pieces(expr))
                    else:
                        return self._parse_expr(expr)
            else:
                raise TypeError("Cannot convert value '{}' of type '{}' to "
                                " SymPy expression".format(repr(expr),
                                                           type(expr)))
        return expr

    def _parse_expr(self, expr):
        expr = self.escape_random_namespace(expr)
        expr = self._escape_relationals(expr)
        try:
            expr = sympy_parse(
                expr, transformations=([self] + self._sympy_transforms),
                local_dict=self.inline_randoms_dict)
        except Exception, e:
            raise NineMLMathParseError(
                "Could not parse math-inline expression: "
                "{}\n\n{}".format(expr, e))
        return self._postprocess(expr)

    def _preprocess(self, tokens):
        """
        Escapes symbols that correspond to objects in SymPy but are not
        reserved identifiers in NineML
        """
        result = []
        # Loop through single tokens
        for toknum, tokval in tokens:
            if toknum == NAME:
                # Escape non-reserved identifiers in 9ML that are reserved
                # keywords in Sympy
                if tokval in self._to_escape:
                    self.escaped_names.add(tokval)
                    tokval = self._escape(tokval)
                # Convert logical identities from ANSI to Python names
                elif tokval == 'true':
                    tokval = 'True'
                elif tokval == 'false':
                    tokval = 'False'
                elif tokval.endswith('__'):
                    tokval = tokval[:-2]
            # Handle multiple negations
            elif toknum == OP and tokval.startswith('!'):
                # NB: Multiple !'s are grouped into the one token
                assert all(t == '!' for t in tokval)
                if len(tokval) % 2:
                    tokval = '~'  # odd number of negation symbols
                else:
                    continue  # even number of negation symbols, cancel out
            result.append((toknum, tokval))
        new_result = []
        # Loop through pairwise combinations
        pair_iterator = izip(result[:-1], result[1:])
        for (toknum, tokval), (next_toknum, next_tokval) in pair_iterator:
            # Handle trivial corner cases where the logical identities
            # (i.e. True and False) are immediately negated
            # as Sympy casts True and False to the Python native objects,
            # and then the '~' gets interpreted as a bitwise shift rather
            # than a negation.
            if toknum == OP and tokval == '~':
                if next_toknum == OP and next_tokval == '~':
                    # Skip this and the next iteration as the double negation
                    # cancels itself out
                    next(pair_iterator)
                    continue
                elif next_toknum == NAME and next_tokval in ('True', 'False'):
                    # Manually drop the negation sign and negate
                    tokval = 'True' if next_tokval is 'False' else 'False'
                    toknum = NAME
                    next(pair_iterator)  # Skip the next iteration
            # Convert the ANSI C89 standard for logical
            # 'and' and 'or', '&&' or '||', to the Sympy format '&' and '|'
            elif ((toknum == OP and tokval in ('&', '|') and
                   next_toknum == OP and next_tokval == tokval)):
                next(pair_iterator)  # Skip the next iteration
            new_result.append((toknum, tokval))
        return new_result

    def _postprocess(self, expr):
        # Convert symbol names that were escaped to avoid clashes with in-built
        # Sympy functions back to their original form
        while self.escaped_names:
            name = self.escaped_names.pop()
            expr = expr.xreplace(
                {sympy.Symbol(self._escape(name)): sympy.Symbol(name)})
        # Convert ANSI C functions to corresponding operator (i.e. 'pow')
        expr = self._func_to_op(expr)
        return expr

    def __call__(self, tokens, local_dict, global_dict):  # @UnusedVariable
        """
        Wrapper function so processor can be passed as a 'transformation'
        to the Sympy parser
        """
        return self._preprocess(tokens)

    def _split_pieces(self, expr):
        if '?' in expr:
            cond, remaining = expr.split('?', 1)
            try:
                if remaining.index('?') < remaining.find(':'):
                    raise NineMLMathParseError(
                        "Nested ternary statements are only permitted in the "
                        "second branch of the enclosing ternary statement: {}"
                        .format(expr))
            except ValueError:
                pass  # If there are no more '?'s in the expression
            try:
                subexpr, remaining = remaining.split(':', 1)
            except ValueError:
                raise NineMLMathParseError(
                    "Missing ':' in ternary statement: {}".format(expr))
            # Concatenate sub expressions of the piecewise.
            pieces = ([(self._parse_expr(subexpr), self._parse_expr(cond))] +
                      self._split_pieces(remaining))
        else:
            pieces = [(self._parse_expr(expr), True)]
        return pieces

    @classmethod
    def _escape(self, s):
        return s + '__escaped__'

    @classmethod
    def _func_to_op(self, expr):
        """Maps functions to SymPy operators (only 'pow' at this stage)"""
        if isinstance(expr, sympy.Function):
            args = (self._func_to_op(a) for a in expr.args)
            try:
                expr = self._func_to_op_map[type(expr)](*args)
            except KeyError:
                expr = expr.__class__(*args)
        return expr

    @classmethod
    def _check_valid_funcs(cls, expr):
        """Checks if the provided Sympy function is a valid 9ML function"""
        if (isinstance(expr, sympy.Function) and
                str(type(expr)) not in chain(
                    cls._valid_funcs, cls.inline_randoms_dict.iterkeys()) and
                not isinstance(expr, sympy.Piecewise)):
            raise NineMLMathParseError(
                "'{}' is a valid function in Sympy but not in 9ML"
                .format(type(expr)))
        for arg in expr.args:
            cls._check_valid_funcs(arg)

    @classmethod
    def escape_random_namespace(cls, expr):
        return cls._escape_random_re.sub(r'random_\1_', expr)

    @classmethod
    def unescape_random_namespace(cls, expr):
        return cls._unescape_random_re.sub(r'random.\1', expr)

    @classmethod
    def inline_random_distributions(cls):
        return cls.inline_randoms_dict.itervalues()

    @classmethod
    def _escape_relationals(cls, expr_string):
        tokenize_re = re.compile(r'\s*(&{1,2}|\|{1,2}|<=?|>=?|==?|'
                                 r'(?:\w+|!|~)?\s*\(|\))\s*')
        open_paren_re = re.compile(r'(?:\w+|!|~)?\s*\(')
        tokens = [t for t in tokenize_re.split(expr_string.strip()) if t]
        # Based on shunting-yard algorithm
        # (see http://en.wikipedia.org/wiki/Shunting-yard_algorithm)
        # with modifications for skipping over non logical/relational operators
        # and associated parentheses.
        operators = []  # stack (in SY algorithm terminology)
        operands = []  # output stream
        to_parse = [False]  # whether the current parenthesis should be parsed
        num_to_concat = [0]  # num. of operands concat when not parsing
        for tok in tokens:
            if open_paren_re.match(tok):
                operators.append(tok)
                to_parse.append(False)
                num_to_concat.append(0)
            elif tok == ')':
                operator = operators.pop()
                k = num_to_concat.pop()
                if to_parse.pop():
                    arg2, arg1 = operands.pop(), operands.pop()
                    operands.append(cls._op2func(operator, arg1, arg2))
                    try:
                        assert operators.pop() == '('
                    except (IndexError, AssertionError):
                        raise NineMLMathParseError(
                            "Unbalanced parentheses in expression: {}"
                            .format(expr_string))
                else:
                    operand = ''.join(operands[-k:])
                    operands = operands[:-k]
                    operands.append(operator + operand + tok)
                    num_to_concat[-1] += 1
            elif tokenize_re.match(tok):
                operators.append(tok)
                to_parse[-1] = True  # parse the last set of parenthesis
            else:
                operands.append(tok)
                num_to_concat[-1] += 1
        for operator in reversed(operators):
            if operator == '(':
                raise NineMLMathParseError(
                    "Unbalanced parentheses in expression: {}"
                    .format(expr_string))
            arg2, arg1 = operands.pop(), operands.pop()
            operands.append(cls._op2func(operator, arg1, arg2))
        return ''.join(operands)

    @classmethod
    def _op2func(cls, operator, arg1, arg2):
        if operator.startswith('&'):
            func = "And__({}, {})".format(arg1, arg2)
        elif operator.startswith('|'):
            func = "Or__({}, {})".format(arg1, arg2)
        elif operator.startswith('='):
            func = "Eq__({}, {})".format(arg1, arg2)
        elif operator == '<':
            func = "Lt__({}, {})".format(arg1, arg2)
        elif operator == '>':
            func = "Gt__({}, {})".format(arg1, arg2)
        elif operator == '<=':
            func = "Le__({}, {})".format(arg1, arg2)
        elif operator == '>=':
            func = "Ge__({}, {})".format(arg1, arg2)
        else:
            assert False
        return func

    @classmethod
    def _escape_equalities(cls, string):
        if '==' in string:
            i = 0
            while i < (len(string) - 1):
                if string[i:i + 2] == '==':
                    before = string[:i]
                    after = string[i + 2:]
                    if before.rstrip().endswith(')'):
                        first = cls._match_bracket(before, open_bracket=')',
                                                   close_bracket='(',
                                                   direction='backwards')
                    else:
                        first = cls._match_first_re.search(before).group(1)
                    if after.lstrip().startswith('('):
                        second = cls._match_bracket(after, open_bracket='(',
                                                      close_bracket=')')
                        second = cls._escape_equalities(second)
                    else:
                        second = cls._match_second_re.match(after).group(0)
                    insert_string = '__equals__({}, {})'.format(first, second)
                    string = (string[:i - len(first)] + insert_string +
                              string[i + len(second) + 2:])
                    i += len(insert_string) - len(first)
                i += 1
        return string

    @classmethod
    def _match_bracket(cls, string, open_bracket, close_bracket,
                       direction='forwards'):
        depth = 0
        if direction == 'backwards':
            string = string[::-1]
        for i, c in enumerate(string):
            if c == open_bracket:
                depth += 1
            elif c == close_bracket:
                depth -= 1
                if depth == 0:
                    output = string[:i + 1]
                    if direction == 'backwards':
                        output = output[::-1]
                    return output
        raise NineMLMathParseError(
            "No matching '{}' found for opening '{}' in string '{}'"
            .format(close_bracket, open_bracket, string))
