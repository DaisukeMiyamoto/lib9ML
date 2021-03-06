
import unittest
from nineml.abstraction import (Expression,
                                      Alias, StateAssignment, TimeDerivative)
from nineml.abstraction.expressions import (
    ExpressionWithSimpleLHS, Constant)
from sympy.functions import Piecewise
from nineml.exceptions import NineMLMathParseError
from nineml.units import coulomb, S_per_cm2, mV
from nineml.abstraction.componentclass.visitors.xml import (
    ComponentClassXMLWriter as XMLWriter, ComponentClassXMLLoader as XMLLoader)
from nineml import Document
import sympy


class Expression_test(unittest.TestCase):

    def test_Valid(self):
        # rhs, expt_vars, expt_funcs, result, values
        valid_rhses = [
            (('a'), ('a'), (), 5, {'a': 5}),
            (('b'), ('b'), (), 7, {'b': 7}),
            (('a+b'), ('a', 'b'), (), 13, {'a': 12, 'b': 1}),
            (('1./(alpha+2*beta)'), ('alpha', 'beta'), (), 0.2,
             {'alpha': 1, 'beta': 2}),
        ]

        for rhs, exp_var, exp_func, exp_res, params in valid_rhses:
            e = Expression(rhs)
            self.assertEquals(set(e.rhs_symbol_names), set(exp_var))
            self.assertEquals(set(str(f) for f in e.rhs_funcs), set(exp_func))
            self.assertAlmostEqual(e.rhs_as_python_func(**params), exp_res,
                                   places=4)

        import numpy
        expr_vars = [
            ["-A/tau_r", ("A", "tau_r"), ()],
            ["V*V", ("V",), ()],
            ["a*(b*V - U)", ("U", "V", "b", "a"), ()],
            [" 0.04*V*V + 5.0*V + 1. + 140.0 - U + Isyn",
             ("V", "U", "Isyn"), ()],
            ["c", ("c"), ()],
            ["1", (), ()],
            ["atan2(sin(x),cos(y))", ("x", "y"),
             ("atan2", "sin", "cos")],
            ["1.*V", ("V"), ()],
            ["1.0", (), ()],
            [".1", (), ()],
            ["1/(1 + mg_conc*eta*exp(-1*gamma*V))", (
                "mg_conc", "eta", "gamma", "V"), ('exp',)],
            ["1 / ( 1 + mg_conc * eta *  exp( -1 * gamma*V))",
             ("mg_conc", "eta", "gamma", "V"), ('exp',)],
            ["1 / ( 1 + mg_conc * sin(0.5 * V) *  exp ( -1 * gamma*V))",
             ("mg_conc", "gamma", "V"), ('exp', "sin")],
            [".1 / ( 1.0 + mg_conc * sin(V) *  exp ( -1.0 * gamma*V))",
             ("mg_conc", "gamma", "V"), ('exp', "sin")],
            ["sin(w)", ("w"), ("sin",)]]

        namespace = {
            "A": 10.0,
            "tau_r": 11.0,
            "V": -70.0,
            "a": 1.2,
            "b": 3.0,
            "U": -80.0,
            "Isyn": 2.0,
            "c": 10.0,
            "mg_conc": 1.0,
            "eta": 2.0,
            "gamma": -20.0,
            "x": 1.0,
            "y": 1.0,
            "w": numpy.arange(10)
        }

        return_values = [-0.909090909091, 4900.0, -156.0, 69.0, 10.0, 1,
                         1.0, -70.0, 1.0, 0.1, 1.0, 1.0, 1.0, 0.1,
                         numpy.sin(namespace['w'])]

        for i, (expr, expt_vars, expt_funcs) in enumerate(expr_vars):
            c = Expression(expr)
            self.assertEqual(set(c.rhs_symbol_names), set(expt_vars))
            self.assertEqual(set(str(f) for f in c.rhs_funcs), set(expt_funcs))

            python_func = c.rhs_as_python_func
            param_dict = dict([(v, namespace[v]) for v in expt_vars])

            v = return_values[i] - python_func(**param_dict)
            self.assertAlmostEqual(numpy.dot(v, v), 0)

    def test_rhs_name_transform_inplace(self):
        # Signature: name(self, name_map)
                # Replace atoms on the RHS with values in the name_map

        e = Expression("V*sin(V)/(eta*mg_conc*exp(-V^2*gamma) + 1)")
        e.rhs_name_transform_inplace({'V': 'VNEW'})
        self.assertEquals(
            e.rhs_str, "VNEW*sin(VNEW)/(eta*mg_conc*exp(-VNEW^2*gamma) + 1)")

        # Don't Change builtin function names:
        e.rhs_name_transform_inplace({'sin': 'SIN'})
        self.assertEquals(
            e.rhs_str, "VNEW*sin(VNEW)/(eta*mg_conc*exp(-VNEW^2*gamma) + 1)")
        e.rhs_name_transform_inplace({'exp': 'EXP'})
        self.assertEquals(
            e.rhs_str, "VNEW*sin(VNEW)/(eta*mg_conc*exp(-VNEW^2*gamma) + 1)")

        # Check the attributes:
        self.assertEquals(set(e.rhs_atoms), set(
            ['VNEW', 'mg_conc', 'eta', 'gamma', 'exp', 'sin']))
        self.assertEquals(set(str(f) for f in e.rhs_funcs),
                          set(['exp', 'sin']))

    def test_escape_of_carets(self):
        self.assertEquals(Expression("a^2").rhs_cstr, 'a*a')
        self.assertEquals(Expression("(a - 2)^2").rhs_cstr,
                          '(a - 2)*(a - 2)')
        self.assertEquals(Expression("(a - (a - 2)^2.5)^2.5").rhs_cstr,
                          'pow(a - pow(a - 2, 2.5), 2.5)')
        self.assertEquals(Expression("a^(a - 2)").rhs_cstr, 'pow(a, a - 2)')


class AnsiC89ToSympy_test(unittest.TestCase):

    def setUp(self):
        self.a = sympy.Symbol('a')
        self.b = sympy.Symbol('b')
        self.c = sympy.Symbol('c')
        self.d = sympy.Symbol('d')
        self.e = sympy.Symbol('e')
        self.f = sympy.Symbol('f')
        self.g = sympy.Symbol('g')
        self.h = sympy.Symbol('h')
        self.i = sympy.Symbol('i')
        self.j = sympy.Symbol('j')

    def test_logical_and(self):
        expr = Expression('a && b')
        self.assertEqual(expr.rhs, sympy.And(self.a, self.b))

    def test_logical_or(self):
        expr = Expression('a || b')
        self.assertEqual(expr.rhs, sympy.Or(self.a, self.b))

    def test_equality(self):
        expr = Expression('a == b')
        self.assertEqual(expr.rhs, sympy.Eq(self.a, self.b))

    def test_equality_combined(self):
        expr = Expression('(a == b) && (c == d) || (e == f)')
        self.assertEqual(
            expr.rhs, sympy.Or(sympy.And(sympy.Eq(self.a, self.b),
                                         sympy.Eq(self.c, self.d)),
                               sympy.Eq(self.e, self.f)))

    def test_nested_relational(self):
        expr = Expression('((a == b) || (c == d)) && ((e == f) || (g < f))')
        self.assertEqual(
            expr.rhs, sympy.And(sympy.Or(sympy.Eq(self.a, self.b),
                                         sympy.Eq(self.c, self.d)),
                                sympy.Or(sympy.Eq(self.e, self.f),
                                         sympy.Lt(self.g, self.f))))

    def test_equality_nested_func(self):
        expr = Expression('((a == b) || (c == pow(d, 2))) && (e == f)')
        self.assertEqual(
            expr.rhs, sympy.And(sympy.Or(sympy.Eq(self.a, self.b),
                                         sympy.Eq(self.c, self.d ** 2)),
                                sympy.Eq(self.e, self.f)))

    def test_pow(self):
        expr = Expression('pow(a, b)')
        self.assertEqual(expr.rhs, self.a ** self.b)

    def test_negation(self):
        expr = Expression('!a')
        self.assertEqual(expr.rhs, sympy.Not(self.a))

    def test_double_negation(self):
        expr = Expression('!!a')
        self.assertEqual(expr.rhs, self.a)

    def test_triple_negation(self):
        expr = Expression('!!!a')
        self.assertEqual(expr.rhs, sympy.Not(self.a))

    def test_ternary_simple(self):
        expr = Expression('a < b ? c : d')
        self.assertEqual(
            expr.rhs, Piecewise((self.c, sympy.Lt(self.a, self.b)),
                                (self.d, True)))

    def test_ternary_bad(self):
        self.assertRaises(NineMLMathParseError,
                          Expression, 'a < b ? (c ? d : e) : f')

    def test_ternary_nested(self):
        expr = Expression('a < b ? c : d > e ? f : g == h ? i : j')
        self.assertEqual(
            expr.rhs, Piecewise((self.c, sympy.Lt(self.a, self.b)),
                                (self.f, sympy.Gt(self.d, self.e)),
                                (self.i, sympy.Eq(self.g, self.h)),
                                (self.j, True)))


class SympyToC89_test(unittest.TestCase):

    def setUp(self):
        self.a = sympy.Symbol('a')
        self.b = sympy.Symbol('b')
        self.c = sympy.Symbol('c')
        self.d = sympy.Symbol('d')
        self.e = sympy.Symbol('e')
        self.f = sympy.Symbol('f')
        self.g = sympy.Symbol('g')
        self.h = sympy.Symbol('h')
        self.i = sympy.Symbol('i')
        self.j = sympy.Symbol('j')

    def test_logical_and(self):
        expr = Expression(sympy.And(self.a, self.b))
        self.assertEqual(expr.rhs_cstr, 'a && b')

    def test_logical_or(self):
        expr = Expression(sympy.Or(self.a, self.b))
        self.assertEqual(expr.rhs_cstr, 'a || b')

    def test_equality(self):
        expr = Expression(sympy.Eq(self.a, self.b))
        self.assertEqual(expr.rhs_cstr, 'a == b')

    def test_equality_combined(self):
        expr = Expression(sympy.Or(sympy.And(sympy.Eq(self.a, self.b),
                                             sympy.Eq(self.c, self.d)),
                                   sympy.Eq(self.e, self.f)))
        self.assertEqual(expr.rhs_cstr, 'a == b && c == d || e == f')

    def test_nested_relational(self):
        expr = Expression(sympy.And(sympy.Or(sympy.Eq(self.a, self.b),
                                             sympy.Eq(self.c, self.d)),
                                    sympy.Or(sympy.Eq(self.e, self.f),
                                             sympy.Lt(self.g, self.f))))
        self.assertEqual(
            expr.rhs_cstr, '(a == b || c == d) && (e == f || g < f)')

    def test_pow(self):
        expr = Expression(self.a ** self.b)
        self.assertEqual(expr.rhs_cstr, 'pow(a, b)')

    def test_negation(self):
        expr = Expression(sympy.Not(self.a))
        self.assertEqual(expr.rhs_cstr, '!a')

    def test_ternary_simple(self):
        expr = Expression(Piecewise((self.c, sympy.Lt(self.a, self.b)),
                                    (self.d, True)))
        self.assertEqual(expr.rhs_cstr, '((a < b) ? (\n   c\n)\n: (\n   d\n))')

    def test_ternary_nested(self):
        expr = Expression(Piecewise((self.c, sympy.Lt(self.a, self.b)),
                                    (self.f, sympy.Gt(self.d, self.e)),
                                    (self.i, sympy.Eq(self.g, self.h)),
                                    (self.j, True)))
        self.assertEqual(
            expr.rhs_cstr, ('((a < b) ? (\n   c\n)\n: ((d > e) ? (\n   '
                            'f\n)\n: ((g == h) ? (\n   i\n)\n: (\n   j\n))))'))


    def test_ternary_bad(self):
        self.assertRaises(NineMLMathParseError,
                          Expression, 'a < b ? (c ? d : e) : f')

    def test_ternary_nested(self):
        expr = Expression('a < b ? c : d > e ? f : g == h ? i : j')
        self.assertEqual(
            expr.rhs, Piecewise((self.c, sympy.Lt(self.a, self.b)),
                                (self.f, sympy.Gt(self.d, self.e)),
                                (self.i, sympy.Eq(self.g, self.h)),
                                (self.j, True)))


class Rationals_test(unittest.TestCase):

    def test_xml(self):
        "Tests conversion of rationals back from the c-code version 1.0L/2.0L"
        expr = Expression('1/2')
        self.assertEqual(expr.rhs_xml, '1/2')

    def test_c89(self):
        "Tests conversion of rationals back from the c-code version 1.0L/2.0L"
        expr = Expression('1/2')
        self.assertEqual(expr.rhs_cstr, '1.0L/2.0L')


class C89_test(unittest.TestCase):

    def test_logical_and(self):
        "Tests conversion of rationals back from the c-code version 1.0L/2.0L"
        expr = Expression('1/2')
        self.assertEqual(str(expr.rhs), '1/2')


class TestVisitor(object):

    def visit(self, obj, **kwargs):
        return obj.accept_visitor(self, **kwargs)


# Testing Skeleton for class: ExpressionWithSimpleLHS
class ExpressionWithSimpleLHS_test(unittest.TestCase):

    def test_lhs(self):

        e = ExpressionWithSimpleLHS('a', 't+t+3 + sin(t) +q')

        self.assertEqual(list(e.lhs), ['a'])
        self.assertEqual(list(e.lhs_atoms), ['a'])
        self.assertEqual(sorted(list(e.rhs_atoms)), sorted(['t', 'sin', 'q']))
        self.assertEqual(sorted(list(e.atoms)), sorted(['a', 't', 'sin', 'q']))

        # RHS transform not affecting LHS:
        e.rhs_name_transform_inplace({'a': 'b'})
        self.assertEqual(sorted(list(e.atoms)), sorted(['a', 't', 'sin', 'q']))

        # LHS transform not affecting RHS:
        e.lhs_name_transform_inplace({'t': 'T'})
        self.assertEqual(sorted(list(e.atoms)), sorted(['a', 't', 'sin', 'q']))

        # name_transform affecting LHS & RHS:
        e.name_transform_inplace({'t': 'T'})
        self.assertEqual(sorted(list(e.atoms)), sorted(['a', 'T', 'sin', 'q']))
        self.assertEqual(sorted(list(e.rhs_atoms)), sorted(['T', 'sin', 'q']))
        self.assertEqual(sorted(list(e.lhs_atoms)), sorted(['a']))

        e.name_transform_inplace({'a': 'A'})
        self.assertEqual(sorted(list(e.atoms)), sorted(['A', 'T', 'sin', 'q']))
        self.assertEqual(sorted(list(e.rhs_atoms)), sorted(['T', 'sin', 'q']))
        self.assertEqual(sorted(list(e.lhs_atoms)), sorted(['A']))


class Alias_test(unittest.TestCase):

    def test_accept_visitor(self):
        # Signature: name(self, visitor, **kwargs)
                # |VISITATION|

        class AliasTestVisitor(TestVisitor):
            def visit_alias(self, component, **kwargs):  # @UnusedVariable
                return kwargs

        c = Alias(lhs='V', rhs='0')
        v = AliasTestVisitor()

        self.assertEqual(
            v.visit(c, kwarg1='Hello', kwarg2='Hello2'),
            {'kwarg1': 'Hello', 'kwarg2': 'Hello2'}
        )


class StateAssignment_test(unittest.TestCase):

    def test_accept_visitor(self):
        # Signature: name(self, visitor, **kwargs)
                # |VISITATION|

        class StateAssignmentTestVisitor(TestVisitor):

            def visit_stateassignment(self, component, **kwargs):  # @UnusedVariable @IgnorePep8
                return kwargs

        c = StateAssignment(lhs='V', rhs='0')
        v = StateAssignmentTestVisitor()

        self.assertEqual(
            v.visit(c, kwarg1='Hello', kwarg2='Hello2'),
            {'kwarg1': 'Hello', 'kwarg2': 'Hello2'}
        )


# Testing Skeleton for class: TimeDerivative
class TimeDerivative_test(unittest.TestCase):

    def test_accept_visitor(self):
        # Signature: name(self, visitor, **kwargs)
                # |VISITATION|

        class TimeDerivativeTestVisitor(TestVisitor):

            def visit_timederivative(self, component, **kwargs):  # @UnusedVariable @IgnorePep8
                return kwargs

        c = TimeDerivative(variable='V', rhs='0')
        v = TimeDerivativeTestVisitor()

        self.assertEqual(
            v.visit(c, kwarg1='Hello', kwarg2='Hello2'),
            {'kwarg1': 'Hello', 'kwarg2': 'Hello2'}
        )

    def test_atoms(self):
        td = TimeDerivative(variable='X',
                            rhs=' y * f - sin(q*q) + 4 * a * exp(Y)')
        self.assertEquals(sorted(td.atoms), sorted(
            ['X', 'y', 'f', 'sin', 'exp', 'q', 'a', 'Y', 't']))
        self.assertEquals(sorted(td.lhs_atoms), sorted(['X', 't']))
        self.assertEquals(sorted(td.rhs_atoms),
                          sorted(['y', 'f', 'sin', 'exp', 'q', 'a', 'Y']))

#   def test_dependent_variable(self):
    def test_independent_variable(self):
        td = TimeDerivative(variable='X',
                            rhs=' y*f - sin(q*q) + 4*a*exp(Y)')
        self.assertEquals(td.independent_variable, 't')
        self.assertEquals(td.variable, 'X')

        # Check substitutions to the LHS:
        td.lhs_name_transform_inplace({'X': 'x'})
        self.assertEquals(td.variable, 'x')

        # Since this is always time, we should not be changing the
        # independent_variable (dt)
        td.lhs_name_transform_inplace({'t': 'T'})
        self.assertEquals(td.independent_variable, 'T')

        # Aand change them again using 'name_transform_inplace'
        # Check substitutions to the LHS:
        td.name_transform_inplace({'x': 'X1'})
        self.assertEquals(td.variable, 'X1')

        # Since this is always time, we should not be changing the
        # independent_variable (dt)
        td.lhs_name_transform_inplace({'T': 'time'})
        self.assertEquals(td.independent_variable, 'time')


class Constant_test(unittest.TestCase):

    def setUp(self):
        self.c = Constant(name="faraday", value=96485.3365, units=coulomb)

    def test_accept_visitor(self):
        # Signature: name(self, visitor, **kwargs)
                # |VISITATION|

        class ConstantTestVisitor(TestVisitor):

            def visit_constant(self, component, **kwargs):  # @UnusedVariable @IgnorePep8
                return kwargs

        v = ConstantTestVisitor()
        self.assertEqual(
            v.visit(self.c, kwarg1='Hello', kwarg2='Hello2'),
            {'kwarg1': 'Hello', 'kwarg2': 'Hello2'}
        )

    def test_xml_roundtrip(self):
        writer = XMLWriter()
        xml = self.c.accept_visitor(writer)
        loader = XMLLoader(Document(coulomb))
        c = loader.load_constant(xml)
        self.assertEqual(c, self.c, "Constant failed xml roundtrip")
