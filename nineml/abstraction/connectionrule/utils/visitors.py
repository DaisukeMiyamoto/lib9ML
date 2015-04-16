"""
docstring needed

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""


from ...componentclass.utils import (
    ComponentActionVisitor, ComponentElementFinder)
from ...componentclass.utils.visitors import ComponentRequiredDefinitions


class ConnectionRuleActionVisitor(ComponentActionVisitor):

    def visit_componentclass(self, component_class, **kwargs):
        super(ConnectionRuleActionVisitor, self).visit_componentclass(
            component_class, **kwargs)


class ConnectionRuleRequiredDefinitions(ComponentRequiredDefinitions,
                                        ConnectionRuleActionVisitor):

    def __init__(self, component_class, expressions):
        ConnectionRuleActionVisitor.__init__(self,
                                             require_explicit_overrides=False)
        ComponentRequiredDefinitions.__init__(self, component_class,
                                              expressions)


class ConnectionRuleElementFinder(ComponentElementFinder,
                                  ConnectionRuleActionVisitor):

    def __init__(self, element):
        ConnectionRuleActionVisitor.__init__(self,
                                             require_explicit_overrides=True)
        ComponentElementFinder.__init__(self, element)
