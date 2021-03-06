"""
This file contains utility classes for modifying components.

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""

from .cloner import RandomDistributionExpandPortDefinition
from .base import RandomDistributionActionVisitor
from ...componentclass.visitors.modifiers import (
    ComponentModifier, ComponentRenameSymbol, ComponentAssignIndices)


class RandomDistributionModifier(ComponentModifier):

    """Utility classes for modifying components"""

    _ExpandPortDefinition = RandomDistributionExpandPortDefinition


class RandomDistributionRenameSymbol(RandomDistributionActionVisitor,
                               ComponentRenameSymbol):

    """ Can be used for:
    Aliases
    """
    pass


class RandomDistributionAssignIndices(ComponentAssignIndices,
                                   RandomDistributionActionVisitor):
    pass
