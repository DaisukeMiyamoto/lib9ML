from __future__ import division
import re
import operator
from sympy import Symbol
from nineml.xmlns import E
from nineml import BaseNineMLObject, DocumentLevelObject
from nineml.annotations import annotate_xml, read_annotations


class Dimension(BaseNineMLObject, DocumentLevelObject):
    """
    Defines the dimension used for quantity units
    """

    element_name = 'Dimension'
    dimension_names = ('m', 'l', 't', 'i', 'n', 'k', 'j')
    SI_units = ('Kg', 'm', 's', 'A', 'mol', 'K', 'cd')
    defining_attributes = ('_dims',)
    _trailing_numbers_re = re.compile(r'(.*)(\d+)$')

    def __init__(self, name, dimensions=None, **kwargs):
        BaseNineMLObject.__init__(self)
        DocumentLevelObject.__init__(self, kwargs.pop('url', None))
        self._name = name
        if dimensions is not None:
            assert len(dimensions) == 7, "Incorrect dimension length"
            self._dims = tuple(dimensions)
        else:
            self._dims = tuple(kwargs.pop(d, 0) for d in self.dimension_names)
        assert not len(kwargs), "Unrecognised kwargs ({})".format(kwargs)

    def __hash__(self):
        return hash(self._dims)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return ("Dimension(name='{}'{})".format(
            self.name, ''.join(' {}={}'.format(n, p) if p != 0 else ''
                               for n, p in zip(self.dimension_names,
                                               self._dims))))

    def __iter__(self):
        return iter(self._dims)

    @property
    def name(self):
        return self._name

    def power(self, name):
        return self._dims[self.dimension_names.index(name)]

    def to_SI_units_str(self):
        numer = '*'.join('({}**{})'.format(si, p) if p > 1 else si
                         for si, p in zip(self.SI_units, self._dims) if p > 0)
        denom = '*'.join('({}**{})'.format(si, p) if p < -1 else si
                         for si, p in zip(self.SI_units, self._dims) if p < 0)
        return '{}/({})'.format(numer, denom)

    def _sympy_(self):
        """
        Create a sympy expression by multiplying symbols representing each of
        the dimensions together
        """
        return reduce(
            operator.mul,
            (Symbol(n) ** p for n, p in zip(self.dimension_names, self._dims)))

    @property
    def m(self):
        return self._dims[0]

    @property
    def l(self):
        return self._dims[1]

    @property
    def t(self):
        return self._dims[2]

    @property
    def i(self):
        return self._dims[3]

    @property
    def n(self):
        return self._dims[4]

    @property
    def k(self):
        return self._dims[5]

    @property
    def j(self):
        return self._dims[6]

    @property
    def time(self):
        return self.t

    @property
    def temperature(self):
        return self.k

    @property
    def luminous_intensity(self):
        return self.j

    @property
    def amount(self):
        return self.n

    @property
    def mass(self):
        return self.m

    @property
    def length(self):
        return self.l

    @property
    def current(self):
        return self.i

    @annotate_xml
    def to_xml(self, **kwargs):  # @UnusedVariable
        kwargs = {'name': self.name}
        kwargs.update(dict(
            (n, str(p))
            for n, p in zip(self.dimension_names, self._dims) if abs(p) > 0))
        return E(self.element_name, **kwargs)

    @classmethod
    @read_annotations
    def from_xml(cls, element, document):
        kwargs = dict(element.attrib)
        name = kwargs.pop('name')
        kwargs = dict((k, int(v)) for k, v in kwargs.items())
        kwargs['url'] = document.url
        return cls(name, **kwargs)

    def __mul__(self, other):
        "self * other"
        if other == 1:
            other = dimensionless
        return Dimension(self.make_name([self.name, other.name]),
                         dimensions=tuple(s + o for s, o in zip(self, other)))

    def __truediv__(self, other):
        "self / expr"
        if other == 1:
            other = dimensionless
        return Dimension(self.make_name([self.name], [other.name]),
                         dimensions=tuple(s - o for s, o in zip(self, other)))

    def __pow__(self, power):
        "self ** expr"
        return Dimension(self.make_name([self.name], power=power),
                         dimensions=tuple(s * power for s in self))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if other == 1:
            other = dimensionless
        return other.__truediv__(self)

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    @classmethod
    def make_name(cls, products=[], divisors=[], power=1):
        """
        Generates a sensible name from the combination of dimensions/units.

        E.g.

        voltage * current -> voltage_current
        voltage / current -> voltage_per_current
        voltage ** 2 -> voltage2
        dimensionless / voltage -> per_voltage
        (voltage * current * dimensionless) ** 3 -> current3_voltage3
        """
        if power == 0:
            return 'dimensionless'
        numerator = []
        denominator = []
        for p in products:
            p_split = p.split('_per_')
            numerator.extend(p_split[0].split('_'))
            if len(p_split) > 1:
                denominator.extend(p_split[1].split('_'))
        for d in divisors:
            d_split = d.split('_per_')
            denominator.extend(d_split[0].split('_'))
            if len(d_split) > 1:
                numerator.extend(d_split[1].split('_'))
        num_denoms = []
        for lst in ((numerator, denominator)
                    if power > 0 else (denominator, numerator)):
            new_lst = []
            for dim_name in lst:
                if dim_name not in ('dimensionless', 'unitless', 'none'):
                    if power != 1:
                        assert isinstance(power, int)
                        match = cls._trailing_numbers_re.match(dim_name)
                        if match:
                            name = match.group(1)
                            pw = int(match.group(2)) * power
                        else:
                            name = dim_name
                            pw = power
                        dim_name = name + str(abs(pw))
                    new_lst.append(dim_name)
            num_denoms.append(new_lst)
        numerator, denominator = num_denoms
        name = '_'.join(sorted(numerator))
        if len(denominator):
            if len(numerator):
                name += '_'
            name += 'per_' + '_'.join(sorted(denominator))
        return name


class Unit(BaseNineMLObject, DocumentLevelObject):
    """
    Defines the units of a quantity
    """

    element_name = 'Unit'
    defining_attributes = ('_dimension', '_power', '_offset')

    def __init__(self, name, dimension, power, offset=0.0, url=None):
        BaseNineMLObject.__init__(self)
        DocumentLevelObject.__init__(self, url)
        self._name = name
        self._dimension = dimension
        self._power = power
        self._offset = offset

    def __hash__(self):
        return hash((self.power, self.offset, self.dimension))

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return ("Unit(name='{}', dimension='{}', power={}{})"
                .format(self.name, self.dimension.name, self.power,
                        (", offset='{}'".format(self.offset)
                         if self.offset else '')))

    def to_SI_units_str(self):
        if self.offset != 0.0:
            raise Exception("Cannot convert to SI units string as offset is "
                            "not zero ({})".format(self.offset))
        return (self.dimension.to_SI_units_str() +
                ' * 10**({})'.format(self.power) if self.power else '')

    def _sympy_(self):
        """
        Create a sympy expression by multiplying symbols representing each of
        the dimensions together
        """
        return self.dimension._sympy_() * 10 ** self.power + self.offset

    @property
    def name(self):
        return self._name

    @property
    def dimension(self):
        return self._dimension

    def set_dimension(self, dimension):
        """
        Used to standardize dimension names across a NineML document. The
        actual dimension (in terms of fundamental dimension powers) should
        not change.
        """
        assert self.dimension == dimension, "dimensions do not match"
        self._dimension = dimension

    @property
    def power(self):
        return self._power

    @property
    def offset(self):
        return self._offset

    @property
    def symbol(self):
        return self.name

    @annotate_xml
    def to_xml(self, **kwargs):  # @UnusedVariable
        kwargs = {'symbol': self.name, 'dimension': self.dimension.name,
                  'power': str(self.power)}
        if self.offset:
            kwargs['offset'] = str(self.offset)
        return E(self.element_name,
                 **kwargs)

    @classmethod
    @read_annotations
    def from_xml(cls, element, document):
        name = element.attrib['symbol']
        dimension = document[element.attrib['dimension']]
        power = int(element.get('power', 0))
        offset = float(element.attrib.get('name', 0.0))
        return cls(name, dimension, power, offset=offset, url=document.url)

    def __mul__(self, other):
        "self * other"
        if other == 1:
            other = unitless
        assert (self.offset == 0 and
                other.offset == 0), "Can't multiply units with nonzero offsets"
        return Unit(Dimension.make_name([self.name, other.name]),
                    dimension=self.dimension * other.dimension,
                    power=(self.power + other.power))

    def __truediv__(self, other):
        "self / expr"
        if other == 1:
            other = unitless
        assert (self.offset == 0 and
                other.offset == 0), "Can't divide units with nonzero offsets"
        return Unit(Dimension.make_name([self.name], [other.name]),
                    dimension=self.dimension / other.dimension,
                    power=(self.power - other.power))

    def __pow__(self, power):
        "self ** expr"
        assert self.offset == 0, "Can't raise units with nonzero offsets"
        return Unit(Dimension.make_name([self.name], power=power),
                    dimension=(self.dimension ** power),
                    power=(self.power * power))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if other == 1:
            other = unitless
        return other.__truediv__(self)

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)


# ----------------- #
# Common dimensions #
# ----------------- #

time = Dimension(name="time", t=1)
per_time = Dimension(name="per_time", t=-1)
voltage = Dimension(name="voltage", m=1, l=2, t=-3, i=-1)
per_voltage = Dimension(name="per_voltage", m=-1, l=-2, t=3, i=1)
conductance = Dimension(name="conductance", m=-1, l=-2, t=3, i=2)
conductanceDensity = Dimension(name="conductanceDensity", m=-1, l=-4, t=3, i=2)
capacitance = Dimension(name="capacitance", m=-1, l=-2, t=4, i=2)
specificCapacitance = Dimension(name="specificCapacitance", m=-1, l=-4, t=4,
                                i=2)
resistance = Dimension(name="resistance", m=1, l=2, t=-3, i=-2)
resistivity = Dimension(name="resistivity", m=2, l=2, t=-3, i=-2)
charge = Dimension(name="charge", i=1, t=1)
charge_per_mole = Dimension(name="charge_per_mole", i=1, t=1, n=-1)
charge_density = Dimension(name="charge_per_mole", i=1, t=1, m=-3)
mass_per_charge = Dimension(name="mass_per_charge", i=-1, t=-1)
current = Dimension(name="current", i=1)
currentDensity = Dimension(name="currentDensity", i=1, l=-2)
current_per_time = Dimension(name="current", i=1, t=-1)
length = Dimension(name="length", l=1)
area = Dimension(name="area", l=2)
volume = Dimension(name="volume", l=3)
concentration = Dimension(name="concentration", l=-3, n=1)
per_time_per_concentration = Dimension(name="concentration", l=3, n=-1, t=-1)
substance = Dimension(name="substance", n=1)
flux = Dimension(name="flux", m=1, l=-3, t=-1)
substance_per_area = Dimension(name="substance", n=1, l=-2)
permeability = Dimension(name="permeability", l=1, t=-1)
temperature = Dimension(name="temperature", k=1)
idealGasConstantDims = Dimension(name="idealGasConstantDims", m=1, l=2, t=-2,
                                 k=-1, n=-1)
rho_factor = Dimension(name="rho_factor", l=-1, n=1, i=-1, t=-1)
dimensionless = Dimension(name="dimensionless")
energy_per_temperature = Dimension(name="energy_per_temperature", m=1, l=2,
                                   t=-2, k=-1)
luminous_intensity = Dimension(name="luminous_intensity", j=1)
voltage_per_time = voltage / time

# ------------ #
# Common units #
# ------------ #

s = Unit(name="s", dimension=time, power=0)
per_s = Unit(name="per_s", dimension=per_time, power=0)
Hz = Unit(name="Hz", dimension=per_time, power=0)
ms = Unit(name="ms", dimension=time, power=-3)
per_ms = Unit(name="per_ms", dimension=per_time, power=3)
m = Unit(name="m", dimension=length, power=0)
cm = Unit(name="cm", dimension=length, power=-2)
um = Unit(name="um", dimension=length, power=-6)
m2 = Unit(name="m2", dimension=area, power=0)
cm2 = Unit(name="cm2", dimension=area, power=-4)
um2 = Unit(name="um2", dimension=area, power=-12)
m3 = Unit(name="m3", dimension=volume, power=0)
cm3 = Unit(name="cm3", dimension=volume, power=-6)
litre = Unit(name="litre", dimension=volume, power=-3)
um3 = Unit(name="um3", dimension=volume, power=-18)
V = Unit(name="V", dimension=voltage, power=0)
mV = Unit(name="mV", dimension=voltage, power=-3)
per_V = Unit(name="per_V", dimension=per_voltage, power=0)
per_mV = Unit(name="per_mV", dimension=per_voltage, power=3)
mV_per_ms = Unit(name="mV_per_ms", dimension=voltage_per_time, power=0)
ohm = Unit(name="ohm", dimension=resistance, power=0)
kohm = Unit(name="kohm", dimension=resistance, power=3)
Mohm = Unit(name="Mohm", dimension=resistance, power=6)
S = Unit(name="S", dimension=conductance, power=0)
mS = Unit(name="mS", dimension=conductance, power=-3)
uS = Unit(name="uS", dimension=conductance, power=-6)
nS = Unit(name="nS", dimension=conductance, power=-9)
pS = Unit(name="pS", dimension=conductance, power=-12)
S_per_m2 = Unit(name="S_per_m2", dimension=conductanceDensity, power=0)
mS_per_cm2 = Unit(name="mS_per_cm2", dimension=conductanceDensity, power=1)
S_per_cm2 = Unit(name="S_per_cm2", dimension=conductanceDensity, power=4)
F = Unit(name="F", dimension=capacitance, power=0)
uF = Unit(name="uF", dimension=capacitance, power=-6)
nF = Unit(name="nF", dimension=capacitance, power=-9)
pF = Unit(name="pF", dimension=capacitance, power=-12)
F_per_m2 = Unit(name="F_per_m2", dimension=specificCapacitance, power=0)
uF_per_cm2 = Unit(name="uF_per_cm2", dimension=specificCapacitance, power=-2)
ohm_m = Unit(name="ohm_m", dimension=resistivity, power=0)
kohm_cm = Unit(name="kohm_cm", dimension=resistivity, power=1)
ohm_cm = Unit(name="ohm_cm", dimension=resistivity, power=-2)
C = Unit(name="C", dimension=charge, power=0)
C_per_mol = Unit(name="C_per_mol", dimension=charge_per_mole, power=0)
A = Unit(name="A", dimension=current, power=0)
mA = Unit(name="mA", dimension=current, power=-3)
uA = Unit(name="uA", dimension=current, power=-6)
nA = Unit(name="nA", dimension=current, power=-9)
pA = Unit(name="pA", dimension=current, power=-12)
A_per_m2 = Unit(name="A_per_m2", dimension=currentDensity, power=0)
uA_per_cm2 = Unit(name="uA_per_cm2", dimension=currentDensity, power=-2)
mA_per_cm2 = Unit(name="mA_per_cm2", dimension=currentDensity, power=1)
mol_per_m3 = Unit(name="mol_per_m3", dimension=concentration, power=0)
mol_per_cm3 = Unit(name="mol_per_cm3", dimension=concentration, power=6)
M = Unit(name="M", dimension=concentration, power=3)
mM = Unit(name="mM", dimension=concentration, power=0)
mol = Unit(name="mol", dimension=substance, power=0)
m_per_s = Unit(name="m_per_s", dimension=permeability, power=0)
cm_per_s = Unit(name="cm_per_s", dimension=permeability, power=-2)
um_per_ms = Unit(name="um_per_ms", dimension=permeability, power=-3)
cm_per_ms = Unit(name="cm_per_ms", dimension=permeability, power=1)
degC = Unit(name="degC", dimension=temperature, power=0, offset=273.15)
K = Unit(name="K", dimension=temperature, power=0)
J_per_K_per_mol = Unit(name="J_per_K_per_mol", dimension=idealGasConstantDims,
                       power=0)
J_per_K = Unit(name="J_per_K", dimension=energy_per_temperature, power=0)
mol_per_m_per_A_per_s = Unit(name="mol_per_m_per_A_per_s",
                             dimension=rho_factor, power=0)
unitless = Unit(name="unitless", dimension=dimensionless, power=0)
coulomb = Unit(name="coulomb", dimension=current_per_time, power=0)
cd = Unit(name="cd", dimension=luminous_intensity, power=0)

if __name__ == '__main__':
    print 1 / voltage
    print (current / voltage) ** -3
