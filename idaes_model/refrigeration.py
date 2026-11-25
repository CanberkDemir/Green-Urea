"""Steady-state R717 vapor-compression utility with fixed operating temperatures.

CoolProp supplies real-fluid state properties. The refrigerant circulation is
solved simultaneously with the process cooling duty in Pyomo.
"""
from CoolProp.CoolProp import PropsSI
from pyomo.environ import Block, Constraint, Expression, Param, Var, units as u


def add_refrigeration(fs, cooling_duty, evaporating_temperature=278.15,
                      condensing_temperature=308.15, superheat=5.0,
                      subcooling=3.0, isentropic_efficiency=0.75,
                      motor_efficiency=0.95):
    if not 0 < isentropic_efficiency <= 1 or not 0 < motor_efficiency <= 1:
        raise ValueError("Compressor efficiencies must be in (0, 1].")
    if condensing_temperature <= evaporating_temperature:
        raise ValueError("Condensing temperature must exceed evaporating temperature.")
    if superheat <= 0 or subcooling <= 0:
        raise ValueError("Superheat and subcooling must be positive.")
    fluid = "Ammonia"
    pe = PropsSI("P", "T", evaporating_temperature, "Q", 1, fluid)
    pc = PropsSI("P", "T", condensing_temperature, "Q", 0, fluid)
    h1 = PropsSI("H", "P", pe, "T", evaporating_temperature + superheat, fluid)
    s1 = PropsSI("S", "P", pe, "T", evaporating_temperature + superheat, fluid)
    h2s = PropsSI("H", "P", pc, "S", s1, fluid)
    h2 = h1 + (h2s - h1) / isentropic_efficiency
    h3 = PropsSI("H", "P", pc, "T", condensing_temperature - subcooling, fluid)
    # Expansion valve conserves specific enthalpy.
    h4 = h3
    if h1 <= h4 or h2 <= h1:
        raise ValueError("Specified states cannot deliver positive refrigeration and work.")
    fs.refrigeration = b = Block()
    b.low_pressure = Param(initialize=pe, units=u.Pa)
    b.high_pressure = Param(initialize=pc, units=u.Pa)
    b.enthalpy = Param([1, 2, 3, 4], initialize={1: h1, 2: h2, 3: h3, 4: h4},
                       units=u.J / u.kg)
    b.mass_flow = Var(initialize=0.06, bounds=(0, None), units=u.kg / u.s)
    b.evaporator_duty = Expression(expr=b.mass_flow * (b.enthalpy[1] - b.enthalpy[4]))
    b.compressor_work = Expression(expr=b.mass_flow * (b.enthalpy[2] - b.enthalpy[1]))
    b.electric_power = Expression(expr=b.compressor_work / motor_efficiency)
    b.condenser_duty = Expression(expr=b.mass_flow * (b.enthalpy[2] - b.enthalpy[3]))
    b.motor_loss = Expression(expr=b.electric_power - b.compressor_work)
    b.cop = Param(initialize=(h1 - h4) * motor_efficiency / (h2 - h1))
    b.cooling_balance = Constraint(expr=b.evaporator_duty == cooling_duty)
    return b
