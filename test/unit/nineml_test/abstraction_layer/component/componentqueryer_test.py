

import unittest
from nineml.abstraction import (Regime, On, AnalogSendPort,
                                AnalogReceivePort, NamespaceAddress, 
                                OutputEvent)
from nineml.abstraction.dynamics import Dynamics as Dynamics


class ComponentClassQueryer_test(unittest.TestCase):

    def test_event_send_receive_ports(self):
        # Signature: name(self)
                # Get the ``recv`` EventPorts
        # from nineml.abstraction.component.componentqueryer import ComponentClassQueryer

        # Check inference of output event ports:
        c = Dynamics(
            name='Comp1',
            regimes=Regime(
                transitions=[
                    On('in_ev1', do=OutputEvent('ev_port1')),
                    On('V < b', do=OutputEvent('ev_port1')),
                    On('V < c', do=OutputEvent('ev_port2')),
                ]
            ),
        )
        self.assertEquals(len(list(c.event_receive_ports)), 1)
        self.assertEquals((list(list(c.event_receive_ports))[0]).name, 'in_ev1')

        self.assertEquals(len(list(c.event_send_ports)), 2)
        self.assertEquals(set(c.event_send_port_names),
                          set(['ev_port1', 'ev_port2']))

        # Check inference of output event ports:
        c = Dynamics(
            name='Comp1',
            regimes=[
                Regime(name='r1',
                       transitions=[
                       On('V > a', do=OutputEvent('ev_port1'), to='r2'),
                       On('in_ev1', do=OutputEvent('ev_port2')),
                       ]
                       ),

                Regime(name='r2',
                       transitions=[
                       On('V > a', do=OutputEvent('ev_port2'), to='r1'),
                       On('in_ev2', do=OutputEvent('ev_port3')),
                       ]
                       )
            ]
        )
        self.assertEquals(len(list(c.event_receive_ports)), 2)
        self.assertEquals(set(c.event_receive_port_names),
                          set(['in_ev1', 'in_ev2']))

        self.assertEquals(len(list(c.event_send_ports)), 3)
        self.assertEquals(set(c.event_send_port_names),
                           set(['ev_port1', 'ev_port2', 'ev_port3']))

        # Check inference of output event ports:
        c = Dynamics(
            name='Comp1',
            regimes=[
                Regime(name='r1',
                       transitions=[
                       On('spikeinput1', do=[]),
                       On('spikeinput2', do=[OutputEvent('ev_port1'),
                          OutputEvent('ev_port2')], to='r2'),
                       ]
                       ),

                Regime(name='r2',
                       transitions=[
                       On('V > a', do=OutputEvent('ev_port2')),
                       On('spikeinput3', do=OutputEvent('ev_port3'), to='r1'),
                       ]
                       )
            ]
        )
        self.assertEquals(len(list(c.event_receive_ports)), 3)
        self.assertEquals(set(c.event_receive_port_names),
                          set(['spikeinput1', 'spikeinput2', 'spikeinput3']))

        self.assertEquals(len(list(c.event_send_ports)), 3)
        self.assertEquals(set(c.event_send_port_names),
                          set(['ev_port1', 'ev_port2', 'ev_port3']))

    def test_get_fully_qualified_port_connections(self):
        # Signature: name(self)
                # Used by the flattening code.
                #
                # This method returns a d list of tuples of the
                # the fully-qualified port connections
        # from nineml.abstraction.component.componentqueryer import ComponentClassQueryer

        # Signature: name(self)
                # Get the namespace address of this component
        d = Dynamics(
            name='D', aliases=['A:=1', 'B:=2'], analog_ports=[AnalogSendPort('A'), AnalogSendPort('B')])
        e = Dynamics(name='E', analog_ports=[AnalogReceivePort('C')])
        f = Dynamics(name='F', analog_ports=[AnalogReceivePort('D')])
        g = Dynamics(name='G', analog_ports=[AnalogReceivePort('E')])
        b = Dynamics(name='B', subnodes={
                           'd': d, 'e': e}, portconnections=[('d.A', 'e.C')])
        c = Dynamics(name='C',
                           aliases=['G:=-1'],
                           analog_ports=[AnalogSendPort('G')],
                           subnodes={'f': f, 'g': g},
                           portconnections=[('G', 'f.D')])

        a = Dynamics(name='A',
                           subnodes={'b': b, 'c': c},
                           analog_ports=[AnalogReceivePort('F')],
                           portconnections=[('b.d.A', 'F')]
                           )

        bNew = a.get_subnode('b')
        cNew = a.get_subnode('c')
        # dNew = a.get_subnode('b.d')
        # eNew = a.get_subnode('b.e')
        # fNew = a.get_subnode('c.f')
        # gNew = a.get_subnode('c.g')

        self.assertEquals(list(a.fully_qualified_port_connections),
                          [(NamespaceAddress('b.d.A'), NamespaceAddress('F'))])

        self.assertEquals(list(bNew.fully_qualified_port_connections),
                          [(NamespaceAddress('b.d.A'), NamespaceAddress('b.e.C'))])

        self.assertEquals(list(cNew.fully_qualified_port_connections),
                          [(NamespaceAddress('c.G'), NamespaceAddress('c.f.D'))])

    def test_ports(self):
        # Signature: name
                # Return an iterator over all the port (Event & Analog) in the
                # component
        # from nineml.abstraction.component.componentqueryer import ComponentClassQueryer


        c = Dynamics(
            name='Comp1',
            regimes=[
                Regime(name='r1',
                       transitions=[
                       On('spikeinput1', do=[]),
                       On('spikeinput2', do=OutputEvent('ev_port2'), to='r2'),
                       ]
                       ),

                Regime(name='r2',
                       transitions=[
                       On('V > a', do=OutputEvent('ev_port2')),
                       On('spikeinput3', do=OutputEvent('ev_port3'), to='r1'),
                       ]
                       )
            ],
            aliases=['A:=0', 'C:=0'],
            analog_ports=[AnalogSendPort('A'), AnalogReceivePort('B'), AnalogSendPort('C')]
        )

        ports = list(list(c.ports))
        port_names = [p.name for p in ports]

        self.assertEquals(len(port_names), 8)
        self.assertEquals(set(port_names),
                          set(['A', 'B', 'C', 'spikeinput1', 'spikeinput2',
                               'spikeinput3', 'ev_port2', 'ev_port3'])
                          )

    def test_regime(self):
        # Signature: name(self, name=None)
                # Find a regime in the component by name
        # from nineml.abstraction.component.componentqueryer import ComponentClassQueryer


        c = Dynamics(name='cl',
                           regimes=[
                                Regime('dX/dt=1/t',
                                       name='r1',
                                       transitions=On('X>X1', do=['X=X0'], to='r2')),
                                Regime('dX/dt=1/t',
                                       name='r2',
                                       transitions=On('X>X1', do=['X=X0'],
                                                      to='r3')),
                                Regime('dX/dt=1/t',
                                       name='r3',
                                       transitions=On('X>X1', do=['X=X0'],
                                                      to='r4')),
                                Regime('dX/dt=1/t',
                                       name='r4',
                                       transitions=On('X>X1', do=['X=X0'],
                                                      to='r1')),
                           ]
                           )
        self.assertEqual(c.regime(name='r1').name, 'r1')
        self.assertEqual(c.regime(name='r2').name, 'r2')
        self.assertEqual(c.regime(name='r3').name, 'r3')
        self.assertEqual(c.regime(name='r4').name, 'r4')

    def test_recurse_all_components(self):
        # Signature: name
                # Returns an iterator over this component and all subcomponents


        d = Dynamics(name='D')
        e = Dynamics(name='E')
        f = Dynamics(name='F')
        g = Dynamics(name='G')

        b = Dynamics(name='B')
        b.insert_subnode(namespace='d', subnode=d)
        b.insert_subnode(namespace='e', subnode=e)

        c = Dynamics(name='C')
        c.insert_subnode(namespace='f', subnode=f)
        c.insert_subnode(namespace='g', subnode=g)

        a = Dynamics(name='A')
        a.insert_subnode(namespace='b', subnode=b)
        a.insert_subnode(namespace='c', subnode=c)

        # Construction of the objects causes cloning to happen:
        # Therefore we test by looking up and checking that there
        # are the correct component names:
        bNew = a.get_subnode('b')
        cNew = a.get_subnode('c')
        dNew = a.get_subnode('b.d')
        eNew = a.get_subnode('b.e')
        fNew = a.get_subnode('c.f')
        gNew = a.get_subnode('c.g')

        self.assertEquals(
            set(a.all_components),
            set([a, bNew, cNew, dNew, eNew, fNew, gNew]))
        self.assertEquals(
            set(bNew.all_components),
            set([bNew, dNew, eNew]))
        self.assertEquals(
            set(cNew.all_components),
            set([cNew, fNew, gNew]))
        self.assertEquals(
            set(dNew.all_components),
            set([dNew]))
        self.assertEquals(
            set(eNew.all_components),
            set([eNew]))
        self.assertEquals(
            set(fNew.all_components),
            set([fNew]))
        self.assertEquals(
            set(gNew.all_components),
            set([gNew]))
