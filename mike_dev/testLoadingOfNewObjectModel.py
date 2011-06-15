


from nineml.utility import LocationMgr, Join, ExpectSingle, FilterExpectSingle
from nineml.abstraction_layer.xmlns import *

import nineml.abstraction_layer as nineml

from collections import defaultdict


LocationMgr.StdAppendToPath()



import logging

logger = logging.getLogger('nineml.xmlreader')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('/tmp/nineml_xmlreader.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
logger.addHandler(fh)
logger.addHandler(ch)






from nineml.abstraction_layer.models.core import ComponentNode

def load_ComponentClass(element):
    name = element.get("name")
    logger.info("Loading ComponentClass: %s"%name)
     
    subnodes = loadBlocks( element,blocks=('Parameter','AnalogPort','EventPort','Dynamics' ) )
    return ComponentNode(   name=name,
                                   parameters = subnodes[nineml.Parameter ] ,
                                   analog_ports = subnodes[nineml.AnalogPort] ,
                                   event_ports = subnodes[nineml.EventPort],
                                   dynamics = ExpectSingle( subnodes[nineml.Dynamics] )
                                   )
   
def load_Parameter(element):
    name = element.get("name")
    logger.info("Loading Parameter: %s"%name)
    return nineml.Parameter(name=name) 

def load_AnalogPort(element):
    name = element.get("name")
    mode = element.get("mode")
    reduce_op = element.get("reduce_op",None)
    logger.info("Loading AnalogPort: %s"%name)
    return nineml.AnalogPort( internal_symbol = name, mode = mode, op = reduce_op )

def load_EventPort(element):
    name = element.get("name")
    mode = element.get("mode")
    logger.info("Loading EventPort: %s"%name)
    return nineml.EventPort( internal_symbol = name, mode = mode )



def load_Dynamics(element):
    logger.info("Loading Dynamics") 
    subnodes = loadBlocks( element, blocks=('Regime','Alias','StateVariable' )  )
    return nineml.Dynamics(  regimes = subnodes[nineml.Regime] ,
                             aliases = subnodes[nineml.Alias] ,
                             state_variables = subnodes[nineml.StateVariable] ,
                             )


def load_RegimeClass(element):
    name = element.get("name")
    logger.info("Loading Regime: %s"%name) 
    subnodes = loadBlocks( element, blocks=('TimeDerivative','OnCondition','OnEvent' )  )
    return nineml.Regime( name=name,
                          time_derivatives = subnodes[nineml.ODE],
                          on_events = subnodes[nineml.OnEvent],
                          on_conditions = subnodes[nineml.OnCondition] )



def load_StateVariable(element):
    name = element.get("name")
    logger.info("Loading StateVariable: %s"%name) 
    return nineml.StateVariable( name=name)


def load_TimeDerivative(element):
    variable = element.get("variable")
    expr = load_SingleInternalMathsBlock(element)
    logger.info("Loading TimeDerivative: %s :-> %s"%(variable, expr)  )
    return nineml.ODE( dependent_variable=variable, indep_variable='t', rhs=expr)
    
def load_Alias(element):
    name = element.get("name")
    rhs =  load_SingleInternalMathsBlock(element)
    logger.info("Loading Alias: %s := %s"%(name, rhs)  )
    return nineml.Alias( lhs=name,  rhs=rhs)


def load_OnCondition(element):
    subnodes = loadBlocks( element, blocks=('Trigger','StateAssignment','EventOut' )  )
    logger.info("Loading OnCondition:") 

    return nineml.OnCondition(  trigger = ExpectSingle( subnodes[nineml.Trigger] ),
                                state_assignments = subnodes[ nineml.Assignment ],
                                event_outputs = subnodes[ nineml.OutputEvent ], )
                                

def load_OnEvent(element):
    logger.info("Loading OnEvent") 
    assert False, 'Not implemented yet'

def load_Trigger(element):
    return load_SingleInternalMathsBlock ( element ) 


def load_StateAssignment(element):
    logger.info("Loading State Assignement") 
    to = element.get('variable')
    expr = load_SingleInternalMathsBlock(element)
    return nineml.Assignment(to=to, expr=expr)

def load_EventOut(element):
    logger.info("Loading EventOut") 
    port = element.get('port')
    return nineml.OutputEvent(port=port)


def load_EventIn(element):
    logger.info("Loading EventIn") 
    return nineml.InputEvent( port = element.get('port') )


def load_SingleInternalMathsBlock(element, checkOnlyBlock=True):
    print element

    if checkOnlyBlock:
        elements = list(element.iterchildren(tag=etree.Element ) ) 
        if  len( elements ) != 1:
            print elements
            assert False, 'Unexpected tags found'

    assert len( element.findall(NS+"MathML") ) == 0
    assert len( element.findall(NS+"MathInline") ) == 1
    
    return ExpectSingle( element.findall(NS+'MathInline') ).text



tag_to_class_dict = {
    "ComponentClass" : nineml.ComponentClass,
    "Dynamics" : nineml.Dynamics,
    "Regime" : nineml.Regime,
    "StateVariable" : nineml.StateVariable,
    "Parameter": nineml.Parameter,
    "EventPort": nineml.EventPort,
    "AnalogPort": nineml.AnalogPort,
    "Dynamics": nineml.Dynamics,
    "OnCondition": nineml.OnCondition,
    "OnEvent": nineml.OnEvent,
    "Alias": nineml.Alias,
    "TimeDerivative": nineml.ODE,
    "Trigger": nineml.Trigger,
    "StateAssignment": nineml.Assignment,
    "EventOut": nineml.OutputEvent,
    "EventIn": nineml.InputEvent,

        }

tag_to_loader = {
    "ComponentClass" : load_ComponentClass,
    "Dynamics" : load_Dynamics,
    "Regime" : load_RegimeClass,
    "StateVariable" : load_StateVariable,
    "Parameter": load_Parameter,
    "EventPort": load_EventPort,
    "AnalogPort": load_AnalogPort,
    "Dynamics": load_Dynamics,
    "OnCondition": load_OnCondition,
    "OnEvent": load_OnEvent,
    "Alias": load_Alias,
    "TimeDerivative": load_TimeDerivative,
    "Trigger": load_Trigger,
    "StateAssignment": load_StateAssignment,
    "EventOut": load_EventOut,
    "EventIn": load_EventIn,
        }

NS = "{CoModL}"


# These blocks map directly in to classes:
def loadBlocks( element, blocks=None, checkForSpuriousBlocks=True ):
    """ 
    Creates a dictionary that maps class-types to instantiated objects
    """

    if blocks:
        for t in element.iterchildren(tag=etree.Element):
            assert t.tag[len(NS):] in blocks  or t.tag in blocks

    
    blocks = blocks if blocks is not None else tag_to_class_dict.keys()

    res = defaultdict( list )
    for blk in blocks:
        cls = tag_to_class_dict[blk]
        loader = tag_to_loader[blk]
        res[cls].extend( [ loader( e ) for e in element.findall(NS+blk) ] )

    return dict( res.iteritems() )



def new_parse( filename):
    doc = etree.parse(filename)
    root = doc.getroot()
    assert root.nsmap[None] == nineml_namespace

    component = root.find(NS + "ComponentClass")
    return load_ComponentClass( component )


sample_xml_dir = Join( LocationMgr.getCatalogDir(), "sample_xml_files")
component = new_parse(  Join( sample_xml_dir, 'PostTF_izhikevich.xml' ) )

#from nineml.abstraction_layer.models import dump_reduced, flags
import pyNN.neuron.nineml as pyNNml



component.write("/tmp/nineml_toxml1.xml")
import sys
sys.exit(0)

celltype_cls = pyNNml.nineml_celltype_from_model(
                        name = "iaf_2coba",
                        nineml_model = component,
                        synapse_components = [
                            pyNNml.CoBaSyn( namespace='cobaExcit',  weight_connector='q' ),
                            pyNNml.CoBaSyn( namespace='cobaInhib',  weight_connector='q' ),
                                   ]
                        )


