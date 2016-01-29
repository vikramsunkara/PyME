"""
A solution-support based domain expansion routine for the FSP algorithm.
"""

import pyme.domain
import pyme.FSP.util
import pyme.lexarrayset 

class SupportExpander(object):
    """
    An FSP expander that expands states around the solution support.
    
    The domain is expanded to include all states reachable using the
    given transitions, up to the given depth, from the states that
    are contained in the support of a compressed epsilon approximation
    of the current solution.
    """
    def __init__(self, transitions, depth, epsilon):
        """
        An FSP expander that expands states around the solution support.
        
        The domain is expanded to include all states reachable using the
        given transitions, up to the given depth, from the states that
        are contained in the support of a compressed epsilon approximation
        of the current solution.
        """
        self.transitions = transitions
        self.depth = depth
        self.epsilon = epsilon
    	print("starting class")
    	self.Gtau = 0.001
    	self.Gtau_default = 0.01
    	
    def expand(self, **kwargs):
        """
        Returns expanded domain states
        """
        p = kwargs['p']
        support = pyme.domain.from_iter(p.compress(self.epsilon))
        expanded_support = pyme.FSP.util.grow_domain(
            support,
            self.transitions,
            self.depth
        )
        domain_states = kwargs['domain_states']
        #return cmepy.lexarrayset.union(domain_states, expanded_support)
	return pyme.lexarrayset.union(domain_states,expanded_support), expanded_support, 0
