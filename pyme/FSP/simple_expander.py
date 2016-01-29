"""
A simple domain expansion routine for the FSP algorithm.
"""

import pyme.FSP.util

class SimpleExpander(object):
    """
    Simple FSP expander that expands the entire domain.
    
    The domain is expanded along the given transitions, up to the specified
    depth.
    """
    def __init__(self, transitions, depth, validity_test=False):
        """
        Simple FSP expander that expands the entire domain.
    
        The domain is expanded along the given transitions, up to the specified
        depth.
        """
        self.transitions = transitions
        self.depth = depth
        self.Gtau = 0.001
        self.Gtau_default = 0.001
        self.name='SE1'
        self.valid_func = validity_test
    
    def expand(self, **kwargs):
        """
        Returns expanded domain states
        """
        return pyme.FSP.util.grow_domain(
            kwargs['domain_states'],
            self.transitions,
            self.depth,
            validity_test = self.valid_func
        )
