"""
Creates solvers for the Chemical Master Equation (CME).
"""

import numpy
import cme_matrix, domain, ode_solver, state_enum
import model as mdl

def create_dense_packing_functions(domain_enum, sink = True):
    if sink:
        pack = lambda (p, p_sink) : numpy.concatenate((p, [p_sink]))
        unpack = lambda y : (y[:-1], y[-1])
    else:
        pack = lambda p : p
        unpack = lambda y : y
    return (pack, unpack)

def create(model,
           sink,
           p_0=None,
           t_0=None,
           sink_0=None,
           time_dependencies=None,
           domain_states=None,
           domain_enum=None,
           validity_test= None):
    """
    Returns a solver for the Chemical Master Equation of the given model.

    Beware! This is an experimental dense solver! You *must* provide
    both domain_states and domain_enum arguments! See below...

    arguments:

        model : the CME model to solve

        sink : If sink is True, the solver will include a 'sink' state used
            to accumulate any probability that may flow outside the domain.
            This can be used to measure the error in the solution due to
            truncation of the domain. If sink is False, the solver will not
            include a 'sink' state, and probability will be artificially
            prevented from flowing outside of the domain.

        p_0 : (optional) mapping from states in the domain to probabilities,
            for the initial probability distribution. If not specified,
            and the initial state of the state space is given by the model,
            defaults to all probability concentrated at the initial state,
            otherwise, a ValueError will be raised.

        t_0 : (optional) initial time, defaults to 0.0

        sink_0 : (optional) initial sink probability, defaults to 0.0
            Only a valid argument if sink is set to True.

        time_dependencies : (optional) By default, reaction propensities are
            time independent. If specified, time_dependencies must be of the
            form { s_1 : phi_1, ..., s_n : phi_n }, where each (s_j, phi_j)
            item satisifes :

                s_j : set of reaction indices
                phi_j : phi_j(t) -> time dependent coefficient

            The propensities of the reactions with indicies contained in s_j
            will all be multiplied by the coefficient phi_j(t), at time t.
            Reactions are indexed according to the ordering of the propensities
            in the model.

            The reaction index sets s_j must be *disjoint*. It is not necessary
            for the union of the s_j to include all the reaction indices.
            If a reaction's index is not contained in any s_j then the reaction
            is treated as time-independent.

        mapping of time dependent coefficient
            functions keyed by subsets of reaction indices, with respect to the
            ordering of reactions determined by the order of the propensity
            functions inside the model. The propensities of the reactions
            with indices included in each subset are multiplied by the time
            dependent coefficient function. By default, no time dependent
            coefficient functions are specified, that is, the CME has
            time-independent propensities.

        domain_states : (required!) array of states in the domain.
            By default, generate the rectangular lattice of states defined by
            the 'shape' entry of the model. A ValueError is raised if both
            domain_states and 'shape' are unspecified.

        domain_enum : (required!) cmepy.state_enum.StateEnum instance,
            representing a bijection between the domain states and
            array indices. You can make one of these from domain_states
            via:

                domain_states = cmepy.state_enum.StateEnum(domain_states)

            This can be accessed from the returned solver object via
            the domain_enum attribute.
    """

    mdl.validate_model(model)

    if sink_0 is not None:
        if not sink:
            raise ValueError('sink_0 may not be specified if sink is False')
        sink_0 = float(sink_0)
    else:
        sink_0 = 0.0

    if domain_states is None:
        raise ValueError('domain_states must be explicitly specified')

    if domain_enum is None:
        raise ValueError('domain_enum must be explicitly specified')

    if validity_test is None:
        validity_test = cme_matrix.non_neg_states

    if t_0 is None:
        t_0 = 0.0

    # compute reaction matrices and use them to define differential equations
    gen_matrices = cme_matrix.gen_reaction_matrices(
        model,
        domain_enum,
        sink,
        validity_test
    )
    reaction_matrices = list(gen_matrices)
    dy_dt = cme_matrix.create_diff_eqs(
        reaction_matrices,
        phi = time_dependencies
    )

    # construct and initialise solver
    if sink:
        p_0 = (p_0, sink_0)
    pack, unpack = create_dense_packing_functions(domain_enum, sink)
    cme_solver = ode_solver.Solver(
        dy_dt,
        y_0 = p_0,
        t_0 = t_0
    )
    cme_solver.set_packing(
        pack,
        unpack,
        transform_dy_dt = False
    )
    cme_solver.domain_enum = domain_enum
    cme_solver.domain_states = domain_states
    return cme_solver
