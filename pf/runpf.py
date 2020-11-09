# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Solves the power flow using a full Newton's method.
"""

import sys

from numpy import array, angle, exp, linalg, conj, r_, Inf

from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve

from pf.dSbus_dV import dSbus_dV
from pf.ppoption import ppoption


def newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt=None):
    """Solves the power flow using a full Newton's method.

    Ybus: the full system admittance matrix (forall buses),
    Sbus: the complex bus power injection vector (for all buses),
    V0:   the initial vector of complex bus voltages,
    ref:  column vectors with the lists of bus indices for the swing bus,
    pv:   column vectors with the lists of bus indices for PV buses
    pq:   column vectors with the lists of bus indices for PQ buses,
    """
    ## default arguments
    ## options
    tol     = ppopt['PF_TOL']
    max_it  = ppopt['PF_MAX_IT']
    verbose = ppopt['VERBOSE']

    ## initialize
    converged = 0
    i = 0
    V = V0
    Va = angle(V)
    Vm = abs(V)

    ## set up indexing for updating V
    pvpq = r_[pv, pq]
    npv = len(pv)
    npq = len(pq)
    j1 = 0;         j2 = npv           ## j1:j2 - V angle of pv buses
    j3 = j2;        j4 = j2 + npq      ## j3:j4 - V angle of pq buses
    j5 = j4;        j6 = j4 + npq      ## j5:j6 - V mag of pq buses

    ## evaluate F(x0)
    mis = V * conj(Ybus * V) - Sbus
    F = r_[  mis[pv].real,
             mis[pq].real,
             mis[pq].imag  ]

    ## check tolerance
    normF = linalg.norm(F, Inf)
    if verbose > 1:
        sys.stdout.write('\n it    max P & Q mismatch (p.u.)')
        sys.stdout.write('\n----  ---------------------------')
        sys.stdout.write('\n%3d        %10.3e' % (i, normF))
    if normF < tol:
        converged = 1
        if verbose > 1:
            sys.stdout.write('\nConverged!\n')

    ## do Newton iterations
    while (not converged and i < max_it):
        ## update iteration counter
        i = i + 1

        ## evaluate Jacobian
        dS_dVm, dS_dVa = dSbus_dV(Ybus, V)

        J11 = dS_dVa[array([pvpq]).T, pvpq].real
        J12 = dS_dVm[array([pvpq]).T, pq].real
        J21 = dS_dVa[array([pq]).T, pvpq].imag
        J22 = dS_dVm[array([pq]).T, pq].imag

        J = vstack([
                hstack([J11, J12]),
                hstack([J21, J22])
            ], format="csr")

        ## compute update step
        dx = -1 * spsolve(J, F)

        ## update voltage
        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        V = Vm * exp(1j * Va)
        Vm = abs(V)            ## update Vm and Va again in case
        Va = angle(V)          ## we wrapped around with a negative Vm

        ## evalute F(x)
        mis = V * conj(Ybus * V) - Sbus
        F = r_[  mis[pv].real,
                 mis[pq].real,
                 mis[pq].imag  ]

        ## check for convergence
        normF = linalg.norm(F, Inf)
        if verbose > 1:
            sys.stdout.write('\n%3d        %10.3e' % (i, normF))
        if normF < tol:
            converged = 1
            if verbose:
                sys.stdout.write("\nNewton's method power flow converged in "
                                 "%d iterations.\n" % i)

    if verbose:
        if not converged:
            sys.stdout.write("\nNewton's method power did not converge in %d "
                             "iterations.\n" % i)

    return V, converged, i


if __name__ == "__main__":
    from cases.case14 import case14
    from pf.makeY import makeY
    caseData = case14()
    Y = makeY(caseData['bus'], caseData['branch'])
    print(123)
