""" Gradient-descent minimization with automatic learning rate selection. Interface inspired by scipy optimize and julia's Optim.jl"""
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_leaves, tree_unflatten
import numpy as np

import time
from types import SimpleNamespace
from tqdm import tqdm
import sys

def minimize(f, x0, a0=None, bounds=None, b=0.0, growth=2.0, backtrack=0.1, maxiter=50, verbose=True, callback=None):
    """ Run alternating gradient descent on f, starting with initial condition x0. 

    Args:
        f: function to optimize, f(x0) returns (loss, grad) where grad is a pytree of dl/dx for every x in x0
        x0: pytree of initial parameters
        a0: pytree of initial step size parameters
        bounds: pytree of bounds for the parameters. Either None or a pytree containing (low, high) for every element in x0
        b: float, tolerance in Armijo rule for backtracking line search
        growth: float, factor by which to increase step size
        backtrack: float, factor by which to decrease step size
        maxiter: int, max number of gradient steps
        verbose: bool, display information about minimization
        callback: function with argument x0, can be used for logging
        
    Returns:
        x: pytree of final parameters
        info: namespace containing optimization info
        
    Notes: 
        Uses backtracking line search. For every parameter in x0, freezes the other parameter and takes a single step
        Automatically tunes the step size for every parameter in x0
    """
    # initialize
    t0 = time.time()
    
    # progress bar
    nparams = len(tree_leaves(x0))
    pbar = tqdm(range(maxiter), leave=True, disable=not verbose, desc=f'minimizing over {nparams} param group{"s" if nparams > 1 else ""}')
    
    x0 = tree_map(lambda x: jnp.array(x), x0)
    if bounds is not None: bounds = tree_map(lambda x, b: jnp.array(b), x0, bounds)
    if bounds is not None: x0 = tree_map(lambda x, bnd: x.clip(bnd[0],bnd[1]), x0, bounds)
    if a0 is None: a0 = tree_map(lambda x: 1.0, x0)
    f0, g0 = f(x0)
    
    # logging
    info = {}
    info['fs'] = info.get('fs', []) + [f0.item()]
    info['a0s'] = info.get('a0s', []) + [a0]
    info['nfeval'] = info.get('nfeval', 0) + 1
    info['niter'] = 0
    
    # main loop
    nparams = len(tree_leaves(x0))
    cannot_decrease = tree_map(lambda x: False, x0)
    cannot_increase = tree_map(lambda x: False, x0)
    # for niter in (pbar := tqdm(range(maxiter), leave=True, disable=not verbose, desc=f'minimizing over {nparams} param group{"s" if nparams > 1 else ""}')):
    for niter in pbar:
        # update each param
        nfeval = 0
        accepted = []
        for param in range(nparams):
            x0, f0, g0, a0, accepted_,nfeval_ = update_param(f, f0, g0, x0, a0, b, bounds, growth, backtrack, param)
            accepted.append(accepted_)
            nfeval += 1
            if (not accepted_) and (nfeval_==0):
                cannot_decrease = tree_update(cannot_decrease, True, param)
            elif (not accepted_) and (nfeval_==1):
                cannot_increase = tree_update(cannot_increase, True, param)
            else:
                cannot_decrease = tree_update(cannot_decrease, False, param)
                cannot_increase = tree_update(cannot_increase, False, param)
        
        # logging 
        info['fs'] = info.get('fs', []) + [f0.item()]
        info['a0s'] = info.get('a0s', []) + [a0]
        info['nfeval'] = info.get('nfeval', 1) + nfeval
        info['niter'] = niter+1
        info['accepted'] = info.get('accepted', []) + [accepted]

        # update display
        pbar.set_postfix({'f': f0})
        
        # callback
        if callback is not None:
            callback(x0)
        
        # convergence check
        cannot_increase_or_decrease = tree_map(lambda a,b: a and b, cannot_increase, cannot_decrease)
        cannot_increase_or_decrease_any = tree_reduce(lambda a,b: a and b, cannot_increase_or_decrease)
        if cannot_increase_or_decrease_any:
            info['converged'] = True
            info['message'] = f'cannot find an update which decreases the loss'
            break

    info['elapsed_time'] = time.time() - t0
    if niter + 1 == maxiter: 
        info['converged'] = False
        info['message'] = f'reached maxiter={maxiter} iterations'

    # format info
    info = SimpleNamespace(**info)
    
    # final printing
    if verbose:
        # status
        # tqdm.write(f' * status: {niter+1} iterations completed, {info.message}', file=sys.stderr)
        tqdm.write(f' * status', file=sys.stderr)
        tqdm.write(f'   {niter+1} iterations completed, {info.message}', file=sys.stderr)
        
        tqdm.write(f'', file=sys.stderr)
        if niter >= 0:
            # objective info
            tqdm.write(f' * objective info', file=sys.stderr)
            tqdm.write(f"   f0={info.fs[0]:.3e}", file=sys.stderr)
            tqdm.write(f"   f1={info.fs[-1]:.3e}", file=sys.stderr)
            tqdm.write(f'', file=sys.stderr)

            # parameter info
            tqdm.write(f' * parameter info', file=sys.stderr)
            if bounds is not None:
                at_min = tree_map(lambda x, g, b: ((x==b[0]) & (g > 0)).astype(x.dtype), x0, g0, bounds)
                at_max = tree_map(lambda x, g, b: ((x==b[1]) & (g < 0)).astype(x.dtype), x0, g0, bounds)
                gnorm = tree_map(lambda g,low,high: jnp.linalg.norm(g*(1-low)*(1-high)).item(), g0, at_min, at_max)
            else: 
                gnorm = tree_map(lambda g: jnp.linalg.norm(g).item(), g0)
            for i,(x_,g_,a_) in enumerate(zip(tree_leaves(x0), tree_leaves(gnorm), tree_leaves(a0))):
                tqdm.write(f"   p{i}: shape={jnp.shape(x_)}, |x|={jnp.linalg.norm(x_):.3e}, |g|={g_:.3e}, eta={a_:.3e}, n_updates={sum([a[i] for a in info.accepted])}", file=sys.stderr)
            tqdm.write(f'', file=sys.stderr)

            # work counters
            tqdm.write(f' * work counters', file=sys.stderr)
            tqdm.write(f'   elapsed time: {time.time()-t0:.3f} seconds', file=sys.stderr)
            tqdm.write(f'   gradient updates: {sum([sum(a) for a in info.accepted])}', file=sys.stderr)
            tqdm.write(f'   function calls: {info.nfeval}', file=sys.stderr)
        
    # return
    return x0, info

def tree_update(tree, update, idx):
    tree, treedef = tree_flatten(tree)
    tree[idx] = update
    tree = tree_unflatten(treedef, tree)
    return tree

def update_param(f, f0, g0, x0, a0, b, bounds, growth, backtrack, which):
    """ find an x satisfying armijo conditions, allowed to take multiple gradient steps
    Args:
        f: returns (loss, grad)
        f0: initial function value corresponding to x0
        g0: pytree of gradients
        x0: pytree of initial parameters
        a0: float, step size
        b: float, tolerance in Armijo rule for backtracking line search
        bounds: pytree of bounds for each variable
        backtrack: float, factor by which to decrease step size inside backtracking line search
        maxls: int, max number of steps inside each line search
    """
    # select param and gradients
    x0_ = tree_leaves(x0)[which]
    g0_ = tree_leaves(g0)[which]
    a0_ = tree_leaves(a0)[which]
    if bounds is not None: bounds_ = tree_leaves(bounds)[which]
    
    # expected decrease
    if bounds is not None:
        at_min = ((x0_==bounds_[0]) & (g0_ > 0)).astype(x0_.dtype)
        at_max = ((x0_==bounds_[1]) & (g0_ < 0)).astype(x0_.dtype)
        expected_decrease = (g0_**2 * (1-at_min) * (1-at_max)).sum()
    else:
        expected_decrease = (g0_**2).sum()
        
    # trial update
    x1_ = x0_-a0_*g0_
    if bounds is not None:
        x1_ = x1_.clip(bounds_[0],bounds_[1])
    
    # conditions
    if jnp.allclose(x0_,x1_): # reject and grow step size
        a0 = tree_update(a0, growth*a0_, which)
        x0, a0, f0, g0, accepted, nfeval = x0, a0, f0, g0, False, 0
    else:
        x1 = tree_update(x0, x1_, which)
        f1, g1 = f(x1) # evaluate
        if f1 <= f0 - b*a0_*expected_decrease: # accept and grow step size 
            a0 = tree_update(a0, growth*a0_, which)
            x0, a0, f0, g0, accepted, nfeval = x1, a0, f1, g1, True, 1
        else: # reject and shrink step size
            a0 = tree_update(a0, backtrack*a0_, which)
            x0, a0, f0, g0, accepted, nfeval = x0, a0, f0, g0, False, 1
      
    return x0, f0, g0, a0, accepted, nfeval

###########
# Example #
###########
def example():
    from jax import value_and_grad
    
    def e(params):
        x,y = params
        return (x**2).sum() + 10*(y**2).sum()
    
    params = (jnp.ones(4),jnp.ones(3))
    f = value_and_grad(e)
    minimize(f, params)
