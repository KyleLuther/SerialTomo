import jax.numpy as jnp
from jax.flatten_util import ravel_pytree#, tree_flatten, tree_unflatten
from jax.tree_util import tree_map, tree_reduce, tree_flatten

import time
from types import SimpleNamespace
from tqdm import tqdm

# def backtracking_line_search(f, f0, g0, x0, a0, p, b, backtrack, maxls):
#     """ find an x satisfying armijo conditions """
#     decrease = b * jnp.dot(ravel_pytree(g0)[0], ravel_pytree(p)[0])
#     converged = False
#     for i in range(maxls):
#         x1 = tree_map(lambda x,p: x+a0*p, x0, p)
#         f1, g1 = f(x1)
        
#         if f1 < f0 + a0*decrease:
#             converged = True
#             return x1, g1, a0, f1, i+1, converged
#         else:
#             a0 = backtrack * a0
        
#     return x1, g1, a0, f1, i+1, converged

# def minimize(f, x0, a0=1.0, b=1e-4, growth=2.0, backtrack=0.1, maxiter=50, maxls=20, gtol=1e-12, rtol=1e-12, rescale=True, verbose=1, callback=None):
#     """
#     f: returns (loss, grad)
#     x0: pytree of initial parameters
#     max_iter: max number of iterations
#     max_ls: max number of line search per iteration
#     """
#     info = {}
#     info['converged'] = False
#     info['message'] = None
#     info['niter'] = 0
#     info['nfeval'] = 1
#     t0 = time.time()
    
#     f0, g0 = f(x0)
#     niter = 0
#     if rescale:
#         dx = tree_map(lambda u: 0*u, x0) # x0
#         dg = tree_map(lambda u: 0*u, g0) # g0
#     # while True:
#     for i in (pbar := tqdm(range(maxiter), leave=False, disable=(verbose == 0))):
#         # search direction
#         p = tree_map(lambda g: -g, g0)
        
#         # rescale search direction
#         if rescale:
#             rescale_ = tree_map(lambda dx_, dg_: jnp.abs(dx_).mean().clip(rtol) / jnp.abs(dg_).mean().clip(rtol), dx, dg)
#             p = tree_map(lambda p, r: p*r, p, rescale_)
            
#         # line search
#         x1, g1, a1, f1, nls, lsconverged = backtracking_line_search(f,f0,g0,x0,a0,p,b,backtrack,maxls)
        
#         # store deltas
#         if rescale:
#             dx = tree_map(lambda u,v: u-v, x1, x0)
#             dg = tree_map(lambda u,v: u-v, g1, g0)
            
#         # reset
#         x0 = x1
#         a0 = growth*a1
#         f0 = f1
#         g0 = g1
#         niter += 1
            
#         # log
#         gnorm = jnp.linalg.norm(ravel_pytree(g0)[0])
#         info['fs'] = info.get('fs', []) + [f1.item()]
#         info['a0s'] = info.get('a0s', []) + [a1]
#         info['gnorms'] = info.get('gnorms', []) + [gnorm.item()]
#         info['nfeval'] = info.get('nfeval', 0) + nls
#         info['niter'] = info.get('niter', 0) + 1
        
#         # display
#         pbar.set_postfix({'f': f0})

#         if rescale:
#             info['rescales'] = info.get('rescales', []) + [tree_map(lambda r: r.item(), rescale_)]
        
#         # callback
#         if callback is not None:
#             callback(x0)
            
#         # print
#         if verbose>=2:
#             print(f'iter {niter}/{maxiter}, f={f1:.3e}, |g|={gnorm:.3e}, a={a1:.2e}, nls={nls}')

#         # convergence checks
#         if niter == maxiter: 
#             info['converged'] = False
#             info['message'] = f'reached maxiter={maxiter} iterations'
#             break
        
#         if lsconverged is False: 
#             info['converged'] = False
#             info['message'] = f'line search reached maxls'
#             break
        
#         if gnorm < gtol:
#             info['converged'] = True
#             info['message'] = f'|g| < gtol'
#             break
        
#     info['elapsed_time'] = time.time() - t0
    
#     # final printing
#     if verbose>=1:
#         print(f'exiting after {niter} iterations, nfeval={info["nfeval"]}, converged={info["converged"]}, msg={info["message"]}, elapsed_time={info["elapsed_time"]:.3e}sec, f_final={f1:.3e}')
        
#     # format info
#     info = SimpleNamespace(**info)
    
#     # return
#     return x0, info

def backtracking_line_search(f, f0, g0, x0, a0, r0, b, backtrack, maxls, nsteps):
    """ find an x satisfying armijo conditions, allowed to take multiple gradient steps """
    converged = False
    for i in range(maxls):
        x1, g1 = x0, g0
        expected_decrease = 0.0
        # take nupdates steps 
        for j in range(nsteps):
            p1 = tree_map(lambda g,r: -a0*r*g, g1, r0) # rescale gradients
            x1 = tree_map(lambda x,p: x+p, x1, p1) # update params
            if j == 0: expected_decrease += jnp.dot(ravel_pytree(g1)[0], ravel_pytree(p1)[0]) # update expected decrease
            f1, g1 = f(x1) # evaluate
        
        if f1 < f0 + b*expected_decrease:
            converged = True
            break
        else:
            a0 = backtrack * a0
        
    return x1, g1, a0, f1, i+1, converged

def minimize(f, x0, a0=1.0, b=1e-4, growth=2.0, backtrack=0.1, maxiter=50, maxls=20, gtol=1e-12, rtol=1e-12, ftol=0.0, autorescale=True, verbose=True, nsteps=1, callback=None):
    """ Gradient descent on f, starting with initial condition x0
    Args:
        f: returns (loss, grad)
        x0: pytree of initial parameters
        a0: float, initial step size
        b: float, tolerance in Armijo rule for backtracking line search
        growth: float, factor by which to increase step size
        backtrack: float, factor by which to decrease step size inside backtracking line search
        maxiter: int, max number of gradient steps
        maxls: int, max number of steps inside each line search
        gtol: float >= 0.0, iteration is stopped when |g| < gtol
        ftol: float >= 0.0, iteration is stopped when f0 - f1 < ftol
        autorescale: bool, if True rescale every group of variables according to Hessian estimate
        rtol: float > 0.0, controls about each group of variables is rescaled by. Only applicable if auto_rescale=True
        verbose: bool, display information about minimization
        nsteps: int, number of gradient steps to take per update
        callback: function which takes in x0, can be used for logging
        
    Returns:
        x: pytree of final parameters
        info: namespace containing optimization info
    """
    info = {}
    info['converged'] = False
    info['message'] = None
    info['niter'] = 0
    info['nfeval'] = 1
    
    # initialize
    t0 = time.time()
    f0, g0 = f(x0)

    # main loop
    for niter in (pbar := tqdm(range(maxiter), leave=True, disable=not verbose, desc='minimize')):
        # rescale search direction
        if autorescale and niter > 0:
            r0 = tree_map(lambda dx_, dg_: (jnp.abs(dx_).mean().clip(rtol) / jnp.abs(dg_).mean().clip(rtol)).item(), dx, dg)
        else:
            r0 = tree_map(lambda x: 1.0, x0)
            
        # line search
        x1, g1, a1, f1, nls, lsconverged = backtracking_line_search(f,f0,g0,x0,a0,r0,b,backtrack,maxls,nsteps)
        
        # store deltas
        df = f0 - f1
        if autorescale:
            dx = tree_map(lambda u,v: u-v, x1, x0)
            dg = tree_map(lambda u,v: u-v, g1, g0)
            
        # reset
        x0 = x1
        a0 = growth*a1
        f0 = f1
        g0 = g1
            
        # log
        gnorm = jnp.linalg.norm(ravel_pytree(g0)[0]).item()
        info['fs'] = info.get('fs', []) + [f1.item()]
        info['a0s'] = info.get('a0s', []) + [a1]
        info['gnorms'] = info.get('gnorms', []) + [gnorm]
        info['nfeval'] = info.get('nfeval', 1) + nls*nsteps
        info['r0s'] = info.get('r0s', []) + [tree_map(lambda r: r, r0)]
        info['niter'] = niter+1

        # update display
        pbar.set_postfix({'f': f0, '|g|': gnorm})
        
        # callback
        if callback is not None:
            callback(x0)
            
        # convergence checks
        if niter + 1 == maxiter: 
            info['converged'] = False
            info['message'] = f'reached maxiter={maxiter} iterations'
            break
        
        if lsconverged is False: 
            info['converged'] = False
            info['message'] = f'line search reached {maxls=} iterations'
            break
        
        if gnorm < gtol:
            info['converged'] = True
            info['message'] = f'|g| < gtol'
            break
            
        if df < ftol:
            info['converged'] = True
            info['message'] = f'f0-f1 < ftol'
            break
            
    info['elapsed_time'] = time.time() - t0
    
    # final printing
    if verbose>=1:
        final_log = tqdm(total=0, bar_format='{desc}', disable=not verbose)
        final_log.set_description_str(f'results: converged={info["converged"]} {info["message"]}, niter={niter}, nfeval={info["nfeval"]}')
        
    # format info
    info = SimpleNamespace(**info)
    
    # return
    return x0, info

###########
# Example #
###########
def runexample():
    from jax import value_and_grad
    
    def e(params):
        x,y = params
        return (x**2).sum() + 10*(y**2).sum()
    
    params = (jnp.ones(4),jnp.ones(3))
    f = value_and_grad(e)
    minimize(f, params)