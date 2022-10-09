import jax.numpy as jnp
from jax.flatten_util import ravel_pytree#, tree_flatten, tree_unflatten
from jax.tree_util import tree_map, tree_reduce, tree_flatten

import time
from types import SimpleNamespace
from tqdm import tqdm

def backtracking_line_search(f, f0, g0, x0, a0, p, b, backtrack, maxls):
    """ find an x satisfying armijo conditions """
    decrease = b * jnp.dot(ravel_pytree(g0)[0], ravel_pytree(p)[0])
    converged = False
    for i in range(maxls):
        x1 = tree_map(lambda x,p: x+a0*p, x0, p)
        f1, g1 = f(x1)
        
        if f1 < f0 + a0*decrease:
            converged = True
            return x1, g1, a0, f1, i+1, converged
        else:
            a0 = backtrack * a0
        
    return x1, g1, a0, f1, i+1, converged

def minimize(f, x0, a0=1.0, b=1e-4, growth=2.0, backtrack=0.1, maxiter=50, maxls=20, gtol=1e-12, rtol=1e-12, rescale=True, verbose=1, callback=None):
    """
    f: returns (loss, grad)
    x0: pytree of initial parameters
    max_iter: max number of iterations
    max_ls: max number of line search per iteration
    """
    info = {}
    info['converged'] = False
    info['message'] = None
    info['niter'] = 0
    info['nfeval'] = 1
    t0 = time.time()
    
    f0, g0 = f(x0)
    niter = 0
    if rescale:
        dx = tree_map(lambda u: 0*u, x0) # x0
        dg = tree_map(lambda u: 0*u, g0) # g0
    # while True:
    for i in (pbar := tqdm(range(maxiter), leave=False, disable=(verbose == 0))):
        # search direction
        p = tree_map(lambda g: -g, g0)
        if rescale:
            rescale_ = tree_map(lambda dx_, dg_: jnp.abs(dx_).mean().clip(rtol) / jnp.abs(dg_).mean().clip(rtol), dx, dg)
            # rsum = tree_reduce(lambda x, y: x.sum() + y.sum(), rescale_)
            # rescale_ = tree_map(lambda r: r / rsum, rescale_)
            p = tree_map(lambda p, r: p*r, p, rescale_)
            
        # line search
        x1, g1, a1, f1, nls, lsconverged = backtracking_line_search(f,f0,g0,x0,a0,p,b,backtrack,maxls)
        
        # store deltas
        if rescale:
            dx = tree_map(lambda u,v: u-v, x1, x0)
            dg = tree_map(lambda u,v: u-v, g1, g0)
            
        # reset
        x0 = x1
        a0 = growth*a1
        f0 = f1
        g0 = g1
        niter += 1
            
        # log
        gnorm = jnp.linalg.norm(ravel_pytree(g0)[0])
        info['fs'] = info.get('fs', []) + [f1.item()]
        info['a0s'] = info.get('a0s', []) + [a1]
        info['gnorms'] = info.get('gnorms', []) + [gnorm.item()]
        info['nfeval'] = info.get('nfeval', 0) + nls
        info['niter'] = info.get('niter', 0) + 1
        
        # display
        pbar.set_postfix({'f': f0})

        if rescale:
            info['rescales'] = info.get('rescales', []) + [tree_map(lambda r: r.item(), rescale_)]
        
        # callback
        if callback is not None:
            callback(x0)
            
        # print
        if verbose>=2:
            print(f'iter {niter}/{maxiter}, f={f1:.3e}, |g|={gnorm:.3e}, a={a1:.2e}, nls={nls}')

        # convergence checks
        if niter == maxiter: 
            info['converged'] = False
            info['message'] = f'reached maxiter={maxiter} iterations'
            break
        
        if lsconverged is False: 
            info['converged'] = False
            info['message'] = f'line search reached maxls'
            break
        
        if gnorm < gtol:
            info['converged'] = True
            info['message'] = f'|g| < gtol'
            break
        
    info['elapsed_time'] = time.time() - t0
    
    # final printing
    if verbose>=1:
        print(f'exiting after {niter} iterations, nfeval={info["nfeval"]}, converged={info["converged"]}, msg={info["message"]}, elapsed_time={info["elapsed_time"]:.3e}sec, f_final={f1:.3e}')
        
    # format info
    info = SimpleNamespace(**info)
    
    # return
    return x0, info

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
#     while True:
#         # search direction
#         p = tree_map(lambda g: -g, g0)
#         if rescale:
#             # if niter == 5:
#                 # return dx, dg
#             # rescale_ = tree_map(lambda dx_, dg_: jnp.abs(dx_).mean().clip(rtol) / jnp.abs(dg_).mean().clip(rtol), dx, dg)
#             rescale_ = tree_map(lambda dx_, dg_: (jnp.abs(dg_).clip(1e-12) / jnp.abs(dx_).clip(1e-12)), dx, dg)
#             rescale_ = tree_map(lambda r: 1 / r.clip(rtol*r.max()), rescale_)
#             rescale_ = tree_map(lambda r: r / r.max(), rescale_)
            
#             # print(rescale_[0].min(), rescale_[0].max())
            
#             p = tree_map(lambda p, r: p*r, p, rescale_)
            
#         # line search
#         # a0 = 1.0
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
#         info['as'] = info.get('as', []) + [a1]
#         info['gnorms'] = info.get('gnorms', []) + [gnorm.item()]
#         info['nfeval'] = info.get('nfeval', 0) + nls
#         info['niter'] = info.get('niter', 0) + 1

#         # if rescale:
#             # info['rescales'] = info.get('rescales', []) + [tree_map(lambda r: r.item(), rescale_)]
        
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