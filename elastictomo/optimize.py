""" Infers volume and imaging parameters given measured projections """
from elastictomo.forward import forward
from elastictomo.interpolate import interpolate, interpolate_vecs, pixelate

# loss
def elastic_regularization(dsp_grid):
    """ elastic regularization loss """
    dy = dsp_grid[1:]-dsp_grid[:-1]
    dx = dsp_grid[:,1:]-dsp_grid[:,:-1]
    
    return (dx**2).sum() + (dy**2).sum()

def loss(vol_grid, dsp_grids, thetas, exposures, lam, tilts, masks):
    """ Computes total loss: \sum_i (pred_i-proj_i)^2 + lam * reg_i """
    # logging
    info = {}

    # setup geometry
    vol_size = vol_grid.shape
    ccd_size = dsp_grids[0].shape[:-1]
    
    # construct continuous volume and displacement fields 
    vol = interpolate(vol_grid)
    dsps = [interpolate_vecs(dsp_grid) for dsp_grid in dsp_grids]
    
    # compute loss for each projection
    loss = 0.0
    for i in range(len(projs)):
        pred = forward(vol, dsps[i], vol_size, ccd_size,
                       orientation=thetas[i], exposure=exposures[i], oversample=(1,1,1))

        mse = ((1-masks[i])*(pred-tilts[i])**2).sum()
        reg = elastic_regularization(dsp_grids[i])
        loss = loss + (mse + lam*reg) / len(projs)
        
        # logging
        info['losses'] = info.get('losses',[]) + [loss]
    
    return loss, info

# initialization
def get_tmats(tilts, zero_ix):
    """ Return affine transform matrices for every tilt """
    sr = StackReg(StackReg.AFFINE)
    tmats1 = np.flip(sr.register_stack(np.flip(img[:zero_ix+1],0), reference='previous'),0)
    tmats2 = sr.register_stack(img[zero_ix:], reference='previous')
    tmats = np.concatenate([tmats1,tmats2[1:]])
    return tmats

def shifts_from_tmats(tmats, size):
    """ Determine shifts from transform matrices """
    shifts = tmats @ np.array([size[0]/2,size[1]/2,0]) - np.array([size[0]/2,size[1]/2,0])
    ty = -shifts[:,1]
    tx = shifts[:,0]
    return np.stack([ty, tx],axis=-1)

def init_shifts(tilts, thetas, downsample=1):
    """ Determine shifts, using downsampled images """
    tilts = nn.avg_pool(tilts[...,None],(downsample,downsample),(downsample,downsample))[...,None]
    zero_ix = np.argmin(np.abs(thetas))
    tmats = get_tmats(tilts,zero_ix)
    shifts = shifts_from_tmats(tmats,tilts[0].shape)
    shifts = downsample * shifts
    return shifts

def inference(projections, init_orientations):
    pass
    # initialize parameters
    
    # optimize
    
    # return volume                   
                   