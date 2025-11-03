import torch as th
import numpy as np


class FourierKAN(th.nn.Module):
    def __init__( self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False):
        super().__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        grid_norm_factor = (th.arange(gridsize) + 1)**2 if smooth_initialization else np.sqrt(gridsize)
        self.fouriercoeffs = th.nn.Parameter(th.randn(2,outdim,inputdim,gridsize) / (np.sqrt(inputdim) * grid_norm_factor ))
        if self.addbias:
            self.bias = th.nn.Parameter(th.zeros(1,outdim))

    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1]+(self.outdim,)
        x = th.reshape(x,(-1,self.inputdim))
        k = th.reshape(th.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))
        xrshp = th.reshape(x,(x.shape[0],1,x.shape[1],1))
        c = th.cos(k*xrshp)
        s = th.sin(k*xrshp)
        y =  th.sum(c*self.fouriercoeffs[0:1],(-2,-1)) 
        y += th.sum(s*self.fouriercoeffs[1:2],(-2,-1))
        if( self.addbias):
            y += self.bias
        y = th.reshape(y, outshape)
        return y