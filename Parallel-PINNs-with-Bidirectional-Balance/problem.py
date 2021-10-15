from matplotlib.pyplot import sca
import numpy as np

class Problem(object):
    def __init__(self, ca, k, T0, Tg, conv, rad, gamma, domain, no_FBM):
        self.bz = 5.67e-8
        self.gamma = gamma
        self.rad = rad
        self.T0 = T0
        self.Tg = Tg

        self.ca = ca
        self.k = k
        self.conv = conv
    
        self.domain = domain
        self.scale = np.array([1,1,1])

        # if use DLB, Restore the relevant partial derivatives by self.re
        self.re = 1e3
        if no_FBM:
            self.re = 1

    def __repr__(self):
        return f'{self.__doc__}'
    
    def set_scale(self, scale=[1e-2,1.27e-4,4.72e-8]):
        self.scale = np.array(scale)
        
    def u0(self, y):
        u0 = (y*self.re - self.T0*self.re)*self.scale[0]
        return u0

    def r_shl(self, y_t, y_xx):
        r = (self.ca[0]*y_t*self.re - self.k[0]*y_xx*(self.re**3))*self.scale[2]
        return r

    def ub_shl(self, y, y_x):
        ub = (- self.k[0]*y_x*(self.re**2) - self.conv[0]*(self.Tg-y)*self.re)*self.scale[1]
        return ub
    
    def r_msr(self, y_t, y_xx):
        r = (self.ca[1]*y_t*self.re - self.k[1]*y_xx*(self.re**3))*self.scale[2]
        return r

    def ub_shlmsr(self, y, yshl, y_x, yshl_x):
        ub1 = (y*self.re - yshl*self.re)*self.scale[0]
        ub2 = (self.k[0]*yshl_x*(self.re**2) - self.k[1]*y_x*(self.re**2))*self.scale[1]
        return ub1, ub2

    def r_lin(self, y_t, y_xx):
        r = (self.ca[2]*y_t*self.re - self.k[2]*y_xx*(self.re**3))*self.scale[2]
        return r

    def ub_msrlin(self, y, ymsr, y_x, ymsr_x):
        ub1 = (y*self.re - ymsr*self.re)*self.scale[0]
        ub2 = (self.k[1]*ymsr_x*(self.re**2) - self.k[2]*y_x*(self.re**2))*self.scale[1]
        return ub1, ub2
    
    def ub_linair(self, y, y_x):
        ub = (- self.k[2]*y_x*(self.re**2) - self.conv[1]*(y-self.T0)*self.re)*self.scale[1]
        return ub
