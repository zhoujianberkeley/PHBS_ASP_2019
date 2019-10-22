    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm

'''
Asymptotic approximation for 0<beta<=1 by Hagan
'''
def bsm_vol(strike, forward, texp, sigma, alpha=0, rho=0, beta=1):
    if(texp<=0.0):
        return( 0.0 )

    powFwdStrk = (forward*strike)**((1-beta)/2)
    logFwdStrk = np.log(forward/strike)
    logFwdStrk2 = logFwdStrk**2
    rho2 = rho*rho

    pre1 = powFwdStrk*( 1 + (1-beta)**2/24 * logFwdStrk2*(1 + (1-beta)**2/80 * logFwdStrk2) )
  
    pre2alp0 = (2-3*rho2)*alpha**2/24
    pre2alp1 = alpha*rho*beta/4/powFwdStrk
    pre2alp2 = (1-beta)**2/24/powFwdStrk**2

    pre2 = 1 + texp*( pre2alp0 + sigma*(pre2alp1 + pre2alp2*sigma) )

    zz = powFwdStrk*logFwdStrk*alpha/np.fmax(sigma, 1e-32)  # need to make sure sig > 0
    if isinstance(zz, float):
        zz = np.array([zz])
    yy = np.sqrt(1 + zz*(zz-2*rho))

    xx_zz = np.zeros(zz.size)

    ind = np.where(abs(zz) < 1e-5)
    xx_zz[ind] = 1 + (rho/2)*zz[ind] + (1/2*rho2-1/6)*zz[ind]**2 + 1/8*(5*rho2-3)*rho*zz[ind]**3
    ind = np.where(zz >= 1e-5)
    xx_zz[ind] = np.log( (yy[[ind]] + (zz[ind]-rho))/(1-rho) ) / zz[ind]
    ind = np.where(zz <= -1e-5)
    xx_zz[ind] = np.log( (1+rho)/(yy[ind] - (zz[ind]-rho)) ) / zz[ind]

    bsmvol = sigma*pre2/(pre1*xx_zz) # bsm vol
    return(bsmvol[0] if bsmvol.size==1 else bsmvol)

'''
Asymptotic approximation for beta=0 by Hagan
'''
def norm_vol(strike, forward, texp, sigma, alpha=0, rho=0):
    # forward, spot, sigma may be either scalar or np.array. 
    # texp, alpha, rho, beta should be scholar values

    if(texp<=0.0):
        return( 0.0 )
    
    zeta = (forward - strike)*alpha/np.fmax(sigma, 1e-32)
    # explicitly make np.array even if args are all scalar or list
    if isinstance(zeta, float):
        zeta = np.array([zeta])
        
    yy = np.sqrt(1 + zeta*(zeta - 2*rho))
    chi_zeta = np.zeros(zeta.size)
    
    rho2 = rho*rho
    ind = np.where(abs(zeta) < 1e-5)
    chi_zeta[ind] = 1 + 0.5*rho*zeta[ind] + (0.5*rho2 - 1/6)*zeta[ind]**2 + 1/8*(5*rho2-3)*rho*zeta[ind]**3

    ind = np.where(zeta >= 1e-5)
    chi_zeta[ind] = np.log( (yy[ind] + (zeta[ind] - rho))/(1-rho) ) / zeta[ind]

    ind = np.where(zeta <= -1e-5)
    chi_zeta[ind] = np.log( (1+rho)/(yy[ind] - (zeta[ind] - rho)) ) / zeta[ind]

    nvol = sigma * (1 + (2-3*rho2)/24*alpha**2*texp) / chi_zeta
 
    return(nvol[0] if nvol.size==1 else nvol)

'''
Hagan model class for 0<beta<=1
'''
class ModelHagan:
    alpha, beta, rho = 0.0, 1.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.beta = beta
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return bsm_vol(strike, forward, texp, sigma, alpha=self.alpha, beta=self.beta, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        bsm_vol = self.bsm_vol(strike, spot, texp, sigma)
        return self.bsm_model.price(strike, spot, texp, bsm_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.bsm_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            bsm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 10)
        if(setval):
            self.sigma = sigma
        return sigma
    
    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or bsm vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        if texp is None:
        	texp = self.texp	
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        if is_vol:
            vol = price_or_vol3                     
        else:
            _price = price_or_vol3
            vol = np.ones(3)
            vol[0] = self.bsm_model.impvol(_price[0], strike3[0], spot, texp, cp_sign= cp_sign)
            vol[1] = self.bsm_model.impvol(_price[1], strike3[1], spot, texp, cp_sign= cp_sign)
            vol[2] = self.bsm_model.impvol(_price[2], strike3[2], spot, texp, cp_sign= cp_sign)
            
        def helper(x):
            return [bsm_vol(strike3[0], forward, texp, x[0], x[1], x[2])-vol[0],
                    bsm_vol(strike3[1], forward, texp, x[0], x[1], x[2])-vol[1],
                    bsm_vol(strike3[2], forward, texp, x[0], x[1], x[2])-vol[2]]
        
        sol = sopt.root(helper,[2,2,0.2],method='hybr') 
     
        if setval:
            self.sigma = sol.x[0]
            self.alpha = sol.x[1]
            self.rho  = sol.x[2]

        return sol.x # sigma, alpha, rho

'''
Hagan model class for beta=0
'''
class ModelNormalHagan:
    alpha, beta, rho = 0.0, 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.beta = 0.0 # not used but put it here
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return norm_vol(strike, forward, texp, sigma, alpha=self.alpha, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        n_vol = self.norm_vol(strike, spot, texp, sigma)
        return self.normal_model.price(strike, spot, texp, n_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            norm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 50)
        if(setval):
            self.sigma = sigma
        return sigma

    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or normal vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        if texp is None:
	        texp = self.texp
        
        forward = spot * np.exp(texp*(self.intr - self.divr))
        if is_vol:
            vol = price_or_vol3                     
        else:
            vol = np.ones(3)
            vol[0] = self.normal_model.impvol(price_or_vol3[0], strike3[0], spot, texp, cp_sign= cp_sign)
            vol[1] = self.normal_model.impvol(price_or_vol3[1], strike3[1], spot, texp, cp_sign= cp_sign)
            vol[2] = self.normal_model.impvol(price_or_vol3[2], strike3[2], spot, texp, cp_sign= cp_sign)
            
        def helper(x):
            return [norm_vol(strike3[0], forward, texp, x[0], x[1], x[2])-vol[0],
                  norm_vol(strike3[1], forward, texp, x[0], x[1], x[2])-vol[1],
                  norm_vol(strike3[2], forward, texp, x[0], x[1], x[2])-vol[2]]
        
        result = sopt.root(helper,[10,10,0.15],method='hybr') 
     
        if setval:
            self.sigma, self.alpha, self.rho = result.x
        
        return result.x # sigma, alpha, rho

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    timestep = 120
    n_sample = 1000
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        if not sigma:
            sigma = self.sigma
        if not texp:
            texp = self.texp
        
        price = self.price(strike, spot, texp, sigma)
        impvol_lst = np.zeros(len(strike))
        for i in range(len(strike)):
            impvol_lst[i] = self.bsm_model.impvol(price[i], strike[i], spot, texp)
 
        return impvol
        
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        if not sigma:
            sigma = self.sigma
        if not texp:
            texp = self.texp
        d1, d2 = np.random.normal(size=(self.n_sample,self.timestep)), np.random.normal(size=(self.n_sample,self.timestep))
        
        w1 = self.rho*d1+np.sqrt(1-self.rho**2)*d2
        path_v, path_p = self.sigma*np.ones((self.n_sample,self.timestep+1)), spot*np.ones((self.n_sample,self.timestep+1))
                           
        delt = texp/self.timestep
        ratio_of_vol = np.exp(self.alpha*np.sqrt(delt)*d1-0.5*self.alpha**2*delt)
        path_v[:,1:] = sigma*np.cumprod(ratio_of_vol,axis=1)
        ratio_p = np.exp(path_v[:,:-1]*np.sqrt(delt)*w1-0.5*path_v[:,:-1]**2*delt)
        path_p[:,1:] = spot*np.cumprod(ratio_p,axis=1)
        
        prices = np.mean(np.fmax(cp_sign*(path_p[:,-1].reshape(self.n_sample,1)-strike),0),0)
                                   
        return prices

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    timestep = 120
    n_sample = 1000
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        if not sigma:
            sigma = self.sigma
        if not texp:
            texp = self.texp
        prices = self.price(strike, spot, texp, sigma)
        impvol_lst = np.zeros(len(strike))
        for i in range(len(strike)):
            impvol_lst[i] = self.normal_model.impvol(prices[i], strike[i], spot, texp)
        return impvol_lst
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        if not sigma:
            sigma = self.sigma
        if not texp:
            texp = self.texp
        d1, d2 = np.random.normal(size=(self.n_sample,self.timestep)), np.random.normal(size=(self.n_sample,self.timestep))

        www = self.rho*d1+np.sqrt(1-self.rho**2)*d2
        vol_path = self.sigma*np.ones((self.n_sample,self.timestep+1))
        price_path = spot*np.ones((self.n_sample,self.timestep+1))
                  
        delta_t = texp/self.timestep
        vol_ratio = np.exp(self.alpha*np.sqrt(delta_t)*d1-0.5*self.alpha**2*delta_t)
        vol_path[:,1:] = sigma*np.cumprod(vol_ratio,axis=1)
        delta_p = vol_path[:,:-1]*www*np.sqrt(delta_t)
        price_path[:,1:] = spot+np.cumsum(delta_p,axis=1)
        
        price_lst = np.mean(np.fmax(cp_sign*(price_path[:,-1].reshape(self.n_sample,1)-strike),0),0) 
        return price_lst

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    timestep = 120
    n_sample = 1000
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        if not sigma:
            sigma = self.sigma
        if not texp:
            texp = self.texp
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        price = self.price(strike, spot, texp, sigma)
        impvol = np.zeros(len(strike))
        for i in range(len(strike)):
            impvol[i] = self.bsm_model.impvol(price[i], strike[i], spot, texp)
        return impvol
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        if not sigma:
            sigma = self.sigma
        if not texp:
            texp = self.texp
        
        z1 = np.random.normal(size=(self.n_sample,self.timestep))
        delta_t = texp/self.timestep
        vol_ratio = np.exp(self.alpha*np.sqrt(delta_t)*z1-0.5*self.alpha**2*delta_t)
        vol_path = self.sigma*np.ones((self.n_sample,self.timestep+1))
        vol_path[:,1:] = sigma*np.cumprod(vol_ratio,axis=1)
        
        coef = np.ones(self.timestep+1)
        coef[1:-2] = np.tile(np.array([4,2]),int(self.timestep/2-1))
        coef[-2] = 4
        var = np.sum(coef*vol_path**2*1/3*delta_t,axis=1)
        con = spot*np.exp(self.rho/self.alpha*(vol_path[:,-1]-vol_path[:,0])-self.rho**2/2*var).reshape(self.n_sample,1)
        vcon = np.sqrt((1-self.rho**2)*var/texp).reshape(self.n_sample,1)
        
        price_lst = np.mean(self.bsm_model.price(strike, con, texp, vcon, cp_sign=cp_sign),0)

        return price_lst

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    timestep = 120
    n_sample = 1000

    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        if not sigma:
            sigma = self.sigma
        if not texp:
            texp = self.texp
        prices = self.price(strike, spot, texp, sigma)
        impvol_lst = np.zeros(len(strike))
        for i in range(len(strike)):
            impvol_lst[i] = self.normal_model.impvol(prices[i], strike[i], spot, texp)
        return impvol_lst
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        np.random.seed(12345)
        if not sigma:
            sigma = self.sigma
        if not texp:
            texp = self.texp

        z1= np.random.normal(size=(self.n_sample,self.timestep))
        delta_t = texp/self.timestep
        vol_ratio = np.exp(self.alpha*np.sqrt(delta_t)*z1-0.5*self.alpha**2*delta_t)
        vol_path = self.sigma*np.ones((self.n_sample,self.timestep+1))
        vol_path[:,1:] = sigma*np.cumprod(vol_ratio,axis=1)
        coef = np.ones(self.timestep+1)
        coef[1:-2] = np.tile(np.array([4,2]),int(self.timestep/2-1))
        coef[-2] = 4
        I_var = np.sum(coef*vol_path**2*1/3*delta_t,axis=1)
        s_con = (spot+self.rho/self.alpha*(vol_path[:,-1]-vol_path[:,0])).reshape(self.n_sample,1)
        vol_con = np.sqrt((1-self.rho**2)*I_var/texp).reshape(self.n_sample,1)
        
        price = np.mean(self.normal_model.price(strike, s_con, texp, vol_con, cp_sign=cp_sign),0)
                        
        return price
       