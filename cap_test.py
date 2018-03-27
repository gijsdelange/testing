import sys


import transport_characterization
from transport_characterization.fitter import fit
from transport_characterization import CV_measurements as cv
import pickle 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import qcodes as qc
table = {1: (100, 'pF'),
         2: (68, 'pF'),
         3: (47, 'pF'),
         4: (33, 'pF'),
         5: (10, 'pF'),
         8: (20, 'kOhm'),
         9: (1, 'pF'),
         10: (0,'open')}

channel = 8
C_Ds = []
C_Derrs = []
C_set = []
plt.figure('cap_test')
plt.clf()
dats = []
for ch in [1,2,3,4,5,9,10]:    
    fname = 'ch{}_{}{}'.format(ch, *table[ch])
    fpath = r'D:\OnedriveMS\OneDrive - Microsoft\Projects\20171010 - SAG Delft MBE\data\CVmeasurements\voltage_bias\{}.pickle'.format(fname)

    C_set += [table[ch][0]]
    dat = pickle.load(open(fpath, 'rb'))
    fs = dat['f (Hz)']
    Is = dat['I (A)'] #- dat['I'][0]
    dats += [Is]
    Ix = Is.real
    Iy = Is.imag
    V_ac = 1e-3
    R_m = 980
    C_p = 1.30e-10
    C_off = 2.6439e-12
    R_Dp = 2e8
    R_Ds = 0.1
    C_D = 20e-12
    cap_model, pars = fit.make_model(cv.V_out_cap_model, 
                                        p0 = (fs, V_ac, C_D, R_Dp, R_Ds, C_off, R_m, C_p))
    pars['V_ac'].value = V_ac;pars['V_ac'].vary = False
    if 0:
        for par in ['C_D', 'R_Dp']:
            pars[par].min = 0.
    pars['R_Ds'].vary = 1
    pars['R_Dp'].vary = 1
    pars['C_D'].vary = 1
    pars['R_m'].vary = 0
    pars['C_off'].vary = 0
    #pars['L_i'].vary = False
    pars['C_p'].vary = 0
    
    
    
    plt.figure(5)
    plt.clf()
    plt.subplot(211)
    plt.plot(fs, Ix, '.')
    plt.plot(fs, cap_model(pars).real, label = 'init_guess')
    plt.subplot(212)
    plt.plot(fs, Iy, '.')
    plt.plot(fs, cap_model(pars).imag, label = 'init_guess')
    
    plt.figure(4)
    plt.clf()
    plt.plot(Is.real, Is.imag, '.', label = 'data')
    plt.plot(cap_model(pars).real, cap_model(pars).imag, 'r', label = 'init_guess')
    result = fit.fit(pars, Is, cap_model)
    plt.plot(cap_model(pars).real, cap_model(pars).imag, 'g', label = 'fit')
    plt.legend()
    
    plt.figure(5)
    
    plt.subplot(211)
    plt.plot(fs, cap_model(pars).real, label = 'fit')
    plt.xlabel('f (Hz)')
    plt.ylabel(r'Re{I} (A)')
    plt.title(fname)
    plt.subplot(212)
    plt.plot(fs, cap_model(pars).imag, label = 'fit')
    plt.legend()
    plt.xlabel('f (Hz)')
    plt.ylabel(r'Im{I} (A)')
    
    fit.print_fitres(pars)
    C_Dm = pars['C_D'].value + C_off
    C_Ds += [pars['C_D'].value]
    C_Derrs += [pars['C_D'].stderr]
    plt.figure('cap_test')
    plt.plot(Is.real/C_Dm, Is.imag/C_Dm, '.', label = '{}{}'.format(*table[ch]))
    
    plt.plot(cap_model(pars).real/C_Dm, cap_model(pars).imag/C_Dm, 'gray')#, label = 'fit')
    plt.ylabel('Im[V_out] (V)')
    plt.xlabel('Re[V_out] (V)')
    plt.legend()
    plt.tight_layout()
plt.figure('cp _cal')
plt.clf()
plt.errorbar(np.array(C_set), np.array(C_Ds)/np.array(C_set), yerr = np.array(C_Derrs)/np.array(C_set), fmt =  'o')

pars.pop('C_D')
ncaps = 7
[pars.add('C_D{}'.format(kk), value = C_set[kk]) for kk in range(ncaps)]

if 0:
    pars['R_Dp'].value = 1e9
    pars['R_Dp'].vary = 0
    pars['R_Ds'].value = 50.
    pars['R_Ds'].vary = 0
    pars['C_p'].vary = 1
    pars['C_off'].vary = 1
    pars['R_m'].vary = 0
    fit.print_fitres(pars)
    def residuals(pars, data):
        
        
        fs = pars['f'].value
        res = np.zeros(2*len(fs)*ncaps)                                                
        for kk in range(ncaps):
            ps = np.abs((pars['f'].value, 
                    pars['V_ac'].value,
                    pars['C_D{}'.format(kk)].value,
                    pars['R_Ds'].value,
                    pars['R_Dp'].value,
                    pars['C_off'].value,
                    pars['R_m'].value,
                    pars['C_p'].value))
            (fs, V_ac, C_D, R_Dp, R_Ds, C_off, R_m, C_p) = ps
            
            res_cmplx = cv.V_out_cap_model(*ps) - data[kk]
            res_kk = np.append(res_cmplx.real, res_cmplx.imag)
            #print(len(res_kk))
            res[2*kk*len(fs): 2*len(fs)*(kk+1)] = (res_kk)
        return res
    result = fit.minimize(residuals, pars, args = (dats,))
    fit.print_fitres(pars)
            
        