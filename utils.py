import numpy as np
from spectres import spectres

def rebin(wlen,flux,wlen_data,flux_err = None, method='linear'):
    #wlen larger than wlen_data
    
    
    #if method == 'linear':
    #extends wlen linearly outside of wlen_data using the spacing on each side
    if method == 'linear':
        stepsize_left = abs(wlen_data[1]-wlen_data[0])
        
        N_left = int((wlen_data[0]-wlen[0])/stepsize_left)-1
        wlen_left = np.linspace(wlen_data[0]-N_left*stepsize_left,
                                wlen_data[0],
                                N_left,
                                endpoint=False)
        
        stepsize_right = wlen_data[-1]-wlen_data[-2]
        
        N_right = int((wlen[-1]-wlen_data[-1])/stepsize_right)-1
        wlen_right = np.linspace(wlen_data[-1]+stepsize_right,
                                wlen_data[-1]+(N_right+1)*stepsize_right,
                                N_right,
                                endpoint=False)
        
        wlen_temp = np.concatenate((wlen_left,wlen_data,wlen_right))
    elif method == 'datalike':
        wlen_temp = wlen_data
    if flux_err is not None:
        assert(np.shape(flux_err)==np.shape(flux))
        flux_temp,flux_new_err = spectres(wlen_temp,wlen,flux,spec_errs = flux_err)
        return wlen_temp,flux_temp,flux_new_err
    else:
        flux_temp = spectres(wlen_temp,wlen,flux)
        return wlen_temp,flux_temp