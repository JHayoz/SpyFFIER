import numpy as np
from spectres import spectres
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from astropy.coordinates import SkyCoord
import astropy.units as u
import skycalc_ipy
from scipy.interpolate import UnivariateSpline,interp1d
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
from sklearn.linear_model import RANSACRegressor,LinearRegression,Ridge
from sklearn.preprocessing import PolynomialFeatures,SplineTransformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
slits_ordering = np.array([9,8,10,7,11,6,12,5,13,4,14,3,15,2,16,1,32,17,31,18,30,19,29,20,28,21,27,22,26,23,25,24],dtype=int)
def plot_data(data,vmin=5,vmax=95):
    plt.figure(figsize=(10,10))
    plt.imshow(data,vmin=np.nanpercentile(data,vmin),vmax=np.nanpercentile(data,vmax),origin='lower')
    plt.show()

def plot_spectrum(wlen,spectrum,range=(2.2,2.45),kwargs={}):
    mask = np.logical_and(wlen > range[0], wlen < range[1])
    if np.sum(mask) == 0:
        return
    sum = np.std(spectrum[mask])
    plt.plot(wlen[mask],spectrum[mask]/sum-np.mean(spectrum[mask]/sum),**kwargs)
    if 'label' in kwargs.keys():
        plt.legend()
def diagnostic_wvl_solution(wlen_ref,wlen_dict,flux_dict,nb_intervals = 3,interval = 0.01,plot_correction=True):
    colors_x = np.linspace(0,1,len(wlen_dict.keys()))
    colors = {key:mpl.colormaps['viridis'](colors_x[key_i]) for key_i,key in enumerate(wlen_dict.keys())}
    if plot_correction:
        plt.figure(figsize=(8,2))
        for key in wlen_dict.keys():
            if key == 'Tellurics':
                continue
            plt.plot(wlen_ref,(wlen_ref-wlen_dict[key])/(wlen_ref[1]-wlen_ref[0]),color=colors[key],label=key)
        plt.title('Wavelength correction')
        plt.ylabel('Correction [px]')
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.legend()
        plt.show()
    range_start = np.linspace(wlen_ref[0],wlen_ref[-1]-interval,nb_intervals)
    for i in np.arange(nb_intervals):
        plt.figure(figsize=(10,3))
        for key in wlen_dict.keys():
            if key == 'Tellurics':
                color = 'r'
            else:
                color = colors[key]
            plot_spectrum(wlen_dict[key],flux_dict[key],range=(range_start[i],range_start[i]+interval),kwargs={'label':key,'color':color})
        plt.legend()
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel(r'Flux (a.u.)')
        plt.show()
def get_sky_calc_model(obj_coord='23 07 28.9014701064 +21 08 02.109792078',date='2023-10-15T03:25:30'):
    astropy.config.set_temp_cache(path='/home/ipa/quanz/user_accounts/jhayoz/Projects/.astropy/cache')
    
    coord_target = SkyCoord('%s:%s:%s %s:%s:%s' % tuple(obj_coord.split(' ')),unit=(u.hourangle,u.degree))
    
    ra=coord_target.ra.value
    dec=coord_target.dec.value
    
    skycalc = skycalc_ipy.SkyCalc()
    skycalc.get_almanac_data(ra=ra, dec=dec,
                             date=date,
                             update_values=True)
    skycalc["msolflux"] = 130       # [sfu] For dates after 2019-01-31
    skycalc['wres'],skycalc['wmin'],skycalc['wmax'] = 50000,1500,3000
    tbl = skycalc.get_sky_spectrum()
    wvl = tbl['lam'].data/1e3
    transm = tbl['trans'].data
    flux = tbl['flux'].data
    return wvl,transm,flux

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

def fit_spline_old(data,mask_bad,x_data=None,s=0.4,k=3,skip_edge=0):
    if x_data is None:
        x = np.arange(len(data))
    else:
        x = x_data
    if skip_edge > 0:
        spl = UnivariateSpline(x[skip_edge:-skip_edge][~mask_bad[skip_edge:-skip_edge]], data[skip_edge:-skip_edge][~mask_bad[skip_edge:-skip_edge]],k=k)
    else:
        spl = UnivariateSpline(x[~mask_bad], data[~mask_bad],k=k)
    spl.set_smoothing_factor(s)
    spline = spl(x)
    return spline

def fit_spline(data,mask_bad,x_data=None,s=0.4,k=3,skip_edge=0):
    if x_data is None:
        x = np.arange(len(data))
    else:
        x = x_data
    
    if skip_edge > 0:
        x_good = x[skip_edge:-skip_edge][~mask_bad[skip_edge:-skip_edge]]
        y_good = data[skip_edge:-skip_edge][~mask_bad[skip_edge:-skip_edge]]
    else:
        x_good = x[~mask_bad]
        y_good = data[~mask_bad]
    X_good = x_good[:,np.newaxis]
    X = x[:,np.newaxis]

    function_fit = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
    
    estimator = RANSACRegressor(estimator = function_fit, random_state = 1, min_samples = 5)
    try:
        estimator.fit(X_good,y_good)
        spline = estimator.predict(X)
    except ValueError:
        print('Value Error (fit_spline): replace spline fit with 0s')
        spline = np.zeros_like(x)

    return spline

# function to fit a linear function to the wavelength shift within a slitlet
def fit_wavelength_error_old(wvl_shift,deg=1,lim_mask=4,lim_sel=1,median=0,outlier_method='simple'):
    # wvl_shift: error in pixels
    if outlier_method=='simple':
        mask_good = np.abs(wvl_shift-median) < lim_mask
    else:
        # method == 'extended'
        med = np.nanmedian(wvl_shift)
        std = median_abs_deviation(wvl_shift-med)
        if std < 4:
            mask_good = np.abs(wvl_shift-med) < lim_mask
        else:
            mask_good = np.abs(wvl_shift-median) < lim_mask
    
    x,y=np.arange(len(wvl_shift)),wvl_shift
    if np.sum(mask_good) < 0.5*len(wvl_shift):
        y_fit=np.ones_like(mask_good)*median
        y_calib=y_fit
        mask_close = mask_good
        print('Too many bad samples, take median')
    else:
        x_good,y_good = x[mask_good],y[mask_good]
        pols = np.polyfit(x_good,y_good,deg=deg)
        p = np.poly1d(pols)
        y_fit = p(x)
        mask_close = np.abs(wvl_shift-y_fit) < lim_sel
        y_calib = np.where(mask_close,wvl_shift,y_fit)
    return y_fit,y_calib,~mask_close

# function to fit a linear function to the wavelength shift within a slitlet. New version works with RANSAC
def fit_wavelength_error(wvl_shift,deg=1,lim_mask=4,lim_sel=1,median=0,outlier_method='simple'):
    # wvl_shift: error in pixels
    if outlier_method=='simple':
        mask_good = np.abs(wvl_shift-median) < lim_mask
    else:
        # method == 'extended'
        med = np.nanmedian(wvl_shift)
        std = median_abs_deviation(wvl_shift-med)
        if std < 4:
            mask_good = np.abs(wvl_shift-med) < lim_mask
        else:
            mask_good = np.abs(wvl_shift-median) < lim_mask
    
    x,y=np.arange(len(wvl_shift)),wvl_shift
    if np.sum(mask_good) < 0.5*len(wvl_shift):
        y_fit=np.ones_like(mask_good)*median
        y_calib=y_fit
        mask_close = mask_good
        print('Too many bad samples, take median')
    else:
        x_good,y_good = x[mask_good],y[mask_good]
        X_good = x_good[:,np.newaxis]
        X = x[:,np.newaxis]
        
        if deg == 1:
            function_fit = LinearRegression()
        else:
            function_fit = make_pipeline(PolynomialFeatures(degree=deg),Ridge(alpha=1e-1))
        
        estimator = RANSACRegressor(estimator = function_fit,random_state=1,min_samples=deg + 1)
        try:
            estimator.fit(X_good,y_good)
            y_fit = estimator.predict(X)
        except ValueError:
            print('Value Error (fit_wavelength_error): replaced with 0s')
        
        mask_close = np.abs(wvl_shift-y_fit) < lim_sel
        y_calib = np.where(mask_close,wvl_shift,y_fit)
    return y_fit,y_calib,~mask_close

def _xcor_spline_wavelength_solution(
    wlen,spectrum_cr,
    tellurics_wlen,tellurics_transm_cr,
    filter_sigma=60,
    accuracy = 20,
    spline_order = 2,spline_smoothing = 0.4,
    window_size = 160,window_shift_ratio=4,
    plot=False
):
    mean_wvl_step = np.mean(wlen[1:]-wlen[:-1])
    
    # # rebin tellurics
    # tellurics_wlen_rebin,tellurics_transm_rebin = rebin(tellurics_wlen,tellurics_transm,wlen,flux_err = None, method='datalike')
    # 
    # # Remove continuum
    # spectrum_smooth = gaussian_filter(spectrum,sigma=filter_sigma)
    # spectrum_cr = spectrum - spectrum_smooth
# 
    # tellurics_smooth = gaussian_filter(tellurics_transm_rebin,sigma=filter_sigma)
    # tellurics_transm_cr = tellurics_transm_rebin - tellurics_smooth
    
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(wlen,spectrum_cr/np.std(spectrum_cr))
        plt.plot(wlen,tellurics_transm_cr/np.std(tellurics_transm_cr))
        plt.show()
    
    # before doing the rolling, just determine a shift
    
    pcc_shift,pcc_error,pcc_phasediff = phase_cross_correlation(
                spectrum_cr,
                tellurics_transm_cr,
                normalization=None,
                upsample_factor=accuracy,
                overlap_ratio=0.5)
    
    shift_wvl_init = pcc_shift.item()
    
    wlen_corr_init = wlen - shift_wvl_init*mean_wvl_step
    
    # define the intervals for the rolling cross-correlation
    window_shift = window_size//window_shift_ratio
    nb_intervals = (len(wlen) - window_size + 1)//window_shift # can't fit all windows at the end of the wavelength array
    
    shift_wvl_px = np.zeros((nb_intervals))
    for itv_i in np.arange(nb_intervals):
        
        tellurics_transm_interv = tellurics_transm_cr[itv_i*window_shift:itv_i*window_shift + window_size]
        spectrum_cr_interv = spectrum_cr[itv_i*window_shift:itv_i*window_shift + window_size]
        
        pcc_shift,pcc_error,pcc_phasediff = phase_cross_correlation(
                    spectrum_cr_interv,
                    tellurics_transm_interv,
                    normalization=None,
                    upsample_factor=accuracy,
                    overlap_ratio=0.5)
        shift_wvl_px[itv_i] = pcc_shift.item()
    
    shift_wvl_px_position = np.array([i*window_shift + window_size//2 for i in np.arange(nb_intervals)])
    mask_bad = np.abs(shift_wvl_px-np.median(shift_wvl_init)) > 1.5

    xs = np.arange(len(wlen))
    
    if np.sum(~mask_bad) < 2*nb_intervals/3:
        print('Warning: skipping spline fitting because too few trustworthy samples')
        shift_wvl_spline = np.zeros((len(xs)))
    else:
        x_good = shift_wvl_px_position[~mask_bad]
        y_good = shift_wvl_px[~mask_bad]
        X_good = x_good[:,np.newaxis]
        X = xs[:,np.newaxis]
        
        # old version without RANSAC
        # spl = UnivariateSpline(shift_wvl_px_position[~mask_bad], shift_wvl_px[~mask_bad],k=spline_order,ext=3)
        # spl.set_smoothing_factor(spline_smoothing)
        # 
        # shift_wvl_spline = spl(xs)
        
        function_fit = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
        
        estimator = RANSACRegressor(estimator = function_fit, random_state = 1, min_samples = 5)
        try:
            estimator.fit(X_good,y_good)
            shift_wvl_spline = estimator.predict(X)
        except ValueError:
            print('Value Error (_xcor_spline_wavelength_solution): replace spline fit with 0s')
            shift_wvl_spline = np.zeros_like(xs)
    
    if plot:
        plt.figure()
        plt.plot(shift_wvl_px_position,shift_wvl_px)
        plt.plot(xs,shift_wvl_spline)
        plt.ylim((-5,5))
        plt.show()
    
    wlen_corrected = wlen - shift_wvl_spline*mean_wvl_step
    
    return wlen_corr_init,wlen_corrected,shift_wvl_spline,wlen[shift_wvl_px_position],shift_wvl_px

def calibrate_wavelength_frame(
    object_data,wavelength,
    tellurics_wlen,tellurics_transm,
    filter_sigma=60,
    accuracy = 10,
    spline_order = 2,spline_smoothing = 0.4,
    window_size = 120,window_shift_ratio=4,
    plot=False,
    method_slit='fov-linear', # raw, corr, median, linear, spline, fov-linear, spline-smooth, parabola
    method_high_order='spline' # 0-order, spline
):
    # get the trend within a slit, then collapse and fit a spline
    assert(len(object_data)==len(wavelength))
    
    lenwvl,lenxy = np.shape(object_data)
    
    # get bounds of defined data
    mask_nans = np.isnan(object_data)
    lower_bound = np.max(np.where(mask_nans[:lenwvl//2])[0]) + 1
    upper_bound = np.min(np.where(mask_nans[lenwvl//2:])[0]) + lenwvl//2

    assert(lower_bound < 250)
    assert(lenwvl-upper_bound < 250)
    
    
    wvl_data = wavelength[lower_bound:upper_bound]
    mean_d_wvl = np.mean(wvl_data[1:]-wvl_data[:-1])
    
    # rebin tellurics
    tellurics_wlen_rebin,tellurics_transm_rebin = rebin(tellurics_wlen,tellurics_transm,wvl_data,flux_err = None, method='datalike')
    tellurics_smooth = gaussian_filter(tellurics_transm_rebin,sigma=filter_sigma)
    tellurics_transm_cr = tellurics_transm_rebin - tellurics_smooth
    
    # results
    wlen_corr_model_frame_px = np.zeros_like(object_data)
    
    # initial error
    wlen_corr_init = np.zeros((lenxy))
    print('Determine initial wavelength shift')
    # for ij in np.arange(lenxy):
    for slit_i in np.arange(32):
        for col_j in np.arange(64):
            ij = slit_i*64 + col_j
            print('Progress %.2f' % ((ij+1)/lenxy*100),end='\r')
            
            # spectrum_data = object_data[lower_bound:upper_bound,ij]
            # take 3 adjacent columns instead of just 1 column
            low_j = max([0,col_j-1])
            high_j = min([63,col_j + 2])
            sel_data = object_data[lower_bound:upper_bound,slit_i*64 + low_j:slit_i*64 + high_j]
            
            spectrum_data = np.nanmean(sel_data,axis=1)
            
            # Remove continuum
            spectrum_smooth = gaussian_filter(spectrum_data,sigma=filter_sigma)
            spectrum_cr = spectrum_data - spectrum_smooth
    
            # cross-correlation
            pcc_shift,pcc_error,pcc_phasediff = phase_cross_correlation(
                        spectrum_cr,
                        tellurics_transm_cr,
                        normalization=None,
                        upsample_factor=accuracy,
                        overlap_ratio=0.5)
            wlen_corr_init[ij] = pcc_shift.item()
    print('')
    print('Finished')
    # plt.figure()
    # plt.plot(wlen_corr_init)
    # plt.ylim((-15,15))
    # plt.show()
    # if method == '0-order':
    #     wlen_corr_model_frame_px[:,:] = np.tile(wlen_corr_init,(lenwvl,1))
    #     
    median_wlen_corr = np.nanmedian(wlen_corr_init)
    print('Median wlen shift: %.2f' % median_wlen_corr)
    # if method == '0-order-median':
    #     wlen_corr_model_frame_px[:,:] = np.tile(median_wlen_corr,(lenwvl,lenxy))
        
    # if method in ['0-order-linear','0-order-spline','spline']:
    wlen_corr_init_slit = wlen_corr_init.reshape((32,64))
    # slit by slit
    wlen_slit_shift = np.zeros_like(wlen_corr_init_slit)
    wlen_slit_spline = np.zeros((lenwvl,32))
    wlen_corr_model = np.zeros((lenwvl,32,64))
    print('Model the wavelength error across the slits')
    for slit_i in np.arange(32):
        print('Progress %.2f' % ((slit_i+1)/32*100),end='\r')
        
        mask_bad = np.abs(wlen_corr_init_slit[slit_i]-median_wlen_corr) > 2
        # decide if the slit has enough signal: if more than half are bad
        if np.sum(mask_bad) > 20:
            # wlen_corr_model[:,slit_i,:] = wavelength[:,np.newaxis] - median_wlen_corr*mean_d_wvl
            wlen_corr_model_frame_px[:,slit_i*64:(slit_i+1)*64] = median_wlen_corr
            wlen_slit_shift[slit_i,:] = median_wlen_corr
            print('')
            print('Using median',median_wlen_corr)
            print('')
        
        # if the slit has enough signal, continue with spline model
        y_fit,y_calib,mask_bad = fit_wavelength_error(wlen_corr_init_slit[slit_i],deg=1,lim_mask=5,lim_sel=3,median=median_wlen_corr,outlier_method='extended')
        if np.sum(mask_bad) > 40:
            wlen_slit_shift[slit_i,:] = median_wlen_corr
            print('Replacing with median wvl corr.')
            continue
        if method_slit == 'raw':
            wlen_slit_shift[slit_i,:] = wlen_corr_init_slit[slit_i]
        elif method_slit == 'corr':
            wlen_slit_shift[slit_i,:] = y_calib
        elif method_slit == 'median':
            wlen_slit_shift[slit_i,:] = median_wlen_corr
        elif 'linear' in method_slit:
            wlen_slit_shift[slit_i,:] = y_fit
        elif method_slit == 'parabola':
            y_fit_2,y_calib_2,mask_bad_2 = fit_wavelength_error(wlen_corr_init_slit[slit_i],deg=2,lim_mask=5,lim_sel=3,median=median_wlen_corr,outlier_method='extended')
            wlen_slit_shift[slit_i,:] = y_fit_2
        elif 'spline' in method_slit:
            if 'smooth' in method_slit:
                y_smooth = gaussian_filter(y_calib,sigma=3)
            else:
                y_smooth = y_calib
            y_spline = fit_spline(y_smooth,mask_bad = mask_bad,s=spline_smoothing,k=2,skip_edge=4)
            wlen_slit_shift[slit_i,:] = y_spline
        if False:
            plt.figure()
            plt.plot(wlen_corr_init_slit[slit_i],color='k',label='Initial measurement')
            plt.plot(wlen_slit_shift[slit_i,:],color='r',label='Model')
            plt.plot(y_calib,color='g',label='y_calib')
            plt.plot(y_fit,color='b',label='y_fit')
            plt.legend()
            plt.ylim((np.min(wlen_slit_shift)-1,np.max(wlen_slit_shift)+1))
            plt.show()
    
    if method_slit == 'fov-linear':
        wlen_slit_shift_resh = wlen_slit_shift.reshape((-1))
        wlen_slit_shift_diff = wlen_slit_shift_resh[1:]-wlen_slit_shift_resh[:-1]
        wlen_slit_shift_diff = np.append(wlen_slit_shift_diff,wlen_slit_shift_diff[-1])
        wlen_slit_shift_diff_crop = wlen_slit_shift_diff.reshape((32,64))[:,1:-1]
        wlen_slit_shift_diff_crop_flat = np.hstack([
            wlen_slit_shift_diff_crop[:,0].reshape((-1,1)),
            wlen_slit_shift_diff_crop,
            wlen_slit_shift_diff_crop[:,-1].reshape((-1,1))
        ]).reshape((-1))
        wlen_slit_shift_diff_crop_line,_,_ = fit_wavelength_error(wlen_slit_shift_diff_crop_flat,deg=3,lim_mask=4,lim_sel=1,median=0)
        model_slit = np.cumsum(wlen_slit_shift_diff_crop_line)
        
        new_wlen_slit_shift = np.zeros_like(wlen_slit_shift)
        for slit_i in np.arange(32):
            popt,pcov = curve_fit(lambda x,b: model_slit[slit_i*64:(slit_i+1)*64]+b,xdata=np.arange(64),ydata=wlen_slit_shift_resh[slit_i*64:(slit_i+1)*64])
            new_wlen_slit_shift[slit_i,:] = model_slit[slit_i*64:(slit_i+1)*64]+popt[0]
        # wlen_slit_shift = new_wlen_slit_shift
    else:
        new_wlen_slit_shift = wlen_slit_shift
    
    print('Determine high-order correction in each slit')
    shift_wvl_px_position_list = []
    shift_wvl_px_list = []
    for slit_i in np.arange(32):
        print('Progress %.2f' % ((slit_i+1)/32*100),end='\r')
        # only continue if we consider higher-order corrections
        if method_high_order == '0-order':
            wlen_corr_model_frame_px[:,slit_i*64:(slit_i+1)*64] = new_wlen_slit_shift[slit_i,:]
            continue
        
        # collapse data within the slit
        wavelength_slit_corr = wvl_data[np.newaxis,:] - new_wlen_slit_shift[slit_i,:,np.newaxis] * mean_d_wvl
        interp_spectra = np.zeros((64,lenwvl))
        for col_i in np.arange(64):
            # spectrum_data = object_data[lower_bound:upper_bound,col_i]
            spectrum_data = object_data[lower_bound:upper_bound,slit_i*64+col_i]
            
            interp_spectra[col_i,:] = interp1d(x=wavelength_slit_corr[col_i],y=spectrum_data,bounds_error=False)(wavelength)
        
        # collapsed_spectrum = np.mean(interp_spectra,axis=0)
        
        # combine the spectra such that the flux is the same in each column
        # mean_std = np.nanstd(interp_spectra,axis=1)
        # norm_flux = interp_spectra/np.where(mean_std == 0, 1, mean_std)[:,np.newaxis]
        # collapsed_spectrum = np.nanmean((norm_flux - np.nanmean(norm_flux,axis=1)[:,np.newaxis]),axis=0)
        # actually no revert back to no normalising the flux
        collapsed_spectrum = np.mean(interp_spectra,axis=0)
        collapsed_spectrum_mask_nans = np.isnan(collapsed_spectrum)
        wvl_collapsed,collapsed_spectrum_sel = wavelength[~collapsed_spectrum_mask_nans],collapsed_spectrum[~collapsed_spectrum_mask_nans]
        
        # Remove continuum
        collapsed_spectrum_smooth = gaussian_filter(collapsed_spectrum_sel,sigma=filter_sigma)
        collapsed_spectrum_cr = collapsed_spectrum_sel - collapsed_spectrum_smooth

        # rebin tellurics
        tellurics_wlen_rebin,tellurics_transm_rebin = rebin(tellurics_wlen,tellurics_transm,wvl_collapsed,flux_err = None, method='datalike')
        tellurics_smooth = gaussian_filter(tellurics_transm_rebin,sigma=filter_sigma)
        tellurics_transm_cr = tellurics_transm_rebin - tellurics_smooth
        
        # Rolling cross-correlation
        _,_,shift_wvl_spline,shift_wvl_px_position,shift_wvl_px = _xcor_spline_wavelength_solution(
            wlen=wvl_collapsed,spectrum_cr=collapsed_spectrum_cr,
            tellurics_wlen=tellurics_wlen_rebin,tellurics_transm_cr=tellurics_transm_cr,
            filter_sigma=filter_sigma,
            accuracy = accuracy,
            spline_order = spline_order,spline_smoothing = spline_smoothing,
            window_size = window_size,window_shift_ratio=window_shift_ratio,
            plot=False
        )
        shift_wvl_px_position_list += [shift_wvl_px_position]
        shift_wvl_px_list += [shift_wvl_px]
        # extend the correction to the whole slit, i.e. at the borders
        slit_lower_bound = np.max(np.where(collapsed_spectrum_mask_nans[:lenwvl//2])[0]) + 1
        slit_upper_bound = np.min(np.where(collapsed_spectrum_mask_nans[lenwvl//2:])[0]) + lenwvl//2
        
        wlen_slit_spline[~collapsed_spectrum_mask_nans,slit_i] = shift_wvl_spline
        wlen_slit_spline[:slit_lower_bound,slit_i] = shift_wvl_spline[0]
        wlen_slit_spline[slit_upper_bound:,slit_i] = shift_wvl_spline[-1]
        
        # define the correction within the bounds of the collapsed spectrum
        full_correction_slit = new_wlen_slit_shift[slit_i,:][np.newaxis,:] + wlen_slit_spline[:,slit_i][:,np.newaxis]
        wlen_corr_model_frame_px[:,slit_i*64:(slit_i+1)*64] = full_correction_slit
        
        # apply the correction
        # wlen_corr_model[:,slit_i,:] = wavelength[:,np.newaxis] - full_correction_slit*mean_d_wvl
    print('')
    print('Finished')
    # reshape the corrected wavelength axis
    # wlen_corr_model_frame = wlen_corr_model.reshape((lenwvl,-1))
    # apply the correction
    wlen_corr_model_frame = wlen_corr_model_frame_px*mean_d_wvl
    
    if plot:
        print('Plotting')
        markevery=2
        mask_bad = np.abs(wlen_corr_init-new_wlen_slit_shift.reshape((-1))) > 1
        col_nb = np.arange(len(wlen_corr_init))
        # plot the shift over the columns
        plt.figure(figsize=(10,5))
        meas_err = plt.errorbar(x=col_nb[~mask_bad],y=wlen_corr_init[~mask_bad],yerr=1/accuracy,
                                color='k',fmt='|',markersize=0.5,elinewidth=0.5,capthick=0.75,capsize=1.25,
                                errorevery=markevery,markevery=markevery,label='Measured error')
        outliers =plt.scatter(col_nb[mask_bad],wlen_corr_init.reshape((-1))[mask_bad],s=5,color='r',marker='o',label='Outliers')
        #plt.plot(col_nb[~mask_bad],wlen_corr_init[~mask_bad],color='k',alpha=0.2)
        initial_fit = plt.plot(col_nb,wlen_slit_shift.reshape((-1)),label='Initial linear fit')
        third_fit = plt.plot(col_nb,new_wlen_slit_shift.reshape((-1)),label='Model')
        median = plt.axhline(median_wlen_corr,color='r',label=r'Median error: %.2f px' % median_wlen_corr)
        
        # y-axis
        if method_slit == 'median':
            plt.ylim((median_wlen_corr-2,median_wlen_corr+2))
        else:
            plt.ylim((np.min(wlen_slit_shift),np.max(wlen_slit_shift)))
        
        # x-axis
        xticks_pos = np.arange(0,2048,64) + 64/2
        plt.xticks(ticks=xticks_pos,labels=slits_ordering)
        plt.xticks(minor=True,ticks=xticks_pos-64/2,labels=slits_ordering)
        plt.tick_params(axis='x',which='major',pad=5,length=10)
        plt.tick_params(axis='x',which='minor',bottom=False,labelbottom=False)
        plt.grid(axis='x',which='minor',alpha=1)
        plt.xlim((0,2048))
        
        plt.title('Wavelength shift over the frame')
        plt.ylabel('Correction [px]')
        plt.xlabel('Slit number')
        plt.legend()
        
        # duplicate y-axis
        ylims = plt.gca().get_ylim()
        ax2 = plt.twinx()
        ylims_nm = np.array(ylims)*mean_d_wvl*1e3
        ax2.set_ylim(ylims_nm)
        ax2.set_ylabel('Wavelength error [nm]')
        
        plt.show()
        
        # plot the splines
        if method_high_order == 'spline':
            plt.figure(figsize=(10,5))
            for slit_i in np.arange(32):
                plt.plot(wavelength,wlen_slit_spline[:,slit_i],color=mpl.colormaps['viridis'](slit_i/32))
                plt.scatter(x=shift_wvl_px_position_list[slit_i],y=shift_wvl_px_list[slit_i],s=2,color=mpl.colormaps['viridis'](slit_i/32))
            plt.title('Wavelength spline fit in each slit')
            plt.ylabel('Correction [px]')
            plt.xlabel(r'Wavelength [$\mu$m]')
            med = np.median(shift_wvl_px_list[slit_i])
            plt.ylim((med-1.5,med+1.5))
            plt.show()
        
        # plot the wavelength correction
        plt.figure(figsize=(10,10))
        #img = (wavelength[:,np.newaxis]-wlen_corr_model_frame)/(mean_d_wvl)
        img = wlen_corr_model_frame_px
        plt.imshow(img,origin='lower')
        plt.title('Wavelength correction')
        plt.colorbar()
        plt.show()
    return wlen_corr_model_frame,wlen_corr_model_frame_px