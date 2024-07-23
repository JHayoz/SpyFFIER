import numpy as np
from spectres import spectres
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from astropy.coordinates import SkyCoord
import astropy.units as u
import skycalc_ipy
from scipy.interpolate import UnivariateSpline,interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def fit_spline(data,mask_bad,s=0.4,k=3):
    x = np.arange(len(data))
    spl = UnivariateSpline(x[~mask_bad], data[~mask_bad],k=k)
    spl.set_smoothing_factor(s)
    spline = spl(x)
    return spline

# function to fit a linear function to the wavelength shift within a slitlet
def fit_wavelength_error(wvl_shift,deg=1,lim_mask=4,lim_sel=1,median=0):
    # wvl_shift: error in pixels
    
    mask_good = np.abs(wvl_shift-median) < lim_mask
    
    x,y=np.arange(len(wvl_shift)),wvl_shift
    if np.sum(mask_good) < 0.5*len(wvl_shift):
        y_fit=np.ones_like(mask_good)*median
        y_calib=y_fit
    else:
        x_good,y_good = x[mask_good],y[mask_good]
        pols = np.polyfit(x_good,y_good,deg=deg)
        p = np.poly1d(pols)
        y_fit = p(x)
        mask_close = np.abs(wvl_shift-y_fit) < lim_sel
        y_calib = np.where(mask_close,wvl_shift,y_fit)
    return y_fit,y_calib

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
    shift_wvl_init, _, _ = phase_cross_correlation(
                spectrum_cr,
                tellurics_transm_cr,
                normalization=None,
                upsample_factor=accuracy,
                overlap_ratio=0.5)
    
    wlen_corr_init = wlen - shift_wvl_init*mean_wvl_step
    
    # define the intervals for the rolling cross-correlation
    window_shift = window_size//window_shift_ratio
    nb_intervals = len(wlen)//window_shift
    
    shift_wvl_px = np.zeros((nb_intervals))
    for itv_i in np.arange(nb_intervals):
        
        tellurics_transm_interv = tellurics_transm_cr[itv_i*window_shift:itv_i*window_shift + window_size]
        spectrum_cr_interv = spectrum_cr[itv_i*window_shift:itv_i*window_shift + window_size]
        
        shift_wvl_px[itv_i], _, _ = phase_cross_correlation(
                    spectrum_cr_interv,
                    tellurics_transm_interv,
                    normalization=None,
                    upsample_factor=accuracy,
                    overlap_ratio=0.5)
    
    shift_wvl_px_position = np.array([(i+1)*window_shift for i in np.arange(nb_intervals)])
    mask_bad = np.abs(shift_wvl_px-np.median(shift_wvl_px)) > 10

    xs = np.arange(len(wlen))
    
    if np.sum(~mask_bad) < nb_intervals/2:
        print('Warning: skipping spline fitting because too few trustworthy samples')
        shift_wvl_spline = np.zeros((len(xs)))
    else:
        spl = UnivariateSpline(shift_wvl_px_position[~mask_bad], shift_wvl_px[~mask_bad],k=spline_order)
        spl.set_smoothing_factor(spline_smoothing)
        
        shift_wvl_spline = spl(xs)
    
    if plot:
        plt.figure()
        plt.plot(shift_wvl_px_position,shift_wvl_px)
        plt.plot(xs,shift_wvl_spline)
        plt.ylim((-5,5))
        plt.show()
    
    wlen_corrected = wlen - shift_wvl_spline*mean_wvl_step
    
    return wlen_corr_init,wlen_corrected,shift_wvl_spline

def calibrate_wavelength_frame(
    object_data,wavelength,
    tellurics_wlen,tellurics_transm,
    filter_sigma=60,
    accuracy = 10,
    spline_order = 2,spline_smoothing = 0.4,
    window_size = 120,window_shift_ratio=4,
    plot=False,method='spline' # '0-order','0-order-linear','0-order-median', '0-order-spline','spline'
):
    
    assert(len(object_data)==len(wavelength))
    
    lenwvl,lenxy = np.shape(object_data)
    
    # get bounds of defined data
    mask_nans = np.isnan(object_data)
    lower_bound = np.max(np.where(mask_nans[:lenwvl//2])[0]) + 1
    upper_bound = np.min(np.where(mask_nans[lenwvl//2:])[0]) + lenwvl//2

    assert(lower_bound < 250)
    assert(lenwvl-upper_bound < 250)
    
    # get the trend within a slit, then collapse and fit a spline
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
    for ij in np.arange(lenxy):
        print('Progress %.2f' % (ij/lenxy*100),end='\r')
        
        spectrum_data = object_data[lower_bound:upper_bound,ij]
        
        # Remove continuum
        spectrum_smooth = gaussian_filter(spectrum_data,sigma=filter_sigma)
        spectrum_cr = spectrum_data - spectrum_smooth

        # cross-correlation
        wlen_corr_init[ij], _, _ = phase_cross_correlation(
                    spectrum_cr,
                    tellurics_transm_cr,
                    normalization=None,
                    upsample_factor=accuracy,
                    overlap_ratio=0.5)
    print('')
    print('Finished')
    if method == '0-order':
        wlen_corr_model_frame_px[:,:] = np.tile(wlen_corr_init,(lenwvl,1))
        
    median_wlen_corr = np.median(wlen_corr_init)
    print('Median wlen shift: %.2f' % median_wlen_corr)
    if method == '0-order-median':
        wlen_corr_model_frame_px[:,:] = np.tile(median_wlen_corr,(lenwvl,lenxy))
        
    if method in ['0-order-linear','0-order-spline','spline']:
        wlen_corr_init_slit = wlen_corr_init.reshape((32,64))
        # slit by slit
        wlen_slit_shift = np.zeros_like(wlen_corr_init_slit)
        wlen_slit_spline = np.zeros((lenwvl,32))
        wlen_corr_model = np.zeros((lenwvl,32,64))
        print('Determine finer correction in each slit')
        for slit_i in np.arange(32):
            print('Progress %.2f' % (slit_i/32*100),end='\r')
            
            mask_bad = np.abs(wlen_corr_init_slit[slit_i]-median_wlen_corr) > 2
            # decide if the slit has enough signal: if more than half are bad
            if np.sum(mask_bad) > 20:
                # wlen_corr_model[:,slit_i,:] = wavelength[:,np.newaxis] - median_wlen_corr*mean_d_wvl
                wlen_corr_model_frame_px[:,slit_i*64:(slit_i+1)*64] = median_wlen_corr
                wlen_slit_shift[slit_i,:] = median_wlen_corr
                print('')
                print('Using median',median_wlen_corr)
                print('')
                continue
            # if the slit has enough signal, continue with spline model
            y_fit,y_calib = fit_wavelength_error(wlen_corr_init_slit[slit_i],deg=1,lim_mask=4,lim_sel=1,median=median_wlen_corr)
            if method == '0-order-linear':
                wlen_corr_model_frame_px[:,slit_i*64:(slit_i+1)*64] = y_calib
                wlen_slit_shift[slit_i,:] = y_calib
                continue
            y_spline = fit_spline(y_calib,mask_bad = np.zeros_like(y_calib,dtype=bool),s=0.1,k=3)
            if method == '0-order-spline':
                wlen_corr_model_frame_px[:,slit_i*64:(slit_i+1)*64] = y_spline
                wlen_slit_shift[slit_i,:] = y_spline
                continue
            wlen_slit_shift[slit_i,:] = y_spline
            
            # collapse data within the slit
            wavelength_slit_corr = wvl_data[np.newaxis,:] - y_spline[:,np.newaxis] * mean_d_wvl
            interp_spectra = np.zeros((64,lenwvl))
            for col_i in np.arange(64):
                spectrum_data = object_data[lower_bound:upper_bound,col_i]
                
                interp_spectra[col_i,:] = interp1d(x=wavelength_slit_corr[col_i],y=spectrum_data,bounds_error=False)(wavelength)
            
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
            _,_,shift_wvl_spline = _xcor_spline_wavelength_solution(
                wlen=wvl_collapsed,spectrum_cr=collapsed_spectrum_cr,
                tellurics_wlen=tellurics_wlen_rebin,tellurics_transm_cr=tellurics_transm_cr,
                filter_sigma=60,
                accuracy = 20,
                spline_order = 2,spline_smoothing = 0.4,
                window_size = 120,window_shift_ratio=4,
                plot=False
            )
            
            # extend the correction to the whole slit, i.e. at the borders
            slit_lower_bound = np.max(np.where(collapsed_spectrum_mask_nans[:lenwvl//2])[0]) + 1
            slit_upper_bound = np.min(np.where(collapsed_spectrum_mask_nans[lenwvl//2:])[0]) + lenwvl//2
            
            wlen_slit_spline[~collapsed_spectrum_mask_nans,slit_i] = shift_wvl_spline
            wlen_slit_spline[:slit_lower_bound,slit_i] = shift_wvl_spline[0]
            wlen_slit_spline[slit_upper_bound:,slit_i] = shift_wvl_spline[-1]
            
            # define the correction within the bounds of the collapsed spectrum
            full_correction_slit = wlen_slit_shift[slit_i,:][np.newaxis,:] + wlen_slit_spline[:,slit_i][:,np.newaxis]
            wlen_corr_model_frame_px[:,slit_i*64:(slit_i+1)*64] = full_correction_slit
            
            # apply the correction
            # wlen_corr_model[:,slit_i,:] = wavelength[:,np.newaxis] - full_correction_slit*mean_d_wvl
    print('')
    print('Finished')
    # reshape the corrected wavelength axis
    # wlen_corr_model_frame = wlen_corr_model.reshape((lenwvl,-1))
    # apply the correction
    wlen_corr_model_frame = wavelength[:,np.newaxis] - wlen_corr_model_frame_px*mean_d_wvl
    
    if plot:
        print('Plotting')
        # plot the shift over the columns
        plt.figure(figsize=(10,5))
        plt.plot(wlen_corr_init,lw=0.3,label='Initial measurement')
        if method in ['0-order-linear','0-order-spline','spline']:
            plt.plot(wlen_slit_shift.reshape((-1)),label='Spline fit')
        plt.axhline(median_wlen_corr,ls=':',label='Median measurement')
        plt.ylim((median_wlen_corr-2,median_wlen_corr+2))
        plt.title('Wavelength shift over the frame')
        plt.ylabel('Correction [px]')
        plt.xlabel('Column number')
        plt.show()
        
        # plot the splines
        if method == 'spline':
            plt.figure(figsize=(10,5))
            for slit_i in np.arange(32):
                plt.plot(wavelength,wlen_slit_spline[:,slit_i],color=mpl.colormaps['viridis'](slit_i/32))
            plt.title('Wavelength spline fit in each slit')
            plt.ylabel('Correction [px]')
            plt.xlabel(r'Wavelength [$\mu$m]')
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