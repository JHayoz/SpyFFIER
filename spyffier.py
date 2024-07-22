import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
import warnings

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import astropy.constants as const
import matplotlib as mpl
import numpy as np
import pandas as pd
from datetime import datetime

import skycalc_ipy

from astropy import time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib import pyplot as plt
from PyAstronomy.pyasl import fastRotBroad
from scipy import interpolate, ndimage, optimize, signal
from skimage.restoration import inpaint
from typeguard import typechecked
from skimage.registration import phase_cross_correlation

from utils import rebin,calibrate_wavelength_frame,fit_wavelength_error


class Pipeline:
    """
    Class for the data reduction pipeline. Each method creates a
    configuration file with default values for the `EsoRex` recipe
    that it will run. If needed, these parameters can be adjusted
    before rerunning a method.
    """

    @typechecked
    def __init__(
        self, 
        esorex_path: str = None, 
        reduction_path: Optional[str] = None, 
        raw_path: Optional[str] = None, 
        wavel_setting: Optional[str] = None
    ) -> None:
        """
        Constructor of the Pipeline class
        
        Parameters
        ----------
        esorex_path : str
            Section title.
        reduction_path : str
            Boundary character for around the section title.
        raw_path : bool
            Extra new line at the beginning.
        wavel_setting : str, None
            Optional name of the `EsoRex` recipe that is used.

        Returns
        -------
        NoneType
            None
        """

        # Absolute path of the main reduction folder

        if reduction_path is None:
            reduction_path = "./"

        self.path = Path(reduction_path)
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        self.esorex_path = Path(esorex_path).resolve()
        
        
        
        print(f"Data reduction folder: {self.path}")

        # manually set spectral setting

        self.wavel_setting = wavel_setting

        if self.wavel_setting:
            print(f"Manually set spectral setting: {self.wavel_setting}")

        # Create attributes with the file paths

        self.header_file = Path(self.path / "header.csv")
        self.excel_file = Path(self.path / "header.xlsx")
        self.json_file = Path(self.path / "files.json")

        # Read or create the CSV file with header data

        if self.header_file.is_file():
            print("Reading header data from header.csv")
            self.header_data = pd.read_csv(self.header_file)

        else:
            print("Creating header DataFrame")
            self.header_data = pd.DataFrame()

        # Read or create the JSON file with filenames for SOF

        if self.json_file.is_file():
            print("Reading filenames and labels from files.json")

            with open(self.json_file, "r", encoding="utf-8") as json_file:
                self.file_dict = json.load(json_file)

        else:
            print("Creating dictionary for filenames")
            self.file_dict = {}

        # Create directory for raw files
        if raw_path is None:
            self.raw_folder = Path(self.path / "raw")
        else:
            self.raw_folder = Path(raw_path)

        if not os.path.exists(self.raw_folder):
            os.makedirs(self.raw_folder)

        # Create directory for calibration files

        self.calib_folder = Path(self.path / "calib")

        if not os.path.exists(self.calib_folder):
            os.makedirs(self.calib_folder)

        # Create directory for product files

        self.product_folder = Path(self.path / "product")

        if not os.path.exists(self.product_folder):
            os.makedirs(self.product_folder)

        # Create directory for configuration files

        self.config_folder = Path(self.path / "config")

        if not os.path.exists(self.config_folder):
            os.makedirs(self.config_folder)

        
        # Print the available recipes for CRIRES+ and Molecfit

        esorex = [self.esorex_path, "--recipes"]

        with subprocess.Popen(
            esorex, cwd=self.config_folder, stdout=subprocess.PIPE, encoding="utf-8"
        ) as proc:
            output, _ = proc.communicate()

        print("\nAvailable EsoRex recipes for ERIS/SPIFFIER:")

        for item in output.split("\n"):
            if item.replace(" ", "")[:8] == "eris_ifu":
                print(f"   -{item}")
        
        # layout of the slitlets of ERIS
        self.slitlet_layout=[9,8,10,7,11,6,12,5,13,4,14,3,15,2,16,1,32,17,31,18,30,19,29,20,28,21,27,22,26,23,25,24]
        """"
        print("\nAvailable EsoRex recipes for Molecfit:")
        
        for item in output.split("\n"):
            if item.replace(" ", "")[:9] == "molecfit_":
                print(f"   -{item}")
        
        
        # Check if there is a new version available

        try:
            pypi_url = "https://pypi.org/pypi/pycrires/json"

            with urllib.request.urlopen(pypi_url, timeout=1.0) as open_url:
                url_content = open_url.read()
                url_data = json.loads(url_content)
                latest_version = url_data["info"]["version"]

        except (urllib.error.URLError, socket.timeout):
            latest_version = None

        if latest_version is not None and pycrires.__version__ != latest_version:
            print(f"\nA new version ({latest_version}) is available!")
            print("Want to stay informed about updates?")
            print("Please have a look at the Github page:")
            print("https://github.com/tomasstolker/pycrires")
        """

    @staticmethod
    @typechecked
    def _print_section(
        sect_title: str,
        bound_char: str = "-",
        extra_line: bool = True,
        recipe_name: Optional[str] = None,
    ) -> None:
        """
        Internal method for printing a section title.

        Parameters
        ----------
        sect_title : str
            Section title.
        bound_char : str
            Boundary character for around the section title.
        extra_line : bool
            Extra new line at the beginning.
        recipe_name : str, None
            Optional name of the `EsoRex` recipe that is used.

        Returns
        -------
        NoneType
            None
        """

        if extra_line:
            print("\n" + len(sect_title) * bound_char)
        else:
            print(len(sect_title) * bound_char)

        print(sect_title)
        print(len(sect_title) * bound_char + "\n")

        if recipe_name is not None:
            print(f"EsoRex recipe: {recipe_name}\n")

    @typechecked
    def _observation_info(
        self
    ) -> None:
        """
        Internal method for printing some details
        about the observations.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Observation details")

        check_key = {
            "OBS.TARG.NAME": "Target",
            "OBS.PROG.ID": "Program ID",
            "INS3.SPGW.NAME": "Grating wheel",
            "INS3.SPXW.NAME": "Pixel scale"
        }

        science_index = self.header_data["DPR.CATG"] == "SCIENCE"
        """
        if "RA" in self.header_data and "DEC" in self.header_data:
            ra_mean = np.mean(self.header_data["RA"][science_index])
            dec_mean = np.mean(self.header_data["DEC"][science_index])

            target_coord = SkyCoord(ra_mean, dec_mean, unit="deg", frame="icrs")

            ra_dec = target_coord.to_string("hmsdms")

            print(f"RA Dec = {ra_dec}")
        """
        for key, value in check_key.items():
            header = self.header_data[key][science_index].to_numpy()

            if isinstance(header[0], str):
                indices = np.argwhere(header)

            else:
                indices = ~np.isnan(header)

                if sum(header == 0.0) != len(header):
                    indices[header == 0.0] = False

            if np.all(header[indices] == header[indices][0]):
                print(f"{value} = {header[0]}")

            else:
                warnings.warn(
                    f"Expecting a single value for {key} but "
                    f"multiple values are found: {header}"
                )

                if isinstance(header[indices][0], np.float64):
                    print(f"{value} = {np.mean(header)}")

        if "OBS.ID" in self.header_data:
            # obs_id = self.header_data['OBS.ID']
            unique_id = pd.unique(self.header_data["OBS.ID"])

            print("\nObservation ID:")

            for item in unique_id:
                if not np.isnan(item):
                    count_files = np.sum(self.header_data["OBS.ID"] == item)

                    if count_files == 1:
                        print(f"   - {item} -> {count_files} file")
                    else:
                        print(f"   - {item} -> {count_files} files")

    @typechecked
    def _export_header(
        self
    ) -> None:
        """
        Internal method for exporting the ``DataFrame`` with header
        data to a CSV and Excel file.

        Returns
        -------
        NoneType
            None
        """

        # Sort DataFrame by the exposure ID

        self.header_data.sort_values(["DET.EXP.ID"], ascending=True, inplace=True)

        # Write DataFrame to CSV file

        print(f"Exporting DataFrame to {self.header_file.name}")

        self.header_data.to_csv(self.header_file, sep=",", header=True, index=False)

        # Write DataFrame to Excel file

        print(f"Exporting DataFrame to {self.excel_file.name}")

        self.header_data.to_excel(
            self.excel_file, sheet_name="ERIS SPIFFIER", header=True, index=False
        )

        # Read header data from CSV file to set the file indices to the sorted order

        self.header_data = pd.read_csv(self.header_file)
    
    @typechecked
    def _rename_products(
        self, 
        fits_files: list = [],
        sof_tag: str = None,
        name_extension: str = '',
        add_arcfile: bool = False
    ) -> None:
        for item in fits_files:
            # rename files
            if add_arcfile:
                hdr = fits.getheader(item)
                arcfile = hdr['ARCFILE'][:-5] # remove '.fits'
                new_name = item.parent / f'{item.name[:-5]}{name_extension}_{arcfile}.fits'
            else:
                new_name = item.parent / f'{item.name[:-5]}{name_extension}.fits'
            
            # if file already exists, delete it
            # if os.path.exists(new_name):
            #     os.remove(new_name)
            #     del self.file_dict[sof_tag][str(new_name)]
            # if file already contains the name extension, delete it
            if name_extension in item.name[:-5] and name_extension != '':
                os.remove(item)
                del self.file_dict[sof_tag][str(item)]
            else:
                os.rename(item,new_name)
                self._update_files(sof_tag, str(new_name))
                # remove old
                if new_name != item and str(item) in self.file_dict[sof_tag].keys():
                    del self.file_dict[sof_tag][str(item)]
    
    @typechecked
    def _update_files(
        self, 
        sof_tag: str, 
        file_name: str
    ) -> None:
        """
        Internal method for updating the dictionary with file
        names and related tag names for the set of files (SOF).

        Parameters
        ----------
        sof_tag : str
            Tag name of ``file_name`` for the set of files (SOF).
        file_name : str
            Absolute path of the file.

        Returns
        -------
        NoneType
            None
        """

        # Print filename and SOF tag

        file_split = file_name.split("/")

        if file_split[-2] == "raw":
            file_print = file_split[-2] + "/" + file_split[-1]
            print(f"   - {file_print} {sof_tag}")

        elif file_split[-3] == "calib":
            file_print = file_split[-3] + "/" + file_split[-2] + "/" + file_split[-1]
            print(f"   - {file_print} {sof_tag}")

        elif file_split[-2] == "product":
            file_print = file_split[-2] + "/" + file_split[-1]
            print(f"   - {file_print} {sof_tag}")

        else:
            print(f"   - {file_name} {sof_tag}")

        # Get FITS header

        if file_name.endswith(".fits"):
            header = fits.getheader(file_name)
        else:
            header = None

        file_dict = {}

        if header is not None and "ESO DET SEQ1 DIT" in header:
            """
            if sof_tag in ["CAL_DARK_MASTER", "CAL_DARK_BPM"]:
                # Use DIT from filename because of issue with
                # cr2res_cal_dark recipe which does not copy
                # the correct header from raw to master dark
                # when processing raw dark with multiple DIT
                file_tmp = file_name.split("/")[-1]
                file_tmp = file_tmp.split("_")[-2]
                file_tmp = file_tmp.split("x")[-2]

                if "." in file_tmp:
                    decimal = len(file_tmp.split(".")[-1])

                    if decimal == 5:
                        for dark_item in self.file_dict["DARK"].values():
                            if float(file_tmp) == round(dark_item["DIT"], 5):
                                file_dict["DIT"] = dark_item["DIT"]
                                break

                else:
                    file_dict["DIT"] = float(file_tmp)

            else:
            """
            file_dict["DIT"] = header["ESO DET SEQ1 DIT"]
        else:
            file_dict["DIT"] = None

        if header is not None and "ESO INS3 SPGW NAME" in header:
            file_dict["SPGW"] = header["ESO INS3 SPGW NAME"]
        else:
            file_dict["SPGW"] = None
            
        if header is not None and "ESO INS3 SPXW NAME" in header:
            file_dict["SPXW"] = header["ESO INS3 SPXW NAME"]
        else:
            file_dict["SPXW"] = None
        
        if sof_tag in self.file_dict:
            if file_name not in self.file_dict[sof_tag]:
                self.file_dict[sof_tag][file_name] = file_dict
        else:
            self.file_dict[sof_tag] = {file_name: file_dict}
    @typechecked
    def _identify_science_dit_ndit(
        self,
        std: bool=False
    ) -> list:
        mask_science = self.header_data['DPR.CATG'] == 'SCIENCE'
        if std:
            mask_science = np.logical_or(self.header_data['DPR.TYPE'] == 'STD',self.header_data['DPR.TYPE'] == 'SKY,STD')
        dit_ndit = self.header_data[mask_science][['DET.SEQ1.DIT','DET.NDIT']].values
        unique_dit_ndit = list(dict.fromkeys(list(map(tuple,dit_ndit))))
        return unique_dit_ndit
        
    @typechecked
    def _identify_obj_sky_groups_by_time(
        self,
        obj_tag: str = 'OBJECT',
        sky_tag: str = 'SKY',
        dit: float = None,
        ndit: int = None,
        spiffier_gw: str = None, 
        spiffier_psw: str = None
    ) -> dict:
        if None in [dit,spiffier_gw,spiffier_psw]:
            raise RuntimeError(
                    f"You need to provide all following parameters: DIT, NDIT, SPGW, SPXW"
                )
        # match by dit, ndit, spgw, spxw
        mask_select = np.ones(len(self.header_data),dtype=bool)
        mask_select = np.logical_and(mask_select,self.header_data['DET.SEQ1.DIT'] == dit)
        if not ndit is None:
            mask_select = np.logical_and(mask_select,self.header_data['DET.NDIT'] == ndit)
        mask_select = np.logical_and(mask_select,self.header_data['INS3.SPGW.NAME'] == spiffier_gw)
        mask_select = np.logical_and(mask_select,self.header_data['INS3.SPXW.NAME'] == spiffier_psw)
        mask_obj_sky = np.logical_or(self.header_data['DPR.TYPE'] == obj_tag,self.header_data['DPR.TYPE'] == sky_tag)
        mask_select = np.logical_and(mask_select,mask_obj_sky)
        nb_sky = np.sum(self.header_data[mask_select]['DPR.TYPE'] == sky_tag)
        nb_object = np.sum(self.header_data[mask_select]['DPR.TYPE'] == obj_tag)
        print(f'{dit,ndit,spiffier_gw,spiffier_psw} # OBJECT: {nb_object}, # SKY: {nb_sky}')

        if nb_sky == 0:
            files_groups = self.select_object_groups(dit=dit)
            return files_groups
        
        sky_groups = {}
        is_in_sky_groups=False
        sky_group_id = 0
        for date,type in self.header_data[mask_select].sort_values(by='DATE-OBS')[['DATE-OBS','DPR.TYPE']].values:
            if is_in_sky_groups:
                if type == sky_tag:
                    sky_groups[sky_group_id] += [[date,type]]
                else:
                    is_in_sky_groups = False
            else:
                if type == sky_tag:
                    is_in_sky_groups=True
                    sky_group_id += 1
                    sky_groups[sky_group_id] = [[date,type]]
        
        sky_groups_times = {}
        for sky_group_id in sky_groups.keys():
            if len(sky_groups[sky_group_id]) == 1:
                sky_groups_times[sky_group_id] = datetime.fromisoformat(sky_groups[sky_group_id][0][0][:23])
            else:
                t0 = datetime.fromisoformat(sky_groups[sky_group_id][-1][0][:23])
                deltat = t0-datetime.fromisoformat(sky_groups[sky_group_id][0][0][:23])
                sky_groups_times[sky_group_id] = t0 + 0.5*deltat

        mask_obj_select = np.logical_and(mask_select,self.header_data['DPR.TYPE'] == obj_tag)
        obj_sky_df = pd.DataFrame()
        obj_sky_df['ARCFILE'] = self.header_data[mask_obj_select]['ARCFILE'].values
        obj_sky_df['DATE-OBS'] = self.header_data[mask_obj_select]['DATE-OBS'].values
        
        sky_group_ids = list(sky_groups_times.keys())
        sky_groups_times_vals = [sky_groups_times[key] for key in sky_group_ids]
        for obj_i,date in enumerate(obj_sky_df['DATE-OBS'].values):
            t0 = datetime.fromisoformat(date[:23])
            argmin = np.argmin(np.abs(t0-np.array(sky_groups_times_vals)))
            sky_group_id = sky_group_ids[argmin]
            obj_sky_df.loc[obj_i,'SKY_BUNDLE_ID'] = sky_group_id
        
        files_groups = {}
        indices_groups = {}
        for sky_group_id in sky_groups.keys():
            mask_obj = obj_sky_df['SKY_BUNDLE_ID'] == sky_group_id
            object_date_obs = list(obj_sky_df[mask_obj]['DATE-OBS'].values)
            sky_date_obs = [sky_groups[sky_group_id][i][0] for i in range(len(sky_groups[sky_group_id]))]
            files_groups[sky_group_id] = {obj_tag:object_date_obs,sky_tag:sky_date_obs}
            
            indices_groups[sky_group_id] = {obj_tag:np.isin(self.header_data['DATE-OBS'],object_date_obs),sky_tag:np.isin(self.header_data['DATE-OBS'],sky_date_obs)}
        return files_groups
    
    @typechecked
    def _create_config(
        self, 
        eso_recipe: str, 
        pipeline_method: str, 
        verbose: bool
    ) -> None:
        """
        Internal method for creating a configuration file with default
        values for a specified `EsoRex` recipe. Also check if `EsorRex`
        is found and raise an error otherwise.

        Parameters
        ----------
        eso_recipe : str
            Name of the `EsoRex` recipe.
        pipeline_method : str
            Name of the ``Pipeline`` method.
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        config_file = self.config_folder / f"{pipeline_method}.rc"

        if shutil.which(self.esorex_path) is None:
            raise RuntimeError(
                "Esorex is not accessible from the command line. "
                "Please make sure that the ESO pipeline is correctly "
                "installed and included in the PATH variable."
            )

        if not os.path.exists(config_file):
            print()

            esorex = [
                self.esorex_path, 
                f"--create-config={config_file}", 
                eso_recipe]

            if verbose:
                stdout = None
            else:
                stdout = subprocess.DEVNULL

                print(
                    f"Creating configuration file: config/{pipeline_method}.rc",
                    end="",
                    flush=True,
                )

            subprocess.run(esorex, cwd=self.config_folder, stdout=stdout, check=True)

            # Open config file and adjust some parameters
            
            with open(config_file, "r", encoding="utf-8") as open_config:
                config_text = open_config.read()
            
            if eso_recipe == "eris_ifu_jitter":
                config_text = config_text.replace(
                    "eris.eris_ifu_jitter.aj-method=7",
                    "eris.eris_ifu_jitter.aj-method=2",
                )
                config_text = config_text.replace(
                    "eris.eris_ifu_jitter.dar-corr=FALSE",
                    "eris.eris_ifu_jitter.dar-corr=TRUE",
                )
                config_text = config_text.replace(
                    "eris.eris_ifu_jitter.flux-calibrate=FALSE",
                    "eris.eris_ifu_jitter.flux-calibrate=TRUE",
                )
            config_text = config_text.replace(
                "eris.eris_ifu_jitter.product_depth=0",
                "eris.eris_ifu_jitter.product_depth=3",
            )
            
            with open(config_file, "w", encoding="utf-8") as open_config:
                open_config.write(config_text)

            if not verbose:
                print(" [DONE]")
            
    @typechecked
    def modify_config(
        self,
        config_file, 
        new_config: dict = {},
        eso_recipe: str = 'eris_ifu_jitter'
    ) -> None:
        print(
                    f"Modifying configuration file: {config_file}",
                    end="\n",
                    flush=True,
                )
        if len(new_config.keys()) == 0:
            raise RuntimeError(
                "The new config is empty"
                )
        
        with open(config_file, "r", encoding="utf-8") as open_config:
            config_text = open_config.read()
        
        split_config_text = config_text.split('\n')
        
        new_config_text = []
        
        for line in split_config_text:
            if '=' not in line:
                new_config_text += [line]
                continue
            found_item = False
            for key,item in new_config.items():
                config_line = f'eris.{eso_recipe}.{key}'
                line_equ_ind = line.index('=')
                if config_line == line[:line_equ_ind]:
                    new_config_line = f'eris.{eso_recipe}.{key}={item}'
                    new_config_text += [new_config_line]
                    print(f'Old parameter: {line}')
                    print(f'New parameter: {new_config_line}')
                    found_item = True
                    break
            if not found_item:
                new_config_text += [line]
                    
        
        new_config_item = '\n'.join(new_config_text)
        
        with open(config_file, "w", encoding="utf-8") as open_config:
            open_config.write(new_config_item)

        print(" [DONE]")
        
    @typechecked
    def extract_header(
        self
    ) -> None:
        """
        Method for extracting relevant header data from the FITS files
        and storing these in a ``DataFrame``. The data will also be
        exported to a CSV and Excel file.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Extracting FITS headers")

        # Create a new DataFrame
        self.header_data = pd.DataFrame()
        print("Creating new DataFrame...\n")

        key_file = os.path.dirname(__file__) + "/keywords.txt"
        keywords = np.genfromtxt(key_file, dtype="str", delimiter=",")

        raw_files = Path(self.raw_folder).glob("*.fits")
        header_dict = {}
        for key_item in keywords:
            header_dict[key_item] = []

        for file_item in raw_files:
            header = fits.getheader(file_item)

            for key_item in keywords:
                if key_item in header:
                    header_dict[key_item].append(header[key_item])
                else:
                    header_dict[key_item].append(None)

        for key_item in keywords:
            column_name = key_item.replace(" ", ".")
            column_name = column_name.replace("ESO.", "")

            self.header_data[column_name] = header_dict[key_item]

        self._export_header()

        indices = np.where(self.header_data["DPR.CATG"] == "SCIENCE")[0]

        if len(indices) > 0:
            self._observation_info()
        else:
            warnings.warn(
                "Could not find any DPR.CATG=SCIENCE data "
                "so there will not be any details printed "
                "about the observations."
            )
    @typechecked
    def _plot_image(
        self, 
        file_type: str, 
        fits_folder: str, 
        save: bool = True,
        vmin_pc: float = 1,
        vmax_pc: float=99
    ) -> None:
        """
        Internal method for plotting the data of a specified file type.

        Parameters
        ----------
        file_type : str
            The file type of which the data should be plotted.
        fits_folder : str
            Folder in which to search for the FITS files of type
            ``file_type``. The argument should be specified relative
            to the main reduction folder (e.g. "calib/cal_dark" or
            "calib/util_calib_flat").

        Returns
        -------
        NoneType
            None
        """
        fits_folder_arr = fits_folder.split('/')[-2:]
        if file_type in self.file_dict:
            print(f"\nPlotting {file_type}:")

            for item in self.file_dict[file_type]:
                file_name = item.split("/")

                if f"{file_name[-3]}/{file_name[-2]}" != f"{fits_folder_arr[-2]}/{fits_folder_arr[-1]}":
                    continue

                print(f"   - calib/{file_name[-2]}/{file_name[-1][:-4]}png")

                with fits.open(item) as hdu_list:
                    plt.figure(figsize=(10, 3.5))

                    header = hdu_list[0].header

                    dit = header["HIERARCH ESO DET SEQ1 DIT"]
                    ndit = header["HIERARCH ESO DET NDIT"]

                    plt.subplot(1, 1, 1)

                    data = fits.getdata(item)
                    
                    data = np.nan_to_num(data)
                    shape = np.shape(data)
                    if len(shape) == 2:
                        img = data
                    elif len(shape) == 3:
                        img = np.mean(data,axis=0)
                    else:
                        img = data

                    vmin, vmax = np.percentile(img, (vmin_pc, vmax_pc))

                    plt.imshow(
                        img, origin="lower", cmap="afmhot", vmin=vmin, vmax=vmax
                    )
                    plt.title(f"ERIS SPIFFIER", fontsize=9)
                    plt.minorticks_on()

                    plt.suptitle(
                        f"{file_name[-1]}, {file_type}, range = [{vmin:.1f}:{vmax:.1f}], "
                        f"DIT = {dit}, NDIT = {ndit}",
                        y=0.95,
                        fontsize=10,
                    )

                    plt.tight_layout()
                    if save:
                        plt.savefig(f"{item[:-4]}png", dpi=300)
                    plt.show()
                    plt.clf()
                    plt.close()

        else:
            warnings.warn(f"Could not find {file_type} files to plot.")

    # esorex recipes
    @typechecked
    def calib_dark(
        self, 
        verbose: bool = True, 
        create_sof: bool = True, 
        dit: float = None
    ) -> None:
        """
        Method for running ``eris_ifu_dark``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.
        create_sof : bool
            Create a new SOF file. Setting the argument to ``True``
            will overwrite the SOF file if already present. Setting
            the argument to ``False`` will allow for manually
            adjusting an existing SOF file if the routine had
            already been previously executed.


        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create master DARK", recipe_name="eris_ifu_dark")

        print(f"Verbose: {verbose}")
        print(f"Create SOF: {create_sof}")

        indices = self.header_data["DPR.TYPE"] == "DARK"

        # Create output folder

        output_dir = self.calib_folder / "calib_dark"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Check unique DIT
        if dit is None:
            unique_dit = set()
            for item in self.header_data[indices]["DET.SEQ1.DIT"]:
                unique_dit.add(item)
    
            if len(unique_dit) == 0:
                print("\nUnique DIT values: none")
            elif len(unique_dit) > 1:
                print(f"\nUnique DIT values: {unique_dit}")
                print(f"\nYou should calibrate darks with different DITS separetely")
            indices_dit = indices
        else:
            print(f"\nOnly considering DIT values: {dit}")
            indices_dit = np.logical_and(self.header_data["DPR.TYPE"] == "DARK",self.header_data["DET.SEQ1.DIT"] == dit)

        # Create SOF file

        sof_file = Path(output_dir / "files.sof")

        if not create_sof and not sof_file.exists():
            warnings.warn(
                f"The SOF file is not found at '{sof_file}' "
                "while 'create_sof' is set to False. "
                "Probably 'cal_dark' has not been "
                "previously executed so forcing "
                "'create_sof' to True."
            )

            create_sof = True

        if create_sof:
            print("\nCreating SOF file:")

            with open(sof_file, "w", encoding="utf-8") as sof_open:
                for item in self.header_data[indices_dit]["ARCFILE"]:
                    sof_open.write(f"{self.raw_folder}/{item} DARK\n")
                    self._update_files("DARK", f"{self.raw_folder}/{item}")

            # Check if any dark frames were found

            if "DARK" not in self.file_dict:
                raise RuntimeError(
                    "The 'raw' folder does not contain any DPR.TYPE=DARK files."
                )

        else:
            print(f"\nFound SOF file: {sof_file}")

        # Create EsoRex configuration file if not found

        self._create_config("eris_ifu_dark", "calib_dark", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "calib_dark.rc"

        esorex = [
            self.esorex_path,
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            '--no-checksum',
            "eris_ifu_dark",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with master dark

        print("Output files:")
        name_extension = f'_{dit}s'

        fits_files = Path(self.path / "calib/calib_dark").glob(
            "eris_ifu_dark_master_dark.fits"
        )
        self._rename_products(list(fits_files),'CALIB_DARK_MASTER',name_extension = name_extension)

        # Update file dictionary with bad pixel map

        fits_files = Path(self.path / "calib/calib_dark").glob(
            "eris_ifu_dark_bpm.fits"
        )
        self._rename_products(list(fits_files),'CALIB_DARK_BPM',name_extension = name_extension)

        # Create plots

        self._plot_image("CALIB_DARK_MASTER", "calib/calib_dark")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def calib_detlin(
        self, 
        verbose: bool = True, 
        create_sof: bool = True
    ) -> None:
        """
        Method for running ``eris_ifu_detlin``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.
        create_sof : bool
            Create a new SOF file. Setting the argument to ``True``
            will overwrite the SOF file if already present. Setting
            the argument to ``False`` will allow for manually
            adjusting an existing SOF file if the routine had
            already been previously executed.


        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create DETLIN", recipe_name="eris_ifu_detlin")

        print(f"Verbose: {verbose}")
        print(f"Create SOF: {create_sof}")

        indices = np.logical_or(self.header_data["DPR.TYPE"] == "LINEARITY,LAMP,DETCHAR",self.header_data["DPR.TYPE"] == "LINEARITY,DARK,DETCHAR")

        # Create output folder

        output_dir = self.calib_folder / "calib_detlin"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Check unique DIT

        unique_dit = set()
        for item in self.header_data[indices]["DET.SEQ1.DIT"]:
            unique_dit.add(item)

        if len(unique_dit) == 0:
            print("\nUnique DIT values: none")
        else:
            print(f"\nUnique DIT values: {unique_dit}")

        # Create SOF file

        sof_file = Path(output_dir / "files.sof")

        if not create_sof and not sof_file.exists():
            warnings.warn(
                f"The SOF file is not found at '{sof_file}' "
                "while 'create_sof' is set to False. "
                "Probably 'cal_dark' has not been "
                "previously executed so forcing "
                "'create_sof' to True."
            )

            create_sof = True

        if create_sof:
            print("\nCreating SOF file:")

            with open(sof_file, "w", encoding="utf-8") as sof_open:
                for item in self.header_data[indices]["ARCFILE"]:
                    sof_open.write(f"{self.raw_folder}/{item} LINEARITY_LAMP\n")
                    self._update_files("LINEARITY_LAMP", f"{self.raw_folder}/{item}")

            # Check if any dark frames were found

            if "LINEARITY_LAMP" not in self.file_dict:
                raise RuntimeError(
                    "The 'raw' folder does not contain any DPR.TYPE=DARK files."
                )

        else:
            print(f"\nFound SOF file: {sof_file}")

        # Create EsoRex configuration file if not found

        self._create_config("eris_ifu_detlin", "calib_detlin", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "calib_detlin.rc"

        esorex = [
            self.esorex_path,
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            '--no-checksum',
            "eris_ifu_detlin",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        print("Output files:")

        # Update file dictionary with detlin bpm

        fits_files = Path(self.path / "calib/calib_detlin").glob(
            "eris_ifu_detlin_bpm.fits"
        )
        self._rename_products(list(fits_files),'CALIB_DETLIN_BPM',name_extension = '')

        # Update file dictionary with detlin bpm filt

        fits_files = Path(self.path / "calib/calib_detlin").glob(
            "eris_ifu_detlin_bpm_filt.fits"
        )
        self._rename_products(list(fits_files),'CALIB_DETLIN_BPM_FILT',name_extension = '')

        # Update file dictionary with detlin gain info

        fits_files = Path(self.path / "calib/calib_detlin").glob(
            "eris_ifu_detlin_gain_info.fits"
        )
        self._rename_products(list(fits_files),'CALIB_DETLIN_GAIN_INFO',name_extension = '')
        
        # Create plots

        # self._plot_image("CALIB_DARK_MASTER", "calib/calib_dark")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def calib_distortion(
        self, 
        verbose: bool = True, 
        create_sof: bool = True, 
        spiffier_gw: str = None, 
        spiffier_psw: str = None
    ) -> None:
        """
        Method for running ``eris_ifu_distortion``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.
        create_sof : bool
            Create a new SOF file. Setting the argument to ``True``
            will overwrite the SOF file if already present. Setting
            the argument to ``False`` will allow for manually
            adjusting an existing SOF file if the routine had
            already been previously executed.


        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create DISTORTION", recipe_name="eris_ifu_distortion")

        print(f"Verbose: {verbose}")
        print(f"Create SOF: {create_sof}")
        
        dpr_types = {
            'NS,DARK':'DARK_NS',
            'NS,SLIT':'FIBRE_NS',
            'NS,FLAT,DARK':'FLAT_NS',
            'NS,FLAT,LAMP':'FLAT_NS',
            'NS,WAVE,DARK':'WAVE_NS',
            'NS,WAVE,LAMP':'WAVE_NS'
        }
        indices_dpr_types = {}

        # match by grating wheel and plate scale
        if not spiffier_gw is None:
            indices_gw = self.header_data["INS3.SPGW.NAME"] == spiffier_gw
        else:
            indices_gw = self.header_data["INS3.SPGW.NAME"] == self.wavel_setting
        if not spiffier_psw is None:
            indices_psw = self.header_data["INS3.SPXW.NAME"] == spiffier_psw
        else:
            indices_psw = np.ones_like(indices_gw,dtype=bool)
        
        indices_gw_pws = np.logical_and(indices_gw,indices_psw)
        
        for dpr_type_i in dpr_types:
            indices_dpr_types[dpr_type_i] = np.logical_and(self.header_data["DPR.TYPE"] == dpr_type_i, indices_gw_pws)
        
        pro_catg = {
            'FIRST_WAVE_FIT':'FIRST_WAVE_FIT',
            'REF_LINE_ARC':'REF_LINE_ARC',
            'WAVE_SETUP':'WAVE_SETUP',
        }
        indices_pro_catg = {}
        for pro_catg_i in pro_catg:
            indices_pro_catg[pro_catg_i] = self.header_data["PRO.CATG"] == pro_catg_i

        # Create output folder

        output_dir = self.calib_folder / "calib_distortion"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        sof_file = Path(output_dir / "files.sof")

        if not create_sof and not sof_file.exists():
            warnings.warn(
                f"The SOF file is not found at '{sof_file}' "
                "while 'create_sof' is set to False. "
                "Probably 'cal_dark' has not been "
                "previously executed so forcing "
                "'create_sof' to True."
            )

            create_sof = True

        if create_sof:
            print("\nCreating SOF file:")

            with open(sof_file, "w", encoding="utf-8") as sof_open:
                
                for dpr_type_i in dpr_types:
                    for item in self.header_data[indices_dpr_types[dpr_type_i]]["ARCFILE"]:
                        sof_open.write(f"{self.raw_folder}/{item} {dpr_types[dpr_type_i]}\n")
                        self._update_files(dpr_types[dpr_type_i], f"{self.raw_folder}/{item}")
                
                for pro_catg_i in pro_catg:
                    for item in self.header_data[indices_pro_catg[pro_catg_i]]["ARCFILE"]:
                        sof_open.write(f"{self.raw_folder}/{item} {pro_catg[pro_catg_i]}\n")
                        self._update_files(pro_catg[pro_catg_i], f"{self.raw_folder}/{item}")

            # Check if any dark frames were found
            
            for action_dpr_type_i in list(dict.fromkeys(dpr_types.values())):
                if action_dpr_type_i not in self.file_dict:
                    raise RuntimeError(
                        f"The 'raw' folder does not contain any DPR.TYPE={action_dpr_type_i} files."
                    )
            for action_pro_catg_i in list(dict.fromkeys(pro_catg.values())):
                if action_pro_catg_i not in self.file_dict:
                    raise RuntimeError(
                        f"The 'raw' folder does not contain any PRO.CATG={action_pro_catg_i} files."
                    )

        else:
            print(f"\nFound SOF file: {sof_file}")

        # Create EsoRex configuration file if not found

        self._create_config("eris_ifu_distortion", "calib_distortion", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "calib_distortion.rc"

        esorex = [
            self.esorex_path,
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            '--no-checksum',
            "eris_ifu_distortion",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        print("Output files:")
        
        name_extension = f'_{spiffier_gw}_{spiffier_psw}'
        
        # Update file dictionary with detlin bpm

        fits_files = Path(self.path / "calib/calib_distortion").glob(
            "eris_ifu_distortion_bpm.fits"
        )
        self._rename_products(list(fits_files),'CALIB_DISTORTION_BPM',name_extension = name_extension)

        # Update file dictionary with detlin bpm filt

        fits_files = Path(self.path / "calib/calib_distortion").glob(
            "eris_ifu_distortion_slitlet_pos.fits"
        )
        self._rename_products(list(fits_files),'CALIB_DISTORTION_SLITLET_POS',name_extension = name_extension)

        # Update file dictionary with detlin gain info

        fits_files = Path(self.path / "calib/calib_distortion").glob(
            "eris_ifu_distortion_distortion.fits"
        )
        self._rename_products(list(fits_files),'CALIB_DISTORTION_DISTORTION',name_extension = name_extension)
        
        # Create plots

        # self._plot_image("CALIB_DARK_MASTER", "calib/calib_dark")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)
    
    @typechecked
    def calib_flat(
        self, 
        verbose: bool = True, 
        create_sof: bool = True, 
        spiffier_gw: str = None, 
        spiffier_psw: str = None,
        dit: float = None,
    ) -> None:
        """
        Method for running ``eris_ifu_flat``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.
        create_sof : bool
            Create a new SOF file. Setting the argument to ``True``
            will overwrite the SOF file if already present. Setting
            the argument to ``False`` will allow for manually
            adjusting an existing SOF file if the routine had
            already been previously executed.


        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create FLAT", recipe_name="eris_ifu_flat")

        print(f"Verbose: {verbose}")
        print(f"Create SOF: {create_sof}")

        # find FLATS
        dpr_types = {
            'FLAT,DARK':'FLAT_LAMP',
            'FLAT,LAMP':'FLAT_LAMP'
        }
        indices_dpr_types = {}
        
        # match by grating wheel and plate scale
        if not spiffier_gw is None:
            indices_gw = self.header_data["INS3.SPGW.NAME"] == spiffier_gw
        else:
            indices_gw = self.header_data["INS3.SPGW.NAME"] == self.wavel_setting
        if not spiffier_psw is None:
            indices_psw = self.header_data["INS3.SPXW.NAME"] == spiffier_psw
        else:
            indices_psw = np.ones_like(indices_gw,dtype=bool)
        
        indices_gw_pws = np.logical_and(indices_gw,indices_psw)
        
        for dpr_type_i in dpr_types:
            indices_dpr_types[dpr_type_i] = np.logical_and(self.header_data["DPR.TYPE"] == dpr_type_i, indices_gw_pws)
        
        # check unique DIT and NDIT
        indices = np.logical_or(indices_dpr_types['FLAT,DARK'],indices_dpr_types['FLAT,LAMP'])
        
        if np.sum(indices) == 0:
            raise RuntimeError(
                    f"The 'raw' folder does not contain any DPR.TYPE=FLAT_LAMP files with matching SPGW={spiffier_gw} and SPXW={spiffier_psw}"
                )
        flat_unique_dit_ndit = set(map(lambda x: (x[0],x[1]),self.header_data[indices][['DET.SEQ1.DIT','DET.NDIT']].values.tolist()))
        nb_files = {}
        for dit_i,ndit_i in flat_unique_dit_ndit:
            nb_files[(dit_i,ndit_i)] = np.sum(np.logical_and(self.header_data[indices]['DET.SEQ1.DIT'] == dit_i, self.header_data[indices]['DET.NDIT'] == ndit_i))
            max_nb_files_dit,max_nb_files_ndit=dit_i,ndit_i
        if len(flat_unique_dit_ndit) > 1:
            print(f'Found Flats of different (DIT,NDIT) matching SPGW={spiffier_gw} and SPXW={spiffier_psw}')
            print(f'Unique (DIT,NDIT) = {flat_unique_dit_ndit}')
            max_nb_files_dit,max_nb_files_ndit = sorted(nb_files.items(), key=lambda item: item[1])[-1][0]
        else:
            for dit_i,ndit_i in flat_unique_dit_ndit:
                max_nb_files_dit,max_nb_files_ndit=dit_i,ndit_i
        print(f'Selected (DIT,NDIT) = {(max_nb_files_dit,max_nb_files_ndit)} with a total of {nb_files[(max_nb_files_dit,max_nb_files_ndit)]} files')
        
        # match DIT and NDIT
        indices_dit_ndit = np.logical_and(self.header_data['DET.SEQ1.DIT'] == max_nb_files_dit, self.header_data['DET.NDIT'] == max_nb_files_ndit)
        for dpr_type_i in dpr_types:
            indices_dpr_types[dpr_type_i] = np.logical_and(indices_dpr_types[dpr_type_i], indices_dit_ndit)
        
        
        # find BPM from DARK, DIST, and DETLIN
        file_dict_search = {
            'CALIB_DARK_BPM':'BPM_DARK',
            'CALIB_DETLIN_BPM':'BPM_DIST',
            'CALIB_DISTORTION_BPM':'BPM_DETLIN'
        }
        match_conditions = {
            'CALIB_DISTORTION_BPM':['SPGW','SPXW'],
            'CALIB_DARK_BPM':['DIT'],
            'CALIB_DETLIN_BPM':[]
        }
        match_values = {'DIT':dit,'SPGW':spiffier_gw,'SPXW':spiffier_psw}
        path_file_search = {}
        for file_type in file_dict_search.keys():
            if file_type in self.file_dict:
                path_file_search[file_type] = []
                for key,value in self.file_dict[file_type].items():
                    if len(match_conditions[file_type]) == 0:
                        # no matching condition for detlin bpm
                        path_file_search[file_type] += [key]
                    else:
                        # for the rest, they have specific matching conditions. Once one condition isn't satisfied, the file isn't added
                        matching = True
                        for match_crit in match_conditions[file_type]:
                            if value[match_crit] != match_values[match_crit]:
                                matching = False
                        if matching:
                            path_file_search[file_type] += [key]
        
        # Create output folder

        output_dir = self.calib_folder / "calib_flat"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        sof_file = Path(output_dir / "files.sof")

        if not create_sof and not sof_file.exists():
            warnings.warn(
                f"The SOF file is not found at '{sof_file}' "
                "while 'create_sof' is set to False. "
                "Probably 'cal_dark' has not been "
                "previously executed so forcing "
                "'create_sof' to True."
            )

            create_sof = True

        if create_sof:
            print("\nCreating SOF file:")

            with open(sof_file, "w", encoding="utf-8") as sof_open:
                for dpr_type_i in dpr_types:
                    for item in self.header_data[indices_dpr_types[dpr_type_i]]["ARCFILE"]:
                        sof_open.write(f"{self.raw_folder}/{item} {dpr_types[dpr_type_i]}\n")
                        self._update_files(dpr_types[dpr_type_i], f"{self.raw_folder}/{item}")
                        
                for file_type,file_type_eso in file_dict_search.items():
                    for item in path_file_search[file_type]:
                        sof_open.write(f"{item} {file_type_eso}\n")
                        self._update_files(file_type, item)

            # Check if any flat lamp frames were found

            if "FLAT_LAMP" not in self.file_dict:
                raise RuntimeError(
                    "The 'raw' folder does not contain any DPR.TYPE=FLAT_LAMP files."
                )
            for file_type,file_type_eso in file_dict_search.items():
                for item in path_file_search[file_type]:
                    if file_type not in self.file_dict:
                        raise RuntimeError(
                            f"The 'raw' folder does not contain any DPR.TYPE={file_type} files."
                        )
        else:
            print(f"\nFound SOF file: {sof_file}")

        # Create EsoRex configuration file if not found

        self._create_config("eris_ifu_flat", "calib_flat", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "calib_flat.rc"

        esorex = [
            self.esorex_path,
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            '--no-checksum',
            "eris_ifu_flat",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        print("Output files:")

        name_extension = f'_{spiffier_gw}_{spiffier_psw}'

        # Update file dictionary with flat bpm

        fits_files = Path(self.path / "calib/calib_flat").glob(
            "eris_ifu_flat_bpm.fits"
        )
        self._rename_products(list(fits_files),'CALIB_FLAT_BPM',name_extension = name_extension)

        # Update file dictionary with master flat

        fits_files = Path(self.path / "calib/calib_flat").glob(
            "eris_ifu_flat_master_flat.fits"
        )
        self._rename_products(list(fits_files),'CALIB_FLAT_MASTER',name_extension = name_extension)
    
        
        # Create plots

        self._plot_image("CALIB_FLAT_MASTER", "calib/calib_flat")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def calib_wavecal(
        self, 
        verbose: bool = True, 
        create_sof: bool = True, 
        spiffier_gw: str = None, 
        spiffier_psw: str = None
    ) -> None:
        """
        Method for running ``eris_ifu_wavecal``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.
        create_sof : bool
            Create a new SOF file. Setting the argument to ``True``
            will overwrite the SOF file if already present. Setting
            the argument to ``False`` will allow for manually
            adjusting an existing SOF file if the routine had
            already been previously executed.


        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create WAVECAL", recipe_name="eris_ifu_wavecal")

        print(f"Verbose: {verbose}")
        print(f"Create SOF: {create_sof}")

        # find WAVE_LAMP
        
        dpr_types = {
            'WAVE,DARK':'WAVE_LAMP',
            'WAVE,LAMP':'WAVE_LAMP'
        }
        
        indices_dpr_types = {}

        # match by grating wheel and plate scale
        if not spiffier_gw is None:
            indices_gw = self.header_data["INS3.SPGW.NAME"] == spiffier_gw
        else:
            spiffier_gw = self.wavel_setting
            indices_gw = self.header_data["INS3.SPGW.NAME"] == self.wavel_setting
        if not spiffier_psw is None:
            indices_psw = self.header_data["INS3.SPXW.NAME"] == spiffier_psw
        else:
            indices_psw = np.ones_like(indices_gw,dtype=bool)
        
        indices_gw_pws = np.logical_and(indices_gw,indices_psw)
        
        for dpr_type_i in dpr_types:
            indices_dpr_types[dpr_type_i] = np.logical_and(self.header_data["DPR.TYPE"] == dpr_type_i, indices_gw_pws)
        
        # find MASTER FLAT, DISTORTION and static calibration files
        file_dict_search = {
            'CALIB_FLAT_MASTER':'MASTER_FLAT',
            'CALIB_DISTORTION_DISTORTION':'DISTORTION',
            'FIRST_WAVE_FIT':'FIRST_WAVE_FIT',
            'REF_LINE_ARC':'REF_LINE_ARC',
            'WAVE_SETUP':'WAVE_SETUP',
        }
        
        path_file_search = {}
        for file_type in file_dict_search.keys():
            if file_type in self.file_dict:
                path_file_search[file_type] = []
                for key,value in self.file_dict[file_type].items():
                    if file_type == 'CALIB_FLAT_MASTER':
                        if value['SPGW'] == spiffier_gw and value['SPXW'] == spiffier_psw:
                            path_file_search[file_type] += [key]
                    elif file_type == 'CALIB_DISTORTION_DISTORTION':
                        if value['SPGW'] == spiffier_gw:
                            path_file_search[file_type] += [key]
                    else:
                        path_file_search[file_type] += [key]
        
        # Create output folder

        output_dir = self.calib_folder / "calib_wavecal"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        sof_file = Path(output_dir / "files.sof")

        if not create_sof and not sof_file.exists():
            warnings.warn(
                f"The SOF file is not found at '{sof_file}' "
                "while 'create_sof' is set to False. "
                "Probably 'cal_dark' has not been "
                "previously executed so forcing "
                "'create_sof' to True."
            )

            create_sof = True

        if create_sof:
            print("\nCreating SOF file:")

            with open(sof_file, "w", encoding="utf-8") as sof_open:
                for dpr_type_i in dpr_types:
                    for item in self.header_data[indices_dpr_types[dpr_type_i]]["ARCFILE"]:
                        sof_open.write(f"{self.raw_folder}/{item} {dpr_types[dpr_type_i]}\n")
                        self._update_files(dpr_types[dpr_type_i], f"{self.raw_folder}/{item}")
                        
                for file_type,file_type_eso in file_dict_search.items():
                    for item in path_file_search[file_type]:
                        sof_open.write(f"{item} {file_type_eso}\n")
                        self._update_files(file_type, item)

            # Check if any wave lamp frames were found

            if "WAVE_LAMP" not in self.file_dict:
                raise RuntimeError(
                    "The 'raw' folder does not contain any DPR.TYPE=WAVE_LAMP files."
                )
            for file_type,file_type_eso in file_dict_search.items():
                for item in path_file_search[file_type]:
                    if file_type not in self.file_dict:
                        raise RuntimeError(
                            f"The 'raw' folder does not contain any DPR.TYPE={file_type} files."
                        )
        else:
            print(f"\nFound SOF file: {sof_file}")

        # Create EsoRex configuration file if not found

        self._create_config("eris_ifu_wavecal", "calib_wavecal", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "calib_wavecal.rc"

        esorex = [
            self.esorex_path,
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            '--no-checksum',
            "eris_ifu_wavecal",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        print("Output files:")

        name_extension = f'_{spiffier_gw}_{spiffier_psw}'

        # Update file dictionary with flat bpm

        fits_files = Path(self.path / "calib/calib_wavecal").glob(
            "eris_ifu_wave_arcImg_resampled.fits"
        )
        self._rename_products(list(fits_files),'WAVE_LAMP_STACKED_RESAMPLED',name_extension = name_extension)

        # Update file dictionary with master flat

        fits_files = Path(self.path / "calib/calib_wavecal").glob(
            "eris_ifu_wave_arcImg_stacked.fits"
        )
        self._rename_products(list(fits_files),'WAVE_LAMP_STACKED',name_extension = name_extension)
        
        
        # Update file dictionary with master flat

        fits_files = Path(self.path / "calib/calib_wavecal").glob(
            "eris_ifu_wave_map.fits"
        )
        self._rename_products(list(fits_files),'WAVE_MAP',name_extension = name_extension)
    
        
        # Create plots

        self._plot_image("WAVE_MAP", "calib/calib_wavecal")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)
    
    @typechecked
    def calib_stdstar_flux(
        self, 
        verbose: bool = True, 
        create_sof: bool = True
    ) -> None:
        """
        Method for running ``eris_ifu_stdstar``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.
        create_sof : bool
            Create a new SOF file. Setting the argument to ``True``
            will overwrite the SOF file if already present. Setting
            the argument to ``False`` will allow for manually
            adjusting an existing SOF file if the routine had
            already been previously executed.


        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create STDSTAR_FLUX", recipe_name="eris_ifu_stdstar")

        print(f"Verbose: {verbose}")
        print(f"Create SOF: {create_sof}")

        # find STD
        indices_std = np.logical_and(self.header_data["DPR.TYPE"] == "STD,FLUX", self.header_data["INS3.SPGW.NAME"] == self.wavel_setting[0] + '_low')

        # find SKY,STD
        indices_sky_std = np.logical_and(self.header_data["DPR.TYPE"] == "SKY,STD,FLUX", self.header_data["INS3.SPGW.NAME"] == self.wavel_setting[0] + '_low')

        # find all calibration files from previous data reduction
        file_dict_search = {
            'CALIB_DISTORTION_DISTORTION':'DISTORTION',
            'CALIB_DISTORTION_SLITLET_POS':'SLITLET_POS',
            'WAVE_MAP':'WAVE_MAP',
            'CALIB_FLAT_MASTER':'MASTER_FLAT',
            'CALIB_DARK_MASTER':'MASTER_DARK',
            'CALIB_DARK_BPM':'BPM_DARK',
            'CALIB_FLAT_BPM':'BPM_FLAT',
            'CALIB_DETLIN_BPM':'BPM_LINEARITY'
        }
        path_file_search = {}
        for file_type in file_dict_search.keys():
            if file_type in self.file_dict:
                path_file_search[file_type] = []
                for key,value in self.file_dict[file_type].items():
                    path_file_search[file_type] += [key]
        
        # find all static calibration files
        pro_catg = {
            'EXTCOEFF_TABLE':'EXTCOEFF_TABLE',
            'OH_SPEC':'OH_SPEC',
            'RESP_FIT_POINTS_CATALOG':'RESP_FIT_POINTS_CATALOG',
            'TELL_MOD_CATALOG':'TELL_MOD_CATALOG',
            'FLUX_STD_CATALOG':'FLUX_STD_CATALOG',
        }
        pro_catg_grating_specific = {
            'RESPONSE_WINDOWS':'RESPONSE_WINDOWS',
            'FIT_AREAS':'FIT_AREAS',
            'QUALITY_AREAS':'QUALITY_AREAS',
            'EFFICIENCY_WINDOWS':'EFFICIENCY_WINDOWS',
        }
        
        indices_pro_catg = {}
        
        for pro_catg_i in pro_catg:
            indices_pro_catg[pro_catg_i] = self.header_data["PRO.CATG"] == pro_catg_i
        
        for pro_catg_i in pro_catg_grating_specific:
            indices_pro_catg[pro_catg_i] = np.logical_and(self.header_data["PRO.CATG"] == pro_catg_i, self.header_data["INS3.SPGW.NAME"] == self.wavel_setting[0] + '_low')
        
        # Create output folder

        output_dir = self.calib_folder / "calib_stdstar_flux"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        sof_file = Path(output_dir / "files.sof")

        if not create_sof and not sof_file.exists():
            warnings.warn(
                f"The SOF file is not found at '{sof_file}' "
                "while 'create_sof' is set to False. "
                "Probably 'cal_dark' has not been "
                "previously executed so forcing "
                "'create_sof' to True."
            )

            create_sof = True

        if create_sof:
            print("\nCreating SOF file:")

            with open(sof_file, "w", encoding="utf-8") as sof_open:
                for item in self.header_data[indices_std]["ARCFILE"]:
                    sof_open.write(f"{self.raw_folder}/{item} STD\n")
                    self._update_files("STD", f"{self.raw_folder}/{item}")
                
                for item in self.header_data[indices_sky_std]["ARCFILE"]:
                    sof_open.write(f"{self.raw_folder}/{item} SKY_STD\n")
                    self._update_files("SKY_STD", f"{self.raw_folder}/{item}")
                
                for file_type,file_type_eso in file_dict_search.items():
                    for item in path_file_search[file_type]:
                        sof_open.write(f"{item} {file_type_eso}\n")
                        self._update_files(file_type, item)
                
                for pro_catg_i in pro_catg.keys():
                    for item in self.header_data[indices_pro_catg[pro_catg_i]]["ARCFILE"]:
                        sof_open.write(f"{self.raw_folder}/{item} {pro_catg[pro_catg_i]}\n")
                        self._update_files(pro_catg[pro_catg_i], f"{self.raw_folder}/{item}")
                
                for pro_catg_i in pro_catg_grating_specific.keys():
                    for item in self.header_data[indices_pro_catg[pro_catg_i]]["ARCFILE"]:
                        sof_open.write(f"{self.raw_folder}/{item} {pro_catg_grating_specific[pro_catg_i]}\n")
                        self._update_files(pro_catg_grating_specific[pro_catg_i], f"{self.raw_folder}/{item}")
            
            # Check if any wave lamp frames were found
            check_file_types = list(pro_catg.keys()) + list(pro_catg_grating_specific.keys()) + list(file_dict_search.keys()) + ['STD','SKY_STD']
            for key in check_file_types:
                if key not in self.file_dict:
                    raise RuntimeError(
                        f"The 'raw' folder does not contain any {key} files."
                    )
        else:
            print(f"\nFound SOF file: {sof_file}")

        # Create EsoRex configuration file if not found

        self._create_config("eris_ifu_stdstar", "calib_stdstar_flux", verbose)
        
        # Run EsoRex

        print()

        #config_file = self.config_folder / "calib_stdstar_flux.rc"
        config_file = self.config_folder / 'eris_ifu_stdstar.rc'
        esorex = [
            self.esorex_path,
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            '--no-checksum',
            "eris_ifu_stdstar",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        print("Output files:")
        
        
        # Update file dictionary

        files_to_update = {
            'STD_FLUX_CUBE_COADD_FLUXCAL':'eris_ifu_stdstar_cube_fluxcal.fits',
            'STD_FLUX_CUBE_MEDIAN':'eris_ifu_stdstar_cube_median.fits',
            'DAR_CORRECTED_CUBE':'eris_ifu_stdstar_dar_cube_*.fits',
            'EXTRACTION_MASK':'eris_ifu_stdstar_extraction_mask.fits',
            'EFFICIENCY':'eris_ifu_stdstar_no_flat_efficiency.fits',
            'RESPONSE':'eris_ifu_stdstar_response.fits',
            'SKY_CUBE':'eris_ifu_stdstar_sky_cube_*.fits',
            'SPECTRUM':'eris_ifu_stdstar_spectrum.fits',
            'SPECTRUM_FLUXCAL':'eris_ifu_stdstar_spectrum_fluxcal.fits',
            'STD_CUBE_MEAN':'eris_ifu_stdstar_std_cube_mean.fits',
            'STD_FLUX_CUBE':'eris_ifu_stdstar_std_flux_cube_*.fits',
            'STD_FLUX_CUBE_COADD':'eris_ifu_stdstar_std_flux_cube_coadd.fits',
            'SKY_TWEAKED_CUBE':'eris_ifu_stdstar_twk_cube_*.fits',
        }

        for file_type,file_name in files_to_update.items():
            fits_files = Path(self.path / "calib/calib_stdstar_flux").glob(
                file_name
            )
    
            for item in fits_files:
                self._update_files(file_type, str(item))
    
        
        # Create plots

        # self._plot_image("STD_FLUX_CUBE_COADD", "calib/calib_stdstar_flux")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def science_ifu_jitter(
        self, 
        verbose: bool = True, 
        create_sof: bool = True, 
        new_config: dict = {}, 
        dit: float = None, 
        ndit: int = None, 
        spiffier_gw: str = None, 
        spiffier_psw: str = None, 
        obj_date_obs: list = None, 
        sky_date_obs: list = None, 
        output_name: str = 'product', 
        use_corr_wavemap: Union[str, None] = None
    ) -> None:
        """
        Method for running ``eris_ifu_jitter``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.
        create_sof : bool
            Create a new SOF file. Setting the argument to ``True``
            will overwrite the SOF file if already present. Setting
            the argument to ``False`` will allow for manually
            adjusting an existing SOF file if the routine had
            already been previously executed.


        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create SCIENCE CUBE", recipe_name="eris_ifu_jitter")

        print(f"Verbose: {verbose}")
        print(f"Create SOF: {create_sof}")
        
        mask_instrument_setting = np.ones(len(self.header_data),dtype=bool)
        mask_instrument_setting = np.logical_and(mask_instrument_setting,self.header_data['DET.SEQ1.DIT'] == dit)
        mask_instrument_setting = np.logical_and(mask_instrument_setting,self.header_data['DET.NDIT'] == ndit)
        mask_instrument_setting = np.logical_and(mask_instrument_setting,self.header_data['INS3.SPGW.NAME'] == spiffier_gw)
        mask_instrument_setting = np.logical_and(mask_instrument_setting,self.header_data['INS3.SPXW.NAME'] == spiffier_psw)
        
        # find OBJ
        if not obj_date_obs is None:
            indices_obj = np.isin(self.header_data["DATE-OBS"],obj_date_obs)
        else:
            indices_obj = np.logical_and(self.header_data["DPR.TYPE"] == "OBJECT", mask_instrument_setting)

        # find SKY
        if not sky_date_obs is None:
            indices_sky = np.isin(self.header_data["DATE-OBS"],sky_date_obs)
        else:
            indices_sky = np.logical_and(self.header_data["DPR.TYPE"] == "SKY", mask_instrument_setting)

        # find all calibration files from previous data reduction
        file_dict_search = {
            'CALIB_DISTORTION_DISTORTION':'DISTORTION',
            'CALIB_DISTORTION_SLITLET_POS':'SLITLET_POS',
            'CALIB_FLAT_MASTER':'MASTER_FLAT',
            'CALIB_DARK_MASTER':'MASTER_DARK',
            'CALIB_DARK_BPM':'BPM_DARK',
            'CALIB_FLAT_BPM':'BPM_FLAT',
            'CALIB_DETLIN_BPM':'BPM_LINEARITY'
        }
        if use_corr_wavemap is None:
            file_dict_search['WAVE_MAP'] = 'WAVE_MAP'
            
        match_conditions = {
            'CALIB_DISTORTION_DISTORTION':['SPGW','SPXW'],
            'CALIB_DISTORTION_SLITLET_POS':['SPGW','SPXW'],
            'WAVE_MAP':['SPGW','SPXW'],
            'CALIB_FLAT_MASTER':['SPGW','SPXW'],
            'CALIB_DARK_MASTER':['DIT'],
            'CALIB_DARK_BPM':['DIT'],
            'CALIB_FLAT_BPM':['SPGW','SPXW'],
            'CALIB_DETLIN_BPM':[]
        }
        match_values = {'DIT':dit,'SPGW':spiffier_gw,'SPXW':spiffier_psw}
        path_file_search = {}
        for file_type in file_dict_search.keys():
            if file_type in self.file_dict:
                path_file_search[file_type] = []
                for key,value in self.file_dict[file_type].items():
                    if len(match_conditions[file_type]) == 0:
                        # no matching condition for detlin bpm
                        path_file_search[file_type] += [key]
                    else:
                        # for the rest, they have specific matching conditions. Once one condition isn't satisfied, the file isn't added
                        matching = True
                        for match_crit in match_conditions[file_type]:
                            if value[match_crit] != match_values[match_crit]:
                                matching = False
                        if matching:
                            path_file_search[file_type] += [key]
                            
        
        # find all static calibration files
        pro_catg = {
            'OH_SPEC':'OH_SPEC',
            'EXTCOEFF_TABLE':'EXTCOEFF_TABLE',
            # 'RESPONSE':'RESPONSE'
        }
        
        indices_pro_catg = {}
        
        for pro_catg_i in pro_catg:
            indices_pro_catg[pro_catg_i] = self.header_data["PRO.CATG"] == pro_catg_i
        
        # Create output folder

        output_dir = self.product_folder / "science_ifu_jitter" / output_name

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        sof_file = Path(output_dir / "files.sof")

        if not create_sof and not sof_file.exists():
            warnings.warn(
                f"The SOF file is not found at '{sof_file}' "
                "while 'create_sof' is set to False. "
                "Probably 'cal_dark' has not been "
                "previously executed so forcing "
                "'create_sof' to True."
            )

            create_sof = True

        if create_sof:
            print("\nCreating SOF file:")

            with open(sof_file, "w", encoding="utf-8") as sof_open:
                for item in self.header_data[indices_obj]["ARCFILE"]:
                    sof_open.write(f"{self.raw_folder}/{item} OBJ\n")
                    self._update_files("OBJ", f"{self.raw_folder}/{item}")
                
                for item in self.header_data[indices_sky]["ARCFILE"]:
                    sof_open.write(f"{self.raw_folder}/{item} SKY_OBJ\n")
                    self._update_files("SKY", f"{self.raw_folder}/{item}")
                
                for file_type,file_type_eso in file_dict_search.items():
                    for item in path_file_search[file_type]:
                        sof_open.write(f"{item} {file_type_eso}\n")
                        self._update_files(file_type, item)
                
                for pro_catg_i in pro_catg.keys():
                    for item in self.header_data[indices_pro_catg[pro_catg_i]]["ARCFILE"]:
                        sof_open.write(f"{self.raw_folder}/{item} {pro_catg[pro_catg_i]}\n")
                        self._update_files(pro_catg[pro_catg_i], f"{self.raw_folder}/{item}")
                
                # handle corrected WAVE_MAP differently
                if not use_corr_wavemap is None:
                    sof_open.write(f"{use_corr_wavemap} {'WAVE_MAP'}\n")
                    self._update_files('WAVE_MAP_CORR', f"{use_corr_wavemap}") # this overwrites from WAVE_MAP_CORR to WAVE_MAP!!!
            
            # Check if any frames were found
            check_file_types = list(pro_catg.keys())+ list(file_dict_search.keys()) + ['OBJ']
            if np.sum(indices_sky) > 0:
                check_file_types += ['SKY']
            for key in check_file_types:
                if key not in self.file_dict:
                    raise RuntimeError(
                        f"The 'raw' folder does not contain any {key} files."
                    )
        else:
            print(f"\nFound SOF file: {sof_file}")

        # Create EsoRex configuration file if not found

        self._create_config("eris_ifu_jitter", "science_ifu_jitter", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / 'science_ifu_jitter.rc'
        
        if len(new_config.keys()) > 0:
            self.modify_config(config_file, new_config = new_config,eso_recipe = 'eris_ifu_jitter')
        
        esorex = [
            self.esorex_path,
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            '--no-checksum',
            "eris_ifu_jitter",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        print("Output files:")
        
        
        # Update file dictionary

        files_to_update = {
            'OBJECT_CUBE':'eris_ifu_jitter_obj_cube_[0-9][0-9][0-9].fits',
            'SKY_CUBE':'eris_ifu_jitter_sky_cube_[0-9][0-9][0-9].fits',
            'SKY_TWEAKED_CUBE':'eris_ifu_jitter_twk_cube_[0-9][0-9][0-9].fits',
            'DAR_CORRECTED_CUBE':'eris_ifu_jitter_dar_cube_[0-9][0-9][0-9].fits',
            'DAR_CORRECTED_CUBE_COADD':'eris_ifu_jitter_dar_cube_coadd.fits',
            'DAR_CORRECTED_CUBE_MEAN':'eris_ifu_jitter_dar_cube_mean.fits',
            'SKY_TWEAKED_CUBE_COADD':'eris_ifu_jitter_twk_cube_coadd.fits',
            'SKY_TWEAKED_CUBE_MEAN':'eris_ifu_jitter_twk_cube_mean.fits',
            'OBJECT_CUBE_COADD':'eris_ifu_jitter_obj_cube_coadd.fits',
            'OBJECT_CUBE_MEAN':'eris_ifu_jitter_obj_cube_mean.fits',
            'BPM_CUBE':'eris_ifu_jitter_bpm_cube_[0-9][0-9][0-9].fits',
            'STD_CUBE_MEDIAN':'eris_ifu_jitter_cube_median.fits',
            'EXTRACTION_MASK':'eris_ifu_jitter_extraction_mask.fits',
            'SPECTRUM':'eris_ifu_jitter_spectrum.fits',
            'SPECTRUM_FLUXCAL':'eris_ifu_jitter_spectrum_fluxcal.fits',
            'OBJECT_CUBE_COADD_FLUXCAL':'eris_ifu_jitter_cube_fluxcal.fits',
            # 'OBJECT_WAVE_B':'eris_ifu_jitter_dbg_waveB_*fits' # keep this one separate to distinguish between OBJECT and SKY
        }
        
        # update OBJECT_WAVE_B
        for file_type in ['OBJECT','SKY']:
            object_files = sorted(output_dir.glob(files_to_update[f'{file_type}_CUBE']))
            object_files_nbs = [Path(file_i).name.split('_')[5][:3] for file_i in object_files]
            object_wave_B_files = [Path(output_dir) / f'eris_ifu_jitter_dbg_waveB_{nb_i}.fits' for nb_i in object_files_nbs]
            
            if not use_corr_wavemap is None:
                self._rename_products(list(object_wave_B_files),f'{file_type}_WAVE_B',name_extension = '_corr_wavemap',add_arcfile=False)
            else:
                self._rename_products(list(object_wave_B_files),f'{file_type}_WAVE_B',name_extension = '_std_wavemap',add_arcfile=False)
        
        for file_type,file_name in files_to_update.items():
            fits_files = output_dir.glob(
                file_name
            )
            add_arcfile = file_type in ['OBJECT_CUBE','DAR_CORRECTED_CUBE','SKY_CUBE','SKY_TWEAKED_CUBE']
            if not use_corr_wavemap is None:
                self._rename_products(list(fits_files),file_type,name_extension = '_corr_wavemap',add_arcfile=add_arcfile)
            else:
                self._rename_products(list(fits_files),file_type,name_extension = '_std_wavemap',add_arcfile=add_arcfile)
            #file updating is done in _rename_products()
            #for item in fits_files:
            #    self._update_files(file_type, str(item))
        
        
        
        # Create plots

        self._plot_image("DAR_CORRECTED_CUBE_MEAN", "product/science_ifu_jitter/" + output_name)
        self._plot_image("SKY_TWEAKED_CUBE_MEAN", "product/science_ifu_jitter/" + output_name)
        #self._plot_image("SKY_CUBE", "product/science_ifu_jitter")
        #self._plot_image("OBJECT_CUBE_COADD", "product/science_ifu_jitter")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def calib_wavelength_cross_corr(
        self,
        accuracy: float = 10, 
        input_dir: str = None,
        output_folder: str = None,
        name_extension: str = None,
        telluric_wvl: Union[list,np.ndarray] = [],
        telluric_transm: Union[list,np.ndarray] = [],
        method: str = 'full',
        plot=True,
        save_result=True
    ) -> dict:
        
        # get WAVE_MAP
        wavemap_all_files = list(self.file_dict['WAVE_MAP'].keys())
        wavemap_files = []
        if len(wavemap_all_files) == 1:
            wavemap_files += [wavemap_all_files[0]]
        else:
            for file_i in wavemap_all_files:
                if not input_dir is None:
                    if Path(file_i).parent.name == input_dir:
                        wavemap_files += [file_i]
                else:
                    wavemap_files += [file_i]
        if len(wavemap_files) == 0:
            raise RuntimeError(
                        "No WAVE_MAP files found."
                    )
        elif len(wavemap_files) > 1:
            print('Several WAVE_MAP files found. Taking the first one.')
        wavemap = fits.getdata(wavemap_files[0])
        wavemap_header = fits.getheader(wavemap_files[0])
        
        # get waveB files
        object_waveB = []
        for file_path,item in self.file_dict['OBJECT_WAVE_B'].items():
            if Path(file_path).parent.name == input_dir:
                right_file = False
                if not name_extension is None:
                    if name_extension in Path(file_path).name:
                        right_file = True
                else:
                    right_file = True
                if not right_file:
                    continue
                print(file_path)
                hdr = fits.getheader(file_path)
                if hdr['ESO DPR TYPE'] == 'OBJECT':
                    object_waveB += [file_path]
                    print(hdr['ESO DPR TYPE'])
        
        if len(object_waveB) == 0:
            raise RuntimeError(
                        "No OBJECT_WAVE_B files found.")

        # Create output folder
        if output_folder is None:
            output_dir = self.calib_folder / "calib_wavelength_cross_corr" / input_dir
        else:
            output_dir = self.calib_folder / "calib_wavelength_cross_corr" / output_folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        wavemap_corr_file_correspondence = {}
        shift_trsm_cc_slitlet_files = np.zeros((len(object_waveB),32,64))
        for object_i,object_filepath in enumerate(object_waveB):
            print(f'Processing OBJECT frame {object_i} out of {len(object_waveB)} frames.',end='\n\n')
            object_data = fits.getdata(object_filepath)
            hdr = fits.getheader(object_filepath)
            lenwvl,lenxy = np.shape(object_data)
            wvl_params = [hdr['CRVAL2'],hdr['CD2_2']]
            wavelength = np.array([wvl_params[0] + i*wvl_params[1] for i in range(lenwvl)])
            wlen_tell_cr,transm_tell_cr = rebin(np.array(telluric_wvl),np.array(telluric_transm),wavelength,flux_err = None, method='datalike')
            mean_wvl_step = np.mean(wavelength[1:]-wavelength[:-1])
            shift_trsm_cc = np.zeros((lenxy))
            for xy in range(lenxy):
                print('Progress %.2f' % (xy/lenxy*100),end='\r')
                spectrum = object_data[:,xy]
                mask_nans = np.isnan(spectrum)
                smooth_spectrum = ndimage.gaussian_filter(spectrum[~mask_nans],sigma=40)
                cr_spectrum = np.zeros_like(spectrum)
                cr_spectrum[~mask_nans] = spectrum[~mask_nans] - smooth_spectrum
                tmp_offset, _, _ = phase_cross_correlation(
                    cr_spectrum,
                    transm_tell_cr,
                    normalization=None,
                    upsample_factor=accuracy,
                    overlap_ratio=0.3)
                shift_trsm_cc[xy] = tmp_offset[0]*mean_wvl_step
            shift_trsm_cc_slitlet = np.array([shift_trsm_cc[64*i:64*(i+1)] for i in range(32)])
            shift_trsm_cc_slitlet_files[object_i,:,:] = shift_trsm_cc_slitlet[:,:]
            
            save_file = f'eris_ifu_wave_map_corr_{Path(object_filepath).stem}'
            wavemap_corr_filepath = output_dir / f'{save_file}.fits'
            wavemap_corr_file_correspondence[str(wavemap_corr_filepath)] = object_filepath
            
            if method != 'model':
                mask_good = np.abs(shift_trsm_cc_slitlet/mean_wvl_step) < 15
                median_shift = np.nanmedian(np.array(shift_trsm_cc_slitlet[mask_good])/mean_wvl_step)
                wvl_calib = np.zeros((2048))
                for slitlet_i in range(len(shift_trsm_cc_slitlet)):
                    wavelength_shift = shift_trsm_cc_slitlet[slitlet_i]/mean_wvl_step
                    y_fit,y_calib=fit_wavelength_error(wavelength_shift,deg=1,lim_mask=4,lim_sel=0.5,median=median_shift)
                    if method == 'median':
                        wvl_calib[slitlet_i*64:(slitlet_i+1)*64]=median_shift*mean_wvl_step*np.ones_like(y_calib)
                    else: 
                        #if method == 'full':
                        wvl_calib[slitlet_i*64:(slitlet_i+1)*64]=y_calib*mean_wvl_step
                if plot:
                    plt.figure(figsize=(15,5))
                    plt.plot(wvl_calib/mean_wvl_step)
                    # plt.plot(shift_trsm_cc/mean_wvl_step)
                    plt.annotate('Slitlet',(0,median_shift-1.6),fontsize=8)
                    for slitlet_i,slitlet_i_nb in enumerate(self.slitlet_layout):
                        plt.annotate(slitlet_i_nb,(slitlet_i*64 + 16,median_shift-2),horizontalalignment='center',fontsize=8)
                    plt.ylim((median_shift-2,median_shift+2))
                    plt.xlabel('Column number')
                    plt.ylabel('Wavelength shift [px]')
                    plt.title(f'WAVE MAP error for frame {Path(object_filepath).stem}')
                    plt.savefig(output_dir / f'{save_file}.png')
                    plt.show()
    
                # apply the correction
                wavemap_corr = wavemap - wvl_calib[np.newaxis,:]
                
                # save the correction
                if save_result:
                    primary_hdu = fits.PrimaryHDU(data=wavemap_corr,header=wavemap_header)
                    hdu_list = fits.HDUList([primary_hdu])
                    hdu_list.writeto(wavemap_corr_filepath,overwrite=True)
                    self._update_files('WAVE_MAP_CORR', str(wavemap_corr_filepath))
        
        if method == 'model':
            # get the median wvl shift
            mask_good = np.abs(shift_trsm_cc_slitlet_files/mean_wvl_step) < 15
            median_shift = np.zeros((len(object_waveB)))
            for obj_i in range(len(object_waveB)):
                median_shift[obj_i] = np.nanmedian(np.array(shift_trsm_cc_slitlet_files[obj_i][mask_good[obj_i]])/mean_wvl_step)
            # subtract the median wvl shift
            shift_trsm_cc_slitlet_files_zero = shift_trsm_cc_slitlet_files - median_shift[:,np.newaxis,np.newaxis]*mean_wvl_step
            # go through each slitlet and build a linear model of the wvl shift across the slitlet, considering all object files
            shift_trsm_cc_slitlet_files_fit = np.zeros((len(object_waveB),32,64))
            for slit_i in range(32):
                shift_trsm_cc_slitlet_files_zero_slit_i = shift_trsm_cc_slitlet_files_zero[:,slit_i,:]
                # only select wvl shifts within 10
                mask_good_i = np.abs(shift_trsm_cc_slitlet_files_zero_slit_i/mean_wvl_step) < 10
                
                x = np.array([np.arange(64) for j in range(len(object_waveB))])
                x_good = x[mask_good_i]
                y_good = shift_trsm_cc_slitlet_files_zero_slit_i[mask_good_i]
                # fit linear model
                pols = np.polyfit(x_good,y_good,deg=1)
                p = np.poly1d(pols)
                y_fit = p(x)
                shift_trsm_cc_slitlet_files_zero_slit_i_fit = np.reshape(y_fit,(len(object_waveB),64))
                shift_trsm_cc_slitlet_files_fit[:,slit_i,:] = shift_trsm_cc_slitlet_files_zero_slit_i_fit + median_shift[:,np.newaxis]*mean_wvl_step
            for object_i,object_filepath in enumerate(object_waveB):
                wvl_calib = np.reshape(shift_trsm_cc_slitlet_files_fit[object_i,:,:],(-1))
                
                save_file = f'eris_ifu_wave_map_corr_{Path(object_filepath).stem}'
                wavemap_corr_filepath = output_dir / f'{save_file}.fits'
                wavemap_corr_file_correspondence[str(wavemap_corr_filepath)] = object_filepath
                
                if plot:
                    plt.figure(figsize=(15,5))
                    plt.plot(wvl_calib/mean_wvl_step)
                    # plt.plot(shift_trsm_cc/mean_wvl_step)
                    plt.annotate('Slitlet',(0,median_shift[obj_i]-1.6),fontsize=8)
                    for slitlet_i,slitlet_i_nb in enumerate(self.slitlet_layout):
                        plt.annotate(slitlet_i_nb,(slitlet_i*64 + 16,median_shift[obj_i]-2),horizontalalignment='center',fontsize=8)
                    plt.ylim((median_shift[obj_i]-2,median_shift[obj_i]+2))
                    plt.xlabel('Column number')
                    plt.ylabel('Wavelength shift [px]')
                    plt.title(f'WAVE MAP error for frame {Path(object_filepath).stem}')
                    plt.savefig(output_dir / f'{save_file}.png')
                    plt.show()
    
                # apply the correction
                wavemap_corr = wavemap - wvl_calib[np.newaxis,:]
                
                # save the correction
                if save_result:
                    primary_hdu = fits.PrimaryHDU(data=wavemap_corr,header=wavemap_header)
                    hdu_list = fits.HDUList([primary_hdu])
                    hdu_list.writeto(wavemap_corr_filepath,overwrite=True)
                    self._update_files('WAVE_MAP_CORR', str(wavemap_corr_filepath))

        # Write updated dictionary to JSON file
        if save_result:
            with open(self.json_file, "w", encoding="utf-8") as json_file:
                json.dump(self.file_dict, json_file, indent=4)
        return wavemap_corr_file_correspondence

    @typechecked
    def calib_wavelength_xcorr_full(
        self,
        input_folder,
        output_folder,
        continuum_sigma=60,
        accuracy=10,
        method = 'spline', # '0-order','0-order-linear','0-order-median','spline'
        spline_order=2,
        spline_smoothing=0.4,
        window_size=120,
        window_shift_ratio=4,
        plot=True,save_result=True
    ) -> None:

        # get the object files
        object_waveB = []
        for file_path,item in self.file_dict['OBJECT_WAVE_B'].items():
            if Path(file_path).parent.name == input_folder:
                print(file_path)
                hdr = fits.getheader(file_path)
                if hdr['ESO DPR TYPE'] == 'OBJECT':
                    object_waveB += [file_path]
                    print(hdr['ESO DPR TYPE'])
        if len(object_waveB) == 0:
            raise RuntimeError(
                "No OBJECT_WAVE_B files found.")
        
        # get the wavemap
        wavemap_all_files = list(self.file_dict['WAVE_MAP'].keys())
        wavemap_files = []
        if len(wavemap_all_files) == 1:
            wavemap_files += [wavemap_all_files[0]]
        else:
            for file_i in wavemap_all_files:
                if not input_dir is None:
                    if Path(file_i).parent.name == input_dir:
                        wavemap_files += [file_i]
                else:
                    wavemap_files += [file_i]
        if len(wavemap_files) == 0:
            raise RuntimeError(
                        "No WAVE_MAP files found."
                    )
        elif len(wavemap_files) > 1:
            print('Several WAVE_MAP files found. Taking the first one.')
        wavemap = fits.getdata(wavemap_files[0])
        wavemap_header = fits.getheader(wavemap_files[0])
        
        # create output folder
        if output_folder is None:
            output_dir = self.calib_folder / "calib_wavelength_cross_corr" / input_dir
        else:
            output_dir = self.calib_folder / "calib_wavelength_cross_corr" / output_folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # go through each object cube
        for obj_i,file_path in enumerate(object_waveB):
            hdr = fits.getdata(file_path)
            date_obs = hdr['DATE-OBS']
            print('Calibrating object cube %s' % date_obs)
            
            object_data = fits.getdata(file_path)
            lenwvl,lenxy = np.shape(object_data)
            wvl_params = [hdr['CRVAL2'],hdr['CD2_2']]
            wavelength = np.array([wvl_params[0] + i*wvl_params[1] for i in np.arange(lenwvl)])
            mean_d_wvl = np.mean(wavelength[:1]-wavelength[:-1])
            # read telluric model
            skycoord = SkyCoord(ra=hdr['RA'],dec=hdr['DEC'],unit='deg')
            string_coord = skycoord.to_string('hmsdms').replace('h',' ').replace('d',' ').replace('m',' ').replace('s','')
            tellurics_wlen,tellurics_transm,tellurics_flux = get_sky_calc_model(obj_coord=string_coord,date=date_obs[:19])
            
            # apply algorithm depending on method, save the px shift to each wavemap pixel
            wlen_corr_model_frame,wlen_corr_model_frame_px = calibrate_wavelength_frame(
                object_data,wavelength,
                tellurics_wlen=tellurics_wlen,tellurics_transm=tellurics_transm,
                filter_sigma=filter_sigma,
                accuracy = accuracy,
                spline_order = spline_order,spline_smoothing = spline_smoothing,
                window_size = window_size,window_shift_ratio=window_shift_ratio,
                method=method,
                plot=plot
                )
            # apply correction to wavemap
            wavemap_corr = np.zeros((2048,2048))
            for ij in np.arange(2048):
                # interpolate the correction to the wavelength of the wavemap
                interp_corr = interp1d(x=wavelength,y=wlen_corr_model_frame_px[:,ij],bounds_error=False,fill_value='extrapolate')
                column_correction = interp_corr(wavemap[:,ij])
                
                wavemap_corr[:,ij] = wavemap[:,ij] - column_correction*mean_d_wvl
            # save new wavemap
            wavemap_corr_filepath = output_dir / ('wavemap_corr_' + date_obs + '.fits')
            if save_result:
                primary_hdu = fits.PrimaryHDU(data=wavemap_corr,header=wavemap_header)
                hdu_list = fits.HDUList([primary_hdu])
                hdu_list.writeto(wavemap_corr_filepath,overwrite=True)
                # update file_dict
                self._update_files('WAVE_MAP_CORR', str(wavemap_corr_filepath))
        
        # Write updated dictionary to JSON file
        if save_result:
            if save_result:
                with open(self.json_file, "w", encoding="utf-8") as json_file:
                    json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def do_standard_reduction(
        self,
        spiffier_gw: str ='K_long',
        spiffier_psw: str = '25mas'
    ) -> None:
        self.extract_header()
        unique_dit_ndit = self._identify_science_dit_ndit()
        unique_dits = list(dict.fromkeys(list(map(lambda x: x[0],unique_dit_ndit))))
        for dit in unique_dits:
            self.calib_dark(dit=dit)
        mask_science = self.header_data['DPR.CATG'] == 'SCIENCE'
        nb_files = {}
        for dit in unique_dits:
            nb_files[dit] = np.sum(self.header_data[mask_science]['DET.SEQ1.DIT'] == dit)
        max_dit = sorted(nb_files.items(),key=lambda x: (x[0],x[1]))[-1][0]
        self.calib_detlin()
        self.calib_distortion(spiffier_gw = spiffier_gw, spiffier_psw = spiffier_psw)
        self.calib_flat(spiffier_gw = spiffier_gw, spiffier_psw = spiffier_psw,dit=max_dit)
        self.calib_wavecal(spiffier_gw = spiffier_gw, spiffier_psw = spiffier_psw)
    @typechecked
    def select_object_groups(
        self,
        dit: float
    ) -> dict:
        mask_science = self.header_data['DPR.CATG'] == 'SCIENCE'
        mask_object = self.header_data['DPR.TYPE'] == 'OBJECT'
        mask_dit = self.header_data['DET.SEQ1.DIT'] == dit
        mask = mask_science & mask_object & mask_dit
        object_files_groups = self.header_data[mask]['DATE-OBS'].sort_values()
        files_groups = {(i+1):{
            'OBJECT':list(object_files_groups[10*i:10*(i+1)].values),
            'SKY':[]} for i in range(int(np.ceil(len(object_files_groups)/10)))}
        return files_groups

    @typechecked
    def science_combine_cubes(
        self,
        obj_cubes: Union[list,np.ndarray] = [],
        offset_list: Union[list,np.ndarray] = [], # 2-column format
        verbose: bool = True, 
        create_sof: bool = True, 
        new_config: dict = {}, 
        output_name: str = 'product', 
    ):
        
        self._print_section("Combine cubes", recipe_name="eris_ifu_combine")

        print(f"Verbose: {verbose}")
        print(f"Create SOF: {create_sof}")
        
        
        # Create output folder

        output_dir = self.product_folder / "science_ifu_combine" / output_name

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create file with list of offsets
        offsets_file = Path(output_dir / "offset.list")
        with open(offsets_file, "w", encoding="utf-8") as offsets_open:
            for x,y in offset_list:
                offsets_open.write(f"{x} {y}\n")

        # Create SOF file

        sof_file = Path(output_dir / "files.sof")

        if not create_sof and not sof_file.exists():
            warnings.warn(
                f"The SOF file is not found at '{sof_file}' "
                "while 'create_sof' is set to False. "
                "Probably 'cal_dark' has not been "
                "previously executed so forcing "
                "'create_sof' to True."
            )

            create_sof = True

        if create_sof:
            print("\nCreating SOF file:")

            with open(sof_file, "w", encoding="utf-8") as sof_open:
                for item in obj_cubes:
                    sof_open.write(f"{item} OBJECT_CUBE\n")
                    self._update_files("OBJECT_CUBE", f"{item}")

            # Check if any frames were found
            check_file_types = ['OBJECT_CUBE']
            for key in check_file_types:
                if key not in self.file_dict:
                    raise RuntimeError(
                        f"The 'raw' folder does not contain any {key} files."
                    )
        else:
            print(f"\nFound SOF file: {sof_file}")

        # Create EsoRex configuration file if not found

        self._create_config("eris_ifu_combine_hdrl", "science_ifu_combine", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / 'science_ifu_combine.rc'
        
        self.modify_config(config_file, new_config = {'name_i':str(offsets_file)}, eso_recipe = 'eris_ifu_combine_hdrl')
        if len(new_config.keys()) > 0:
            self.modify_config(config_file, new_config = new_config, eso_recipe = 'eris_ifu_combine_hdrl')
        
        esorex = [
            self.esorex_path,
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            '--no-checksum',
            "eris_ifu_combine_hdrl",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        print("Output files:")
        
        
        # Update file dictionary

        files_to_update = {
            #'OBJECT_CUBE':'eris_ifu_jitter_obj_cube_[0-9][0-9][0-9].fits',
        }
        for file_type,file_name in files_to_update.items():
            fits_files = output_dir.glob(
                file_name
            )
            add_arcfile = file_type in ['OBJECT_CUBE','DAR_CORRECTED_CUBE','SKY_CUBE','SKY_TWEAKED_CUBE']
            self._rename_products(list(fits_files),file_type,name_extension = '',add_arcfile=True)
        
        
        
        # Create plots

        #self._plot_image("DAR_CORRECTED_CUBE_MEAN", "product/science_ifu_jitter/" + output_name)
        #self._plot_image("SKY_TWEAKED_CUBE_MEAN", "product/science_ifu_jitter/" + output_name)
        #self._plot_image("SKY_CUBE", "product/science_ifu_jitter")
        #self._plot_image("OBJECT_CUBE_COADD", "product/science_ifu_jitter")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)