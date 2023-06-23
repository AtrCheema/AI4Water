
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:02:10 2018
"""

__all__ = ["HdfFile"]

from typing import Union

import warnings

import numpy as np
import pandas as pd
from easy_mpl import plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from HSP2.HYDR import hydr

plt.rcParams["font.family"] = "Times New Roman"


m2_to_acre = 0.0002471   # meter square to Acre
cfs_to_lps = 28.3
lps_to_cms = 0.001
cfs_to_cms = 0.0283
lps_to_cfs = 0.0353
in_to_mm = 25.4

# units conversion constants, 1 ACRE is 43560 sq ft. assumes input in acre-ft
VFACT = 43560.0

plot_props = {'q': ['Streamflow', 'upper left'],
              'lat': ['Lateral Flow', 'upper right'],
              'suro': ['Surface Runoff', 'center right']}


class HdfFile(object):
    """
    class to get and set and process HDF5 file's data

    methods:

    :attributes
       sim_start: `str`, set or get simulation start time
       sim_end: `str`, set or get simulation end time
       units:
       no_of_rchres:
       rchres_ids
       no_of_perlnd:
       perlnd_ids:
       no_of_implnd:
       implnd_ids:
       freq:
       sim_len:
       sim_idx:
       general: dictionary

    :methods
        rchres_paras: fetches the specified parameter for a specific rchres
        rchres_flags: fetches a specific flag value/values for a specific rchres
        ftables: fetches ftable for specified rchres
        get_uc: fetches all parameters as dictionary in order to run a simulation
        hru_flow: fetches flow as in/ivld for a specific hru e.g. PERO for P101 or SURO for I103 etc
        sim_lat_cfs: calculates lateral flow as cfs by rerunning HYDR module
        sim_suro_cfs: calculates surface runoff as cfs by rerunning HYDR module
    """

    def __init__(
            self,
            name:str,
            uci_obj=None,
            observed_data=None
    ):
        if not os.path.exists(name):
            raise ValueError("{} file does not exit".format(name))

        self.hdfname = name
        self.uci_obj = uci_obj
        self.data_dir = os.path.dirname(name)

        with pd.HDFStore(self.hdfname) as store:
            self.links:pd.DataFrame = store.get('CONTROL/LINKS')
            self.ext_src:pd.DataFrame = store.get('CONTROL/EXT_SOURCES')

        if observed_data is not None:
            self.obs_df = observed_data.copy()

    @property
    def op_seq(self)->pd.DataFrame:
        with pd.HDFStore(self.hdfname) as store:
            return store.get('CONTROL/OP_SEQUENCE')

    @property
    def gblock(self)->pd.DataFrame:
        with pd.HDFStore(self.hdfname) as store:
            return store.get('CONTROL/GLOBAL')

    def change_time_step(
            self,
            new_time_step:int
    ):
        """
        changes the time step of all the operations

        Parameters
        -----------
        new_time_step : int
            the new time step in minutes
        """
        new_time_step = int(new_time_step)
        assert isinstance(new_time_step, int)

        with pd.HDFStore(self.hdfname) as store:
            op_seq = store.get('CONTROL/OP_SEQUENCE')
            op_seq.loc[:, 'INDELT_minutes'] = new_time_step
            self.put(store, 'CONTROL/OP_SEQUENCE', op_seq)
        return

    @property
    def sim_start(self)->str:
        return self.gblock.loc['Start', 'Info']

    @sim_start.setter
    def sim_start(self, value):

        with pd.HDFStore(self.hdfname) as store:
            gdata = store.get('CONTROL/GLOBAL')
            gdata.loc['Start'] = value
            self.put(store, 'CONTROL/GLOBAL', gdata)

    @staticmethod
    def put(store:pd.HDFStore, path, data):
        store.put(path, data, format='t', data_columns=True)

    @property
    def sim_end(self)->str:
        return self.gblock.loc['Stop', 'Info']

    @sim_end.setter
    def sim_end(self, value):
        """sets the simulation end
        >>> h5 = HdfFile(...)
        >>> h5.sim_end = "2022-08-26 00:00"
        """
        with pd.HDFStore(self.hdfname) as store:
            gdata = store.get('CONTROL/GLOBAL')
            gdata.loc['Stop'] = value
            self.put(store, 'CONTROL/GLOBAL', gdata)

    @property
    def units(self)->int:
        return int(self.gblock.loc['Units', 'Info'])

    @units.setter
    def units(self, value):
        """change sthe units"""
        with pd.HDFStore(self.hdfname) as store:
            gdata = store.get('CONTROL/GLOBAL')
            gdata.loc['units'] = value
            self.put(store, 'CONTROL/GLOBAL', gdata)

    @property
    def rchres_ids(self)->np.ndarray:
        """returns array consisting of names of RCHRES"""
        return self.op_seq.loc[self.op_seq['OPERATION']=='RCHRES'].loc[:, 'SEGMENT'].values

    @property
    def num_rchres(self):
        return len(self.rchres_ids)

    @property
    def perlnd_ids(self)->np.ndarray:
        """returns array consisting of ID of PERLNDS"""
        return self.op_seq.loc[self.op_seq['OPERATION']=='PERLND'].loc[:, 'SEGMENT'].values

    @property
    def num_perlnd(self)->int:
        return len(self.perlnd_ids)

    @property
    def implnd_ids(self)->np.ndarray:
        """returns array consisting of ID of IMPLND"""
        return self.op_seq.loc[self.op_seq['OPERATION']=='IMPLND'].loc[:, 'SEGMENT'].values

    @property
    def num_implnd(self):
        return len(self.implnd_ids)

    @property
    def perlnd_afactors(self)->pd.DataFrame:
        """
        returns a dataframe with `SVOLNO` and `AFACTR` columns
        """
        return self.links.loc[self.links['SVOL'] == 'PERLND'][['SVOLNO', 'AFACTR']]

    @property
    def tot_perlnd_area(self):
        """
        returns total area in Acres supposing landuse change file had area in acres
        """
        return self.links.loc[self.links['SVOL'] == 'PERLND']['AFACTR'].sum()

    @property
    def implnd_afactors(self):
        """
        returns a dataframe with `SVOLNO` and `AFACTR` columns
        """
        return self.links.loc[self.links['SVOL'] == 'IMPLND'][['SVOLNO', 'AFACTR']]

    @property
    def tot_implnd_area(self)->float:
        """
        returns total area in Acres supposing landuse change file had area in acres
        """
        return self.links.loc[self.links['SVOL'] == 'IMPLND']['AFACTR'].sum()

    @property
    def tot_area(self)->float:
        return self.tot_perlnd_area + self.tot_implnd_area

    def get_prec(self, name='PREC'):
        """ First finds the path of precipitation timeseries in hdf5 and then
        gets precipitation timeseries from hdf5 file."""
        return self._get_ts_by_tmemn(name)

    def set_prec(self, val):
        """sets the value of TIMESERIES"""
        self._set_ts_by_tmemn(val, "PREC")

    def plot_prec(self, start=None, end=None, **kwargs):
        """plots precipitation TIMESERIES
        >>> h5 = HdfFile(...)
        >>> h5.plot_prec(end="2021-07-30")
        """
        ts = slice_ts(self.get_prec(), st=start, en=end)
        return plot(ts,  **kwargs)

    def get_wind(self, name="WIND"):
        """gets the wind speed time series from hdf5 TIMESRIES blocks"""
        return self._get_ts_by_tmemn(name)

    def set_wind(self, val):
        """sets the value of TIMESERIES"""
        self._set_ts_by_tmemn(val, "WIND")

    def plot_wind(self, start=None, end=None, **kwargs):
        ts = slice_ts(self.get_wind(), st=start, en=end)
        return plot(ts, **kwargs)

    def get_cloud(self, name="CLOUD"):
        """gets the CLOUD time series from hdf5 TIMESRIES blocks"""
        return self._get_ts_by_tmemn(name)

    def set_cloud(self, val):
        """sets the value of TIMESERIES"""
        self._set_ts_by_tmemn(val, "CLOUD")

    def plot_cloud(self, start=None, end=None, **kwargs):
        ts = slice_ts(self.get_cloud(), st=start, en=end)
        return plot(ts, **kwargs)

    def get_dew_temp(self, name="DEWTMP"):
        """gets the dew temperature time series from hdf5 TIMESRIES blocks"""
        return self._get_ts_by_tmemn(name)

    def set_dew_temp(self, val):
        """sets the value of TIMESERIES"""
        self._set_ts_by_tmemn(val, "DEWTMP")

    def plot_dew_temp(self, start=None, end=None, **kwargs):
        ts = slice_ts(self.get_dew_temp(), st=start, en=end)
        return plot(ts, **kwargs)

    def get_sol_rad(self, name="SOLRAD"):
        """
        gets the solar radiation time series from hdf5 TIMESRIES blocks
        """
        return self._get_ts_by_tmemn(name)

    def set_sol_rad(self, val):
        """sets the value of TIMESERIES"""
        self._set_ts_by_tmemn(val, "SOLRAD")

    def plot_sol_rad(self, start=None, end=None, **kwargs):
        ts = slice_ts(self.get_sol_rad(), st=start, en=end)
        return plot(ts, **kwargs)

    def get_pot_etp(self, name='PETINP'):
        """ First finds the path of precipitation timeseries in hdf5 and then
        gets precipitation timeseries from hdf5 file."""
        return self._get_ts_by_tmemn(name)

    def set_pot_etp(self, val):
        self._set_ts_by_tmemn(val, "PETINP")
        return

    def plot_pot_etp(self, start=None, end=None, **kwargs):
        """plots potential evapotranspiration"""
        ts = slice_ts(self.get_pot_etp(), st=start, en=end)
        return plot(ts, **kwargs)

    def get_air_temp(self, name='GATMP'):
        """ First finds the path of air temp timeseries in hdf5 and then
        gets precipitation timeseries from hdf5 file."""
        return self._get_ts_by_tmemn(name)

    def set_air_temp(self, val):
        self._set_ts_by_tmemn(val, "GATMP")
        return

    def plot_air_temp(self, start=None, end=None, **kwargs):
        """plots air temperature"""
        ts = slice_ts(self.get_air_temp(), st=start, en=end)
        return plot(ts, **kwargs)

    def _set_ts_by_svolno(self, value, svolno):
        """sets the value of a TIMESERIES"""
        esp = self.ext_src.loc[self.ext_src['SVOLNO'] == svolno]
        _ts = esp['SVOLNO'].unique()
        assert len(_ts) == 1
        ts_name = _ts.item()
        return self.set_ts_by_name(value, ts_name)

    def _set_ts_by_tmemn(self, value, tmemn):
        """sets the value of a TIMESERIES"""
        esp = self.ext_src.loc[self.ext_src['TMEMN'] == tmemn]
        _ts = esp['SVOLNO'].unique()
        assert len(_ts) == 1
        ts_name = _ts.item()
        return self.set_ts_by_name(value, ts_name)

    def set_ts_by_name(self, ts, name):
        """writes the time series at TIMESERIES/name"""
        if self.sim_len != len(ts):
            warnings.warn(f"""
            Given array has length {len(ts)} while sim length is {self.sim_len}""")
        elif isinstance(ts, pd.Series):
            if isinstance(ts.index, pd.DatetimeIndex):
                pass
            else:
                ts = pd.Series(ts, index=self.sim_idx)
        else:
            ts = pd.Series(ts, index=self.sim_idx)

        if ts.index.freq is None:
            ts.index.freq = pd.infer_freq(ts.index)

        assert ts.index.freq is not None

        ts_path = "TIMESERIES/" + name
        with pd.HDFStore(self.hdfname) as store:
            self.put(store, ts_path, ts)
        return

    def _get_ts_by_tmemn(self, tmemn):
        esp = self.ext_src.loc[self.ext_src['TMEMN'] == tmemn]
        mem_ts = esp['SVOLNO'].unique()
        assert len(mem_ts) == 1, mem_ts
        mem_ts = mem_ts.item()
        return self.get_ts_by_name(mem_ts)

    def _get_ts_by_svolno(self, svolno):
        esp = self.ext_src.loc[self.ext_src['SVOLNO'] == svolno]
        prec_ts = esp['SVOLNO'].unique()
        assert len(prec_ts) == 1, prec_ts
        prec_ts = prec_ts.item()
        return self.get_ts_by_name(prec_ts)

    def get_ts_by_name(self, name):
        with pd.HDFStore(self.hdfname) as store:
            pcp = store.get(f'TIMESERIES/{name}')

        return pcp[self.sim_start: self.sim_end]

    def get_ts_names(self):
        return self.ext_src.loc[:, 'SVOLNO'].unique()

    @property
    def freq(self):
        return self.op_seq['INDELT_minutes'].unique().item()

    @property
    def sim_idx(self)->pd.DatetimeIndex:
        return pd.date_range(self.sim_start, self.sim_end, freq=str(self.freq) + "min")

    @property
    def sim_len(self)->int:
        return len(self.sim_idx)

    @property
    def general(self)->dict:
        gen = {'sim_len': self.sim_len,
               'sim_delt': self.freq,
               'delt': self.freq,
               'steps': self.sim_len,
               'tindex': self.sim_idx,
               'sim_start': self.sim_start,
               'sim_end': self.sim_end,
               'units': self.units,
               }
        return gen

    def hru_area_fact(self, hru, rchres, afact_df=None):
        if hru not in list(self.perlnd_ids) + list(self.implnd_ids):
            raise ValueError
        if rchres not in self.rchres_ids:
            raise ValueError

        if afact_df is None:
            with pd.HDFStore(self.hdfname) as store:
                if "CONTROL/SCHEMATIC/TS" in store:
                    key = hru + rchres[0] + str(int(rchres[1:]))
                    return store["/CONTROL/SCHEMATIC/TS"][key].values
                else:
                    afactor = self.links.loc[self.links['SVOLNO'] == hru].loc[self.links['TVOLNO'] == rchres]['AFACTR']
                    if len(afactor) > 0:
                        return float(afactor)
                    else:
                        return 0.0
        else:
            key = hru + rchres[0] + str(int(rchres[1:]))
            return afact_df[key].values

    def ivol_lat(self, hru_name=None, afactor_df=None):
        """ fetches amount of lateral flow in a given rchres """
        ivol_ifwo = self.ivol_flow('PERLND', 'IFWO',
                                   hru_name=hru_name, afactor_df=afactor_df)
        ivol_agwo = self.ivol_flow('PERLND', 'AGWO',
                                   hru_name=hru_name, afactor_df=afactor_df)

        return ivol_ifwo + ivol_agwo

    def ivol_suro(self, hru_name=None, afactor_df=None):
        """ fetches amount of total surface runoff in a given rchres """
        ivol_isuro = self.ivol_flow('IMPLND', 'SURO',
                                    hru_name=hru_name, afactor_df=afactor_df)
        ivol_psuro = self.ivol_flow('PERLND', 'SURO',
                                    hru_name=hru_name, afactor_df=afactor_df)

        return ivol_isuro + ivol_psuro

    def ivol_flow(self, landuse_type, flow_name,
                  hru_name=None, rchres_id='all', afactor_df=None):
        """
        Returns a flow quantity/flow_name (e.g. SURO) calculated either from
        IWATER or from PWATER or from both
         hru_name: must be one of self.perlnd_ids or self.implnd_ids
         """
        if landuse_type not in ['PERLND', 'IMPLND']:
            raise ValueError

        if landuse_type == "PERLND":
            module = "PWATER"
        else:
            module = "IWATER"

        if flow_name not in ['SURO', 'PERO', 'AGWO', 'IFWO']:
            raise ValueError

        if rchres_id != 'all':
            if rchres_id not in self.rchres_ids:
                raise ValueError("unknown rchres id {}".format(rchres_id))
            rchres_ids = [rchres_id]
        else:
            rchres_ids = self.rchres_ids

        if hru_name is None:
            hrus = self.perlnd_ids if landuse_type == 'PERLND' else self.implnd_ids
        else:
            if hru_name not in list(list(self.perlnd_ids) + list(self.implnd_ids)):
                raise ValueError
            hrus = [hru_name]

        ivol_flows = []
        for hru in hrus:
            for rch in rchres_ids:
                ivol_flow = self.hru_flow(
                    landuse_type, module, flow_name, hru) * 0.08333 * self.hru_area_fact(
                    hru, rch, afact_df=afactor_df)

                ivol_flows.append(ivol_flow)

        return np.sum(ivol_flows, axis=0)

    def get_uc(self, rchres_id):

        if rchres_id not in self.rchres_ids:
            raise ValueError

        ui = {
              'rchtab': self.ftable(rchres_id),
              'CONVF': 1.0
              }

        for flg in ['ICAT', 'ODFVF', 'COLIN', 'OUTDG', 'AUX1FG', 'ODGTF',
                    'FUNCT', 'VOL', 'VCONFG', 'AUX2FG', 'AUX3FG']:
            ui[flg] = self.rchres_flags(flg, rchres_id)

        for para_name in ['KS', 'IREXIT', 'IRMINV', 'LEN', 'DB50', 'DELTH',
                          'FTBUCI', 'FTBW', 'STCOR']:
            ui[para_name] = self.rchres_paras(para_name, rchres_id)

        for ginf in ['NEXITS',  'LKFG', 'BUNITE', 'BUNITM', 'PUNITE',
                     'PUNITM', 'RCHID', 'LANDUSE', 'OUNITS', 'IUNITS']:
            ui[ginf] = self.rchres_gen_info(ginf, rchres_id)

        return ui

    def rchres_gen_info(self, para_name, rchres_id='general'):
        """
        gets number of exits.
        :param rchres_id:
        :param para_name:
        :return:
        """

        with pd.HDFStore(self.hdfname) as store:
            geninfo_df = store.get("RCHRES/GENERAL_INFO")
        if para_name not in geninfo_df.columns:
            raise ValueError("""{} does not exist in {} of {} file"""
                             .format(para_name, "RCHRES/GENERAL_INFO", self.hdfname))

        if rchres_id != 'general':
            if rchres_id not in self.rchres_ids:
                raise ValueError("unknown rchres id")
        else:
            rchres_id = self.rchres_ids[0]

        gen_info = geninfo_df[para_name][rchres_id]

        gen_info = int(gen_info) if para_name != 'LANDUSE' else gen_info

        return gen_info

    def rchres_flags(self, flag_name='ODFVF', rchres_id='general'):

        if flag_name not in ['ODFVF', 'COLIN', 'OUTDG', 'ODGTF', 'AUX1FG', 'AUX2FG',
                             'AUX3FG', 'FUNCT', 'VOL', 'VCONFG', 'ICAT']:
            raise ValueError("{} does not exist in {} of {} file".format(flag_name, "RCHRES/HYDR/FLAGS", self.hdfname))

        if rchres_id != 'general':
            if rchres_id not in self.rchres_ids:
                raise ValueError("unknown rchres id")
            else:
                rchres_id = rchres_id
        else:
            rchres_id = self.rchres_ids[0]

        if flag_name in ['ODFVF', 'COLIN', 'OUTDG', 'ODGTF', 'FUNCT']:
            flags = [flag_name + str(i+1) for i in range(5)]

            with pd.HDFStore(self.hdfname) as store:
                vals = store.get("RCHRES/HYDR/FLAGS")[flags].loc[rchres_id].values
            return vals
        else:
            flags = [flag_name]

            with pd.HDFStore(self.hdfname) as store:
                vals = store.get("RCHRES/HYDR/FLAGS")[flags].loc[rchres_id].values

            if flag_name in ['VOL', 'ICAT']:
                return float(vals)
            else:
                return int(vals)

    def rchres_paras(self, para_name, rchres_id):
        with pd.HDFStore(self.hdfname) as store:
            paras = store.get("RCHRES/HYDR/PARAMETERS")
        if para_name not in paras.columns:
            raise ValueError("""{} does not exist in {} of {} file. Paras are {}
            """.format(para_name, "RCHRES/HYDR/PARAMETERS", self.hdfname, paras))

        if rchres_id not in self.rchres_ids:
            raise ValueError

        val = paras[para_name].loc[rchres_id]

        if para_name in ['FTBUCI']:
            return str(val)
        elif para_name in ['IREXIT']:
            return int(val)
        else:
            return float(val)

    def ftable(self, rchres_id):
        if rchres_id not in self.rchres_ids:
            raise ValueError

        table_id = 'FT' + rchres_id[1:]

        with pd.HDFStore(self.hdfname) as store:
            ftable = store.get(f"FTABLES/{table_id}")
        return ftable

    def get_group_hydr(self, flow_name, rchres_id='last'):

        rchres_id = self._check_rchres_id(rchres_id)

        path = "/RESULTS/" + "RCHRES_" + rchres_id + "/HYDR"
        saved_flows = pd.read_hdf(self.hdfname, path).columns

        if flow_name not in saved_flows:
            raise ValueError("Flow {} not found at {} in {}".format(flow_name, path, self.hdfname))
        else:
            return pd.read_hdf(self.hdfname, path)[flow_name]

    def get_flow_by_rerun(self, flow_name='lat', rchres_id='last', afactor_df=None):
        """
         calculates lateral flow or surface runoff contribution to total discharge
         by first calculating IVOL and
        then performing channel routing i.e. by running HYDR module for the given IVOL.
        """
        rchres_id = self._check_rchres_id(rchres_id)

        if flow_name not in ['suro', 'lat']:
            raise ValueError

        path = f"/RESULTS/RCHRES_{rchres_id}/HYDR"

        with pd.HDFStore(self.hdfname) as store:
            saved_flows = store.get(path)
            saved_flows = saved_flows.columns

        #saved_flows = pd.read_hdf(self.hdfname, path).columns
        saved_flow_name = 'ro_' + flow_name

        if saved_flow_name not in saved_flows:
            #ui = self.get_uc(rchres_id)
            ui = self.uci_obj.uci[("RCHRES", "HYDR", rchres_id)]

            _func = 'ivol_' + flow_name
            ivol = getattr(self, _func)(afactor_df=afactor_df)
            ts = {'IVOL': ivol}

            uc_ = ui.copy()
            tss = ts.copy()

            if 'PARAMETERS' not in uc_:
                uc_['PARAMETERS'] = {}
            uc_['PARAMETERS']['NEXITS'] = self.rchres_gen_info("NEXITS", rchres_id)

            _, _ = hydr('', siminfo=self.general, uci=uc_, ts=tss,
                        ftables=self.ftable(rchres_id))
            sim_flow = tss['RO']

            with pd.HDFStore(self.hdfname) as store:
                saved_flows = store.get(path)

                saved_flows[saved_flow_name] = sim_flow

                store.put(path, saved_flows, data_columns=saved_flows.columns)
        else:
            with pd.HDFStore(self.hdfname) as store:
                sim_flow = store.get(path)[saved_flow_name]

        return sim_flow

    def _check_rchres_id(self, rchres_id):
        if rchres_id != 'last':
            if rchres_id not in self.rchres_ids:
                raise ValueError("invalid rchres name {}".format(rchres_id))
        else:
            rchres_id = self.rchres_ids[-1]
        return rchres_id

    def sim_lat_cfs(self, rchres_id='last', afactor_df=None):

        return self.get_flow_by_rerun('lat', rchres_id, afactor_df=afactor_df)

    def sim_lat_cms(self, rchres_id='last', afactor_df=None):
        mfac = cfs_to_cms if int(self.units) == 1 else 1.0
        return self.sim_lat_cfs(rchres_id=rchres_id, afactor_df=afactor_df) * mfac

    def sim_suro_cfs(self, rchres_id='last', afactor_df=None):

        return self.get_flow_by_rerun('suro', rchres_id, afactor_df=afactor_df)

    def sim_suro_cms(self, rchres_id='last', afactor_df=None):
        mfac = cfs_to_cms if int(self.units) == 1 else 1.0
        return self.sim_suro_cfs(rchres_id=rchres_id, afactor_df=afactor_df) * mfac

    def sim_q_cfs(self, rch_no):
        mfac = 1.0 if int(self.units) == 1 else cfs_to_cms
        rch = str(rch_no).rjust(3, '0')

        with pd.HDFStore(self.hdfname) as store:
            sim_q = store.get('RESULTS/RCHRES_R' + rch + '/HYDR')['RO'] * mfac

        return sim_q[pd.Timestamp(self.sim_start):pd.Timestamp(self.sim_end)]

    def sim_q_cms(self, rchres_id='last'):

        if rchres_id != 'last':
            if rchres_id not in self.rchres_ids:
                raise ValueError("invalid rchres name {}".format(rchres_id))
        else:
            rchres_id = self.rchres_ids[-1]

        mfac = 1.0 if int(self.units) == 2 else cfs_to_cms
        # rch = str(rchres_id).rjust(3, '0')

        with pd.HDFStore(self.hdfname) as store:
            sim_q = store.get(f"RESULTS/RCHRES_{rchres_id}/HYDR")["RO"] * mfac

        return sim_q[pd.Timestamp(self.sim_start):pd.Timestamp(self.sim_end)]

    @property
    def phru_ids(self):
        return [pid[1:] for pid in self.perlnd_ids]

    @property
    def ihru_ids(self):
        return [pid[1:] for pid in self.implnd_ids]

    @property
    def hru_ids(self):
        return np.unique(self.phru_ids + self.ihru_ids)

    def hru_ivol(
            self,
            flow_name:str,
            hru_id:str,
            rchres_id='all',
            module='water', afactor_df=None):
        """
        Example
        -------
        >>> h5 = HdfFile(...)
        >>> h5.hru_ivol('SURO', '101')
        """
        assert hru_id in self.hru_ids

        phru = f"P{hru_id}"
        ihru = f"I{hru_id}"

        flow_name = flow_name.upper()

        if module != 'water':
            raise NotImplementedError

        ivol_pflow = self.ivol_flow('PERLND', flow_name, hru_name=phru,
                                    rchres_id=rchres_id,
                                    afactor_df=afactor_df)
        ivol_iflow = self.ivol_flow('IMPLND', flow_name, hru_name=ihru,
                                    rchres_id=rchres_id,
                                    afactor_df=afactor_df)

        return ivol_pflow + ivol_iflow

    def hru_flow(
            self,
            hru_type:str='PERLND',
            module:str='PWATER',
            flow_name:str='PERO',
            p_id='all')->np.ndarray:
        """
        Example
        --------

        """
        if hru_type not in ['PERLND', 'IMPLND', 'RCHRES']:
            raise ValueError("invalid hru name {}".format(hru_type))

        _pid = {'PERLND': self.perlnd_ids,
                'IMPLND': self.implnd_ids,
                'RCHRES': self.rchres_ids}

        if p_id == 'all':
            p_id = _pid[hru_type]
        elif p_id not in list(list(self.implnd_ids) + list(self.perlnd_ids) + list(self.rchres_ids)):
            raise ValueError("pid {} not name of any hru".format(p_id))
        else:
            p_id = [p_id]

        if module not in ['PWATER', 'IWATER', 'HYDR']:
            raise ValueError("invalid module name {}".format(module))

        flows = []
        for pid in p_id:

            with pd.HDFStore(self.hdfname) as store:
                flow = store.get(f"RESULTS/{hru_type}_{pid}/{module}")[flow_name].values

            #flow = pd.read_hdf(self.hdfname, "RESULTS/" + hru_type + "_" + pid + "/" + module)[flow_name].values
            flows.append(flow)

        return np.sum(flows, axis=0)

    def obs_lat_cfs(self):
        mfac = 1.0 if int(self.units) == 1 else lps_to_cfs
        return self.obs_df['sub_lps'][self.sim_start:self.sim_end] * mfac

    def obs_lat_cms(self):
        mfac = 1.0 if int(self.units) == 2 else lps_to_cms
        return self.obs_df['sub_lps'][self.sim_start:self.sim_end] * mfac

    def obs_suro_cfs(self):
        mfac = 1.0 if int(self.units) == 1 else lps_to_cfs
        return self.obs_df['ovf_lps'][self.sim_start:self.sim_end] * mfac

    def obs_suro_cms(self):
        mfac = 1.0 if int(self.units) == 2 else lps_to_cms
        return self.obs_df['ovf_lps'][self.sim_start:self.sim_end] * mfac

    def obs_q_cfs(self):
        return self.obs_df['q_lps'][self.sim_start:self.sim_end] * lps_to_cfs

    def obs_q_cms(self):
        return self.obs_df['q_lps'][self.sim_start:self.sim_end] * lps_to_cms

    # def plot_hru_flow(self, hru_name, flow_name='SURO', save_name=None, st=None, en=None, ylim=None):
    #     """ Plots flow for an HRU in cms """
    #     flow_name = flow_name.upper()
    #
    #     hru_flow = self.hru_ivol(flow_name, hru_name) * cfs_to_cms
    #     hru_flow = pd.DataFrame(hru_flow, index=self.sim_idx, columns=[flow_name])
    #
    #     fig, (ax1, ax2) = plt.subplots(2, sharex='all')
    #     fig.set_figwidth(14)
    #     fig.set_figheight(9)
    #
    #     ax1.plot(self.prec[st:en] * in_to_mm, '-', markersize=2 ,color='b', label='Precipitation')
    #     ax1.set_ylim(ax1.get_ylim()[::-1])
    #     set_axis_props(ax1, 'Milimeters', 'Time', 'k', True, "center left", 20, 10)
    #     ax1.get_xaxis().set_visible(False)
    #     ax1.spines['bottom'].set_visible(False)
    #
    #     ax2.plot(hru_flow[st:en], '.', markersize=3, color='r', label='Simulated ' + str(flow_name))
    #     set_axis_props(ax2, 'CMS', 'Time', 'k', False, "upper left", 20, 10, y_max_lim=ylim)
    #
    #     plt.subplots_adjust(wspace=0.05, hspace=0.01)
    #     if save_name:
    #         plt.savefig(save_name + '.png', dpi=300, bbox_inches='tight')
    #     plt.show()

    def get_dqals(self, rchres:list, start=None, end=None)->pd.DataFrame:
        """Gets Dissolved concentration of qual from multiple RCHRES."""
        assert isinstance(rchres, list)
        dqals = []
        for rch in rchres:
            dqal =self.get_dqal(rch, start, end)
            # replace values below -1e30 with nans
            dqal[dqal.loc[dqal<-1e30].index] = np.nan
            dqals.append(dqal)

        df = pd.concat(dqals, axis=1)
        df.columns = rchres
        return df

    def get_tiqals(self, rchres:list, start=None, end=None)->pd.DataFrame:
        """Gets Total inflow of qual in multiple RCHRES."""
        assert isinstance(rchres, list)
        tiqals = []
        for rch in rchres:
            tiqals.append(self.get_tiqal(rch, start, end))

        tiqals = pd.concat(tiqals, axis=1)
        tiqals.columns = rchres
        return tiqals

    def get_troqals(self, rchres:list, start=None, end=None)->pd.DataFrame:
        """Gets Total outflow of qual from multiple RCHRES."""
        assert isinstance(rchres, list)
        troqals = []
        for rch in rchres:
            troqals.append(self.get_troqal(rch, start, end))

        troqals = pd.concat(troqals, axis=1)
        troqals.columns = rchres
        return troqals

    def get_rodqals(self, rchres:list, start=None, end=None)->pd.DataFrame:
        """Get Total outflow of dissolved qual from multiple RCHRES."""
        assert isinstance(rchres, list)
        rodqals = []
        for rch in rchres:
            rodqals.append(self.get_rodqal(rch, start, end))

        rodqals = pd.concat(rodqals, axis=1)
        rodqals.columns = rchres
        return rodqals

    def get_idqals(self, rchres:list, start=None, end=None)->pd.DataFrame:
        """Get Sum of inflows of dissolved qual from multiple RCHRES."""
        assert isinstance(rchres, list)
        idqals = []
        for rch in rchres:
            idqals.append(self.get_idqal(rch, start, end))

        idqals = pd.concat(idqals, axis=1)
        idqals.columns = rchres
        return idqals

    def get_rdqals(self, rchres:list, start=None, end=None)->pd.DataFrame:
        """Get Total storage of qual in dissolved form from multiple RCHRES."""
        assert isinstance(rchres, list)
        rdqals = []
        for rch in rchres:
            rdqals.append(self.get_rdqal(rch, start, end))

        rdqals = pd.concat(rdqals, axis=1)
        rdqals.columns = rchres
        return rdqals

    def plot_dqals(self, rches:list, start=None, end=None, **kwargs):
        """Plots Dissolved concentration of qual from multiple RCHRES."""
        dqals = self.get_dqals(rches, start, end)
        def_ax_kws = dict(title="Dissolved concentration of qual in RCHRES (DQAL)")
        ax_kws = kwargs.pop('ax_kws', def_ax_kws)
        return plot(dqals, ax_kws=ax_kws, **kwargs)

    def plot_rodqals(self, rches:list, start=None, end=None, **kwargs):
        """plots Total outflow of dissolved qual from multiple RCHRES."""
        rodqals = self.get_rodqals(rches, start, end)
        def_ax_kws = dict(title="Total outflow of dissolved qual (RODQAL)")
        ax_kws = kwargs.pop('ax_kws', def_ax_kws)
        return plot(rodqals, ax_kws=ax_kws, **kwargs)

    def plot_tiqals(self, rches:list, start=None, end=None, **kwargs):
        """Plots Total inflow of qual in multiple RCHRES."""
        tiqals = self.get_tiqals(rches, start, end)
        def_ax_kws = dict(title="Total inflow of qual (TIQAL)")
        ax_kws = kwargs.pop('ax_kws', def_ax_kws)
        return plot(tiqals, ax_kws=ax_kws, **kwargs)

    def plot_troqals(self, rches:list, start=None, end=None, **kwargs):
        """Plots Total outflow of qual from multiple RCHRES."""
        troqals = self.get_troqals(rches, start, end)
        def_ax_kws = dict(title="Total outflow of qual (TROQAL)")
        ax_kws = kwargs.pop('ax_kws', def_ax_kws)
        return plot(troqals, ax_kws=ax_kws, **kwargs)

    def plot_idqals(self, rches:list, start=None, end=None, **kwargs):
        """Plots Total inflow of dissolved qual in multiple RCHRES."""
        idqals = self.get_idqals(rches, start, end)
        def_ax_kws = dict(title="Total inflow of dissolved qual (IDQAL)")
        ax_kws = kwargs.pop('ax_kws', def_ax_kws)
        return plot(idqals, ax_kws=ax_kws, **kwargs)

    def plot_rdqals(self, rches:list, start=None, end=None, **kwargs):
        """Plots Total storage of dissolved qual in multiple RCHRES."""
        rdqals = self.get_rdqals(rches, start, end)
        def_ax_kws = dict(title="Total storage of dissolved qual (RDQAL)")
        ax_kws = kwargs.pop('ax_kws', def_ax_kws)
        return plot(rdqals, ax_kws=ax_kws, **kwargs)

    def get_rodqal(self, rchres, start=None, end=None):
        """Total outflow of dissolved qual from the RCHRES."""
        return self.get_gquals("GQUAL1_RODQAL", rchres, start, end)

    def get_rdqal(self, rchres, start=None, end=None)->pd.Series:
        """Get Total storage of qual in dissolved from the RCHRES."""
        return self.get_gquals("GQUAL1_RDQAL", rchres, start, end)

    def get_idqal(self, rchres, start=None, end=None)->pd.Series:
        """Get Sum of inflows of dissolved qual from the RCHRES."""
        return self.get_gquals("GQUAL1_IDQAL", rchres, start, end)

    def get_dqal(self, rchres:str, start=None, end=None)->pd.Series:
        """Dissolved concentration of qual from a RCHRES."""
        return self.get_gquals("GQUAL1_DQAL", rchres, start, end)

    def get_tiqal(self, rchres, start=None, end=None)->pd.Series:
        """ Gets Total inflow of qual in RCHRES."""
        return self.get_gquals("GQUAL1_TIQAL", rchres, start, end)

    def get_troqal(self, rchres, start=None, end=None)->pd.Series:
        """ Gets Total outflow of qual from RCHRES."""
        return self.get_gquals("GQUAL1_TROQAL", rchres, start, end)

    def get_msts(
            self,
            perlnd_id:str,
            start=None,
            end=None
    )->pd.DataFrame:
        """
        Get water content in surface, upper principle, upper auxiliary
        lower and groundwater storages from a perlnd_id as pandas Dataframe
        """
        assert perlnd_id in self.perlnd_ids

        with pd.HDFStore(self.hdfname) as store:
            df = store.get(f"RESULTS/PERLND_{perlnd_id}/MSTLAY")
            df = df[['MST1', 'MST2', 'MST3', 'MST4', 'MST5']]
        return slice_ts(df, start, end)

    def get_fracs(
            self,
            perlnd_id:str,
            start=None,
            end=None
    )->pd.DataFrame:
        """Gets the fractional fluxes through soil from a perlnd id"""

        assert perlnd_id in self.perlnd_ids

        with pd.HDFStore(self.hdfname) as store:
            df = store.get(f"RESULTS/PERLND_{perlnd_id}/MSTLAY")
            df = df[['FRAC1', 'FRAC2', 'FRAC3', 'FRAC4', 'FRAC5', 'FRAC6', 'FRAC7',
                     'FRAC8']]
        return slice_ts(df, start, end)

    def plot_fracs(
            self,
            perlnd_ids,
            start=None,
            end=None,
            share_axes=False,
            **kwargs
    )->plt.Axes:

        df = self.get_fracs(perlnd_ids, start, end)
        return plot(df, share_axes=share_axes, **kwargs)

    def plot_msts(
            self,
            perlnd_id:str,
            start=None,
            end=None,
            share_axes:bool = False,
            **kwargs
    )->plt.Axes:
        """
        Parameters
        -----------
        perlnd_id : str
            the perlnd id from which MSTs are to be fetched
        start :
        end :
        share_axes : bool
            whether to plot all MSTs on same axes or on separate axes
        """
        df = self.get_msts(perlnd_id, start, end)
        return plot(df, share_axes=share_axes, **kwargs)

    def plot_all_popst(
            self,
            start=None,
            end=None,
            **kwargs
    ):
        """
        plots sum of popst (Total outflow of solution pesticide) from all
        perlnds

        parameters
        ----------
        start :
            start time
        end :
            end time
        """
        df = self.get_all_popst(start, end)
        return plot(df, **kwargs)

    def get_all_popst(
            self,
            start=None,
            end=None,
    )->pd.Series:
        """
        gets sum of popst (Total outflow of solution pesticide) from all perlnds

        """
        with pd.HDFStore(self.hdfname) as store:
            dfs = []
            for perlnd_id in self.perlnd_ids:
                df = store.get(f"RESULTS/PERLND_{perlnd_id}/PEST")
                dfs.append(df['POPST'])

            df = pd.concat(dfs, axis=1)
            df = df.sum(axis="columns")
        return slice_ts(df, start, end)

    def get_gquals(self, name:str, rchres:str, start=None, end=None)->pd.Series:
        """Get qual associated time series from the RCHRES."""
        with pd.HDFStore(self.hdfname) as store:
            df = store.get(f"RESULTS/RCHRES_{rchres}/GQUAL")

        gqual = df[name]
        if start is None:
            start = self.sim_start

        if end is None:
            end = self.sim_end

        gqual = gqual.loc[pd.Timestamp(start):pd.Timestamp(end)]
        return gqual

    def get_total_seds_in_rchres(
            self,
            rchres_id:str = "last",
            start=None,
            end=None):
        """gets the SSED4 (total suspended sediments in a rchres
        """
        if rchres_id != 'last':
            if rchres_id not in self.rchres_ids:
                raise ValueError("invalid rchres name {}".format(rchres_id))
        else:
            rchres_id = self.rchres_ids[-1]

        return self.get_sediments("SSED4", rchres_id, start, end)

    def get_sediments(
            self,
            name:str,
              rchres:str,
              start=None,
              end=None
              )->pd.Series:
        """Get qual associated time series from the RCHRES."""
        with pd.HDFStore(self.hdfname) as store:
            df = store.get(f"RESULTS/RCHRES_{rchres}/SEDTRN")

        sediments = df[name]
        if start is None:
            start = self.sim_start

        if end is None:
            end = self.sim_end

        sediments = sediments.loc[pd.Timestamp(start):pd.Timestamp(end)]
        return sediments

    def plot_rodqal(self, rchres:str, start=None, end=None, **kwargs):

        rodqal = self.get_rodqal(rchres, start, end)
        return plot(rodqal, **kwargs)

    def plot_dqal(self, rchres:str, start=None, end=None, **kwargs):

        dqal = self.get_dqal(rchres, start, end)
        return plot(dqal, **kwargs)

    def plot_tiqal(self, rchres:str, start=None, end=None, **kwargs):

        tiqal = self.get_tiqal(rchres, start, end)
        return plot(tiqal, **kwargs)

    def plot_troqal(self, rchres:str, start=None, end=None, **kwargs):

        tiqal = self.get_troqal(rchres, start, end)
        return plot(tiqal, **kwargs)

    def plot_flows(self, flow_name, save_name=None, st=None, en=None, ylim=None):

        if st is None:
            st = self.sim_start
        if en is None:
            en = self.sim_end

        if not isinstance(flow_name, list):
            if flow_name not in ['suro', 'lat', 'q']:
                raise ValueError
            flow_name = [flow_name]  # because onely one value is given
        else:
            for content in flow_name:
                if content not in ['suro', 'lat', 'q']:
                    raise ValueError

        fig, (ax1, ax2) = plt.subplots(2, sharex='all')
        fig.set_figwidth(14)
        fig.set_figheight(9)

        ax1.plot(self.prec[st:en] * in_to_mm, '-', markersize=2,
                 color='b', label='Precipitation')
        set_axis_props(ax1, y_label='Milimeters', x_label='Time',
                       c='k', top_spine=True, leg_pos="center left",
                       leg_fs=20, leg_ms=10, show_xaxis=False,
                       bottom_spine=False, invert_y_axis=True)

        c = np.array([0.23495847, 0.28413737, 0.6598559])

        for flow in flow_name:
            _units = "cms"
            sim_func = "sim_" + flow + "_" + _units
            sim_flow = getattr(self, sim_func)()
            sim_flow = pd.Series(sim_flow, index=self.sim_idx, name='sim_' + flow)
            obs_func = "obs_" + flow + "_" + _units
            obs_flow = getattr(self, obs_func)()
            obs_flow = pd.Series(obs_flow, index=self.sim_idx, name='sim_' + flow)

            ax2.plot(obs_flow[st:en], '.', markersize=1, color=c,
                     label='Observed ' + plot_props[flow][0])
            ax2.plot(sim_flow[st:en], '.', markersize=1, color='r',
                     label='Simulated ' + plot_props[flow][0])
            set_axis_props(ax2, 'CMS', 'Time', 'k', False, "upper left",
                           20, 10, y_max_lim=ylim)

        plt.subplots_adjust(wspace=0.05, hspace=0.01)
        if save_name:
            plt.savefig(save_name + '.png', dpi=300, bbox_inches='tight')
            plt.close('all')
        else:
            plt.show()

    def sim_dqal(self, rchres_id='last'):
        if rchres_id != 'last':
            if rchres_id not in self.rchres_ids:
                raise ValueError("invalid rchres name {}".format(rchres_id))
        else:
            rchres_id = self.rchres_ids[-1]

        return pd.read_hdf(self.hdfname, '/RESULTS/RCHRES_' + rchres_id + '/GQUAL/F_Coli')['dqal'] * 10.0

    def sim_dqal_load(self, rchres_id='last'):

        return self.sim_dqal(rchres_id) * self.sim_q_cms(rchres_id)

    def sim_qual(self, qual_name='idqal', rchres_id='last'):

        if rchres_id != 'last':
            if rchres_id not in self.rchres_ids:
                raise ValueError("invalid rchres name {}".format(rchres_id))
        else:
            rchres_id = self.rchres_ids[-1]

        if qual_name not in ['idqal', 'rodqal']:
            raise ValueError("unknown qual property {}".format(qual_name))

        return pd.read_hdf(self.hdfname, '/RESULTS/RCHRES_' + rchres_id + '/GQUAL/F_Coli')[qual_name] * 0.0353

    def switch_off_wq(
            self,
            dependencies:bool = False):
        """swithc off pqual, gqual, and iqual
        parameters
        ----------
        dependencies : bool (default=False)
            if True, then all the modules which are required to run
            wq are also turned off. This includes CONS, ADCALC, SEDTRN
            in RCHRES, SOLIDS in IMPLND and SEDMNT in PERLND
        """
        self.switch_off_iqual()
        self.switch_off_pqual()
        self.switch_off_gqual()

        if dependencies:
            self.switch_off_cons()
            # self.switch_off_adcalc() # todo
            self.switch_off_sediment_simulation(True)
        return

    def switch_off_pqual(self):
        path = "PERLND/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "PQUAL")

    def switch_off_iqual(self):
        path = "IMPLND/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "IQUAL")

    def switch_off_sedmnt(self):
        path = "PERLND/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "SEDMNT")

    def switch_off_solids(self):
        path = "IMPLND/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "SOLIDS")

    def switch_off_gqual(self):
        path = "RCHRES/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "GQUAL")

    def switch_on_gqual(self):
        path = "RCHRES/GENERAL/ACTIVITY"
        self.switch_on_actiivty(path, "GQUAL")

    def switch_off_adcalc(self):
        path = "RCHRES/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "ADCALC")
        return

    def switch_on_adcalc(self):
        path = "RCHRES/GENERAL/ACTIVITY"
        self.switch_on_actiivty(path, "ADCALC")
        return

    def switch_on_pstemp(self):
        path = "PERLND/GENERAL/ACTIVITY"
        self.switch_on_actiivty(path, "PSTEMP")
        return

    def switch_off_pstemp(self):
        path = "PERLND/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "PSTEMP")
        return

    def switch_on_atemp(self):
        path = "PERLND/GENERAL/ACTIVITY"
        self.switch_on_actiivty(path, "ATEMP")
        return

    def switch_off_atemp(self):
        path = "PERLND/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "ATEMP")
        return

    def switch_on_mstlay(self):
        path = "PERLND/GENERAL/ACTIVITY"
        self.switch_on_actiivty(path, "MSTLAY")
        return

    def switch_off_pest(
            self,
            dependencies:bool = False
    ):
        path = "PERLND/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "PEST")
        if dependencies:
            self.switch_off_pstemp()

        return

    def switch_on_pest(self):
        """this will turn on PEST module of PERLND in all PERLNDs"""
        path = "PERLND/GENERAL/ACTIVITY"
        self.switch_on_actiivty(path, "PEST")

        self.switch_on_atemp()  # calculates air temperature which is required for pstemp
        self.switch_on_pstemp()  # calculates temperature of soil layers which is required for pest
        self.switch_on_mstlay()
        return

    def switch_off_mstlay(self):
        path = "PERLND/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "MSTLAY")
        return

    def switch_off_sediment_simulation(
            self,
            dependencies:bool = False
    ):
        """makse sure that modules for sediment simulation are not run
         in PERLND, IMPLND and RCHRES."""
        self.switch_off_actiivty("IMPLND/GENERAL/ACTIVITY", "SOLIDS")
        self.switch_off_actiivty("PERLND/GENERAL/ACTIVITY", "SEDMNT")
        self.switch_off_actiivty("RCHRES/GENERAL/ACTIVITY", "SEDTRN")
        if dependencies:
            self.switch_off_actiivty("RCHRES/GENERAL/ACTIVITY", "ADCALC")
        return

    def switch_off_cons(self):
        path = "RCHRES/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "CONS")
        return

    def switch_off_htrch(self):
        path = "RCHRES/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "HTRCH")
        return

    def switch_off_actiivty(self, path, module_name):
        # switching off module
        with pd.HDFStore(self.hdfname) as store:
            activity = store.get(path)

            assert module_name in activity

            activity.loc[:, module_name] = 0
            self.put(store, path, activity)
        return

    def switch_on_actiivty(self, path, module_name):
        # switching off module
        with pd.HDFStore(self.hdfname) as store:
            activity = store.get(path)

            assert module_name in activity

            activity.loc[:, module_name] = 1
            self.put(store, path, activity)
        return

    def switch_on_sedtrn(self):
        path = "RCHRES/GENERAL/ACTIVITY"
        self.switch_on_actiivty(path, "SEDTRN")
        return

    def switch_off_sedtrn(self):
        path = "RCHRES/GENERAL/ACTIVITY"
        self.switch_off_actiivty(path, "SEDTRN")
        return

    def switch_on_solids(self):
        path = "IMPLND/GENERAL/ACTIVITY"
        self.switch_on_actiivty(path, "SOLIDS")
        return

    def switch_on_sedmnt(self):
        path = "PERLND/GENERAL/ACTIVITY"
        self.switch_on_actiivty(path, "SEDMNT")
        return

    def swith_on_sediment_simulation(self):
        """makes the sediment simulation module on in PERLND/IMPLND and RCHRES"""
        self.switch_on_solids()
        self.switch_on_sedmnt()
        self.switch_on_sedtrn()
        return

def set_axis_props(axis,
                   y_label,
                   x_label,
                   c,
                   top_spine,
                   leg_pos,
                   leg_fs,
                   leg_ms,
                   xl_fs=14,
                   y_max_lim=None,
                   show_xaxis=True,
                   bottom_spine=True,
                   invert_y_axis=False,
                   verbosity=0):
    if y_max_lim:
        axis.set_ylim(0.0, y_max_lim)

    axis.legend(loc=leg_pos, fontsize=leg_fs, markerscale=leg_ms)

    if invert_y_axis:
        axis.set_ylim(axis.get_ylim()[::-1])

    axis.set_ylabel(y_label, fontsize=20, color=c)
    axis.set_xlabel(x_label, fontsize=20)
    axis.tick_params(axis="y", which='major', labelsize=20, colors=c)
    axis.tick_params(axis="x", which='major', labelsize=20)

    axis.get_xaxis().set_visible(show_xaxis)
    if show_xaxis:
        axis.set_xlabel(x_label, fontsize=xl_fs)

    axis.spines['top'].set_visible(top_spine)
    axis.spines['bottom'].set_visible(bottom_spine)

    loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
    axis.xaxis.set_major_locator(loc)
    fmt = mdates.AutoDateFormatter(loc)
    axis.xaxis.set_major_formatter(fmt)
    return


def slice_ts(
        ts:Union[pd.Series, pd.DataFrame],
        st = None,
        en = None
)->Union[pd.Series, pd.DataFrame]:
    """slice the time series from st and end"""
    if st is not None:
        # integer
        if isinstance(st, int):
            # end is given
            if en is not None:
                ts = ts.iloc[st:en]
            else:
                ts = ts.iloc[st:]

        else:   # label based
            if en is not None:  # end is given
                ts = ts.loc[st:en]
            else:
                ts = ts.loc[st]

    elif en is not None: # st is not given but en is given
        if isinstance(en, int):  # integer
            if st is not None:   # start is given
                ts = ts.iloc[st:en]
            else:
                ts = ts.iloc[:en]

        else:  # label based
            if st is not None: # start is given
                ts = ts.loc[st:en]
            else:
                ts = ts.loc[:en]

    return ts