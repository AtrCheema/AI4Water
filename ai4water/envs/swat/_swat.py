
import os
import datetime
from typing import Union, List

from ai4water.backend import np, pd, plt
from ai4water.utils.utils import dateandtime_now

from ._global_vars import DEF_WQL_COLUMNS, RES_COLUMNS, DEF_RCH_COLUMNS, \
    RCH_OUT_CODES, HRU_OUT_CODES, SUB_OUT_CODES, RCH_COL_MAP


class SWAT(object):
    """interface to swat model

    parameters
    -----------
    path : str
        path where SWAT files are located included the executable.
    weir_codes : list
        The codes for weirs

    """
    def __init__(
            self,
            path,
            weir_codes:List[int] = None
    ):

        self.path = path
        self.weir_codes = weir_codes

        self._rch_cols = self._output_rch_cols()  # todo

    def get_wq_rch(self, wq_name, rch_id):
        """get wq data for weir"""
        if not isinstance(wq_name, list):
            wq_name = [wq_name]
        if not isinstance(rch_id, list):
            rch_id = [rch_id]

        df = self.read_wql_output(rch_id)

        return df[wq_name + ["rch_id"]]

    def read_wql_output(self,
                        rch_id:Union[str, int, list]
                        )->pd.DataFrame:

        if not isinstance(rch_id, list):
            rch_id = [rch_id]

        # read SWAT results
        fname = os.path.join(self.path, "output.wql")

        with open(fname, 'r') as f:
            lines = f.readlines()

        lines = [line.split() for line in lines]

        pred = pd.DataFrame(lines[1:], columns=DEF_WQL_COLUMNS)

        pred = pred[[
            "rch_id", "day", 'WTEMP(C)', 'ALGAE_INppm', "ALGAE_Oppm", "H20VOLUMEm3",
            "NH4_OUTppm", "NO2_OUTppm", "NO3_OUTppm", "ORGN_OUTppm", "ORGP_OUTppm", "SOLP_OUTppm",
            "CBOD_OUTppm"]]

        pred = pred.astype({
            "rch_id": np.int16,
            "day": np.int16,
            'WTEMP(C)': np.float32,
            'ALGAE_INppm': np.float32,
            "ALGAE_Oppm": np.float32,
            "H20VOLUMEm3": np.float32,
            "NH4_OUTppm": np.float32,
            "NO2_OUTppm": np.float32,
            "NO3_OUTppm": np.float32,
            "ORGN_OUTppm": np.float32,
            "ORGP_OUTppm": np.float32,
            "SOLP_OUTppm": np.float32,
            "CBOD_OUTppm": np.float32,
        })

        pred = pred.loc[pred['rch_id' ].isin(rch_id), :]

        date_range = pd.date_range(self.sim_start(), end=self.sim_end(), freq="D")

        if len(pred)==len(date_range):
            pred.index = date_range
        else:
            # when getting values for multiple rch_ids,
            assert len(rch_id)>1, f"rch_id is {rch_id} pred: {pred.shape}"

        return pred

    def read_res(self, res_id, year, skip_rows=8):
        """reads reservoir output file (output.rsv) and returns data
        for a particular reservoir

        parameters
        -----------
        res_id :
        year :
        skip_rows :
        """
        fname = os.path.join(self.path, "output.rsv")

        with open(fname, 'r') as f:
            lines = f.readlines()

        lines = [line.split() for line in lines[skip_rows:]]

        res_df = pd.DataFrame(lines[1:],
                              columns=RES_COLUMNS)
        res_df = res_df.iloc[:, 1:]
        res_df = res_df.astype(np.float32)
        res_df['RES'] = res_df['RES'].astype(np.int32)
        res_df['MON'] = res_df['MON'].astype(np.int32)
        res_df['year'] = res_df['year'].astype(np.int32)

        res = res_df.loc[res_df['RES' ]==res_id, :]

        res_yr = res[res['year' ]==year]
        date_range = pd.date_range(self.sim_start(), end=self.sim_end(), freq="D")
        res_yr.index = date_range

        return res_yr

    def read_rch(
            self,
            rch_id:Union[int, list],
            year=None,
            skip_rows=8
    )->pd.DataFrame:
        """
        reads main channel output (output.rch) file and returns data for a
        particular reach
        """
        fname = os.path.join(self.path, "output.rch")

        with open(fname, 'r') as f:
            lines = f.readlines()

        lines = [line.split() for line in lines[skip_rows:]]

        rch_df = pd.DataFrame(lines[1:])
        rch_df = rch_df.iloc[:, 1:]

        rch_df.columns = self._rch_cols

        rch_df = rch_df.astype(np.float32)
        rch_df['RCH'] = rch_df['RCH'].astype(np.int32)
        rch_df['MON'] = rch_df['MON'].astype(np.int32)

        if not isinstance(rch_id, list):
            rch_id = [rch_id]

        rch_df = rch_df.loc[rch_df['RCH' ].isin(rch_id), :]

        if len(rch_id)>1:
            assert year is None
        else:
            if self.sim_len()>365:
                year_ = int(self.sim_start_yr()) + int(self.num_skip_years())
            else:
                year_ = year
            start_date = f"{year_}{jday_to_monthday(int(self.sim_start_day()), year_)}"
            date_range = pd.date_range(start_date, periods=len(rch_df), freq="D")
            rch_df.index = date_range

            if year:
                rch_df = rch_df.loc[rch_df.index.year == year]

        return rch_df

    def sim_len(self):
        return len(pd.date_range(self.sim_start(), self.sim_end(), freq="D"))

    def get_weir_chla(self, weir_name, rch_id):
        """get chla for a single weir"""
        df = self.read_results(rch_id)
        return df[weir_name]

    def sim_start(self )->str:
        return jday_to_date(int(self.sim_start_yr()), int(self.sim_start_day()))

    def sim_end(self )->str:
        final_yr = int(self.sim_start_yr()) + int(self.sim_num_yr()) - 1
        return jday_to_date(final_yr, int(self.sim_end_day()))  #f"{final_yr}1231"

    def sim_num_yr(self):
        """number of years of simulation NBYR"""
        return self._read_cio_para('NBYR')

    def sim_end_day(self )->str:
        """reads value of IDAL from .cio file"""
        return self._read_cio_para('IDAL').rjust(3, "0")

    def sim_start_day(self )->str:
        """reads value of IDAL from .cio file"""
        return self._read_cio_para('IDAF').rjust(3, "0")

    def sim_start_yr(self):
        """reads value of IYR from .cio file"""
        return self._read_cio_para("IYR")

    def num_skip_years(self)->str:
        """reads NYSKIP from .cio file"""
        return self._read_cio_para("NYSKIP")

    def sol_rad_file(self)->str:
        """returns path of file which contains solar radiation"""
        fname = self._read_cio_para('SLRFILE')
        fpath = os.path.join(self.path, fname)
        assert os.path.exists(fpath), f"{fname} not found at {self.path}"
        return fpath

    def rel_hum_file(self)->str:
        """returns path of file which contains relative humidity data"""
        fname = self._read_cio_para('RHFILE')
        fpath = os.path.join(self.path, fname)
        assert os.path.exists(fpath), f"{fname} not found at {self.path}"
        return fpath

    def wind_speed_file(self)->str:
        """returns path of file which contains wind speed"""
        fname = self._read_cio_para('WNDFILE')
        fpath = os.path.join(self.path, fname)
        assert os.path.exists(fpath), f"{fname} not found at {self.path}"
        return fpath

    def temp_files(self)->List[str]:
        """
        returns names of temperature files i.e. those that have .tmp extension
        """
        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        files = []
        for line in lines:
            if ".tmp" in line:
                files.append(line.replace('\n', ''))

        assert len(files) == self.num_pcp_files()
        return files

    def _temp_fpath(self):
        fname = self.temp_files()
        assert len(fname)==1
        fname = fname[0]
        return os.path.join(self.path, fname)

    def num_pcp_files(self)->int:
        return int(self._read_cio_para("NRGAGE"))

    def pcp_files(self)->List[str]:
        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        files = []
        for line in lines:
            if ".pcp" in line:
                files.append(line.replace('\n', ''))

        assert len(files) == self.num_pcp_files()
        return files

    def _pcp_fpath(self):
        fname = self.pcp_files()
        assert len(fname)==1
        fname = fname[0]
        return os.path.join(self.path, fname)

    def pcp_stations(self):
        """names of stations in .pcp file"""
        with open(self._pcp_fpath(), 'r') as f:
            lines = f.readlines()
        return lines[0]

    def pcp_lat(self):
        """latitute of stations in .pcp file"""
        with open(self._pcp_fpath(), 'r') as f:
            lines = f.readlines()
        return lines[1]

    def pcp_long(self):
        """longitude of stations in .pcp file"""
        with open(self._pcp_fpath(), 'r') as f:
            lines = f.readlines()
        return lines[2]

    def pcp_elev(self):
        """elevation of stations in .pcp file"""
        with open(self._pcp_fpath(), 'r') as f:
            lines = f.readlines()
        return lines[3]

    @staticmethod
    def read_ts_file(fpath:str, width:int, nrows_skip:int)->pd.DataFrame:
        date = []
        data = []
        with open(fpath, 'r') as f:
            for idx, line in enumerate(f):
                # >= because python indexing starts from 0
                if idx>=nrows_skip:
                    data_ = line[7:]
                    data.append([data_[i:i+width] for i in range(0, len(data_), width)])
                    date.append(jday_to_date(int(line[0:4]), int(line[4:7])))

        data = pd.DataFrame(data)
        data = data.iloc[:, 0:-1].astype(np.float32)
        data.index = pd.to_datetime(date)
        return data

    def read_temp(self)->pd.DataFrame:
        """reads a temperature .tmp file """
        temp = self.read_ts_file(self._temp_fpath(), 5, 4)

        assert temp.shape[1] == int(self._read_cio_para('NTTOT'))*2, f"{temp.shape} {self._read_cio_para('NTTOT')}"

        return temp

    def read_pcp(self)->pd.DataFrame:
        """reads a precipitation file .pcp"""
        pcp = self.read_ts_file(self._pcp_fpath(), 5, 4)

        assert pcp.shape[1] == int(self._read_cio_para('NRGFIL'))

        return pcp

    def read_sol_rad(self)->pd.DataFrame:
        """reads a solar radiation data from .slr file"""
        slr = self.read_ts_file(self.sol_rad_file(), 8, 1)

        assert slr.shape[1] == int(self._read_cio_para('NSTOT'))

        return slr

    def read_rel_hum(self)->pd.DataFrame:
        """reads a relative humidity data from .slr file"""
        hmd = self.read_ts_file(self.rel_hum_file(), 8, 1)

        assert hmd.shape[1] == int(self._read_cio_para('NHTOT'))

        return hmd

    def read_wind_speed(self)->pd.DataFrame:
        """reads wind speed data from .slr file"""
        wnd = self.read_ts_file(self.wind_speed_file(), 8, 1)

        assert wnd.shape[1] == int(self._read_cio_para('NWTOT'))

        return wnd

    def _read_cio_para(self, para_name):
        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if para_name in line:
                return line.split()[0]
        return

    def wq_rches(self, rch_ids, wq_name:str = "ALGAE_INppm")->pd.DataFrame:
        """returns water quality of one or multiple rch_ids"""
        rch_wq_df = self.get_wq_rch(wq_name, rch_ids)
        groups = rch_wq_df.groupby('rch_id')
        rch_wq_df = pd.DataFrame(np.column_stack([grp[wq_name] for grp_name, grp in groups]))
        rch_wq_df.columns = list(groups.groups.keys())

        del groups

        date_range = pd.date_range(self.sim_start(), end=self.sim_end(), freq="D")
        rch_wq_df.index = date_range
        return rch_wq_df

    def wq_all_weirs(self, wq_name:str = "ALGAE_INppm")->pd.DataFrame:
        """
        reads water quality of all weirs
        """
        # todo, are we reading weirs or rches
        rch_wq_df = self.wq_rches(self.weir_codes, wq_name)
        #cols = {v:k for k,v in self.weir_codes.items()}
        #rch_wq_df.rename(cols, axis='columns', inplace=True)

        return rch_wq_df

    def plot_wq_all_weirs(
            self,
            wq_name:str="ALGAE_INppm",
            save:bool = True,
            show:bool = True,
            **kwargs
    )->pd.DataFrame:
        """plot chla for all weirs
        plot_chla_all_weirs()
        """
        rch_wq_df = self.wq_all_weirs(wq_name=wq_name)

        out = rch_wq_df.plot(subplots=True, figsize=(8, 12), **kwargs)
        if save:
            plt.savefig(f"{self.path}_rch_chla_obs", bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        return out

    def __call__(self, executable:str="swat2012.exe")->None:
        """run the swat model"""

        old_wd = os.getcwd()

        os.chdir(self.path)
        os.system(executable)

        os.chdir(old_wd)

        return

    def write_outflow(
            self,
            rch_id:Union[int, str],
            outflow:np.ndarray
    ):
        """writes the daily outflow from a rch/weir/reservoir
        parameters
        ----------
        rch_id : int/str
        outflow :
        """
        day = str(rch_id).rjust(3, '0')
        fname = f'{self.path}\\00{day}0000.day'

        header = f"Daily Reservoir Outflow file: .day file Subbasin:{rch_id}  created {dateandtime_now()} \n"

        with open(fname, 'w') as f:
            f.write(header)
            for val in outflow.reshape(-1,).tolist():
                f.write(f"{val:10.3f}\n")
        return

    def change_start_day(self, day:int):
        """change simulation start day i.e. IDAF variabel in file.cio
        parameters
        ----------
        day : int
            julian day to start the simulation
        """

        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for l in lines:
            if "IDAF" in l:
                l = l[0:13] + str(day).ljust(5) + l[18:]
            new_lines.append(l)

        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'w') as f:
           f.writelines(new_lines)

        return

    def change_end_day(self, day:int):
        """changes the simulation end day (IDAL) in file.cio"""
        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for l in lines:
            if "IDAL" in l:
                l = l[0:13] + str(day).ljust(5) + l[18:]
            new_lines.append(l)

        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'w') as f:
            f.writelines(new_lines)
        return

    def change_num_sim_years(self, years:int):
        """changes the number of years to simulate (NBYR) in file.cio"""
        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for l in lines:
            if "NBYR" in l:
                l = l[0:14] + str(years).rjust(2) + l[16:]
            new_lines.append(l)

        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'w') as f:
            f.writelines(new_lines)
        return

    def change_sim_start_yr(self, year):
        """changes simulation start year i.e. IYR variable in file.cio"""

        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for l in lines:
            if "IYR" in l:
                l = l[0:12] + str(year).rjust(4) + l[16:]
            new_lines.append(l)

        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'w') as f:
            f.writelines(new_lines)
        return

    def get_weirs_outflow(self, start_date=None)->pd.DataFrame:
        weir_outflow = {}
        for idx, code in enumerate(self.weir_codes):

            _day = str(code).rjust(3, '0')
            fname = f'{self.path}\\00{_day}0000.day'

            day_weir = pd.read_csv(fname, skiprows=0, encoding= 'unicode_escape')
            index = pd.date_range(self.sim_start(), periods=len(day_weir), freq='D')
            day_weir.index = index
            weir_outflow[code] = day_weir

        df = pd.concat(weir_outflow.values(), axis=1)
        df.columns = list(weir_outflow.keys())
        if start_date:
            df = df.loc[start_date:, :]
        return df

    def get_weir_outflow(self, weir_id:int, start_date="20000101"):
        """get outflow for a single weir"""

        day = str(weir_id).rjust(3, '0')
        fname = f'{self.path}\\00{day}0000.day'
        weir_outflow = pd.read_csv(fname, skiprows=0, encoding='unicode_escape')
        index = pd.date_range(start_date, periods=len(weir_outflow), freq='D')
        weir_outflow.index = index

        return weir_outflow

    def precip_for_sub(self,
                       basin_id:Union[int, str]
                       )->pd.Series:
        """returns precipitation data for a sub-basin"""
        gage_id = self.precip_gage_for_sub(basin_id)
        pcp = self.read_pcp()
        # +1 because python uses 0 based indexing and gage_ids start from 1
        return pcp.iloc[:, gage_id+1]

    def temp_for_sub(
            self,
            basin_id:Union[int, str]
    )->pd.DataFrame:
        """returns maximum and minimum temperature data for a sub-basin"""
        gage_id = self.temp_gage_for_sub(basin_id)
        temp = self.read_temp()
        # there are two columns for each gage one for min and one for max
        # both columns for a gage are located side by side
        st, en = gage_id*2 - 2, gage_id*2 - 1
        return temp.iloc[:, st: en+1]

    def wind_for_sub(
            self,
            basin_id:Union[int, str]
    )->pd.DataFrame:
        """returns wind speed data for a sub-basin"""
        gage_id = self.wind_gage_for_sub(basin_id)
        wind = self.read_wind_speed()
        # +1 because python uses 0 based indexing and gage_ids start from 1
        return wind.iloc[:, gage_id+1]

    def rel_hum_for_sub(
            self,
            basin_id:Union[int, str]
    )->pd.DataFrame:
        """returns relative humidity data for a sub-basin"""
        gage_id = self.rel_hum_gage_for_sub(basin_id)
        rel_hum = self.read_rel_hum()
        # +1 because python uses 0 based indexing and gage_ids start from 1
        return rel_hum.iloc[:, gage_id+1]

    def sol_rad_for_sub(
            self,
            basin_id:Union[int, str]
    )->pd.DataFrame:
        """returns relative humidity data for a sub-basin"""
        gage_id = self.sol_rad_gage_for_sub(basin_id)
        rel_hum = self.read_sol_rad()
        # +1 because python uses 0 based indexing and gage_ids start from 1
        return rel_hum.iloc[:, gage_id+1]

    def precip_gage_for_sub(self, basin_id)->int:
        """returns the gage measuring/containing precipitation data for a sub-basin"""
        return int(self.gage_for_sub('IRGAGE', basin_id))

    def temp_gage_for_sub(self, basin_id)->int:
        """returns the gage measuring/containing temperature data for a sub-basin"""
        return int(self.gage_for_sub('ITGAGE', basin_id))

    def sol_rad_gage_for_sub(self, basin_id)->int:
        """returns the gage measuring/containing solar radiation data for a sub-basin"""
        return int(self.gage_for_sub('ISGAGE', basin_id))

    def rel_hum_gage_for_sub(self, basin_id)->int:
        """returns the gage measuring/containing humidity data for a sub-basin"""
        return int(self.gage_for_sub('IHGAGE', basin_id))

    def wind_gage_for_sub(self, basin_id)->int:
        """returns the gage measuring/containing wind speed data for a sub-basin"""
        return int(self.gage_for_sub('IWGAGE', basin_id))

    def gage_for_sub(self, para_name:str, basin_id:Union[int, str])->str:
        basin_id = str(basin_id).rjust(3, '0')
        fpath = os.path.join(self.path, f"00{basin_id}0000.sub")

        with open(fpath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if para_name in line:
                return line.split()[0]
        raise ValueError

    def subbasins(self)->List[int]:
        """returns the subbasin number for all the subbasins being modeled.
        It reads the .fig file and returns SUB_NUM values for subbasin command
        """
        sub_nums = []
        with open(os.path.join(self.path, 'fig.fig'), 'r') as fp:
            for idx, line in enumerate(fp):
                if line.startswith('subbasin'):
                    _, hyd_stor, sub_command, sub_num, *sub_gis = line.split()
                    sub_nums.append(int(sub_num))
        return sub_nums

    def reaches(self)->List[int]:
        """returns the reaches being modeled in the sequence as they are given
        in .fig file

        Returns
        -------
        list
        """
        rch_nums = []
        with open(os.path.join(self.path, 'fig.fig'), 'r') as fp:
            for idx, line in enumerate(fp):
                if line.startswith('route'):
                    _, rch_command, hyd_stor, rch_num, *_ = line.split()
                    rch_nums.append(int(rch_num))

        return rch_nums

    def reservoirs(self)->List[int]:
        """returns reservoir ids (RES_NUM) being modeled."""
        res_nums = []
        with open(os.path.join(self.path, 'fig.fig'), 'r') as fp:
            for idx, line in enumerate(fp):
                if line.startswith('routres'):
                    _, res_command, hyd_stor, res_num, *_ = line.split()
                    res_nums.append(int(res_num))
        return res_nums

    def downstream_rch_to_rch(self, upstream_rch_num:int)->int:
        """returns the downstream reach to a reach"""
        reaches = self.reaches()
        if upstream_rch_num == reaches[-1]:
            raise ValueError(f"{upstream_rch_num} is the last reach. No downstream reach found")

        val = [reaches[idx + 1] for idx, val in enumerate(reaches) if val == upstream_rch_num]
        assert len(val)==1
        return val[0]

    def upstream_rch_to_rch(self, downstream_rch_num:int)->int:
        """returns the downstream reach to a reach"""
        reaches = self.reaches()
        if downstream_rch_num == reaches[0]:
            raise ValueError(f"{downstream_rch_num} is the first reach. No upstream reach found")

        val = [reaches[idx - 1] for idx, val in enumerate(reaches) if val == downstream_rch_num]
        assert len(val) == 1
        return val[0]

    def customize_rch_output(
            self,
            parameters:Union[str, int, List[str], List[int]]):
        """
        define IPDVAR i.e., parameters which are to printed/written in
        output.rch file

        Examples
        ---------
        >>> swat = SWAT('/path/to/swat/files')
        >>> swat.customize_rch_output(0)
        will write following
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        >>> swat.customize_rch_output(1)
        will write following
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        >>> swat.customize_rch_output([1, 2, 3])
        will write following
        1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        >>> swat.customize_rch_output("FLOW_IN")
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        >>> swat.customize_rch_output(["FLOW_IN", "FLOW_OUT", "EVAP"])
        will write following
        1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        """
        self._custom_out_files(parameters, RCH_OUT_CODES, 64)
        # the column names must be redefined
        self._rch_cols = self._output_rch_cols()
        return

    def customize_sub_output(
            self,
            parameters:Union[str, int, List[str], List[int]]):
        """
        define IPDVAB i.e., parameters which are to printed/written in
        output.sub file

        Examples
        --------
        >>> swat = SWAT('/path/to/swat/files')
        >>> swat.customize_sub_output(0)
        will write following
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        >>> swat.customize_sub_output(1)
        will write following
        1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        >>> swat.customize_sub_output([1, 2, 3])
        will write following
        1   2   3   0   0   0   0   0   0   0   0   0   0   0   0
        >>> swat.customize_sub_output("PRECIP")
        will write following
        1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        >>> swat.customize_sub_output(["PRECIP", "SNOMELT", "PET"])
        will write following
        1   2   3   0   0   0   0   0   0   0   0   0   0   0   0
        """
        self._custom_out_files(parameters, SUB_OUT_CODES, 66, max_paras=15)
        return

    def customize_hru_output(
            self,
            parameters
    ):
        """
        define IPDVAS i.e. parameters which are to printed/written in output.sub
        file

        Examples
        --------
        >>> swat = SWAT('/path/to/swat/files')
        >>> swat.customize_hru_output(0)
        will write following
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        >>> swat.customize_hru_output(1)
        will write following
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        >>> swat.customize_hru_output([1, 2, 3])
        will write following
        1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        >>> swat.customize_hru_output("PRECIP")
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        >>> swat.customize_hru_output(["PRECIP", "SNOFALL", "SNOMELT"])
        will write following
        1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        """
        self._custom_out_files(parameters, HRU_OUT_CODES, 68)
        return

    def _custom_out_files(self, parameters, CODES,
                          line_num:int,
                          max_paras:int=20):

        parameters = verify_paras(parameters, CODES, max_paras)

        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'r') as fp:
            lines = fp.readlines()

        lines[line_num] = '   ' + '   '.join(str(val) for val in parameters) + '\n'

        with open(fname, 'w') as f:
           f.writelines(lines)
        return

    def _output_rch_cols(self):
        """returns column names for output.rch file"""
        fname = os.path.join(self.path, "file.cio")
        with open(fname, 'r') as fp:
            lines = fp.readlines()

        idpvar = [int(val) for val in lines[64].split()]
        if sum(idpvar)==0:
            cols = DEF_RCH_COLUMNS
        else:
            idpvar = [RCH_COL_MAP[col] for col in idpvar]
            cols = ['RCH', 'GIS', 'MON', 'AREAkm2'] + idpvar

        return cols

    def _output_wql_cols(self):
        """returns column names for output.wql file"""
        return


def verify_paras(
        parameters:Union[int, str, List[str], List[int]],
        CODE:dict,
        max_paras:int = 20
)->list:

    if isinstance(parameters, list):

        assert len(parameters)<=max_paras

        if isinstance(parameters[0], str):
            assert all([isinstance(para, str) for para in parameters])
            assert all([para in CODE.values() for para in parameters])
            _CODE = {v:k for k,v in CODE.items()}
            parameters = [_CODE[para] for para in parameters]

        elif isinstance(parameters[0], int):
            assert all([isinstance(para, int) for para in parameters])
            assert all([para in CODE.keys() for para in parameters])

    elif isinstance(parameters, int):
        parameters = [parameters]

    elif isinstance(parameters, str):
        assert parameters in CODE.values()
        _CODE = {v: k for k, v in CODE.items()}
        parameters = [_CODE[parameters]]

    zeros = [0 for _ in range(max_paras - len(parameters))]
    parameters += zeros

    return parameters


def jday_to_monthday(jday:int, year=2000):

    date = datetime.datetime(year, 1, 1) + datetime.timedelta(jday - 1)

    return f"{str(date.month).rjust(2, '0')}{str(date.day).rjust(2, '0')}"


def jday_to_date(year:int, jday:int)->str:

    date = datetime.datetime(year, 1, 1) + datetime.timedelta(jday - 1)

    return f"{date.year}{str(date.month).rjust(2, '0')}{str(date.day).rjust(2, '0')}"


if __name__ == "__main__":

    pass

