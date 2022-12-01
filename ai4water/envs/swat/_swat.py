
import os
import datetime

from ai4water.backend import np, pd, plt
from ai4water.utils.utils import dateandtime_now


class SWAT(object):
    """interface to swat model"""
    def __init__(self, dir_name, weir_codes:dict):

        self.dir_name = dir_name
        self.weir_codes = weir_codes

    def get_wq_rch(self, wq_name, rch_id):
        """get wq data for a single weir"""

        df = self.read_wql_output(rch_id)

        return df[wq_name]

    def read_wql_output(self, rch_id):

        # read SWAT results
        fname = os.path.join(self.dir_name, "output.wql")

        with open(fname, 'r') as f:
            lines = f.readlines()

        lines = [line.split() for line in lines]

        columns = [
            'junk', 'rch_id', 'day', 'WTEMP(C)',  'ALGAE_INppm', "ALGAE_Oppm",
            "ORGN_INppm", "ORGN_OUTppm", "NH4_INppm",  "NH4_OUTppm",   "NO2_INppm",
            "NO2_OUTppm", "NO3_INppm", "NO3_OUTppm",  "ORGP_INppm", "ORGP_OUTppm",
            "SOLP_INppm", "SOLP_OUTppm",  "CBOD_INppm", "CBOD_OUTppm", "SAT_OXppm",
            "DISOX_INppm",  "DISOX_Oppm", "H20VOLUMEm3", "TRVL_TIMEhr",
        ]

        pred = pd.DataFrame(lines[1:],
                            columns=columns)

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

        pred = pred.loc[pred['rch_id' ] == rch_id, :]

        date_range = pd.date_range(self.sim_start(), end=self.sim_end(), freq="D")

        pred.index = date_range

        return pred

    def read_res(self, res_id, year, skip_rows=8):

        fname = os.path.join(self.dir_name, "output.rsv")

        with open(fname, 'r') as f:
            lines = f.readlines()

        lines = [line.split() for line in lines[skip_rows:]]

        columns = [
            'junk', 'RES', 'MON', 'VOLUMEm3', 'FLOW_INcms', 'FLOW_OUTcms', 'PRECIPm3',
            'EVAPm3', 'SEEPAGEm3', 'SED_INtons', 'SED_OUTtons', 'SED_CONCppm', 'ORGN_INkg',
            'ORGN_OUTkg', 'RES_ORGNppm', 'ORGP_INkg', 'ORGP_OUTkg', 'RES_ORGPppm', 'NO3_INkg',
            'NO3_OUTkg', 'RES_NO3ppm', 'NO2_INkg', 'NO2_OUTkg', 'RES_NO2ppm', 'NH3_INkg',
            'NH3_OUTkg', 'RES_NH3ppm', 'MINP_INkg', 'MINP_OUTkg', 'RES_MINPppm', 'CHLA_INkg',
            'CHLA_OUTkg', 'SECCHIDEPTHm',
            'PEST_INmg',
            'REACTPSTmg',
            'VOLPSTmg',
            'SETTLPSTmg', 'RESUSP_PSTmg', 'DIFFUSEPSTmg', 'REACBEDPSTmg',
            'BURYPSTmg',
            'PEST_OUTmg', 'PSTCNCWmg/m3', 'PSTCNCBmg/m3', 'year']

        res_df = pd.DataFrame(lines[1:],
                              columns=columns)
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

    def read_rch(self, rch_id, year, skip_rows=8):

        start_date = f"{year}{jday_to_monthday(int(self.sim_start_day()), year)}"

        fname = os.path.join(self.dir_name, "output.rch")

        with open(fname, 'r') as f:
            lines = f.readlines()

        lines = [line.split() for line in lines[skip_rows:]]

        columns = [
            'RCH', 'GIS',
            'MON', 'AREAkm2', 'FLOW_INcms', 'FLOW_OUTcms', 'EVAPcms', 'TLOSScms',
            'SED_INtons', 'SED_OUTtons', 'SEDCONCmg/kg', 'ORGN_INkg',
            'ORGN_OUTkg', 'ORGP_INkg', 'ORGP_OUTkg', 'NO3_INkg', 'NO3_OUTkg',
            'NH4_INkg', 'NH4_OUTkg', 'NO2_INkg', 'NO2_OUTkg', 'MINP_INkg', 'MINP_OUTkg',
            'CHLA_INkg', 'CHLA_OUTkg', 'CBOD_INkg', 'CBOD_OUTkg', 'DISOX_INkg', 'DISOX_OUTkg',
            'SOLPST_INmg', 'SOLPST_OUTmg', 'SORPST_INmg', 'SORPST_OUTmg',
            'REACTPSTmg', 'VOLPSTmg', 'SETTLPSTmg', 'RESUSP_PSTmg', 'DIFFUSEPSTmg', 'REACBEDPSTmg',
            'BURYPSTmg', 'BED_PSTmg', 'BACTP_OUTct', 'BACTLP_OUTct', 'CMETAL#1kg', 'CMETAL#2kg',
            'CMETAL#3kg', 'TOT_Nkg', 'TOT_Pkg', 'NO3ConcMg/l', 'WTMPdegc'
        ]

        rch_df = pd.DataFrame(lines[1:])
        rch_df = rch_df.iloc[:, 1:]

        rch_df.columns = columns

        rch_df = rch_df.astype(np.float32)
        rch_df['RCH'] = rch_df['RCH'].astype(np.int32)
        rch_df['MON'] = rch_df['MON'].astype(np.int32)

        rch = rch_df.loc[rch_df['RCH' ]==rch_id, :]

        date_range = pd.date_range(start_date, periods=len(rch), freq="D")

        rch.index = date_range
        return rch

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

    def num_skip_years(self):
        """reads NYSKIP from .cio file"""
        return self._read_cio_para("NYSKIP")

    def _read_cio_para(self, para_name):
        fname = os.path.join(self.dir_name, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if para_name in line:
                return line.split()[0]
        return

    def plot_wq_all_weirs(self, wq_name:str="ALGAE_INppm"):
        """plot chla for all weirs
        plot_chla_all_weirs()
        """
        rch_wq = {}
        for k ,v in self.weir_codes.items():

            rch_wq[k] = self.get_wq_rch(wq_name, v)[wq_name]

        rch_wq_df = pd.concat(rch_wq.values(), axis=1)
        rch_wq_df.columns = list(rch_wq.keys())

        rch_wq_df.plot(subplots=True, figsize=(8, 12))
        plt.savefig(f"{self.dir_name}_rch_chla_obs", bbox_inches="tight", dpi=300)
        return rch_wq_df

    def __call__(self, executable:str="swat2012.exe")->None:
        """run the swat model"""

        old_wd = os.getcwd()

        os.chdir(self.dir_name)
        os.system(executable)

        os.chdir(old_wd)

        return

    def write_outflow(self, rich_id, outflow):

        day = str(rich_id).rjust(3, '0')
        fname = f'{self.dir_name}\\00{day}0000.day'

        header = f"Daily Reservoir Outflow file: .day file Subbasin:{rich_id}  created {dateandtime_now()} \n"

        with open(fname, 'w') as f:
            f.write(header)
            for val in outflow.reshape(-1,).tolist():
                f.write(f"{val:10.3f}\n")
        return

    def change_start_day(self, day:int):
        fname = os.path.join(self.dir_name, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for l in lines:
            if "IDAF" in l:
                l = l[0:13] + str(day).ljust(5) + l[18:]
            new_lines.append(l)

        fname = os.path.join(self.dir_name, "file.cio")
        with open(fname, 'w') as f:
           f.writelines(new_lines)

        return

    def change_end_day(self, day:int):
        fname = os.path.join(self.dir_name, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for l in lines:
            if "IDAL" in l:
                l = l[0:13] + str(day).ljust(5) + l[18:]
            new_lines.append(l)

        fname = os.path.join(self.dir_name, "file.cio")
        with open(fname, 'w') as f:
            f.writelines(new_lines)
        return

    def change_nbyr(self, years):
        fname = os.path.join(self.dir_name, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for l in lines:
            if "NBYR" in l:
                l = l[0:14] + str(years).rjust(2) + l[16:]
            new_lines.append(l)

        fname = os.path.join(self.dir_name, "file.cio")
        with open(fname, 'w') as f:
            f.writelines(new_lines)
        return

    def change_iyr(self, year):
        fname = os.path.join(self.dir_name, "file.cio")
        with open(fname, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for l in lines:
            if "IYR" in l:
                l = l[0:12] + str(year).rjust(4) + l[16:]
            new_lines.append(l)

        fname = os.path.join(self.dir_name, "file.cio")
        with open(fname, 'w') as f:
            f.writelines(new_lines)
        return

    def get_weirs_outflow(self, start_date="20000101"):
        weir_outflow = {}
        for idx, (k, v) in enumerate(self.weir_codes.items()):

            _day = str(v).rjust(3, '0')
            fname = f'{self.dir_name}\\00{_day}0000.day'

            day_weir = pd.read_csv(fname, skiprows=0, encoding= 'unicode_escape')
            index = pd.date_range(start_date, periods=len(day_weir), freq='D')
            day_weir.index = index
            weir_outflow[k] = day_weir

        _df = pd.concat(weir_outflow.values(), axis=1)
        _df.columns = list(weir_outflow.keys())
        return _df, weir_outflow

    def get_weir_outflow(self, weir_id:int, start_date="20000101"):
        """get outflow for a single weir"""

        day = str(weir_id).rjust(3, '0')
        fname = f'{self.dir_name}\\00{day}0000.day'
        weir_outflow = pd.read_csv(fname, skiprows=0, encoding='unicode_escape')
        index = pd.date_range(start_date, periods=len(weir_outflow), freq='D')
        weir_outflow.index = index

        return weir_outflow


def jday_to_monthday(jday:int, year=2000):

    date = datetime.datetime(year, 1, 1) + datetime.timedelta(jday - 1)

    return f"{str(date.month).rjust(2, '0')}{str(date.day).rjust(2, '0')}"


def jday_to_date(year:int, jday:int)->str:

    date = datetime.datetime(year, 1, 1) + datetime.timedelta(jday - 1)

    return f"{date.year}{str(date.month).rjust(2, '0')}{str(date.day).rjust(2, '0')}"


if __name__ == "__main__":

    pass

