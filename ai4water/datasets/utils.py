
import os
import sys
import ssl
import glob
import shutil
import zipfile
import tempfile
from typing import Union, List
import urllib.request as ulib
import urllib.parse as urlparse

from ai4water.backend import pd


try:
    import requests
except ModuleNotFoundError:
    requests = None


# following files must exist withing data folder for CAMELS-GB data
DATA_FILES = {
    'CAMELS-GB': [
        'CAMELS_GB_climatic_attributes.csv',
        'CAMELS_GB_humaninfluence_attributes.csv',
        'CAMELS_GB_hydrogeology_attributes.csv',
        'CAMELS_GB_hydrologic_attributes.csv',
        'CAMELS_GB_hydrometry_attributes.csv',
        'CAMELS_GB_landcover_attributes.csv',
        'CAMELS_GB_soil_attributes.csv',
        'CAMELS_GB_topographic_attributes.csv'
    ],
    'HYSETS': [  # following files must exist in a folder containing HYSETS dataset.
        'HYSETS_2020_ERA5.nc',
        'HYSETS_2020_ERA5Land.nc',
        'HYSETS_2020_ERA5Land_SWE.nc',
        'HYSETS_2020_Livneh.nc',
        'HYSETS_2020_nonQC_stations.nc',
        'HYSETS_2020_SCDNA.nc',
        'HYSETS_2020_SNODAS_SWE.nc',
        'HYSETS_elevation_bands_100m.csv',
        'HYSETS_watershed_boundaries.zip',
        'HYSETS_watershed_properties.txt'
    ]
}


def download_all_http_directory(url, outpath=None, filetypes=".zip", match_name=None):
    """
    Download all the files which are of category filetypes at the location of
    outpath. If a file is already present. It will not be downloaded.
    filetypes str: extension of files to be downloaded. By default only .zip files
        are downloaded.
    mathc_name str: if not None, then only those files will be downloaded whose name
        have match_name string in them.
    """
    try:
        import bs4
    except (ModuleNotFoundError, ImportError) as e:
        raise e(f"You must install bs4 library e.g. by using"
                f"pip install bs4")

    if os.name == 'nt':
        ssl._create_default_https_context = ssl._create_unverified_context
    page = list(urlparse.urlsplit(url))[2].split('/')[-1]
    basic_url = url.split(page)[0]

    r = requests.get(url)
    data = bs4.BeautifulSoup(r.text, "html.parser")
    match_name = filetypes if match_name is None else match_name

    for l in data.find_all("a"):

        if l["href"].endswith(filetypes) and match_name in l['href']:
            _outpath = outpath
            if outpath is not None:
                _outpath = os.path.join(outpath, l['href'])

            if os.path.exists(_outpath):
                print(f"file {l['href']} already exists at {outpath}")
                continue
            download(basic_url + l["href"], _outpath)
            print(r.status_code, l["href"], )
    return


def download(url, out=None):
    """High level function, which downloads URL into tmp file in current
    directory and then renames it to filename autodetected from either URL
    or HTTP headers.

    :param url:
    :param out: output filename or directory
    :return:    filename where URL is downloaded to
    """
    # detect of out is a directory
    if out is not None:
        outdir = os.path.dirname(out)
        out_filename = os.path.basename(out)
        if outdir == '':
            outdir = os.getcwd()
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    else:
        outdir = os.getcwd()
        out_filename = None

    # get filename for temp file in current directory
    prefix = filename_from_url(url)
    (fd, tmpfile) = tempfile.mkstemp(".tmp", prefix=prefix, dir=".")
    os.close(fd)
    os.unlink(tmpfile)

    # set progress monitoring callback
    def callback_charged(blocks, block_size, total_size):
        # 'closure' to set bar drawing function in callback
        callback_progress(blocks, block_size, total_size, bar_function=bar)

    callback = callback_charged

    # Python 3 can not quote URL as needed
    binurl = list(urlparse.urlsplit(url))
    binurl[2] = urlparse.quote(binurl[2])
    binurl = urlparse.urlunsplit(binurl)

    (tmpfile, headers) = ulib.urlretrieve(binurl, tmpfile, callback)
    filename = filename_from_url(url)

    if out_filename:
        filename = out_filename

    filename = outdir + "/" + filename

    # add numeric ' (x)' suffix if filename already exists
    if os.path.exists(filename):
        filename = filename + '1'
    shutil.move(tmpfile, filename)

    # print headers
    return filename


__current_size = 0


def callback_progress(blocks, block_size, total_size, bar_function):
    """callback function for urlretrieve that is called when connection is
    created and when once for each block

    draws adaptive progress bar in terminal/console

    use sys.stdout.write() instead of "print,", because it allows one more
    symbol at the line end without linefeed on Windows

    :param blocks: number of blocks transferred so far
    :param block_size: in bytes
    :param total_size: in bytes, can be -1 if server doesn't return it
    :param bar_function: another callback function to visualize progress
    """
    global __current_size

    width = 100

    if sys.version_info[:3] == (3, 3, 0):  # regression workaround
        if blocks == 0:  # first call
            __current_size = 0
        else:
            __current_size += block_size
        current_size = __current_size
    else:
        current_size = min(blocks * block_size, total_size)
    progress = bar_function(current_size, total_size, width)
    if progress:
        sys.stdout.write("\r" + progress)


def filename_from_url(url):
    """:return: detected filename as unicode or None"""
    # [ ] test urlparse behavior with unicode url
    fname = os.path.basename(urlparse.urlparse(url).path)
    if len(fname.strip(" \n\t.")) == 0:
        return None
    return fname


def bar(current_size, total_size, width):
    percent = current_size/total_size * 100
    if round(percent % 1, 4) == 0.0:
        print(f"{round(percent)}% of {round(total_size*1e-6, 2)} MB downloaded")
    return


def check_attributes(attributes, check_against: list) -> list:
    if attributes == 'all' or attributes is None:
        attributes = check_against
    elif not isinstance(attributes, list):
        assert isinstance(attributes, str)
        assert attributes in check_against
        attributes = [attributes]
    else:
        assert isinstance(attributes, list), f'unknown attributes {attributes}'

    assert all(elem in check_against for elem in attributes)

    return attributes


def sanity_check(dataset_name, path, url=None):
    if dataset_name in DATA_FILES:
        if dataset_name == 'CAMELS-GB':
            if not os.path.exists(os.path.join(path, 'data')):
                raise FileNotFoundError(f"No folder named `data` exists inside {path}")
            else:
                data_path = os.path.join(path, 'data')
                for file in DATA_FILES[dataset_name]:
                    if not os.path.exists(os.path.join(data_path, file)):
                        raise FileNotFoundError(f"File {file} must exist inside {data_path}")
    _maybe_not_all_files_downloaded(path, url)
    return


def _maybe_not_all_files_downloaded(
        path:str,
        url:Union[str, list, dict]
):
    if isinstance(url, dict):
        available_files = os.listdir(path)

        for fname, link in url.items():
            if fname not in available_files:
                print(f"file {fname} is not available so downloading it now.")
                download_and_unzip(path, {fname:link})

    return


def check_st_en(
        df:pd.DataFrame,
        st:Union[int, str, pd.DatetimeIndex]=None,
        en:Union[int, str, pd.DatetimeIndex]=None
)->pd.DataFrame:
    """slices the dataframe based upon st and en"""
    if isinstance(st, int):
        if en is None:
            en = len(df)
        else:
            assert isinstance(en, int)
        df = df.iloc[st:en]

    elif isinstance(st, (str, pd.DatetimeIndex)):
        if en is None:
            en = df.index[-1]
        df = df.loc[st:en]

    elif isinstance(en, int):
        st = 0 # st must be none here
        df = df.iloc[st:en]
    elif isinstance(en, (str, pd.DatetimeIndex)):
        st = df.index[0]
        df = df.loc[st:en]

    return df


def unzip_all_in_dir(dir_name, ext=".gz"):
    gz_files = glob.glob(f"{dir_name}/*{ext}")
    for f in gz_files:
        shutil.unpack_archive(f, dir_name)
    return


def maybe_download(ds_dir,
                   url:Union[str, List[str], dict],
                   overwrite:bool=False,
                   name=None,
                   include:list=None,
                   **kwargs):

    if os.path.exists(ds_dir) and len(os.listdir(ds_dir)) > 0:
        if overwrite:
            print(f"removing previous data directory {ds_dir} and downloading new")
            shutil.rmtree(ds_dir)
            download_and_unzip(ds_dir, url=url, include=include, **kwargs)
        else:
            print(f"""
    Not downloading the data since the directory 
    {ds_dir} already exists.
    Use overwrite=True to remove previously saved files and download again""")
            sanity_check(name, ds_dir, url)
    else:
        download_and_unzip(ds_dir, url=url, include=include, **kwargs)
    return


def download_and_unzip(path,
                       url:Union[str, List[str], dict],
                       include=None,
                       **kwargs):
    """

    url :

    include :
        files to download. Files which are not in include will not be
        downloaded.
    kwargs :
        any keyword arguments for download_from_zenodo function
    """
    from .download_zenodo import download_from_zenodo

    if not os.path.exists(path):
        os.makedirs(path)
    if isinstance(url, str):
        if 'zenodo' in url:
            download_from_zenodo(path, doi=url, include=include, **kwargs)
        else:
            download(url, path)
        _unzip(path)
    elif isinstance(url, list):
        for url in url:
            if 'zenodo' in url:
                download_from_zenodo(path, url, include=include, **kwargs)
            else:
                download(url, path)
        _unzip(path)
    elif isinstance(url, dict):
        for fname, url in url.items():
            if 'zenodo' in url:
                download_from_zenodo(path, doi=url, include=include, **kwargs)
            else:
                download(url, os.path.join(path, fname))
        _unzip(path)

    else:
        raise ValueError(f"Invalid url: {path}, {url}")

    return


def _unzip(ds_dir, dirname=None):
    """unzip all the zipped files in a directory"""
    if dirname is None:
        dirname = ds_dir

    all_files = glob.glob(f"{dirname}/*.zip")
    for f in all_files:
        src = os.path.join(dirname, f)
        trgt = os.path.join(dirname, f.split('.zip')[0])
        if not os.path.exists(trgt):
            print(f"unzipping {src} to {trgt}")
            with zipfile.ZipFile(os.path.join(dirname, f), 'r') as zip_ref:
                try:
                    zip_ref.extractall(os.path.join(dirname, f.split('.zip')[0]))
                except OSError:
                    filelist = zip_ref.filelist
                    for _file in filelist:
                        if '.txt' in _file.filename or '.csv' in _file.filename or '.xlsx' in _file.filename:
                            zip_ref.extract(_file)

    # extracting tar.gz files todo, check if zip files can also be unpacked by the following oneliner
    gz_files = glob.glob(f"{ds_dir}/*.gz")
    for f in gz_files:
        shutil.unpack_archive(f, ds_dir)

    return


class OneHotEncoder(object):
    """
    >>> data, _, _ = mg_photodegradation()
    >>> cat_enc1 = OneHotEncoder()
    >>> cat_ = cat_enc1.fit_transform(data['Catalyst_type'].values)
    >>> _cat = cat_enc1.inverse_transform(cat_)
    >>> all([a==b for a,b in zip(data['Catalyst_type'].values, _cat)])
    """
    def fit(self, X:np.ndarray):
        assert len(X) == X.size
        categories, inverse = np.unique(X, return_inverse=True)
        X = np.eye(categories.shape[0])[inverse]
        self.categories_ = [categories]
        return X

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return self.transform(self.fit(X))

    def inverse_transform(self, X):
        return pd.DataFrame(X, columns=self.categories_[0]).idxmax(1).values


class LabelEncoder(object):
    """
    >>> data, _, _ = mg_photodegradation()
    >>> cat_enc1 = LabelEncoder()
    >>> cat_ = cat_enc1.fit_transform(data['Catalyst_type'].values)
    >>> _cat = cat_enc1.inverse_transform(cat_)
    >>> all([a==b for a,b in zip(data['Catalyst_type'].values, _cat)])
    """
    def fit(self, X):
        assert len(X) == X.size
        categories, inverse = np.unique(X, return_inverse=True)
        self.categories_ = [categories]
        labels = np.unique(inverse)
        self.mapper_ = {category:label for category,label in zip(categories, labels)}
        return inverse

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return self.transform(self.fit(X))

    def inverse_transeform(self, X):
        return pd.Series(X).map(self.mapper_)


def ohe_column(df:pd.DataFrame, col_name:str)->tuple:
    """one hot encode a column in datatrame"""
    assert isinstance(col_name, str)
    assert isinstance(df, pd.DataFrame)

    encoder = OneHotEncoder()
    ohe_cat = encoder.fit_transform(df[col_name].values.reshape(-1, 1))
    cols_added = [f"{col_name}_{i}" for i in range(ohe_cat.shape[-1])]

    df[cols_added] = ohe_cat

    df.pop(col_name)

    return df, cols_added, encoder


def le_column(df:pd.DataFrame, col_name)->tuple:
    """label encode a column in dataframe"""
    encoder = LabelEncoder()
    df[col_name] = encoder.fit_transform(df[col_name])
    return df, encoder

