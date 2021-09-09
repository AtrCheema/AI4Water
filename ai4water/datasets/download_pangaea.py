# -*- coding: utf-8 -*-
"""
The code in this file has been modified after https://github.com/pangaea-data-publisher/pangaeapy/blob/master/pangaeapy/pandataset.py
"""
import json
import os
import xml.etree.ElementTree as ET
import re
import io

import requests
import pandas as pd


class PanParam:
    """ PANGAEA Parameter
    Shoud be used to create PANGAEA parameter objects. Parameter is used here to represent 'measured variables'

    Attributes
    ----------
    ID : int
        the identifier for the parameter
    name : str
        A long name or title used for the parameter
    shortName : str
        A short name or label to identify the parameter
    synonym : dict
        A diconary of synonyms for the parameter whcih e.g. is used by other archives or communities.
        The dict key indicates the namespace (possible values currently are CF and OS)
    type : str
        indicates the data type of the parameter (string, numeric, datetime etc..)
    source : str
        defines the category or source for a parameter (e.g. geocode, data, event)... very PANGAEA specific ;)
    unit : str
        the unit of measurement used with this parameter (e.g. m/s, kg etc..)
    format: str
        the number format string given by PANGAEA e.g ##.000 which defines the displayed precision of the number
    """

    def __init__(self, ID, name, shortName, param_type, source, unit=None, fmt=None):
        self.ID = ID
        self.name = name
        self.shortName = shortName
        # Synonym namespace dict predefined keys are CF: CF variables (), OS:OceanSites, SD:SeaDataNet abbreviations (TEMP, PSAL etc..)
        ns = ('CF', 'OS', 'SD')
        self.synonym = dict.fromkeys(ns)
        self.type = param_type
        self.source = source
        self.unit = unit
        self.format = fmt


class PanDataSet:
    """ PANGAEA DataSet
    The PANGAEA PanDataSet class enables the creation of objects which hold the necessary information, including data as well as metadata, to analyse a given PANGAEA dataset.

    Parameters
    ----------
    ID : str
        The identifier of a PANGAEA dataset. An integer number or a DOI is accepted here
    deleteFlag : str
        in case quality flags are avialable, this parameter defines a flag for which data should not be included in the data dataFrame.
        Possible values are listed here: https://wiki.pangaea.de/wiki/Quality_flag
    addQC : boolean
        adds a QC column for each parameter which contains QC flags

    Attributes
    ----------
    ID : str
        The identifier of a PANGAEA dataset. An integer number or a DOI is accepted here
    params : list of PanParam
        a list of all PanParam objects (the parameters) used in this dataset
    events : list of PanEvent
        a list of all PanEvent objects (the events) used in this dataset
    projects : list of PanProject
        a list containing the PanProjects objects referenced by this dataset
    data : pandas.DataFrame
        a pandas dataframe holding all the data
    loginstatus : str
        a label which indicates if the data set is protected or not default value: 'unrestricted'
    """

    def __init__(self, ID=None, paramlist=None, deleteFlag='', addQC=False):
        ### The constructor allows the initialisation of a PANGAEA dataset object either by using an integer dataset ID or a DOI
        self.ID = setID(ID)
        self.ns = {'md': 'http://www.pangaea.de/MetaData'}
        # Mapping should be moved to e.g netCDF class/module??
        # moddir = os.path.dirname(os.path.abspath(__file__))
        # self.CFmapping=pd.read_csv(moddir+'\\PANGAEA_CF_mapping.txt',delimiter='\t',index_col='ID')
        self.params = dict()
        self.defaultparams = ['Latitude', 'Longitude', 'Event', 'Elevation', 'Date/Time']
        self.paramlist = paramlist
        self.paramlist_index = []
        self.events = []
        self.projects = []
        # allowed geocodes for netcdf generation which are used as xarray dimensions not needed in the moment
        self._geocodes = {1599: 'Date_Time', 1600: 'Latitude', 1601: 'Longitude', 1619: 'Depth water'}
        self.loginstatus = 'unrestricted'
        self.deleteFlag = deleteFlag
        self.metadata = {}
        # print('trying to load data and metadata from PANGAEA')
        self.set_metadata()
        self.defaultparams = [s for s in self.defaultparams if s in self.params.keys()]
        self.addQC = addQC
        if self.paramlist is not None:
            if len(self.paramlist) != len(self.paramlist_index):
                raise ValueError


    def _setParameters(self, panXMLMatrixColumn):
        """
        Initializes the list of parameter objects from the metadata XML info
        """
        coln = dict()
        if panXMLMatrixColumn is not None:
            for matrix in panXMLMatrixColumn:
                paramstr = matrix.find("md:parameter", self.ns)
                panparID = int(_getID(str(paramstr.get('id'))))

                panparShortName = ''
                if paramstr.find('md:shortName', self.ns) is not None:
                    panparShortName = paramstr.find('md:shortName', self.ns).text
                    # Rename duplicate column headers
                    if panparShortName in coln:
                        coln[panparShortName] += 1
                        panparShortName = panparShortName + '_' + str(coln[panparShortName])
                    else:
                        coln[panparShortName] = 1
                panparType = matrix.get('type')
                panparUnit = None
                if paramstr.find('md:unit', self.ns) is not None:
                    panparUnit = paramstr.find('md:unit', self.ns).text
                panparFormat = matrix.get('format')

                self.params[panparShortName] = PanParam(panparID, paramstr.find('md:name', self.ns).text,
                                                        panparShortName, panparType, matrix.get('source'), panparUnit,
                                                        panparFormat)

    def get_data(self, addQC=False):
        """
        This method populates the data DataFrame with data from a PANGAEA dataset.
        In addition to the data given in the tabular ASCII file delivered by PANGAEA.

        Parameters:
        -----------
        addQC : boolean
            If this is set to True, pangaeapy adds a QC column in which the quality flags are separated.
            Each new column is named after the orgininal column plus a "_qc" suffix.
        """

        if self.metadata['hierarchyLevel'] is not None:
            if self.metadata['hierarchyLevel'].get('value') == 'parent':
                raise ValueError(f"""
        Data set is of type parent, please select one of its child datasets.
        The {len(self.children())} child datasets are \n{self.children()}""")

        # converting list of parameters` short names (from user input) to the list of parameters` indexes
        # the list of parameters` indexes is an argument for pd.read_csv
        if self.paramlist is not None:
            self.paramlist += self.defaultparams
            for parameter in self.paramlist:
                _iter = 0
                for shortName in self.params.keys():
                    if parameter == shortName:
                        self.paramlist_index.append(_iter)
                    _iter += 1
            if len(self.paramlist) != len(self.paramlist_index):
                raise ValueError("Error entering parameters`short names!")
        else:
            self.paramlist_index = None

        dataURL = "https://doi.pangaea.de/10.1594/PANGAEA." + str(self.ID) + "?format=textfile"
        panDataTxt = requests.get(dataURL).text
        panData = re.sub(r"/\*(.*)\*/", "", panDataTxt, 1, re.DOTALL).strip()
        # Read in PANGAEA Data
        data = pd.read_csv(io.StringIO(panData), index_col=False, error_bad_lines=False, sep=u'\t',
                                usecols=self.paramlist_index, names=list(self.params.keys()), skiprows=[0])
        # add geocode/dimension columns from Event

        # -- delete values with given QC flags
        if self.deleteFlag != '':
            if self.deleteFlag == '?' or self.deleteFlag == '*':
                self.deleteFlag = "\\" + self.deleteFlag
            data.replace(regex=r'^' + self.deleteFlag + '{1}.*', value='', inplace=True)
            # --- Replace Quality Flags for numeric columns
        if not addQC:
            data.replace(regex=r'^[\?/\*#\<\>]', value='', inplace=True)
        # --- Delete empty columns
        data = data.dropna(axis=1, how='all')
        for paramcolumn in list(self.params.keys()):
            if paramcolumn not in data.columns:
                del self.params[paramcolumn]
            # --- add QC columns
            elif addQC:
                if self.params[paramcolumn].type == 'numeric':
                    data[[paramcolumn + '_qc', paramcolumn]] = data[paramcolumn].astype(str).str.extract(
                        r'(^[\*/\?])?(.+)')
        # --- Adjust Column Data Types
        data = data.apply(pd.to_numeric, errors='ignore')
        if 'Date/Time' in data.columns:
            data.index = pd.to_datetime(data['Date/Time'], format='%Y/%m/%dT%H:%M:%S')
            data.index.names = ['index']
            data.pop('Date/Time')
        return data

    def download(self, path, name=None, **kwargs):

        if name is None:
            name = self.metadata['title'].replace(' ', '_')
            name = name.replace('-', '_')
        path = os.path.join(path, name+'.txt')
        self.get_data().to_csv(path, **kwargs)

        if 'hierarchyLevel' in self.metadata and self.metadata['hierarchyLevel'] is not None:
            self.metadata['hierarchyLevel'] = self.metadata['hierarchyLevel'].text
        fname = os.path.join(os.path.dirname(path), f'{name}_metadata.json')
        with open(fname, 'w') as fp:
            json.dump(self.metadata, fp, indent=4, sort_keys=False)

        return name


    def set_metadata(self):
        """
        The method initializes the metadata of the PanDataSet object using the information of a PANGAEA metadata XML file.
        """
        _metadata = {}
        metaDataURL = "https://doi.pangaea.de/10.1594/PANGAEA." + str(self.ID) + "?format=metainfo_xml"
        r = requests.get(metaDataURL)
        if r.status_code != 404:
            try:
                r.raise_for_status()
                xmlText = r.text
                xml = ET.fromstring(xmlText)
                self.metadata['loginstatus'] = xml.find('./md:technicalInfo/md:entry[@key="loginOption"]', self.ns).get('value')
                if self.metadata['loginstatus'] != 'unrestricted':
                    raise ValueError('Data set is protected')

                self.metadata['hierarchyLevel'] = xml.find('./md:technicalInfo/md:entry[@key="hierarchyLevel"]', self.ns)

                self.metadata['title'] = xml.find("./md:citation/md:title", self.ns).text
                self.metadata['year'] = xml.find("./md:citation/md:year", self.ns).text
                self.metadata['author_info'] = self.find_author_info(xml)
                self.metadata['project_info'] = self.find_project_info(xml)
                self.metadata['license_info'] = self.find_license_info(xml)

                topotypeEl = xml.find("./md:extent/md:topoType", self.ns)
                if topotypeEl is not None:
                    self.topotype = topotypeEl.text
                else:
                    self.topotype = None

                panXMLMatrixColumn = xml.findall("./md:matrixColumn", self.ns)
                self._setParameters(panXMLMatrixColumn)

            except requests.exceptions.HTTPError as e:
                print(e)

    def children(self):
        """Finds the child datasets of a parent dataset"""
        kinder = []
        childqueryURL = "https://www.pangaea.de/advanced/search.php?q=incollection:" + str(self.ID) + "&count=1000"
        r = requests.get(childqueryURL)
        if r.status_code != 404:
            s = r.json()
            for p in s['results']:
                kinder.append(p['URI'])
        return kinder

    def find_license_info(self, xml)->dict:
        lizenz = {}
        idx = 0
        for _license in xml.findall("./md:license", self.ns):

            l = _license.find("md:label", self.ns)
            lizenz[f'label_{idx}'] = l.text if l is not None else l
            n = _license.find("md:name", self.ns)
            lizenz[f'name_{idx}'] = n.text if n is not None else n
            u = _license.find("md:URI", self.ns)
            lizenz[f'URI_{idx}'] = u.text if u is not None else u

            idx += 1

        return lizenz

    def find_project_info(self, xml)->dict:
        projekt_info = {}
        idx = 0
        for project in xml.findall("./md:project", self.ns):

            l = project.find("md:label", self.ns)
            projekt_info[f'label_{idx}'] = l.text if l is not None else l
            n = project.find("md:name", self.ns)
            projekt_info['name_{idx}'] = n.text if n is not None else n
            u = project.find("md:URI", self.ns)
            projekt_info[f'URI_{idx}'] = u.text if u is not None else u
            uri = project.find("md:award/md:URI", self.ns)
            projekt_info[f'awardURI_{idx}'] = uri.text if uri is not None else uri

            idx += 1

        return projekt_info

    def find_author_info(self, xml)->dict:
        autor = {}
        idx = 0

        for author in xml.findall("./md:citation/md:author", self.ns):

            autor[f'lastname_{idx}'] = author.find("md:lastName", self.ns).text
            autor[f'firstname_{idx}'] = author.find("md:firstName", self.ns).text
            orcid = author.find("md:orcid", self.ns)
            autor[f'orcid_{idx}'] = orcid.text if orcid is not None else orcid

            idx += 1

        return autor


def setID(ID):
    """
    Initialize the ID of a data set in case it was not defined in the constructur
    Parameters
    ----------
    ID : str
        The identifier of a PANGAEA dataset. An integer number or a DOI is accepted here
    """
    idmatch = re.search(r'10\.1594\/PANGAEA\.([0-9]+)$', ID)

    if idmatch is not None:
        return idmatch[1]
    else:
       return ID

def _getID(panparidstr):
    panparidstr = panparidstr[panparidstr.rfind('.') + 1:]
    panparId = re.match(r"([a-z]+)([0-9]+)", panparidstr)
    if panparId:
        return panparId.group(2)
    else:
        return False

if __name__ == "__main__":
    # ds = PanDataSet('10.1594/PANGAEA.898217')
    # print(ds.data.shape)
    # ds = PanDataSet('10.1594/PANGAEA.882613')
    # print(ds.data.shape)
    # ds = PanDataSet('10.1594/PANGAEA.879494')
    # print(ds.data.shape)
    # ds = PanDataSet('10.1594/PANGAEA.831196')
    # print(ds.data.shape)
    ds = PanDataSet('10.1594/PANGAEA.919103')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.919104')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.909880')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.908290')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.892384')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.883587')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.882178')
    print(ds.get_data().shape)
    # ds = PanDataSet('10.1594/PANGAEA.811992')
    # print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.807883')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.778629')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.774595')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.746240')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.226925')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.905446')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.900958')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.890070')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.882611')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.879507')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.842446')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.841977')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.831193')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.811072')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.811076')
    print(ds.get_data().shape)
    ds = PanDataSet('10.1594/PANGAEA.912582')
    print(ds.get_data().shape)
