
RES_COLUMNS = [
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


# column names for output.wql
DEF_WQL_COLUMNS = [
            'junk', 'rch_id', 'day', 'WTEMP(C)',  'ALGAE_INppm', "ALGAE_Oppm",
            "ORGN_INppm", "ORGN_OUTppm", "NH4_INppm",  "NH4_OUTppm",   "NO2_INppm",
            "NO2_OUTppm", "NO3_INppm", "NO3_OUTppm",  "ORGP_INppm", "ORGP_OUTppm",
            "SOLP_INppm", "SOLP_OUTppm",  "CBOD_INppm", "CBOD_OUTppm", "SAT_OXppm",
            "DISOX_INppm",  "DISOX_Oppm", "H20VOLUMEm3", "TRVL_TIMEhr",
        ]


DEF_RCH_COLUMNS = [
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

RCH_COL_MAP = {
    "FLOW_IN": "FLOW_INcms",
    "FLOW_OUT": "FLOW_OUTcms",
    "EVAP": "EVAPcms",
    "TLOSS": "TLOSScms",
    "SED_IN": "SED_INtons",
    "SED_OUT": "SED_OUTtons",
    "SEDCONC": "SEDCONCmg/kg",
    "ORGN_IN": "ORGN_INkg",
    "ORGN_OUT": "ORGN_OUTkg",
    "ORGP_IN": "ORGP_INkg",
    "ORGP_OUT": "ORGP_OUTkg",
    "NO3_IN": "NO3_INkg",
    "NO3_OUT": "NO3_OUTkg",
    "NH4_IN": "NH4_INkg",
    "NH4_OUT": "NH4_OUTkg",
    "NO2_IN": "NO2_INkg",
    "NO2_OUT": "NO2_OUTkg",
    "MINP_IN": "MINP_INkg",
    "MINP_OUT": "MINP_OUTkg",
    "CHLA_IN": "CHLA_INkg",
    "CHLA_OUT": "CHLA_OUTkg",
    "CBOD_IN": "CBOD_INkg",
    "CBOD_OUT": "CBOD_OUTkg",
    "DISOX_IN": "DISOX_INkg",
    "DISOX_OUT": "DISOX_OUTkg",
    "SOLPST_IN": "SOLPST_INmg",
    "SOLPST_OUT": "SOLPST_OUTmg",
    "SORPST_IN": "SORPST_INmg",
    "SORPST_OUT": "SORPST_OUTmg",
    "REACTPST": "REACTPSTmg",
    "VOLPST": "VOLPSTmg",
    "SETTLPST": "SETTLPSTmg",
    "RESUSP_PST": "RESUSP_PSTmg",
    "DIFFUSE": "DIFFUSEmg",
    "REACBEDPST": "REACBEDPSTmg",
    "BURYPST": "BURYPSTmg",
    "BED_PST": "BED_PSTmg",
    "BACTP_OUT": "BACTP_OUTct",
    "BACTLP_OUT": "BACTLP_OUTct",
    "CMETAL#1": "CMETAL#1kg",
    "CMETAL#2": "CMETAL#2kg",
    "CMETAL#3": "CMETAL#3kg"
}

# codes for output.rch file
RCH_OUT_CODES = {
    1: "FLOW_IN",
    2: "FLOW_OUT",
    3: "EVAP",
    4: "TLOSS",
    5: "SED_IN",
    6: "SED_OUT",
    7: "SEDCONC",
    8: "ORGN_IN",
    9: "ORGN_OUT",
    10: "ORGP_IN",
    11: "ORGP_OUT",
    12: "NO3_IN",
    13: "NO3_OUT",
    14: "NH4_IN",
    15: "NH4_OUT",
    16: "NO2_IN",
    17: "NO2_OUT",
    18: "MINP_IN",
    19: "MINP_OUT",
    20: "CHLA_IN",
    21: "CHLA_OUT",
    22: "CBOD_IN",
    23: "CBOD_OUT",
    24: "DISOX_IN",
    25: "DISOX_OUT",
    26: "SOLPST_IN",
    27: "SOLPST_OUT",
    28: "SORPST_IN",
    29: "SORPST_OUT",
    30: "REACTPST",
    31: "VOLPST",
    32: "SETTLPST",
    33: "RESUSP_PST",
    34: "DIFFUSEPST",
    35: "REACBEDPST",
    36: "BURYPST",
    37: "BED_PST",
    38: "BACTP_OUT",
    39: "BACTLP_OUT",
    40: "CAMETAL#1",
    41: "CAMETAL#2",
    42: "CAMETAL#3"
}

# codes for output.sub file
SUB_OUT_CODES = {
    1: "PRECIP",
    2: "SNOMELT",
    3: "PET",
    4: "ET",
    5: "SW",
    6: "PERC",
    7: "SURQ",
    8: "GW_Q",
    9: "WYLD",
    10: "SYLD",
    11: "ORGN",
    12: "ORGP",
    13: "NSURQ",
    14: "SOLP",
    15: "SEDP"
}

# codes for output.sub file
HRU_OUT_CODES = {
    1: "PRECIP",
    2: "SNOFALL",
    3: "SNOMELT",
    4: "IRR",
    5: "PET",
    6: "ET",
    7: "SW_INIT",
    8: "SW_END",
    9: "PERC",
    10: "GW_RCHG",
    11: "DA_RCHG",
    12: "REVAP",
    13: "SA_IRR",
    14: "DA_IRR",
    15: "SA_ST",
    16: "DA_ST",
    17: "SURQ_GEN",
    18: "SURQ_CNT",
    19: "TLOSS",
    20: "LATQ",
    21: "GW_Q",
    22: "WYLD",
    23: "DAILYCN",
    24: "TMP_AV",
    25: "TMP_MX",
    26: "TMP_MN",
    27: "SOL_TMP",
    28: "SOLAR",
    29: "SYLD",
    30: "USLE",
    31: "N_APP",
    32: "P_APP",
    33: "NAUTO",
    34: "PAUTO",
    35: "NGRZ",
    36: "PGRZ",
    37: "CFERTN",
    38: "CFERTP",
    39: "NRAIN",
    40: "NFIX",
    41: "F_MMN",
    42: "A_MN",
    43: "A_SN",
    44: "F_MP",
    45: "AO_LP",
    46: "L_AP",
    47: "A_SP",
    48: "DNIT",
    49: "NUP",
    50: "PUP",
    51: "ORGN",
    52: "ORGP",
    53: "SEDP",
    54: "NSURQ",
    55: "NLATQ",
    56: "NO3L",
    57: "NO3GW",
    58: "SOLP",
    59: "P_GW",
    60: "W_STRS",
    61: "TMP_STRS",
    62: "N_STRS",
    63: "P_STRS",
    64: "BIOM",
    65: "LAI",
    66: "YLD",
    67: "BACTP",
    68: "BACTLP",
}
