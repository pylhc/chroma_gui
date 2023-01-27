# Files to be saved
FILENAME = "./TIMBER_DATA.csv"
BACKUP_FILENAME = "./TIMBER_DATA_{now}.csv"

FILENAME_PKL = "./TIMBER_RAW_BBQ.pkl.gz"
BACKUP_FILENAME_PKL = "./TIMBER_RAW_BBQ_{now}.pkl.gz"

FILENAME_HDF = "./TIMBER_RAW_BBQ.hdf"
BACKUP_FILENAME_HDF = "./TIMBER_RAW_BBQ_{now}.hdf"

# Variables to query for
TIMBER_VARS = ['LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_FREQ_1',  # Good BBQ data usually, sometimes noisy
             'LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_FREQ_2',
             'LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_FREQ_1',
             'LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_FREQ_2',
             'LHC.BQBBQ.CONTINUOUS_HS.B1:ACQ_DATA_H',  # Raw BBQ
             'LHC.BQBBQ.CONTINUOUS_HS.B1:ACQ_DATA_V',
             'LHC.BQBBQ.CONTINUOUS_HS.B2:ACQ_DATA_H',  # Raw BBQ
             'LHC.BQBBQ.CONTINUOUS_HS.B2:ACQ_DATA_V',
             'LHC.BOFSU:RADIAL_TRIM_B1',
             'BFC.LHC:TuneFBAcq:tuneB1H',  # Completely off for some reason sometimes
             'BFC.LHC:TuneFBAcq:tuneB1V',
             'BFC.LHC:TuneFBAcq:tuneB2H',
             'BFC.LHC:TuneFBAcq:tuneB2V',
             'BFC.LHC:RadialLoopFBAcq:fradialLoopTrim',  # DPP data, doesn't always contain something
             'ALB.SR4.B1:FGC_FREQ',  # RF Frequency
             'ALB.SR4.B2:FGC_FREQ'
             ]

# Vars not to be saved in the CSV
TIMBER_RAW_VARS = ['LHC.BQBBQ.CONTINUOUS_HS.B1:ACQ_DATA_H',
                   'LHC.BQBBQ.CONTINUOUS_HS.B1:ACQ_DATA_V',
                   'LHC.BQBBQ.CONTINUOUS_HS.B2:ACQ_DATA_H',
                   'LHC.BQBBQ.CONTINUOUS_HS.B2:ACQ_DATA_V',
                  ]
