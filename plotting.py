from utils import get_dataframe
from chip import CDS


IF = get_dataframe('IF')
cd = CDS(IF.index, IF.CLOSE, IF.TURN)
cd.plot_dist('2019-01-02', thresh=0.01, bin_size=10)