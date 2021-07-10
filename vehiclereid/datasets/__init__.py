from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .veri import VeRi
from .vehicleid import VehicleID
from .aic20_reid import AIC20_ReID
from .aic20_reid_type import AIC20_ReID_Type
from .aic20_reid_color import AIC20_ReID_Color
from .aic20_reid_camid import AIC20_ReID_CamID
from .aic20_reid_simu import AIC20_ReID_Simu
from .aic20_reid_full import AIC20_ReID_Full
from .aic20_reid_simu_color import AIC20_ReID_Simu_Color
from .aic20_reid_simu_type import AIC20_ReID_Simu_Type

__imgreid_factory = {
    'veri': VeRi,
    'vehicleID': VehicleID,
    'AIC20_ReID': AIC20_ReID,
    'AIC20_ReID_Type': AIC20_ReID_Type,
    'AIC20_ReID_Color': AIC20_ReID_Color,
    'AIC20_ReID_CamID': AIC20_ReID_CamID,
    'AIC20_ReID_Simu': AIC20_ReID_Simu,
    'AIC20_ReID_Full': AIC20_ReID_Full,
    'AIC20_ReID_Simu_Color': AIC20_ReID_Simu_Color,
    'AIC20_ReID_Simu_Type': AIC20_ReID_Simu_Type
}


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)

