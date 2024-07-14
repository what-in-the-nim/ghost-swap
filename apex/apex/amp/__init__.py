from ._amp_state import _amp_state, master_params
from .amp import (
    float_function,
    half_function,
    init,
    promote_function,
    register_float_function,
    register_half_function,
    register_promote_function,
)
from .frontend import initialize, load_state_dict, state_dict
from .handle import disable_casts, scale_loss
