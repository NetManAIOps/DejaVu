import torch


# noinspection PyUnresolvedReferences,PyProtectedMember
def set_jit_enabled(enabled: bool):
    """ Enables/disables JIT """
    if torch.__version__ < "1.7":
        torch.jit._enabled = enabled
    else:
        if enabled:
            torch.jit._state.enable()
        else:
            torch.jit._state.disable()
