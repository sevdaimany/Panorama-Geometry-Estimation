import sys
import torch
import torch.amp
import torch.cuda.amp as cuda_amp
import torch.nn as nn

# =========================================================================
# MONKEY PATCH V4: Real AMP + Eager Mode Enforcer
# =========================================================================

# 1. AMP fwd/bwd decorators (Inject directly into the REAL torch.amp)
def compatible_custom_fwd(*args, **kwargs):
    kwargs.pop("device_type", None)
    if len(args) == 0:
        def decorator(f):
            return cuda_amp.custom_fwd(f, **kwargs)
        return decorator
    return cuda_amp.custom_fwd(args[0], **kwargs)

def compatible_custom_bwd(*args, **kwargs):
    kwargs.pop("device_type", None)
    if len(args) == 0:
        def decorator(f):
            return cuda_amp.custom_bwd(f, **kwargs)
        return decorator
    return cuda_amp.custom_bwd(args[0], **kwargs)

# We NO LONGER replace the whole module. We just add what's missing.
torch.amp.custom_fwd = compatible_custom_fwd
torch.amp.custom_bwd = compatible_custom_bwd

# 2. NEUTRALIZE DYNAMO (torch.compile)
# Dynamo crashes when trying to trace monkey-patched code.
# By making torch.compile a pass-through, the model runs safely in eager mode.
torch.compile = lambda model, *args, **kwargs: model

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# 3. Deep Intercept for Dynamo Config
import torch._dynamo.config
_orig_dynamo_setattr = torch._dynamo.config.__class__.__setattr__

def _patched_dynamo_setattr(self, name, value):
    if name == 'accumulated_cache_size_limit':
        name = 'cache_size_limit'
        if not hasattr(self, name):
            return
    return _orig_dynamo_setattr(self, name, value)

torch._dynamo.config.__class__.__setattr__ = _patched_dynamo_setattr

# 4. Polyfill for torch.nn.Buffer
if not hasattr(nn, 'Buffer'):
    class _MockBuffer:
        def __init__(self, tensor):
            self.tensor = tensor

    nn.Buffer = lambda t, *args, **kwargs: _MockBuffer(t)

    _orig_module_setattr = nn.Module.__setattr__
    def _patched_module_setattr(self, name, value):
        if isinstance(value, _MockBuffer):
            self.register_buffer(name, value.tensor)
        else:
            _orig_module_setattr(self, name, value)
    
    nn.Module.__setattr__ = _patched_module_setattr

# =========================================================================
# END MONKEY PATCH
# =========================================================================