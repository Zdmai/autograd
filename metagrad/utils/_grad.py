def need_grad(val):
    def need_grad_fn(fn):
        if val.require_grad:
            return fn
        else :
            return lambda : None
    return need_grad_fn
