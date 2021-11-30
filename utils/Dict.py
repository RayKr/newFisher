
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict_to_obj(v)
    return d
