import time

COUNT = 'cnt'
TIME = 'time'


def profile(target_function):

    def wrapper(*args, **kwds):
        tstart = time.time()
        obj = args[0]
        if not hasattr(obj, '_perf_info'):
            obj._perf_info = {}
        fname = target_function.__name__
        fun_info = obj._perf_info.get(fname)
        if fun_info is None:
            obj._perf_info[fname] = fun_info = {COUNT: 0, TIME: 0.}
        func = target_function(*args, **kwds)
        fun_info[COUNT] += 1
        fun_info[TIME] += time.time() - tstart
        return func

    return wrapper
