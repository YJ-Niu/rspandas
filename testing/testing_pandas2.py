import time
start_time = time.time()

import rsnumpy as np  # noqa: E402

import rspandas as pd  # noqa: E402


def print_series(num, s):
    print("++++++++++++++++++++", num)
    print(s)
    print()


s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
print_series(1, s)

end_time = time.time()
print("end_time - start_time:", end_time - start_time)
