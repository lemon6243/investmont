import numpy as np
Xr = np.load("dataset_all_3d/X_raw_3d.npy")
absmax = np.abs(Xr)
print("500(=5m) 넘는 값 비율:", round(float((absmax > 500).mean()), 5))
print("1000(=10m) 넘는 값 비율:", round(float((absmax > 1000).mean()), 6))
print("99.9 퍼센타일:", round(float(np.percentile(absmax, 99.9)), 1))
print("99 퍼센타일:", round(float(np.percentile(absmax, 99)), 1))
