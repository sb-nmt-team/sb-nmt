import numpy as np
import pickle
res = []
with open("translation_logs.pkl", "rb") as f:
    while True:
        try:
            res.append(pickle.load(f))
            print(" ".join(res[-1][0][1:-1]))
        except:
            break
