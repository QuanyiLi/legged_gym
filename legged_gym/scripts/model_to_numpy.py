import torch as th
import numpy as np

if __name__=="__main__":
    path = "/home/quanyi/legged_gym/logs/rough_anymal_c/Jul28_17-17-21_/model_1500.pt"
    weights = th.load(path)["model_state_dict"]
    ret = {}
    for layer, weight in weights.items():
        ret[layer] = weight.detach().cpu().numpy()
    np.savez_compressed("anymal_elu", **ret)
    print("model is converted and saved!")
