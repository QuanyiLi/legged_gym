import torch as th
import numpy as np

if __name__=="__main__":
    path = "/home/quanyi/neurips/forward/Jul30_18-26-29_/model_900.pt"
    weights = th.load(path)["model_state_dict"]
    ret = {}
    for layer, weight in weights.items():
        ret[layer] = weight.detach().cpu().numpy()
    np.savez_compressed("forward_cassie_elu.npz", **ret)
    print("model is converted and saved!")
