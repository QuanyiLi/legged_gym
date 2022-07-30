import torch as th
import numpy as np

if __name__=="__main__":
    path = "/home/quanyi/neurips/anymal_tanh/model_1500.pt"
    weights = th.load(path)["model_state_dict"]
    ret = {}
    for layer, weight in weights.items():
        ret[layer] = weight.detach().cpu().numpy()
    np.savez_compressed("anymal_tanh", **ret)
    print("model is converted and saved!")
