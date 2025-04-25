import glob, os
import torch
tmp_path= "param_data/tmp_250414_180816"



for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
    buffers = torch.load(file)
    
    param = []
    for key in buffers.keys():
        print(key)