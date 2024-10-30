import glob, os
import torch
tmp_path= "./param_data/tmp_241028_112725"



for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
    buffers = torch.load(file)
    
    param = []
    for key in buffers.keys():
        print(key)