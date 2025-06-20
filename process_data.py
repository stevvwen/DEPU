from hmac import new
import torch




data= torch.load("param_data/Pendulum-v1_current_policy/data.pt")

param_data= data['pdata']

new_param_data= []

for index in range(len(param_data)):
    new_param_data.append(param_data[index][:1217])

new_param_data= new_param_data[:1000]  # Limit to the first 1000 entries

data['pdata']= torch.stack(new_param_data, dim=0)
torch.save(data, "param_data/Pendulum-v1_current_policy/data.pt")