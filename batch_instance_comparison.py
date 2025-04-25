import torch
import numpy as np
import matplotlib.pyplot as plt




batchnorm_data=[-1300.0167083631754, -593.9976375800082, -220.42892400067794, -445.66000351392506, -211.64263372950845, -284.04419045963584,
                -220.38537319094831, -266.49526441296615, -262.23019269448616, -220.38537319094831, -211.64263372950845]
insnorm_data= [-1446.8864645671486, -1180.823678308878, -1105.8379027441508, -1058.2985137943017, -964.1647655571285, -1066.5416431744507,
               -1087.6719175245403, -960.364224624044, -1095.6511776600896, -1033.133125402705, -950.3918900406621]


best_batch_norm= -148.27415765067843
stable_baseline_performance= -135

x_axis= [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

plt.plot(x_axis, batchnorm_data, label='Batch Normalization', color='blue')
plt.plot(x_axis, insnorm_data, label='Instance Normalization', color='red')
plt.axhline(y=best_batch_norm, color='green', linestyle='--', label='Best Batch Norm Model')
plt.axhline(y=stable_baseline_performance, color='orange', linestyle='--', label='Stable Baseline Performance')
plt.xlabel('Epoch')
plt.ylabel('Mean Generated Model Reward')
plt.title('Batch Normalization vs Instance Normalization')
plt.legend()
plt.grid()
plt.savefig('batchnorm_vs_insnorm.png')