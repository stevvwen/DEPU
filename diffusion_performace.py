import torch
import numpy as np
import matplotlib.pyplot as plt


diffusion_data= [-1267.2107479009412, -1256.85530783230077, -1269.5418044366834, -1254.4317158324725, -1248.735259810553]


average_performance_before= -1402.4046816525931
x_axis= [50, 100, 150, 200, 250]


plt.plot(x_axis, diffusion_data, label='Diffusion Model', marker='o')
plt.axhline(y=average_performance_before, color='r', linestyle='--', label='Average Performance Before Upscaling')
plt.title('Diffusion Model Performance')
plt.xlabel('Epochs')
plt.ylabel('Performance')
plt.legend()
plt.grid()
plt.savefig('diffusion_performance.png')