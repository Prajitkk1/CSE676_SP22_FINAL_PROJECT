# -*- coding: utf-8 -*-


import pandas as pd

import matplotlib.pyplot as plt




data_default = pd.read_csv("model_history_classification_default.csv") 
data_gan = pd.read_csv("model_history_classification_withGAN.csv") 



plt.plot(data_default['epoch'], data_default['accuracy'], label = "Default Dataset")
plt.plot(data_default['epoch'],  data_gan['accuracy'], label = "Modified Dataset")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Plot")
plt.savefig("Accuracy.png", dpi=400)
plt.show()
# data_default.plot.line(x='epoch', y='accuracy', ax=axes)
# data_gan.plot.line(x='epoch', y='accuracy', ax=axes)

plt.plot(data_default['epoch'], data_default['loss'], label = "Default Dataset")
plt.plot(data_default['epoch'],  data_gan['loss'], label = "Modified Dataset")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Plot")
plt.savefig("Loss.png", dpi=400)
plt.show()


