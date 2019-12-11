import matplotlib.pyplot as plt
import pandas as pd
isi = pd.read_csv('/cptjack/totem/barrylee/codes/real64---three-hepa/2019_10_17hepa-128-0.001_log.csv')
loss = isi['loss']
val_loss = isi['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training And Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
acc = isi['acc']
val_acc = isi['val_acc']
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training And Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
