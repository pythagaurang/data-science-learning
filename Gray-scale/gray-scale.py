# Image transformation with matplotlib

import matplotlib.pyplot as plt

filename='t9np10qvti931.png'

img=plt.imread(filename)

plt.subplot(2,1,1)
plt.imshow(img)
plt.axis('off')

# to see what intensity does just 
# remove the cmap argument from
# plt.imshow function

intensity=img.sum(axis=2)
plt.subplot(2,1,2)
plt.imshow(intensity,cmap='grey')
plt.axis('off')

plt.tight_layout()
plt.show()
