import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('j.png', 1)

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(img)
ax[0].set_title("Original")
ax[1].imshow(opening, cmap=plt.cm.gray)
ax[1].set_title("Opening")
fig.tight_layout()
plt.show()
