import matplotlib.pyplot as plt
import numpy as np
import cv2


'''color_1 = [255, 0, 0]   #red
color_2 = [0, 255, 0]   #green
color_3 = [0, 0, 255]   #blue
color_4 = [30, 127, 150]   #grey

plt.imshow(np.array([
    [color_1, color_2],
    [color_3, color_4],
]))

plt.show()'''

#SIZE = 2
#SIZE = 10
SIZE = 100

colors = np.array(
    np.array([
        np.array([np.random.randint(0, 255, 3) for x in range(SIZE)]) for x in range(SIZE)
    ])
)
print(np.array(colors).shape)
plt.imshow(colors)
plt.show()


#read image
'''image = cv2.imread('chewbacca.jpeg')    #BRG
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Gray

WIDTH = 300
HEIGHT = 300
resized = cv2.resize(image, (WIDTH, HEIGHT))

print(type(resized))
print(resized.shape)
plt.imshow(resized, cmap='gray')
plt.show()'''
