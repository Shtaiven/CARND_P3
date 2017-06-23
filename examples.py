import cv2
import matplotlib.image as mpimg

img = mpimg.imread('example.jpg')

# Color space change
cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

# Flipped image
flip_img = cv2.flip(img, 1)

# Cropped image
h = img.shape[0]
crop_img = img[50:h-20,:] # 50px cropped from top, 20px cropped from bottom

# Save the examples
mpimg.imsave('y_channel.jpg', img[:,:,0], cmap='gray')
mpimg.imsave('u_channel.jpg', img[:,:,1], cmap='gray')
mpimg.imsave('v_channel.jpg', img[:,:,2], cmap='gray')
mpimg.imsave('flipped.jpg', flip_img)
mpimg.imsave('cropped.jpg', crop_img)
