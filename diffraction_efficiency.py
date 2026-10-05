import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from skimage.feature import peak_local_max
import scipy.ndimage as ndi


image_path = r"xm17.tif"

img = cv2.imread(image_path)

x_start = 300
x_stop = 500
y_start = 250
y_stop = 850

roi_length = 20

def get_diffraction_efficiency(img, x_start, x_stop, y_start, y_stop, roi_length):

    img_matrix = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[x_start:x_stop, y_start:y_stop]
    plt.imshow(img_matrix, cmap='viridis')

    maxima = peak_local_max(img_matrix, min_distance=50)
    maxima = maxima[np.argsort(maxima[:, 1])]

    print(maxima)

    y0, x0 = maxima[0]
    y1, x1 = maxima[-1]

    print(x0, y0)
    print(x1, y1)

    plt.plot([x0, x1], [y0, y1], 'b-', linewidth=.3)

    intensities = np.array([])
    for maximum in maxima:
        roi_x_start = maximum[1]-roi_length
        roi_x_stop = maximum[1]+roi_length
        roi_y_start = maximum[0]-roi_length
        roi_y_stop = maximum[0]+roi_length
        plt.gca().add_patch(
            Rectangle((roi_x_start, roi_y_start), 2 * roi_length, 2 * roi_length, linewidth=1, edgecolor='r',
                      facecolor='none'))

        order_intensity = np.sum(img_matrix[roi_y_start:roi_y_stop, roi_x_start:roi_x_stop])
        intensities = np.append(intensities, order_intensity)

    plt.plot(maxima[:, 1], maxima[:, 0], 'r.')
    plt.show()
    plt.close()

    normalized_intensities = intensities / np.sum(intensities)
    max_order = int(np.floor(len(intensities)/2))
    bar_x = np.arange(-max_order, max_order+1, 1)

    plt.bar(bar_x, normalized_intensities)
    plt.show()
    plt.close()

    y0, x0 = maxima[0]
    y1, x1 = maxima[-1]
    num = 1000
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    zi = ndi.map_coordinates(img_matrix, np.vstack((y,x)))

    plt.plot(zi)
    plt.show()
    plt.close()


get_diffraction_efficiency(img, x_start, x_stop, y_start, y_stop, roi_length)