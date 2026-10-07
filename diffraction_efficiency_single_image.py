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

    span_y = np.ptp(maxima[:, 0])
    span_x = np.ptp(maxima[:, 1])
    sort_col = 1 if span_x >= span_y else 0

    maxima = maxima[np.argsort(maxima[:, sort_col])]

    y0, x0, y1, x1 = extend_plot_line(maxima[0], maxima[-1], 50)

    plt.plot([x0, x1], [y0, y1], '-', c="lightblue", linewidth=.3)

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

    num = 1000
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    zi = ndi.map_coordinates(img_matrix, np.vstack((y,x)))

    plt.plot(zi)
    plt.show()
    plt.close()

def extend_plot_line(p1: list, p2: list, width: int):
    assert len(p1) == len(p2) == 2

    alpha = np.arctan2((p2[0]-p1[0]), (p2[1]-p1[1]))
    delta_y = np.sin(alpha)*width
    delta_x = np.cos(alpha)*width

    y0 = p1[0]-delta_y
    x0 = p1[1]-delta_x
    y1 = p2[0]+delta_y
    x1 = p2[1]+delta_x

    return y0, x0, y1, x1



get_diffraction_efficiency(img, x_start, x_stop, y_start, y_stop, roi_length)