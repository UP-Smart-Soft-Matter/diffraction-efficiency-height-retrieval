import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from skimage.feature import peak_local_max
import scipy.ndimage as ndi


image_path = r"xm17.tif"

img = cv2.imread(image_path)

class DiffractionEfficiency(object):
    def __init__(self, image, image_crop_x=(300, 500), image_crop_y=(250, 850), roi_length=20, peak_local_max_min_distance=50):
        self.__img_matrix = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[image_crop_x[0]:image_crop_x[1], image_crop_y[0]:image_crop_y[1]]
        self.__roi_length = roi_length
        self.__peak_local_max_min_distance = peak_local_max_min_distance

        self._intensities = np.array([])

        maxima = peak_local_max(self.__img_matrix, min_distance=self.__peak_local_max_min_distance)

        span_y = np.ptp(maxima[:, 0])
        span_x = np.ptp(maxima[:, 1])
        sort_col = 1 if span_x >= span_y else 0

        self.__maxima = maxima[np.argsort(maxima[:, sort_col])]

        for maximum in maxima:
            roi_x_start = maximum[1] - self.__roi_length
            roi_x_stop = maximum[1] + self.__roi_length
            roi_y_start = maximum[0] - self.__roi_length
            roi_y_stop = maximum[0] + self.__roi_length


            order_intensity = np.sum(self.__img_matrix[roi_y_start:roi_y_stop, roi_x_start:roi_x_stop])
            self._intensities = np.append(self._intensities, order_intensity)

        self._normalized_intensities = self._intensities / np.sum(self._intensities)

    def plot_image(self):

        plt.imshow(self.__img_matrix, cmap='viridis')

        for maximum in self.__maxima:
            roi_x_start = maximum[1] - self.__roi_length
            roi_y_start = maximum[0] - self.__roi_length
            plt.gca().add_patch(
                Rectangle((roi_x_start, roi_y_start), 2 * self.__roi_length, 2 * self.__roi_length, linewidth=1, edgecolor='r', facecolor='none'))

        y0, x0, y1, x1 = extend_plot_line(self.__maxima[0], self.__maxima[-1], 50)

        plt.plot([x0, x1], [y0, y1], '-', c="lightblue", linewidth=.3)

        plt.plot(self.__maxima[:, 1], self.__maxima[:, 0], 'r.')
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_diffraction_efficiency(self):
        max_order = int(np.floor(len(self._normalized_intensities) / 2))
        bar_x = np.arange(-max_order, max_order + 1, 1)

        plt.bar(bar_x, self._normalized_intensities)
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_profile (self):
        y0, x0, y1, x1 = extend_plot_line(self.__maxima[0], self.__maxima[-1], 50)
        num = 1000
        x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
        zi = ndi.map_coordinates(self.__img_matrix, np.vstack((y,x)))

        plt.plot(zi)
        plt.tight_layout()
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


de = DiffractionEfficiency(img)

de.plot_image()
de.plot_diffraction_efficiency()
de.plot_profile()
