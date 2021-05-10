import os
import cv2
import copy
import math
import pytesseract
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert


class Image:
    def __init__(self, path: str):
        self.path = path
        self.directory = os.path.join(*path.split("/")[:-1])
        self.file = path.split("/")[-1]
        self.line_thickness = 12

    def get_image(self):
        img = cv2.imread(self.path)
        img2 = img[:]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{self.directory}/01_gray.png', gray)

        return self.delete_border_auto(img, img2)

    def draw_lines(self, img: np.ndarray, lines: np.ndarray) -> None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x2 = x2 + 1
            y2 = y2 + 1
            img[y1:y2, x1:x2] = 255
            img[(y2 - 1):y1, x1:x2] = 255

    def draw_key_points(self, img, contours):
        cells = []
        for contour in contours[-1:0:-1]:
            sort_points = sorted(contour, key=lambda m: m[0][0] + m[0][1])
            point_1 = tuple(sort_points[0][0])
            point_2 = tuple(sort_points[-1][0])
            center = (int((point_2[0] + point_1[0]) / 2), int((point_2[1] + point_1[1]) / 2))
            cells.append((point_1, point_2, center))
            cv2.drawMarker(img, point_1, (0, 255, 0), 4, 20, 5)
            cv2.drawMarker(img, point_2, (0, 255, 0), 4, 20, 5)
            cv2.drawMarker(img, center, (0, 0, 255), 6, 20, 5)
        return cells

    def join(self, arr1, axis):
        arr2 = np.concatenate((arr1[:, 0, axis], arr1[:, 1, axis]))
        arr2 = np.sort(arr2)
        print("np_cells1")
        print(arr2)
        return arr2

    def cluster_axis(self, cells, axis):
        np_cells1 = self.join(cells, axis)

        cells_cluster = self.cluster_number(np_cells1, 15)
        print("cells_cluster")
        print(cells_cluster)

        arr = cells[:, 0:2, axis]
        print("Исходный массив: ", arr)
        centroids = sorted(list(set(cells_cluster)))
        print("centroids")
        print(centroids)

        tresholds = []
        for i, num in enumerate(centroids):
            tresholds.append(math.ceil((centroids[i + 1 - len(centroids)] + num) / 2))
        tresholds.pop()
        print("Пороговые значения: ", tresholds)
        centroids = np.array(centroids, np.uint64)

        indices = np.searchsorted(tresholds, arr)
        cells[:, 0:2, axis] = centroids[indices]

    def delete_border_auto(self, gray, input_img):
        kernel = np.ones((5, 5), 'uint8')  # Ядро 5х5
        eroded = cv2.erode(gray, kernel, iterations=1)
        cv2.imwrite(f'{self.directory}/02_eroded.png', eroded)
        _, threshold_img = cv2.threshold(eroded, 240, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'{self.directory}/03_threshold.png', threshold_img)
        invert_img = invert(threshold_img / 255)
        cv2.imwrite(f'{self.directory}/04_invert_img.png', invert_img * 255)
        skeleton = skeletonize(invert_img)
        skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{self.directory}/05_skeleton.png', skeleton)
        skeleton_eroded = cv2.dilate(skeleton, kernel, iterations=1)
        cv2.imwrite(f'{self.directory}/06_skeleton_eroded.png', skeleton_eroded)
        _, skeleton_threshold = cv2.threshold(skeleton_eroded, 5, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'{self.directory}/07_skeleton_threshold.png', skeleton_threshold)
        lines = cv2.HoughLinesP(skeleton_threshold, 1, np.pi/180, 100, minLineLength=100, maxLineGap=5)
        self._draw_lines_p(input_img, lines)
        cv2.imwrite(f'{self.directory}/08_red_lines.png', input_img)

        empty_img = np.zeros(input_img.shape[0:2], np.uint8)
        self.draw_lines(empty_img, lines)
        _, empty_threshold = cv2.threshold(empty_img, 10, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'{self.directory}/09_empty_threshold.png', empty_threshold)

        contours, hierarchy = cv2.findContours(empty_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        print("Contours: ", len(contours))
        for i in range(len(contours)):
            cv2.drawContours(gray, contours, i, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 2)
        cells = self.draw_key_points(gray, contours)

        full_path = f'{self.directory}/{self.file[:-4]}_10_img_markers.png'
        cv2.imwrite(full_path, gray)

        cells.sort(key=lambda x: (x[0][0], x[0][1]))
        np_cells = np.array(cells, np.uint64)
        print("cells")
        for cell in cells:
            print(cell)

        self.cluster_axis(np_cells, 0)
        print("Конечный массив: ")
        for cell in cells:
            print(cell[0], cell[1], cell[2])

        self.cluster_axis(np_cells, 1)
        print("Конечный массив: ")
        for cell in cells:
            print(cell[0], cell[1], cell[2])

        print("lexsort", np.lexsort((np_cells[:, 0, 1], np_cells[:, 0, 0])))

        cells_3 = np_cells[np.lexsort((np_cells[:, 0, 0], np_cells[:, 0, 1]))]
        print("Sorted: ")
        os.mkdir(f'{self.directory}/{self.file[:-4]}')
        print(input_img.shape)
        for i, cell in enumerate(cells_3):
            print(cell[0], cell[1], cell[2])
            cell_img = input_img[cell[0][1]:cell[1][1], cell[0][0]:cell[1][0]]
            print(cell_img.shape)
            # cell_path = f'{self.directory}/{self.file[:-4]}/{i}.png'
            # cv2.imwrite(cell_path, cell_img)
            # self.separate_numbers(cell_img)

        return full_path

    @staticmethod
    def cluster_number(arr, lim) -> np.ndarray:
        print("Clusterizing...")
        final_arr = copy.copy(arr)
        temp_arr = copy.copy(arr)
        temp = []
        for i, number in enumerate(arr):
            if i > 0:
                if number - arr[i - 1] > lim:
                    np.place(final_arr, temp_arr < 1, int(sum(temp) / len(temp)))
                    temp_arr = copy.copy(arr)
                    temp_arr[i] = 0
                    temp = [number, ]
                else:
                    temp_arr[i] = 0
                    temp.append(number)
            else:
                temp_arr[i] = 0
                temp.append(number)
        np.place(final_arr, temp_arr < 1, int(sum(temp) / len(temp)))
        return final_arr

    def _draw_lines_p(self, img2, lines):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 5)


def ocr_image(image, config):
    return pytesseract.image_to_string(
        image,
        config=config
    )
