import os
import copy
import math
import cv2
import xlsxwriter
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

from itertools import chain
from collections import Counter
from skimage.morphology import skeletonize
from skimage.util import invert
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans


class Image:
    def __init__(self, path: str):
        self.path = path
        self.directory = os.path.join(*path.split("/")[:-1])
        self.file = path.split("/")[-1]
        self.line_thickness = 12
        self.rows = []

    def get_image(self):
        img = cv2.imread(self.path)
        img2 = img[:]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{self.directory}/01_gray.png', gray)

        cells = self.delete_border_auto(img, img2)
        self.get_rows(cells)
        self.count_sub_cells()
        result = self.reset_sub_cells()
        return result

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
        # print("Contours: ", len(contours))
        for i in range(len(contours)):
            cv2.drawContours(gray, contours, i, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 2)
        cells = self.draw_key_points(gray, contours)

        full_path = f'{self.directory}/{self.file[:-4]}_10_img_markers.png'
        cv2.imwrite(full_path, gray)
        return cells

    def get_rows(self, cells):
        cells.sort(key=lambda x: (x[0][0], x[0][1]))
        np_cells = np.array(cells, np.uint64)

        self.cluster_axis(np_cells, 0)
        self.cluster_axis(np_cells, 1)

        print("lexsort", np.lexsort((np_cells[:, 0, 1], np_cells[:, 0, 0])))

        cells_3 = np_cells[np.lexsort((np_cells[:, 0, 0], np_cells[:, 0, 1]))]

        u, indices = np.unique(cells_3[:, 0, 1], return_inverse=True)

        for i in u:
            row = Row(i)
            for cell in cells_3[np.where(cells_3[:, 0, 1] == i)]:
                cell = Cell(cell[0], cell[1], cell[2])
                row.cells.append(cell)

            self.rows.append(row)

    def count_sub_cells(self):
        files_dir = f'{self.directory}/{self.file[:-4]}'
        os.mkdir(files_dir)
        img_to_cut = cv2.imread(self.path)

        for i, row in enumerate(self.rows[2:]):
            for j, cell in enumerate(row.cells):
                print(cell.start, cell.end, cell.center)
                cell_img = img_to_cut[cell.start[1]:cell.end[1], cell.start[0]:cell.end[0]]

                n = self.transform_number(i)
                m = self.transform_number(j)

                cell_path = f'{files_dir}/{n}-{m}.png'

                try:
                    cv2.imwrite(cell_path, cell_img)
                    cell.ocr(cell_img)
                except cv2.error:
                    pass

    def reset_sub_cells(self):
        img_to_cut = cv2.imread(self.path)
        print()
        print(img_to_cut.shape)
        for row in self.rows[2:]:
            print("height", row.height)
            for cell in row.cells:
                # print("До пересчета", cell.sub_cells)
                # if cell.height != row.height:
                cell.reocr(img_to_cut, row.height)
                print("Конечные значения", cell.sub_cells)

        for row in self.rows[2:]:
            print("fullness", row.fullness)
            X = np.array(row.fullness)
            cluster_min, cluster_max = KMeans(2).fit(X.reshape(-1, 1)).cluster_centers_
            print(cluster_min, cluster_max)
            mean = (cluster_min[0] + cluster_max[0]) / 2
            print(mean)
            row.min_fullness = mean

            plt.scatter([i for i in range(len(row.fullness))], row.fullness)
            plt.plot([0, 40], [mean, mean])
        plt.savefig(f"111.png")
        return self.rows[2:]

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
        # print("np_cells1")
        # print(arr2)
        return arr2

    def cluster_axis(self, cells, axis):
        np_cells1 = self.join(cells, axis)

        cells_cluster = self.cluster_number(np_cells1, 15)
        # print("cells_cluster")
        # print(cells_cluster)

        arr = cells[:, 0:2, axis]
        # print("Исходный массив: ", arr)
        centroids = sorted(list(set(cells_cluster)))
        # print("centroids")
        # print(centroids)

        tresholds = []
        for i, num in enumerate(centroids):
            tresholds.append(math.ceil((centroids[i + 1 - len(centroids)] + num) / 2))
        tresholds.pop()
        # print("Пороговые значения: ", tresholds)
        centroids = np.array(centroids, np.uint64)

        indices = np.searchsorted(tresholds, arr)
        cells[:, 0:2, axis] = centroids[indices]

    @staticmethod
    def cluster_number(arr, lim) -> np.ndarray:
        # print("Clusterizing...")
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

    @staticmethod
    def _draw_lines_p(img2, lines):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 5)

    @staticmethod
    def transform_number(n):
        if len(str(n)) == 1:
            return f'00{n}'
        elif len(str(n)) == 2:
            return f'0{n}'
        else:
            return str(n)


def crop_to_text(image):
    MAX_COLOR_VAL = 255
    BLOCK_SIZE = 15
    SUBTRACT_FROM_MEAN = -2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    img_bin = cv2.adaptiveThreshold(
        ~image,
        MAX_COLOR_VAL,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        SUBTRACT_FROM_MEAN,
    )

    img_h, img_w = image.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_w * 0.5), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(img_h * 0.7)))
    horizontal_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)
    both = horizontal_lines + vertical_lines
    cleaned = img_bin - both

    # Get rid of little noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    opened = cv2.dilate(opened, kernel)

    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rects = [cv2.boundingRect(c) for c in contours]
    NUM_PX_COMMA = 6
    MIN_CHAR_AREA = 5 * 9
    char_sized_bounding_rects = [(x, y, w, h) for x, y, w, h in bounding_rects if w * h > MIN_CHAR_AREA]
    if char_sized_bounding_rects:
        minx, miny, maxx, maxy = math.inf, math.inf, 0, 0
        for x, y, w, h in char_sized_bounding_rects:
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x + w)
            maxy = max(maxy, y + h)
        x, y, w, h = minx, miny, maxx - minx, maxy - miny
        cropped = image[y:min(img_h, y+h+NUM_PX_COMMA), x:min(img_w, x+w)]
    else:
        # If we morphed out all of the text, assume an empty image.
        cropped = MAX_COLOR_VAL * np.ones(shape=(20, 100), dtype=np.uint8)
    bordered = cv2.copyMakeBorder(cropped, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, 255)
    return bordered


def ocr_image(image, config):
    return pytesseract.image_to_string(
        image,
        config=config
    )


def search_peaks(img, distance, axis):
    # kernel = np.ones((kernel_x, kernel_y), 'uint8')
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # w, h = img.shape[0:2]
    # cv2.imwrite(f'{self.directory}/result3{axis}.png', img)

    if axis == 'row':
        Sum = np.sum(img / 255, axis=1)
    else:
        Sum = np.sum(img / 255, axis=0)

    gaussian_curve = gaussian_filter(Sum, sigma=distance * 0.2)
    gaussian_peaks = argrelextrema(gaussian_curve, np.greater)
    print(gaussian_peaks)
    print("Peaks: ", len(gaussian_peaks[0]))
    plt.plot(Sum)
    plt.plot(gaussian_curve)
    # plt.plot(gaussian_peaks, range(len(gaussian_peaks)), "o")
    plt.savefig("mygraph.png")
    plt.close()
    print(img.shape)
    borders = [0, *gaussian_peaks[0], img.shape[0]]
    print(borders)
    divide_images = []
    for i in range(len(borders) - 1):
        divide_images.append(img[borders[i]:borders[i + 1]])
        print(i, img[borders[i]:borders[i + 1]].shape)
    return divide_images


class Row:
    def __init__(self, start):
        self.start = start
        self.cells = []

    @property
    def height(self):
        counter = Counter(self.heights)
        most_common = counter.most_common()[0][0]
        return most_common

    @property
    def heights(self):
        return [cell.height for cell in self.cells]

    @property
    def fullness(self):
        fullness = [cell.fullness for cell in self.cells]
        print(len(list(chain(*fullness))))
        return list(chain(*fullness))


class Cell:
    def __init__(self, start, end, center):
        self.start = [int(start[0]), int(start[1])]
        self.end = [int(end[0]), int(end[1])]
        self.center = [int(center[0]), int(center[1])]
        self.sub_cells = []

    @property
    def height(self):
        return len(self.sub_cells)

    @property
    def fullness(self):
        return [sub_cell.fullness for sub_cell in self.sub_cells]

    def ocr(self, image):
        cropped = crop_to_text(image)
        cv2.imwrite("static/images/11_cropped.jpg", cropped)
        divide_images = search_peaks(cropped, 50, axis="row")
        print(len(divide_images))

        for divide_image in divide_images:
            cv2.imwrite("static/images/13_cropped.jpg", divide_image)
            sub_cell = SubCell(divide_image)
            # sub_cell.ocr()
            self.sub_cells.append(sub_cell)

    def reocr(self, img_to_cut, height):
        print("Пересчет!", self.start, self.end)
        dif = int((self.end[1] - self.start[1]) / height)
        self.sub_cells = []

        for i in range(height):
            y_start = self.start[1] + i * dif
            y_end = self.start[1] + (i + 1) * dif
            print(y_start, y_end)
            cell_img = img_to_cut[y_start:y_end, self.start[0]:self.end[0]]
            sub_cell = SubCell(cell_img)
            sub_cell.ocr()
            self.sub_cells.append(sub_cell)

    def __repr__(self):
        return self.sub_cells


class SubCell:
    def __init__(self, img):
        self.img = img
        self.fullness = np.sum(np.invert(img, dtype=np.uint8))

    def ocr(self):
        tessdata_dir = "tessdata"
        tess_args = ["--psm", "7", "-l", "table-ocr", "--tessdata-dir", tessdata_dir]
        self.value = ocr_image(self.img, " ".join(tess_args)).strip()

    def __repr__(self):
        return f"{self.value} - {self.fullness}"


full_path = "static/images/p005020210518201354441326.png"
image = Image(full_path)
rows = image.get_image()

workbook = xlsxwriter.Workbook('Expenses01.xlsx')
worksheet = workbook.add_worksheet()

i = 1
j = 1

for row in rows:
    print("row.min_fullness", row.min_fullness)
    j = 1
    for cell in row.cells:
        m = 0
        for sub_cell in cell.sub_cells:
            print("sub_cell.fullness", sub_cell.fullness)
            if sub_cell.fullness > row.min_fullness:
                worksheet.write(i + m, j, sub_cell.value)
            else:
                worksheet.write(i + m, j, 0)
            m += 1
        j += 1
    i += row.height
workbook.close()
