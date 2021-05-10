import os
import cv2
import json
import copy
import math
import numpy as np
import numpy.ma as ma
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema, convolve2d
from skimage.morphology import skeletonize
from skimage.util import invert
from sklearn.cluster import KMeans


from prettytable import PrettyTable
from table_ocr import table_ocr
from cell_ocr import cell_ocr


class Table:
    def __init__(self, path, model):
        self.image = Image(path)
        self.model = model
        self._model_name = self._get_model_name()
        self._model_fields = self._get_model_fields()
        self.rows = [[1, 2, 3, 4], [2, 5, 6, 8]]

    def _get_model_name(self):
        return str(self.model).split(".")[-1][:-2]

    def _get_model_fields(self):
        fields = []
        for k, v in self.model.__dict__.items():
            if not k.endswith('__') and not 'function' in str(type(v)):
                fields.append(k)
        return fields

    def _create_json_fixture(self):
        fixture = []
        for i, row in enumerate(self.rows):
            fields = {}
            for j, cell in enumerate(row):
                fields[self._model_fields[j]] = cell

            fixture.append(
                {
                    "model": self._model_name,
                    "pk": i + 1,
                    "fields": fields
                }
            )
        return fixture

    def write_fixture_to_json(self):
        fixture = self._create_json_fixture()
        with open(f"{fixture[0]['model']}.json", "w") as j:
            json.dump(fixture, j, indent=4)


class Image:
    def __init__(self, path: str):
        self.path = path
        self.directory = os.path.join(*path.split("/")[:-1])
        self.file = path.split("/")[-1]
        self.line_thickness = 12

    def _draw_lines(self, img, lines, min_theta, max_theta):
        final_lines = []
        temp_lines = []

        for i, line in enumerate(lines):
            rho, theta = line
            if min_theta < theta < max_theta:
                if final_lines:
                    if temp_lines:
                        if rho < temp_lines[-1][0] + self.line_thickness:
                            temp_lines.append((rho, theta))
                        else:
                            final_lines.append((sum([i[0] for i in temp_lines])/len(temp_lines), theta))
                            temp_lines = [(rho, theta)]
                    else:
                        if rho < final_lines[-1][0] + self.line_thickness:
                            temp_lines.append(final_lines[-1])
                            temp_lines.append((rho, theta))
                        else:
                            temp_lines.append((rho, theta))
                else:
                    if temp_lines:
                        if rho < temp_lines[-1][0] + self.line_thickness:
                            temp_lines.append((rho, theta))
                        else:
                            final_lines.append((sum([i[0] for i in temp_lines])/len(temp_lines), theta))
                            temp_lines = [(rho, theta)]
                    else:
                        temp_lines.append((rho, theta))
        final_lines.append((sum([i[0] for i in temp_lines]) / len(temp_lines), theta))

        print("final_lines: ", final_lines)
        sizes = []
        for i, line in enumerate(final_lines):
            sizes.append(final_lines[i + 1 - len(final_lines)][0] - line[0])
        print(sizes[:-1])
        min_size = min(sizes[:-1])
        print("Min size: ", min_size)

        for i, line in enumerate(final_lines):
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 - 5000 * (-b))
            y1 = int(y0 - 5000 * a)
            x2 = int(x0 + 5000 * (-b))
            y2 = int(y0 + 5000 * a)

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), self.line_thickness)

        return min_size

    def delete_borders_manual(self, img, gray, vertical_threshold, horizontal_threshold):
        edges = cv2.Canny(gray, 50, 150, apertureSize=5)
        cv2.imwrite(f'{self.directory}/edges.png', edges)
        for_vertical_lines = cv2.HoughLines(image=edges, rho=1, theta=np.pi / 180, threshold=vertical_threshold)
        for_horizontal_lines = cv2.HoughLines(image=edges, rho=1, theta=np.pi / 180, threshold=horizontal_threshold)
        vertical_lines_mask = for_vertical_lines[:, :, 1] < 0.01
        horizontal_lines_mask = for_horizontal_lines[:, :, 1] > 1.56

        vertical_lines = np.sort(for_vertical_lines[vertical_lines_mask], axis=0)
        horizontal_lines = np.sort(for_horizontal_lines[horizontal_lines_mask], axis=0)

        print(f"Horizontal lines number: {len(horizontal_lines)}")
        print(f"Horizontal lines: {horizontal_lines}")
        min_size_horizontal = self._draw_lines(img, horizontal_lines, 1.56, 1.58)

        print(f"Vertical lines number: {len(vertical_lines)}")
        print(f"Vertical lines: {vertical_lines}")
        min_size_vertical = self._draw_lines(img, vertical_lines, -0.01, 0.01)

        cv2.imwrite(f'{self.directory}/result.png', img)
        return min_size_horizontal, min_size_vertical

    def delete_border_auto(self, gray, input_img):
        kernel = np.ones((5, 5), 'uint8')  # Ядро 5х5 для размытия
        eroded = cv2.erode(gray, kernel, iterations=1)
        cv2.imwrite(f'{self.directory}/02_eroded.png', eroded)
        _, threshold_img = cv2.threshold(eroded, 240, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'{self.directory}/03_threshold.png', threshold_img)
        invert_img = invert(threshold_img / 255)
        cv2.imwrite(f'{self.directory}/04_invert_img.png', invert_img * 255)
        skeleton = skeletonize(invert_img)
        skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{self.directory}/05_skeleton.png', skeleton)
        skeleton_erode_kernel = np.ones((5, 5), 'uint8')
        skeleton_eroded = cv2.dilate(skeleton, skeleton_erode_kernel, iterations=1)
        cv2.imwrite(f'{self.directory}/06_skeleton_eroded.png', skeleton_eroded)
        # skeleton_gray = cv2.imread(f'{self.directory}/skeleton_eroded.png', cv2.IMREAD_GRAYSCALE)

        _, skeleton_threshold = cv2.threshold(skeleton_eroded, 5, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'{self.directory}/07_skeleton_threshold.png', skeleton_threshold)
        lines = cv2.HoughLinesP(skeleton_threshold, 1, np.pi/180, 100, minLineLength=100, maxLineGap=5)
        self._draw_lines_p(input_img, lines)
        cv2.imwrite(f'{self.directory}/08_red_lines.png', input_img)

        empty_img = np.zeros(input_img.shape[0:2], np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x2 = x2 + 1
            y2 = y2 + 1
            # print(x1, y1, x2, y2)
            empty_img[y1:y2, x1:x2] = 255
            empty_img[(y2 - 1):y1, x1:x2] = 255
        # empty_img = cv2.dilate(empty_img, np.ones((1, 1), 'uint8'))
        _, empty_threshold = cv2.threshold(empty_img, 10, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'{self.directory}/09_empty_threshold.png', empty_threshold)

        contours, hierarchy = cv2.findContours(empty_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        cv2.drawContours(gray, contours, 1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 2)
        cells = []
        for contour in contours[-1:0:-1]:
            for point in contour:
                point = tuple(point[0])
                cv2.drawMarker(gray, point, (0, 255, 0), 1, 10)
            sort_points = sorted(contour, key=lambda m: m[0][0] + m[0][1])
            point_1 = tuple(sort_points[0][0])
            point_2 = tuple(sort_points[-1][0])
            center = (int((point_2[0] + point_1[0]) / 2), int((point_2[1] + point_1[1]) / 2))
            cells.append((point_1, point_2, center))
            cv2.drawMarker(gray, point_1, (0, 255, 0), 4, 20)
            cv2.drawMarker(gray, point_2, (0, 255, 0), 4, 20)
            cv2.drawMarker(gray, center, (0, 0, 255), 6, 20)

        cv2.imwrite(f'{self.directory}/10_img_markers.png', gray)

        cells.sort(key=lambda x: (x[0][0], x[0][1]))
        np_cells = np.array(cells, np.uint64)
        print("cells")
        for cell in cells:
            print(cell)
        np_cells1 = np.concatenate((np_cells[:, 0, 0], np_cells[:, 1, 0]))
        np_cells1 = np.sort(np_cells1)
        print("np_cells1")
        print(np_cells1)

        cells_cluster = self.cluster_number(np_cells1, 15)
        print("cells_cluster")
        print(cells_cluster)

        arr = np_cells[:, 0:2, 0]
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
        np_cells[:, 0:2, 0] = centroids[indices]
        print("Конечный массив: ")
        for cell in np_cells:
            print(cell[0], cell[1], cell[2])





        np_cells2 = np.concatenate((np_cells[:, 0, 1], np_cells[:, 1, 1]))
        np_cells2 = np.sort(np_cells2)
        print("np_cells2")
        print(np_cells2)

        cells_cluster2 = self.cluster_number(np_cells2, 15)
        print("cells_cluster2")
        print(cells_cluster2)

        arr2 = np_cells[:, 0:2, 1]
        print("Исходный массив: ", arr2)
        centroids = sorted(list(set(cells_cluster2)))
        print(centroids)

        tresholds = []
        for i, num in enumerate(centroids):
            tresholds.append(math.ceil((centroids[i + 1 - len(centroids)] + num) / 2))
        tresholds.pop()
        print("Пороговые значения2: ", tresholds)
        centroids = np.array(centroids, np.uint64)

        indices = np.searchsorted(tresholds, arr2)
        np_cells[:, 0:2, 1] = centroids[indices]
        print("Конечный массив2: ")
        for cell in np_cells:
            print(cell[0], cell[1], cell[2])






        print("lexsort", np.lexsort((np_cells[:, 0, 1], np_cells[:, 0, 0])))

        cells_3 = np_cells[np.lexsort((np_cells[:, 0, 0], np_cells[:, 0, 1]))]
        print("Sorted: ")
        for cell in cells_3:
            print(cell[0], cell[1], cell[2])

        # cluster_cells = self.cluster(np_cells, 15)
        # for cell in cluster_cells:
        #     print(cell[0], cell[1], cell[2])
        #
        # indexis = np.argsort(cluster_cells[:, 0, 1])
        # cluster_cells_sort = cluster_cells[indexis, :]
        # print("Sort_cluster")
        # for cell in cluster_cells_sort:
        #     print(cell[0], cell[1], cell[2])
        #
        # cluster_cells_sort[:, 0, [0, 1]] = cluster_cells_sort[:, 0, [1, 0]]
        # print("Replace_cluster")
        # for cell in cluster_cells_sort:
        #     print(cell[0], cell[1], cell[2])
        #
        # cluster_cells_sort2 = self.cluster(cluster_cells_sort, 15)
        # cluster_cells_sort2[:, 0, [0, 1]] = cluster_cells_sort2[:, 0, [1, 0]]
        #
        # print("Cluster_y")
        # for cell in cluster_cells_sort2:
        #     print(cell[0], cell[1], cell[2])
        #
        # indexis = np.argsort(cluster_cells_sort2[:, 0, 0])
        # cluster_cells_sort3 = cluster_cells_sort2[indexis, :]
        #
        # print("Cluster_xy")
        # for cell in cluster_cells_sort3:
        #     print(cell[0], cell[1], cell[2])
        #
        # indexis = np.argsort(cluster_cells_sort2[:, 1, 0])
        # cluster_cells_sort4 = cluster_cells_sort2[indexis, :]
        #
        # print("Sort by p2")
        # for cell in cluster_cells_sort4:
        #     print(cell[0], cell[1], cell[2])
        #
        # cluster_cells_sort4[:, [0, 1, 2]] = cluster_cells_sort4[:, [1, 0, 2]]
        # print("Sort by p2 replace")
        # for cell in cluster_cells_sort4:
        #     print(cell[0], cell[1], cell[2])
        #
        # cluster_cells_sort5 = self.cluster(cluster_cells_sort4, 15)
        # cluster_cells_sort5[:, [0, 1, 2]] = cluster_cells_sort5[:, [1, 0, 2]]
        # print("Sort by p2 sorted")
        # for cell in cluster_cells_sort5:
        #     print(cell[0], cell[1], cell[2])
        #
        # cv2.imwrite(f'{self.directory}/img35.png', gray)

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

    @staticmethod
    def cluster(coords, lim) -> np.ndarray:
        print("Clusterizing...")
        final_coords = copy.copy(coords)
        temp_coords = copy.copy(coords)
        temp = []
        for i, coord in enumerate(coords):
            p1, p2, c = coord
            if i > 0:
                if coord[0][0] - coords[i - 1][0][0] > lim:
                    np.place(final_coords[:, 0, 0], temp_coords[:, 0, 0] < 1, int(sum(temp) / len(temp)))
                    temp_coords = copy.copy(coords)
                    temp_coords[i] = ((0, coord[0][1]), coord[1], coord[2])
                    temp = [coord[0][0], ]
                else:
                    temp_coords[i] = ((0, coord[0][1]), coord[1], coord[2])
                    temp.append(coord[0][0])
            else:
                temp_coords[i] = ((0, coord[0][1]), coord[1], coord[2])
                temp.append(coord[0][0])
        np.place(final_coords[:, 0], temp_coords[:, 0] < 1, int(sum(temp) / len(temp)))
        return final_coords

    @staticmethod
    def cluster_old(coords, lim, axis=0):
        if axis == 1:
            coords = coords[:, [1, 0]]
        print("Clusterizing...")
        final_coord = copy.copy(coords)
        temp_coord = copy.copy(coords)
        temp = []
        for i, coord in enumerate(coords):
            x, y = coord
            if i > 0:
                if x - coords[i - 1][0] > lim:
                    np.place(final_coord[:, 0], temp_coord[:, 0] < 1, int(sum(temp) / len(temp)))
                    temp_coord = copy.copy(coords)
                    temp_coord[i] = (0, y)
                    temp = [x, ]
                else:
                    temp_coord[i] = (0, y)
                    temp.append(x)
            else:
                temp_coord[i] = (0, y)
                temp.append(x)
        np.place(final_coord[:, 0], temp_coord[:, 0] < 1, int(sum(temp) / len(temp)))
        if axis == 1:
            final_coord = final_coord[:, [1, 0]]
        return final_coord

    def _draw_lines_p(self, img2, lines):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 5)

    def get_image(self):
        img = cv2.imread(self.path)
        img2 = img[:]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{self.directory}/01_gray.png', gray)

        # min_size_horizontal, min_size_vertical = self.delete_borders_manual(img, gray, 300, 380)
        self.delete_border_auto(img, img2)

        # result = cv2.imread(f'{self.directory}/result.png', cv2.IMREAD_GRAYSCALE)
        # result = cv2.bitwise_not(result)
        # _, result = cv2.threshold(result, 200, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(f'{self.directory}/result2.png', result)
        #
        # y_points = self.search_peaks(result, 1, int(0.5*min_size_horizontal), min_size_horizontal, axis='row')
        # x_points = self.search_peaks(result, int(0.5*min_size_vertical), 1, min_size_vertical, axis='column')
        #
        # a = int(min_size_horizontal * 0.3)
        # b = int(min_size_vertical * 0.3)
        # max_value = 4*a*b
        # for i, x in enumerate(x_points[0]):
        #     for j, y in enumerate(y_points[0]):
        #         x1 = x - b
        #         y1 = y - a
        #         x2 = x + b
        #         y2 = y + a
        #         region = result[y1:y2, x1:x2]
        #         value = np.sum(region / 255)
        #         # print(i, j, value)
        #         if value > max_value/10:
        #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        #
        # cv2.imwrite(f'{self.directory}/rectangles.png', img)

    def search_peaks(self, img, kernel_x, kernel_y, distance, axis):
        # kernel = np.ones((kernel_x, kernel_y), 'uint8')
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        # w, h = img.shape[0:2]
        # cv2.imwrite(f'{self.directory}/result3{axis}.png', img)

        if axis == 'row':
            Sum = np.sum(img / 255, axis=1)
        else:
            Sum = np.sum(img / 255, axis=0)

        gaussian_curve = gaussian_filter(Sum, sigma=distance*0.2)
        gaussian_peaks = argrelextrema(gaussian_curve, np.greater)
        plt.plot(Sum)
        plt.plot(gaussian_curve)
        plt.plot(gaussian_peaks, range(len(gaussian_peaks)), "o")
        plt.show()
        print(gaussian_peaks)
        print("Peaks: ", len(gaussian_peaks[0]))
        return gaussian_peaks


if __name__ == '__main__':
    image = Image("images/p0050.png")
    arg = image.get_image()
    # table = table_ocr(arg)
    # result_table = {}
    # index = 1

    # for i, row in enumerate(table):
    #     if i in [0, 1]:
    #         continue
    #     for j, cell in enumerate(row):
    #         if j in [0, ]:
    #             continue
    #         value = cell_ocr(cell, None).strip()
    #         result_table.setdefault(index, []).append(value)
    #     index += 1
    #
    # print(result_table)
    # prettytable = PrettyTable(['1', '2', '3', '4', '5', '6', '7', '8'])
    # for key, value in result_table.items():
    #     prettytable.add_row(value)
    #
    # print(prettytable)

