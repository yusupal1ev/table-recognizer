import os
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema

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

    def get_image(self):
        img = cv2.imread(self.path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{self.directory}/gray.png', gray)
        edges = cv2.Canny(gray, 50, 150, apertureSize=5)
        cv2.imwrite(f'{self.directory}/edges.png', edges)
        for_vertical_lines = cv2.HoughLines(image=edges, rho=1, theta=np.pi / 180, threshold=300)
        for_horizontal_lines = cv2.HoughLines(image=edges, rho=1, theta=np.pi / 180, threshold=380)
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
        result = cv2.imread(f'{self.directory}/result.png', cv2.IMREAD_GRAYSCALE)
        result = cv2.bitwise_not(result)
        _, result = cv2.threshold(result, 200, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'{self.directory}/result2.png', result)

        y_points = self.search_peaks(result, 1, int(0.5*min_size_horizontal), min_size_horizontal, axis='row')
        x_points = self.search_peaks(result, int(0.5*min_size_vertical), 1, min_size_vertical, axis='column')

        a = int(min_size_horizontal * 0.3)
        b = int(min_size_vertical * 0.3)
        max_value = 4*a*b
        for i, x in enumerate(x_points[0]):
            for j, y in enumerate(y_points[0]):
                x1 = x - b
                y1 = y - a
                x2 = x + b
                y2 = y + a
                region = result[y1:y2, x1:x2]
                value = np.sum(region / 255)
                print(i, j, value)
                if value > max_value/10:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.imwrite(f'{self.directory}/rectangles.png', img)

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


