import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    src_filename = "krosh.png"

    src = cv.imread(f"image_base/{src_filename}", 0)

    #cv.imshow("help", src)

    rows = src.shape[0]
    circles = cv.HoughCircles(src, cv.HOUGH_GRADIENT, 1, rows/32, 
                              Guilparam1=70, param2=30, minRadius=50, maxRadius=100)


    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # Рисуем центр круга
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # Рисуем границы круга
            radius = i[2]
            cv.circle(src, center, radius, (0, 0, 255), 3)
    else:
        print("Круги на изображении не обнаружены")

    cv.imshow("detected circles", src)
    cv.imwrite(f"image_result/result_of_{src_filename}", src)
    cv.waitKey(0)