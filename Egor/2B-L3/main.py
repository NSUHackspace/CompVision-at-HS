import cv2 as cv
import numpy as np

if __name__ == "__main__":
    # Референсные файлы
    src_filename = "car.jpg"
    template_filename = "car_template.jpg"

    # Подгрузка изображений
    src = cv.imread(f"image_base/{src_filename}")
    tmpl = cv.imread(f"image_base/{template_filename}",0)
    
    out = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    ballard = cv.createGeneralizedHoughBallard()
    ballard.setTemplate(tmpl)

    #guil = cv.createGeneralizedHoughGuil()
    #guil.setTemplate(tmpl)

    [b_positions, b_votes] = ballard.detect(out)
    #[g_positions, g_votes] = guil.detect(out)
    #print("Guil true") 

    for position in b_positions:
        for pps in position:
            pos = [pps[0], pps[1]]
            scale = pps[2]
            angle = pps[3]

            rect = cv.RotatedRect(pos, [tmpl.shape[0]*scale, tmpl.shape[0]*scale], angle)

            opts = {'Color': [0, 0, 255], 'Thickness': 2, 'LineType': 'AA'}

            box = cv.boxPoints(rect)
            box = np.int32(box)

            src = cv.drawContours(src, [box], 0, (0, 255, 255), 2)
            

    cv.imshow("result", src)
    cv.imwrite(f"image_result/result_of_{src_filename}", src)
    cv.waitKey(0)