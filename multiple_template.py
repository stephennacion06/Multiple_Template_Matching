from datetime import datetime
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

test_img_path = 'test_image/*.png'
test_img_list = glob.glob(test_img_path)

template_img_path = 'template_image/*.png'
template_img_list = glob.glob(template_img_path)
name_count = 0


for test_img in test_img_list:
    img_rgb = cv2.imread(test_img)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    print(test_img_list)

    for template_img in template_img_list:
        try:
            template = cv2.imread(template_img, 0)
            w, h = template.shape[::-1]

            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        except:
            print("error on", test_img, template_img)
        name_count += 1

    cv2.imwrite('output_image/{}_{}.png'.format(datetime.now().strftime('%d-%m-%Y-%H-%M'), name_count), img_rgb)
