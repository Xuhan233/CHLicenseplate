import cv2 as cv
import numpy as np

#change the path for reading and saving image
img_path = r'.\example1.jpg'
save_path = r'.\test'
# remove the noise of image deduct the pixel value in the image which cut the similarly part in this image
def absdiff(img):
    r = 15
    h = w = r * 2 + 1
    kernel = np.zeros((h, w), np.uint8)
    cv.circle(kernel, (r, r), r, 1, -1)
    img_opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img_absdiff = cv.absdiff(img, img_opening)
    cv.imshow("Opening", img_opening)
    return img_absdiff

# Binarization of the img the binarization is used to specify the color into black and white
# To change the image into black and white can remove the large area of same part in the image
def binarization(img):
    max = float(img.max())
    min = float(img.min())
    x = max - ((max - min) / 2)
    ret, img_binary = cv.threshold(img, x, 255, cv.THRESH_BINARY)
    return img_binary

#To detect the edge of each same area. The edge of the lincense plate must be included.

def canny(img):
    img_canny = cv.Canny(img, img.shape[0], img.shape[1])
    return img_canny

# locate the license plate by opening and closing calculation of open cv The opening and closing calculation is
# remove the small bright area which could remove the noise edge and area for reecongintion of license plate.
def opening_closing(img):
    kernal2 = np.ones((5,27),np.float64)
    kernel = np.ones((5, 23), np.uint8)
    img_closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    cv.imshow("Closing", img_closing)
    kernal2 = kernel
    img_opening1 = cv.morphologyEx(img_closing, cv.MORPH_OPEN, kernel)
    cv.imshow("Opening_1", img_opening1)
    kernel = np.ones((11, 6), np.uint8)
    img_opening2 = cv.morphologyEx(img_opening1, cv.MORPH_OPEN, kernel)
    return img_opening2

# Find the rectangular box of the license plate
def find_rectangle(contour):
    wedith, length = [], []
    # to find the shape of area with a rectangle
    for p in contour:
        wedith.append(p[0][0])
        length.append(p[0][1])
    out = [min(wedith), min(length), max(wedith), max(length)]
    return out

# Resize the image into fixed shape and pixel since the testing image might be different pixel and shape.
# Resize the image and set the scale to 400 since 400 is the best pixel that openCV recommended.
def resize_img(img, max_size):
    height, wedigh = img.shape[0:2]
    scale = max_size / max(height, wedigh)
    # print(cv.resize(img, None, fx=scale, fy=scale,
    #                             interpolation=cv.INTER_CUBIC))
    return cv.resize(img, None, fx=scale, fy=scale,
                            interpolation=cv.INTER_CUBIC)




# split the license plate out for a single picture
def locate_license(original, img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_contours = original.copy()
    img_contours = cv.drawContours(img_contours, contours, -1, (255, 0, 0), 6)
    cv.imshow("Contours", img_contours)
    # To calculate out the area of the image
    area_locked = []
    for im in contours:
        ractangle = find_rectangle(im)
        # To calculate out the ratio between length and width of the rectangle.
        ratio = (ractangle[2] - ractangle[0]) / (ractangle[3] - ractangle[1])
        # To find out the location of upper left conner and lower right corner to calculate out the area.
        area = (ractangle[2] - ractangle[0]) * (ractangle[3] - ractangle[1]) # calculate out the area.
        area_locked.append([ractangle, area, ratio])
    # use the sorting algorithm to sort out the largest five area.
    area_locked = sorted(area_locked, key=lambda bl: bl[1])[-5:]
    # select the area by color since the chinese license plate was in blue/ yellow color.
    # If want to recognize the yellow license plate change the blue index to yellow index.
    max_w, max_i = 0, -1
    for i in range(len(area_locked)):
        print('block', area_locked[i])
        # To limit the length and width's ratio.
        if 2 <= area_locked[i][2] <= 4 and 1000 <= area_locked[i][1] <= 20000:
            # to use thee threshold value to select the best area that most like the license plate out.
            area = original[area_locked[i][0][1]: area_locked[i][0][3], area_locked[i][0][0]: area_locked[i][0][2]]
            hsv = cv.cvtColor(area, cv.COLOR_BGR2HSV)
            lower = np.array([100, 50, 50])
            mid = np.array([120,150,150])
            upper = np.array([140, 255, 255])
            mask = cv.inRange(hsv, lower, upper)
            w1 = 0
            for m in mask:
                w1 += m / 255
            w2 = 0
            for n in w1:
                w2 += n
            if w2 > max_w:
                max_i = i
                max_w = w2

    rect = area_locked[max_i][0]
    return rect

#Stretching the image Strenting is aimed at  strenthen the contrast ratio of the image
def stretching(img):
    maxi = float(img.max())
    mini = float(img.min())
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = 255 / (maxi - mini) * img[i, j] - (255 * mini) / (maxi - mini)
    img_stretched = img
    return img_stretched
#Change the image to gray for use of split the character out
def preprocessing(img):
    img_resized = resize_img(img, 400)
    cv.imshow('Original', img_resized)
    img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', img_gray)
    # uncomment of gaussian
    # img_gaussian = cv.GaussianBlur(img_gray, (3,3), 0)
    # cv.imshow("Gaussian_Blur", img_gaussian)
    img_stretched = stretching(img_gray)
    cv.imshow('Stretching', img_stretched)
    img_absdiff = absdiff(img_stretched)
    cv.imshow("Absdiff", img_absdiff)
    img_binary = binarization(img_absdiff)
    cv.imshow('Binarization', img_binary)
    img_canny = canny(img_binary)
    cv.imshow("Canny", img_canny)
    img_opening2 = opening_closing(img_canny)
    cv.imshow("Opening_2", img_opening2)
    rect = locate_license(img_resized, img_opening2)
    print("rect:", rect)
    # make the license plate image a little bigger to avoid cut part of the character out
    rect[0] = rect[0]-5;
    rect[1] = rect[1] - 5;
    rect[2] = rect[2]+5;
    rect[3] = rect[3] + 5;
    img_copy = img_resized.copy()
    cv.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    cv.imshow('License', img_copy)
    return rect, img_resized

# To cut the plate out by the coordinate.
def cut_license(original, rect):
    return original[rect[1]:rect[3], rect[0]:rect[2]]

# find_waves function was used to remove all noise like screw of the license plate.
def find_waves(threshold, histogram):
    is_peak = False
    up = -1
    is_threshold = False
    if histogram[0] > threshold:
        up = 0
        is_peak = True
    wave_peaks = []
    for index, index1 in enumerate(histogram):
        if is_peak and index1 < threshold:
            if index - up > 2:
                is_peak = False
                is_threshold = False
                wave_peaks.append((up, index))
        elif not is_peak and index1 >= threshold:
            is_peak = True
            is_threshold = True
            up = index
    if is_peak and up != -1 and index - up > 4:
        wave_peaks.append((up, index))
    return wave_peaks

# To remove the border of the license plate like the license plate frame.
def remove_upanddown_border(img):
    plate_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, plate_binary_img = cv.threshold(plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    histogram = np.sum(plate_binary_img, axis=1)
    min_row = np.min(histogram)
    average = np.sum(histogram) / plate_binary_img.shape[0]
    th = (min_row + average) / 2
    peak = find_waves(th, histogram)
    wave_span = 0.0
    selected_wave = []
    for wave_peak in peak:
        span = wave_peak[1] - wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    return plate_binary_img

def find_end(start, arg, black, white, width, black_max, white_max):
    end = start + 1
    for m in range(start + 1, width - 1):
        if (black[m] if arg else white[m]) > (0.95 * black_max if arg else 0.95 * white_max):
            end = m
            break
    return end

# Separate the character out
def char_segmentation(thresh):
    max_w = 0
    max_b = 0
    partofwhite, partofblack = [], []
    height, width = thresh.shape

    for i in range(width):
        countwhite = 0 #
        countblack = 0
        for j in range(height):
            if thresh[j][i] == 255:
                countwhite += 1
            if thresh[j][i] == 0:
                countblack += 1
        max_w = max(max_w, countwhite)
        max_b = max(max_b, countblack)
        partofwhite.append(countwhite)
        partofblack.append(countblack)
    # arg true means it was black background with white character
    # are false means it was white background with black character
    arg = True
    if max_b < max_w:
        arg = False
    # split the character out.
    n = 1
    while n < width - 2:
        n += 1
        if (partofwhite[n] if arg else partofblack[n]) > (0.05 * max_w if arg else 0.05 * max_b):
            start = n
            end = find_end(start, arg, partofblack, partofwhite, width, max_b, max_w)
            n = end
            if end - start > 5 or end > (width * 3 / 7):
                cropImg = thresh[0:height, start - 1:end + 1]
                cropImg = cv.resize(cropImg, (34, 56))
                cv.imwrite(save_path + '\\{}.bmp'.format(n), cropImg)
                cv.imshow('Char_{}'.format(n), cropImg)
def main():
    original = cv.imread(img_path)
    rectagular, img_resized = preprocessing(original)
    plate_img = cut_license(img_resized, rectagular)
    cv.imshow('Plate_cutted', plate_img)
    character_img = remove_upanddown_border(plate_img)
    cv.imshow('plate_in_binary', character_img)
    char_segmentation(character_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()