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
# Binarization of the img
def binarization(img):
    maxi = float(img.max())
    mini = float(img.min())
    x = maxi - ((maxi - mini) / 2)
    ret, img_binary = cv.threshold(img, x, 255, cv.THRESH_BINARY)
    return img_binary
#To detect the edge of the license plate
def canny(img):
    img_canny = cv.Canny(img, img.shape[0], img.shape[1])
    return img_canny

# locate the license plate by opening and closing calculation of open cv
def opening_closing(img):
    kernel = np.ones((5, 23), np.uint8)
    img_closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    cv.imshow("Closing", img_closing)
    img_opening1 = cv.morphologyEx(img_closing, cv.MORPH_OPEN, kernel)
    cv.imshow("Opening_1", img_opening1)
    kernel = np.ones((11, 6), np.uint8)
    img_opening2 = cv.morphologyEx(img_opening1, cv.MORPH_OPEN, kernel)
    return img_opening2

# Find the rectangular box of the license plate
def find_rectangle(contour):
    y, x = [], []
    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])
    return [min(y), min(x), max(y), max(x)]
# Resize the image and set the scale to 400
def resize_img(img, max_size):
    h, w = img.shape[0:2]
    scale = max_size / max(h, w)
    img_resized = cv.resize(img, None, fx=scale, fy=scale,
                            interpolation=cv.INTER_CUBIC)
    # print(img_resized.shape)
    return img_resized

# split the license plate out for a single picture
def locate_license(original, img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_cont = original.copy()
    img_cont = cv.drawContours(img_cont, contours, -1, (255, 0, 0), 6)
    cv.imshow("Contours", img_cont)
    block = []
    for c in contours:
        r = find_rectangle(c)
        a = (r[2] - r[0]) * (r[3] - r[1])
        s = (r[2] - r[0]) / (r[3] - r[1])
        block.append([r, a, s])
    block = sorted(block, key=lambda bl: bl[1])[-5:]
    maxweight, maxindex = 0, -1
    for i in range(len(block)):
        print('block', block[i])
        if 2 <= block[i][2] <= 4 and 1000 <= block[i][1] <= 20000:  # 对矩形区域高宽比及面积进行限制
            b = original[block[i][0][1]: block[i][0][3], block[i][0][0]: block[i][0][2]]
            hsv = cv.cvtColor(b, cv.COLOR_BGR2HSV)
            lower = np.array([100, 50, 50])
            upper = np.array([140, 255, 255])
            mask = cv.inRange(hsv, lower, upper)
            w1 = 0
            for m in mask:
                w1 += m / 255
            w2 = 0
            for n in w1:
                w2 += n
            if w2 > maxweight:
                maxindex = i
                maxweight = w2

    rect = block[maxindex][0]
    return rect

#Stretching the image
#To strenthen the contrast ratio of the image
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

def cut_license(original, rect):
    license_img = original[rect[1]:rect[3], rect[0]:rect[2]]
    return license_img


def find_waves(threshold, histogram):
    up_point = -1
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def remove_upanddown_border(img):
    plate_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, plate_binary_img = cv.threshold(plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    row_histogram = np.sum(plate_binary_img, axis=1)  # 数组的每一行求和
    row_min = np.min(row_histogram)
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)
    wave_span = 0.0
    selected_wave = []
    for wave_peak in wave_peaks:
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
    white, black = [], []
    height, width = thresh.shape
    white_max = 0
    black_max = 0
    for i in range(width):
        line_white = 0
        line_black = 0
        for j in range(height):
            if thresh[j][i] == 255:
                line_white += 1
            if thresh[j][i] == 0:
                line_black += 1
        white_max = max(white_max, line_white)
        black_max = max(black_max, line_black)
        white.append(line_white)
        black.append(line_black)
        # print('white_max', white_max)
        # print('black_max', black_max)
    arg = True
    if black_max < white_max:
        arg = False
    n = 1
    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):  # 这点没有理解透彻
            start = n
            end = find_end(start, arg, black, white, width, black_max, white_max)
            n = end
            if end - start > 5 or end > (width * 3 / 7):
                cropImg = thresh[0:height, start - 1:end + 1]
                cropImg = cv.resize(cropImg, (34, 56))
                cv.imwrite(save_path + '\\{}.bmp'.format(n), cropImg)
                cv.imshow('Char_{}'.format(n), cropImg)
def main():
    image = cv.imread(img_path)
    rect, img_resized = preprocessing(image)
    license_img = cut_license(img_resized, rect)
    cv.imshow('License', license_img)
    plate_b_img = remove_upanddown_border(license_img)
    cv.imshow('plate_binary', plate_b_img)
    char_segmentation(plate_b_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()