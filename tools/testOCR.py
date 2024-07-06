import easyocr
import cv2
reader = easyocr.Reader(lang_list=['en'], gpu = True, detector=False)
img = cv2.imread("assets/testtext.png")
print(reader.recognize(img))