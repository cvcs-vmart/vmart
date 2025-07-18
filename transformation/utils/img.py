import cv2

def show_img(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_img(img, path):
    cv2.imwrite(path, img)