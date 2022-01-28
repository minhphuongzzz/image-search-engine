from tensorflow.keras.preprocessing import image
import method.color as color
import method.hog as hog
import method.vgg16 as vgg16
import method.orb as orb
import cv2

# LOAD & RESIZE IMAGE FUNCTION 
def load_image(img_path, target_size):

  img = cv2.imread(img_path)
  if target_size == -1:
    return img

  return cv2.resize(img, target_size)

# EXTRACT IMAGE FUNCTION
def extract_img(img_path, method):

    extractor = None
    
    if method == "color":
        img = load_image(img_path, -1)
        extractor = color.Color()
    elif method == "orb":
        img = load_image(img_path, -1)
        extractor = orb.Orb()
    elif method == "hog":
        img = load_image(img_path, (224, 224))
        extractor = hog.Hog()
    elif method == 'vgg16':
        img = load_image(img_path, (224, 224))
        extractor = vgg16.Vgg16()

    feature = extractor.extract(img)

    return feature
