import os
from .curl_inference import curl 
from enlighten_inference import EnlightenOnnxModel
import cv2



def enhance() :
    enhance_curl()
    enhance_engan()
    # enhance_mirnet()

def enhance_curl() :
    curl("./media/images", "./media/images_y")

def enhance_engan() :
    directory = './media/images'
 
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            img = cv2.imread(f)
            model = EnlightenOnnxModel()

            processed = model.predict(img)
            cv2.imwrite('./media/images_x/' + filename, processed)
            # os.unlink(path + filename)