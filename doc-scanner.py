import cv2
import pytesseract
from PIL import Image

# tell pytesseract where your tesseract executable is located (if it's not in your PATH)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load an image (replace with your image path)
image_path = "sample.png"
img = cv2.imread(image_path)

# preprocess the image (convert to grayscale for better accuracy)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# OCR extract text, assuming English language
text = pytesseract.image_to_string(gray, lang='eng')

# Optional, get bounding box estimates
h, w, _ = img.shape
boxes = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

# to view detected text over bounding boxes
for i in range(len(boxes["text"])):
    if boxes["text"][i].strip() != "":
        x, y, bw, bh = boxes["left"][i], boxes["top"][i], boxes["width"][i], boxes["height"][i]
        cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
        cv2.putText(img, boxes["text"][i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
# Display the image with bounding boxes
cv2.imshow("OCR Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows() 