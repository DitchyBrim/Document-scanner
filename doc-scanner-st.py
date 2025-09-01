import cv2
import numpy as np
import pytesseract
import streamlit as st

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def gray2text(gray_image: np.ndarray, lang: str = 'eng') -> str:
    text = pytesseract.image_to_string(gray_image, lang=lang)
    text_blocks = []
    return {
        'full_text': text,
        'text_blocks': text_blocks,
        'word_count': len(text.split()),
        'average_condfidence': 0,
    }

def scan_document(image: np.ndarray, lang='eng') -> np.ndarray:

    gray = detect_edges(image)
    text = gray2text(gray, lang=lang)
    boxes = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    return text, boxes

st.set_page_config(page_title="Document Scanner & OCR", layout="wide")

st.title("📄 Document Scanner & Text Extraction")
st.markdown("Upload an image of a document to scan and extract text")

uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width='stretch')

    with st.spinner("Processing..."):
        text_data, boxes = scan_document(image)
        text = text_data['full_text']
        img_copy = image.copy()
        h, w, _ = img_copy.shape
        for i in range(len(boxes['text'])):
            if boxes["text"][i].strip() != "":
                x, y, bw, bh = boxes["left"][i], boxes["top"][i], boxes["width"][i], boxes["height"][i]
                cv2.rectangle(img_copy, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                cv2.putText(img_copy, boxes["text"][i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    with col2:
        st.subheader("Scanned Document")
        st.image(img_copy, width='stretch')
    
    st.subheader("Extracted Text")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.text_area("Extracted Text", text, height=300)

    with col2:
        st.metric("Word Count", text_data['word_count'])

        # Download options
        st.subheader("Download")
        
        # Text file download
        st.download_button(
            "Download as TXT",
            text_data['full_text'],
            file_name="extracted_text.txt",
            mime="text/plain"
        )