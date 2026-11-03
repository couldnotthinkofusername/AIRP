import hashlib
import io
import re
import cv2
import fitz
import numpy as np
import pdfplumber
import streamlit as st
import torch
import torchvision
from paddleocr import PaddleOCR
from docx import Document
from PIL import Image, ImageDraw
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights)
from torchvision.transforms.functional import to_tensor
from ultralytics import YOLO






                                    # UI

st.set_page_config(
    page_title = "AIRP",
    layout = "centered",
    initial_sidebar_state = "auto"
)

st.title("AIRP")
st.markdown("Autonomous Information Redaction Privacy")

file = st.file_uploader(
    label = "Upload",
    type = [ "pdf", "docx", "jpg", "png", "jpeg"],
    accept_multiple_files = False,
)

device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

YOLO_Model = YOLO('yolo26n.pt')

patterns = [re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.IGNORECASE),
            re.compile(r"\+?\d[\d\s().-]{7,}\d")]




                                 # Image Preprocessing

def resized(image_file):
    return image_file


def gamma_scale(resized, gamma = 1.0):
    gamma_table = np.array([((x / 255.0) ** gamma) * 255 for x in range(256)]).astype(np.uint8)
    
    return cv2.LUT(resized, gamma_table)


def Denoised(resized):
    return cv2.fastNlMeansDenoisingColored(resized, None, 10, 10, 7, 21)


                                         # Backend

def Imagefiles(file):
    
    file.seek(0)

    image_file = Image.open(file).convert("RGB")

    image_file = cv2.cvtColor(
        np.array(image_file),
        cv2.COLOR_RGB2BGR,
    )

    image_file = resized(image_file)
    image_file = gamma_scale(image_file)
    image_file = Denoised(image_file)

    image_file = Image.fromarray(
        cv2.cvtColor(
            image_file, 
            cv2.COLOR_BGR2RGB,
            )
        )
    

    return image_file






                         # Targeting Object regions using Faster R-CNN

@st.cache_resource
def Target_Region():
    weights = (FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    model_detect = (torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = weights))

    model_detect.eval()
    model_detect.to(device)

    return model_detect


def detecting_regions(image_file):



    model_detect = Target_Region()

    image_tensor = to_tensor(image_file.convert("RGB")).to(device)

    with torch.no_grad():
        detect_region = model_detect([image_tensor])[0]

    return [[int(value) for value in box.tolist()]
            for box, label, score in zip(
            detect_region['boxes'],
            detect_region['labels'],
            detect_region['scores'],
        )   if score.item() >= 0.70]



def detecting_text_regions(image_file):

    model = yolo_model()

    image_array = np.array(
        image_file.convert("RGB")
    )

    results = model.predict(
        source = image_array,
        conf = 0.40,
        verbose = False,
    )

    text_boxes = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            coordinates = box.xyxy[0].tolist()

            text_boxes.append(
                [
                    int(coordinates[0]),
                    int(coordinates[1]),
                    int(coordinates[2]),
                    int(coordinates[3]),
                ]
            )

    return text_boxes




                         # Using Calamari OCR for detecting text in Images


@st.cache_resource
def yolo_model():
    return YOLO_Model


@st.cache_resource
def paddle_ocr():

    return PaddleOCR(use_doc_orientation_classify = False,
                     use_doc_unwarping = False,
                     use_textline_orientation = False,
                     lang = 'en'
                     )



def detecting_text_lines(image_file):

    image_array = np.array(image_file.convert('RGB'))

    grayscaling = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    binary = cv2.adaptiveThreshold(
        grayscaling,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

    connected_text = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        horizontal_kernel
    )

    contours, _ = cv2.findContours(
        connected_text,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    text_boxes = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if width < 30:
            continue

        if height < 10:
            continue

        if height > 120:
            continue

        text_boxes.append([x, y, x + width, y + height])

    text_boxes.sort(key = lambda box: (box[1], box[0]))

    return text_boxes



def paddle_prediction(image_file, box):
    
    model = paddle_ocr()

    x1, y1, x2, y2 = box

    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image_file.width, x2)
    y2 = min(image_file.height, y2)

    if x2 <= x1 or y2 <= y1:

        return ''

    line_image = image_file.crop((x1, y1, x2, y2)).convert('RGB')


    line_array = cv2.cvtColor(np.array(line_image), cv2.COLOR_RGB2BGR)

    try: 
        recognised_text = []
    

        for result in model.predict(line_array):

            recognised_text.extend(result.get('rec_texts', []))

        return ''.join(recognised_text)

    except Exception as error:

        raise RuntimeError(f'OCR failed. Redaction stopped: {error}') from error 



def is_sensitive_text(text):

    if not text:
        
        return False

    return any(pattern.search(text) for pattern in patterns)





                            # Apply Redaction techniques using OpenCV + PIL

def Redaction_models(image_file):

    def Mask(image_file, box):

        mask = image_file.copy()

        ImageDraw.Draw(mask).rectangle(box, fill = "black")

        return mask



    def BlackBox(image_file, box):
        
        return Mask(image_file, box)



    def Pixelate(image_file, box):
        
        image_arr = np.array(image_file).copy()

        x1, y1, x2, y2 = box

        ROI = image_arr[y1:y2, x1:x2]

        if ROI.size:
            h, w = ROI.shape[:2]

            reshape = cv2.resize(
                ROI,
                (max(1, w // 12),
                 max(1, h // 12)
                )
            )

            ROI = cv2.resize(
                reshape,
                (w, h),
                interpolation = cv2.INTER_NEAREST
            )

            image_arr[y1:y2, x1:x2] = ROI


        return Image.fromarray(image_arr)


    def Blurring(image_file, box):
        image_arr = np.array(image_file).copy()

        x1, y1, x2, y2 = box

        ROI = image_arr[y1:y2, x1:x2]

        if ROI.size:

            blur = cv2.GaussianBlur(
                ROI,
                (61, 61),
                0
                )
            
            image_arr[y1:y2, x1:x2] = blur

        return Image.fromarray(image_arr)

    return Mask, Pixelate, BlackBox, Blurring






                            # Selection of Redaction techniques

def Choosing_Redaction(Mask, BlackBox, Pixelate, Blurring, image_file, extension):
    
    if extension in ["docx", "pdf"]:
        
        return "Hashing"

    else:
        return "Black_Box"




                                         # Applying redaction

def Apply_Redaction(image_file, boxes, method):

    Mask, Pixelate, BlackBox, Blurring = Redaction_models(image_file)

    output = image_file.copy()

    for box in boxes:
        if method == "Blurring":
            output = Blurring(output, box)

        elif method == "Black_Box":
            output = BlackBox(output, box)

        elif method == "Pixelation":
            output = Pixelate(output, box)

        elif method == "Masking":
            output = Mask(output, box)
    

    return output




                             # Verifying if sensitive data remains

def Veryfying_Image_Redaction(image_file, max_iteration = 5):
    
    output = image_file.copy()

    for i in range(max_iteration):

        text_boxes = detecting_text_lines(output)

        sensitive_boxes = []

        for box in text_boxes:
            
            text = paddle_prediction(output, box)

            if is_sensitive_text(text):
                sensitive_boxes.append(box)

        if not sensitive_boxes:
            
            return output

        output = Apply_Redaction(output, sensitive_boxes, "Black_Box")

    return output


def Image_Redaction(image_file):

    boxes = detecting_regions(image_file)

    text_boxes = detecting_text_regions(image_file)

    sensitive_boxes = []

    for box in text_boxes:

        text = paddle_prediction(image_file, box)

        if is_sensitive_text(text):

            sensitive_boxes.append(box)

    redact_boxes = boxes + sensitive_boxes

    if not redact_boxes:

        return image_file


    redaction = Apply_Redaction(image_file, redact_boxes, "Black_Box")

    return Veryfying_Image_Redaction(redaction, max_iteration = 5)





                                          # Text Processing

def document(file):
    
    file.seek(0)
    
    docx_file = Document(file)

    content = []


    for para in docx_file.paragraphs:
        
        if para.text.strip():
            
            content.append(para.text.strip())


    for table in docx_file.tables:
        
        for row in table.rows:
           
            row_text = " | ".join(cell.text.strip()for cell in row.cells)

            if row_text:
                
                content.append(row_text)

    return "\n".join(content)


def Pdf(file):
    
    file.seek(0)

    pdf_file = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:
            
            text = page.extract_text()

            if text:
                pdf_file += (text + "\n")

            tables = page.extract_tables()

            for table in tables:
                
                for row in table:
                    
                    pdf_file += (" ".join(str(cell or "") for cell in row) + "\n")

    return pdf_file


def Textfiles(file):
    
    extension = file.name.lower().split(".")[-1]

    if extension == "docx":
        
        return document(file)

    if extension == "pdf":
        
        return Pdf(file)

    else:
        return " "


def File_redaction(file):
    
    extension = file.name.lower().split(".")[-1]

    if extension == "docx":
        
        file.seek(0)

        doc = Document(file)
        
        def hash(match):
            
            return hashlib.sha256(match.group(0).encode('utf-8')).hexdigest()

        for paragraph in doc.paragraphs:
                
                redacted_text = paragraph.text
        
                for pattern in patterns:
                    redacted_text = pattern.sub(hash, redacted_text)
                
                paragraph.text = redacted_text

         
        for table in doc.tables:

            for row in table.rows:

                for cell in row.cells:

                    for paragraph in cell.paragraphs:

                        redacted_text = paragraph.text

                        for pattern in patterns:

                            redacted_text = pattern.sub(hash, redacted_text)

                            paragraph.text = redacted_text


        output = io.BytesIO()

        doc.save(output)
        
        output.seek(0)

        return output, "Hashing"


    if extension == "pdf":
        
        file.seek(0)

        pdf_bytes = file.read()

        pdf = fitz.open(
            stream = pdf_bytes,
            filetype = "pdf"
            )

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf_open:

            for page_number, plumber_page in enumerate(pdf_open.pages):
                page = pdf[page_number]

                words = plumber_page.extract_words()

                for word in words:
                    
                    word_text = word["text"]

                    sensitive = any(pattern.search(word_text) for pattern in patterns)

                    if sensitive:
                        rect = fitz.Rect(
                            word["x0"],
                            word["top"],
                            word["x1"],
                            word["bottom"]
                        )

                        page.add_redact_annot(rect, fill = (0, 0, 0))

                page.apply_redactions()

        output = io.BytesIO()

        pdf.save(output,garbage = 4, deflate = True)

        pdf.close()

        output.seek(0)

        return output, "Black_Box"

    return None, "Unsupported"




                        #What Happens after the redaction button pressed 

    

if file is not None:
        
    file_extension = file.name.rsplit('.', 1)[-1].lower()
        
    if file_extension in ('jpeg', 'jpg' , 'png'):
            
        process = Imagefiles(file)

        st.image(process, caption = 'Processed Image')

        redacted_image = Image_Redaction(process)

        st.image(redacted_image, caption = 'Redacted_Image')

        image_output = io.BytesIO()

        redacted_image.save(image_output, format = 'png')

        st.download_button(
            label = 'Download Image File',
            file_name = 'Redacted_Image.png',
            data = image_output.getvalue(),
            mime = 'image/png'
        )
        
    elif file_extension in ('docx' , 'pdf'):
            
        st.text_area('Text Area', Textfiles(file), height = 200)
    
        output, method = File_redaction(file)
    
        st.write('Technique: ', method)

        if file_extension == 'docx':
            name = 'Redacted_File.docx'
            ext_type = ('application/vnd.openxmlformats-officedocument.'
                        'wordprocessingml.document'
                        )
        
        elif file_extension == 'pdf':
            name = 'Redacted_File.pdf'
            ext_type = 'application/pdf'
        
        st.download_button(
            label = 'Download File',
            file_name = name,
            data = output.getvalue(),
            mime = ext_type
        )


