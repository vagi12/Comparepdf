import PyPDF2
import math
import re
from collections import Counter
import cv2
import os
import numpy as np

WORD = re.compile(r"\w+")
a =[]

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def get_cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if denominator == 0:
        return 0.0
    else:
        return float(numerator) / denominator

def cosine_best_match(File1,File2,dictm):
    comparison_pages = []
    File1_pages =  len(File1.pages)
    File2_pages = len(File2.pages)

    if File2_pages==File1_pages:
        for page in range(File2_pages):
            comparison_pages.append((page,page))
    else:
        pdf1 = PyPDF2.PdfReader(open(File1, 'rb'))
        pdf2 = PyPDF2.PdfReader(open(File2, 'rb'))

        num_pages_pdf1 = len(pdf1.pages)
        num_pages_pdf2 = len(pdf2.pages)

        visited_pages = []
        similarity_matrix = [[None] * num_pages_pdf2 for _ in range(num_pages_pdf1)]

        for i in range(num_pages_pdf1):
            text1 = pdf1.pages[i].extract_text()
            vector1 = text_to_vector(text1)
            max_similarity = -1
            matching_page = None

            for j in range(num_pages_pdf2):
                if j in visited_pages:
                    continue

                if similarity_matrix[i][j] is None:
                    text2 = pdf2.pages[j].extract_text()
                    vector2 = text_to_vector(text2)
                    similarity = get_cosine_similarity(vector1, vector2)
                    similarity_matrix[i][j] = similarity
                else:
                    similarity = similarity_matrix[i][j]

                if similarity > max_similarity:
                    max_similarity = similarity
                    matching_page = j

            visited_pages.append(matching_page)

            if max_similarity > 0.8:
                # print(f"({i}, {matching_page}) => {max_similarity}")
                comparison_pages.append((i,matching_page))
                a.append(matching_page)
            else:
                print(f"Page {i} from PDF1 has no matching page in PDF2")
        
        b = []
        for i in range(len(visited_pages)):
            b.append(visited_pages[i])
            for j in range(len(a)):
                if b[i] == a[j]:
                    b[i] = 0
                    
        for k in range(len(b)):
            if b[k] != 0:
                print(f"Page {b[k]} from PDF2 has no matching page in PDF1")

        # Print extra pages in PDF2
        for j in range(num_pages_pdf2):
            if j not in visited_pages:
                print(f"Extra page {j} found in PDF2")

    return comparison_pages
      
def compare_images(img1,img2,output_folder):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(img1_gray, img2_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    composite_image = np.hstack((img1, img2))
    # create directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # initialize file number to 1
    file_num = 1
    # save composite_image in output_folder with incremented file number
    filename = f'composite_image_{file_num}.jpg'
    while os.path.exists(f'{output_folder}/{filename}'):
        file_num += 1
        filename = f'composite_image_{file_num}.jpg'
    Image.fromarray(composite_image).save(f'{output_folder}/{filename}')
