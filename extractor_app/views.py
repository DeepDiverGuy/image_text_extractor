from rest_framework.decorators import api_view
from rest_framework.response import Response

import os
import base64
import fitz
import numpy as np
import pytesseract
from PIL import Image
import cv2
from io import BytesIO  # (development)



# Create your views here.


'''
REQUEST INFORMATION -

API: /api/process_pdf

Method: POST

JS example: 
  const formData = new FormData();
  formData.append('pdf_file', pdfFile);

  fetch('/api/process_pdf/', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    // handle response
  })
  .catch(error => {
    // handle error
  });
'''

@api_view(['POST'])
def process_pdf(request):

    error_messages = {}

    # for obvious reasons, we can't use fractions for 'multiply_count'
    # we can change it for better quality pics. It should be integer values.
    # eg: 1 for dpi=100
    multiply_count = 3
    # print(multiply_count)

    # for obvious reasons, we are using multiples of 100 for 'dpi_to_use'
    dpi_to_use = 100*multiply_count

    # signature-starting-point co-ordinates are assigned to these
    bottom_left_x = multiply_count*840  # 845
    bottom_left_y = multiply_count*320  # 320

    pytess_lang = 'eng+Bengali'  # 'Bengali' worked well in my Ubuntu, but in another Windows, I had to use 'ben'
    os.environ["TESSDATA_PREFIX"] = os.path.join(os.getcwd(), 'tessdata')

    # Get the uploaded file from the request
    pdf_file = request.FILES['pdf_file']
        
    # extracted text from OpenCV is written in this variable
    final_dict = {
    "Present Address": {},
    "Permanent Address": {},
    }

    # extracted image of face is populated in this variable as base64 encoded string
    face_pic_base64 = None

    # extracted image of signature is populated in this variable as base64 encoded string
    signature_pic_base64 = None

    # Load the PDF document
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    except:
        return Response({
            "error": "PDF file couldn't be read"
        })

    # iterate through the pages
    for page in doc:

        # render page to an image
        current_page = page.get_pixmap(dpi = dpi_to_use)

        # store image as a PNG, for testing (development)
        # current_page.save("page-%i.png" % page.number+1)

        # Convert the pixmap image into bytes & the bytes to a numpy array
        current_page_array = np.frombuffer(current_page.tobytes(), dtype=np.uint8)

        # Decode the numpy array using OpenCV and convert to grayscale
        img_color = cv2.imdecode(current_page_array, cv2.IMREAD_UNCHANGED)
        img = cv2.imdecode(current_page_array, cv2.IMREAD_GRAYSCALE)


        # Getting all the texts from the current page along with their metadata
        # Used: PIL & pytesseract
        # Function: specified both the english & bangla trained model (keep in mind, as named inside tesseract directory)

        # Convert the numpy array to a PIL image object
        current_page_pil = Image.fromarray(img).convert('L')

        # Perform OCR and get the results as a list of dictionaries
        results = pytesseract.image_to_data(current_page_pil, lang=pytess_lang, output_type='dict')
        # print(results) (development)

        # Define a confidence threshold
        confidence_threshold = 40

        # Create an empty dictionary to store the filtered results
        all_texts = {}

        text_index = 0
        # Loop through the results and filter based on confidence
        for index, c_l in enumerate(results['conf']):

            confidence_level = int(c_l)

            if confidence_level >= confidence_threshold:

                # WARNING: don't change the format below, you will end-up getting same word once only
                all_texts[text_index] = {
                    'text': f"{results['text'][index]}",
                    'top': int(results['top'][index]),
                    'left': int(results['left'][index]),
                    'height': int(results['height'][index]),
                    'width': int(results['width'][index]),
                }
                text_index += 1

        
        # Getting co-ordinates of all the cells of the table
        # Used: CV2
        # Function: [Complex]

        # apply binary thresholding
        ret,thresh = cv2.threshold(img,247,255,0)

        # test show of the threshe d image (development)
        # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        # cv2.imshow("test", thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Find contours in the binary image and loop through them to find the rectangles
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE+cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        rectangle_box_list = []

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                # each cell we wanna grab has height more than 20*multiply_count pixel and can have one of three different widths (approx) - 180, 550, 135
                if (20*multiply_count<h) and (175*multiply_count<w<185*multiply_count or 545*multiply_count<w<555*multiply_count or 130*multiply_count<w<140*multiply_count):
                    # file.write(pytesseract.image_to_string(image[y:y+h, x:x+w], lang='eng+Bengali'))
                    # file.write(f"{y}, {x}, {h}, {w}\n")
                    rectangle_box_list.append([y, x, h, w])

        rectangle_box_list_sorted = sorted(rectangle_box_list, key=lambda x: (x[0], x[1]))

        rectangle_box_dict = {i: {'box': item, 'box_text': ''} for i, item in enumerate(rectangle_box_list_sorted)}


        def is_inside(in_box, out_box):
            """Checks if in_box is completely contained within out_box"""
            top1, left1, height1, width1 = in_box
            top2, left2, height2, width2 = out_box

            bottom1 = top1 + height1
            right1 = left1 + width1
            bottom2 = top2 + height2
            right2 = left2 + width2

            return (top1+5*multiply_count) >= top2 and (left1+5*multiply_count) >= left2 and bottom1 <= (bottom2+5*multiply_count) and right1 <= (right2+5*multiply_count)

        for i in rectangle_box_dict:
            for text_index in all_texts:
                t_top, t_left, t_height, t_width = all_texts[text_index]['top'], all_texts[text_index]['left'], all_texts[text_index]['height'], all_texts[text_index]['width'],
                if is_inside([t_top, t_left, t_height, t_width], rectangle_box_dict[i]['box']):
                    rectangle_box_dict[i]['box_text'] += f" {all_texts[text_index]['text']}"


        rowcolumn_index_count = 0

        if page.number == 0:

            # Extracting text from page 0 (1st page)

            try:          
                

                # # text = pytesseract.image_to_string(Image.open("page-%i.png" % page.number), lang='eng+Bengali')
                # text += pytesseract.image_to_string(current_page_pil, lang=pytess_lang)

                for row in range(1,43):

                    if 23<row<=30 or 31<row<=38:
                        
                        if 23<row<=30:
                            final_dict["Present Address"][rectangle_box_dict[rowcolumn_index_count]['box_text'].strip()] = rectangle_box_dict[rowcolumn_index_count+1]['box_text'].strip()
                            final_dict["Present Address"][rectangle_box_dict[rowcolumn_index_count+2]['box_text'].strip()] = rectangle_box_dict[rowcolumn_index_count+3]['box_text'].strip()
                        else:
                            final_dict["Permanent Address"][rectangle_box_dict[rowcolumn_index_count]['box_text'].strip()] = rectangle_box_dict[rowcolumn_index_count+1]['box_text'].strip()
                            final_dict["Permanent Address"][rectangle_box_dict[rowcolumn_index_count+2]['box_text'].strip()] = rectangle_box_dict[rowcolumn_index_count+3]['box_text'].strip()
                        rowcolumn_index_count += 4

                    elif row==31:
                        final_dict["Present Address"][rectangle_box_dict[rowcolumn_index_count]['box_text'].strip()] = rectangle_box_dict[rowcolumn_index_count+1]['box_text'].strip()
                        rowcolumn_index_count += 3
                    
                    elif row==39:
                        final_dict["Permanent Address"][rectangle_box_dict[rowcolumn_index_count]['box_text'].strip()] = rectangle_box_dict[rowcolumn_index_count+1]['box_text'].strip()
                        rowcolumn_index_count += 2

                    else:
                        final_dict[rectangle_box_dict[rowcolumn_index_count]['box_text'].strip()] = rectangle_box_dict[rowcolumn_index_count+1]['box_text'].strip()
                        rowcolumn_index_count += 2
                        if row==23:
                            rowcolumn_index_count += 1

                final_dict["Sl No"] = final_dict.pop("SI No")

            except:
                error_messages[f"text extraction error - {page.number}"] = f"error occurred on page: {page.number}"


            # Getting face image
            # Used: Fitz
            # Function: Loop once to get the first image object (the face image) on this page

            if page.get_images() == []:
                error_messages["face extraction error"] = "no image found in the first page"

            else:
                for image_obj in page.get_images():

                    xref = image_obj[0]
                    # print(image_obj)
                    face_pic = fitz.Pixmap(doc, xref)

                    # if pixmap.n > 4:
                    #     face_pic = fitz.Pixmap(fitz.csRGB, face_pic)

                    # Convert the pixmap image to a byte string & encode the byte string as base64
                    face_pic_base64 = base64.b64encode(face_pic.tobytes()).decode('utf-8')

                    if face_pic_base64 == None:
                        error_messages["face extraction error"] = f"error occurred while extracting picture of face"

                    # face_pic.save(f"face.png")
                    # face_pic = None

                    break
                   

            # Getting signature image. Because Fitz (Pixmap) was producing incorrect image, I had to use OpenCV.
            # Used: OpenCV
            # Function: [complex]

            try:               

                if img is None:
                    raise Exception("file could not be read, check with os.path.exists()")

                # Define the area of interest by specifying the top-left coordinate
                # and the width and height of the region to search for the quadrilateral
                left = bottom_left_x
                top = bottom_left_y
                width = 200*multiply_count  # 185 is the main width, if started from exact co-ordinate
                height = 400*multiply_count

                # Crop the image to the area of interest. cropped_image - region of interest
                cropped_image_color = img_color[top:top+height, left:left+width]
                cropped_image = img[top:top+height, left:left+width]

                # apply binary thresholding
                ret,thresh = cv2.threshold(cropped_image,200,255,cv2.THRESH_BINARY_INV)

                # test show of the threshed image (development)
                # cv2.imshow("test", thresh)
                # cv2.waitKey(5000)
                # cv2.destroyAllWindows()

                # Find contours in the binary image and loop through them to find the largest quadrilateral
                # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE+cv2.THRESH_OTSU)
                contours, hierarchy = cv2.findContours(thresh, 1, 2)

                largest_area = 0
                largest_contour = None

                for contour in contours:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    if len(approx) == 4:
                        area = cv2.contourArea(approx)
                        if area > largest_area:
                            largest_area = area
                            largest_contour = approx

                # Extract the largest quadrilateral as a new image and return it
                # mask = np.zeros(thresh.shape, np.uint8)
                # cv2.drawContours(mask, [largest_contour], 0, 255, -1)
                # result = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
                
                # Draw the bounding rectangle around the quadrilateral and extract the cropped_image
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                signature_pic = cropped_image_color[y:y+h, x:x+w]

                # test show of the final image (development)
                # cv2.imshow("Signature", result)
                # cv2.waitKey(5000)
                # cv2.destroyAllWindows()

                # Convert the signature_pic to bytes
                signature_pic_bytes = cv2.imencode('.png', signature_pic)[1].tobytes()

                # Convert the bytes to base64 encoded string
                signature_pic_base64 = base64.b64encode(signature_pic_bytes).decode('utf-8')

                # cv2.imwrite('signature.png', signature_pic)

            except:
                error_messages["signature extraction error"] = f"error occurred while extracting picture of signature"


        elif page.number == 1:

            # Extracting text from page 1 (2nd page)

            for row in range(1,23):
                try:
                    final_dict[rectangle_box_dict[rowcolumn_index_count]['box_text'].strip()] = rectangle_box_dict[rowcolumn_index_count+1]['box_text'].strip()
                    rowcolumn_index_count += 2
                except:
                    error_messages[f"text extraction error - {page.number}"] = f"error occurred on page: {page.number}"

    doc.close()

    # the 2 lines below are for testing (development)
    # Image.open(BytesIO(base64.b64decode(face_pic_base64))).save("face.png")
    # Image.open(BytesIO(base64.b64decode(signature_pic_base64))).save("signature.png")

    # Return the extracted text and images in the response
    response = {
        'error_messages': error_messages,
        'image_encoding_type': 'base64',
        'text_data': final_dict,
        'face': face_pic_base64,
        'signature': signature_pic_base64,
        
    }
    return Response(response)
