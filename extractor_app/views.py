from rest_framework.decorators import api_view
from rest_framework.response import Response

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
    multiply_count = 2
    # print(multiply_count)

    # for obvious reasons, we are using multiples of 100 for 'dpi_to_use'
    dpi_to_use = 100*multiply_count

    # signature-starting-point co-ordinates is assigned to these
    bottom_left_x = multiply_count*840  # 845
    bottom_left_y = multiply_count*320  # 320

    # Get the uploaded file from the request
    pdf_file = request.FILES['pdf_file']
        
    # extracted text from OpenCV is written in this variable
    text = ''

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


        # Extracting text from the image
        # Used: pytesseract
        # Function: specified both the english & bangla trained model (keep in mind, as named inside tesseract directory)
        try:          
            # Convert the numpy array to a PIL image object
            current_page_pil = Image.fromarray(img)

            # text = pytesseract.image_to_string(Image.open("page-%i.png" % page.number), lang='eng+Bengali')
            text += pytesseract.image_to_string(current_page_pil, lang='eng+Bengali')  # 'Bengali' worked well in my Ubuntu, but in another Windows, I had to use 'ben'
        except:
            error_messages["text extraction error"] = f"error occurred on page: {page.number+1}"


        if page.number == 0:

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

    doc.close()

    # the 2 lines below are for testing (development)
    # Image.open(BytesIO(base64.b64decode(face_pic_base64))).save("face.png")
    # Image.open(BytesIO(base64.b64decode(signature_pic_base64))).save("signature.png")

    # Return the extracted text and images in the response
    response = {
        'error_messages': error_messages,
        'image_encoding_type': 'base64',
        'text': text,
        'face': face_pic_base64,
        'signature': signature_pic_base64,
        
    }
    return Response(response)
