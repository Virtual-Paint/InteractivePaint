from PIL import Image
from io import BytesIO
import base64


class InputProcessing:
    @classmethod
    def bytes_to_image(cls, byte_data):
        print(byte_data)
        
        base64_data = byte_data.split(',')[1]

        # Decode the base64 data
        binary_data = base64.b64decode(base64_data)

        # Create a BytesIO object to simulate a file-like object
        image_stream = BytesIO(binary_data)

        # Open the image using Pillow
        image = Image.open(image_stream)

        # Display the image (optional)
        image.save('siema.jpg')
