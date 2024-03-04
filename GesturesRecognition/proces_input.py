from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from io import BytesIO
import base64


class InputProcessing:
    @classmethod
    def bytes_to_image(cls, byte_data):
        base64_data = byte_data.split(',')[1]

        # Decode the base64 data
        binary_data = base64.b64decode(base64_data)

        # Create a BytesIO object to simulate a file-like object
        image_stream = BytesIO(binary_data)

        # Open the image using Pillow
        image = Image.open(image_stream)

        myFont = ImageFont.truetype('FreeMono.ttf', 65)
        I1 = ImageDraw.Draw(image)
        I1.text((100, 100), "processed", font=myFont, fill=(255, 0, 0))

        # Display the image (optional)
        #image.save(f'test/{idx}.jpg')

        buffer = BytesIO()

        # Save the image to the buffer
        image.save(buffer, format="JPEG")

        # Get the base64-encoded string
        mime_type = "image/jpeg"  # Adjust based on your image format

        # Get the base64-encoded string with MIME type
        base64_image = f"{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
        return base64_image
