# my code for black and white image
from PIL import Image
import os

imPath  = "original_images"
cropPath = "cropped_images"
imFiles = os.listdir(imPath) 

for filename in imFiles:
    image = Image.open(os.path.join(imPath,filename)) # name of the image file
    rgb_image = image.convert('RGB')
    width, height = image.size

    output = Image.new('RGB', (width, height))
    # cropped: x:180-920, y: 210-550, => 740x340
    cropped = Image.new('RGB', (740, 340))

    for row in range(height):
        for col in range(width):
            r, g, b = image.getpixel((col, row))

            p = (int)((0.3 * r) + (0.3 * g) + (0.4 * b))
            if ( p < 128 ):
                p = max(0, 128 -(128-p)*2+64)
            else:
                p = min(255, 128 + (p-128)*2+64)

            output.putpixel((col, row),(p,p,p))

            if col >= 180 and col < 920 and row >= 210 and row < 550 :
                cropped.putpixel((col-180, row-210),(p,p,p))

    # output.save("manometro_clarendon.jpg")

    cropped.save(os.path.join(cropPath,"cropped_"+filename))
