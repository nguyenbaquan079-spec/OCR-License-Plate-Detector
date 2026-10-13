import cv2
import easyocr
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "englishquote.jpg")

reader = easyocr.Reader(['en'])

img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not read image at {image_path}")
    exit()

results = reader.readtext(img)

for bbox, text, confidence in results:

    print(f"Text: {text}")
    print(f"Confidence: {confidence:.2f}")

    # Convert bounding box coordinates to integers
    points = [(int(x), int(y)) for x, y in bbox]

    # Draw bounding box
    cv2.polylines(
        img,
        [cv2.convexHull(
            __import__('numpy').array(points)
        )],
        True,
        (0, 255, 0),
        2
    )

    # Put recognized text on image
    x, y = points[0]

    cv2.putText(
        img,
        text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

output_path = os.path.join(script_dir, "result.jpg")
cv2.imwrite(output_path, img)
print(f"Result saved to {output_path}")

