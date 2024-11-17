import cv2
import glob


images_paths = glob.glob("../data/*")

for image_path in images_paths:
    img = cv2.imread(image_path)
    #cv2.imshow("blabla", img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners_resp = cv2.cornerHarris(gray_image, blockSize=3, ksize=3, k=0.04)
    print(corners_resp)
    corner_response = cv2.dilate(src=corners_resp, kernel=None)
    #cv2.imshow("bab", r)
    threshold = 0.002 * corner_response.max()
    img[corner_response > threshold] = [0, 0, 255]  # Mark corners in red

    # Display the result
    cv2.imshow('Harris Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

