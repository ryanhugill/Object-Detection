import cv2
from random import randint


# Load pre-trained data on objects
trained_object_data = cv2.CascadeClassifier('xmls/haarcascade_frontalface_default.xml')


def random_color():
    return [randint(0, 255), randint(0, 255), randint(0, 255)]


def object_detection_img(img_path):
    # Read in the image data to detect object in
    img = cv2.imread(img_path)

    # Convert chosen image to grayscale
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Finds pixel co-ordinates of object
    object_coordinates = trained_object_data.detectMultiScale(img_grayscale)

    # Draw a rectangle around the object(s)
    for object_found in object_coordinates:
        print("Object found at:", object_found)
        x, y, object_width, object_height = object_found
        cv2.rectangle(img, (x, y), (x + object_width, y + object_height), (0, 255, 0), 4)

    cv2.imshow('Object(s) With Rectangles', img)
    print("\nPress any key to close the image...")

    cv2.waitKey()
    cv2.destroyAllWindows()


def object_detection_video(video_path):
    # Read in the image data from the primary webcam
    webcam = cv2.VideoCapture(video_path)

    # Iterate over frames
    while True:
        reading_frame_success, frame = webcam.read()

        if reading_frame_success:
            # Convert current frame to grayscale
            frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Finds pixel co-ordinates of object
            object_coordinates = trained_object_data.detectMultiScale(frame_grayscale)

            # Draw a rectangle around the object(s)
            for object_found in object_coordinates:
                print("Object found at:", object_found)
                x, y, object_width, object_height = object_found
                cv2.rectangle(frame, (x, y), (x + object_width, y + object_height), (0, 255, 0), 4)

            cv2.imshow('Object(s) With Rectangles', frame)
            print("\nPress Q to close the video...")

        # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # object_detection_img('images/object.jpg')
    # object_detection_video('videos/object.mp4')
    object_detection_video(0)
