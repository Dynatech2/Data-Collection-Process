import cv2
from threading import Thread, Lock
import os
import time
import numpy as np


# List of RTSP stream URLs
urls = [
    "rtsp://admin:Dyna1234@180.74.167.65:551/stream0",
    "rtsp://admin:Dyna1234@180.74.167.65:552/stream0",
    "rtsp://admin:Dyna1234@180.74.167.65:553/stream0",
    "rtsp://admin:Dyna1234@180.74.167.65:554/stream0",
]

display_width, display_height = 640, 480
frame_locks = [Lock() for _ in urls]
latest_frames = [None for _ in urls]
running = True
screenshot_interval = 10

base_download_dir = "D:/Image_5"
os.makedirs(base_download_dir, exist_ok=True)


def capture_images(index, url):
    global latest_frames, running
    last_screenshot_time = time.time()
    image_counter = 1
    camera_dir = os.path.join(base_download_dir, f"camera_{index}")
    os.makedirs(camera_dir, exist_ok=True)

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)

    while running:
        ret, frame = cap.read()
        if not ret:
            print(f"[Error] Failed to read frame from {url}. Reinitializing...")
            cap.release()
            time.sleep(10)  # Wait before trying to reconnect
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)
            continue

        frame = cv2.resize(frame, (display_width, display_height))
        with frame_locks[index]:
            latest_frames[index] = frame.copy()

        current_time = time.time()
        if (current_time - last_screenshot_time) >= screenshot_interval:
            output_path = os.path.join(camera_dir, f"Image_{image_counter}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved {output_path}")
            last_screenshot_time = current_time
            image_counter += 1

    cap.release()


def display_streams():
    global running
    while running:
        frames = []

        for i, frame_lock in enumerate(frame_locks):
            with frame_lock:
                frame = latest_frames[i]
                if frame is not None:
                    frames.append(frame)
                else:
                    # If a frame is missing, create a blank one
                    frames.append(np.zeros((display_height, display_width, 3), dtype=np.uint8))

        # Create a 2x2 grid of the four camera streams
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        grid = np.vstack((top_row, bottom_row))

        # Display the combined frame in one window
        cv2.imshow("Multi-Camera Display", grid)

        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Start capture threads
    capture_threads = [Thread(target=capture_images, args=(i, url)) for i, url in enumerate(urls)]
    for t in capture_threads:
        t.start()

    # Start display thread
    display_thread = Thread(target=display_streams)
    display_thread.start()

    # Join capture threads
    for t in capture_threads:
        t.join()

    # Join display thread
    display_thread.join()
