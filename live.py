#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
#Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(0)

frame1 = 0
frame2 = 0
def motionDetection():
    global frame1
    global frame2

    while camera.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 2000:
                continue
            print("motion detected !!!!!")
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (255, 0, 0), 3)

        # cv.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        toshow = frame1
        frame1 = frame2
        ret, frame2 = camera.read()
        return toshow



def gen_frames():
    global frame1
    global frame2
    while True:
        success, frame1 = camera.read()
        success, frame2 = camera.read()# read the camera frame
        if not success:
            break
        else:
            toshow = motionDetection()
            ret, buffer = cv2.imencode('.jpg', toshow)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='192.168.18.16', port=5000)
