import numpy as np
import cv2
import pickle
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX


# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRAInED MODEL
pickle_in = open("model_trained.p", "rb")  # rb = READ BYTE
model = pickle.load(pickle_in)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)

    img = equalize(img)
    img = img / 255
    return img


def gcn(classno):
    if classno == 0:
        return 'Speed Limit 20 km/h'
    elif classno == 1:
        return 'Speed Limit 30 km/h'
    elif classno == 2:
        return 'Speed Limit 50 km/h'
    elif classno == 3:
        return 'Speed Limit 60 km/h'
    elif classno == 4:
        return 'Speed Limit 70 km/h'
    elif classno == 5:
        return 'Speed Limit 80 km/h'
    elif classno == 6:
        return 'End of Speed Limit 80 km/h'
    elif classno == 7:
        return 'Speed Limit 100 km/h'
    elif classno == 8:
        return 'Speed Limit 120 km/h'
    elif classno == 9:
        return 'No passing'
    elif classno == 10:
        return 'No passing for vechiles over 3.5 metric tons'
    elif classno == 11:
        return 'Right-of-way at the next intersection'
    elif classno == 12:
        return 'Priority road'
    elif classno == 13:
        return 'Yield'
    elif classno == 14:
        return 'Stop'
    elif classno == 15:
        return 'No vechiles'
    elif classno == 16:
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classno == 17:
        return 'No entry'
    elif classno == 18:
        return 'General caution'
    elif classno == 19:
        return 'Dangerous curve to the left'
    elif classno == 20:
        return 'Dangerous curve to the right'
    elif classno == 21:
        return 'Double curve'
    elif classno == 22:
        return 'Bumpy road'
    elif classno == 23:
        return 'Slippery road'
    elif classno == 24:
        return 'Road narrows on the right'
    elif classno == 25:
        return 'Road work'
    elif classno == 26:
        return 'Traffic signals'
    elif classno == 27:
        return 'Pedestrians'
    elif classno == 28:
        return 'Children crossing'
    elif classno == 29:
        return 'Bicycles crossing'
    elif classno == 30:
        return 'Beware of ice/snow'
    elif classno == 31:
        return 'Wild animals crossing'
    elif classno == 32:
        return 'End of all speed and passing limits'
    elif classno == 33:
        return 'Turn right ahead'
    elif classno == 34:
        return 'Turn left ahead'
    elif classno == 35:
        return 'Ahead only'
    elif classno == 36:
        return 'Go straight or right'
    elif classno == 37:
        return 'Go straight or left'
    elif classno == 38:
        return 'Keep right'
    elif classno == 39:
        return 'Keep left'
    elif classno == 40:
        return 'Roundabout mandatory'
    elif classno == 41:
        return 'End of no passing'
    elif classno == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'


while True:

    # READ IMAGE
    success, imgOrignal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        # print(gcn(classIndex))
        cv2.putText(imgOrignal, str(classIndex) + " " + str(gcn(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
