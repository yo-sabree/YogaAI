import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from ultralytics import YOLO


model = YOLO('yolov8n.pt')

yoga_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(107, activation='softmax')
])
yoga_model.load_weights('yoga-model.h5')


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


yoga_poses = ["adho mukha svanasana", "adho mukha vriksasana", "agnistambhasana", "ananda balasana",
              "anantasana", "anjaneyasana", "ardha bhekasana", "ardha chandrasana", "ardha matsyendrasana",
              "ardha pincha mayurasana", "ardha uttanasana", "ashtanga namaskara", "astavakrasana",
              "baddha konasana", "bakasana", "balasana", "bhairavasana", "bharadvajasana i", "bhekasana",
              "bhujangasana", "bhujapidasana", "bitilasana", "camatkarasana", "chakravakasana",
              "chaturanga dandasana", "dandasana", "dhanurasana", "durvasasana", "dwi pada viparita dandasana",
              "eka pada koundinyanasana i", "eka pada koundinyanasana ii", "eka pada rajakapotasana",
              "eka pada rajakapotasana ii", "ganda bherundasana", "garbha pindasana", "garudasana",
              "gomukhasana", "halasana", "hanumanasana", "janu sirsasana", "kapotasana", "krounchasana",
              "kurmasana", "lolasana", "makara adho mukha svanasana", "makarasana", "malasana",
              "marichyasana i", "marichyasana iii", "marjaryasana", "matsyasana", "mayurasana",
              "natarajasana", "padangusthasana", "padmasana", "parighasana", "paripurna navasana",
              "parivrtta janu sirsasana", "parivrtta parsvakonasana", "parivrtta trikonasana",
              "parsva bakasana", "parsvottanasana", "pasasana", "paschimottanasana", "phalakasana",
              "pincha mayurasana", "prasarita padottanasana", "purvottanasana", "salabhasana",
              "salamba bhujangasana", "salamba sarvangasana", "salamba sirsasana", "savasana",
              "setu bandha sarvangasana", "simhasana", "sukhasana", "supta baddha konasana",
              "supta matsyendrasana", "supta padangusthasana", "supta virasana", "tadasana",
              "tittibhasana", "tolasana", "tulasana", "upavistha konasana", "urdhva dhanurasana",
              "urdhva hastasana", "urdhva mukha svanasana", "urdhva prasarita eka padasana",
              "ustrasana", "utkatasana", "uttana shishosana", "uttanasana", "utthita ashwa sanchalanasana",
              "utthita hasta padangustasana", "utthita parsvakonasana", "utthita trikonasana",
              "vajrasana", "vasisthasana", "viparita karani", "virabhadrasana i", "virabhadrasana ii",
              "virabhadrasana iii", "virasana", "vriksasana", "vrischikasana", "yoganidrasana"]


def imgProcess(image):
    img = cv2.resize(image, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = yoga_model.predict(img)
    return np.argmax(prediction)



def yogaPoseDetect():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    frame_count = 0
    detected_pose = "Detecting..."

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)

        for r in results:
            for box in r.boxes:
                id = int(box.cls[0])
                class_name = model.names[id]

                if class_name == "person":
                    pose_results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    if frame_count % 10 == 0:  # Predict every 10 frames to reduce lag
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped_img = img[y1:y2, x1:x2]
                        if cropped_img.size != 0:
                            predicted_pose = imgProcess(cropped_img)
                            detected_pose = yoga_poses[predicted_pose]

        frame_count += 1

        cv2.putText(img, detected_pose, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Yoga Pose Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


yogaPoseDetect()
