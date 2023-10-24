import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import time
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import Adam

# กำหนด path ของ Data set โดยแต่ละชนิดจะแบ่งเป็นแต่ละโฟลเดอร์
data_dir = "Dataset_with_bg"
categories = os.listdir(data_dir)

# สร้างรายการเก็บข้อมูล
data = []
labels = []
# นำรูปภาพแต่ละรูปไปเก็บไว้ใน data และเก็บชนิดไว้ใน labels
for category in categories:
    path = os.path.join(data_dir, category)
    print(path)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        if image is not None and not image.size == 0: # ถ้ามีขนาดเป็น 0 ให้ข้าม
            image = cv2.resize(image, (224, 224))  # ปรับขนาดรูปภาพเป็น 224x224 (MobileNetV2)
            data.append(image)
            labels.append(category)

# แปลง labels ให้เป็นตัวเลขโดยใช้ LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

# บันทึก LabelEncoder ลงในไฟล์
joblib.dump(le, 'Label_encoder.joblib')

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(np.array(data), labels, test_size=0.2, random_state=42)

# โหลดโมเดล MobileNetV2 (pre-trained)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(len(categories), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)

# แชร์ layers แรกของโมเดล (Freeze)
for layer in base_model.layers:
    layer.trainable = False

# คอมไพล์และคอมไพล์โมเดล
model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ปรับค่า pixel ให้อยู่ในรูปแบบที่ MobileNetV2 รับได้
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

# ฝึกโมเดล CNN
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# บันทึกโมเดล
model_filename = "CNN_model.h5"
model.save(model_filename)

# ทดสอบโมเดล
y_pred = model.predict(X_test)
y_cnn_pred = np.argmax(y_pred, axis=1)

# คำนวณค่า Accuracy
accuracy = accuracy_score(y_test, y_cnn_pred)
print("Accuracy:", accuracy)

# คำนวณค่า Recall
recall = recall_score(y_test, y_cnn_pred, average='weighted', zero_division=1)
print("Recall:", recall)

# คำนวณค่า F1 Score
f1 = f1_score(y_test, y_cnn_pred, average='weighted')
print("F1 Score:", f1)

# คำนวณ Confusion Matrix
confusion = confusion_matrix(y_test, y_cnn_pred)
print("Confusion Matrix:")
print(confusion)