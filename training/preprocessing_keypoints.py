from ultralytics import YOLO
import os
import cv2
import pandas as pd
import numpy as np

# Lade das YOLO Pose Modell
pose_model = YOLO("models/yolo11x-pose.pt")

# Lade das YOLO Detection Modell für Spieler und Ball
detection_model = YOLO("models/ball_detection_V4.pt")

# Datenverzeichnis
data_dir = "/Users/luca/Downloads/data_challenge_gathered"
output_dir = "training/Ergebnisse_preprocessing"
classes = sorted(os.listdir(data_dir))  # 17 Klassen

# Erstelle den Ergebnisse-Ordner, falls er nicht existiert
os.makedirs(output_dir, exist_ok=True)

# Keypoints-Bezeichnungen
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Spaltenüberschriften für CSV
columns = ["Bildname", "image_width", "image_height"]  # Neu: Breite & Höhe
for kp in keypoint_names:
    columns.append(f"{kp}_x")
    columns.append(f"{kp}_y")
columns += ["label", "player_count", "ball_count", "ball_x", "ball_y"]

data = []

# Funktion für Distanz-Berechnung
def euclidean_distance(x1, y1, x2, y2):
    if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):  # Falls einer der Punkte nicht existiert
        return np.nan  # Oder eine andere sinnvolle Behandlung
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Funktion für Winkel-Berechnung
def calculate_angle(a, b, c):
    if (a[0] == 0 and a[1] == 0) or (b[0] == 0 and b[1] == 0) or (c[0] == 0 and c[1] == 0):
        return np.nan  # Keine Berechnung, da einer der Punkte nicht existiert

    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan  # Falls zwei Punkte identisch sind, ist der Winkel nicht definiert

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


#################### Jetzt Keypoint Extraction und Object Detection #######################################
# Gehe durch jede Klasse
for class_label, class_name in enumerate(classes):
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        if img_name.startswith('.') or not os.path.isfile(img_path):
            continue

        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Bild laden
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_height, image_width = image.shape[:2]  # Bildgröße ermitteln

        # YOLO-Pose ausführen
        results = pose_model(img_path, conf=0.6)

        if len(results) == 0:
            continue

        keypoints = results[0].keypoints.xy.cpu().numpy()
        keypoints_flat = []
        for i in range(len(keypoint_names)):
            keypoints_flat.append(keypoints[0][i, 0])  # x-Wert
            keypoints_flat.append(keypoints[0][i, 1])  # y-Wert

        # YOLO Object Detection ausführen
        detection_results = detection_model(img_path, conf=0.8)

        # Zähle die erkannten Objekte
        player_count = 0
        ball_detected = 0  # Boolean: 1 = mind. ein Ball erkannt, 0 = kein Ball erkannt
        ball_x, ball_y = 0, 0  # Position des Balls (falls vorhanden)

        for result in detection_results:
            for box, cls in zip(result.boxes.xywh.cpu().numpy(), result.boxes.cls.int().cpu().numpy()):
                class_name = result.names[cls]
                if class_name == "player":
                    player_count += 1
                elif class_name == "ball":
                    ball_detected = 1
                    ball_x, ball_y = int(box[0]), int(box[1])  # Ball-Koordinaten (Mittelpunkt)

        # Daten in Liste speichern
        data.append([img_name, image_width, image_height] + keypoints_flat + [class_label, player_count, ball_detected, ball_x, ball_y])

# DataFrame speichern
df = pd.DataFrame(data, columns=columns)
df.to_csv(os.path.join(output_dir, "yolo_pose_data.csv"), index=False)

print("✅ Keypoints, Objektdetektionen und Bildgrößen erfolgreich gespeichert!")



#################### Jetzt Caluclation #######################################
# CSV-Datei einlesen
df = pd.read_csv(os.path.join(output_dir, "yolo_pose_data.csv"))

features = []

for _, row in df.iterrows():
    image_name = row["Bildname"]
    
    # Berechnung der Abstände
    shoulder_width = euclidean_distance(row.get("left_shoulder_x"), row.get("left_shoulder_y"),
                                        row.get("right_shoulder_x"), row.get("right_shoulder_y"))
    hip_width = euclidean_distance(row.get("left_hip_x"), row.get("left_hip_y"),
                                        row.get("right_hip_x"), row.get("right_hip_y"))
    arm_length_left = euclidean_distance(row.get("left_shoulder_x"), row.get("left_shoulder_y"), row.get("left_elbow_x"), row.get("left_elbow_y")) + \
                      euclidean_distance(row.get("left_elbow_x"), row.get("left_elbow_y"), row.get("left_wrist_x"), row.get("left_wrist_y"))
    arm_length_right = euclidean_distance(row.get("right_shoulder_x"), row.get("right_shoulder_y"), row.get("right_elbow_x"), row.get("right_elbow_y")) + \
                       euclidean_distance(row.get("right_elbow_x"), row.get("right_elbow_y"), row.get("right_wrist_x"), row.get("right_wrist_y"))
    
    # Berechnung der Winkel
    shoulder_angle_rigth = calculate_angle(
        (row.get("left_shoulder_x"), row.get("left_shoulder_y")),
        (row.get("right_shoulder_x"), row.get("right_shoulder_y")),
        (row.get("right_hip_x"), row.get("right_hip_y"))
    ) 

    shoulder_angle_left = calculate_angle(
        (row.get("right_shoulder_x"), row.get("right_shoulder_y")),
        (row.get("left_shoulder_x"), row.get("left_shoulder_y")),
        (row.get("left_hip_x"), row.get("left_hip_y"))
    )

    underarm_angle_rigth = calculate_angle(
        (row.get("left_elbow_x"), row.get("left_elbow_y")),
        (row.get("right_shoulder_x"), row.get("right_shoulder_y")),
        (row.get("right_hip_x"), row.get("right_hip_y"))
    )

    underarm_angle_rigth = calculate_angle(
        (row.get("right_elbow_x"), row.get("right_elbow_y")),
        (row.get("left_shoulder_x"), row.get("left_shoulder_y")),
        (row.get("left_hip_x"), row.get("left_hip_y"))
    )

    underarm_angle_left = calculate_angle(
        (row.get("left_elbow_x"), row.get("left_elbow_y")),
        (row.get("right_shoulder_x"), row.get("right_shoulder_y")),
        (row.get("right_hip_x"), row.get("right_hip_y"))
    )

    elbow_angle_left = calculate_angle(
        (row.get("left_shoulder_x"), row.get("left_shoulder_y")),
        (row.get("left_elbow_x"), row.get("left_elbow_y")),
        (row.get("left_wrist_x"), row.get("left_wrist_y"))
    )
    elbow_angle_right = calculate_angle(
        (row.get("right_shoulder_x"), row.get("right_shoulder_y")),
        (row.get("right_elbow_x"), row.get("right_elbow_y")),
        (row.get("right_wrist_x"), row.get("right_wrist_y"))
    )
    
    knee_angle_left = calculate_angle(
        (row.get("left_hip_x"), row.get("left_hip_y")),
        (row.get("left_knee_x"), row.get("left_knee_y")),
        (row.get("left_ankle_x"), row.get("left_ankle_y"))
    )
    knee_angle_right = calculate_angle(
        (row.get("right_hip_x"), row.get("right_hip_y")),
        (row.get("right_knee_x"), row.get("right_knee_y")),
        (row.get("right_ankle_x"), row.get("right_ankle_y"))
    )

    nose_angle = calculate_angle(
        (row.get("left_eye_x"), row.get("left_eye_y")),
        (row.get("nose_x"), row.get("nose_y")),
        (row.get("right_eye_x"), row.get("right_eye_y"))
    )
    
    feature_row = row.to_dict()  # Konvertiert die gesamte Zeile in ein Dictionary
    feature_row.update({
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "arm_length_left": arm_length_left,
        "arm_length_right": arm_length_right,
        "shoulder_angle_rigth": shoulder_angle_rigth,
        "shoulder_angle_left": shoulder_angle_left,
        "underarm_angle_rigth": underarm_angle_rigth,
        "underarm_angle_left": underarm_angle_left,
        "elbow_angle_left": elbow_angle_left,
        "elbow_angle_right": elbow_angle_right,
        "knee_angle_left": knee_angle_left,
        "knee_angle_right": knee_angle_right,
        "nose_angle": nose_angle
    })
    
    features.append(feature_row)

# Ergebnisse in eine neue CSV speichern
features_df = pd.DataFrame(features)
features_df.fillna(0, inplace=True)  # Fehlende Werte durch 0 ersetzen
features_df.to_csv(os.path.join(output_dir, "features.csv"), index=False)

print("Feature-Extraktion abgeschlossen und gespeichert in Features.csv")

#################### Jetzt Normalisierung #######################################

# CSV einlesen
df = pd.read_csv(os.path.join(output_dir, "features.csv"))

# Spalten mit Koordinaten erkennen
coordinate_columns = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]

# Normalisierung der Koordinaten durchführen
for col in coordinate_columns:
    if col.endswith('_x'):
        df[col] = df[col] / df['image_width']
    elif col.endswith('_y'):
        df[col] = df[col] / df['image_height']

# Normalisierung der Abstände durchführen
if 'shoulder_width' in df.columns:
    df['shoulder_width'] = df['shoulder_width'] / df['image_width']

if 'hip_width' in df.columns:
    df['hip_width'] = df['hip_width'] / df['image_width']

if 'arm_length_left' in df.columns:
    df['arm_length_left'] = df['arm_length_left'] / df['image_height']

if 'arm_length_right' in df.columns:
    df['arm_length_right'] = df['arm_length_right'] / df['image_height']

# Die normalisierte CSV speichern
df.to_csv(os.path.join(output_dir, "normalized_data.csv"), index=False)