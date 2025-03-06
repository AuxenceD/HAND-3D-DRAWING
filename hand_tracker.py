import cv2
import mediapipe as mp
import time
import json

# Initialiser MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)  # 0 est généralement la webcam intégrée

# Fichier pour stocker les coordonnées des points
output_file = "hand_coordinates.txt"

print("Démarrage de la capture vidéo...")
print("Appuyez sur 'q' pour quitter")
print("Appuyez sur 'c' pour capturer un point")

# Pour stocker les points capturés
captured_points = []

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Échec de la lecture de la vidéo")
        break

    # Convertir l'image en RGB (MediaPipe utilise RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Dessiner les repères de la main sur l'image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Point de l'index (pour dessiner)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convertir les coordonnées normalisées en coordonnées de pixels
            h, w, c = image.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Dessiner un cercle à la pointe de l'index
            cv2.circle(image, (cx, cy), 10, (0, 255, 0), -1)
            
            # Afficher les coordonnées 3D
            coords_text = f"X: {index_finger_tip.x:.2f}, Y: {index_finger_tip.y:.2f}, Z: {index_finger_tip.z:.2f}"
            cv2.putText(image, coords_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Afficher l'image
    cv2.imshow('MediaPipe Hands', image)
    
    # Contrôles clavier
    key = cv2.waitKey(5) & 0xFF
    
    if key == ord('q'):  # Quitter
        break
    elif key == ord('c') and results.multi_hand_landmarks:  # Capturer un point
        # Obtenir les coordonnées 3D du bout de l'index
        index_tip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        point = [index_tip.x, index_tip.y, index_tip.z]
        captured_points.append(point)
        print(f"Point capturé: {point}")
        
        # Enregistrer tous les points dans un fichier
        with open(output_file, 'w') as f:
            json.dump(captured_points, f)
        
        print(f"Total de points capturés: {len(captured_points)}")

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
print(f"Points capturés enregistrés dans {output_file}")