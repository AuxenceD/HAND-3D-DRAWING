import cv2
import mediapipe as mp
import numpy as np
import json
import time

# Initialiser MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Seuil de distance pour considérer un "pincement"
PINCH_THRESHOLD = 0.05  # Ajuster selon la sensibilité souhaitée

# Fichier pour stocker les coordonnées des points
output_file = "hand_coordinates.txt"

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

# Variables pour le dessin
is_drawing = False
drawing_points = []
current_line = []

print("Démarrage de la capture vidéo...")
print("Pince ton pouce et ton index pour commencer à dessiner")
print("Sépare-les pour arrêter de dessiner")
print("Appuie sur 'r' pour réinitialiser le dessin")
print("Appuie sur 's' pour sauvegarder le dessin")
print("Appuie sur 'q' pour quitter")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Échec de la lecture de la vidéo")
        break
        
    # Retourner l'image horizontalement pour un effet "miroir"
    image = cv2.flip(image, 1)
    
    # Convertir en RGB pour MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Dessiner les repères de la main
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner les connecteurs et landmarks de la main
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Obtenir les coordonnées du bout de l'index (point 8)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Obtenir les coordonnées du bout du pouce (point 4)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Calculer la distance entre les deux points
            distance = np.sqrt((index_tip.x - thumb_tip.x)**2 + 
                              (index_tip.y - thumb_tip.y)**2 + 
                              (index_tip.z - thumb_tip.z)**2)
            
            # Calculer le point milieu
            mid_x = (index_tip.x + thumb_tip.x) / 2
            mid_y = (index_tip.y + thumb_tip.y) / 2
            mid_z = (index_tip.z + thumb_tip.z) / 2
            
            # Convertir en coordonnées de pixels pour affichage
            h, w, c = image.shape
            mid_px = int(mid_x * w)
            mid_py = int(mid_y * h)
            
            # Gérer le dessin
            if distance < PINCH_THRESHOLD:
                if not is_drawing:
                    # Commencer une nouvelle ligne
                    print("Début du dessin")
                    is_drawing = True
                    current_line = []
                
                # Ajouter le point milieu à la ligne actuelle
                current_line.append([mid_x, mid_y, mid_z])
                
                # Dessiner un cercle rouge pour indiquer qu'on dessine
                cv2.circle(image, (mid_px, mid_py), 10, (0, 0, 255), -1)
            else:
                if is_drawing:
                    # Arrêter la ligne actuelle
                    print("Fin du dessin")
                    is_drawing = False
                    if len(current_line) > 1:  # Ne pas ajouter des lignes vides
                        drawing_points.append(current_line)
                
                # Dessiner un cercle vert pour indiquer qu'on ne dessine pas
                if 'mid_px' in locals() and 'mid_py' in locals():
                    cv2.circle(image, (mid_px, mid_py), 10, (0, 255, 0), -1)
            
            # Afficher la distance
            cv2.putText(image, f"Distance: {distance:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if distance < PINCH_THRESHOLD else (0, 255, 0), 2)
            
            # Afficher les coordonnées
            cv2.putText(image, f"X: {mid_x:.2f}, Y: {mid_y:.2f}, Z: {mid_z:.2f}", 
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Dessiner les lignes déjà capturées (historique)
    for line in drawing_points:
        if len(line) > 1:
            points = np.array([(int(p[0] * w), int(p[1] * h)) for p in line], np.int32)
            cv2.polylines(image, [points], False, (255, 0, 255), 2)
    
    # Dessiner la ligne en cours
    if is_drawing and len(current_line) > 1:
        points = np.array([(int(p[0] * w), int(p[1] * h)) for p in current_line], np.int32)
        cv2.polylines(image, [points], False, (0, 255, 255), 2)
    
    # Afficher l'image
    cv2.imshow('MediaPipe Hands', image)
    
    # Contrôles clavier
    key = cv2.waitKey(5) & 0xFF
    
    if key == ord('q'):  # Quitter
        break
    elif key == ord('r'):  # Réinitialiser
        drawing_points = []
        current_line = []
        is_drawing = False
        print("Dessin réinitialisé")
    elif key == ord('s'):  # Sauvegarder
        # Ajouter la ligne en cours si on est en train de dessiner
        if is_drawing and len(current_line) > 1:
            drawing_points.append(current_line)
        
        # Préparer les données pour la sauvegarde
        all_points = []
        for line in drawing_points:
            all_points.extend(line)
            # Ajouter un point "null" pour séparer les lignes
            if len(line) > 0:
                all_points.append(None)
        
        # Filtrer les None pour le format JSON
        json_points = [p for p in all_points if p is not None]
        
        # Sauvegarder
        with open(output_file, 'w') as f:
            json.dump({"points": json_points, "lines": drawing_points}, f)
        print(f"Dessin sauvegardé dans {output_file}")

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()