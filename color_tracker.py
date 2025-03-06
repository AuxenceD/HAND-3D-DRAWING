import cv2
import numpy as np
import json
import time

# Fonction pour rien faire (utilisée pour les trackbars)
def nothing(x):
    pass

# Crée une fenêtre pour les contrôles de couleur
cv2.namedWindow('Contrôles')
cv2.createTrackbar('H min', 'Contrôles', 0, 179, nothing)
cv2.createTrackbar('S min', 'Contrôles', 0, 255, nothing)
cv2.createTrackbar('V min', 'Contrôles', 0, 255, nothing)
cv2.createTrackbar('H max', 'Contrôles', 179, 179, nothing)
cv2.createTrackbar('S max', 'Contrôles', 255, 255, nothing)
cv2.createTrackbar('V max', 'Contrôles', 255, 255, nothing)

# Valeurs par défaut pour détecter du bleu
cv2.setTrackbarPos('H min', 'Contrôles', 90)  # Bleu commence vers 90
cv2.setTrackbarPos('S min', 'Contrôles', 50)
cv2.setTrackbarPos('V min', 'Contrôles', 50)
cv2.setTrackbarPos('H max', 'Contrôles', 130)  # Bleu finit vers 130
cv2.setTrackbarPos('S max', 'Contrôles', 255)
cv2.setTrackbarPos('V max', 'Contrôles', 255)

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

# Fichier pour stocker les coordonnées des points
output_file = "hand_coordinates.txt"

print("Démarrage de la capture vidéo...")
print("Ajuste les curseurs pour détecter la couleur que tu veux suivre")
print("Appuie sur 'q' pour quitter")
print("Appuie sur 'c' pour capturer un point")

# Pour stocker les points capturés
captured_points = []

# Pour estimer la profondeur
min_size = 100  # Taille de référence pour Z (à ajuster)
max_size = 500  # Taille max quand l'objet est très proche

while True:
    ret, frame = cap.read()
    if not ret:
        print("Impossible de lire la vidéo")
        break
    
    # Redimensionne l'image pour qu'elle soit plus rapide à traiter
    frame = cv2.resize(frame, (640, 480))
    
    # Convertir en HSV (plus facile pour la détection de couleur)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Obtenir les valeurs actuelles des trackbars
    h_min = cv2.getTrackbarPos('H min', 'Contrôles')
    s_min = cv2.getTrackbarPos('S min', 'Contrôles')
    v_min = cv2.getTrackbarPos('V min', 'Contrôles')
    h_max = cv2.getTrackbarPos('H max', 'Contrôles')
    s_max = cv2.getTrackbarPos('S max', 'Contrôles')
    v_max = cv2.getTrackbarPos('V max', 'Contrôles')
    
    # Créer une plage de couleur à détecter
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    # Créer un masque pour isoler la couleur
    mask = cv2.inRange(hsv, lower, upper)
    
    # Éliminer le bruit avec une opération d'ouverture
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Trouver les contours dans le masque
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Si des contours sont trouvés
    if contours:
        # Trouver le plus grand contour (probablement notre objet)
        c = max(contours, key=cv2.contourArea)
        
        # Calculer la zone et le centre du contour
        area = cv2.contourArea(c)
        
        if area > 100:  # Ignorer les petits contours (bruit)
            # Obtenir un rectangle autour du contour
            x, y, w, h = cv2.boundingRect(c)
            
            # Dessiner un rectangle autour de l'objet
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculer le centre
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Dessiner un cercle au centre
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Calculer les coordonnées normalisées (0-1)
            norm_x = center_x / frame.shape[1]
            norm_y = center_y / frame.shape[0]
            
            # Estimer la profondeur Z basée sur la taille de l'objet
            # Plus l'objet est petit, plus il est loin
            norm_z = 1 - min(1, max(0, (area - min_size) / (max_size - min_size)))
            
            # Afficher les coordonnées
            text = f"X: {norm_x:.2f}, Y: {norm_y:.2f}, Z: {norm_z:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    # Afficher le résultat
    cv2.imshow('Suivi Couleur', frame)
    cv2.imshow('Masque', mask)
    
    # Contrôle par clavier
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # Quitter
        break
    elif key == ord('c') and contours:  # Capturer un point
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        if area > 100:
            # Obtenir un rectangle autour du contour
            x, y, w, h = cv2.boundingRect(c)
            
            # Calculer le centre
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Calculer les coordonnées normalisées (0-1)
            norm_x = center_x / frame.shape[1]
            norm_y = center_y / frame.shape[0]
            
            # Estimer la profondeur Z basée sur la taille
            norm_z = 1 - min(1, max(0, (area - min_size) / (max_size - min_size)))
            
            # Ajouter le point à notre liste
            point = [norm_x, norm_y, norm_z]
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