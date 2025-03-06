import cv2
import mediapipe as mp
import pygame
import sys
import traceback
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# Dimensions de la fenêtre
WIDTH, HEIGHT = 1280, 720

# Configuration MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class Camera:
    def __init__(self):
        # Position et orientation de la caméra
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = -10
        
        # Plan de projection
        self.projection_distance = 5  # Distance du plan de projection
        self.projection_matrix = np.eye(4)
        
        # Facteur de sensibilité pour la rotation
        self.rotation_speed = 10.0  # Sensibilité encore plus élevée pour des rotations rapides

def rotate_point(point, rx, ry):
    """Faire pivoter un point selon les rotations de caméra"""
    # Conversion des rotations en radians
    rx_rad = math.radians(rx)
    ry_rad = math.radians(ry)
    
    # Matrice de rotation
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx_rad), -math.sin(rx_rad)],
        [0, math.sin(rx_rad), math.cos(rx_rad)]
    ])
    
    Ry = np.array([
        [math.cos(ry_rad), 0, math.sin(ry_rad)],
        [0, 1, 0],
        [-math.sin(ry_rad), 0, math.cos(ry_rad)]
    ])
    
    # Appliquer les rotations
    point_rotated = np.dot(Ry, np.dot(Rx, point))
    
    return point_rotated

class Drawing3D:
    def __init__(self, camera):
        self.lines = []  # Liste de lignes (chaque ligne = liste de points)
        self.current_line = None
        self.is_drawing = False
        self.camera = camera
        self.midpoint = None  # Point milieu entre pouce et index
        self.last_navigation_point = None  # Dernier point de navigation
        self.navigation_active = False  # Indique si la navigation est active
        
        # Paramètres pour le lissage des mouvements
        self.smoothing_factor = 0.3  # Facteur de lissage réduit pour plus de réactivité
        self.prev_dx = 0
        self.prev_dy = 0

    def start_new_line(self, point):
        """Commence une nouvelle ligne"""
        self.current_line = [point]
        self.lines.append(self.current_line)
        self.is_drawing = True

    def add_point(self, point):
        """Ajoute un point à la ligne en cours"""
        if self.is_drawing and self.current_line is not None:
            self.current_line.append(point)

    def set_midpoint(self, point):
        """Définit le point milieu entre pouce et index"""
        self.midpoint = point

    def stop_drawing(self):
        """Arrête le dessin de la ligne en cours"""
        self.is_drawing = False
        self.current_line = None

    def start_navigation(self, point):
        """Débute une nouvelle session de navigation"""
        self.navigation_active = True
        self.last_navigation_point = np.array(point).copy()
        self.prev_dx = 0
        self.prev_dy = 0
        
    def stop_navigation(self):
        """Arrête la navigation"""
        self.navigation_active = False
        self.last_navigation_point = None

    def navigate(self, current_point):
        """Gère la navigation dans l'environnement 3D avec un mouvement lissé et SANS RESTRICTION"""
        if not self.navigation_active or self.last_navigation_point is None:
            return
            
        # Convertir le point actuel en array numpy
        current_point_array = np.array(current_point)
        
        # Calculer le déplacement brut
        raw_dx = (current_point_array[0] - self.last_navigation_point[0]) * self.camera.rotation_speed
        raw_dy = (current_point_array[1] - self.last_navigation_point[1]) * self.camera.rotation_speed
        
        # Appliquer le lissage pour éviter les mouvements brusques
        dx = self.prev_dx * self.smoothing_factor + raw_dx * (1 - self.smoothing_factor)
        dy = self.prev_dy * self.smoothing_factor + raw_dy * (1 - self.smoothing_factor)
        
        # Enregistrer pour le prochain frame
        self.prev_dx = dx
        self.prev_dy = dy
        
        # Mettre à jour la rotation de la caméra SANS AUCUNE RESTRICTION
        self.camera.rotation_y += dx
        self.camera.rotation_x += dy  # Non inversé pour une sensation plus naturelle
        
        # Mettre à jour le dernier point de navigation
        self.last_navigation_point = current_point_array.copy()

    def draw(self):
        """Dessine toutes les lignes en 3D et le point milieu"""
        # Dessin des lignes existantes
        for line in self.lines:
            if len(line) > 1:
                glBegin(GL_LINE_STRIP)
                glColor3f(0, 1, 1)  # Couleur cyan
                for point in line:
                    glVertex3f(*point)
                glEnd()
        
        # Dessin du point milieu
        if self.midpoint is not None:
            glPointSize(10)  # Taille du point
            glBegin(GL_POINTS)
            glColor3f(1, 0, 0)  # Couleur rouge pour le point milieu
            glVertex3f(*self.midpoint)
            glEnd()

    def clear(self):
        """Efface tous les dessins"""
        self.lines = []
        self.current_line = None
        self.is_drawing = False
        self.midpoint = None
        self.navigation_active = False
        self.last_navigation_point = None

def setup_opengl():
    """Configuration détaillée d'OpenGL"""
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def main():
    try:
        # Initialisation de Pygame
        pygame.init()
        
        # Créer la fenêtre avec des flags spécifiques
        display = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | HWSURFACE)
        pygame.display.set_caption("Dessin 3D sur Plan Dynamique")
        
        # Configuration OpenGL
        setup_opengl()
        
        # Initialisation de la capture vidéo
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erreur : Impossible d'ouvrir la caméra")
            return
        
        # Initialisation de MediaPipe Hands
        hands = mp_hands.Hands(
            max_num_hands=2,  # Détecter deux mains
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Création de la caméra et de l'objet de dessin
        camera = Camera()
        drawing = Drawing3D(camera)
        
        # Variables pour suivre l'état de pincement des mains
        left_hand_pinched = False
        right_hand_pinched = False
        
        # Boucle principale
        clock = pygame.time.Clock()
        running = True
        
        # Contrôle clavier alternatif pour les tests
        keys_pressed = {}
        
        while running:
            # Gestion des événements Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    keys_pressed[event.key] = True
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        drawing.clear()
                    elif event.key == pygame.K_r:  # Réinitialiser la rotation de la caméra
                        camera.rotation_x = 0
                        camera.rotation_y = 0
                elif event.type == pygame.KEYUP:
                    if event.key in keys_pressed:
                        keys_pressed[event.key] = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Molette haut (zoom in)
                        camera.zoom += 1.0
                    elif event.button == 5:  # Molette bas (zoom out)
                        camera.zoom -= 1.0
            
            # Contrôle clavier alternatif pour les rotations
            if pygame.K_LEFT in keys_pressed and keys_pressed[pygame.K_LEFT]:
                camera.rotation_y -= 3.0
            if pygame.K_RIGHT in keys_pressed and keys_pressed[pygame.K_RIGHT]:
                camera.rotation_y += 3.0
            if pygame.K_UP in keys_pressed and keys_pressed[pygame.K_UP]:
                camera.rotation_x += 3.0
            if pygame.K_DOWN in keys_pressed and keys_pressed[pygame.K_DOWN]:
                camera.rotation_x -= 3.0
            
            # Capture de la frame vidéo
            ret, frame = cap.read()
            if not ret:
                print("Erreur : Impossible de capturer la frame")
                break
            
            # Retourner l'image horizontalement
            frame = cv2.flip(frame, 1)
            
            # Convertir l'image en RGB pour MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # Réinitialiser la zone de dessin
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            glTranslatef(0, 0, camera.zoom)
            
            # Rotation de la vue
            glRotatef(camera.rotation_x, 1, 0, 0)
            glRotatef(camera.rotation_y, 0, 1, 0)
            
            # Dessin des axes 3D pour repère
            glBegin(GL_LINES)
            # Axe X (Rouge)
            glColor3f(1, 0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(5, 0, 0)
            
            # Axe Y (Vert)
            glColor3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 5, 0)
            
            # Axe Z (Bleu)
            glColor3f(0, 0, 1)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 5)
            glEnd()
            
            # Variables pour stocker les points de navigation et de dessin
            navigation_point = None
            drawing_point = None
            
            # Si des mains sont détectées
            if results.multi_hand_landmarks:
                # Séparer les mains gauche et droite
                left_hand = None
                right_hand = None
                
                for idx, hand_handedness in enumerate(results.multi_handedness):
                    if hand_handedness.classification[0].label == "Left":
                        left_hand = results.multi_hand_landmarks[idx]
                    else:
                        right_hand = results.multi_hand_landmarks[idx]
                
                # Gestion de la navigation (main gauche)
                if left_hand:
                    # Obtenir les coordonnées du pouce et de l'index
                    thumb_tip = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Calculer la distance entre le pouce et l'index
                    distance = math.sqrt(
                        (thumb_tip.x - index_tip.x)**2 + 
                        (thumb_tip.y - index_tip.y)**2 + 
                        (thumb_tip.z - index_tip.z)**2
                    )
                    
                    # Point de navigation (point milieu entre pouce et index)
                    midpoint_x = (thumb_tip.x + index_tip.x) / 2
                    midpoint_y = (thumb_tip.y + index_tip.y) / 2
                    
                    navigation_point = np.array([
                        (midpoint_x - 0.5) * 10,  # X
                        -(midpoint_y - 0.5) * 10,  # Y inversé
                        camera.projection_distance  # Toujours à la même distance du point de vue
                    ])
                    
                    # Rotation du point selon la vue actuelle
                    rotated_navigation_point = rotate_point(
                        navigation_point, 
                        -camera.rotation_x, 
                        -camera.rotation_y
                    )
                    
                    # Convertir en liste pour OpenGL
                    navigation_point = rotated_navigation_point.tolist()
                    
                    # Seuil de pincement pour la navigation
                    PINCH_THRESHOLD = 0.05
                    
                    # Gérer l'état de pincement
                    currently_pinched = distance < PINCH_THRESHOLD
                    
                    # Si le pincement vient de commencer
                    if currently_pinched and not left_hand_pinched:
                        drawing.start_navigation(navigation_point)
                    
                    # Si le pincement continue
                    if currently_pinched and left_hand_pinched:
                        drawing.navigate(navigation_point)
                    
                    # Si le pincement s'est terminé
                    if not currently_pinched and left_hand_pinched:
                        drawing.stop_navigation()
                    
                    # Mettre à jour l'état
                    left_hand_pinched = currently_pinched
                
                # Gestion du dessin (main droite)
                if right_hand:
                    # Obtenir les coordonnées du pouce et de l'index
                    thumb_tip = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Calculer le point milieu entre le pouce et l'index
                    midpoint_x = (thumb_tip.x + index_tip.x) / 2
                    midpoint_y = (thumb_tip.y + index_tip.y) / 2
                    
                    midpoint = np.array([
                        (midpoint_x - 0.5) * 10,  # X
                        -(midpoint_y - 0.5) * 10,  # Y inversé
                        camera.projection_distance  # Toujours à la même distance du point de vue
                    ])
                    
                    # Rotation du point milieu selon la vue actuelle
                    rotated_midpoint = rotate_point(
                        midpoint, 
                        -camera.rotation_x, 
                        -camera.rotation_y
                    )
                    
                    # Convertir en liste pour OpenGL
                    drawing_point = rotated_midpoint.tolist()
                    
                    # Définir le point milieu qui sera toujours visible
                    drawing.set_midpoint(drawing_point)
                    
                    # Calculer la distance entre le pouce et l'index
                    distance = math.sqrt(
                        (thumb_tip.x - index_tip.x)**2 + 
                        (thumb_tip.y - index_tip.y)**2 + 
                        (thumb_tip.z - index_tip.z)**2
                    )
                    
                    # Seuil de pincement pour le dessin
                    PINCH_THRESHOLD = 0.05
                    
                    # Gérer l'état de pincement
                    currently_pinched = distance < PINCH_THRESHOLD
                    
                    # Commencer/arrêter de dessiner
                    if currently_pinched and not right_hand_pinched:
                        drawing.start_new_line(drawing_point)
                    elif currently_pinched and right_hand_pinched:
                        drawing.add_point(drawing_point)
                    elif not currently_pinched and right_hand_pinched:
                        drawing.stop_drawing()
                    
                    # Mettre à jour l'état
                    right_hand_pinched = currently_pinched
            
            # Dessiner les lignes et le point milieu
            drawing.draw()
            
            # Mettre à jour l'affichage
            pygame.display.flip()
            
            # Limiter à 60 FPS
            clock.tick(60)
    
    except Exception as e:
        print(f"Erreur fatale : {e}")
        traceback.print_exc()
    
    finally:
        # Libérer les ressources
        try:
            cap.release()
        except:
            pass
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()