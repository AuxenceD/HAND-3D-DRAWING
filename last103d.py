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

class AirCanvasCamera:
    def __init__(self):
        # Position et orientation de la caméra
        self.position = np.array([0.0, 0.0, -10.0])  # Position initiale de la caméra
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        
        # Sensibilité des mouvements et rotations
        self.move_sensitivity = 0.05
        self.rotation_sensitivity = 0.8  # Augmenté pour des rotations plus fluides
    
    def move(self, dx, dy, dz):
        """Déplace la caméra dans l'espace 3D"""
        # Créer la matrice de rotation
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(self.rotation_x)), -np.sin(np.radians(self.rotation_x))],
            [0, np.sin(np.radians(self.rotation_x)), np.cos(np.radians(self.rotation_x))]
        ])
        
        Ry = np.array([
            [np.cos(np.radians(self.rotation_y)), 0, np.sin(np.radians(self.rotation_y))],
            [0, 1, 0],
            [-np.sin(np.radians(self.rotation_y)), 0, np.cos(np.radians(self.rotation_y))]
        ])
        
        # Créer le vecteur de mouvement
        movement = np.dot(Ry, np.dot(Rx, np.array([dx, dy, dz])))
        
        # Mettre à jour la position
        self.position += movement * self.move_sensitivity

class Drawing3D:
    def __init__(self, camera):
        self.camera = camera
        self.lines = []  # Liste de lignes (chaque ligne = liste de points)
        self.current_line = None
        self.is_drawing = False
        self.midpoint = None
        
        # Pour la navigation de caméra
        self.last_hand_position = None
    
    def start_new_line(self, point):
        """Commence une nouvelle ligne"""
        self.current_line = [point]
        self.lines.append(self.current_line)
        self.is_drawing = True
    
    def add_point(self, point):
        """Ajoute un point à la ligne en cours"""
        if self.is_drawing and self.current_line is not None:
            # N'ajouter le point que s'il est suffisamment éloigné du dernier point
            if not self.current_line or np.linalg.norm(np.array(point) - np.array(self.current_line[-1])) > 0.1:
                self.current_line.append(point)
    
    def set_midpoint(self, point):
        """Définit le point milieu entre pouce et index"""
        self.midpoint = point
    
    def stop_drawing(self):
        """Arrête le dessin de la ligne en cours"""
        self.is_drawing = False
        self.current_line = None
    
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
            glPointSize(10)
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
        self.last_hand_position = None

def setup_opengl():
    """Configuration détaillée d'OpenGL"""
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def transform_hand_point(landmark, camera):
    """Transformer le point de la main en tenant compte de l'orientation de la caméra"""
    # Convertir la position de la main en coordonnées 3D mondiales
    transformed_point = np.array([
        (landmark.x - 0.5) * 10,   # X
        -(landmark.y - 0.5) * 10,  # Y (inversé)
        landmark.z * 10            # Z profondeur
    ])
    
    # Rotation du point selon l'orientation de la caméra
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(-camera.rotation_x)), -np.sin(np.radians(-camera.rotation_x))],
        [0, np.sin(np.radians(-camera.rotation_x)), np.cos(np.radians(-camera.rotation_x))]
    ])
    
    Ry = np.array([
        [np.cos(np.radians(-camera.rotation_y)), 0, np.sin(np.radians(-camera.rotation_y))],
        [0, 1, 0],
        [-np.sin(np.radians(-camera.rotation_y)), 0, np.cos(np.radians(-camera.rotation_y))]
    ])
    
    # Appliquer les rotations
    rotated_point = np.dot(Ry, np.dot(Rx, transformed_point))
    
    return list(rotated_point)

def main():
    try:
        # Initialisation de Pygame
        pygame.init()
        
        # Créer la fenêtre avec des flags spécifiques
        display = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | HWSURFACE)
        pygame.display.set_caption("Canvas 3D Avancé")
        
        # Configuration OpenGL
        setup_opengl()
        
        # Initialisation de la capture vidéo
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erreur : Impossible d'ouvrir la caméra")
            return
        
        # Initialisation de MediaPipe Hands
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Création de la caméra et de l'objet de dessin
        camera = AirCanvasCamera()
        drawing = Drawing3D(camera)
        
        # Variables pour suivre l'état des pincements
        left_hand_pinched = False
        right_hand_pinched = False
        
        # Boucle principale
        clock = pygame.time.Clock()
        running = True
        
        # Pour le contrôle au clavier
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
                    elif event.key == pygame.K_r:  # Réinitialiser la caméra
                        camera.position = np.array([0.0, 0.0, -10.0])
                        camera.rotation_x = 0.0
                        camera.rotation_y = 0.0
                elif event.type == pygame.KEYUP:
                    if event.key in keys_pressed:
                        keys_pressed[event.key] = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Molette haut (zoom in)
                        camera.position[2] += 1.0
                    elif event.button == 5:  # Molette bas (zoom out)
                        camera.position[2] -= 1.0
            
            # Contrôle au clavier pour les déplacements
            if pygame.K_LEFT in keys_pressed and keys_pressed[pygame.K_LEFT]:
                camera.move(-0.2, 0, 0)
            if pygame.K_RIGHT in keys_pressed and keys_pressed[pygame.K_RIGHT]:
                camera.move(0.2, 0, 0)
            if pygame.K_UP in keys_pressed and keys_pressed[pygame.K_UP]:
                camera.move(0, 0.2, 0)
            if pygame.K_DOWN in keys_pressed and keys_pressed[pygame.K_DOWN]:
                camera.move(0, -0.2, 0)
            if pygame.K_w in keys_pressed and keys_pressed[pygame.K_w]:
                camera.move(0, 0, 0.2)
            if pygame.K_s in keys_pressed and keys_pressed[pygame.K_s]:
                camera.move(0, 0, -0.2)
            if pygame.K_a in keys_pressed and keys_pressed[pygame.K_a]:
                camera.rotation_y -= 2.0
            if pygame.K_d in keys_pressed and keys_pressed[pygame.K_d]:
                camera.rotation_y += 2.0
            
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
            
            # Appliquer les transformations de caméra
            glTranslatef(*camera.position)
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
            
            # Si des mains sont détectées
            if results.multi_hand_landmarks:
                # Séparer les mains gauche et droite
                left_hand = None
                right_hand = None
                for idx, hand_label in enumerate(results.multi_handedness):
                    if hand_label.classification[0].label == "Left":
                        left_hand = results.multi_hand_landmarks[idx]
                    else:
                        right_hand = results.multi_hand_landmarks[idx]
                
                # Gestion de la navigation (main gauche)
                if left_hand:
                    thumb_tip = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Calculer la distance entre le pouce et l'index
                    distance = math.sqrt(
                        (thumb_tip.x - index_tip.x)**2 + 
                        (thumb_tip.y - index_tip.y)**2 + 
                        (thumb_tip.z - index_tip.z)**2
                    )
                    
                    # Transformer le point de navigation
                    navigation_point = transform_hand_point(thumb_tip, camera)
                    
                    # Seuil de pincement pour la navigation
                    PINCH_THRESHOLD = 0.05
                    
                    # Navigation avec geste de pincement
                    currently_pinched = distance < PINCH_THRESHOLD
                    
                    if currently_pinched:
                        # Rotation de caméra
                        dx = (thumb_tip.x - 0.5) * camera.rotation_sensitivity
                        dy = (thumb_tip.y - 0.5) * camera.rotation_sensitivity
                        
                        # Utiliser la position absolue pour la rotation
                        target_rotation_y = dx * 360  # Plage complète de rotation horizontale
                        target_rotation_x = dy * 180  # Plage complète de rotation verticale
                        
                        # Appliquer progressivement la rotation (interpolation)
                        camera.rotation_y += (target_rotation_y - camera.rotation_y) * 0.1
                        camera.rotation_x += (target_rotation_x - camera.rotation_x) * 0.1
                    
                    # Mettre à jour l'état de pincement
                    left_hand_pinched = currently_pinched
                
                # Gestion du dessin (main droite)
                if right_hand:
                    thumb_tip = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Calculer le point milieu entre le pouce et l'index
                    midpoint_x = (thumb_tip.x + index_tip.x) / 2
                    midpoint_y = (thumb_tip.y + index_tip.y) / 2
                    midpoint_z = (thumb_tip.z + index_tip.z) / 2
                    
                    # Créer un objet landmark-like pour le point milieu
                    class MidpointLandmark:
                        def __init__(self, x, y, z):
                            self.x = x
                            self.y = y
                            self.z = z
                    
                    midpoint = MidpointLandmark(midpoint_x, midpoint_y, midpoint_z)
                    
                    # Calculer la distance entre le pouce et l'index
                    distance = math.sqrt(
                        (thumb_tip.x - index_tip.x)**2 + 
                        (thumb_tip.y - index_tip.y)**2 + 
                        (thumb_tip.z - index_tip.z)**2
                    )
                    
                    # Transformer le point de dessin
                    drawing_point = transform_hand_point(midpoint, camera)
                    
                    # Définir le point milieu pour la visualisation
                    drawing.set_midpoint(drawing_point)
                    
                    # Seuil de pincement pour le dessin
                    PINCH_THRESHOLD = 0.05
                    
                    # État de pincement actuel
                    currently_pinched = distance < PINCH_THRESHOLD
                    
                    # Commencer/arrêter de dessiner
                    if currently_pinched and not right_hand_pinched:
                        drawing.start_new_line(drawing_point)
                    elif currently_pinched:
                        drawing.add_point(drawing_point)
                    elif not currently_pinched and right_hand_pinched:
                        drawing.stop_drawing()
                    
                    # Mettre à jour l'état de pincement
                    right_hand_pinched = currently_pinched
            
            # Dessiner les lignes et le point milieu
            drawing.draw()
            
            # Mettre à jour l'affichage
            pygame.display.flip()
            
            # Limiter à 60 FPS
            clock.tick(60)
            
            # Option : afficher le FPS
            # pygame.display.set_caption(f"Canvas 3D Avancé - {int(clock.get_fps())} FPS")
    
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