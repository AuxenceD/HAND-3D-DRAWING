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
        
        # Facteurs de sensibilité pour la rotation
        self.rotation_sensitivity = 4.0  # Augmenté pour une rotation plus rapide

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
        
        # Pour réduire le nombre de points (optimisation)
        self.min_point_distance = 0.05  # Distance minimale entre les points
        self.last_point = None

    def start_new_line(self, point):
        """Commence une nouvelle ligne"""
        self.current_line = [point]
        self.lines.append(self.current_line)
        self.is_drawing = True
        self.last_point = point

    def add_point(self, point):
        """Ajoute un point à la ligne en cours si la distance est suffisante"""
        if self.is_drawing and self.current_line is not None:
            # Vérifier la distance avec le dernier point pour réduire le nombre de points
            if self.last_point is None or np.linalg.norm(np.array(point) - np.array(self.last_point)) > self.min_point_distance:
                self.current_line.append(point)
                self.last_point = point

    def set_midpoint(self, point):
        """Définit le point milieu entre pouce et index"""
        self.midpoint = point

    def stop_drawing(self):
        """Arrête le dessin de la ligne en cours"""
        self.is_drawing = False
        self.current_line = None
        self.last_navigation_point = None
        self.last_point = None

    def navigate(self, current_point):
        """Gère la navigation dans l'environnement 3D"""
        if self.last_navigation_point is not None:
            # Calculer le déplacement avec un facteur d'accélération plus élevé
            dx = (current_point[0] - self.last_navigation_point[0]) * self.camera.rotation_sensitivity
            dy = (current_point[1] - self.last_navigation_point[1]) * self.camera.rotation_sensitivity
            
            # Ajuster la rotation de la caméra (en inversant dy pour correspondre au mouvement naturel)
            self.camera.rotation_y += dx
            self.camera.rotation_x -= dy  # Inversé pour correspondre au mouvement naturel
        
        # Mettre à jour le dernier point de navigation
        self.last_navigation_point = current_point

    def draw(self):
        """Dessine toutes les lignes en 3D et le point milieu"""
        # Utiliser les display lists pour optimiser le rendu
        glColor3f(0, 1, 1)  # Couleur cyan
        
        # Dessin des lignes existantes
        for line in self.lines:
            if len(line) > 1:
                glBegin(GL_LINE_STRIP)
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
        self.last_navigation_point = None
        self.last_point = None

def setup_opengl():
    """Configuration détaillée d'OpenGL"""
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    
    # Activer des optimisations OpenGL
    glEnable(GL_CULL_FACE)
    glShadeModel(GL_SMOOTH)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    
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
        
        # Réduire la résolution de la webcam pour optimiser les performances
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialisation de MediaPipe Hands avec des paramètres optimisés
        hands = mp_hands.Hands(
            max_num_hands=2,  # Détecter deux mains
            min_detection_confidence=0.6,  # Légèrement réduit pour plus de fluidité
            min_tracking_confidence=0.6,
            model_complexity=0  # Utiliser le modèle le plus léger
        )
        
        # Création de la caméra et de l'objet de dessin
        camera = Camera()
        drawing = Drawing3D(camera)
        
        # Variables pour limiter le traitement des frames
        skip_frames = 0
        process_every_n_frames = 2  # Traiter une frame sur 2
        
        # Boucle principale
        clock = pygame.time.Clock()
        running = True
        
        # Pour le contrôle au clavier
        keys_down = set()
        
        while running:
            # Gestion des événements Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    keys_down.add(event.key)
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        drawing.clear()
                elif event.type == pygame.KEYUP:
                    if event.key in keys_down:
                        keys_down.remove(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Molette haut (zoom in)
                        camera.zoom += 1.0  # Augmenté pour un zoom plus rapide
                    elif event.button == 5:  # Molette bas (zoom out)
                        camera.zoom -= 1.0  # Augmenté pour un zoom plus rapide
            
            # Contrôle par clavier (alternatives pour la rotation de caméra)
            if pygame.K_LEFT in keys_down:
                camera.rotation_y -= 2.0
            if pygame.K_RIGHT in keys_down:
                camera.rotation_y += 2.0
            if pygame.K_UP in keys_down:
                camera.rotation_x += 2.0
            if pygame.K_DOWN in keys_down:
                camera.rotation_x -= 2.0
            
            # Capture de la frame vidéo
            ret, frame = cap.read()
            if not ret:
                print("Erreur : Impossible de capturer la frame")
                break
            
            # Traitement d'une frame sur N pour économiser des ressources
            process_frame = (skip_frames % process_every_n_frames) == 0
            skip_frames += 1
            
            # Variables pour stocker les points de navigation et de dessin
            navigation_point = None
            drawing_point = None
            
            # Traitement de la détection de mains seulement si nécessaire
            if process_frame:
                # Retourner l'image horizontalement
                frame = cv2.flip(frame, 1)
                
                # Convertir l'image en RGB pour MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
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
                        
                        # Point de navigation
                        navigation_point = np.array([
                            (thumb_tip.x - 0.5) * 10,  # X
                            -(thumb_tip.y - 0.5) * 10,  # Y inversé
                            camera.projection_distance  # Toujours à la même distance du point de vue
                        ])
                        
                        # Rotation du point selon la vue actuelle
                        rotated_navigation_point = rotate_point(
                            navigation_point, 
                            -camera.rotation_x, 
                            -camera.rotation_y
                        )
                        
                        # Convertir en liste pour OpenGL
                        navigation_point = list(rotated_navigation_point)
                        
                        # Seuil de pincement pour la navigation
                        PINCH_THRESHOLD = 0.06  # Légèrement augmenté
                        
                        # Naviguer si pincement
                        if distance < PINCH_THRESHOLD:
                            drawing.navigate(navigation_point)
                    
                    # Gestion du dessin (main droite)
                    if right_hand:
                        # Obtenir les coordonnées du pouce et de l'index
                        thumb_tip = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index_tip = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        
                        # Calculer le point milieu entre le pouce et l'index
                        midpoint = np.array([
                            (thumb_tip.x - 0.5) * 10,  # X
                            -(thumb_tip.y - 0.5) * 10,  # Y inversé
                            camera.projection_distance  # Toujours à la même distance du point de vue
                        ])
                        
                        # Rotation du point milieu selon la vue actuelle
                        rotated_midpoint = rotate_point(
                            midpoint, 
                            -camera.rotation_x, 
                            -camera.rotation_y
                        )
                        
                        # Convertir en liste pour OpenGL
                        drawing_point = list(rotated_midpoint)
                        
                        # Définir le point milieu qui sera toujours visible
                        drawing.set_midpoint(drawing_point)
                        
                        # Calculer la distance entre le pouce et l'index
                        distance = math.sqrt(
                            (thumb_tip.x - index_tip.x)**2 + 
                            (thumb_tip.y - index_tip.y)**2 + 
                            (thumb_tip.z - index_tip.z)**2
                        )
                        
                        # Seuil de pincement pour le dessin
                        PINCH_THRESHOLD = 0.06  # Légèrement augmenté
                        
                        # Commencer/arrêter de dessiner
                        if distance < PINCH_THRESHOLD:
                            if not drawing.is_drawing:
                                drawing.start_new_line(drawing_point)
                            else:
                                drawing.add_point(drawing_point)
                        elif drawing.is_drawing:
                            drawing.stop_drawing()
            
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
            
            # Dessiner les lignes et le point milieu
            drawing.draw()
            
            # Mettre à jour l'affichage
            pygame.display.flip()
            
            # Limiter à 60 FPS
            clock.tick(60)
            
            # Afficher les FPS actuels dans la console (optionnel, pour le débogage)
            # print(f"FPS: {clock.get_fps():.1f}", end="\r")
    
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