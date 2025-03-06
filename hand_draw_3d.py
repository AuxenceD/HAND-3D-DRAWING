import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import json
import os

# ================================================
# CONFIGURATION
# ================================================
# Dimensions des fenêtres
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Couleurs (R, G, B)
BACKGROUND_COLOR = (0, 0, 0)
LINE_COLOR = (0, 0.8, 1)
GRID_COLOR = (0.2, 0.2, 0.2)

# Paramètres du dessin
PINCH_THRESHOLD = 0.05  # Seuil pour détecter un pincement (distance)
LINE_THICKNESS = 3.0    # Épaisseur des lignes

# Configuration MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ================================================
# CLASSES
# ================================================
class Line3D:
    """Classe pour stocker une ligne 3D avec ses points et sa couleur"""
    def __init__(self, color=LINE_COLOR, thickness=LINE_THICKNESS):
        self.points = []
        self.color = color
        self.thickness = thickness

    def add_point(self, point):
        """Ajoute un point à la ligne"""
        self.points.append(point)

    def draw(self):
        """Dessine la ligne en OpenGL"""
        if len(self.points) < 2:
            return

        glColor3f(*self.color)
        glLineWidth(self.thickness)
        glBegin(GL_LINE_STRIP)
        for point in self.points:
            glVertex3f(*point)
        glEnd()

class Scene3D:
    """Classe pour gérer la scène 3D"""
    def __init__(self):
        self.lines = []
        self.current_line = None
        self.is_drawing = False
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = -5
        self.pan_x = 0
        self.pan_y = 0

    def start_new_line(self, color=LINE_COLOR, thickness=LINE_THICKNESS):
        """Commence une nouvelle ligne"""
        self.current_line = Line3D(color, thickness)
        self.lines.append(self.current_line)
        self.is_drawing = True

    def add_point(self, point):
        """Ajoute un point à la ligne en cours"""
        if self.is_drawing and self.current_line:
            self.current_line.add_point(point)

    def stop_drawing(self):
        """Arrête le dessin en cours"""
        self.is_drawing = False
        self.current_line = None

    def clear(self):
        """Efface toutes les lignes"""
        self.lines = []
        self.current_line = None
        self.is_drawing = False

    def save(self, filename="drawing_3d.json"):
        """Sauvegarde le dessin au format JSON"""
        data = {
            "lines": [
                {
                    "points": line.points,
                    "color": line.color,
                    "thickness": line.thickness
                }
                for line in self.lines
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Dessin sauvegardé dans {filename}")

    def load(self, filename="drawing_3d.json"):
        """Charge un dessin depuis un fichier JSON"""
        if not os.path.exists(filename):
            print(f"Fichier {filename} introuvable")
            return False
            
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.clear()
        for line_data in data["lines"]:
            line = Line3D(
                color=tuple(line_data["color"]),
                thickness=line_data["thickness"]
            )
            line.points = [tuple(p) for p in line_data["points"]]
            self.lines.append(line)
        
        print(f"Dessin chargé depuis {filename}")
        return True

    def draw(self):
        """Dessine toute la scène 3D"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Appliquer le zoom et la translation
        glTranslatef(self.pan_x, self.pan_y, self.zoom)
        
        # Appliquer les rotations
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        # Dessiner la grille de référence
        self.draw_grid()
        
        # Dessiner les axes
        self.draw_axes()
        
        # Dessiner toutes les lignes
        for line in self.lines:
            line.draw()

    def draw_grid(self, size=10, step=1):
        """Dessine une grille de référence"""
        glBegin(GL_LINES)
        glColor3f(*GRID_COLOR)
        
        # Lignes horizontales
        for i in range(-size, size+1, step):
            glVertex3f(-size, 0, i)
            glVertex3f(size, 0, i)
            glVertex3f(i, 0, -size)
            glVertex3f(i, 0, size)
            
        glEnd()

    def draw_axes(self, length=3):
        """Dessine les axes x, y, z"""
        glBegin(GL_LINES)
        # Axe X en rouge
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(length, 0, 0)
        
        # Axe Y en vert
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, length, 0)
        
        # Axe Z en bleu
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, length)
        glEnd()

# ================================================
# FONCTIONS PRINCIPALES
# ================================================
def setup_opengl(width, height):
    """Configure l'environnement OpenGL"""
    glClearColor(*BACKGROUND_COLOR, 1)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / height), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0, 0, -5)

def process_video_frame(frame, hands, scene):
    """Traite une image de la webcam pour la détection des mains"""
    # Convertir en RGB pour MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Variables pour stocker les infos de la main
    index_tip = None
    thumb_tip = None
    mid_point = None
    is_pinching = False
    
    # Si des mains sont détectées
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner les repères de la main
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Obtenir les coordonnées du bout de l'index
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Obtenir les coordonnées du bout du pouce
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Calculer la distance entre l'index et le pouce
            distance = math.sqrt(
                (index_tip.x - thumb_tip.x)**2 + 
                (index_tip.y - thumb_tip.y)**2 + 
                (index_tip.z - thumb_tip.z)**2
            )
            
            # Calculer le point milieu
            mid_x = (index_tip.x + thumb_tip.x) / 2
            mid_y = (index_tip.y + thumb_tip.y) / 2
            mid_z = (index_tip.z + thumb_tip.z) / 2
            
            # Ajuster les coordonnées pour OpenGL
            # Inverser Y, centrer X, ajuster Z
            opengl_x = (mid_x - 0.5) * 10
            opengl_y = -(mid_z) * 10  # Utiliser Z comme hauteur
            opengl_z = (mid_y - 0.5) * 10
            mid_point = (opengl_x, opengl_y, opengl_z)
            
            # Dessiner un cercle au milieu pour montrer le point de dessin
            h, w, c = frame.shape
            cv2.circle(
                frame, 
                (int(mid_x * w), int(mid_y * h)), 
                10, 
                (0, 0, 255) if distance < PINCH_THRESHOLD else (0, 255, 0), 
                -1
            )
            
            # Afficher la distance
            cv2.putText(
                frame, 
                f"Distance: {distance:.3f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 255) if distance < PINCH_THRESHOLD else (0, 255, 0), 
                2
            )
            
            # Vérifier si on pince
            is_pinching = distance < PINCH_THRESHOLD
            
            # Gérer le dessin
            if is_pinching:
                if not scene.is_drawing:
                    scene.start_new_line()
                scene.add_point(mid_point)
            elif scene.is_drawing:
                scene.stop_drawing()
    
    return frame, is_pinching, mid_point

def main():
    """Fonction principale"""
    # Initialiser Pygame
    pygame.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Dessin 3D par gestes de main")
    
    # Configurer OpenGL
    setup_opengl(WINDOW_WIDTH, WINDOW_HEIGHT)
    
    # Initialiser la scène 3D
    scene = Scene3D()
    
    # Initialiser la capture vidéo
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    
    # Initialiser MediaPipe
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Variables pour la navigation
    mouse_pressed = False
    prev_mouse_pos = (0, 0)
    
    # Afficher les instructions
    print("=== INSTRUCTIONS ===")
    print("SOURIS:")
    print("  - Bouton gauche: Faire tourner la vue 3D")
    print("  - Molette: Zoomer/Dézoomer")
    print("  - Bouton droit: Déplacer la vue")
    print("CLAVIER:")
    print("  - C: Effacer le dessin")
    print("  - S: Sauvegarder le dessin")
    print("  - L: Charger un dessin")
    print("  - ESC ou Q: Quitter")
    print("MAIN:")
    print("  - Pince index+pouce: Dessiner")
    print("  - Séparer index+pouce: Arrêter de dessiner")
    
    # Boucle principale
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Gérer les événements Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Gestion du clavier
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_c:
                    scene.clear()
                    print("Dessin effacé")
                elif event.key == pygame.K_s:
                    scene.save()
                elif event.key == pygame.K_l:
                    scene.load()
            
            # Gestion de la souris
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Bouton gauche: rotation
                    mouse_pressed = True
                    prev_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 3:  # Bouton droit: translation
                    mouse_pressed = True
                    prev_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # Molette haut: zoom in
                    scene.zoom += 0.5
                elif event.button == 5:  # Molette bas: zoom out
                    scene.zoom -= 0.5
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 3):
                    mouse_pressed = False
            
            elif event.type == pygame.MOUSEMOTION and mouse_pressed:
                x, y = pygame.mouse.get_pos()
                dx = x - prev_mouse_pos[0]
                dy = y - prev_mouse_pos[1]
                prev_mouse_pos = (x, y)
                
                buttons = pygame.mouse.get_pressed()
                if buttons[0]:  # Bouton gauche: rotation
                    scene.rotation_y += dx * 0.5
                    scene.rotation_x += dy * 0.5
                elif buttons[2]:  # Bouton droit: translation
                    scene.pan_x += dx * 0.01
                    scene.pan_y -= dy * 0.01
        
        # Capturer et traiter l'image de la webcam
        ret, frame = cap.read()
        if not ret:
            break
            
        # Retourner l'image horizontalement pour effet miroir
        frame = cv2.flip(frame, 1)
        
        # Traiter l'image pour la détection des mains
        frame, is_pinching, mid_point = process_video_frame(frame, hands, scene)
        
        # Dessiner la scène 3D
        scene.draw()
        pygame.display.flip()
        
        # Convertir l'image pour Pygame et l'afficher dans un coin
        frame = cv2.resize(frame, (VIDEO_WIDTH//2, VIDEO_HEIGHT//2))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.image.frombuffer(
            frame_rgb.tobytes(), (frame_rgb.shape[1], frame_rgb.shape[0]), 'RGB')
        
        # Afficher la vidéo dans le coin inférieur droit
        display.blit(frame_surface, (WINDOW_WIDTH - frame_rgb.shape[1] - 10, WINDOW_HEIGHT - frame_rgb.shape[0] - 10))
        pygame.display.update()
        
        # Limiter à 60 FPS
        clock.tick(60)
    
    # Nettoyer
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()