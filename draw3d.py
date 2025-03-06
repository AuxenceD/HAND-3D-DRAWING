import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# Dimensions de la fenêtre
WIDTH, HEIGHT = 1280, 720

# Configuration MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class Drawing3D:
    def __init__(self):
        # Liste pour stocker les points dessinés
        self.points = []
        # Indique si on est en train de dessiner
        self.is_drawing = False

    def add_point(self, point):
        """Ajoute un point au dessin"""
        if self.is_drawing:
            self.points.append(point)

    def draw(self):
        """Dessine les points en 3D"""
        glBegin(GL_LINE_STRIP)
        glColor3f(0, 1, 1)  # Couleur cyan
        for point in self.points:
            glVertex3f(*point)
        glEnd()

    def clear(self):
        """Efface tous les points"""
        self.points = []

def setup_opengl():
    """Configuration initiale d'OpenGL"""
    glClearColor(0, 0, 0, 1)
    glEnable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0, 0, -5)

def main():
    # Initialisation de Pygame
    pygame.init()
    display = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Dessin 3D par pincement")
    
    # Configuration OpenGL
    setup_opengl()
    
    # Initialisation de la capture vidéo
    cap = cv2.VideoCapture(0)
    
    # Initialisation de MediaPipe Hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Création de l'objet de dessin
    drawing = Drawing3D()
    
    # Boucle principale
    running = True
    while running:
        # Gestion des événements Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    # Touche 'C' pour effacer
                    drawing.clear()
        
        # Capture de la frame vidéo
        ret, frame = cap.read()
        if not ret:
            break
        
        # Retourner l'image horizontalement
        frame = cv2.flip(frame, 1)
        
        # Convertir l'image en RGB pour MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Réinitialiser la zone de dessin
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, 0, -5)
        
        # Si une main est détectée
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Obtenir les coordonnées du pouce et de l'index
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Calculer la distance entre le pouce et l'index
                distance = math.sqrt(
                    (thumb_tip.x - index_tip.x)**2 + 
                    (thumb_tip.y - index_tip.y)**2 + 
                    (thumb_tip.z - index_tip.z)**2
                )
                
                # Convertir les coordonnées pour OpenGL
                x = (thumb_tip.x - 0.5) * 10
                y = (thumb_tip.y - 0.5) * 10
                z = (thumb_tip.z) * 10
                
                # Seuil de pincement (à ajuster)
                PINCH_THRESHOLD = 0.05
                
                # Commencer/arrêter de dessiner
                if distance < PINCH_THRESHOLD:
                    if not drawing.is_drawing:
                        drawing.is_drawing = True
                    drawing.add_point((x, y, z))
                else:
                    drawing.is_drawing = False
        
        # Dessiner les points
        drawing.draw()
        
        # Mettre à jour l'affichage
        pygame.display.flip()
        
        # Limiter à 60 FPS
        pygame.time.Clock().tick(60)
    
    # Libérer les ressources
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()