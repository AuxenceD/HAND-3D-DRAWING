import cv2
import mediapipe as mp
import pygame
import sys
import traceback
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
        self.lines = []  # Liste de lignes (chaque ligne = liste de points)
        self.current_line = None
        self.is_drawing = False
        
        # Paramètres de vue
        self.camera_rotation_x = 0
        self.camera_rotation_y = 0
        self.camera_zoom = -10  # Distance initiale de la caméra

    def start_new_line(self, point):
        """Commence une nouvelle ligne"""
        self.current_line = [point]
        self.lines.append(self.current_line)
        self.is_drawing = True

    def add_point(self, point):
        """Ajoute un point à la ligne en cours"""
        if self.is_drawing and self.current_line is not None:
            self.current_line.append(point)

    def stop_drawing(self):
        """Arrête le dessin de la ligne en cours"""
        self.is_drawing = False
        self.current_line = None

    def draw(self):
        """Dessine toutes les lignes en 3D"""
        # Appliquer les rotations de caméra
        glRotatef(self.camera_rotation_x, 1, 0, 0)
        glRotatef(self.camera_rotation_y, 0, 1, 0)

        # Dessin des lignes
        for line in self.lines:
            if len(line) > 1:
                glBegin(GL_LINE_STRIP)
                glColor3f(0, 1, 1)  # Couleur cyan
                for point in line:
                    glVertex3f(*point)
                glEnd()

    def clear(self):
        """Efface tous les dessins"""
        self.lines = []
        self.current_line = None
        self.is_drawing = False

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
        pygame.display.set_caption("Dessin 3D Interactif")
        
        # Configuration OpenGL
        setup_opengl()
        
        # Initialisation de la capture vidéo
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erreur : Impossible d'ouvrir la caméra")
            return
        
        # Initialisation de MediaPipe Hands
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Création de l'objet de dessin
        drawing = Drawing3D()
        
        # Variables pour la navigation souris
        mouse_pressed = False
        prev_mouse_pos = (0, 0)
        
        # Boucle principale
        clock = pygame.time.Clock()
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
                        drawing.clear()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Bouton gauche
                        mouse_pressed = True
                        prev_mouse_pos = pygame.mouse.get_pos()
                    elif event.button == 4:  # Molette haut (zoom in)
                        drawing.camera_zoom += 0.5
                    elif event.button == 5:  # Molette bas (zoom out)
                        drawing.camera_zoom -= 0.5
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        mouse_pressed = False
                elif event.type == pygame.MOUSEMOTION and mouse_pressed:
                    x, y = pygame.mouse.get_pos()
                    dx = x - prev_mouse_pos[0]
                    dy = y - prev_mouse_pos[1]
                    prev_mouse_pos = (x, y)
                    
                    # Rotation de la caméra
                    drawing.camera_rotation_y += dx * 0.5
                    drawing.camera_rotation_x += dy * 0.5
            
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
            glTranslatef(0, 0, drawing.camera_zoom)
            
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
                    # Utiliser la profondeur (z) pour la troisième dimension
                    x = (thumb_tip.x - 0.5) * 10  # Position horizontale
                    y = (thumb_tip.y - 0.5) * 10  # Position verticale
                    z = (1 - thumb_tip.z) * 10    # Profondeur (inversée pour effet de distance)
                    
                    # Seuil de pincement
                    PINCH_THRESHOLD = 0.05
                    
                    # Commencer/arrêter de dessiner
                    if distance < PINCH_THRESHOLD:
                        if not drawing.is_drawing:
                            drawing.start_new_line((x, y, z))
                        else:
                            drawing.add_point((x, y, z))
                    elif drawing.is_drawing:
                        drawing.stop_drawing()
            
            # Dessiner les lignes
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