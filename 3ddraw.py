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
        self.points = []
        self.is_drawing = False

    def add_point(self, point):
        if self.is_drawing:
            self.points.append(point)

    def draw(self):
        """Dessine les points en 3D avec plus de débogage"""
        if not self.points:
            return
        
        # Configuration pour assurer que les points sont visibles
        glColor3f(1, 1, 0)  # Jaune vif
        glPointSize(5.0)  # Points plus gros
        
        glBegin(GL_POINTS)  # Changé de LINE_STRIP à POINTS pour débogage
        for point in self.points:
            glVertex3f(*point)
        glEnd()

    def clear(self):
        self.points = []

def setup_opengl():
    """Configuration détaillée d'OpenGL"""
    try:
        # Couleur de fond (gris foncé pour voir la différence)
        glClearColor(0.2, 0.2, 0.2, 1.0)
        
        # Activer le test de profondeur
        glEnable(GL_DEPTH_TEST)
        
        # Configuration de la projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
        
        # Configuration de la vue
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -5)
        
        # Vérifier les erreurs OpenGL
        error = glGetError()
        if error != GL_NO_ERROR:
            print(f"Erreur OpenGL lors de la configuration : {error}")
    
    except Exception as e:
        print(f"Erreur lors de la configuration OpenGL : {e}")
        traceback.print_exc()

def main():
    try:
        # Initialisation de Pygame
        pygame.init()
        
        # Créer la fenêtre avec des flags spécifiques
        display = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | HWSURFACE)
        pygame.display.set_caption("Dessin 3D par pincement - Débogage")
        
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
            glTranslatef(0, 0, -5)
            
            # Dessiner un repère pour s'assurer que quelque chose est visible
            glBegin(GL_LINES)
            glColor3f(1, 0, 0)  # Rouge
            glVertex3f(0, 0, 0)
            glVertex3f(1, 0, 0)
            glColor3f(0, 1, 0)  # Vert
            glVertex3f(0, 0, 0)
            glVertex3f(0, 1, 0)
            glColor3f(0, 0, 1)  # Bleu
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 1)
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
                        print(f"Point ajouté : {x}, {y}, {z}")
                    else:
                        drawing.is_drawing = False
            
            # Dessiner les points
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