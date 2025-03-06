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
import os
from datetime import datetime

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
        self.rotation_sensitivity = 100.0  # Sensibilité élevée pour rotations rapides
    
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
        self.navigation_active = False
        self.last_hand_position = None
        self.start_pinch_position = None
        
        # Pour les messages à l'écran
        self.message = None
        self.message_timer = 0
    
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
    
    def start_navigation(self, hand_position):
        """Commence une nouvelle session de navigation"""
        self.navigation_active = True
        self.start_pinch_position = np.array([hand_position.x, hand_position.y])
        self.last_hand_position = np.array([hand_position.x, hand_position.y])
    
    def navigate(self, hand_position):
        """Effectue la rotation de la caméra de manière cumulative"""
        if not self.navigation_active or self.last_hand_position is None:
            return
        
        # Calculer le déplacement de la main depuis la dernière position
        current_position = np.array([hand_position.x, hand_position.y])
        delta = current_position - self.last_hand_position
        
        # Appliquer la rotation de façon cumulative
        self.camera.rotation_y += delta[0] * self.camera.rotation_sensitivity
        self.camera.rotation_x -= delta[1] * self.camera.rotation_sensitivity
        
        # Mettre à jour la dernière position de la main
        self.last_hand_position = current_position
    
    def stop_navigation(self):
        """Arrête la navigation"""
        self.navigation_active = False
        self.last_hand_position = None
        self.start_pinch_position = None
    
    def export_to_obj(self, filename="drawing3d.obj"):
        """
        Exporte les lignes 3D vers un fichier OBJ compatible avec Blender.
        """
        try:
            # Créer le répertoire exports s'il n'existe pas
            export_dir = "exports"
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # Construire le chemin complet du fichier
            filepath = os.path.join(export_dir, filename)
            
            with open(filepath, 'w') as f:
                # Écrire l'en-tête du fichier OBJ
                f.write("# OBJ file created by 3D Drawing Application\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Compteur pour les vertex
                vertex_count = 1
                
                # Pour chaque ligne dans le dessin
                for line_index, line in enumerate(self.lines):
                    if len(line) < 2:
                        continue  # Ignorer les lignes avec moins de 2 points
                    
                    # Stocker l'index du premier vertex de cette ligne
                    line_start_vertex = vertex_count
                    
                    # Écrire tous les sommets (vertex) de cette ligne
                    for point in line:
                        # Format OBJ: v x y z
                        f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                        vertex_count += 1
                    
                    # Écrire les segments de ligne
                    # Format OBJ: l v1 v2
                    f.write(f"g line_{line_index+1}\n")  # Groupe pour cette ligne
                    
                    for i in range(len(line) - 1):
                        # Les indices de vertex dans OBJ commencent à 1, pas à 0
                        v1 = line_start_vertex + i
                        v2 = line_start_vertex + i + 1
                        f.write(f"l {v1} {v2}\n")
                
                # Afficher un message
                self.show_message(f"Exporté vers {filepath}")
                return True
                
        except Exception as e:
            self.show_message(f"Erreur: {e}")
            print(f"Erreur lors de l'exportation OBJ : {e}")
            return False
    
    def export_to_stl(self, filename="drawing3d.stl", thickness=0.1):
        """
        Exporte les lignes 3D vers un fichier STL en les transformant en tubes.
        Note: Cette fonction est plus complexe car STL nécessite des surfaces.
        Pour être vraiment utile, il faudrait convertir les lignes en tubes 3D.
        """
        try:
            # Créer le répertoire exports s'il n'existe pas
            export_dir = "exports"
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # Construire le chemin complet du fichier
            filepath = os.path.join(export_dir, filename)
            
            # Pour une implémentation STL complète, il faudrait:
            # 1. Convertir chaque segment de ligne en un "tube" avec une certaine épaisseur
            # 2. Générer les triangles qui constituent ce tube
            # 3. Écrire ces triangles au format STL
            
            # Cette version simplifiée crée juste un message
            self.show_message("STL nécessite conversion en surfaces")
            print("L'exportation STL nécessiterait de convertir les lignes en tubes 3D avec triangulation.")
            
            # Pour implémenter réellement STL, utilisez une bibliothèque comme numpy-stl
            return False
            
        except Exception as e:
            self.show_message(f"Erreur: {e}")
            print(f"Erreur lors de l'exportation STL : {e}")
            return False
    
    def show_message(self, text, duration=180):
        """Affiche un message à l'écran pendant un certain nombre de frames"""
        self.message = text
        self.message_timer = duration  # ~3 secondes à 60 FPS
    
    def update_message(self):
        """Met à jour le timer de message"""
        if self.message_timer > 0:
            self.message_timer -= 1
            if self.message_timer <= 0:
                self.message = None
    
    def draw_message(self, surface):
        """Dessine le message sur la surface Pygame"""
        if self.message and self.message_timer > 0:
            # Sauvegarde la projection OpenGL actuelle
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
            
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            # Désactiver le test de profondeur temporairement
            glDisable(GL_DEPTH_TEST)
            
            # Créer un fond semi-transparent
            glColor4f(0, 0, 0, 0.7)
            glBegin(GL_QUADS)
            glVertex2f(0, HEIGHT - 50)
            glVertex2f(WIDTH, HEIGHT - 50)
            glVertex2f(WIDTH, HEIGHT)
            glVertex2f(0, HEIGHT)
            glEnd()
            
            # Retour aux paramètres OpenGL normaux
            glEnable(GL_DEPTH_TEST)
            
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            
            # Rendu du texte avec Pygame
            # Cette partie doit être faite après le flip() de pygame.display
            # car OpenGL et Pygame ne partagent pas le même contexte de rendu
            
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
        self.navigation_active = False
        self.last_hand_position = None
        self.start_pinch_position = None
        self.show_message("Dessin effacé")

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

def setup_opengl():
    """Configuration détaillée d'OpenGL"""
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def render_text(screen, message):
    """Rendu d'un texte sur l'écran Pygame"""
    if not message:
        return
    
    # Initialiser la police (si ce n'est pas déjà fait)
    if not hasattr(render_text, "font"):
        pygame.font.init()
        render_text.font = pygame.font.SysFont(None, 32)
    
    # Créer la surface de texte
    text_surface = render_text.font.render(message, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(WIDTH//2, HEIGHT-25))
    
    # Dessiner sur l'écran
    screen.blit(text_surface, text_rect)

def main():
    try:
        # Initialisation de Pygame
        pygame.init()
        
        # Créer la fenêtre avec des flags spécifiques
        display = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | HWSURFACE)
        pygame.display.set_caption("Canvas 3D avec Exportation")
        
        # Pour les messages texte
        text_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
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
        
        # Afficher les instructions de commande
        drawing.show_message("O: Exporter en OBJ | S: Exporter en STL | C: Effacer", 300)
        
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
                        drawing.show_message("Caméra réinitialisée")
                    elif event.key == pygame.K_o:  # Exporter en OBJ
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"drawing3d_{timestamp}.obj"
                        drawing.export_to_obj(filename)
                    elif event.key == pygame.K_s:  # Exporter en STL
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"drawing3d_{timestamp}.stl"
                        drawing.export_to_stl(filename)
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
            if pygame.K_s in keys_pressed and keys_pressed[pygame.K_s] and not (event.type == pygame.KEYDOWN and event.key == pygame.K_s):
                camera.move(0, 0, -0.2)
            if pygame.K_a in keys_pressed and keys_pressed[pygame.K_a]:
                camera.rotation_y -= 5.0
            if pygame.K_d in keys_pressed and keys_pressed[pygame.K_d]:
                camera.rotation_y += 5.0
            
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
            
            # Réinitialiser la zone de dessin OpenGL
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
                    
                    # Définir le point pour la navigation
                    class HandPoint:
                        def __init__(self, x, y, z):
                            self.x = x
                            self.y = y
                            self.z = z
                    
                    # Utiliser le point du pouce pour la navigation
                    hand_point = HandPoint(thumb_tip.x, thumb_tip.y, thumb_tip.z)
                    
                    # Seuil de pincement pour la navigation
                    PINCH_THRESHOLD = 0.05
                    
                    # État actuel du pincement
                    currently_pinched = distance < PINCH_THRESHOLD
                    
                    # Gestion des états de pincement
                    if currently_pinched and not left_hand_pinched:
                        # Début d'un nouveau pincement
                        drawing.start_navigation(hand_point)
                    elif currently_pinched and left_hand_pinched:
                        # Pincement continu - naviguer
                        drawing.navigate(hand_point)
                    elif not currently_pinched and left_hand_pinched:
                        # Fin du pincement
                        drawing.stop_navigation()
                    
                    
                    # Mettre à jour l'état du pincement
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
            
            # Mettre à jour le message à l'écran
            drawing.update_message()
            
            # Mettre à jour l'affichage
            pygame.display.flip()
            
            # Si un message est actif, le dessiner par-dessus l'écran OpenGL
            if drawing.message:
                # Effacer la surface de texte
                text_surface.fill((0, 0, 0, 0))
                
                # Créer un fond semi-transparent
                bg_rect = pygame.Rect(0, HEIGHT - 50, WIDTH, 50)
                pygame.draw.rect(text_surface, (0, 0, 0, 180), bg_rect)
                
                # Rendu du texte
                render_text(text_surface, drawing.message)
                
                # Afficher la surface de texte sur l'écran principal
                display.blit(text_surface, (0, 0))
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