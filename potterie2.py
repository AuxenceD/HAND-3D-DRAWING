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

class PotteryCamera:
    def __init__(self):
        # Position et orientation de la caméra
        self.position = np.array([0.0, 0.0, -10.0])
        self.rotation_x = 15.0  # Légère inclinaison vers le bas pour voir le tour
        self.rotation_y = 0.0
        
        # Paramètres du mode poterie
        self.rotation_speed = 4.0  # Vitesse de rotation 4 fois plus rapide (en degrés par frame)
        self.auto_rotate = True    # Rotation automatique activée par défaut
        
        # Pour les contrôles souris
        self.mouse_sensitivity = 0.3
        self.mouse_dragging = False
        self.last_mouse_pos = None
    
    def update_rotation(self):
        """Met à jour la rotation du tour de potier"""
        if self.auto_rotate:
            self.rotation_y += self.rotation_speed
            if self.rotation_y >= 360:
                self.rotation_y -= 360
    
    def start_mouse_drag(self, mouse_pos):
        """Commence le déplacement de la caméra avec la souris"""
        self.mouse_dragging = True
        self.last_mouse_pos = mouse_pos
    
    def update_mouse_drag(self, mouse_pos):
        """Met à jour la position/rotation de la caméra avec la souris"""
        if not self.mouse_dragging or self.last_mouse_pos is None:
            return
        
        # Calculer le déplacement de la souris
        dx = mouse_pos[0] - self.last_mouse_pos[0]
        dy = mouse_pos[1] - self.last_mouse_pos[1]
        
        # Rotation avec le bouton gauche de la souris
        if pygame.mouse.get_pressed()[0]:  # Bouton gauche
            self.rotation_y += dx * self.mouse_sensitivity
            self.rotation_x += dy * self.mouse_sensitivity
            
            # Limiter la rotation verticale pour éviter de se retourner
            self.rotation_x = max(-90, min(90, self.rotation_x))
        
        # Déplacement avec le bouton droit de la souris
        elif pygame.mouse.get_pressed()[2]:  # Bouton droit
            # Déplacement latéral (gauche/droite)
            self.position[0] += dx * self.mouse_sensitivity * 0.05
            
            # Déplacement vertical (haut/bas)
            self.position[1] -= dy * self.mouse_sensitivity * 0.05
        
        # Mettre à jour la dernière position de la souris
        self.last_mouse_pos = mouse_pos
    
    def stop_mouse_drag(self):
        """Arrête le déplacement de la caméra avec la souris"""
        self.mouse_dragging = False
        self.last_mouse_pos = None


class PotteryDrawing:
    def __init__(self, camera):
        self.camera = camera
        self.lines = []  # Liste de lignes (chaque ligne = liste de points)
        self.current_line = None
        self.is_drawing = False
        self.midpoint = None
        
        # Pour les messages à l'écran
        self.message = None
        self.message_timer = 0
        
        # Grille du tour de potier
        self.pottery_radius = 5.0
        self.pottery_height = 10.0
        self.pottery_segments = 36
    
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
    
    def adjust_rotation_speed(self, increment):
        """Ajuste la vitesse de rotation du tour"""
        self.camera.rotation_speed += increment
        
        # Limiter la vitesse entre 0.5 et 10.0
        self.camera.rotation_speed = max(0.5, min(10.0, self.camera.rotation_speed))
        
        self.show_message(f"Vitesse de rotation: {self.camera.rotation_speed:.1f}")
    
    def toggle_auto_rotate(self):
        """Active ou désactive la rotation automatique"""
        self.camera.auto_rotate = not self.camera.auto_rotate
        
        if self.camera.auto_rotate:
            self.show_message("Rotation automatique activée")
        else:
            self.show_message("Rotation automatique désactivée")
    
    def draw_pottery_wheel(self):
        """Dessine le tour de potier et sa grille de référence"""
        # Dessiner la base du tour (disque)
        glColor3f(0.5, 0.5, 0.5)  # Gris
        
        # Disque inférieur (base)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, -self.pottery_height/2, 0)  # Centre
        for i in range(self.pottery_segments + 1):
            angle = 2.0 * math.pi * i / self.pottery_segments
            x = self.pottery_radius * math.cos(angle)
            z = self.pottery_radius * math.sin(angle)
            glVertex3f(x, -self.pottery_height/2, z)
        glEnd()
        
        # Dessiner la grille cylindrique semi-transparente
        glColor4f(0.7, 0.7, 0.7, 0.3)  # Gris clair semi-transparent
        
        # Lignes verticales
        glBegin(GL_LINES)
        for i in range(self.pottery_segments):
            angle = 2.0 * math.pi * i / self.pottery_segments
            x = self.pottery_radius * math.cos(angle)
            z = self.pottery_radius * math.sin(angle)
            
            glVertex3f(x, -self.pottery_height/2, z)
            glVertex3f(x, self.pottery_height/2, z)
        glEnd()
        
        # Cercles horizontaux
        num_circles = 6
        for j in range(num_circles):
            y = -self.pottery_height/2 + j * (self.pottery_height / (num_circles-1))
            
            glBegin(GL_LINE_LOOP)
            for i in range(self.pottery_segments):
                angle = 2.0 * math.pi * i / self.pottery_segments
                x = self.pottery_radius * math.cos(angle)
                z = self.pottery_radius * math.sin(angle)
                
                glVertex3f(x, y, z)
            glEnd()
        
        # Axe central
        glColor3f(0.3, 0.3, 0.3)  # Gris foncé
        glBegin(GL_LINES)
        glVertex3f(0, -self.pottery_height/2, 0)
        glVertex3f(0, self.pottery_height/2, 0)
        glEnd()
    
    def export_to_obj(self, filename="pottery.obj"):
        """Exporte les lignes 3D vers un fichier OBJ compatible avec Blender."""
        try:
            # Créer le répertoire exports s'il n'existe pas
            export_dir = "exports"
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # Construire le chemin complet du fichier
            filepath = os.path.join(export_dir, filename)
            
            with open(filepath, 'w') as f:
                # Écrire l'en-tête du fichier OBJ
                f.write("# OBJ file created by Pottery App\n")
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
    
    def draw(self):
        """Dessine toutes les lignes en 3D et le point milieu"""
        # Dessiner d'abord le tour de potier
        self.draw_pottery_wheel()
        
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
    
    # Activer la transparence pour la grille
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
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
        pygame.display.set_caption("Tour de Potier 3D")
        
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
        
        # Création de la caméra et du dessin
        camera = PotteryCamera()
        drawing = PotteryDrawing(camera)
        
        # Variables pour suivre l'état de pincement
        right_hand_pinched = False
        
        # Boucle principale
        clock = pygame.time.Clock()
        running = True
        
        # Pour le contrôle au clavier
        keys_pressed = {}
        
        # Afficher les instructions de commande
        drawing.show_message("Souris: Navigation | ESPACE: On/Off rotation | +/-: Vitesse | C: Effacer | O: Exporter", 300)
        
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
                    elif event.key == pygame.K_SPACE:
                        drawing.toggle_auto_rotate()
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        drawing.adjust_rotation_speed(0.5)
                    elif event.key == pygame.K_MINUS:
                        drawing.adjust_rotation_speed(-0.5)
                    elif event.key == pygame.K_o:  # Exporter en OBJ
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"pottery_{timestamp}.obj"
                        drawing.export_to_obj(filename)
                elif event.type == pygame.KEYUP:
                    if event.key in keys_pressed:
                        keys_pressed[event.key] = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Molette haut (zoom in)
                        camera.position[2] += 1.0
                    elif event.button == 5:  # Molette bas (zoom out)
                        camera.position[2] -= 1.0
                    elif event.button in [1, 3]:  # Bouton gauche ou droit
                        camera.start_mouse_drag(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button in [1, 3]:  # Bouton gauche ou droit
                        camera.stop_mouse_drag()
                elif event.type == pygame.MOUSEMOTION:
                    camera.update_mouse_drag(event.pos)
            
            # Mettre à jour la rotation du tour
            camera.update_rotation()
            
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
            
            # Si des mains sont détectées
            if results.multi_hand_landmarks:
                # Chercher la main droite pour le dessin
                right_hand = None
                for idx, hand_label in enumerate(results.multi_handedness):
                    if hand_label.classification[0].label == "Right":
                        right_hand = results.multi_hand_landmarks[idx]
                        break
                
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
            
            # Dessiner les lignes et le tour de potier
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