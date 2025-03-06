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

# Window dimensions
WIDTH, HEIGHT = 1280, 720

# MediaPipe configuration
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class AirCanvasCamera:
    def __init__(self):
        # Camera positioning and orientation
        self.position = np.array([0.0, 0.0, -10.0])  # Initial camera position
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        
        # Movement and rotation sensitivity
        self.move_sensitivity = 0.05
        self.rotation_sensitivity = 0.5
    
    def move(self, dx, dy, dz):
        """Move the camera in 3D space"""
        # Create rotation matrix
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
        
        # Create movement vector
        movement = np.dot(Ry, np.dot(Rx, np.array([dx, dy, dz])))
        
        # Update position
        self.position += movement * self.move_sensitivity

class Drawing3D:
    def __init__(self, camera):
        self.camera = camera
        self.lines = []  # List of lines (each line = list of points)
        self.current_line = None
        self.is_drawing = False
        self.midpoint = None
    
    def start_new_line(self, point):
        """Start a new line"""
        self.current_line = [point]
        self.lines.append(self.current_line)
        self.is_drawing = True
    
    def add_point(self, point):
        """Add a point to the current line"""
        if self.is_drawing and self.current_line is not None:
            # Only add point if it's sufficiently far from the last point
            if not self.current_line or np.linalg.norm(np.array(point) - np.array(self.current_line[-1])) > 0.1:
                self.current_line.append(point)
    
    def set_midpoint(self, point):
        """Set the midpoint between thumb and index"""
        self.midpoint = point
    
    def stop_drawing(self):
        """Stop drawing the current line"""
        self.is_drawing = False
        self.current_line = None
    
    def draw(self):
        """Draw all lines in 3D and the midpoint"""
        # Draw existing lines
        for line in self.lines:
            if len(line) > 1:
                glBegin(GL_LINE_STRIP)
                glColor3f(0, 1, 1)  # Cyan color
                for point in line:
                    glVertex3f(*point)
                glEnd()
        
        # Draw midpoint
        if self.midpoint is not None:
            glPointSize(10)
            glBegin(GL_POINTS)
            glColor3f(1, 0, 0)  # Red color for midpoint
            glVertex3f(*self.midpoint)
            glEnd()
    
    def clear(self):
        """Clear all drawings"""
        self.lines = []
        self.current_line = None
        self.is_drawing = False
        self.midpoint = None

def setup_opengl():
    """Detailed OpenGL configuration"""
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def transform_hand_point(landmark, camera):
    """Transform hand point considering camera orientation"""
    # Convert hand position to 3D world coordinates
    transformed_point = np.array([
        (landmark.x - 0.5) * 10,   # X
        -(landmark.y - 0.5) * 10,  # Y (inverted)
        landmark.z * 10            # Z depth
    ])
    
    # Rotate point based on camera orientation
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
    
    # Apply rotations
    rotated_point = np.dot(Ry, np.dot(Rx, transformed_point))
    
    return list(rotated_point)

def main():
    try:
        # Initialize Pygame
        pygame.init()
        
        # Create window with specific flags
        display = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | HWSURFACE)
        pygame.display.set_caption("Advanced 3D Air Canvas")
        
        # OpenGL setup
        setup_opengl()
        
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        # Initialize MediaPipe Hands
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Create camera and drawing objects
        camera = AirCanvasCamera()
        drawing = Drawing3D(camera)
        
        # Main loop
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Pygame event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        drawing.clear()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Mouse wheel up (zoom in)
                        camera.position[2] += 0.5
                    elif event.button == 5:  # Mouse wheel down (zoom out)
                        camera.position[2] -= 0.5
            
            # Capture video frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot capture frame")
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert image to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # Clear and reset drawing area
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            # Apply camera transformations
            glTranslatef(*camera.position)
            glRotatef(camera.rotation_x, 1, 0, 0)
            glRotatef(camera.rotation_y, 0, 1, 0)
            
            # Draw 3D axes for reference
            glBegin(GL_LINES)
            # X axis (Red)
            glColor3f(1, 0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(5, 0, 0)
            
            # Y axis (Green)
            glColor3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 5, 0)
            
            # Z axis (Blue)
            glColor3f(0, 0, 1)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 5)
            glEnd()
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                # Separate left and right hands
                left_hand = None
                right_hand = None
                for idx, hand_label in enumerate(results.multi_handedness):
                    if hand_label.classification[0].label == "Left":
                        left_hand = results.multi_hand_landmarks[idx]
                    else:
                        right_hand = results.multi_hand_landmarks[idx]
                
                # Handle left hand for camera navigation
                if left_hand:
                    thumb_tip = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Calculate pinch distance
                    distance = math.sqrt(
                        (thumb_tip.x - index_tip.x)**2 + 
                        (thumb_tip.y - index_tip.y)**2 + 
                        (thumb_tip.z - index_tip.z)**2
                    )
                    
                    # Transform navigation point
                    navigation_point = transform_hand_point(thumb_tip, camera)
                    
                    # Navigation with pinch gesture
                    PINCH_THRESHOLD = 0.05
                    if distance < PINCH_THRESHOLD:
                        # Camera rotation/movement
                        camera.rotation_y += (thumb_tip.x - 0.5) * 10
                        camera.rotation_x += (thumb_tip.y - 0.5) * 10
                
                # Handle right hand for drawing
                if right_hand:
                    thumb_tip = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Calculate midpoint 
                    midpoint_x = (thumb_tip.x + index_tip.x) / 2
                    midpoint_y = (thumb_tip.y + index_tip.y) / 2
                    midpoint_z = (thumb_tip.z + index_tip.z) / 2
                    
                    # Create a landmark-like object for midpoint
                    class MidpointLandmark:
                        def __init__(self, x, y, z):
                            self.x = x
                            self.y = y
                            self.z = z
                    
                    midpoint = MidpointLandmark(midpoint_x, midpoint_y, midpoint_z)
                    
                    # Calculate pinch distance
                    distance = math.sqrt(
                        (thumb_tip.x - index_tip.x)**2 + 
                        (thumb_tip.y - index_tip.y)**2 + 
                        (thumb_tip.z - index_tip.z)**2
                    )
                    
                    # Transform drawing point
                    drawing_point = transform_hand_point(thumb_tip, camera)
                    
                    # Set midpoint for visualization
                    drawing.set_midpoint(transform_hand_point(midpoint, camera))
                    
                    # Drawing with pinch gesture
                    PINCH_THRESHOLD = 0.05
                    if distance < PINCH_THRESHOLD:
                        if not drawing.is_drawing:
                            drawing.start_new_line(drawing_point)
                        else:
                            drawing.add_point(drawing_point)
                    elif drawing.is_drawing:
                        drawing.stop_drawing()
            
            # Draw lines and midpoint
            drawing.draw()
            
            # Update display
            pygame.display.flip()
            
            # Limit to 60 FPS
            clock.tick(60)
    
    except Exception as e:
        print(f"Fatal Error: {e}")
        traceback.print_exc()
    
    finally:
        # Release resources
        try:
            cap.release()
        except:
            pass
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()