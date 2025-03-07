import cv2
import mediapipe as mp
import pygame
import sys
import traceback
import numpy as np
import math
import os
import subprocess
from datetime import datetime
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from collections import deque

# Window dimensions
WIDTH, HEIGHT = 1280, 720

# Constants for drawing modes
MODE_LINE = 0
MODE_POTTERY = 1  # Direct 3D sculpting mode
MODE_PROFILE_POTTERY = 2  # Profile-based pottery wheel mode

# MediaPipe configuration
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SmoothingFilter:
    """
    Implements a moving average filter to stabilize hand movements
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.points_buffer = deque(maxlen=window_size)
        self.weights = np.linspace(0.5, 1.0, window_size)  # More weight on recent points
        self.weights = self.weights / np.sum(self.weights)  # Normalize weights
    
    def update(self, new_point):
        """
        Add a new point to the buffer and return the smoothed point
        """
        if new_point is None:
            return None
            
        try:
            # Ensure point is converted to a consistent format
            if isinstance(new_point, (list, tuple, np.ndarray)):
                point = np.array(new_point)
            elif hasattr(new_point, 'x') and hasattr(new_point, 'y') and hasattr(new_point, 'z'):
                point = np.array([new_point.x, new_point.y, new_point.z])
            else:
                print(f"Invalid point type for smoothing: {type(new_point)}")
                return new_point
            
            self.points_buffer.append(point)
            
            # If not enough points yet, return the current point
            if len(self.points_buffer) < 2:
                return point.tolist()
                
            # Calculate weighted average
            point_array = np.array(list(self.points_buffer))
            weights = self.weights[-len(self.points_buffer):]
            weights = weights / np.sum(weights)  # Re-normalize weights
            
            smoothed_point = np.zeros_like(point_array[0], dtype=float)
            for i, pt in enumerate(point_array):
                smoothed_point += pt * weights[i]
                
            return smoothed_point.tolist()
        except Exception as e:
            print(f"Error in smoothing: {e}")
            return new_point
    
    def reset(self):
        """Clear the buffer"""
        self.points_buffer.clear()

class AirCanvasCamera:
    """Camera control system for 3D space navigation with smoothing"""
    
    def __init__(self):
        # Camera position in 3D space
        self.position = np.array([0.0, 0.0, -10.0])
        
        # Camera rotation angles (degrees)
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        
        # Navigation state tracking
        self.is_navigating = False
        self.last_nav_point = None
        
        # Navigation sensitivity
        self.rotation_sensitivity = 20.0  # Reduced for smoother movement
        self.movement_sensitivity = 5.0   # Reduced for smoother movement
        
        # Smoothing filter for camera movement
        self.smoother = SmoothingFilter(window_size=15)
    
    def move(self, dx, dy, dz):
        """Move the camera position with smoothing"""
        try:
            # Apply small damping factor to prevent too sudden movements
            damping = 0.8
            self.position[0] += dx * damping
            self.position[1] += dy * damping
            self.position[2] += dz * damping
        except Exception as e:
            print(f"Camera movement error: {e}")
    
    def rotate(self, dx, dy):
        """Rotate the camera view with smoothing"""
        try:
            # Apply damping for smoother rotation
            damping = 0.5
            self.rotation_y += dx * self.rotation_sensitivity * damping
            self.rotation_x += dy * self.rotation_sensitivity * damping
            
            # Clamp rotation on x-axis to prevent flipping
            self.rotation_x = max(-90, min(90, self.rotation_x))
        except Exception as e:
            print(f"Camera rotation error: {e}")
    
    def start_navigation(self, hand_point):
        """Begin camera navigation using hand tracking"""
        try:
            self.is_navigating = True
            self.last_nav_point = hand_point
            self.smoother.reset()  # Reset smoother for new navigation
        except Exception as e:
            print(f"Navigation start error: {e}")
    
    def navigate(self, hand_point):
        """Update camera based on hand movement with smoothing"""
        try:
            if self.is_navigating and self.last_nav_point:
                # Apply smoothing to the hand point
                smoothed_point = self.smoother.update(hand_point)
                
                if smoothed_point and hasattr(smoothed_point, 'x'):
                    # Calculate movement delta
                    dx = smoothed_point.x - self.last_nav_point.x
                    dy = smoothed_point.y - self.last_nav_point.y
                    
                    # Apply rotation based on hand movement
                    self.rotate(dx * 100, dy * 100)
                    
                    # Update the last navigation point gradually to reduce jitter
                    self.last_nav_point = smoothed_point
                elif isinstance(smoothed_point, (list, np.ndarray)) and len(smoothed_point) >= 2:
                    # For list or numpy array inputs
                    dx = smoothed_point[0] - self.last_nav_point.x
                    dy = smoothed_point[1] - self.last_nav_point.y
                    
                    # Apply rotation based on hand movement
                    self.rotate(dx * 100, dy * 100)
                    
                    # Update the last navigation point
                    self.last_nav_point = type('Point', (), {'x': smoothed_point[0], 'y': smoothed_point[1]})()
        except Exception as e:
            print(f"Navigation update error: {e}")
    
    def stop_navigation(self):
        """End camera navigation"""
        self.is_navigating = False
        self.last_nav_point = None

# Function for robust hand point transformation
def transform_hand_point(landmark, camera):
    """
    Transform hand point considering camera orientation with robust error handling
    
    Args:
        landmark: Mediapipe hand landmark
        camera: AirCanvasCamera instance
    
    Returns:
        List of 3D coordinates with robust error handling
    """
    try:
        # Enhanced input validation
        if not hasattr(landmark, 'x') or not hasattr(landmark, 'y') or not hasattr(landmark, 'z'):
            print("Invalid landmark object")
            return [0, 0, 0]  # Default safe value
            
        # More comprehensive NaN and inf checks
        coordinates = [landmark.x, landmark.y, landmark.z]
        if any(math.isnan(coord) or math.isinf(coord) for coord in coordinates):
            print("Invalid landmark coordinates")
            return [0, 0, 0]  # Default safe value
            
        # Convert hand position to 3D world coordinates
        transformed_point = np.array([
            (landmark.x - 0.5) * 10,  # X
            -(landmark.y - 0.5) * 10, # Y (inverted)
            landmark.z * 10           # Z depth
        ])

        # Rotation matrices with additional error checks
        def safe_rotation_matrix(angle, axis):
            try:
                rads = np.radians(-angle)
                if axis == 'x':
                    return np.array([
                        [1, 0, 0],
                        [0, np.cos(rads), -np.sin(rads)],
                        [0, np.sin(rads), np.cos(rads)]
                    ])
                elif axis == 'y':
                    return np.array([
                        [np.cos(rads), 0, np.sin(rads)],
                        [0, 1, 0],
                        [-np.sin(rads), 0, np.cos(rads)]
                    ])
            except Exception as e:
                print(f"Error creating rotation matrix: {e}")
                return np.eye(3)  # Identity matrix as fallback

        Rx = safe_rotation_matrix(camera.rotation_x, 'x')
        Ry = safe_rotation_matrix(camera.rotation_y, 'y')

        # Apply rotations safely
        try:
            rotated_point = np.dot(Ry, np.dot(Rx, transformed_point))
        except Exception as e:
            print(f"Matrix multiplication error: {e}")
            rotated_point = transformed_point  # Fallback to original point
        
        # Final safety check
        if np.any(np.isnan(rotated_point)) or np.any(np.isinf(rotated_point)):
            print("Invalid result after transformation")
            return [0, 0, 0]  # Default safe value
            
        return list(rotated_point)
    except Exception as e:
        print(f"Comprehensive error in transform_hand_point: {e}")
        return [0, 0, 0]  # Default safe value


class Vertex:
    """
    Represents a vertex in 3D space with capabilities for connecting to other vertices
    """
    def __init__(self, position, snap_radius=0.5):
        self.position = position  # [x, y, z]
        self.connected_to = []    # List of vertices this connects to
        self.snap_radius = snap_radius  # Radius for snapping to other vertices
        
    def try_snap(self, vertices):
        """
        Try to snap this vertex to any nearby vertices
        Returns the vertex it snapped to, or None
        """
        for vertex in vertices:
            if vertex is self:
                continue
                
            # Calculate distance
            dist = np.linalg.norm(np.array(self.position) - np.array(vertex.position))
            
            if dist < self.snap_radius:
                # Snap to this vertex
                self.position = vertex.position.copy()
                return vertex
                
        return None
        
    def add_connection(self, vertex):
        """Add a connection to another vertex"""
        if vertex not in self.connected_to:
            self.connected_to.append(vertex)
            
    def remove_connection(self, vertex):
        """Remove a connection to another vertex"""
        if vertex in self.connected_to:
            self.connected_to.remove(vertex)


class Line3D:
    """
    Enhanced line representation for better 3D structure awareness
    """
    def __init__(self, vertices=None):
        self.vertices = vertices or []  # List of Vertex objects
        self.start_vertex = None
        self.end_vertex = None
        self.is_complete = False
        
    def add_vertex(self, position):
        """Add a vertex to the line"""
        new_vertex = Vertex(position)
        self.vertices.append(new_vertex)
        
        # Track start and end vertices
        if len(self.vertices) == 1:
            self.start_vertex = new_vertex
        else:
            self.end_vertex = new_vertex
            
        return new_vertex
        
    def complete(self):
        """Mark the line as complete"""
        self.is_complete = True


class GridPlane:
    """
    Represents a reference grid plane for easier 3D alignment
    """
    def __init__(self, normal_axis='y', size=10, spacing=1.0):
        self.normal_axis = normal_axis  # Axis perpendicular to the plane ('x', 'y', or 'z')
        self.size = size                # Grid size (extends in both directions)
        self.spacing = spacing          # Grid line spacing
        self.position = 0               # Position along the normal axis
        self.visible = True
        self.snap_enabled = True
        self.snap_tolerance = 0.3       # Tolerance for snapping to the grid plane
        
    def draw(self):
        """Draw the grid plane"""
        if not self.visible:
            return
            
        # Draw grid plane with slightly transparent lines
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glLineWidth(1.0)
        glBegin(GL_LINES)
        
        # Set grid color based on axis (with transparency)
        if self.normal_axis == 'x':
            glColor4f(0.7, 0.3, 0.3, 0.3)  # Reddish for X plane
        elif self.normal_axis == 'y':
            glColor4f(0.3, 0.7, 0.3, 0.3)  # Greenish for Y plane
        else:  # z
            glColor4f(0.3, 0.3, 0.7, 0.3)  # Bluish for Z plane
            
        # Draw the grid lines
        min_coord = -self.size * self.spacing
        max_coord = self.size * self.spacing
        
        if self.normal_axis == 'x':
            # Draw Z lines
            for i in range(-self.size, self.size + 1):
                z = i * self.spacing
                glVertex3f(self.position, min_coord, z)
                glVertex3f(self.position, max_coord, z)
                
            # Draw Y lines
            for i in range(-self.size, self.size + 1):
                y = i * self.spacing
                glVertex3f(self.position, y, min_coord)
                glVertex3f(self.position, y, max_coord)
                
        elif self.normal_axis == 'y':
            # Draw X lines
            for i in range(-self.size, self.size + 1):
                x = i * self.spacing
                glVertex3f(x, self.position, min_coord)
                glVertex3f(x, self.position, max_coord)
                
            # Draw Z lines
            for i in range(-self.size, self.size + 1):
                z = i * self.spacing
                glVertex3f(min_coord, self.position, z)
                glVertex3f(max_coord, self.position, z)
                
        else:  # z
            # Draw X lines
            for i in range(-self.size, self.size + 1):
                x = i * self.spacing
                glVertex3f(x, min_coord, self.position)
                glVertex3f(x, max_coord, self.position)
                
            # Draw Y lines
            for i in range(-self.size, self.size + 1):
                y = i * self.spacing
                glVertex3f(min_coord, y, self.position)
                glVertex3f(max_coord, y, self.position)
        
        glEnd()
        glDisable(GL_BLEND)
        
    def snap_to_grid(self, point):
        """
        Snap a point to the grid if it's close to the plane
        Returns the snapped point
        """
        if not self.snap_enabled:
            return point
            
        # Extract the normal axis coordinate
        if self.normal_axis == 'x':
            axis_idx = 0
        elif self.normal_axis == 'y':
            axis_idx = 1
        else:  # z
            axis_idx = 2
            
        # Check if the point is close to the plane
        if abs(point[axis_idx] - self.position) < self.snap_tolerance:
            # Snap the point to the plane
            snapped_point = point.copy()
            snapped_point[axis_idx] = self.position
            
            # Snap the other coordinates to the grid
            for i in range(3):
                if i != axis_idx:
                    snapped_point[i] = round(snapped_point[i] / self.spacing) * self.spacing
                    
            return snapped_point
            
        return point


class SurfaceGenerator:
    """Advanced surface generation system for creating 3D printable meshes"""

    def __init__(self):
        self.lines = []  # Raw drawing lines (Line3D objects)
        self.vertices = []  # All vertices
        self.surfaces = []  # Generated surfaces
        self.surface_threshold = 0.2  # Distance to consider lines connected
        
        # Direct sculpting pottery mode variables
        self.pottery_mesh = None
        self.pottery_radius = 2.0
        self.pottery_height = 4.0
        self.pottery_slices = 16  # Number of vertical slices
        self.pottery_rings = 12   # Number of horizontal rings
        self.sculpt_mode = 0      # 0 = add material, 1 = subtract material
        self.sculpt_strength = {
            0: 0.3,  # Add strength
            1: -0.3  # Subtract strength (negative)
        }
        
        # Profile-based pottery variables
        self.profile_points = []  # 2D profile points for revolution
        self.pottery_surface_vertices = []  # Vertices of the pottery surface
        self.pottery_surface_faces = []     # Faces of the pottery surface
        self.pottery_surface_normals = []   # Normals for lighting
        self.surface_segments = 36  # Number of segments for revolution
        self.pottery_render_mode = 'solid'  # 'solid', 'wireframe', 'points'
        self.show_profile = True    # Show the profile line
        self.previous_profiles = []  # For undo functionality
        self.max_profiles = 5
        
    def add_line(self, line):
        """Add a new line to the drawing"""
        self.lines.append(line)
        
        # Add vertices to the global list
        for vertex in line.vertices:
            if vertex not in self.vertices:
                self.vertices.append(vertex)
                
    def find_closest_vertex(self, position, max_distance=0.5, exclude=None):
        """
        Find the closest vertex to a given position within a maximum distance
        Returns the vertex or None
        """
        closest_vertex = None
        min_dist = max_distance
        
        for vertex in self.vertices:
            if exclude and vertex is exclude:
                continue
                
            dist = np.linalg.norm(np.array(vertex.position) - np.array(position))
            
            if dist < min_dist:
                min_dist = dist
                closest_vertex = vertex
                
        return closest_vertex
        
    def connect_lines(self, line1, line2):
        """
        Explicitly connect two lines by their endpoints
        Returns True if successful
        """
        if not line1.is_complete or not line2.is_complete:
            return False
            
        # Connect the endpoints
        line1.end_vertex.add_connection(line2.start_vertex)
        line2.start_vertex.add_connection(line1.end_vertex)
        
        return True
    
    def find_closest_line(self, position, max_distance=1.0):
        """
        Find the closest line to a given position
        Returns (line, distance) tuple or (None, None)
        """
        closest_line = None
        min_distance = max_distance
        
        for line in self.lines:
            if len(line.vertices) < 2:
                continue
                
            # Check distance to each segment in the line
            for i in range(len(line.vertices) - 1):
                p1 = np.array(line.vertices[i].position)
                p2 = np.array(line.vertices[i + 1].position)
                p = np.array(position)
                
                # Calculate distance from point to line segment
                d = self._point_to_segment_distance(p, p1, p2)
                
                if d < min_distance:
                    min_distance = d
                    closest_line = line
                    
        return closest_line, min_distance
    
    def _point_to_segment_distance(self, p, p1, p2):
        """Calculate the distance from a point to a line segment"""
        segment = p2 - p1
        segment_length_squared = np.dot(segment, segment)
        
        # Avoid division by zero
        if segment_length_squared == 0:
            return np.linalg.norm(p - p1)
            
        # Calculate projection
        t = max(0, min(1, np.dot(p - p1, segment) / segment_length_squared))
        projection = p1 + t * segment
        
        # Return distance to the projection
        return np.linalg.norm(p - projection)
        
    def find_closed_loops(self):
        """
        Find closed loops of connected vertices that can form faces
        """
        loops = []
        visited = set()
        
        # Use a recursive helper function
        def find_loops_from_vertex(start, current, path, min_length=3):
            # If we're back at the start and path is long enough, we found a loop
            if current is start and len(path) >= min_length:
                loops.append(path.copy())
                return
                
            # Avoid revisiting vertices (except the start when the path is long enough)
            if current in visited and (current is not start or len(path) < min_length):
                return
                
            visited.add(current)
            
            # Try all connections from this vertex
            for next_vertex in current.connected_to:
                path.append(next_vertex)
                find_loops_from_vertex(start, next_vertex, path, min_length)
                path.pop()
                
            visited.remove(current)
            
        # Start from each vertex
        for vertex in self.vertices:
            find_loops_from_vertex(vertex, vertex, [vertex])
            
        return loops
        
    def generate_surfaces(self):
        """
        Generate surfaces from connected lines and explicit connections
        Uses advanced triangulation techniques
        """
        self.surfaces = []
        
        # First approach: use the existing line-to-line approach for backward compatibility
        self._generate_from_adjacent_lines()
        
        # Second approach: find closed loops and create faces
        self._generate_from_closed_loops()
        
        # Enhanced approach: Find nearby lines that could form surfaces
        self._generate_from_nearby_lines()
        
    def _generate_from_adjacent_lines(self):
        """Generate surfaces by connecting adjacent line segments"""
        # If fewer than 2 lines, no surfaces possible
        if len(self.lines) < 2:
            return
            
        # Get point lists from Line3D objects
        point_lines = []
        for line in self.lines:
            if len(line.vertices) >= 2:
                point_lines.append([v.position for v in line.vertices])
                
        # Basic surface generation by connecting adjacent lines
        for i in range(len(point_lines) - 1):
            line1 = point_lines[i]
            line2 = point_lines[i + 1]
            
            # Ensure lines have enough points
            if len(line1) < 2 or len(line2) < 2:
                continue
                
            # Create surfaces by connecting line points
            surface = []
            for j in range(min(len(line1), len(line2)) - 1):
                # Create two triangles to form a quad-like surface
                triangle1 = [
                    line1[j],
                    line1[j+1],
                    line2[j]
                ]
                
                triangle2 = [
                    line1[j+1],
                    line2[j+1],
                    line2[j]
                ]
                
                surface.extend([triangle1, triangle2])
                
            if surface:
                self.surfaces.append(surface)
    
    def _generate_from_nearby_lines(self):
        """Generate surfaces by connecting lines that are close to each other"""
        # Threshold distance for connecting lines
        connection_threshold = 1.0
        
        # Store lines that are already connected to avoid duplicate surfaces
        connected_pairs = set()
        
        # Check each pair of lines
        for i, line1 in enumerate(self.lines):
            if len(line1.vertices) < 2:
                continue
                
            for j, line2 in enumerate(self.lines):
                if i == j or (i, j) in connected_pairs or (j, i) in connected_pairs:
                    continue
                    
                if len(line2.vertices) < 2:
                    continue
                    
                # Check if lines are close enough
                min_distance = float('inf')
                for v1 in line1.vertices:
                    for v2 in line2.vertices:
                        dist = np.linalg.norm(np.array(v1.position) - np.array(v2.position))
                        min_distance = min(min_distance, dist)
                
                # If lines are close enough, connect them
                if min_distance <= connection_threshold:
                    surface = self._create_surface_between_lines(line1, line2)
                    if surface:
                        self.surfaces.append(surface)
                        connected_pairs.add((i, j))
    
    def _create_surface_between_lines(self, line1, line2):
        """
        Create a surface between two lines by connecting their vertices
        Returns a list of triangles forming the surface
        """
        # Extract vertex positions from both lines
        points1 = [v.position for v in line1.vertices]
        points2 = [v.position for v in line2.vertices]
        
        # Ensure there are enough points in each line
        if len(points1) < 2 or len(points2) < 2:
            return None
            
        # Create a surface by triangulating between the two lines
        surface = []
        
        # Determine which line has fewer points
        if len(points1) <= len(points2):
            shorter, longer = points1, points2
        else:
            shorter, longer = points2, points1
            
        # Calculate parameterization for the longer line
        params = np.linspace(0, 1, len(shorter))
        
        # Create triangles
        for i in range(len(shorter) - 1):
            # Get points on shorter line
            s1 = shorter[i]
            s2 = shorter[i + 1]
            
            # Get corresponding points on longer line
            idx1 = int(params[i] * (len(longer) - 1))
            idx2 = int(params[i + 1] * (len(longer) - 1))
            
            # Ensure indices are valid
            idx1 = max(0, min(idx1, len(longer) - 1))
            idx2 = max(0, min(idx2, len(longer) - 1))
            
            l1 = longer[idx1]
            l2 = longer[idx2]
            
            # Create two triangles (or one if points are the same)
            if idx1 != idx2:
                triangle1 = [s1, s2, l1]
                triangle2 = [s2, l2, l1]
                surface.extend([triangle1, triangle2])
            else:
                triangle = [s1, s2, l1]
                surface.append(triangle)
                
        return surface
                
    def _generate_from_closed_loops(self):
        """Generate surfaces from identified closed loops"""
        loops = self.find_closed_loops()
        
        for loop in loops:
            # Extract positions
            positions = [vertex.position for vertex in loop]
            
            if len(positions) < 3:
                continue
                
            # Use simple triangulation for convex faces (fan triangulation)
            surface = []
            for i in range(1, len(positions) - 1):
                triangle = [
                    positions[0],
                    positions[i],
                    positions[i+1]
                ]
                surface.append(triangle)
                
            if surface:
                self.surfaces.append(surface)
    
    def initialize_pottery(self):
        """
        Initialize a pottery mesh by creating a cylindrical shape
        
        This method is crucial for direct pottery sculpting mode:
        1. Creates an initial 3D mesh representing a cylinder
        2. Gives the pottery a slight hourglass shape for more natural look
        3. Prepares the mesh for deformation
        """
        # Reset pottery mesh
        self.pottery_mesh = []
        
        # Create a cylinder as the initial pottery shape
        vertices = []
        
        # Create vertices
        for i in range(self.pottery_rings):
            # Calculate Y coordinate, centered around zero
            y = -self.pottery_height/2 + i * (self.pottery_height / (self.pottery_rings - 1))
            
            # Use a slight hourglass shape for the initial pottery
            # This creates a more natural, tapering form
            radius_factor = 1.0 - 0.2 * math.sin(math.pi * i / (self.pottery_rings - 1))
            ring_radius = self.pottery_radius * radius_factor
            
            ring = []
            for j in range(self.pottery_slices):
                angle = 2.0 * math.pi * j / self.pottery_slices
                x = ring_radius * math.cos(angle)
                z = ring_radius * math.sin(angle)
                
                ring.append([x, y, z])
            
            vertices.append(ring)
        
        # Create triangular faces
        for i in range(self.pottery_rings - 1):
            for j in range(self.pottery_slices):
                j_next = (j + 1) % self.pottery_slices
                
                # Define the four corners of a quad on the cylinder
                v1 = vertices[i][j]
                v2 = vertices[i][j_next]
                v3 = vertices[i+1][j_next]
                v4 = vertices[i+1][j]
                
                # Create two triangles from the quad
                triangle1 = [v1, v2, v3]
                triangle2 = [v1, v3, v4]
                
                self.pottery_mesh.append([triangle1, triangle2])
        
        # Add to main surfaces for rendering
        self.surfaces = []  # Clear existing surfaces
        for mesh_triangles in self.pottery_mesh:
            self.surfaces.extend(mesh_triangles)
        
        return True
    
    def deform_pottery(self, point, radius=1.0, mode=0):
        """
        Deform the pottery mesh based on a 3D point
        - point: The 3D point to deform towards
        - radius: The radius of influence around the point
        - mode: 0 = add material (push outward), 1 = subtract material (push inward)
        """
        if not self.pottery_mesh:
            print("Initializing pottery mesh for sculpting")
            self.initialize_pottery()
        
        # Save the current sculpting mode    
        self.sculpt_mode = mode
            
        # Convert point to NumPy array for easy calculations
        try:
            deform_point = np.array(point)
            
            # Track whether any vertices were modified
            modified = False
            
            # Strength factors for each mode
            add_strength = 0.5      # Stronger for adding material
            subtract_strength = 0.7  # Stronger for subtracting
            
            # Go through all triangles in the pottery mesh
            for quad_triangles in self.pottery_mesh:
                for triangle in quad_triangles:
                    for i, vertex in enumerate(triangle):
                        # Convert to NumPy array
                        vertex_point = np.array(vertex)
                        
                        # Calculate distance to deformation point
                        distance = np.linalg.norm(vertex_point - deform_point)
                        
                        # Check if within radius of influence
                        if distance < radius:
                            # Calculate influence factor (closer points are affected more)
                            influence = (1.0 - distance / radius)
                            
                            if mode == 0:  # Add material (push outward from center)
                                try:
                                    # Calculate direction from center of pottery (approx at 0,0,0)
                                    # Keep y-coordinate the same to maintain the pottery's height profile
                                    center = np.array([0, vertex_point[1], 0])
                                    direction = vertex_point - center
                                    
                                    # Skip points too close to center
                                    if np.linalg.norm(direction) < 0.1:
                                        continue
                                    
                                    # Normalize direction vector - safely
                                    direction_length = np.linalg.norm(direction)
                                    if direction_length > 0.001:  # Avoid division by zero
                                        direction = direction / direction_length
                                        # Move vertex outward from center with stronger effect
                                        vertex_point = vertex_point + direction * influence * add_strength
                                    else:
                                        continue  # Skip this vertex if direction is near zero
                                
                                except Exception as e:
                                    print(f"Error in add deformation: {e}")
                                    continue  # Skip this vertex if there's an error
                                
                            else:  # Subtract material (push inward toward center)
                                try:
                                    # Calculate direction toward center of pottery
                                    center = np.array([0, vertex_point[1], 0])
                                    direction = center - vertex_point
                                    
                                    # Skip points too close to center
                                    if np.linalg.norm(direction) < 0.1:
                                        continue
                                    
                                    # Normalize direction - safely
                                    direction_length = np.linalg.norm(direction)
                                    if direction_length > 0.001:
                                        direction = direction / direction_length
                                    else:
                                        continue
                                    
                                    # Push vertex toward center based on distance from cursor
                                    push_direction = deform_point - vertex_point
                                    push_strength = influence * subtract_strength
                                    
                                    # Combined movement that pushes toward both cursor and center - safely
                                    if np.linalg.norm(push_direction) > 0.001:
                                        push_direction = push_direction / np.linalg.norm(push_direction)
                                        movement = (push_direction * 0.7 + direction * 0.3) * push_strength
                                        vertex_point = vertex_point + movement
                                
                                except Exception as e:
                                    print(f"Error in subtract deformation: {e}")
                                    continue  # Skip this vertex if there's an error
                            
                            # Update the vertex in the triangle
                            triangle[i] = vertex_point.tolist()
                            modified = True
            
            return modified
            
        except Exception as e:
            print(f"Error deforming pottery: {e}")
            return False
    
    def rotate_pottery(self, angle_degrees):
        """
        Rotate the pottery mesh around the Y axis
        - angle_degrees: Rotation angle in degrees
        """
        if not self.pottery_mesh:
            return False
            
        # Convert angle to radians
        angle = math.radians(angle_degrees)
        
        # Create rotation matrix around Y axis
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        # Rotate all vertices in the pottery mesh
        for quad_triangles in self.pottery_mesh:
            for triangle in quad_triangles:
                for i, vertex in enumerate(triangle):
                    # Convert to NumPy array for matrix multiplication
                    vertex_point = np.array([vertex[0], vertex[1], vertex[2]])
                    
                    # Apply rotation
                    rotated_point = np.dot(rotation_matrix, vertex_point)
                    
                    # Update the vertex in the triangle
                    triangle[i] = rotated_point.tolist()
                    
        return True
    
    def start_profile_drawing(self, point):
        """Start drawing a new profile for pottery"""
        # Save the previous profile if it exists
        if len(self.profile_points) > 2:
            self.previous_profiles.append(self.profile_points.copy())
            if len(self.previous_profiles) > self.max_profiles:
                self.previous_profiles.pop(0)
        
        # Projection of the point to ensure it's on the positive X side
        profile_point = [abs(point[0]), point[1], 0]  # Keep only X positive for the profile
        
        # Reset the profile and start a new one
        self.profile_points = [profile_point]
        
        return True
    
    def add_profile_point(self, point):
        """Add a point to the current profile"""
        if not self.profile_points:
            return False
        
        # Projection of the point
        profile_point = [abs(point[0]), point[1], 0]  # Keep only X positive
        
        # Only add if it's far enough from the last point
        if np.linalg.norm(np.array(profile_point[:2]) - np.array(self.profile_points[-1][:2])) > 0.1:
            self.profile_points.append(profile_point)
            # Generate the surface immediately
            self.generate_pottery_surface()
            return True
        
        return False
    
    def generate_pottery_surface(self):
        """Generate a 3D surface by revolving the profile"""
        if len(self.profile_points) < 2:
            return False
        
        # Reset surface data
        self.pottery_surface_vertices = []
        self.pottery_surface_faces = []
        self.pottery_surface_normals = []
        self.surfaces = []  # Explicitly reset surfaces
        
        # Generate points by rotating around the Y axis
        for point_idx, point in enumerate(self.profile_points):
            x, y, _ = point
            
            # For each profile point, create a complete ring of points
            for segment in range(self.surface_segments):
                angle = 2.0 * math.pi * segment / self.surface_segments
                
                # 3D coordinates after rotation
                new_x = x * math.cos(angle)
                new_z = x * math.sin(angle)
                
                # Add the vertex
                self.pottery_surface_vertices.append([new_x, y, new_z])
                
                # Calculate approximate normal (pointing outward)
                normal_x = math.cos(angle)
                normal_z = math.sin(angle)
                self.pottery_surface_normals.append([normal_x, 0, normal_z])
        
        # Generate faces (quads) between the rings
        for point_idx in range(len(self.profile_points) - 1):
            for segment in range(self.surface_segments):
                # Calculate indices for the 4 vertices of a face
                v1 = point_idx * self.surface_segments + segment
                v2 = point_idx * self.surface_segments + ((segment + 1) % self.surface_segments)
                v3 = (point_idx + 1) * self.surface_segments + ((segment + 1) % self.surface_segments)
                v4 = (point_idx + 1) * self.surface_segments + segment
                
                # Add the face
                self.pottery_surface_faces.append([v1, v2, v3, v4])
        
        # Create triangular surfaces
        for face in self.pottery_surface_faces:
            if len(face) == 4:  # Quad face
                v1, v2, v3, v4 = face
                
                # Ensure vertex indices are valid
                try:
                    # Create two triangles
                    triangle1 = [
                        self.pottery_surface_vertices[v1],
                        self.pottery_surface_vertices[v2],
                        self.pottery_surface_vertices[v3]
                    ]
                    
                    triangle2 = [
                        self.pottery_surface_vertices[v1],
                        self.pottery_surface_vertices[v3],
                        self.pottery_surface_vertices[v4]
                    ]
                    
                    self.surfaces.append(triangle1)
                    self.surfaces.append(triangle2)
                except IndexError:
                    print(f"Invalid vertex indices: {v1}, {v2}, {v3}, {v4}")
        
        return True
    
    def undo_profile(self):
        """Restore the previous profile"""
        if self.previous_profiles:
            self.profile_points = self.previous_profiles.pop()
            self.generate_pottery_surface()
            return True
        return False

    def export_to_stl(self, filename="3d_drawing.stl"):
        """
        Export surfaces to STL for 3D printing
        Uses trimesh for robust mesh generation
        """
        # Prepare mesh vertices and faces
        vertices = []
        faces = []
        vertex_map = {}

        # Create export directory if it doesn't exist
        export_dir = "exports"
        try:
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
                print(f"Created exports directory: {os.path.abspath(export_dir)}")
        except Exception as e:
            print(f"Failed to create exports directory: {e}")
            return False

        # Full path for the export file
        export_path = os.path.join(export_dir, filename)

        try:
            print(f"Preparing to export STL to: {os.path.abspath(export_path)}")
            print(f"Number of surfaces to export: {len(self.surfaces)}")
            
            # Collect unique vertices and map them
            for surface in self.surfaces:
                for triangle in surface:
                    if len(triangle) != 3:
                        continue  # Skip non-triangles
                        
                    triangle_indices = []
                    for point in triangle:
                        try:
                            # Ensure point is valid
                            if not isinstance(point, (list, tuple, np.ndarray)) or len(point) < 3:
                                continue
                                
                            # Convert to tuple for dictionary key
                            point_tuple = tuple(float(p) for p in point[:3])
                            
                            # Check for invalid values
                            if any(math.isnan(p) or math.isinf(p) for p in point_tuple):
                                continue
                                
                            # Add to vertex map
                            if point_tuple not in vertex_map:
                                vertex_map[point_tuple] = len(vertices)
                                vertices.append(point_tuple)
                                
                            triangle_indices.append(vertex_map[point_tuple])
                        except Exception as e:
                            print(f"Error processing point {point}: {e}")
                            # Continue with next point
                    
                    # Only add valid triangles
                    if len(triangle_indices) == 3:
                        faces.append(triangle_indices)

            # Create mesh
            if not vertices or not faces:
                print("No valid geometry to export")
                return False

            # Import trimesh if available, otherwise use a simple STL exporter
            try:
                import trimesh
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                
                # Ensure manifold (watertight) mesh
                mesh.fix_normals()
                
                # Export STL
                mesh.export(export_path)
                print(f"Successfully exported STL to: {os.path.abspath(export_path)}")
                
            except ImportError:
                # Trimesh not available, use simple STL exporter
                print("trimesh not available, using simple STL exporter")
                self._export_simple_stl(vertices, faces, export_path)
                
            # Try to open the directory
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(export_dir)
                else:  # macOS, Linux
                    try:
                        subprocess.call(['open', export_dir])
                    except:
                        subprocess.call(['xdg-open', export_dir])
            except Exception as e:
                print(f"Could not open exports directory: {e}")
                
            return True

        except Exception as e:
            print(f"Mesh export error: {e}")
            traceback.print_exc()
            return False
    
    def _export_simple_stl(self, vertices, faces, filepath):
        """
        A simple STL exporter that doesn't require trimesh
        """
        try:
            with open(filepath, 'w') as f:
                f.write("solid Exported3DModel\n")
                
                for face in faces:
                    if len(face) != 3:
                        continue
                        
                    # Get vertices of this face
                    v1 = vertices[face[0]]
                    v2 = vertices[face[1]]
                    v3 = vertices[face[2]]
                    
                    # Calculate normal
                    try:
                        vect1 = np.array(v2) - np.array(v1)
                        vect2 = np.array(v3) - np.array(v1)
                        normal = np.cross(vect1, vect2)
                        length = np.linalg.norm(normal)
                        if length > 0:
                            normal = normal / length
                        else:
                            normal = [0, 0, 1]  # Default normal if calculation fails
                    except:
                        normal = [0, 0, 1]  # Default normal
                    
                    # Write the facet data
                    f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                    f.write("    outer loop\n")
                    f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                    f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                    f.write(f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
                
                f.write("endsolid Exported3DModel\n")
                
            print(f"Successfully exported simple STL to: {os.path.abspath(filepath)}")
            return True
            
        except Exception as e:
            print(f"Simple STL export error: {e}")
            return False

    def clear(self):
        """Reset all drawing data"""
        self.lines.clear()
        self.vertices.clear()
        self.surfaces.clear()
        self.pottery_mesh = None
        self.profile_points = []
        self.pottery_surface_vertices = []
        self.pottery_surface_faces = []
        self.pottery_surface_normals = []


class Drawing3D:
    """Handles 3D drawing and rendering with enhanced stability and shape generation"""
    
    def __init__(self, camera):
        # Reference to the camera for perspective transforms
        self.camera = camera
        
        # Drawing elements
        self.lines = []  # List of Line3D objects
        self.current_line = None  # Currently active Line3D
        
        # Grid planes for reference
        self.grid_planes = {
            'xy': GridPlane(normal_axis='z', size=10, spacing=1.0),
            'xz': GridPlane(normal_axis='y', size=10, spacing=1.0),
            'yz': GridPlane(normal_axis='x', size=10, spacing=1.0)
        }
        self.active_grid = 'xy'  # Default active grid
        
        # Cursor visualization
        self.cursor_position = None
        self.smoothed_cursor = None
        
        # Surface generation for 3D printing
        self.surface_generator = SurfaceGenerator()
        
        # State management
        self.is_drawing = False
        self.sculpt_mode = 0  # 0 = add, 1 = subtract
        
        # Drawing mode
        self.drawing_mode = MODE_LINE
        
        # User interface
        self.message = ""
        self.message_timeout = 0
        
        # Navigation tracking
        self.nav_start_point = None
        
        # Smoothing filters
        self.cursor_smoother = SmoothingFilter(window_size=8)
        self.drawing_smoother = SmoothingFilter(window_size=5)
        
        # Connect vertices when they're close to each other
        self.auto_connect = True
        self.connect_radius = 0.5
        
        # Show connection points for easier line joining
        self.show_vertices = True
        
        # Pottery mode
        self.pottery_rotation_active = False
        self.pottery_rotation_speed = 2.0  # Degrees per frame
        
        # Cursor colors
        self.cursor_colors = {
            MODE_LINE: (1.0, 0.0, 0.0),      # Red for line mode
            MODE_POTTERY: {
                0: (0.0, 1.0, 0.3),          # Green for adding material
                1: (1.0, 0.5, 0.0)           # Orange for subtracting material
            },
            MODE_PROFILE_POTTERY: (1.0, 1.0, 0.0)  # Yellow for profile pottery
        }
        
    def toggle_drawing_mode(self):
        """Toggle between drawing modes"""
        if self.drawing_mode == MODE_LINE:
            self.drawing_mode = MODE_POTTERY
            self.surface_generator.initialize_pottery()
            self.show_message("Direct Pottery Sculpting mode activated")
        elif self.drawing_mode == MODE_POTTERY:
            self.drawing_mode = MODE_PROFILE_POTTERY
            # Initialize profile pottery mode
            if not self.surface_generator.profile_points:
                self.init_profile_pottery()
            self.show_message("Profile Pottery mode activated")
        else:
            self.drawing_mode = MODE_LINE
            self.show_message("Line mode activated")
            
    def init_profile_pottery(self):
        """Initialize the profile-based pottery mode"""
        # Clear current profile if any
        self.surface_generator.profile_points = []
        self.surface_generator.pottery_surface_vertices = []
        self.surface_generator.pottery_surface_faces = []
        self.surface_generator.pottery_surface_normals = []
            
    def toggle_active_grid(self):
        """Cycle through the reference grids"""
        if self.active_grid == 'xy':
            self.active_grid = 'xz'
            self.show_message("XZ grid plane active")
        elif self.active_grid == 'xz':
            self.active_grid = 'yz'
            self.show_message("YZ grid plane active")
        else:
            self.active_grid = 'xy'
            self.show_message("XY grid plane active")
            
    def start_new_line(self, point):
        """Begin a new line at the specified point"""
        # For direct sculpting pottery mode
        if self.drawing_mode == MODE_POTTERY:
            self.is_drawing = True
            return
            
        # For profile-based pottery mode
        if self.drawing_mode == MODE_PROFILE_POTTERY:
            self.surface_generator.start_profile_drawing(point)
            self.is_drawing = True
            return
            
        # For line drawing mode
        self.is_drawing = True
        
        # Apply grid snapping
        snapped_point = self._snap_to_active_grid(point)
        
        # Apply smoothing
        self.drawing_smoother.reset()
        smoothed_point = self.drawing_smoother.update(snapped_point)
        
        # Create a new Line3D
        self.current_line = Line3D()
        
        # Add the first vertex
        self.current_line.add_vertex(smoothed_point)
        
        # Try to connect to an existing vertex
        if self.auto_connect:
            closest_vertex = self.surface_generator.find_closest_vertex(
                smoothed_point, 
                self.connect_radius
            )
            
            if closest_vertex:
                # Update the start point to match the existing vertex
                self.current_line.vertices[0].position = closest_vertex.position.copy()
                
                # Establish connection
                self.current_line.vertices[0].add_connection(closest_vertex)
                closest_vertex.add_connection(self.current_line.vertices[0])
        
    def add_point(self, point):
        """Add a point to the current line"""
        if not self.is_drawing:
            return
        
        # For direct sculpting pottery mode
        if self.drawing_mode == MODE_POTTERY:
            self.surface_generator.deform_pottery(point, mode=self.sculpt_mode)
            return
        
        # For profile-based pottery mode
        if self.drawing_mode == MODE_PROFILE_POTTERY:
            self.surface_generator.add_profile_point(point)
            return
            
        # For line drawing mode
        if self.current_line is not None:
            # Apply grid snapping
            snapped_point = self._snap_to_active_grid(point)
            
            # Apply smoothing
            smoothed_point = self.drawing_smoother.update(snapped_point)
            
            # Only add points if they're sufficiently different from the last point
            if len(self.current_line.vertices) == 0:
                self.current_line.add_vertex(smoothed_point)
            else:
                last_vertex = self.current_line.vertices[-1]
                distance = self._distance(smoothed_point, last_vertex.position)
                
                if distance > 0.05:
                    self.current_line.add_vertex(smoothed_point)
    
    def stop_drawing(self):
        """Finish the current line and add it to the collection"""
        if not self.is_drawing:
            return
            
        # For direct sculpting pottery mode
        if self.drawing_mode == MODE_POTTERY:
            self.is_drawing = False
            return
            
        # For profile-based pottery mode
        if self.drawing_mode == MODE_PROFILE_POTTERY:
            # Complete the profile and generate the final surface
            if len(self.surface_generator.profile_points) >= 2:
                self.surface_generator.generate_pottery_surface()
            self.is_drawing = False
            return
                
        # For line mode
        if self.current_line is not None:
            # Only add lines with at least 2 vertices
            if len(self.current_line.vertices) >= 2:
                # Mark the line as complete
                self.current_line.complete()
                
                # Try to connect the end to an existing vertex
                if self.auto_connect:
                    last_vertex = self.current_line.vertices[-1]
                    closest_vertex = self.surface_generator.find_closest_vertex(
                        last_vertex.position, 
                        self.connect_radius,
                        exclude=last_vertex
                    )
                    
                    if closest_vertex:
                        # Update the end point to match the existing vertex
                        last_vertex.position = closest_vertex.position.copy()
                        
                        # Establish connection
                        last_vertex.add_connection(closest_vertex)
                        closest_vertex.add_connection(last_vertex)
                
                # Add the line to our collections
                self.lines.append(self.current_line)
                self.surface_generator.add_line(self.current_line)
            
        # Reset drawing state
        self.current_line = None
        self.is_drawing = False
    
    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two 3D points"""
        return math.sqrt(
            (point1[0] - point2[0])**2 + 
            (point1[1] - point2[1])**2 + 
            (point1[2] - point2[2])**2
        )
    
    def _snap_to_active_grid(self, point):
        """Snap a point to the active grid plane"""
        active_grid = self.grid_planes[self.active_grid]
        return active_grid.snap_to_grid(point)
    
    def set_midpoint(self, point):
        """Update the cursor visualization position with smoothing"""
        # Apply smoothing to the cursor
        self.smoothed_cursor = self.cursor_smoother.update(point)
        
        # Store both the raw and smoothed positions
        self.cursor_position = self.smoothed_cursor if self.smoothed_cursor is not None else point
    
    def start_navigation(self, hand_point):
        """Start camera navigation"""
        self.camera.start_navigation(hand_point)
        self.nav_start_point = hand_point
    
    def navigate(self, hand_point):
        """Update camera navigation"""
        self.camera.navigate(hand_point)
    
    def stop_navigation(self):
        """Stop camera navigation"""
        self.camera.stop_navigation()
        self.nav_start_point = None
    
    def update(self):
        """Update state for animations and continuous effects"""
        # Handle pottery rotation for direct sculpting mode
        if self.drawing_mode == MODE_POTTERY and self.pottery_rotation_active:
            self.surface_generator.rotate_pottery(self.pottery_rotation_speed)
        
        # Handle pottery rotation for profile pottery mode
        if self.drawing_mode == MODE_PROFILE_POTTERY and self.pottery_rotation_active:
            # Rotate camera around Y axis for pottery wheel effect
            self.camera.rotation_y += self.pottery_rotation_speed
    
    def clear(self):
        """Clear all drawing data"""
        self.lines = []
        self.current_line = None
        self.surface_generator.clear()
        self.show_message("Drawing cleared")
    
    def _draw_pottery_wheel(self):
        """Draw the pottery wheel and its reference grid"""
        # Draw the wheel base (disc)
        glColor3f(0.5, 0.5, 0.5)  # Gray
        
        # Get pottery dimensions from the surface generator
        radius = self.surface_generator.pottery_radius
        height = self.surface_generator.pottery_height
        segments = self.surface_generator.surface_segments
        
        # Lower disk (base)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, -height/2, 0)  # Center
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            glVertex3f(x, -height/2, z)
        glEnd()
        
        # Draw the semi-transparent cylindrical grid
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glColor4f(0.7, 0.7, 0.7, 0.3)  # Light gray, semi-transparent
        
        # Vertical lines
        glBegin(GL_LINES)
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            
            glVertex3f(x, -height/2, z)
            glVertex3f(x, height/2, z)
        glEnd()
        
        # Horizontal circles
        num_circles = 6
        for j in range(num_circles):
            y = -height/2 + j * (height / (num_circles-1))
            
            glBegin(GL_LINE_LOOP)
            for i in range(segments):
                angle = 2.0 * math.pi * i / segments
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
                
                glVertex3f(x, y, z)
            glEnd()
        
        # Central axis
        glColor3f(0.3, 0.3, 0.3)  # Dark gray
        glBegin(GL_LINES)
        glVertex3f(0, -height/2, 0)
        glVertex3f(0, height/2, 0)
        glEnd()
        
        glDisable(GL_BLEND)
    
    def _draw_profile(self):
        """Draw the 2D profile line"""
        if not self.surface_generator.profile_points:
            return
            
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        glColor3f(1, 1, 0)  # Yellow for the profile
        
        for point in self.surface_generator.profile_points:
            glVertex3f(*point)
            
        glEnd()
        glLineWidth(1.0)
    
    def _draw_vertices(self):
        """Draw all vertices as connection points"""
        if not self.surface_generator.vertices:
            return
            
        glPointSize(6.0)
        glBegin(GL_POINTS)
        glColor3f(0.8, 0.8, 0.0)  # Yellow for connection points
        
        for vertex in self.surface_generator.vertices:
            glVertex3f(vertex.position[0], vertex.position[1], vertex.position[2])
            
        glEnd()
        glPointSize(1.0)
    
    def draw(self):
        """Render all drawing elements"""
        # Draw the active grid planes for line mode
        if self.drawing_mode == MODE_LINE:
            for name, grid in self.grid_planes.items():
                grid.draw()
            
            # Draw connection points (vertices)
            if self.show_vertices:
                self._draw_vertices()
        
        # For profile pottery mode, draw the wheel
        if self.drawing_mode == MODE_PROFILE_POTTERY:
            self._draw_pottery_wheel()
            
            # Draw the profile line
            if self.surface_generator.show_profile:
                self._draw_profile()
        
        # Draw completed lines (for line mode)
        glLineWidth(2.0)
        for line in self.lines:
            if len(line.vertices) < 2:
                continue
                
            glBegin(GL_LINE_STRIP)
            glColor3f(1.0, 1.0, 1.0)  # White for lines
            
            for vertex in line.vertices:
                glVertex3f(vertex.position[0], vertex.position[1], vertex.position[2])
                
            glEnd()
        
        # Draw current line (in line mode)
        if self.drawing_mode == MODE_LINE and self.is_drawing and self.current_line and len(self.current_line.vertices) > 0:
            glBegin(GL_LINE_STRIP)
            glColor3f(0.0, 1.0, 0.0)  # Green for active line
            
            for vertex in self.current_line.vertices:
                glVertex3f(vertex.position[0], vertex.position[1], vertex.position[2])
                
            # Add a line to the current cursor position
            if self.cursor_position and len(self.current_line.vertices) > 0:
                last_vertex = self.current_line.vertices[-1]
                glVertex3f(last_vertex.position[0], last_vertex.position[1], last_vertex.position[2])
                glVertex3f(self.cursor_position[0], self.cursor_position[1], self.cursor_position[2])
                
            glEnd()
        
        # Draw cursor
        if self.cursor_position:
            glPointSize(10.0)
            glBegin(GL_POINTS)
            
            # Choose color based on drawing mode
            if self.drawing_mode == MODE_LINE:
                color = self.cursor_colors[MODE_LINE]
                glColor3f(*color)
            elif self.drawing_mode == MODE_POTTERY:
                # Use sculpt mode specific colors for direct pottery
                color = self.cursor_colors[MODE_POTTERY][self.sculpt_mode]
                glColor3f(*color)
            else:  # Profile pottery mode
                glColor3f(1.0, 1.0, 0.0)  # Yellow for profile drawing
                
            glVertex3f(
                self.cursor_position[0],
                self.cursor_position[1],
                self.cursor_position[2]
            )
            glEnd()
            
            # For direct pottery mode, draw an indication of sculpt mode
            if self.drawing_mode == MODE_POTTERY:
                glPointSize(5.0)
                glBegin(GL_POINTS)
                
                # Draw small point near cursor to indicate mode
                offset = 0.3
                if self.sculpt_mode == 0:  # Add mode - show plus sign
                    glColor3f(0.0, 1.0, 0.3)  # Green for add
                    glVertex3f(self.cursor_position[0] + offset, self.cursor_position[1], self.cursor_position[2])
                    glVertex3f(self.cursor_position[0] - offset, self.cursor_position[1], self.cursor_position[2])
                    glVertex3f(self.cursor_position[0], self.cursor_position[1] + offset, self.cursor_position[2])
                    glVertex3f(self.cursor_position[0], self.cursor_position[1] - offset, self.cursor_position[2])
                else:  # Subtract mode - show minus sign
                    glColor3f(1.0, 0.5, 0.0)  # Orange for subtract
                    glVertex3f(self.cursor_position[0] + offset, self.cursor_position[1], self.cursor_position[2])
                    glVertex3f(self.cursor_position[0] - offset, self.cursor_position[1], self.cursor_position[2])
                
                glEnd()
        
        # Draw surfaces from the generator
        self.draw_surfaces()
    
    def draw_surfaces(self):
        """Render the generated surfaces"""
        if not self.surface_generator.surfaces:
            return
        
        # Choose render mode for profile pottery
        if self.drawing_mode == MODE_PROFILE_POTTERY:
            render_mode = self.surface_generator.pottery_render_mode
        else:
            render_mode = 'solid'  # Default for other modes
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set polygon mode based on render mode
        if render_mode == 'wireframe':
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        for surface in self.surface_generator.surfaces:
            # Add robust checking for surface validity
            if not isinstance(surface, list) or len(surface) == 0:
                continue
            
            # Ensure each triangle is valid
            try:
                # Validate triangle structure
                if len(surface) == 3 and all(len(vertex) == 3 for vertex in surface):
                    glBegin(GL_TRIANGLES)
                    
                    # Calculate normal vector
                    try:
                        v1 = np.array(surface[0])
                        v2 = np.array(surface[1])
                        v3 = np.array(surface[2])
                        
                        # Calculate normal vector
                        normal = np.cross(v2 - v1, v3 - v1)
                        
                        # Normalize
                        norm_length = np.linalg.norm(normal)
                        if norm_length > 0:
                            normal = normal / norm_length
                            glNormal3f(normal[0], normal[1], normal[2])
                    except Exception:
                        # Fallback normal if calculation fails
                        glNormal3f(0, 1, 0)
                    
                    # Different colors for different modes
                    if self.drawing_mode == MODE_POTTERY:
                        glColor4f(0.8, 0.6, 0.4, 0.8)  # Clay color for pottery
                    elif self.drawing_mode == MODE_PROFILE_POTTERY:
                        glColor4f(0.76, 0.6, 0.42, 0.8)  # Natural clay color
                    else:
                        glColor4f(0.2, 0.6, 1.0, 0.6)  # Blue for line mode
                    
                    # Draw triangle vertices
                    for vertex in surface:
                        glVertex3f(vertex[0], vertex[1], vertex[2])
                    
                    glEnd()
            except Exception as e:
                print(f"Error processing surface: {e}")
        
        # Reset polygon mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_BLEND)
    
    def show_message(self, text, duration=3.0):
        """Display a message to the user"""
        self.message = text
        self.message_timeout = pygame.time.get_ticks() + (duration * 1000)
        print(f"Message: {text}")  # Also print to console for clarity
    
    def update_message(self):
        """Update message display status"""
        if self.message and pygame.time.get_ticks() > self.message_timeout:
            self.message = ""
    
    def toggle_pottery_rotation(self):
        """Toggle pottery rotation on/off"""
        self.pottery_rotation_active = not self.pottery_rotation_active
        
        if self.pottery_rotation_active:
            self.show_message("Pottery rotation activated")
        else:
            self.show_message("Pottery rotation stopped")
    
    def export_mesh(self):
        """Export the 3D model to STL or OBJ for printing"""
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create exports directory if it doesn't exist
        export_dir = "exports"
        try:
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
                print(f"Created exports directory: {os.path.abspath(export_dir)}")
        except Exception as e:
            self.show_message(f"Failed to create exports directory: {str(e)}")
            print(f"Error creating exports directory: {e}")
            return False
        
        # For profile pottery mode, export as OBJ
        if self.drawing_mode == MODE_PROFILE_POTTERY:
            filename = f"pottery_{timestamp}.obj"
            filepath = os.path.join(export_dir, filename)
            
            try:
                # Check if we have profile points
                if len(self.surface_generator.profile_points) < 2:
                    self.show_message("No profile to export")
                    return False
                    
                # Create the OBJ file
                with open(filepath, 'w') as f:
                    # Write header
                    f.write("# OBJ file created by Air Canvas\n")
                    f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    # Write vertices
                    for vertex in self.surface_generator.pottery_surface_vertices:
                        f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                    
                    # Write normals
                    for normal in self.surface_generator.pottery_surface_normals:
                        f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                    
                    # Material and group
                    f.write("g pottery_surface\n")
                    
                    # Write faces
                    for face in self.surface_generator.pottery_surface_faces:
                        face_str = "f"
                        for idx in face:
                            face_str += f" {idx+1}//{idx+1}"
                        f.write(face_str + "\n")
                
                print(f"Successfully exported OBJ to: {os.path.abspath(filepath)}")
                self.show_message(f"Exported as {filename}. See {os.path.abspath(export_dir)}")
                
                # Try to open the directory for the user
                try:
                    if os.name == 'nt':  # Windows
                        os.startfile(export_dir)
                    else:  # macOS, Linux
                        try:
                            subprocess.call(['open', export_dir])
                        except:
                            subprocess.call(['xdg-open', export_dir])
                except Exception as e:
                    print(f"Could not open exports directory: {e}")
                    
                return True
                
            except Exception as e:
                print(f"Error exporting OBJ: {e}")
                traceback.print_exc()  # Print full stack trace
                self.show_message(f"Export failed: {str(e)}")
                return False
            
        # For other modes, export as STL
        else:
            filename = f"3d_drawing_{timestamp}.stl"
            
            # Generate surfaces if none exist
            if not self.surface_generator.surfaces:
                self.surface_generator.generate_surfaces()
            
            # Export the mesh
            success = self.surface_generator.export_to_stl(filename)
            
            if success:
                self.show_message(f"Model exported as {filename}. See {os.path.abspath(export_dir)}")
            else:
                self.show_message("Export failed. No valid geometry.")
            
            return success


def transform_hand_point(landmark, camera):
    """Transform hand point considering camera orientation with error handling"""
    try:
        # Validate inputs
        if not hasattr(landmark, 'x') or not hasattr(landmark, 'y') or not hasattr(landmark, 'z'):
            print("Invalid landmark object")
            return [0, 0, 0]  # Default safe value
            
        # Check for NaN or inf values
        if (math.isnan(landmark.x) or math.isnan(landmark.y) or math.isnan(landmark.z) or
            math.isinf(landmark.x) or math.isinf(landmark.y) or math.isinf(landmark.z)):
            print("Invalid landmark coordinates")
            return [0, 0, 0]  # Default safe value
            
        # Convert hand position to 3D world coordinates
        transformed_point = np.array([
            (landmark.x - 0.5) * 10,  # X
            -(landmark.y - 0.5) * 10, # Y (inverted)
            landmark.z * 10           # Z depth
        ])

        # Rotation of the point according to camera orientation
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
        
        # Check for NaN or inf in result
        if np.any(np.isnan(rotated_point)) or np.any(np.isinf(rotated_point)):
            print("Invalid result after transformation")
            return [0, 0, 0]  # Default safe value
            
        return list(rotated_point)
    except Exception as e:
        print(f"Error in transform_hand_point: {e}")
        return [0, 0, 0]  # Default safe value


def setup_opengl():
    """Detailed OpenGL configuration"""
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Enable lighting for better 3D visualization
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    
    # Set up light 0
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
    glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
    
    # Set material properties
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])


def render_text(screen, text_surface, message):
    """Render text on the Pygame screen"""
    if not message:
        return

    # Initialize font (if not already done)
    if not hasattr(render_text, "font"):
        pygame.font.init()
        render_text.font = pygame.font.SysFont(None, 32)

    # Create text surface
    text_surface.fill((0, 0, 0, 0))

    # Create semi-transparent background
    bg_rect = pygame.Rect(0, HEIGHT - 50, WIDTH, 50)
    pygame.draw.rect(text_surface, (0, 0, 0, 180), bg_rect)

    # Render text
    text_surf = render_text.font.render(message, True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=(WIDTH//2, HEIGHT-25))

    # Draw on text surface
    text_surface.blit(text_surf, text_rect)


def main():
    try:
        # Initialize Pygame
        pygame.init()

        # Create window with specific flags
        display = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | HWSURFACE)
        pygame.display.set_caption("Enhanced 3D Drawing with Pottery Modes")

        # Surface for text messages
        text_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        # OpenGL configuration
        setup_opengl()

        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open camera")
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

        # Variables to track pinch state
        left_hand_pinched = False
        right_hand_index_pinched = False
        right_hand_middle_pinched = False
        
        # For rotating profile pottery
        pottery_wheel_rotation = True

        # Variable to pause video capture (during dialogs)
        cap_paused = False

        # Main loop
        clock = pygame.time.Clock()
        running = True

        # For keyboard control
        keys_pressed = {}

        # Create exports directory at startup
        export_dir = "exports"
        try:
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
                print(f"Created exports directory: {os.path.abspath(export_dir)}")
        except Exception as e:
            print(f"Failed to create exports directory: {e}")

        # Initial message
        drawing.show_message("3D Drawing with Line, Direct Sculpting, and Profile Pottery Modes")

        while running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    keys_pressed[event.key] = True
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        drawing.clear()
                    elif event.key == pygame.K_r:  # Reset camera or render mode
                        if pygame.key.get_mods() & pygame.KMOD_CTRL:  # Ctrl+R for camera reset
                            camera.position = np.array([0.0, 0.0, -10.0])
                            camera.rotation_x = 0.0
                            camera.rotation_y = 0.0
                            drawing.show_message("Camera reset")
                        elif drawing.drawing_mode == MODE_PROFILE_POTTERY:
                            # Cycle through render modes
                            modes = ['solid', 'wireframe', 'points']
                            current_index = modes.index(drawing.surface_generator.pottery_render_mode)
                            drawing.surface_generator.pottery_render_mode = modes[(current_index + 1) % len(modes)]
                            drawing.show_message(f"Render mode: {drawing.surface_generator.pottery_render_mode}")
                    elif event.key == pygame.K_e:  # Export STL mesh
                        drawing.export_mesh()
                    elif event.key == pygame.K_g:  # Generate surfaces
                        drawing.surface_generator.generate_surfaces()
                        drawing.show_message("Surfaces generated")
                    elif event.key == pygame.K_i:  # Display drawing info
                        line_count = len(drawing.lines)
                        surface_count = len(drawing.surface_generator.surfaces)
                        drawing.show_message(f"Lines: {line_count}, Surfaces: {surface_count}")
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:  # Toggle drawing mode
                        drawing.toggle_drawing_mode()
                    elif event.key == pygame.K_TAB:  # Toggle active grid
                        drawing.toggle_active_grid()
                    elif event.key == pygame.K_v:  # Toggle vertex visibility
                        drawing.show_vertices = not drawing.show_vertices
                        drawing.show_message(f"Vertices {'visible' if drawing.show_vertices else 'hidden'}")
                    elif event.key == pygame.K_SPACE:  # Toggle pottery rotation
                        if drawing.drawing_mode == MODE_POTTERY or drawing.drawing_mode == MODE_PROFILE_POTTERY:
                            drawing.toggle_pottery_rotation()
                            # Update the rotation state for profile pottery
                            pottery_wheel_rotation = drawing.pottery_rotation_active
                        else:
                            drawing.show_message("Pottery rotation only available in pottery modes")
                    elif event.key == pygame.K_p:  # Toggle profile display in profile pottery mode
                        if drawing.drawing_mode == MODE_PROFILE_POTTERY:
                            drawing.surface_generator.show_profile = not drawing.surface_generator.show_profile
                            drawing.show_message(f"Profile {'visible' if drawing.surface_generator.show_profile else 'hidden'}")
                    elif event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):  # Ctrl+Z for undo
                        # Undo for profile pottery mode
                        if drawing.drawing_mode == MODE_PROFILE_POTTERY:
                            if drawing.surface_generator.undo_profile():
                                drawing.show_message("Undid last profile")
                            else:
                                drawing.show_message("No previous profile to undo")
                elif event.type == pygame.KEYUP:
                    if event.key in keys_pressed:
                        keys_pressed[event.key] = False

            # Keyboard control for movement
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

            # If not paused, capture and process video
            if not cap_paused:
                # Capture video frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Unable to capture frame")
                    break

                # Flip image horizontally
                frame = cv2.flip(frame, 1)

                # Convert image to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(frame_rgb)
            else:
                # If paused, reset results
                results = None

            # Reset OpenGL drawing area
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            # Apply camera transformations
            glTranslatef(*camera.position)
            glRotatef(camera.rotation_x, 1, 0, 0)
            glRotatef(camera.rotation_y, 0, 1, 0)

            # Draw 3D axes for reference
            glDisable(GL_LIGHTING)  # Disable lighting for axes
            glBegin(GL_LINES)
            # X-axis (Red)
            glColor3f(1, 0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(5, 0, 0)
            # Y-axis (Green)
            glColor3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 5, 0)
            # Z-axis (Blue)
            glColor3f(0, 0, 1)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 5)
            glEnd()
            glEnable(GL_LIGHTING)  # Re-enable lighting

            # Update pottery rotation if active
            drawing.update()

            # If hands are detected and capture is not paused
            if results and results.multi_hand_landmarks and not cap_paused:
                # Separate left and right hands
                left_hand = None
                right_hand = None
                for idx, hand_label in enumerate(results.multi_handedness):
                    if hand_label.classification[0].label == "Left":
                        left_hand = results.multi_hand_landmarks[idx]
                    else:
                        right_hand = results.multi_hand_landmarks[idx]

                # Navigation handling (left hand)
                if left_hand:
                    thumb_tip = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # Calculate distance between thumb and index
                    distance = math.sqrt(
                        (thumb_tip.x - index_tip.x)**2 +
                        (thumb_tip.y - index_tip.y)**2 +
                        (thumb_tip.z - index_tip.z)**2
                    )

                    # Define point for navigation
                    hand_point = type('HandPoint', (), {'x': thumb_tip.x, 'y': thumb_tip.y, 'z': thumb_tip.z})()

                    # Pinch threshold for navigation
                    PINCH_THRESHOLD = 0.05

                    # Current pinch state
                    currently_pinched = distance < PINCH_THRESHOLD

                    # Handle pinch states
                    if currently_pinched and not left_hand_pinched:
                        # Start a new pinch
                        drawing.start_navigation(hand_point)
                    elif currently_pinched and left_hand_pinched:
                        # Continuous pinch - navigate
                        drawing.navigate(hand_point)
                    elif not currently_pinched and left_hand_pinched:
                        # End of pinch
                        drawing.stop_navigation()

                    # Update pinch state
                    left_hand_pinched = currently_pinched

                # Drawing handling (right hand)
                if right_hand:
                    thumb_tip = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = right_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    # Calculate distances between fingers
                    thumb_index_distance = math.sqrt(
                        (thumb_tip.x - index_tip.x)**2 +
                        (thumb_tip.y - index_tip.y)**2 +
                        (thumb_tip.z - index_tip.z)**2
                    )

                    thumb_middle_distance = math.sqrt(
                        (thumb_tip.x - middle_tip.x)**2 +
                        (thumb_tip.y - middle_tip.y)**2 +
                        (thumb_tip.z - middle_tip.z)**2
                    )

                    # For index finger pinch (adds material or draws)
                    index_midpoint_x = (thumb_tip.x + index_tip.x) / 2
                    index_midpoint_y = (thumb_tip.y + index_tip.y) / 2
                    index_midpoint_z = (thumb_tip.z + index_tip.z) / 2

                    # For middle finger pinch (subtracts material)
                    middle_midpoint_x = (thumb_tip.x + middle_tip.x) / 2
                    middle_midpoint_y = (thumb_tip.y + middle_tip.y) / 2
                    middle_midpoint_z = (thumb_tip.z + middle_tip.z) / 2

                    # Create landmark-like objects for midpoints
                    index_midpoint = type('MidpointLandmark', (), {
                        'x': index_midpoint_x,
                        'y': index_midpoint_y,
                        'z': index_midpoint_z
                    })()

                    middle_midpoint = type('MidpointLandmark', (), {
                        'x': middle_midpoint_x,
                        'y': middle_midpoint_y,
                        'z': middle_midpoint_z
                    })()

                    # Transform drawing points
                    index_drawing_point = transform_hand_point(index_midpoint, camera)
                    middle_drawing_point = transform_hand_point(middle_midpoint, camera)

                    # Pinch threshold
                    PINCH_THRESHOLD = 0.05

                    # Current pinch states
                    currently_index_pinched = thumb_index_distance < PINCH_THRESHOLD
                    currently_middle_pinched = thumb_middle_distance < PINCH_THRESHOLD

                    # Handle pottery sculpting mode (add/subtract)
                    if drawing.drawing_mode == MODE_POTTERY:
                        # Choose the appropriate sculpt mode based on which finger is pinched
                        if currently_index_pinched:
                            drawing.sculpt_mode = 0  # Add material
                            drawing.set_midpoint(index_drawing_point)
                        elif currently_middle_pinched:
                            drawing.sculpt_mode = 1  # Subtract material
                            drawing.set_midpoint(middle_drawing_point)
                        
                        # For pottery mode, handle pinch states for adding material
                        if currently_index_pinched and not right_hand_index_pinched:
                            drawing.start_new_line(index_drawing_point)
                        elif currently_index_pinched and right_hand_index_pinched:
                            drawing.add_point(index_drawing_point)
                        elif not currently_index_pinched and right_hand_index_pinched:
                            drawing.stop_drawing()

                        # For pottery mode, handle pinch states for subtracting material
                        if currently_middle_pinched and not right_hand_middle_pinched:
                            drawing.start_new_line(middle_drawing_point)
                        elif currently_middle_pinched and right_hand_middle_pinched:
                            drawing.add_point(middle_drawing_point)
                        elif not currently_middle_pinched and right_hand_middle_pinched:
                            drawing.stop_drawing()
                    else:
                        # For other drawing modes, just use index finger
                        drawing.set_midpoint(index_drawing_point)
                        
                        if currently_index_pinched and not right_hand_index_pinched:
                            drawing.start_new_line(index_drawing_point)
                        elif currently_index_pinched and right_hand_index_pinched:
                            drawing.add_point(index_drawing_point)
                        elif not currently_index_pinched and right_hand_index_pinched:
                            drawing.stop_drawing()

                    # Update pinch states
                    right_hand_index_pinched = currently_index_pinched
                    right_hand_middle_pinched = currently_middle_pinched

            # Draw lines and surfaces
            glDisable(GL_LIGHTING)  # Disable lighting for wireframes
            drawing.draw()
            glEnable(GL_LIGHTING)   # Re-enable lighting

            # Update on-screen message
            drawing.update_message()

            # Update display
            pygame.display.flip()

            # Draw text message
            render_text(display, text_surface, drawing.message)

            # Limit to 60 FPS
            clock.tick(60)

    except Exception as e:
        print(f"Fatal error: {e}")
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