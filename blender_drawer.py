import bpy
import json
import os
import time

# Chemin vers le fichier de coordonnées (même répertoire ou spécifiez le chemin complet)
file_path = "hand_coordinates.txt"

# Vérifier si le fichier existe
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier {file_path} n'existe pas. Exécutez d'abord hand_tracker.py.")

# Lire les coordonnées depuis le fichier
with open(file_path, 'r') as f:
    points = json.load(f)

# Vérifier si nous avons suffisamment de points
if len(points) < 2:
    raise ValueError(f"Pas assez de points pour créer une ligne ({len(points)} trouvés, besoin d'au moins 2)")

print(f"Création d'une ligne avec {len(points)} points")

# Supprimer tous les objets existants (optionnel)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Créer une courbe Bézier
curve_data = bpy.data.curves.new('LigneDeLaMain', type='CURVE')
curve_data.dimensions = '3D'
curve_data.resolution_u = 12

# Créer l'objet courbe et le lier à la scène
curve_object = bpy.data.objects.new('LigneDeLaMain', curve_data)
bpy.context.collection.objects.link(curve_object)

# Créer un spline dans la courbe
spline = curve_data.splines.new(type='POLY')  # POLY pour ligne droite entre points

# Définir le nombre de points
spline.points.add(len(points) - 1)  # -1 car il y a déjà un point par défaut

# Facteur d'échelle pour rendre les mouvements plus visibles
scale_factor = 10

# Ajuster les coordonnées pour Blender (y et z sont souvent inversés)
for i, point in enumerate(points):
    # Conversion de l'espace caméra à l'espace Blender
    # Les coordonnées x, y sont normalisées entre 0 et 1, nous les étendons
    # La coordonnée z est négative (plus proche = plus grand en valeur absolue)
    x = (point[0] - 0.5) * scale_factor  # centrer et mettre à l'échelle
    y = (point[2]) * scale_factor * -5  # utiliser z comme profondeur, inverser et mettre à l'échelle
    z = (0.5 - point[1]) * scale_factor  # inverser y et mettre à l'échelle
    
    # Attribuer les coordonnées (x, y, z, w)
    spline.points[i].co = (x, y, z, 1)

# Définir le matériau (couleur)
mat = bpy.data.materials.new(name="CouleurLigne")
mat.diffuse_color = (0, 0.8, 1, 1)  # Bleu clair (R,G,B,A)
curve_object.data.materials.append(mat)

# Définir l'épaisseur
curve_object.data.bevel_depth = 0.1

# Mettre à jour la vue
bpy.context.view_layer.update()

print("Ligne créée avec succès!")