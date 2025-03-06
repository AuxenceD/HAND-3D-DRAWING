# HAND-3D-DRAWING
This repository is all our work during the microchallenge II week in the master of IAAC, MDEF. We tried to draw in 3D with only using hands, with the hands recognition and we used python, mediapipe, opencv and pygame.


# Introduction
This documentation explores the different stages of our project, developed during the Microchallenge II workshop at IAAC in the MDEF master's program. Our goal was to create an algorithm that allows real-time 3D drawing, visualized as a hologram, and later transformed into a printable 3D model.

To achieve this, we experimented with various software tools and programming languages, refining our approach to find the most effective solution. While the project is not perfect, it marks an exciting starting point and has been a valuable learning experience for our team.

The idea emerged from the fusion of two previous projects. One involved 2D drawing using hand gestures, with AI recognizing movements and a pressure sensor adjusting line thickness and color. The other focused on holographic visualization. Combining these concepts, we aimed to create an accessible alternative to VR and AR tools—one that feels more intuitive and hands-on, reducing the barrier between the user and the machine.

In this documentation, we will walk through each step of our journey, from concept to execution, sharing the challenges, discoveries, and insights gained along the way.

# First Approaches with p5.js
In the initial phase of our project, we explored different ways to implement 3D drawing using p5.js. Each of us focused on different aspects: one worked on 2D drawing with hand gesture recognition, while the other explored 3D drawing in space.

One of the main challenges we faced was that p5.js struggled with depth perception. When attempting to draw in 3D using the mouse, the system often interpreted strokes as being on the same plane, making it difficult to create true depth. To address this, we tried implementing a movable plane, allowing users to draw on a surface that could be repositioned. However, this solution also proved unreliable and difficult to control.

Eventually, we managed to create a form of 3D drawing, but the results were unpredictable and inconsistent. The lack of precision and control made it clear that p5.js had significant limitations for our vision, pushing us to explore alternative tools and methods.

So after all these tries we decided to move to use other software.

# Exploring Three.js
After facing limitations with p5.js, we turned to Three.js, a library specifically designed for 3D rendering. At first, it seemed like a promising tool for our project, offering greater flexibility and precision for creating 3D drawings in space.

However, we quickly encountered a major challenge: we were unfamiliar with Three.js, and integrating it with MediaPipe and OpenCV—which are primarily used with Python—proved to be difficult. Despite exploring various resources, we found little that directly addressed our specific needs.

Due to these obstacles and the steep learning curve, we decided to abandon this approach early on and look for alternative solutions that better suited our workflow and technical constraints.

# Running Python, OpenCV, and MediaPipe on Blender
Another approach we explored was integrating Python, OpenCV, and MediaPipe into Blender to draw lines directly within the 3D environment. This seemed like an interesting solution, as Blender provides a powerful 3D space to work with, and using Python would allow us to leverage hand-tracking and computer vision tools more effectively.

However, we quickly ran into compatibility issues. Blender required specific versions of Python and OpenCV, while MediaPipe had its own version dependencies. Trying to make everything work together turned into a frustrating process, as resolving one issue often created another. We spent a lot of time troubleshooting these conflicts but made little progress toward our actual goal.

Due to these constant technical barriers, we decided not to run our Python code directly inside Blender and instead looked for alternative workflows to achieve our desired results.

# Drawing in 3D with Python and Pygame
After multiple iterations and unsuccessful attempts with other tools, we finally managed to draw in 3D space using Python and Pygame. By leveraging Pygame to visualize a 3D environment, we were able to create an interactive space where users could draw freely.

Our first breakthrough was successfully drawing 3D lines using the mouse while allowing real-time rotation of the environment. This was a major improvement over our previous attempts with p5.js, as it gave us full control over depth and perspective. However, achieving this functionality required many iterations before we got it to work properly. In our GitHub repository, this functionality is implemented in the file draw3d.py.

Once we had basic 3D drawing with the mouse, we moved on to a more advanced interaction method: drawing and controlling the camera using hand tracking with MediaPipe and OpenCV. After numerous refinements, we reached a somewhat convincing result, though there is still significant room for improvemen

The core of our system relies on a dynamic drawing plane, which is always perpendicular to the camera’s normal. When the camera moves, the plane moves accordingly, ensuring that the drawing surface adapts to the user’s perspective. The plane’s coordinates are stored in a matrix and continuously updated based on the camera’s orientation.

For gesture-based drawing, we detect the distance between the thumb and index finger. If this distance falls below a predefined threshold—meaning the user is making a "pinching" motion—points are created and connected with lines, forming a continuous drawing.

To enhance functionality, we also implemented:

Screenshot capture to save different stages of the drawing.
3D object import, allowing users to draw around existing models.
Export to .obj format, making it possible to open and refine the drawings in software like Blender or Rhino.
This approach finally gave us the flexibility and depth control we had been searching for, bringing us closer to our original vision of an intuitive, hand-controlled 3D drawing tool.

This final code is disponible in our github in the name of essai13d.py

# Exploring New Applications: Digital Pottery
Since we reached our initial goal earlier than expected, we decided to explore alternative ways to use our 3D drawing system. One idea that emerged was simulating a pottery wheel to create digital pottery forms.

To achieve this, we modified our environment so that it rotates at a constant speed, mimicking the motion of a traditional pottery wheel. As we drew in space, our strokes formed circular patterns, allowing us to shape something resembling a 3D pottery form.

However, we quickly encountered a limitation: our drawings were still just lines, not real 3D surfaces. This meant that, although we could visualize pottery-like shapes, they could not be directly 3D printed.

We conducted research and ran several tests in Blender to find a way to convert these lines into solid 3D models, but despite our efforts, we were unable to achieve a satisfying result. This remains an open challenge for further exploration.

You can find this code in the repository at the name of potterie2.py .
