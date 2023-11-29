import cv2
import numpy as np

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Inicializar el detector de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Crear una ventana para dibujar
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
drawing_mode = False
drawing_coords = []
drawing_radius = 10  # Radio inicial del círculo
drawing_color = (0, 255, 0)  # Color inicial (verde)

# Factor de escala inicial (puedes ajustarlo según tus necesidades)
initial_scale_factor = 0.8

# Función de retroalimentación del mouse
def draw_circle(event, x, y, flags, param):
    global drawing_mode, canvas, drawing_coords, drawing_radius, drawing_color

    if drawing_mode:
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(canvas, (x, y), drawing_radius, drawing_color, -1)
            drawing_coords.append((x, y))

# Configurar la ventana y vincular la función del mouse
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', draw_circle)

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()

    # Convertir a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Modo de dibujo
    if drawing_mode:
        # Crear una capa gris semi transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 480), (128, 128, 128), -1)

        # Agregar un cuadrado en el centro para indicar dónde dibujar
        center_square_size = 300
        center_square = ((640 - center_square_size) // 2, (480 - center_square_size) // 2,
                         (640 + center_square_size) // 2, (480 + center_square_size) // 2)
        cv2.rectangle(overlay, (center_square[0], center_square[1]), (center_square[2], center_square[3]), (0, 0, 255), 2)

        result = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Dibujar el contenido del dibujo en la posición actual
        for coord in drawing_coords:
            cv2.circle(result, coord, drawing_radius, drawing_color, -1)

    else:
        result = frame.copy()

        # Mover y escalar el dibujo con la cara detectada solo después de salir del modo de dibujo
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Tomar la primera cara
            face_center = (x + w // 2, y + h // 2)

            # Calcular el factor de escala basado en el tamaño de la cara
            scale_factor = initial_scale_factor * (w + h) / 400.0  # Puedes ajustar 400 según tus necesidades

            for coord in drawing_coords:
                adjusted_coord = (
                    int(face_center[0] + scale_factor * (coord[0] - 320)),
                    int(face_center[1] + scale_factor * (coord[1] - 240))
                )
                cv2.circle(result, adjusted_coord, int(drawing_radius * scale_factor), drawing_color, -1)

    # Mostrar la imagen resultante
    cv2.imshow('Camera', result)

    # Capturar la tecla presionada
    key = cv2.waitKey(1)

    # Cambiar al modo de dibujo cuando se presiona la tecla 'p'
    if key == ord('p'):
        drawing_mode = not drawing_mode

        # Limpiar el dibujo cuando cambias al modo de dibujo
        if drawing_mode:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            drawing_coords = []  # Limpiar las coordenadas del dibujo cuando entras en el modo de dibujo

    # Cambiar el color al hacer clic con el botón derecho
    elif key == ord('r'):
        drawing_color = (0, 0, 255)  # Rojo
    elif key == ord('g'):
        drawing_color = (0, 255, 0)  # Verde
    elif key == ord('b'):
        drawing_color = (255, 0, 0)  # Azul
    elif key == ord('m'):
        drawing_color = (255, 0, 255)  # Morado
    elif key == ord('y'):
        drawing_color = (0, 255, 255)  # Amarillo

    # Salir del bucle si se presiona la tecla 'q'
    elif key == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
