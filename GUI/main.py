import customtkinter
from tkinter import messagebox, simpledialog
import cv2 as cv
import os
import imutils
import numpy as np
from datetime import datetime
from time import time
from PIL import Image, ImageTk

# Configuración de apariencia
customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

# Crear la ventana principal
app = customtkinter.CTk()
app.state('zoomed')  # Maximizar la ventana principal
app.resizable(False, False)  # Deshabilitar el redimensionado
app.title("Sistema de Reconocimiento Facial")

# Función para abrir el formulario de captura de rostros
def abrir_fase_captura():
    app.withdraw()  # Oculta el formulario principal
    form_captura.deiconify()  # Muestra el formulario de captura

    nombre_estudiante = simpledialog.askstring("Nombre del Estudiante", "Ingrese el nombre del estudiante:", parent=form_captura)
    if not nombre_estudiante:
        messagebox.showerror("Error", "Debe ingresar un nombre de estudiante.")
        volver_al_principal(form_captura)
        return

    ruta1 = 'C:/Users/USUARIO/Downloads/Proyecto/reconocimientofacial1/Data'
    rutacompleta = os.path.join(ruta1, nombre_estudiante)
    if not os.path.exists(rutacompleta):
        os.makedirs(rutacompleta)

    camara = cv.VideoCapture(0)
    ruidos = cv.CascadeClassifier(r'C:\Users\USUARIO\Downloads\Proyecto\entrenamientos opencv ruidos\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
    id = 0

    label_captura_video.configure(text="Espere mientras toma las fotos...")

    def mostrar_camara():
        nonlocal id
        respuesta, captura = camara.read()
        if not respuesta:
            return
        captura = imutils.resize(captura, width=640)
        grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
        idcaptura = captura.copy()
        cara = ruidos.detectMultiScale(grises, 1.3, 5)
        for (x, y, e1, e2) in cara:
            cv.rectangle(captura, (x, y), (x + e1, y + e2), (0, 255, 0), 2)
            rostrocapturado = idcaptura[y:y + e2, x:x + e1]
            rostrocapturado = cv.resize(rostrocapturado, (160, 160), interpolation=cv.INTER_CUBIC)
            cv.imwrite(os.path.join(rutacompleta, f'imagen_{id}.jpg'), rostrocapturado)
            id += 1
        img = cv.cvtColor(captura, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        label_captura_video.imgtk = img
        label_captura_video.configure(image=img)
        if id < 350:
            label_captura_video.after(10, mostrar_camara)
        else:
            camara.release()
            cv.destroyAllWindows()
            messagebox.showinfo("Registro Completo", "El registro del rostro ha finalizado.")
            label_captura_video.configure(image='', text='')  # Quitar la cámara del formulario

    mostrar_camara()

# Función para abrir el formulario de entrenamiento
def abrir_fase_entrenamiento():
    # Deshabilitar botones
    button_captura.configure(state="disabled")
    button_entrenamiento.configure(state="disabled")
    button_validacion.configure(state="disabled")

    dataRuta = 'C:/Users/USUARIO/Downloads/Proyecto/reconocimientofacial1/Data'
    listaData = os.listdir(dataRuta)
    ids = []
    rostrosData = []
    id = 0
    tiempoInicial = time()
    progress_bar = customtkinter.CTkProgressBar(master=app, width=400)
    progress_bar.place(relx=0.5, rely=0.4, anchor=customtkinter.CENTER)
    progress_bar.set(0)
    app.update_idletasks()

    for i, fila in enumerate(listaData):
        rutacompleta = os.path.join(dataRuta, fila)
        for archivo in os.listdir(rutacompleta):
            ids.append(id)
            rostrosData.append(cv.imread(os.path.join(rutacompleta, archivo), 0))
        id += 1
        progress_bar.set(i / len(listaData))
        app.update_idletasks()

    entrenamientoEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
    entrenamientoEigenFaceRecognizer.train(rostrosData, np.array(ids))
    entrenamientoEigenFaceRecognizer.write('EntrenamientoEigenFaceRecognizer.xml')
    tiempoTotalEntrenamiento = time() - tiempoInicial
    progress_bar.set(1)
    app.update_idletasks()
    progress_bar.destroy()
    messagebox.showinfo("Entrenamiento Completo", f"El entrenamiento ha finalizado en {tiempoTotalEntrenamiento:.2f} segundos.")

    # Habilitar botones
    button_captura.configure(state="normal")
    button_entrenamiento.configure(state="normal")
    button_validacion.configure(state="normal")

# Función para abrir el formulario de validación
def abrir_fase_validacion():
    app.withdraw()  # Oculta el formulario principal
    form_validacion.deiconify()  # Muestra el formulario de validación
    dataRuta = 'C:/Users/USUARIO/Downloads/Proyecto/reconocimientofacial1/Data'
    listaData = os.listdir(dataRuta)
    nombres = {i: nombre for i, nombre in enumerate(listaData)}
    entrenamientoEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
    entrenamientoEigenFaceRecognizer.read('EntrenamientoEigenFaceRecognizer.xml')
    ruidos = cv.CascadeClassifier(r'C:\Users\USUARIO\Downloads\Proyecto\entrenamientos opencv ruidos\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
    camara = cv.VideoCapture(0)

    def mostrar_camara():
        respuesta, captura = camara.read()
        if not respuesta:
            return
        captura = imutils.resize(captura, width=640)
        grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
        idcaptura = grises.copy()
        cara = ruidos.detectMultiScale(grises, 1.3, 5)
        for (x, y, e1, e2) in cara:
            rostrocapturado = idcaptura[y:y + e2, x:x + e1]
            rostrocapturado = cv.resize(rostrocapturado, (160, 160), interpolation=cv.INTER_CUBIC)
            resultado = entrenamientoEigenFaceRecognizer.predict(rostrocapturado)
            nombre = nombres.get(resultado[0], "Desconocido")
            color = (0, 255, 0) if resultado[1] < 8000 else (0, 0, 255)
            cv.putText(captura, nombre, (x, y-20), 2, 1.1, color, 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x + e1, y + e2), color, 2)
        img = cv.cvtColor(captura, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        label_validacion_video.imgtk = img
        label_validacion_video.configure(image=img)
        label_validacion_video.after(10, mostrar_camara)

    mostrar_camara()

# Función para regresar al formulario principal desde cualquier fase
def volver_al_principal(form):
    form.withdraw()  # Oculta el formulario actual (fase)
    app.deiconify()  # Muestra el formulario principal

# Creación de formularios para las 3 fases
form_captura = customtkinter.CTkToplevel()
form_captura.state('zoomed')  # Maximizar la ventana secundaria
form_captura.resizable(False, False)  # Deshabilitar redimensionado
form_captura.title("Fase de Captura de Rostros")
form_captura.withdraw()  # Ocultar al inicio

form_validacion = customtkinter.CTkToplevel()
form_validacion.state('zoomed')  # Maximizar la ventana secundaria
form_validacion.resizable(False, False)  # Deshabilitar redimensionado
form_validacion.title("Fase de Validación")
form_validacion.withdraw()  # Ocultar al inicio

# Contenido del formulario principal
label_principal = customtkinter.CTkLabel(master=app, text="Sistema de Reconocimiento Facial", font=("Arial", 24))
label_principal.place(relx=0.5, rely=0.3, anchor=customtkinter.CENTER)

button_captura = customtkinter.CTkButton(master=app, text="Fase de Captura de Rostros", command=abrir_fase_captura)
button_captura.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)

button_entrenamiento = customtkinter.CTkButton(master=app, text="Fase de Entrenamiento", command=abrir_fase_entrenamiento)
button_entrenamiento.place(relx=0.5, rely=0.6, anchor=customtkinter.CENTER)

button_validacion = customtkinter.CTkButton(master=app, text="Fase de Validación", command=abrir_fase_validacion)
button_validacion.place(relx=0.5, rely=0.7, anchor=customtkinter.CENTER)

# Contenido del formulario de captura de rostros
label_captura = customtkinter.CTkLabel(master=form_captura, text="Captura de Rostros", font=("Arial", 24))
label_captura.place(relx=0.5, rely=0.1, anchor=customtkinter.CENTER)

label_captura_video = customtkinter.CTkLabel(master=form_captura)
label_captura_video.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)

button_volver_captura = customtkinter.CTkButton(master=form_captura, text="Volver", command=lambda: volver_al_principal(form_captura))
button_volver_captura.place(relx=0.5, rely=0.9, anchor=customtkinter.CENTER)

# Contenido del formulario de validación
label_validacion = customtkinter.CTkLabel(master=form_validacion, text="Validación de Rostros", font=("Arial", 24))
label_validacion.place(relx=0.5, rely=0.1, anchor=customtkinter.CENTER)

label_validacion_video = customtkinter.CTkLabel(master=form_validacion)
label_validacion_video.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)

button_volver_validacion = customtkinter.CTkButton(master=form_validacion, text="Volver", command=lambda: volver_al_principal(form_validacion))
button_volver_validacion.place(relx=0.5, rely=0.9, anchor=customtkinter.CENTER)

# Iniciar la ventana de la aplicación principal
app.mainloop()