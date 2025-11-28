import streamlit as st
import tensorflow.lite as tflite
from PIL import Image, ImageOps
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Minecraft Scanner", page_icon="⛏️")

st.title("⛏️ Scanner de Mobs (TFLite)")
st.write("Versión optimizada con TensorFlow Lite.")


# --- CARGAR MODELO TFLITE ---
@st.cache_resource
def load_tflite_model():
    # Carga el intérprete (el cerebro de la IA)
    interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    return interpreter


try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    st.stop()

# --- CARGAR ETIQUETAS ---
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    class_names = ["Clase 0", "Clase 1"]

# --- DESCRIPCIONES ---
descripciones = {
    "Zombie": "🧟 **Zombie**: Hostil. Se quema con el sol.",
    "Aldeano": "🧑‍🌾 **Aldeano**: Pacífico. Sirve para comerciar.",
}

# --- CÁMARA ---
camera_image = st.camera_input("📸 Foto")

if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

    # --- PREPROCESAMIENTO ---
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Normalizar
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Preparar datos para TFLite
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # --- PREDICCIÓN CON TFLITE ---
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    nombre_limpio = class_name[2:] if class_name[0].isdigit() else class_name
    nombre_limpio = nombre_limpio.strip()

    # --- RESULTADOS ---
    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, width=150)

    with col2:
        if confidence_score > 0.6:
            st.success(f"¡Es un **{nombre_limpio}**!")
            st.caption(f"Confianza: {confidence_score:.1%}")
            st.info(descripciones.get(nombre_limpio, ""))
        else:
            st.warning(f"Parece un {nombre_limpio}...")
