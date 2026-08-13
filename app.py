import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Test Definitivo: Pokémon y Gimnasio", page_icon="🔮", layout="centered")

st.title("Escáner Psicológico Pokémon")
st.markdown("Responde a estas 50 situaciones. El algoritmo analizará 8 dimensiones de tu personalidad y cruzará tus mecánicas de juego para encontrar tu especie exacta y tu especialidad como Líder de Gimnasio.")

# --- 2. CARGAR BASE DE DATOS (.JSON) ---
@st.cache_data
def cargar_datos():
    # Asegúrate de que este nombre coincide con el archivo JSON que subiste a GitHub
    return pd.read_json("vectores_pokemon_absoluto.json")

df_pokemon = cargar_datos()
# Las 8 dimensiones exactas que generamos en el laboratorio
columnas_dimensiones = ['Agresividad', 'Sociabilidad', 'Energia', 'Intelecto', 'Misticismo', 'Lealtad', 'Caos', 'Orgullo']

# --- 3. BASE DE DATOS DE 50 PREGUNTAS (PESOS CRUZADOS) ---
# Orden del array: [Agresividad, Sociabilidad, Energia, Intelecto, Misticismo, Lealtad, Caos, Orgullo]
preguntas_test = [
    # BLOQUE 1: Dilemas Cotidianos y Sociales
    {"pregunta": "1. Estás en un grupo de trabajo y alguien no hace su parte. Tú...", "opciones": {
        "A) Le expongo frente al resto y asumo el mando": {"perfil": [3, 0, 1, 0, 0, 0, 0, 2], "tipos": ["Fuego", "Lucha"]},
        "B) Cubro su parte en silencio para que el proyecto no fracase": {"perfil": [-1, 2, 0, 0, 0, 3, 0, -1], "tipos": ["Hada", "Normal"]},
        "C) Rediseño la estructura para aislar su parte y que no nos afecte": {"perfil": [0, -1, 0, 4, 0, 0, 1, 0], "tipos": ["Acero", "Psíquico"]},
        "D) Saboteo sutilmente su nota final sin que se entere": {"perfil": [1, -2, 0, 2, 1, 0, 3, 0], "tipos": ["Siniestro", "Veneno"]}}},
    {"pregunta": "2. Un amigo íntimo te pide consejo a las 3 AM por un problema amoroso...", "opciones": {
        "A) Le escucho pacientemente hasta que se duerma": {"perfil": [-2, 4, -1, 0, 0, 3, 0, 0], "tipos": ["Agua", "Hada"]},
        "B) Le digo las verdades que duelen para que espabile": {"perfil": [2, 0, 0, 1, 0, 1, 0, 1], "tipos": ["Roca", "Hielo"]},
        "C) Le propongo salir de fiesta ahora mismo para que se olvide": {"perfil": [0, 2, 4, 0, 0, 1, 2, 0], "tipos": ["Eléctrico", "Fuego"]},
        "D) Analizo los patrones de comportamiento de su pareja objetivamente": {"perfil": [0, -1, 0, 4, 0, 0, 0, 0], "tipos": ["Psíquico", "Acero"]}}},
    {"pregunta": "3. Se va la luz en todo tu barrio durante una tormenta...", "opciones": {
        "A) Me quejo en alto e intento buscar velas tropezando con todo": {"perfil": [1, 0, 2, -1, 0, 0, 2, 0], "tipos": ["Fuego", "Eléctrico"]},
        "B) Aprovecho para dormir o relajarme escuchando la lluvia": {"perfil": [-2, 0, -3, 0, 0, 0, 0, 0], "tipos": ["Agua", "Planta"]},
        "C) Excelente. Me encanta la oscuridad absoluta": {"perfil": [0, -2, 0, 0, 4, 0, 1, 0], "tipos": ["Fantasma", "Siniestro"]},
        "D) Enciendo mi batería externa y sigo con lo mío organizado": {"perfil": [0, 0, 0, 3, 0, 0, -1, 1], "tipos": ["Acero", "Normal"]}}},
    {"pregunta": "4. Estás organizando los gastos del mes con tus compañeros...", "opciones": {
        "A) Hago un Excel milimétrico donde cuadra hasta el último céntimo": {"perfil": [0, 0, 0, 4, 0, 1, -2, 1], "tipos": ["Acero", "Psíquico"]},
        "B) Pago yo un poco más si hace falta para evitar discusiones": {"perfil": [-1, 3, 0, 0, 0, 2, 0, -1], "tipos": ["Hada", "Agua"]},
        "C) Propongo un bote común, confío en que nadie se pase": {"perfil": [0, 3, 1, -1, 0, 1, 1, 0], "tipos": ["Normal", "Planta"]},
        "D) Me escaqueo de las compras comunes si puedo": {"perfil": [0, -2, 0, 1, 0, -2, 3, 0], "tipos": ["Veneno", "Siniestro"]}}},
    {"pregunta": "5. En una fiesta donde hay mucha gente que no conoces...", "opciones": {
        "A) Acabo siendo el centro de atención contando anécdotas": {"perfil": [1, 3, 3, 0, 0, 0, 0, 2], "tipos": ["Fuego", "Eléctrico"]},
        "B) Me pego a la única persona que conozco toda la noche": {"perfil": [0, 1, -1, 0, 0, 3, 0, 0], "tipos": ["Bicho", "Planta"]},
        "C) Me busco una esquina tranquila para observar a los demás": {"perfil": [0, -2, -1, 2, 2, 0, 0, 0], "tipos": ["Fantasma", "Hielo"]},
        "D) Hablo con grupos distintos intentando entender sus dinámicas": {"perfil": [0, 2, 0, 3, 0, 0, 0, 0], "tipos": ["Psíquico", "Agua"]}}},

    # BLOQUE 2: Entretenimiento y Aficiones
    {"pregunta": "6. Eliges un género para leer manga o ver anime...", "opciones": {
        "A) Shonen de peleas, torneos, superación y sangre": {"perfil": [4, 1, 3, 0, 0, 1, 0, 0], "tipos": ["Lucha", "Fuego"]},
        "B) Seinen oscuro, psicológico y con mundos decadentes": {"perfil": [1, -2, 0, 2, 4, 0, 1, 0], "tipos": ["Siniestro", "Fantasma"]},
        "C) Misterio, detectives o ciencia ficción dura": {"perfil": [0, 0, 0, 4, 0, 0, 0, 1], "tipos": ["Acero", "Psíquico"]},
        "D) Slice of life, comedia o romance para desconectar": {"perfil": [-2, 3, 0, 0, 0, 1, 0, 0], "tipos": ["Hada", "Normal"]}}},
    {"pregunta": "7. ¿Cómo es tu rutina ideal de ejercicio físico?", "opciones": {
        "A) Calistenia dura, dominadas y romper mis límites": {"perfil": [3, 0, 3, 1, 0, 0, 0, 2], "tipos": ["Lucha", "Roca"]},
        "B) Correr o saltar a la cuerda, cosas de ritmo muy rápido": {"perfil": [0, 0, 4, 0, 0, 0, 1, 0], "tipos": ["Volador", "Eléctrico"]},
        "C) Estiramientos, yoga, control corporal absoluto": {"perfil": [-1, 0, -1, 2, 1, 0, -1, 0], "tipos": ["Psíquico", "Planta"]},
        "D) Evito el deporte, prefiero entrenar la mente": {"perfil": [-2, 0, -3, 4, 0, 0, 0, 1], "tipos": ["Acero", "Bicho"]}}},
    {"pregunta": "8. A la hora de tocar un instrumento o escuchar música...", "opciones": {
        "A) Batería o percusión: ritmos rápidos, caos y energía": {"perfil": [2, 0, 4, 0, 0, 0, 2, 0], "tipos": ["Eléctrico", "Fuego"]},
        "B) Guitarra o piano: técnica perfecta, teoría musical": {"perfil": [0, 0, 0, 4, 0, 0, -1, 1], "tipos": ["Acero", "Hielo"]},
        "C) Canto o coros: conectar con la emoción y los demás": {"perfil": [-1, 4, 0, 0, 0, 1, 0, 0], "tipos": ["Hada", "Agua"]},
        "D) Sintetizadores pesados, bajos oscuros, ambient": {"perfil": [0, -2, -1, 0, 4, 0, 0, 0], "tipos": ["Siniestro", "Veneno"]}}},
    {"pregunta": "9. Jugando a un videojuego RPG, ¿cuál es tu clase base?", "opciones": {
        "A) Guerrero/Bárbaro: A melee, rompiendo cosas": {"perfil": [4, 0, 2, -1, 0, 1, 0, 1], "tipos": ["Lucha", "Tierra"]},
        "B) Mago/Hechicero: Control de campo y daño en área": {"perfil": [0, 0, 0, 4, 2, 0, 0, 1], "tipos": ["Psíquico", "Fuego"]},
        "C) Pícaro/Asesino: Sigilo, trampas, críticos por la espalda": {"perfil": [2, -2, 1, 1, 1, 0, 3, 0], "tipos": ["Siniestro", "Veneno"]},
        "D) Clérigo/Bardo: Curar, bufos y salvar al equipo": {"perfil": [-2, 4, 0, 0, 1, 4, 0, 0], "tipos": ["Hada", "Planta"]}}},
    {"pregunta": "10. Creas un modpack para un juego. ¿En qué te centras?", "opciones": {
        "A) Gráficos increíbles, animaciones y estética visual": {"perfil": [0, 0, 1, 0, 0, 0, 0, 4], "tipos": ["Hada", "Agua"]},
        "B) Optimización del código, rendimiento y automatización": {"perfil": [0, -1, 0, 4, 0, 0, -2, 1], "tipos": ["Acero", "Eléctrico"]},
        "C) Añadir monstruos durísimos, armas y acción constante": {"perfil": [3, 0, 3, 0, 0, 0, 1, 0], "tipos": ["Dragón", "Lucha"]},
        "D) Magia oscura, dimensiones alternativas y secretos": {"perfil": [0, -1, 0, 1, 4, 0, 2, 0], "tipos": ["Fantasma", "Siniestro"]}}},

    # BLOQUE 3: Reacciones e Instintos
    {"pregunta": "11. Te enteras de que alguien ha hablado mal de ti a tus espaldas...", "opciones": {
        "A) Le encaro inmediatamente y le pido explicaciones": {"perfil": [4, 0, 2, 0, 0, 0, 0, 2], "tipos": ["Fuego", "Lucha"]},
        "B) Me duele, pero intento entender por qué lo ha hecho": {"perfil": [-1, 2, 0, 1, 0, 0, 0, -1], "tipos": ["Agua", "Planta"]},
        "C) Analizo sus debilidades y arruino su reputación sin que se note": {"perfil": [1, -2, 0, 3, 1, 0, 3, 0], "tipos": ["Siniestro", "Veneno"]},
        "D) Le ignoro por completo, no merece mi tiempo": {"perfil": [0, -1, -1, 0, 0, 0, 0, 4], "tipos": ["Hielo", "Dragón"]}}},
    {"pregunta": "12. Tienes que caminar por una calle muy oscura de noche...", "opciones": {
        "A) Voy rápido y alerta, listo para correr o pelear": {"perfil": [2, 0, 3, 0, 0, 0, 0, 0], "tipos": ["Eléctrico", "Lucha"]},
        "B) Me pongo los cascos para aislarme y voy a mi rollo": {"perfil": [0, -1, 0, 0, 0, 0, 1, 0], "tipos": ["Normal", "Acero"]},
        "C) Voy en tensión, imaginando monstruos o cosas ocultas": {"perfil": [0, 0, 1, 2, 3, 0, 0, 0], "tipos": ["Psíquico", "Fantasma"]},
        "D) Me encanta. Las sombras son mi elemento natural": {"perfil": [0, -2, 0, 0, 4, 0, 2, 0], "tipos": ["Siniestro", "Veneno"]}}},
    {"pregunta": "13. Vas andando y un perro callejero te ladra agresivamente...", "opciones": {
        "A) Le grito o finjo tirarle una piedra para intimidarlo": {"perfil": [3, 0, 1, 0, 0, 0, 0, 1], "tipos": ["Fuego", "Roca"]},
        "B) Le hablo suave y le ofrezco la mano para que huela": {"perfil": [-2, 3, 0, 0, 0, 2, 0, 0], "tipos": ["Hada", "Planta"]},
        "C) Mantengo contacto visual frío y me alejo lentamente": {"perfil": [0, -1, 0, 3, 0, 0, 0, 1], "tipos": ["Hielo", "Psíquico"]},
        "D) Salgo corriendo a toda velocidad, paso de problemas": {"perfil": [0, 0, 4, -1, 0, 0, 1, 0], "tipos": ["Volador", "Normal"]}}},
    {"pregunta": "14. Estás montando algo y las piezas no encajan por un milímetro...", "opciones": {
        "A) Aprieto y golpeo hasta que ceda por pura fuerza": {"perfil": [4, 0, 2, -1, 0, 0, 1, 0], "tipos": ["Lucha", "Roca"]},
        "B) Lo dejo, respiro hondo, y vuelvo en otro momento": {"perfil": [-2, 1, -2, 0, 0, 0, 0, 0], "tipos": ["Agua", "Planta"]},
        "C) Mido los ángulos y limo la pieza para que encaje matemáticamente": {"perfil": [0, 0, 0, 4, 0, 0, -2, 1], "tipos": ["Acero", "Psíquico"]},
        "D) Hago una chapuza con pegamento y cruzo los dedos": {"perfil": [0, 0, 1, 0, 0, 0, 4, -1], "tipos": ["Veneno", "Bicho"]}}},
    {"pregunta": "15. ¿Qué te pone más nervioso/a?", "opciones": {
        "A) La inactividad, esperar colas o el aburrimiento": {"perfil": [1, 0, 4, 0, 0, 0, 1, 0], "tipos": ["Eléctrico", "Fuego"]},
        "B) El conflicto, los gritos y la tensión ambiental": {"perfil": [-2, 3, 0, 0, 0, 1, 0, 0], "tipos": ["Hada", "Normal"]},
        "C) La incompetencia, la falta de lógica o el desorden": {"perfil": [1, -1, 0, 4, 0, 0, -2, 1], "tipos": ["Acero", "Psíquico"]},
        "D) Sentirme expuesto, vulnerable o sin escapatoria": {"perfil": [0, -2, 0, 1, 2, -1, 1, 0], "tipos": ["Fantasma", "Siniestro"]}}},

    # BLOQUE 4: Gustos, Estética y Metodología
    {"pregunta": "16. Para crear una página web (ej. un e-commerce)...", "opciones": {
        "A) Uso plantillas llenas de animaciones, color y efectos dinámicos": {"perfil": [0, 2, 3, 0, 0, 0, 1, 1], "tipos": ["Eléctrico", "Hada"]},
        "B) Diseños limpios, claros, minimalistas y fáciles de leer": {"perfil": [-1, 1, 0, 2, 0, 0, -1, 0], "tipos": ["Agua", "Normal"]},
        "C) Me meto en el código fuente, optimizo el CSS y las bases de datos": {"perfil": [0, -1, 0, 4, 0, 0, -1, 1], "tipos": ["Acero", "Psíquico"]},
        "D) Pongo colores oscuros, menús ocultos y diseño misterioso": {"perfil": [0, -2, 0, 1, 3, 0, 1, 0], "tipos": ["Siniestro", "Fantasma"]}}},
    {"pregunta": "17. Para decorar tu espacio de trabajo o habitación...", "opciones": {
        "A) Trofeos, pósters de acción, colores agresivos y neones": {"perfil": [2, 0, 2, 0, 0, 0, 0, 3], "tipos": ["Fuego", "Dragón"]},
        "B) Plantas reales, luz cálida, cojines y tonos tierra": {"perfil": [-2, 2, -1, 0, 0, 0, 0, 0], "tipos": ["Planta", "Bicho"]},
        "C) Simetría perfecta, todo blanco/metálico, sin nada a la vista": {"perfil": [0, -1, 0, 3, 0, 0, -2, 1], "tipos": ["Hielo", "Acero"]},
        "D) Calaveras, cosas antiguas, luz muy tenue y estética gótica": {"perfil": [0, -2, 0, 1, 4, 0, 1, 0], "tipos": ["Fantasma", "Veneno"]}}},
    {"pregunta": "18. ¿Cómo sueles vestir en tu día a día?", "opciones": {
        "A) Ropa deportiva, técnica, ancha y para moverme bien": {"perfil": [1, 0, 3, -1, 0, 0, 0, 0], "tipos": ["Lucha", "Volador"]},
        "B) Ropa muy suave, cómoda, tonos pastel o naturales": {"perfil": [-2, 2, -1, 0, 0, 0, 0, 0], "tipos": ["Normal", "Hada"]},
        "C) Elegante, sobrio, camisas, buscando proyectar respeto": {"perfil": [0, -1, 0, 2, 0, 0, -1, 4], "tipos": ["Hielo", "Dragón"]},
        "D) Tonos oscuros, cuero, cadenas o prendas asimétricas": {"perfil": [1, -2, 0, 0, 3, 0, 2, 0], "tipos": ["Siniestro", "Veneno"]}}},
    {"pregunta": "19. ¿Cuál de estas mascotas tendrías?", "opciones": {
        "A) Un perro grande y fuerte que imponga respeto": {"perfil": [3, 1, 2, 0, 0, 3, 0, 1], "tipos": ["Fuego", "Lucha"]},
        "B) Un perrito pequeño o un conejo súper cariñoso": {"perfil": [-2, 4, 1, 0, 0, 1, 0, 0], "tipos": ["Hada", "Planta"]},
        "C) Un terrario complejo con reptiles o insectos exóticos": {"perfil": [0, -1, 0, 3, 1, 0, 0, 0], "tipos": ["Bicho", "Acero"]},
        "D) Un gato negro, un cuervo o algo independiente": {"perfil": [0, -2, 0, 2, 3, -1, 1, 0], "tipos": ["Fantasma", "Siniestro"]}}},
    {"pregunta": "20. Cuando tienes que estudiar algo pesado...", "opciones": {
        "A) Lo leo en voz alta caminando por la casa sin parar": {"perfil": [0, 0, 4, 1, 0, 0, 1, 0], "tipos": ["Eléctrico", "Volador"]},
        "B) Hago videollamada con amigos para estudiarlo juntos": {"perfil": [-1, 4, 0, 1, 0, 1, 0, 0], "tipos": ["Agua", "Hada"]},
        "C) Mapas mentales, colores, resúmenes hiperestructurados": {"perfil": [0, -1, 0, 4, 0, 0, -2, 1], "tipos": ["Psíquico", "Acero"]},
        "D) Dejo todo para la noche antes y me lo aprendo bajo presión": {"perfil": [1, -1, 2, 0, 0, 0, 3, 0], "tipos": ["Siniestro", "Veneno"]}}},

    # BLOQUE 5: Trabajo en Equipo y Rol
    {"pregunta": "21. Si estuvieras en un atraco de ciencia ficción, serías...", "opciones": {
        "A) El asalto frontal: armas pesadas y explosivos": {"perfil": [4, 0, 2, -1, 0, 1, 1, 0], "tipos": ["Lucha", "Fuego"]},
        "B) El conductor: adrenalina, reflejos y vehículo de huida": {"perfil": [1, 0, 4, 1, 0, 1, 0, 0], "tipos": ["Volador", "Acero"]},
        "C) El cerebro: hackeando cámaras desde una furgoneta segura": {"perfil": [0, -2, 0, 4, 1, 0, -1, 1], "tipos": ["Psíquico", "Hielo"]},
        "D) El infiltrado: disfraces, mentiras y venenos": {"perfil": [1, 2, 0, 2, 1, -1, 3, 0], "tipos": ["Siniestro", "Veneno"]}}},
    {"pregunta": "22. Creando el Lore de un villano para una partida...", "opciones": {
        "A) Un señor de la guerra imparable que busca conquistar todo": {"perfil": [4, 0, 1, 0, 0, 0, 0, 3], "tipos": ["Dragón", "Lucha"]},
        "B) Un líder sectario manipulador que cree estar haciendo el bien": {"perfil": [0, 2, 0, 3, 1, 2, -1, 1], "tipos": ["Hada", "Psíquico"]},
        "C) Una inteligencia artificial fría y desprovista de emociones": {"perfil": [0, -3, 0, 4, 0, 0, -2, 2], "tipos": ["Acero", "Hielo"]},
        "D) Un ente cósmico eldritch incomprensible y oscuro": {"perfil": [0, -2, -1, 2, 4, 0, 3, 0], "tipos": ["Fantasma", "Siniestro"]}}},
    {"pregunta": "23. Durante un debate acalorado...", "opciones": {
        "A) Levanto la voz e impongo mis argumentos con pasión": {"perfil": [3, 0, 2, 0, 0, 0, 0, 2], "tipos": ["Fuego", "Dragón"]},
        "B) Intento mediar y buscar un punto intermedio entre ambas partes": {"perfil": [-2, 3, 0, 1, 0, 1, -1, 0], "tipos": ["Agua", "Normal"]},
        "C) Uso datos irrebatibles y desmonto falacias lógicas en frío": {"perfil": [0, -1, 0, 4, 0, 0, -1, 2], "tipos": ["Acero", "Psíquico"]},
        "D) Lanzo comentarios sarcásticos para confundir al rival": {"perfil": [1, 0, 0, 2, 1, 0, 3, 1], "tipos": ["Veneno", "Siniestro"]}}},
    {"pregunta": "24. En un juego cooperativo donde solo queda un botiquín...", "opciones": {
        "A) Lo pillo yo, confío en mis habilidades para ganar la partida": {"perfil": [2, -1, 1, 0, 0, -1, 0, 4], "tipos": ["Dragón", "Fuego"]},
        "B) Se lo dejo al aliado que esté peor de vida": {"perfil": [-2, 3, 0, 0, 0, 4, 0, -1], "tipos": ["Hada", "Planta"]},
        "C) Analizo matemáticamente quién tiene más probabilidades de usarlo bien": {"perfil": [0, -1, 0, 4, 0, 0, -1, 0], "tipos": ["Acero", "Psíquico"]},
        "D) Lo robo, no aviso a nadie, y me escondo": {"perfil": [0, -3, 0, 1, 1, -2, 3, 0], "tipos": ["Siniestro", "Fantasma"]}}},
    {"pregunta": "25. Tu planazo de vacaciones sería...", "opciones": {
        "A) Irte de aventura, hacer puenting, alpinismo o surf": {"perfil": [2, 1, 4, 0, 0, 0, 0, 0], "tipos": ["Agua", "Roca"]},
        "B) Una casa rural con amigos, barbacoa y juegos de mesa": {"perfil": [-1, 4, -1, 0, 0, 2, 0, 0], "tipos": ["Normal", "Planta"]},
        "C) Visitar museos históricos, tecnología, o rutas culturales precisas": {"perfil": [0, -1, 0, 4, 0, 0, -1, 0], "tipos": ["Acero", "Psíquico"]},
        "D) Rutas por castillos abandonados, catacumbas o lugares con misterio": {"perfil": [0, -2, 0, 2, 4, 0, 1, 0], "tipos": ["Fantasma", "Veneno"]}}},

    # BLOQUE 6: El Entorno Físico y Emocional
    {"pregunta": "26. ¿Con qué paisaje sientes más conexión?", "opciones": {
        "A) Un volcán en erupción o un desierto implacable": {"perfil": [3, -1, 2, 0, 0, 0, 1, 1], "tipos": ["Fuego", "Tierra"]},
        "B) Un prado florecido en primavera con brisa fresca": {"perfil": [-2, 2, -1, 0, 0, 0, 0, 0], "tipos": ["Planta", "Hada"]},
        "C) La geometría perfecta de un glaciar o una montaña nevada": {"perfil": [0, -2, 0, 2, 1, 0, -1, 1], "tipos": ["Hielo", "Acero"]},
        "D) Un bosque denso en plena noche sin luna": {"perfil": [0, -2, 0, 0, 4, 0, 1, 0], "tipos": ["Siniestro", "Fantasma"]}}},
    {"pregunta": "27. Cuando llueve de forma torrencial...", "opciones": {
        "A) Salgo a mojarme o a correr, la tormenta me carga las pilas": {"perfil": [1, 0, 4, 0, 0, 0, 1, 0], "tipos": ["Eléctrico", "Agua"]},
        "B) Me acurruco en el sofá con una bebida caliente": {"perfil": [-2, 1, -3, 0, 0, 0, 0, 0], "tipos": ["Normal", "Fuego"]},
        "C) Me pongo a observar cómo fluye el agua y sus patrones lógicos": {"perfil": [0, -1, 0, 3, 1, 0, -1, 0], "tipos": ["Agua", "Psíquico"]},
        "D) Me gusta el ruido oscuro de los truenos retumbando": {"perfil": [0, -1, 0, 0, 3, 0, 2, 0], "tipos": ["Siniestro", "Dragón"]}}},
    {"pregunta": "28. Un rasgo negativo que la gente podría decir de ti es...", "opciones": {
        "A) Tienes demasiada mala leche o eres impaciente": {"perfil": [4, -1, 2, -1, 0, 0, 0, 1], "tipos": ["Fuego", "Lucha"]},
        "B) Eres un poco vago o demasiado blando con los demás": {"perfil": [-3, 3, -3, 0, 0, 1, 0, -1], "tipos": ["Normal", "Planta"]},
        "C) Pareces un robot frío, distante o pedante": {"perfil": [-1, -3, 0, 4, 0, -1, -1, 2], "tipos": ["Acero", "Hielo"]},
        "D) Eres rencoroso, maquiavélico o muy reservado": {"perfil": [1, -2, 0, 1, 2, -2, 3, 0], "tipos": ["Veneno", "Siniestro"]}}},
    {"pregunta": "29. ¿Qué virtud valoras más en tu entorno?", "opciones": {
        "A) La fuerza, la valentía y el arrojo para tomar riesgos": {"perfil": [3, 0, 2, 0, 0, 2, 0, 1], "tipos": ["Dragón", "Lucha"]},
        "B) La empatía, el cariño y el cuidado mutuo": {"perfil": [-2, 4, 0, 0, 0, 2, 0, -1], "tipos": ["Hada", "Agua"]},
        "C) La inteligencia, el intelecto y el ingenio": {"perfil": [0, -1, 0, 4, 0, 0, 0, 1], "tipos": ["Psíquico", "Acero"]},
        "D) La supervivencia, la astucia y saber engañar al sistema": {"perfil": [1, -2, 0, 2, 1, -1, 4, 0], "tipos": ["Veneno", "Siniestro"]}}},
    {"pregunta": "30. Encuentras una reliquia en una tienda de segunda mano...", "opciones": {
        "A) Una espada antigua, oxidada pero imponente": {"perfil": [3, 0, 1, 0, 1, 1, 0, 2], "tipos": ["Acero", "Lucha"]},
        "B) Un colgante de piedra preciosa con formas orgánicas": {"perfil": [-1, 2, 0, 0, 1, 0, 0, 0], "tipos": ["Roca", "Hada"]},
        "C) Un astrolabio mecánico o un reloj de bolsillo complejo": {"perfil": [0, -1, 0, 4, 0, 0, -1, 1], "tipos": ["Acero", "Psíquico"]},
        "D) Un grimorio gastado, escrito en un idioma incomprensible": {"perfil": [0, -2, 0, 2, 4, 0, 1, 0], "tipos": ["Fantasma", "Siniestro"]}}},

    # BLOQUE 7: Supuestos Fantásticos
    {"pregunta": "31. Si te concedieran un deseo mágico absurdo...", "opciones": {
        "A) Tener superfuerza para romper muros de un puñetazo": {"perfil": [4, 0, 2, -1, 0, 0, 0, 1], "tipos": ["Lucha", "Roca"]},
        "B) Poder hablar con los animales y curarlos": {"perfil": [-2, 3, 0, 1, 1, 2, 0, -1], "tipos": ["Planta", "Hada"]},
        "C) Leer mentes y tener telequinesis": {"perfil": [0, -1, 0, 4, 2, 0, -1, 1], "tipos": ["Psíquico", "Acero"]},
        "D) Convertirte en sombra e invocar pesadillas": {"perfil": [1, -2, 0, 1, 4, -1, 2, 0], "tipos": ["Siniestro", "Fantasma"]}}},
    {"pregunta": "32. Escoge un medio de transporte fantástico:", "opciones": {
        "A) Un dragón rojo que arrase los cielos a tu paso": {"perfil": [3, 0, 2, 0, 1, 1, 0, 4], "tipos": ["Dragón", "Fuego"]},
        "B) Volar levitando tú mismo o sobre una nube mullida": {"perfil": [-1, 1, -1, 0, 1, 0, 0, 0], "tipos": ["Volador", "Hada"]},
        "C) Una nave antigravedad perfectamente calibrada": {"perfil": [0, -1, 3, 3, 0, 0, -2, 1], "tipos": ["Acero", "Eléctrico"]},
        "D) Teletransporte viajando por las grietas del inframundo": {"perfil": [0, -2, 2, 1, 4, 0, 1, 0], "tipos": ["Fantasma", "Veneno"]}}},
    {"pregunta": "33. ¿Qué prefieres en un combate de fantasía?", "opciones": {
        "A) Golpear rápido y de frente, sin miedo al dolor": {"perfil": [4, 0, 3, -1, 0, 2, 0, 1], "tipos": ["Lucha", "Eléctrico"]},
        "B) Proteger a mis aliados tras un escudo inquebrantable": {"perfil": [-1, 2, 0, 1, 0, 4, -1, 1], "tipos": ["Acero", "Roca"]},
        "C) Controlar el campo de batalla: paralizar, dormir, alterar la gravedad": {"perfil": [0, 0, -1, 4, 2, 0, 0, 0], "tipos": ["Psíquico", "Hielo"]},
        "D) Envenenar al rival y verle caer poco a poco desde las sombras": {"perfil": [1, -2, 0, 2, 1, 0, 4, 0], "tipos": ["Veneno", "Siniestro"]}}},
    {"pregunta": "34. Un oráculo te permite ver una escena del futuro...", "opciones": {
        "A) Veo mi mayor victoria y los trofeos que he ganado": {"perfil": [2, 0, 1, 0, 0, 0, 0, 4], "tipos": ["Lucha", "Fuego"]},
        "B) Veo a mi familia y amigos a salvo y felices": {"perfil": [-2, 4, -1, 0, 0, 3, 0, -1], "tipos": ["Agua", "Planta"]},
        "C) Veo los números de la lotería y los avances tecnológicos": {"perfil": [0, -1, 0, 4, 0, 0, -1, 0], "tipos": ["Acero", "Psíquico"]},
        "D) Veo cómo mueren mis enemigos": {"perfil": [2, -3, 0, 1, 3, -1, 3, 1], "tipos": ["Siniestro", "Fantasma"]}}},
    {"pregunta": "35. Eres el jefe de un castillo. Tu primera trampa es...", "opciones": {
        "A) Fosos de lava y rodillos con pinchos que aplastan al intruso": {"perfil": [4, -1, 2, -1, 0, 0, 1, 0], "tipos": ["Fuego", "Acero"]},
        "B) Un laberinto de plantas somníferas que confunden dulcemente": {"perfil": [-1, 1, -1, 1, 1, 0, 1, 0], "tipos": ["Planta", "Hada"]},
        "C) Un puzzle lógico con lásers y espejos de una precisión absurda": {"perfil": [0, -1, 0, 4, 0, 0, -2, 1], "tipos": ["Psíquico", "Eléctrico"]},
        "D) Ilusiones ópticas de los peores miedos del aventurero": {"perfil": [0, -2, 0, 2, 4, 0, 3, 0], "tipos": ["Fantasma", "Siniestro"]}}},

    # BLOQUE 8: Vida Diaria e Instintos Secundarios
    {"pregunta": "36. Frente a tu comida favorita...", "opciones": {
        "A) Me la como súper rápido y devoro grandes raciones": {"perfil": [2, 0, 3, -1, 0, 0, 1, 0], "tipos": ["Dragón", "Lucha"]},
        "B) La comparto y saboreo poco a poco, comiendo sin prisa": {"perfil": [-2, 2, -2, 0, 0, 1, 0, 0], "tipos": ["Normal", "Agua"]},
        "C) Planifico la receta, peso los ingredientes y disfruto el proceso": {"perfil": [0, -1, 0, 3, 0, 0, -1, 0], "tipos": ["Acero", "Psíquico"]},
        "D) Le echo picante extremo o sabores extrañísimos": {"perfil": [1, 0, 1, 0, 1, 0, 3, 0], "tipos": ["Veneno", "Fuego"]}}},
    {"pregunta": "37. Un conocido muy torpe acaba de tropezar y caerse...", "opciones": {
        "A) Me echo a reír a carcajadas sin poder evitarlo": {"perfil": [1, 1, 2, 0, 0, 0, 3, 0], "tipos": ["Eléctrico", "Fuego"]},
        "B) Voy corriendo asustado a ayudarle a levantarse": {"perfil": [-1, 3, 0, 0, 0, 2, -1, 0], "tipos": ["Hada", "Planta"]},
        "C) Le explico por qué el centro de gravedad le ha fallado": {"perfil": [0, -2, 0, 4, 0, 0, -1, 1], "tipos": ["Psíquico", "Acero"]},
        "D) No hago nada. Finjo que no le he visto": {"perfil": [0, -2, -1, 0, 1, 0, 0, 1], "tipos": ["Fantasma", "Hielo"]}}},
    {"pregunta": "38. Tienes que mentir para salvarte de un problema...", "opciones": {
        "A) Me niego, afronto el problema aunque haya pelea": {"perfil": [3, 0, 1, 0, 0, 2, -1, 2], "tipos": ["Lucha", "Dragón"]},
        "B) Miento fatal, me pongo rojo y confieso la verdad llorando": {"perfil": [-2, 1, -1, 0, 0, 1, 0, -1], "tipos": ["Normal", "Agua"]},
        "C) Tejo una mentira con 14 datos falsos irrefutables lógicamente": {"perfil": [0, -1, 0, 4, 0, 0, -1, 0], "tipos": ["Psíquico", "Acero"]},
        "D) Miento con una frialdad y maestría absolutas. Soy experto": {"perfil": [1, -2, 0, 2, 2, -1, 4, 0], "tipos": ["Siniestro", "Veneno"]}}},
    {"pregunta": "39. En tu grupo de amigos tienes fama de ser...", "opciones": {
        "A) El cañero, impulsivo o el que propone locuras": {"perfil": [3, 1, 3, -1, 0, 0, 2, 0], "tipos": ["Fuego", "Eléctrico"]},
        "B) El "mamá/papá" del grupo, que cuida y abraza": {"perfil": [-2, 4, -1, 0, 0, 3, 0, 0], "tipos": ["Hada", "Planta"]},
        "C) La wikipedia andante que sabe datos inútiles": {"perfil": [0, 0, 0, 4, 0, 0, -1, 1], "tipos": ["Psíquico", "Acero"]},
        "D) El misterioso, borde o del humor negro indescifrable": {"perfil": [1, -2, 0, 1, 3, 0, 3, 0], "tipos": ["Siniestro", "Fantasma"]}}},
    {"pregunta": "40. Haciendo una maleta para viajar...", "opciones": {
        "A) Meto las cosas a puñados cinco minutos antes de salir": {"perfil": [1, 0, 3, -2, 0, 0, 2, 0], "tipos": ["Volador", "Eléctrico"]},
        "B) Llevo 5 jerseys "por si acaso" hace frío, muy previsor": {"perfil": [-1, 1, -1, 1, 0, 2, -1, 0], "tipos": ["Agua", "Normal"]},
        "C) Doblo todo al estilo Marie Kondo. Espacio optimizado al 100%": {"perfil": [0, -1, 0, 4, 0, 0, -2, 1], "tipos": ["Acero", "Psíquico"]},
        "D) Llevo poca cosa, toda negra. Paso desapercibido": {"perfil": [0, -2, 0, 1, 2, 0, 1, 0], "tipos": ["Fantasma", "Siniestro"]}}},

    # BLOQUE 9: Estrategia y Resolución
    {"pregunta": "41. Te pierdes conduciendo por una zona desconocida...", "opciones": {
        "A) Sigo conduciendo rápido y por instinto, ya saldré": {"perfil": [1, 0, 3, -1, 0, 0, 1, 0], "tipos": ["Fuego", "Volador"]},
        "B) Me agobio un poco y le pregunto a algún vecino": {"perfil": [-1, 3, 0, 0, 0, 0, 0, 0], "tipos": ["Hada", "Normal"]},
        "C) Aparco, saco el mapa, calculo las coordenadas y trazo la ruta": {"perfil": [0, -1, 0, 4, 0, 0, -2, 1], "tipos": ["Acero", "Psíquico"]},
        "D) Conduzco por los peores callejones hasta encontrar atajos": {"perfil": [0, -1, 0, 1, 1, 0, 3, 0], "tipos": ["Veneno", "Siniestro"]}}},
    {"pregunta": "42. Te proponen apuntarte a un escape room de terror...", "opciones": {
        "A) Voy el primero, derribando puertas si hace falta": {"perfil": [3, 0, 2, 0, 0, 2, 0, 1], "tipos": ["Lucha", "Roca"]},
        "B) Voy escondido detrás de mi amigo más grande gritando": {"perfil": [-2, 2, 1, 0, 0, 0, 1, 0], "tipos": ["Hada", "Normal"]},
        "C) Ignoro los sustos y resuelvo los candados con candidez": {"perfil": [0, -1, 0, 4, 1, 0, -1, 1], "tipos": ["Psíquico", "Hielo"]},
        "D) Me alío con el actor que asusta para aterrorizar a mis amigos": {"perfil": [1, 0, 0, 1, 3, -2, 4, 0], "tipos": ["Fantasma", "Siniestro"]}}},
    {"pregunta": "43. Un juego de mesa tiene reglas demasiado complejas...", "opciones": {
        "A) Me frustro y digo que juguemos a otra cosa más de acción": {"perfil": [3, 0, 2, -2, 0, 0, 1, 0], "tipos": ["Fuego", "Lucha"]},
        "B) Escucho a quien lo explica e intento aprender sobre la marcha": {"perfil": [-1, 2, 0, 1, 0, 1, 0, 0], "tipos": ["Agua", "Planta"]},
        "C) Me leo el manual de 40 páginas, memorizo todo y exploto el meta": {"perfil": [0, -1, 0, 4, 0, 0, -1, 1], "tipos": ["Acero", "Psíquico"]},
        "D) Hago trampas camuflando cartas cuando no miran": {"perfil": [0, -2, 0, 2, 0, -1, 4, 0], "tipos": ["Siniestro", "Veneno"]}}},
    {"pregunta": "44. Se te rompe el móvil justo antes de un viaje importante...", "opciones": {
        "A) Rompo algo de la rabia y me voy sin móvil": {"perfil": [4, -1, 1, -1, 0, 0, 1, 0], "tipos": ["Dragón", "Lucha"]},
        "B) Pido ayuda a un familiar o amigo para que me preste uno viejo": {"perfil": [-1, 3, 0, 0, 0, 1, 0, 0], "tipos": ["Hada", "Normal"]},
        "C) Lo desmonto y puenteo la placa base para revivirlo 24 horas": {"perfil": [0, -2, 0, 4, 0, 0, -1, 1], "tipos": ["Eléctrico", "Acero"]},
        "D) Le robo el cargador o el móvil a mi hermano a escondidas": {"perfil": [1, -2, 0, 1, 0, -2, 4, 0], "tipos": ["Siniestro", "Fantasma"]}}},
    {"pregunta": "45. Para relajarte completamente necesitas...", "opciones": {
        "A) Adrenalina física, sudar y luego caer desplomado en la cama": {"perfil": [2, 0, 4, -1, 0, 0, 1, 0], "tipos": ["Fuego", "Roca"]},
        "B) Un spa, naturaleza, masajes y silencio total": {"perfil": [-2, 1, -2, 0, 0, 0, 0, 0], "tipos": ["Agua", "Planta"]},
        "C) Un videojuego de puzzles, ajedrez o sudoku hipercomplicado": {"perfil": [0, -2, 0, 4, 0, 0, -2, 1], "tipos": ["Psíquico", "Acero"]},
        "D) Cementerios, true crime, leer sobre cosas perturbadoras": {"perfil": [0, -2, 0, 1, 4, 0, 1, 0], "tipos": ["Fantasma", "Veneno"]}}},

    # BLOQUE 10: Valores Finales
    {"pregunta": "46. Lo que más odias en una persona es...", "opciones": {
        "A) La cobardía, la debilidad o la pereza": {"perfil": [3, 0, 2, 0, 0, 0, 0, 2], "tipos": ["Dragón", "Lucha"]},
        "B) La crueldad, el egoísmo o la falta de corazón": {"perfil": [-2, 4, 0, 0, 0, 2, -1, 0], "tipos": ["Hada", "Planta"]},
        "C) La estupidez, la ignorancia o la irracionalidad": {"perfil": [0, -2, 0, 4, 0, 0, -1, 2], "tipos": ["Psíquico", "Hielo"]},
        "D) La hipocresía. Prefiero a un villano honesto": {"perfil": [1, -2, 0, 1, 2, 0, 3, 0], "tipos": ["Siniestro", "Fantasma"]}}},
    {"pregunta": "47. Cuando el semáforo se pone en naranja...", "opciones": {
        "A) Acelero a fondo, paso seguro": {"perfil": [2, 0, 4, -1, 0, 0, 1, 0], "tipos": ["Fuego", "Volador"]},
        "B) Freno suavemente para no asustar al de atrás": {"perfil": [-1, 2, -1, 1, 0, 1, -1, 0], "tipos": ["Hada", "Agua"]},
        "C) Calculo mi velocidad de frenada y la distancia. Me detengo exacto": {"perfil": [0, -1, 0, 4, 0, 0, -1, 1], "tipos": ["Acero", "Psíquico"]},
        "D) Me lo salto, y si me pitan, pito yo más fuerte": {"perfil": [2, -2, 1, 0, 0, 0, 4, 1], "tipos": ["Siniestro", "Veneno"]}}},
    {"pregunta": "48. Si encontraras una cartera en la calle con 500€...", "opciones": {
        "A) Busco al dueño para dársela cara a cara, por honor": {"perfil": [1, 1, 0, 0, 0, 4, 0, 2], "tipos": ["Lucha", "Acero"]},
        "B) La llevo a la policía, me daría cargo de conciencia quedármela": {"perfil": [-1, 3, 0, 1, 0, 2, -1, 0], "tipos": ["Normal", "Agua"]},
        "C) La dejo donde está, no quiero meterme en problemas legales ilógicos": {"perfil": [0, -2, 0, 3, 0, 0, -1, 0], "tipos": ["Psíquico", "Hielo"]},
        "D) Me quedo el dinero, tiro la cartera a la basura": {"perfil": [1, -3, 0, 1, 1, -3, 4, 0], "tipos": ["Siniestro", "Fantasma"]}}},
    {"pregunta": "49. El elemento de la naturaleza que mejor te representa es...", "opciones": {
        "A) Un incendio forestal indomable o un terremoto": {"perfil": [4, -1, 3, 0, 0, 0, 1, 1], "tipos": ["Fuego", "Tierra"]},
        "B) Un río cristalino que fluye constante o un bosque": {"perfil": [-2, 2, -1, 0, 0, 1, 0, 0], "tipos": ["Agua", "Planta"]},
        "C) Un relámpago fugaz o el cristal perfecto de un diamante": {"perfil": [0, -1, 3, 3, 0, 0, -1, 1], "tipos": ["Eléctrico", "Acero"]},
        "D) El humo tóxico, la niebla o la nada del espacio exterior": {"perfil": [0, -2, 0, 1, 4, -1, 2, 0], "tipos": ["Veneno", "Siniestro"]}}},
    {"pregunta": "50. Finalmente, escoge tu Lema de Liderazgo:", "opciones": {
        "A) 'La victoria pertenece a los que golpean más fuerte'": {"perfil": [4, 0, 2, -1, 0, 1, 0, 2], "tipos": ["Lucha", "Dragón"]},
        "B) 'Crecemos juntos, protegemos a la manada'": {"perfil": [-2, 4, 0, 0, 0, 3, -1, 0], "tipos": ["Hada", "Normal"]},
        "C) 'Conocimiento absoluto, control absoluto'": {"perfil": [0, -2, 0, 4, 1, 0, -1, 3], "tipos": ["Psíquico", "Acero"]},
        "D) 'Bienvenido al abrazo de las sombras'": {"perfil": [1, -2, -1, 1, 4, -1, 3, 1], "tipos": ["Fantasma", "Siniestro"]}}}
]

# --- 4. FORMULARIO STREAMLIT ---
with st.form("formulario_test"):
    respuestas = []
    
    for p in preguntas_test:
        st.markdown(f"**{p['pregunta']}**")
        opcion = st.radio("Opciones", list(p["opciones"].keys()), index=None, label_visibility="collapsed")
        respuestas.append(opcion)
        st.write("---")
        
    enviado = st.form_submit_button("Analizar mi Psique Completa", type="primary")

# --- 5. MOTOR DE CÁLCULO ABSOLUTO ---
if enviado:
    preguntas_faltantes = [str(i + 1) for i, resp in enumerate(respuestas) if resp is None]
    
    if preguntas_faltantes:
        st.error(f"⚠️ ¡Faltan datos! Responde estas preguntas para ajustar las matemáticas: {', '.join(preguntas_faltantes)}.")
    else:
        with st.spinner('Procesando 8 dimensiones e interceptando PokéAPI...'):
            # Array inicial [Agr, Soc, Ene, Int, Mis, Lea, Cao, Org]
            perfil_personalidad = np.zeros(8)
            puntuacion_tipos = {t: 0 for t in ['Normal', 'Fuego', 'Agua', 'Eléctrico', 'Planta', 'Hielo', 'Lucha', 'Veneno', 'Tierra', 'Volador', 'Psíquico', 'Bicho', 'Roca', 'Fantasma', 'Dragón', 'Siniestro', 'Acero', 'Hada']}
            
            # Acumulación Cruzada
            for index, respuesta_usuario in enumerate(respuestas):
                opciones_pregunta = preguntas_test[index]["opciones"]
                datos = opciones_pregunta[respuesta_usuario]
                
                # Sumamos/restamos los pesos exactos de la dimensión
                perfil_personalidad += np.array(datos["perfil"])
                # Asignamos el tipo de Gimnasio
                for tipo in datos["tipos"]:
                    puntuacion_tipos[tipo] += 1

            # Evitamos negativos absolutos (para que el coseno no explote)
            perfil_personalidad = np.clip(perfil_personalidad, 0, None)
            if np.sum(perfil_personalidad) == 0:
                perfil_personalidad += 0.1 # Seguro anti-ceros
                
            # 1. MATEMÁTICAS DEL POKÉMON (Coseno en 8D)
            vector_usuario = perfil_personalidad.reshape(1, -1)
            coincidencias = cosine_similarity(vector_usuario, df_pokemon[columnas_dimensiones].values)[0]
            mejor_indice = np.argmax(coincidencias)
            mejor_pokemon = df_pokemon.iloc[mejor_indice]
            porcentaje_afinidad = round(coincidencias[mejor_indice] * 100, 1)
            
            # 2. LÓGICA DE GIMNASIO DUAL
            tipos_ordenados = sorted(puntuacion_tipos.items(), key=lambda x: x[1], reverse=True)
            tipo_primario = tipos_ordenados[0][0]
            tipo_secundario = tipos_ordenados[1][0]
            puntos_1 = tipos_ordenados[0][1]
            puntos_2 = tipos_ordenados[1][1]
            
            # Si el segundo tipo está a menos de 3 puntos del primero, es un Gimnasio Dual
            es_dual = (puntos_1 - puntos_2) <= 3
            texto_gimnasio = f"{tipo_primario} y {tipo_secundario}" if es_dual else tipo_primario
            
            # 3. EXTRACCIÓN DE POKÉAPI SEGURA
            url_api = f"https://pokeapi.co/api/v2/pokemon/{mejor_pokemon['id']}/"
            try:
                respuesta_api = requests.get(url_api).json()
                sprites = respuesta_api.get('sprites', {}).get('other', {}).get('official-artwork', {})
                imagen_url = sprites.get('front_default')
            except Exception:
                imagen_url = None

            # 4. GENERACIÓN DE PORCENTAJES PARA EL INFORME
            total_puntos = np.sum(perfil_personalidad)
            porcentajes = (perfil_personalidad / total_puntos) * 100
            estadisticas = sorted(zip(columnas_dimensiones, porcentajes), key=lambda x: x[1], reverse=True)
            
            # --- RENDERIZADO DEL RESULTADO ---
            st.success("¡Análisis completado con éxito!")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if imagen_url:
                    st.image(imagen_url, use_container_width=True)
                else:
                    st.warning("Imagen no disponible.")
                
            with col2:
                st.subheader(f"Tu alma gemela es {mejor_pokemon['nombre']}")
                st.write(f"**Afinidad estructural:** {porcentaje_afinidad}%")
                if es_dual:
                    st.write(f"🏆 **Líder de Gimnasio Dual:** {texto_gimnasio}")
                else:
                    st.write(f"🏆 **Líder de Gimnasio Especialista:** {texto_gimnasio}")
                
            st.divider()
            
            st.subheader("📊 Tu Hoja de Estadísticas (Base Stats)")
            
            # Mostramos el Top 3 de dimensiones para que la gente compare
            st.write(f"1. **{estadisticas[0][0]}** dominante ({estadisticas[0][1]:.1f}%)")
            st.write(f"2. Fuerte presencia de **{estadisticas[1][0]}** ({estadisticas[1][1]:.1f}%)")
            if estadisticas[2][1] > 10.0:
                st.write(f"3. Matices de **{estadisticas[2][0]}** ({estadisticas[2][1]:.1f}%)")
                
            st.write(f"\nEsta combinación exacta en tu mapa de 8 dimensiones es la que el algoritmo ha emparejado milimétricamente con el *lore* y mecánicas internas de {mejor_pokemon['nombre']}.")