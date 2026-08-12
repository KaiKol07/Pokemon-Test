import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Test Pokémon y Gimnasio", page_icon="⚡", layout="centered")

st.title("Escáner de Personalidad Pokémon")
st.markdown("Responde a las 50 preguntas para descubrir tu especie y el tipo de Gimnasio que liderarías.")

# --- 2. CARGAR BASE DE DATOS (.JSON) ---
# Usamos caché para que no tenga que leer el archivo cada vez que alguien hace clic
@st.cache_data
def cargar_datos():
    return pd.read_json("vectores_pokemon.json")

df_pokemon = cargar_datos()
columnas_dimensiones = ['Agresivo_Audaz', 'Calmado_Afable', 'Energico_Activo', 'Intelectual_Astuto', 'Misterioso_Oscuro']

# --- 3. BASE DE DATOS DE PREGUNTAS ---
preguntas_test = [
    # --- BLOQUE 1: Convivencia y Amigos ---
    {"pregunta": "1. Un compañero de piso lleva días sin limpiar su parte de la casa...", "opciones": {
        "A) Se lo digo directamente y sin rodeos para que lo haga ya": {"perfil": [4,0,3,0,0], "tipos": ["Fuego", "Lucha", "Roca"]},
        "B) Lo limpio yo porque no me gusta discutir ni ver suciedad": {"perfil": [0,5,0,0,0], "tipos": ["Agua", "Hada", "Planta"]},
        "C) Le propongo organizar un cuadrante de tareas justo para todos": {"perfil": [0,2,0,5,0], "tipos": ["Acero", "Normal", "Psíquico"]},
        "D) Le dejo su basura en la puerta de su habitación sutilmente": {"perfil": [0,0,0,0,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "2. Una amiga cercana está súper agobiada en plena época de exámenes...", "opciones": {
        "A) La obligo a salir a dar una vuelta para que suelte adrenalina": {"perfil": [2,0,5,0,0], "tipos": ["Eléctrico", "Lucha", "Volador"]},
        "B) Le preparo un café, la escucho quejarse y le doy ánimos": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Agua", "Planta"]},
        "C) Le ayudo a organizar sus esquemas y horarios de estudio": {"perfil": [0,0,0,5,2], "tipos": ["Psíquico", "Acero", "Bicho"]},
        "D) Le mando memes extraños o bromas sin contexto para distraerla": {"perfil": [0,2,0,0,4], "tipos": ["Fantasma", "Siniestro", "Veneno"]}}},
    {"pregunta": "3. Estáis organizando un viaje en tren de 5 personas para ir a otra ciudad...", "opciones": {
        "A) Tomo el mando, decido la hora y compro los billetes del tirón": {"perfil": [4,0,3,0,0], "tipos": ["Dragón", "Fuego", "Lucha"]},
        "B) Me adapto a lo que diga la mayoría, voy a pasarlo bien": {"perfil": [0,5,0,0,0], "tipos": ["Normal", "Planta", "Agua"]},
        "C) Calculo los precios exactos, descuentos y busco los mejores asientos": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) Me busco el asiento más alejado del ruido para ir durmiendo": {"perfil": [0,2,0,0,4], "tipos": ["Siniestro", "Fantasma", "Roca"]}}},
    {"pregunta": "4. En unas fiestas locales famosas por las multitudes y los petardos...", "opciones": {
        "A) Estoy en primera fila saltando y sintiendo el ruido": {"perfil": [3,0,5,0,0], "tipos": ["Fuego", "Eléctrico", "Lucha"]},
        "B) Disfruto del ambiente, pero desde una zona tranquila": {"perfil": [0,4,0,2,0], "tipos": ["Agua", "Planta", "Hada"]},
        "C) Observo la estructura de las fallas/monumentos y cómo las montan": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Tierra", "Psíquico"]},
        "D) Me encierro en casa con las persianas bajadas hasta que acaben": {"perfil": [0,0,0,0,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "5. Llegas a una fiesta donde casi no conoces a nadie...", "opciones": {
        "A) Me convierto en el centro de atención rápidamente": {"perfil": [2,0,5,0,0], "tipos": ["Fuego", "Eléctrico", "Lucha"]},
        "B) Me acerco a hablar con alguien que también esté solo": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Planta", "Agua"]},
        "C) Analizo los grupos desde una esquina antes de integrarme": {"perfil": [0,0,0,5,2], "tipos": ["Psíquico", "Acero", "Hielo"]},
        "D) Encuentro a la mascota del anfitrión y me quedo con ella": {"perfil": [0,4,0,0,4], "tipos": ["Fantasma", "Siniestro", "Bicho"]}}},
    
    # --- BLOQUE 2: Ocio y Hobbies ---
    {"pregunta": "6. Quieres hacerle un buen regalo a tu madre o a un familiar. Le compras...", "opciones": {
        "A) Un vale para una experiencia de aventura o deporte": {"perfil": [3,0,4,0,0], "tipos": ["Lucha", "Volador", "Fuego"]},
        "B) Un cuento ilustrado, plantas o algo relajante": {"perfil": [0,5,0,0,0], "tipos": ["Planta", "Hada", "Agua"]},
        "C) El último modelo de un gadget tecnológico útil": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Eléctrico", "Normal"]},
        "D) Una buena novela negra, un thriller o algo de misterio criminal": {"perfil": [0,2,0,3,4], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "7. Si decides hacer ejercicio en casa o en un parque...", "opciones": {
        "A) Hago ejercicios intensos, dominadas y busco superar mis límites": {"perfil": [4,0,3,0,0], "tipos": ["Lucha", "Roca", "Fuego"]},
        "B) Hago estiramientos, uso una esterilla y priorizo la calma": {"perfil": [0,5,0,0,0], "tipos": ["Planta", "Agua", "Hada"]},
        "C) Salgo a correr largas distancias o monto en bicicleta": {"perfil": [0,0,5,0,0], "tipos": ["Volador", "Eléctrico", "Normal"]},
        "D) Prefiero ejercitar la mente; el deporte físico no es para mí": {"perfil": [0,0,0,5,3], "tipos": ["Psíquico", "Siniestro", "Acero"]}}},
    {"pregunta": "8. Si formaras parte de una banda de música, ¿cuál sería tu rol?", "opciones": {
        "A) Cantante principal o guitarra solista (centro del escenario)": {"perfil": [3,0,5,0,0], "tipos": ["Fuego", "Eléctrico", "Dragón"]},
        "B) Bajo o teclado (marcando la armonía y apoyando)": {"perfil": [0,5,0,2,0], "tipos": ["Agua", "Planta", "Hada"]},
        "C) Batería o percusión (manteniendo el ritmo técnico y complejo)": {"perfil": [0,0,4,4,0], "tipos": ["Acero", "Normal", "Lucha"]},
        "D) Productor en las sombras, mezclando los sonidos": {"perfil": [0,0,0,4,4], "tipos": ["Fantasma", "Siniestro", "Veneno"]}}},
    {"pregunta": "9. Jugando a un juego de mesa o a rol con tus amigos...", "opciones": {
        "A) Soy el guerrero que va directo a golpear a los monstruos": {"perfil": [5,0,3,0,0], "tipos": ["Lucha", "Roca", "Tierra"]},
        "B) Soy el apoyo que cura y ayuda al resto del equipo": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Planta", "Agua"]},
        "C) Soy el estratega que lee todas las reglas para ganar ventaja": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Hielo"]},
        "D) Soy el que hace trampas en secreto o juega al despiste": {"perfil": [0,0,0,2,5], "tipos": ["Siniestro", "Veneno", "Fantasma"]}}},
    {"pregunta": "10. ¿Cuál es tu plan ideal para un domingo lluvioso?", "opciones": {
        "A) Invitar a gente a casa, pedir pizza y jugar a algo competitivo": {"perfil": [3,0,4,0,0], "tipos": ["Fuego", "Eléctrico", "Lucha"]},
        "B) Manta, sofá, té caliente y dormir la siesta": {"perfil": [0,5,0,0,0], "tipos": ["Normal", "Planta", "Hada"]},
        "C) Leer un buen libro, ver documentales o aprender algo nuevo": {"perfil": [0,2,0,5,0], "tipos": ["Psíquico", "Acero", "Agua"]},
        "D) Maratón de películas de terror con las luces apagadas": {"perfil": [0,1,0,0,5], "tipos": ["Fantasma", "Siniestro", "Veneno"]}}},

    # --- BLOQUE 3: Reacciones y Personalidad ---
    {"pregunta": "11. Te despiertas por la mañana y te das cuenta de que llegas tarde...", "opciones": {
        "A) Salto de la cama, me visto en 30 segundos y salgo corriendo": {"perfil": [3,0,5,0,0], "tipos": ["Eléctrico", "Volador", "Fuego"]},
        "B) Bueno, ya he llegado tarde. Me tomo el café con calma": {"perfil": [0,5,0,0,0], "tipos": ["Normal", "Planta", "Snorlax"]}, # Snorlax guiño
        "C) Calculo mentalmente la ruta más óptima para minimizar el retraso": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Hielo"]},
        "D) Me invento una excusa elaboradísima para justificarlo": {"perfil": [0,0,0,3,4], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "12. Alguien se cuela en la cola del supermercado justo delante de ti...", "opciones": {
        "A) Le llamo la atención alzando la voz inmediatamente": {"perfil": [5,0,0,0,0], "tipos": ["Fuego", "Lucha", "Dragón"]},
        "B) Resoplo pero no digo nada, no merece la pena el conflicto": {"perfil": [0,5,0,0,0], "tipos": ["Agua", "Hada", "Planta"]},
        "C) Le explico con calma y educación que la fila empieza atrás": {"perfil": [0,3,0,4,0], "tipos": ["Acero", "Normal", "Psíquico"]},
        "D) Le pongo la zancadilla sin que nadie me vea": {"perfil": [2,0,0,0,5], "tipos": ["Siniestro", "Veneno", "Fantasma"]}}},
    {"pregunta": "13. ¿Qué cualidad valoras más en las personas?", "opciones": {
        "A) La valentía y la pasión para hacer las cosas": {"perfil": [3,0,4,0,0], "tipos": ["Fuego", "Lucha", "Dragón"]},
        "B) La bondad, la empatía y que sepan escuchar": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Agua", "Planta"]},
        "C) La inteligencia, el ingenio y el sentido del humor lógico": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Eléctrico"]},
        "D) La lealtad ciega y que sepan guardar secretos": {"perfil": [0,2,0,0,4], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "14. Un defecto que sueles tener a veces es...", "opciones": {
        "A) Ser demasiado impulsivo, impaciente o cabezota": {"perfil": [5,0,2,0,0], "tipos": ["Fuego", "Dragón", "Roca"]},
        "B) Ser un poco perezoso o evitar tomar decisiones difíciles": {"perfil": [0,5,0,0,0], "tipos": ["Normal", "Planta", "Hada"]},
        "C) Pensar demasiado las cosas y parecer muy distante": {"perfil": [0,0,0,5,2], "tipos": ["Hielo", "Psíquico", "Acero"]},
        "D) Ser un poco rencoroso o desconfiado con la gente": {"perfil": [0,0,0,2,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "15. Estás intentando montar un mueble y no encajan las piezas...", "opciones": {
        "A) Fuerzo la pieza a golpes hasta que encaje": {"perfil": [4,0,3,0,0], "tipos": ["Lucha", "Roca", "Tierra"]},
        "B) Lo dejo a medias y me voy a hacer otra cosa relajante": {"perfil": [0,4,0,0,0], "tipos": ["Normal", "Agua", "Planta"]},
        "C) Desmonto todo y reviso el manual desde la página uno": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) Uso pegamento extra fuerte o apaños dudosos para ocultar el fallo": {"perfil": [0,0,0,2,4], "tipos": ["Veneno", "Siniestro", "Fantasma"]}}},

    # --- BLOQUE 4: Entornos y Gustos Cotidianos ---
    {"pregunta": "16. ¿Cuál de estos animales elegirías de mascota?", "opciones": {
        "A) Un perro grande y enérgico que necesite correr mucho": {"perfil": [2,0,5,0,0], "tipos": ["Lucha", "Normal", "Fuego"]},
        "B) Un conejo o algo suave y achuchable": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Planta", "Agua"]},
        "C) Un acuario o un terrario fascinante de observar": {"perfil": [0,2,0,4,0], "tipos": ["Bicho", "Agua", "Acero"]},
        "D) Un gato negro muy independiente o un reptil exótico": {"perfil": [0,0,0,2,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "17. Para vestirte en el día a día prefieres...", "opciones": {
        "A) Ropa deportiva, zapatillas anchas y colores vivos": {"perfil": [0,0,5,0,0], "tipos": ["Eléctrico", "Fuego", "Volador"]},
        "B) Ropa muy cómoda, ancha y de colores pastel o suaves": {"perfil": [0,5,0,0,0], "tipos": ["Normal", "Hada", "Planta"]},
        "C) Estilo ordenado, camisas, colores sobrios y simétricos": {"perfil": [0,0,0,5,0], "tipos": ["Hielo", "Acero", "Psíquico"]},
        "D) Ropa oscura, accesorios metálicos, cuero o estilo gótico": {"perfil": [0,0,0,0,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "18. Tu paisaje o clima favorito para perderte sería...", "opciones": {
        "A) Un día de sol abrasador en una zona rocosa o playa": {"perfil": [3,0,3,0,0], "tipos": ["Fuego", "Roca", "Tierra"]},
        "B) Un bosque verde con brisa suave o después de llover": {"perfil": [0,5,0,0,0], "tipos": ["Planta", "Bicho", "Agua"]},
        "C) Una tormenta eléctrica desde la ventana de una ciudad alta": {"perfil": [0,2,4,4,0], "tipos": ["Eléctrico", "Acero", "Volador"]},
        "D) Una noche de niebla espesa donde no se ve la luna": {"perfil": [0,0,0,2,5], "tipos": ["Fantasma", "Siniestro", "Hielo"]}}},
    {"pregunta": "19. ¿Cuál de estas películas verías en el cine?", "opciones": {
        "A) Acción pura, explosiones, superhéroes y artes marciales": {"perfil": [4,0,3,0,0], "tipos": ["Lucha", "Fuego", "Dragón"]},
        "B) Comedia romántica, animación bonita o documentales de naturaleza": {"perfil": [0,5,0,2,0], "tipos": ["Hada", "Planta", "Agua"]},
        "C) Ciencia ficción dura, viajes en el tiempo o puzzles mentales": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Eléctrico"]},
        "D) Terror psicológico, asesinatos o cine negro": {"perfil": [0,0,0,2,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "20. Cuando cocinas, tu estilo es...", "opciones": {
        "A) Fuego alto, rápido y con mucho picante o especias fuertes": {"perfil": [4,0,3,0,0], "tipos": ["Fuego", "Dragón", "Lucha"]},
        "B) Platos tradicionales, caseros y reconfortantes (sopas, guisos)": {"perfil": [0,5,0,0,0], "tipos": ["Agua", "Planta", "Normal"]},
        "C) Sigo las recetas al miligramo midiendo todo en báscula": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) Mezclo sobras al azar a ver qué sale; a veces funciona": {"perfil": [0,0,2,0,4], "tipos": ["Veneno", "Bicho", "Siniestro"]}}},

    # --- BLOQUE 5: Fantasía y Supuestos ---
    {"pregunta": "21. Si te dieran a elegir un superpoder, escogerías...", "opciones": {
        "A) Súper fuerza y resistencia invulnerable": {"perfil": [5,0,3,0,0], "tipos": ["Lucha", "Roca", "Acero"]},
        "B) Volar libremente o hablar con los animales": {"perfil": [0,4,4,0,0], "tipos": ["Volador", "Planta", "Hada"]},
        "C) Leer mentes, mover cosas con el pensamiento o súper intelecto": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Eléctrico", "Hielo"]},
        "D) Hacerte invisible o atravesar paredes": {"perfil": [0,0,0,2,5], "tipos": ["Fantasma", "Siniestro", "Veneno"]}}},
    {"pregunta": "22. En un apocalipsis zombi, tu arma principal sería...", "opciones": {
        "A) Un bate de béisbol con clavos o una motosierra": {"perfil": [5,0,2,0,0], "tipos": ["Fuego", "Lucha", "Acero"]},
        "B) Botiquines de primeros auxilios y raciones de supervivencia": {"perfil": [0,5,0,0,0], "tipos": ["Normal", "Planta", "Agua"]},
        "C) Una radio de onda corta para contactar con supervivientes": {"perfil": [0,0,0,5,0], "tipos": ["Eléctrico", "Psíquico", "Volador"]},
        "D) Una ballesta sigilosa y veneno para las flechas": {"perfil": [0,0,0,2,5], "tipos": ["Veneno", "Siniestro", "Fantasma"]}}},
    {"pregunta": "23. Si encontraras una lámpara mágica, tu primer deseo sería...", "opciones": {
        "A) Convertirme en el mejor deportista/luchador del mundo": {"perfil": [4,0,4,0,0], "tipos": ["Lucha", "Fuego", "Dragón"]},
        "B) Paz mundial, salud infinita o que todo el mundo sea feliz": {"perfil": [0,5,0,2,0], "tipos": ["Hada", "Agua", "Planta"]},
        "C) Saber la respuesta a cualquier pregunta del universo": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Eléctrico"]},
        "D) Dinero infinito con el que manipular el mundo en la sombra": {"perfil": [0,0,0,2,5], "tipos": ["Siniestro", "Veneno", "Fantasma"]}}},
    {"pregunta": "24. En un cuento de hadas, tú serías...", "opciones": {
        "A) El caballero temerario que va a cazar al dragón": {"perfil": [4,0,3,0,0], "tipos": ["Acero", "Lucha", "Fuego"]},
        "B) El aldeano bondadoso que ayuda al héroe en su camino": {"perfil": [0,5,0,0,0], "tipos": ["Normal", "Hada", "Planta"]},
        "C) El mago sabio ermitaño que vive en la torre": {"perfil": [0,0,0,5,2], "tipos": ["Psíquico", "Hielo", "Dragón"]},
        "D) La bruja misteriosa del pantano o el villano incomprendido": {"perfil": [0,0,0,2,5], "tipos": ["Veneno", "Fantasma", "Siniestro"]}}},
    {"pregunta": "25. Si pudieras viajar a otra época, irías a...", "opciones": {
        "A) La época de los vikingos o gladiadores, pura acción": {"perfil": [5,0,2,0,0], "tipos": ["Lucha", "Roca", "Tierra"]},
        "B) A los años 60, música tranquila, paz y naturaleza": {"perfil": [0,5,0,0,0], "tipos": ["Planta", "Agua", "Hada"]},
        "C) Al futuro lejano, para ver naves espaciales y cíborgs": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Eléctrico", "Psíquico"]},
        "D) Al Londres victoriano, callejones nublados y misterios": {"perfil": [0,0,0,2,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},

    # --- BLOQUE 6: Trabajo, Dinero y Organización ---
    {"pregunta": "26. Te encuentras un billete de 50€ en la calle...", "opciones": {
        "A) Me voy corriendo a gastarlo en salir de fiesta o invitar a amigos": {"perfil": [0,0,5,0,0], "tipos": ["Fuego", "Eléctrico", "Lucha"]},
        "B) Lo dono a alguien que lo necesite o compro un regalo": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Planta", "Agua"]},
        "C) Lo guardo en la cuenta de ahorro para futuros gastos": {"perfil": [0,0,0,5,0], "tipos": ["Normal", "Acero", "Psíquico"]},
        "D) Miro a los lados, me lo guardo en silencio y me voy rápido": {"perfil": [0,0,0,0,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "27. Al hacer la maleta para un viaje...", "opciones": {
        "A) Meto todo a presión a última hora y cruzo los dedos": {"perfil": [2,0,4,0,0], "tipos": ["Lucha", "Fuego", "Eléctrico"]},
        "B) Llevo ropa cómoda de más por si acaso": {"perfil": [0,4,0,0,0], "tipos": ["Normal", "Agua", "Hada"]},
        "C) Hago una lista previa y organizo todo con organizadores por colores": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) Llevo lo mínimo indispensable y en tonos muy oscuros": {"perfil": [0,2,0,0,4], "tipos": ["Siniestro", "Veneno", "Bicho"]}}},
    {"pregunta": "28. Te enfrentas a un día lleno de tareas aburridas...", "opciones": {
        "A) Me pongo música a tope y las hago todas del tirón sin parar": {"perfil": [2,0,5,0,0], "tipos": ["Eléctrico", "Fuego", "Volador"]},
        "B) Las voy haciendo poco a poco intercalando descansos": {"perfil": [0,5,0,0,0], "tipos": ["Planta", "Agua", "Normal"]},
        "C) Planifico el orden exacto para optimizar mi tiempo al máximo": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Bicho"]},
        "D) Procrastino hasta la noche y luego lo hago todo bajo presión": {"perfil": [0,0,0,2,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "29. ¿Qué tal llevas lo de seguir normas estrictas?", "opciones": {
        "A) Mal, me rebelo a la mínima si no tienen sentido": {"perfil": [5,0,2,0,0], "tipos": ["Fuego", "Lucha", "Dragón"]},
        "B) Bien, sigo la corriente para no causar problemas": {"perfil": [0,5,0,0,0], "tipos": ["Agua", "Normal", "Hada"]},
        "C) Excelente, las reglas sostienen el orden de la sociedad": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) Hago como que las sigo, pero encuentro los huecos legales": {"perfil": [0,0,0,3,4], "tipos": ["Siniestro", "Veneno", "Fantasma"]}}},
    {"pregunta": "30. Cuando visitas una ciudad que no conoces...", "opciones": {
        "A) Ando sin rumbo fijo explorando hasta cansarme": {"perfil": [0,0,5,0,0], "tipos": ["Volador", "Eléctrico", "Lucha"]},
        "B) Me siento en una terraza bonita a ver a la gente pasar": {"perfil": [0,5,0,0,0], "tipos": ["Planta", "Agua", "Hada"]},
        "C) Llevo marcados en Google Maps todos los museos y monumentos": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Roca"]},
        "D) Prefiero perderme por los callejones estrechos o locales ocultos": {"perfil": [0,2,0,0,4], "tipos": ["Fantasma", "Siniestro", "Veneno"]}}},

    # --- BLOQUE 7: Emociones y Relaciones ---
    {"pregunta": "31. Un amigo te pide consejo sobre un problema amoroso...", "opciones": {
        "A) Le digo que deje a esa persona y sea un líder fuerte": {"perfil": [4,0,0,0,0], "tipos": ["Fuego", "Lucha", "Dragón"]},
        "B) Le ofrezco un abrazo y pañuelos; las emociones son complicadas": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Agua", "Planta"]},
        "C) Analizo los pros y contras de la relación de forma objetiva": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Hielo"]},
        "D) Le sugiero darle celos a su pareja para ver cómo reacciona": {"perfil": [0,0,0,2,5], "tipos": ["Siniestro", "Veneno", "Fantasma"]}}},
    {"pregunta": "32. Cuando estás triste o de bajón, lo que más te ayuda es...", "opciones": {
        "A) Salir a hacer deporte intenso hasta no poder más": {"perfil": [3,0,4,0,0], "tipos": ["Lucha", "Fuego", "Roca"]},
        "B) Comer algo rico, taparme con una manta y llorar un poco": {"perfil": [0,5,0,0,0], "tipos": ["Agua", "Normal", "Hada"]},
        "C) Racionalizar mis sentimientos; buscar el porqué estoy así": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Eléctrico"]},
        "D) Aislarme por completo en mi cuarto a oscuras": {"perfil": [0,1,0,0,5], "tipos": ["Fantasma", "Siniestro", "Veneno"]}}},
    {"pregunta": "33. Si ganas un torneo o una competición...", "opciones": {
        "A) Lo celebro por todo lo alto gritando y presumiendo": {"perfil": [4,0,4,0,0], "tipos": ["Fuego", "Lucha", "Eléctrico"]},
        "B) Agradezco a todos la participación con humildad": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Agua", "Planta"]},
        "C) Pienso que era obvio, estadísticamente era el mejor": {"perfil": [0,0,0,5,2], "tipos": ["Psíquico", "Hielo", "Acero"]},
        "D) Me río fríamente de mis rivales en secreto": {"perfil": [0,0,0,2,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "34. ¿Qué te suele hacer reír más?", "opciones": {
        "A) Las caídas absurdas o los sustos (humor físico)": {"perfil": [2,0,4,0,0], "tipos": ["Lucha", "Eléctrico", "Roca"]},
        "B) Los vídeos adorables de perritos o gatos torpes": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Planta", "Normal"]},
        "C) El sarcasmo fino y el humor inteligente o referencias cultas": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Hielo"]},
        "D) El humor negrísimo y políticamente incorrecto": {"perfil": [0,0,0,0,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "35. Un extraño te saluda por la calle creyendo que eres otra persona...", "opciones": {
        "A) Me echo a reír a carcajadas y le digo que se ha equivocado": {"perfil": [0,0,5,0,0], "tipos": ["Fuego", "Volador", "Normal"]},
        "B) Le sonrío amablemente y le saco del error con suavidad": {"perfil": [0,5,0,0,0], "tipos": ["Agua", "Hada", "Planta"]},
        "C) Le ignoro y sigo caminando rápido fingiendo no oírle": {"perfil": [0,3,0,4,0], "tipos": ["Hielo", "Acero", "Psíquico"]},
        "D) Le sigo la corriente fingiendo ser esa persona un rato": {"perfil": [0,0,0,3,5], "tipos": ["Siniestro", "Fantasma", "Bicho"]}}},

    # --- BLOQUE 8: Temas Variados y Cierre ---
    {"pregunta": "36. En el colegio/instituto, ¿qué asignatura preferías?", "opciones": {
        "A) Educación Física; me encantaba moverme": {"perfil": [2,0,5,0,0], "tipos": ["Lucha", "Eléctrico", "Volador"]},
        "B) Arte, Dibujo o Ciencias de la Naturaleza": {"perfil": [0,5,0,2,0], "tipos": ["Planta", "Agua", "Bicho"]},
        "C) Matemáticas, Física o Informática": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) No iba mucho a clase, me escondía en los pasillos": {"perfil": [0,0,0,0,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "37. ¿Qué tipo de clima extremo te gusta más?", "opciones": {
        "A) Una ola de calor en pleno agosto": {"perfil": [4,0,3,0,0], "tipos": ["Fuego", "Tierra", "Dragón"]},
        "B) Una lluvia suave pero que no para en todo el día": {"perfil": [0,5,0,0,0], "tipos": ["Agua", "Bicho", "Planta"]},
        "C) Un día congelado con nieve o tormenta eléctrica brutal": {"perfil": [0,2,4,4,0], "tipos": ["Hielo", "Eléctrico", "Acero"]},
        "D) Eclipse solar o una noche súper oscura sin luna": {"perfil": [0,1,0,2,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "38. Te regalan un jardín vacío. ¿Qué pones en él?", "opciones": {
        "A) Una zona de barbacoa gigante y un ring de entrenamiento": {"perfil": [4,0,4,0,0], "tipos": ["Fuego", "Lucha", "Roca"]},
        "B) Un estanque con peces, flores preciosas y una hamaca": {"perfil": [0,5,0,0,0], "tipos": ["Planta", "Agua", "Hada"]},
        "C) Un invernadero automatizado y simétrico": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Tierra"]},
        "D) Plantas carnívoras, setas raras y estatuas de piedra": {"perfil": [0,2,0,0,4], "tipos": ["Veneno", "Fantasma", "Siniestro"]}}},
    {"pregunta": "39. Cuando usas las redes sociales...", "opciones": {
        "A) Subo historias haciendo deporte, saliendo o de fiesta": {"perfil": [0,0,5,0,0], "tipos": ["Fuego", "Eléctrico", "Lucha"]},
        "B) Pongo fotos de mis mascotas, comida o paisajes bonitos": {"perfil": [0,5,0,0,0], "tipos": ["Normal", "Planta", "Hada"]},
        "C) Comparto artículos científicos, noticias o debates técnicos": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) No subo nada nunca; solo observo desde las sombras (stalkeo)": {"perfil": [0,2,0,0,5], "tipos": ["Fantasma", "Siniestro", "Veneno"]}}},
    {"pregunta": "40. En un buffet libre...", "opciones": {
        "A) Me voy directo a la carne y me lleno el plato hasta arriba": {"perfil": [5,0,3,0,0], "tipos": ["Dragón", "Fuego", "Lucha"]},
        "B) Pruebo un poquito de cada cosa, sobre todo los postres": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Normal", "Planta"]},
        "C) Me hago un plato perfectamente equilibrado en macros": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) Me como los mariscos y lo más caro para que pierdan dinero": {"perfil": [0,0,0,2,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "41. Vas de copiloto en un coche largo. Tú...", "opciones": {
        "A) Pongo la música a tope y canto a gritos": {"perfil": [0,0,5,0,0], "tipos": ["Eléctrico", "Normal", "Fuego"]},
        "B) Me quedo dormido a los cinco minutos": {"perfil": [0,5,0,0,0], "tipos": ["Planta", "Agua", "Hada"]},
        "C) Voy guiando con el GPS e indicando las salidas correctas": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) Me quejo de los demás conductores por lo bajo": {"perfil": [2,0,0,0,4], "tipos": ["Veneno", "Siniestro", "Fantasma"]}}},
    {"pregunta": "42. Te pierdes en un bosque y empieza a anochecer...", "opciones": {
        "A) Hago una hoguera gigante y aúllo para que me escuchen": {"perfil": [4,0,4,0,0], "tipos": ["Fuego", "Lucha", "Dragón"]},
        "B) Me subo a un árbol y espero tranquilo a que amanezca": {"perfil": [0,5,0,0,0], "tipos": ["Bicho", "Planta", "Normal"]},
        "C) Me guío por la estrella polar y el musgo de los árboles": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Hielo"]},
        "D) Excelente, la noche es mi terreno natural": {"perfil": [0,0,0,0,5], "tipos": ["Fantasma", "Siniestro", "Veneno"]}}},
    {"pregunta": "43. Una persona cuenta un chiste malísimo...", "opciones": {
        "A) Le abucheo en broma y le tiro un cojín": {"perfil": [3,0,4,0,0], "tipos": ["Lucha", "Fuego", "Eléctrico"]},
        "B) Me río por pena para que no se sienta mal": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Agua", "Planta"]},
        "C) Le explico por qué su chiste carece de gracia lógicamente": {"perfil": [0,0,0,5,2], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) Le clavo una mirada fría y no muevo un músculo": {"perfil": [0,3,0,0,4], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}},
    {"pregunta": "44. Un insecto entra en tu habitación...", "opciones": {
        "A) Lo aplasto sin piedad con el zapato": {"perfil": [5,0,0,0,0], "tipos": ["Fuego", "Roca", "Tierra"]},
        "B) Lo cojo con un vasito y lo saco por la ventana": {"perfil": [0,5,0,0,0], "tipos": ["Planta", "Agua", "Bicho"]},
        "C) Analizo qué especie es antes de tomar medidas": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Normal"]},
        "D) Lo enveneno con insecticida lentamente": {"perfil": [0,0,0,0,5], "tipos": ["Veneno", "Fantasma", "Siniestro"]}}},
    {"pregunta": "45. Para decorar tu cuarto eliges...", "opciones": {
        "A) Pósters de deportes, coches o cosas brillantes": {"perfil": [0,0,5,0,0], "tipos": ["Eléctrico", "Lucha", "Fuego"]},
        "B) Muchas plantas, cojines peludos y luces cálidas": {"perfil": [0,5,0,0,0], "tipos": ["Hada", "Planta", "Agua"]},
        "C) Estanterías simétricas, libros ordenados y luz blanca": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) Cortinas opacas, calaveras o estética oscura": {"perfil": [0,0,0,0,5], "tipos": ["Fantasma", "Siniestro", "Veneno"]}}},
    {"pregunta": "46. Te invitan a ir de acampada...", "opciones": {
        "A) ¡Genial! Escalar, hacer senderismo y cansarme": {"perfil": [3,0,5,0,0], "tipos": ["Roca", "Tierra", "Lucha"]},
        "B) Vale, pero yo me quedo junto al río leyendo tranquilamente": {"perfil": [0,5,0,2,0], "tipos": ["Agua", "Normal", "Planta"]},
        "C) Me llevo todo tipo de gadgets solares y de supervivencia": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Eléctrico", "Psíquico"]},
        "D) Odio los bichos y dormir en el suelo; no voy": {"perfil": [0,0,0,0,5], "tipos": ["Siniestro", "Veneno", "Bicho"]}}},
    {"pregunta": "47. Cuando caminas por la calle...", "opciones": {
        "A) Piso fuerte y voy rápido, abriéndome paso": {"perfil": [4,0,4,0,0], "tipos": ["Fuego", "Lucha", "Dragón"]},
        "B) Voy despacio, mirando los escaparates o los árboles": {"perfil": [0,5,0,0,0], "tipos": ["Planta", "Agua", "Hada"]},
        "C) Voy rápido, concentrado, esquivando a la gente matemáticamente": {"perfil": [0,0,0,5,0], "tipos": ["Psíquico", "Acero", "Eléctrico"]},
        "D) Camino pegado a las paredes, observando todo sin ser visto": {"perfil": [0,3,0,0,5], "tipos": ["Fantasma", "Siniestro", "Veneno"]}}},
    {"pregunta": "48. Vas al parque de atracciones. Tu favorita es...", "opciones": {
        "A) La montaña rusa más rápida, alta y violenta": {"perfil": [4,0,5,0,0], "tipos": ["Fuego", "Volador", "Dragón"]},
        "B) Los troncos de agua o la noria": {"perfil": [0,5,0,0,0], "tipos": ["Agua", "Hada", "Normal"]},
        "C) Los simuladores 3D o las escape rooms": {"perfil": [0,0,0,5,2], "tipos": ["Acero", "Psíquico", "Hielo"]},
        "D) La casa del terror": {"perfil": [0,0,0,0,5], "tipos": ["Fantasma", "Siniestro", "Veneno"]}}},
    {"pregunta": "49. Frente a un plato que nunca has probado...", "opciones": {
        "A) Me lo como entero sin dudar, me encantan los retos": {"perfil": [4,0,3,0,0], "tipos": ["Lucha", "Fuego", "Tierra"]},
        "B) Pruebo un poquito con cuidado para ver si me gusta": {"perfil": [0,5,0,0,0], "tipos": ["Normal", "Planta", "Agua"]},
        "C) Pregunto por todos y cada uno de los ingredientes primero": {"perfil": [0,0,0,5,0], "tipos": ["Acero", "Psíquico", "Bicho"]},
        "D) Me niego en rotundo, a saber qué le han echado": {"perfil": [0,2,0,0,4], "tipos": ["Veneno", "Siniestro", "Fantasma"]}}},
    {"pregunta": "50. Finalmente, ¿cuál sería el lema de tu Gimnasio Pokémon?", "opciones": {
        "A) 'El fuego de la pasión lo arrasa todo'": {"perfil": [5,0,2,0,0], "tipos": ["Dragón", "Lucha", "Fuego"]},
        "B) 'Paciencia, adaptación y corazón puro'": {"perfil": [0,5,0,2,0], "tipos": ["Agua", "Planta", "Hada"]},
        "C) 'La mente siempre vence a la fuerza bruta'": {"perfil": [0,0,0,5,2], "tipos": ["Psíquico", "Acero", "Eléctrico"]},
        "D) 'Bienvenido a tu peor pesadilla'": {"perfil": [0,2,0,0,5], "tipos": ["Siniestro", "Fantasma", "Veneno"]}}}

]

# --- 4. INTERFAZ DEL FORMULARIO ---
# st.form evita que la web se recargue con cada clic, solo calcula al pulsar "Enviar"
with st.form("formulario_test"):
    respuestas = []
    
    for p in preguntas_test:
        st.markdown(f"**{p['pregunta']}**")
        # index=None hace que los botones empiecen desmarcados
        opcion = st.radio("Opciones", list(p["opciones"].keys()), index=None, label_visibility="collapsed")
        respuestas.append(opcion)
        st.write("---") # Línea separadora
        
    enviado = st.form_submit_button("Analizar mi Personalidad", type="primary")

# --- 5. MOTOR DE CÁLCULO ---
if enviado:
    preguntas_faltantes = [str(i + 1) for i, resp in enumerate(respuestas) if resp is None]
    
    if preguntas_faltantes:
        st.error(f"⚠️ ¡Alto ahí! Te has dejado sin marcar las siguientes preguntas: {', '.join(preguntas_faltantes)}.")
    else:
        with st.spinner('Calculando distancias vectoriales y analizando tu perfil...'):
            perfil_personalidad = np.zeros(5)
            puntuacion_tipos = {t: 0 for t in ['Normal', 'Fuego', 'Agua', 'Eléctrico', 'Planta', 'Hielo', 'Lucha', 'Veneno', 'Tierra', 'Volador', 'Psíquico', 'Bicho', 'Roca', 'Fantasma', 'Dragón', 'Siniestro', 'Acero', 'Hada']}
            
            for index, respuesta_usuario in enumerate(respuestas):
                opciones_pregunta = preguntas_test[index]["opciones"]
                datos = opciones_pregunta[respuesta_usuario]
                
                perfil_personalidad += np.array(datos["perfil"])
                for tipo in datos["tipos"]:
                    puntuacion_tipos[tipo] += 1

            # Matemáticas
            vector_usuario = perfil_personalidad.reshape(1, -1)
            coincidencias = cosine_similarity(vector_usuario, df_pokemon[columnas_dimensiones].values)[0]
            mejor_indice = np.argmax(coincidencias)
            mejor_pokemon = df_pokemon.iloc[mejor_indice]
            porcentaje_afinidad = round(coincidencias[mejor_indice] * 100, 1)
            tipo_ganador = max(puntuacion_tipos, key=puntuacion_tipos.get)
            
            # PokéAPI
            url_api = f"https://pokeapi.co/api/v2/pokemon/{mejor_pokemon['id']}/"
            respuesta_api = requests.get(url_api).json()
            
            # Extracción segura por si la PokéAPI falla o devuelve None
            sprites = respuesta_api.get('sprites', {}).get('other', {}).get('official-artwork', {})
            imagen_url = sprites.get('front_default')
            
            # Cálculos del informe
            nombres_display = ['Agresivo / Audaz', 'Calmado / Afable', 'Enérgico / Activo', 'Intelectual / Astuto', 'Misterioso / Oscuro']
            total_puntos = np.sum(perfil_personalidad)
            porcentajes = (perfil_personalidad / total_puntos) * 100
            estadisticas = sorted(zip(nombres_display, porcentajes), key=lambda x: x[1], reverse=True)
            rasgo_principal = estadisticas[0]
            rasgo_secundario = estadisticas[1]
            
            # --- RENDERIZADO DEL RESULTADO ---
            st.success("¡Análisis completado!")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Verificación de imagen y uso del parámetro actualizado
                if imagen_url:
                    st.image(imagen_url, use_container_width=True)
                else:
                    st.warning("Imagen del Pokémon temporalmente no disponible.")
                
            with col2:
                st.subheader(f"Tu alma gemela es {mejor_pokemon['nombre']}")
                st.write(f"**Afinidad matemática:** {porcentaje_afinidad}%")
                st.write(f"**Gimnasio:** Especialista de tipo **{tipo_ganador}**")
                
            st.divider()
            
            st.subheader("Análisis de tu Personalidad")
            texto_analisis = f"El algoritmo ha determinado que tu rasgo dominante es ser **{rasgo_principal[0]}** ({rasgo_principal[1]:.1f}%). "
            if rasgo_secundario[1] > 15.0:
                texto_analisis += f"Esto se combina con una fuerte tendencia hacia lo **{rasgo_secundario[0]}** ({rasgo_secundario[1]:.1f}%). "
            texto_analisis += f"Esta mezcla exacta en tu temperamento es lo que te vincula con {mejor_pokemon['nombre']}."
            st.write(texto_analisis)
            
            st.write("**TU GIMNASIO:**")
            st.write(f"Tus decisiones diarias y aficiones han sumado la mayoría de puntos de aptitud hacia el tipo **{tipo_ganador}**, definiendo tu estilo de liderazgo.")