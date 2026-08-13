import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import euclidean_distances

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
    # BLOQUE 1: Vida diaria y curro
    {"pregunta": "1. Llevas tres días seguidos durmiendo mal y hoy tienes reunión importante. ¿Cómo llegas?", "opciones": {
        "A) Como si nada, funciono igual de bien reventado que descansado": {"perfil": [1, 0, 1, 0, 0, 0, 0, 3], "tipos": ["Roca", "Acero"]},
        "B) Con un café doble y quejándome a todo el que se cruce": {"perfil": [2, 1, 2, 0, 0, 0, 1, 0], "tipos": ["Eléctrico", "Fuego"]},
        "C) Voy pero aviso desde ya que hoy rindo la mitad": {"perfil": [0, 2, -2, 0, 0, 2, 0, -1], "tipos": ["Normal", "Agua"]},
        "D) Cancelo lo que pueda esperar, mi cuerpo manda hoy": {"perfil": [-1, -1, -3, 1, 0, -1, 1, 1], "tipos": ["Planta", "Hielo"]}}},
    {"pregunta": "2. Un compañero de piso lleva semanas sin fregar sus platos. Un día explota la cosa...", "opciones": {
        "A) Los meto en una bolsa y se los dejo delante de su puerta": {"perfil": [2, -1, 0, 1, 0, -1, 2, 1], "tipos": ["Veneno", "Siniestro"]},
        "B) Hablo con él tranquilo, seguro que tiene un mal momento": {"perfil": [-2, 3, 0, 0, 0, 3, 0, -1], "tipos": ["Hada", "Agua"]},
        "C) Hago una tabla de turnos y la pego en la nevera": {"perfil": [0, 0, 0, 4, 0, 1, -2, 0], "tipos": ["Acero", "Psíquico"]},
        "D) Los friego yo, total ya estoy de mal humor": {"perfil": [-3, 1, -1, 0, 0, 2, 0, -2], "tipos": ["Planta", "Normal"]}}},
    {"pregunta": "3. Te escriben a las 23:00 pidiéndote un favor urgente que no te apetece nada hacer.", "opciones": {
        "A) Digo que sí aunque por dentro esté maldiciendo un poco": {"perfil": [0, 2, 0, 0, 0, 4, 0, -1], "tipos": ["Normal", "Hada"]},
        "B) Pregunto primero qué gano yo antes de mover un dedo": {"perfil": [1, -1, 0, 2, 0, -2, 2, 1], "tipos": ["Siniestro", "Veneno"]},
        "C) Contesto al día siguiente como si no hubiera visto nada": {"perfil": [0, -2, -1, 1, 0, -2, 1, 1], "tipos": ["Fantasma", "Hielo"]},
        "D) Le doy la solución rápida sin ni siquiera moverme del sitio": {"perfil": [0, 0, 0, 3, 1, 1, 0, 0], "tipos": ["Psíquico", "Acero"]}}},
    {"pregunta": "4. En el súper hay una cola larguísima y solo una caja abierta.", "opciones": {
        "A) Me pongo a hablar con el de delante para matar el rato": {"perfil": [0, 3, 0, 0, 0, 0, 0, -1], "tipos": ["Normal", "Volador"]},
        "B) Calculo mentalmente si me sale a cuenta ir a otra tienda": {"perfil": [0, -1, 0, 3, 0, 0, 0, 0], "tipos": ["Acero", "Tierra"]},
        "C) Suspiro fuerte esperando que alguien abra otra caja por presión social": {"perfil": [1, 0, 1, 0, 0, 0, 1, 1], "tipos": ["Fuego", "Eléctrico"]},
        "D) Dejo el carro y me voy, ya compraré online": {"perfil": [0, -2, -1, 1, 0, -1, 0, 2], "tipos": ["Hielo", "Dragón"]}}},
    {"pregunta": "5. Alguien te corrige en público sobre algo que dijiste mal.", "opciones": {
        "A) Le doy la razón al instante, prefiero eso a discutir por tonterías": {"perfil": [-2, 1, 0, 0, 0, 1, 0, -2], "tipos": ["Agua", "Normal"]},
        "B) Busco el matiz por el que en realidad no estaba tan equivocado": {"perfil": [0, 0, 0, 3, 0, 0, 1, 2], "tipos": ["Psíquico", "Roca"]},
        "C) Me sienta fatal aunque no lo demuestre en la cara": {"perfil": [0, -1, -1, 0, 1, 0, 0, 3], "tipos": ["Fantasma", "Hielo"]},
        "D) Me río y lo uso de chiste antes de que lo haga otro": {"perfil": [0, 3, 2, 0, 0, 0, 2, -1], "tipos": ["Eléctrico", "Normal"]}}},

    # BLOQUE 2: Aficiones y tiempo libre
    {"pregunta": "6. Te regalan una tarde libre sin planes ni obligaciones. ¿Qué haces de verdad, no lo que dirías en una entrevista?", "opciones": {
        "A) Me tiro en el sofá a hacer scroll sin sentido durante horas": {"perfil": [0, 0, -3, -1, 0, 0, 1, 0], "tipos": ["Normal", "Agua"]},
        "B) Aprovecho para adelantar algo pendiente, no sé estar parado": {"perfil": [1, 0, 2, 1, 0, 0, -1, 1], "tipos": ["Acero", "Lucha"]},
        "C) Salgo a caminar sin rumbo y ver a dónde llego": {"perfil": [0, 0, 1, 0, 3, 0, 1, 0], "tipos": ["Volador", "Fantasma"]},
        "D) Llamo a alguien para que me acompañe, solo se me hace raro": {"perfil": [0, 4, 1, 0, 0, 1, 0, 0], "tipos": ["Hada", "Normal"]}}},
    {"pregunta": "7. En un juego de mesa con amigos, alguien está a punto de hacer una jugada que le hace ganar.", "opciones": {
        "A) Le ayudo a verla igual, gano si gano jugando bien": {"perfil": [-2, 2, 0, 1, 0, 3, -1, 0], "tipos": ["Hada", "Planta"]},
        "B) Cierro la boca y dejo que se equivoque solo": {"perfil": [1, -2, 0, 2, 0, -1, 2, 0], "tipos": ["Siniestro", "Psíquico"]},
        "C) Cambio de tema para distraerle un segundo, nada personal": {"perfil": [1, 1, 1, 1, 0, -1, 3, 0], "tipos": ["Veneno", "Eléctrico"]},
        "D) Me tiro un farol enorme para que dude de su propia jugada": {"perfil": [2, 0, 1, 2, 1, -1, 2, 1], "tipos": ["Siniestro", "Dragón"]}}},
    {"pregunta": "8. Descubres una serie o videojuego que te engancha muchísimo.", "opciones": {
        "A) Lo devoro en un fin de semana entero sin salir de casa": {"perfil": [0, -1, 1, 0, 0, 0, 1, 0], "tipos": ["Bicho", "Fantasma"]},
        "B) Lo voy dosificando para que dure y no se acabe pronto": {"perfil": [-1, 0, -1, 2, 0, 1, -1, 1], "tipos": ["Hielo", "Planta"]},
        "C) Necesito comentarlo con alguien mientras lo veo, si no pierde la gracia": {"perfil": [0, 3, 0, 0, 0, 0, 0, 0], "tipos": ["Normal", "Volador"]},
        "D) Me pongo a leer teorías y análisis en vez de seguir viéndolo": {"perfil": [0, -1, 0, 4, 1, 0, 0, 0], "tipos": ["Psíquico", "Acero"]}}},
    {"pregunta": "9. Estás decorando o montando algo para tu casa/habitación.", "opciones": {
        "A) Sigo un plano exacto, medido con regla y todo": {"perfil": [0, 0, 0, 4, 0, 0, -2, 0], "tipos": ["Acero", "Roca"]},
        "B) Voy improvisando según lo que me gusta al verlo puesto": {"perfil": [0, 0, 1, -1, 1, 0, 2, 0], "tipos": ["Planta", "Hada"]},
        "C) Pido opinión a todo el mundo aunque luego haga lo mío": {"perfil": [0, 2, 0, 0, 0, 0, 0, 0], "tipos": ["Normal", "Agua"]},
        "D) Lo dejo a medias, ya lo terminaré otro día que me apetezca más": {"perfil": [0, 0, -2, 0, 0, -1, 1, 0], "tipos": ["Fantasma", "Bicho"]}}},
    {"pregunta": "10. En un videojuego cooperativo, tu personaje muere por un error tuyo estúpido.", "opciones": {
        "A) Me río el primero, la cago fuerte pero se puede reír": {"perfil": [0, 2, 1, 0, 0, 0, 1, -2], "tipos": ["Normal", "Eléctrico"]},
        "B) Le echo la culpa al lag antes de asumir nada": {"perfil": [1, 0, 0, 0, 0, -1, 2, 1], "tipos": ["Veneno", "Siniestro"]},
        "C) Me quedo callado un rato dándole vueltas a cómo evitarlo la próxima": {"perfil": [0, -1, 0, 3, 0, 0, -1, 1], "tipos": ["Acero", "Psíquico"]},
        "D) Me frustro en serio, odio quedar mal delante de otros": {"perfil": [2, -1, 1, 0, 0, 0, 0, 3], "tipos": ["Fuego", "Dragón"]}}},

    # BLOQUE 3: Reacciones automáticas
    {"pregunta": "11. Te pisan sin querer en el metro y ni se disculpan.", "opciones": {
        "A) Lo dejo pasar, seguro ni se ha dado cuenta": {"perfil": [-2, 0, 0, 0, 0, 0, 0, -1], "tipos": ["Agua", "Normal"]},
        "B) Carraspeo fuerte a ver si capta la indirecta": {"perfil": [1, 0, 0, 0, 0, 0, 1, 1], "tipos": ["Eléctrico", "Roca"]},
        "C) Se lo digo directamente pero sin mala leche": {"perfil": [1, 1, 0, 1, 0, 0, -1, 0], "tipos": ["Lucha", "Normal"]},
        "D) Me quedo rumiándolo el resto del trayecto en silencio": {"perfil": [0, -2, -1, 1, 0, 0, 0, 2], "tipos": ["Fantasma", "Hielo"]}}},
    {"pregunta": "12. Notas que una araña grande se pasea por tu habitación de noche.", "opciones": {
        "A) La observo un rato con curiosidad antes de decidir qué hacer": {"perfil": [0, 0, 0, 2, 2, 0, 0, 0], "tipos": ["Bicho", "Psíquico"]},
        "B) Grito y salgo pitando de la habitación": {"perfil": [0, 1, 3, -1, 0, 0, 1, -1], "tipos": ["Volador", "Normal"]},
        "C) Cojo un vaso y la saco fuera con cuidado de no hacerle daño": {"perfil": [-2, 1, 0, 0, 0, 2, 0, 0], "tipos": ["Hada", "Planta"]},
        "D) La aplasto sin pensarlo dos veces, no hay negociación posible": {"perfil": [3, -1, 0, 0, 0, 0, 0, 1], "tipos": ["Lucha", "Roca"]}}},
    {"pregunta": "13. Alguien te cuenta un secreto muy fuerte de otra persona que conocéis los dos.", "opciones": {
        "A) Se me escapa con el tiempo, no soy bueno guardando cosas así": {"perfil": [0, 1, 0, -1, 0, -2, 2, 0], "tipos": ["Normal", "Volador"]},
        "B) Me lo llevo a la tumba, ni loco lo repito": {"perfil": [0, 0, 0, 0, 0, 4, -1, 0], "tipos": ["Acero", "Siniestro"]},
        "C) Le doy vueltas para entender por qué me lo ha contado a mí": {"perfil": [0, -1, 0, 3, 1, 0, 0, 0], "tipos": ["Psíquico", "Fantasma"]},
        "D) Lo guardo pero lo uso de as bajo la manga si hace falta": {"perfil": [1, -1, 0, 1, 0, -2, 2, 1], "tipos": ["Siniestro", "Veneno"]}}},
    {"pregunta": "14. Vas conduciendo o en bici y un coche te corta el paso de mala manera.", "opciones": {
        "A) Toco el pito y suelto un par de cosas por la ventana": {"perfil": [3, 0, 1, 0, 0, 0, 0, 0], "tipos": ["Fuego", "Eléctrico"]},
        "B) Freno, respiro y sigo, no vale la pena el disgusto": {"perfil": [-2, 0, -1, 1, 0, 0, -1, 0], "tipos": ["Agua", "Acero"]},
        "C) Me acuerdo de la matrícula por si acaso algún día coincide otra vez": {"perfil": [0, 0, 0, 2, 0, 0, 1, 2], "tipos": ["Dragón", "Siniestro"]},
        "D) Me da un vuelco el corazón y voy con miedo el resto del camino": {"perfil": [0, 0, 0, 0, 1, 0, 0, -2], "tipos": ["Hielo", "Agua"]}}},
    {"pregunta": "15. Te avisan quince minutos antes de que va a venir gente a tu casa y está hecha un desastre.", "opciones": {
        "A) Limpio a toda velocidad como si me fuera la vida en ello": {"perfil": [0, 0, 3, 0, 0, 1, 0, 1], "tipos": ["Eléctrico", "Lucha"]},
        "B) Escondo el desorden en cualquier armario y listo": {"perfil": [0, 0, 1, 0, 0, -1, 3, 0], "tipos": ["Veneno", "Normal"]},
        "C) Aviso de que va a estar así, que no se asusten": {"perfil": [-1, 2, 0, 0, 0, 0, 0, -2], "tipos": ["Hada", "Agua"]},
        "D) Ni me inmuto, el que venga que venga como está": {"perfil": [0, -1, -1, 0, 0, 0, 0, 3], "tipos": ["Roca", "Dragón"]}}},

    # BLOQUE 4: Dilemas con más filo
    {"pregunta": "16. Tienes que elegir entre decir una mentira piadosa o una verdad que duele.", "opciones": {
        "A) Verdad siempre, aunque sea incómoda para todos": {"perfil": [1, -1, 0, 1, 0, 1, 0, 2], "tipos": ["Roca", "Lucha"]},
        "B) Mentira si eso evita hacer daño a quien quiero": {"perfil": [-2, 2, 0, 0, 0, 3, 0, -1], "tipos": ["Hada", "Agua"]},
        "C) Busco una tercera opción que no sea ni una cosa ni otra": {"perfil": [0, 0, 0, 3, 0, 0, 1, 0], "tipos": ["Psíquico", "Acero"]},
        "D) Depende de si me conviene más a mí, siendo sincero": {"perfil": [0, -1, 0, 1, 0, -2, 2, 0], "tipos": ["Siniestro", "Veneno"]}}},
    {"pregunta": "17. En un grupo, dos personas que te caen bien empiezan a discutir fuerte.", "opciones": {
        "A) Me meto en medio a poner paz aunque no me lo pidan": {"perfil": [-1, 3, 1, 0, 0, 2, -1, 0], "tipos": ["Hada", "Normal"]},
        "B) Me quedo a un lado observando cómo se desarrolla todo": {"perfil": [0, -1, 0, 2, 1, 0, 0, 0], "tipos": ["Fantasma", "Psíquico"]},
        "C) Suelto una broma para bajar la tensión de golpe": {"perfil": [0, 2, 1, 0, 0, 0, 1, -1], "tipos": ["Eléctrico", "Normal"]},
        "D) Me quedo con quien tenga razón, sin más": {"perfil": [1, -1, 0, 2, 0, -1, 0, 1], "tipos": ["Acero", "Roca"]}}},
    {"pregunta": "18. Te piden opinión sobre algo que la otra persona ha hecho con mucho esfuerzo pero no te gusta.", "opciones": {
        "A) Destaco lo bueno y suavizo lo que no me convence": {"perfil": [-1, 2, 0, 1, 0, 1, 0, -1], "tipos": ["Hada", "Planta"]},
        "B) Digo exactamente lo que pienso, para eso me pregunta": {"perfil": [1, 0, 0, 1, 0, 0, 0, 2], "tipos": ["Roca", "Dragón"]},
        "C) Le pregunto qué opina él primero para no comprometerme": {"perfil": [0, 1, 0, 2, 0, -1, 1, 0], "tipos": ["Psíquico", "Siniestro"]},
        "D) Cambio de tema con disimulo antes de que me insista": {"perfil": [0, 0, 0, 0, 0, 0, 3, -1], "tipos": ["Fantasma", "Veneno"]}}},
    {"pregunta": "19. Metes la pata en algo delante de gente que te importa impresionar.", "opciones": {
        "A) Lo reconozco en el momento y me río de mí mismo": {"perfil": [0, 2, 1, 0, 0, 0, 1, -2], "tipos": ["Normal", "Eléctrico"]},
        "B) Improviso una excusa creíble sobre la marcha": {"perfil": [0, 1, 0, 2, 0, -1, 2, 0], "tipos": ["Siniestro", "Volador"]},
        "C) Se me queda grabado días, aunque nadie más se acuerde": {"perfil": [0, -1, -1, 0, 1, 0, 0, 2], "tipos": ["Hielo", "Fantasma"]},
        "D) Sigo como si nada hubiera pasado, sin dar explicaciones": {"perfil": [0, -1, 0, 0, 0, 0, -1, 3], "tipos": ["Roca", "Dragón"]}}},
    {"pregunta": "20. Un desconocido te pide dinero por la calle.", "opciones": {
        "A) Le doy lo que llevo suelto sin pensarlo mucho": {"perfil": [-1, 2, 0, 0, 0, 1, 0, -1], "tipos": ["Hada", "Agua"]},
        "B) Le pregunto para qué lo necesita antes de decidir": {"perfil": [0, 1, 0, 2, 0, 0, 0, 0], "tipos": ["Psíquico", "Normal"]},
        "C) Sigo andando, prefiero no involucrarme en la calle": {"perfil": [0, -2, 0, 1, 0, -1, 0, 1], "tipos": ["Acero", "Hielo"]},
        "D) Le ofrezco comprarle algo de comer en vez de dinero": {"perfil": [0, 1, 0, 1, 0, 2, -1, 0], "tipos": ["Planta", "Tierra"]}}},

    # BLOQUE 5: Gustos y estética
    {"pregunta": "21. Eliges música para un viaje largo en coche con más gente.", "opciones": {
        "A) Pongo lo que a mí me gusta, ya se acostumbrarán": {"perfil": [1, -1, 0, 0, 0, -1, 0, 2], "tipos": ["Roca", "Dragón"]},
        "B) Voy alternando gustos para que todos tengan su rato": {"perfil": [-1, 3, 0, 0, 0, 1, 0, -1], "tipos": ["Normal", "Hada"]},
        "C) Hago una lista rara con cosas que nadie conoce, a ver qué opinan": {"perfil": [0, 0, 0, 1, 2, 0, 1, 0], "tipos": ["Fantasma", "Psíquico"]},
        "D) Me da igual, con el silencio o el ruido de fondo voy bien": {"perfil": [-2, -1, -1, 0, 0, 0, 0, 0], "tipos": ["Agua", "Normal"]}}},
    {"pregunta": "22. Vas a comprar ropa nueva. ¿Qué criterio pesa más?", "opciones": {
        "A) Que sea cómoda, lo demás me da bastante igual": {"perfil": [-1, -1, 0, 0, 0, 0, 0, 0], "tipos": ["Normal", "Planta"]},
        "B) Que llame la atención, que se note que estoy ahí": {"perfil": [1, 1, 1, 0, 0, 0, 0, 3], "tipos": ["Fuego", "Volador"]},
        "C) Que combine bien con todo lo que ya tengo, tipo cápsula": {"perfil": [0, 0, 0, 3, 0, 0, -2, 1], "tipos": ["Acero", "Hielo"]},
        "D) Algo distinto a lo normal, oscuro o con rollo raro": {"perfil": [0, -1, 0, 0, 3, 0, 1, 0], "tipos": ["Siniestro", "Fantasma"]}}},
    {"pregunta": "23. En un restaurante, el plato te llega mal hecho o frío.", "opciones": {
        "A) Lo devuelvo sin problema, para eso pago": {"perfil": [1, 0, 0, 0, 0, 0, 0, 1], "tipos": ["Fuego", "Roca"]},
        "B) Me lo como igual, no quiero dar la lata": {"perfil": [-2, 1, 0, 0, 0, 0, 0, -2], "tipos": ["Agua", "Planta"]},
        "C) Se lo comento al camarero pero de forma suave y con una sonrisa": {"perfil": [-1, 2, 0, 1, 0, 1, 0, 0], "tipos": ["Hada", "Normal"]},
        "D) No digo nada pero no vuelvo nunca más a ese sitio": {"perfil": [0, -1, 0, 1, 0, 0, 1, 2], "tipos": ["Hielo", "Siniestro"]}}},
    {"pregunta": "24. Un olor muy fuerte y desagradable inunda de repente la habitación donde estás.", "opciones": {
        "A) Abro ventanas y me pongo a buscar el origen del problema": {"perfil": [0, 0, 1, 2, 0, 0, -1, 0], "tipos": ["Volador", "Acero"]},
        "B) Aguanto ahí como si no pasara nada especial": {"perfil": [0, 0, -1, 0, 0, 0, 0, 1], "tipos": ["Veneno", "Roca"]},
        "C) Me quejo alto para que alguien más se encargue": {"perfil": [1, 1, 0, 0, 0, 0, 0, 0], "tipos": ["Eléctrico", "Normal"]},
        "D) Me voy de la habitación sin dar más explicaciones": {"perfil": [0, -2, 1, 0, 1, 0, 0, 0], "tipos": ["Fantasma", "Hielo"]}}},
    {"pregunta": "25. Escoge sin pensarlo mucho: ¿qué franja horaria sientes más tuya?", "opciones": {
        "A) La madrugada, cuando no hay ni un alma despierta": {"perfil": [0, -2, 0, 1, 3, 0, 1, 0], "tipos": ["Siniestro", "Fantasma"]},
        "B) Primera hora de la mañana, con todo por estrenar": {"perfil": [0, 1, 2, 0, 0, 0, -1, 0], "tipos": ["Volador", "Normal"]},
        "C) El mediodía, cuando todo está en marcha y hay movimiento": {"perfil": [1, 3, 2, 0, 0, 0, 0, 0], "tipos": ["Fuego", "Eléctrico"]},
        "D) El atardecer, ese rato tranquilo antes de que caiga la noche": {"perfil": [-2, 0, -2, 0, 1, 0, 0, 0], "tipos": ["Agua", "Planta"]}}},

    # BLOQUE 6: Rol, fantasía y absurdo
    {"pregunta": "26. En una partida de rol, el máster te ofrece un objeto mágico misterioso sin explicar qué hace.", "opciones": {
        "A) Lo uso ya mismo, ya se verá qué pasa": {"perfil": [1, 0, 2, -1, 1, 0, 2, 0], "tipos": ["Dragón", "Eléctrico"]},
        "B) Investigo primero preguntando o buscando pistas dentro del juego": {"perfil": [0, 0, 0, 3, 1, 0, -1, 0], "tipos": ["Psíquico", "Bicho"]},
        "C) Se lo regalo a otro jugador, que se arriesgue él": {"perfil": [-1, 1, 0, 0, 0, -1, 1, 0], "tipos": ["Normal", "Veneno"]},
        "D) Lo guardo sin usarlo, prefiero tenerlo por si acaso": {"perfil": [0, -1, 0, 1, 0, 1, -1, 1], "tipos": ["Acero", "Roca"]}}},
    {"pregunta": "27. Si pudieras tener un poder pero con una desventaja fea incluida, ¿cuál eliges?", "opciones": {
        "A) Fuerza sobrehumana, aunque se me agote la energía enseguida": {"perfil": [3, 0, 1, -1, 0, 0, 0, 1], "tipos": ["Lucha", "Tierra"]},
        "B) Leer mentes, aunque no pueda desconectarlo nunca": {"perfil": [0, 0, 0, 3, 1, 0, 0, 0], "tipos": ["Psíquico", "Fantasma"]},
        "C) Invisibilidad, aunque nadie vuelva a acordarse de mí": {"perfil": [0, -3, 0, 0, 2, 0, 1, -1], "tipos": ["Fantasma", "Siniestro"]},
        "D) Curar a otros, aunque a mí me cueste el doble sanar": {"perfil": [-2, 2, 0, 0, 0, 3, 0, 0], "tipos": ["Hada", "Planta"]}}},
    {"pregunta": "28. Construyendo tu base o fortaleza ideal en un videojuego de supervivencia.", "opciones": {
        "A) La lleno de trampas y defensas por todos lados": {"perfil": [2, -1, 0, 1, 0, 0, 1, 0], "tipos": ["Veneno", "Roca"]},
        "B) La hago acogedora, con sitio para que quepan más personas": {"perfil": [-1, 3, 0, 0, 0, 1, 0, 0], "tipos": ["Hada", "Normal"]},
        "C) La escondo bien camuflada, que nadie sepa que existe": {"perfil": [0, -2, 0, 2, 1, 0, 0, 0], "tipos": ["Fantasma", "Bicho"]},
        "D) La hago enorme y vistosa, que se vea desde lejos": {"perfil": [0, 0, 1, 0, 0, 0, 0, 4], "tipos": ["Dragón", "Roca"]}}},
    {"pregunta": "29. Vas a nombrar a tu mascota o personaje. ¿Qué tipo de nombre eliges de verdad?", "opciones": {
        "A) Algo gracioso o absurdo que me haga reír cada vez que lo diga": {"perfil": [0, 2, 1, 0, 0, 0, 2, 0], "tipos": ["Normal", "Eléctrico"]},
        "B) Un nombre serio y con peso, que suene a algo importante": {"perfil": [1, 0, 0, 1, 0, 0, -1, 2], "tipos": ["Dragón", "Roca"]},
        "C) Algo tierno, relacionado con cariño o con alguien que quiero": {"perfil": [-2, 2, 0, 0, 0, 2, 0, 0], "tipos": ["Hada", "Agua"]},
        "D) Algo raro sacado de mitología o de un idioma que me suena bien": {"perfil": [0, 0, 0, 1, 3, 0, 0, 0], "tipos": ["Psíquico", "Fantasma"]}}},
    {"pregunta": "30. Encuentras una puerta cerrada con un cartel de NO PASAR en un sitio abandonado.", "opciones": {
        "A) La abro, la curiosidad me puede siempre": {"perfil": [1, 0, 1, 0, 2, 0, 1, 0], "tipos": ["Fantasma", "Siniestro"]},
        "B) Paso de largo, un cartel es un cartel": {"perfil": [-1, 0, 0, 1, 0, 1, -1, 0], "tipos": ["Acero", "Normal"]},
        "C) Busco otra forma de entrar, tipo ventana, por si acaso": {"perfil": [1, -1, 0, 2, 0, 0, 1, 0], "tipos": ["Siniestro", "Bicho"]},
        "D) Se lo cuento a alguien más para ir juntos a mirar": {"perfil": [0, 2, 1, 0, 1, 0, 0, 0], "tipos": ["Volador", "Normal"]}}},

    # BLOQUE 7: Valores y forma de ser
    {"pregunta": "31. ¿Qué te cuesta más perdonar de verdad, aunque digas que ya lo has superado?", "opciones": {
        "A) Que me mientan directamente a la cara": {"perfil": [1, 0, 0, 0, 0, 2, -1, 1], "tipos": ["Lucha", "Acero"]},
        "B) Que me dejen de lado sin motivo aparente": {"perfil": [0, 1, 0, 0, 0, 1, 0, -2], "tipos": ["Hada", "Fantasma"]},
        "C) Que no valoren el esfuerzo que he puesto en algo": {"perfil": [0, 0, 0, 1, 0, 0, 0, 3], "tipos": ["Roca", "Dragón"]},
        "D) Que hablen sin saber y aun así opinen como si supieran": {"perfil": [0, -1, 0, 3, 0, 0, 0, 1], "tipos": ["Psíquico", "Acero"]}}},
    {"pregunta": "32. En el trabajo o clase, ¿cómo prefieres que te reconozcan un buen resultado?", "opciones": {
        "A) En privado, no necesito que se entere todo el mundo": {"perfil": [-1, -1, 0, 0, 0, 0, 0, 1], "tipos": ["Agua", "Acero"]},
        "B) Delante de todos, que se sepa el curro que ha costado": {"perfil": [1, 2, 1, 0, 0, 0, 0, 3], "tipos": ["Fuego", "Dragón"]},
        "C) Con algo concreto, un aumento o un beneficio real": {"perfil": [0, -1, 0, 2, 0, 0, 1, 0], "tipos": ["Siniestro", "Acero"]},
        "D) La verdad, ni lo necesito, sé lo que he hecho yo mismo": {"perfil": [0, -2, 0, 1, 0, 0, -1, 2], "tipos": ["Hielo", "Roca"]}}},
    {"pregunta": "33. Un amigo te pide que le acompañes a hacer algo que a ti te da bastante pereza.", "opciones": {
        "A) Voy sin quejarme, para eso están los amigos": {"perfil": [-1, 2, 0, 0, 0, 3, 0, -1], "tipos": ["Hada", "Normal"]},
        "B) Le digo que no esta vez, sinceramente": {"perfil": [0, -1, 0, 1, 0, -1, 0, 1], "tipos": ["Acero", "Roca"]},
        "C) Voy pero le hago saber que me está costando el esfuerzo": {"perfil": [1, 1, 0, 0, 0, 1, 0, 1], "tipos": ["Tierra", "Normal"]},
        "D) Le propongo un plan alternativo que nos guste a los dos": {"perfil": [0, 1, 0, 2, 0, 1, 0, 0], "tipos": ["Planta", "Psíquico"]}}},
    {"pregunta": "34. Sientes que llevas un tiempo estancado, sin avanzar en nada importante.", "opciones": {
        "A) Lo cambio todo de golpe, necesito una sacudida fuerte": {"perfil": [1, 0, 3, 0, 0, -1, 2, 0], "tipos": ["Eléctrico", "Dragón"]},
        "B) Me paro a analizar en qué punto exacto me quedé atascado": {"perfil": [0, -1, 0, 4, 0, 0, -1, 0], "tipos": ["Acero", "Psíquico"]},
        "C) Espero, confío en que estas cosas pasan solas con tiempo": {"perfil": [-2, 0, -2, 0, 1, 0, 0, 0], "tipos": ["Agua", "Planta"]},
        "D) Me machaco bastante por dentro, aunque por fuera no se note": {"perfil": [0, -1, 0, 0, 1, 0, 0, -3], "tipos": ["Fantasma", "Hielo"]}}},
    {"pregunta": "35. En una foto de grupo, ¿dónde acabas normalmente colocado?", "opciones": {
        "A) En el centro, sin buscarlo demasiado pero ahí acabo": {"perfil": [1, 1, 0, 0, 0, 0, 0, 2], "tipos": ["Dragón", "Fuego"]},
        "B) Al lado de mi gente de confianza, esté donde esté eso": {"perfil": [-1, 2, 0, 0, 0, 3, 0, -1], "tipos": ["Hada", "Normal"]},
        "C) En una esquina, o directamente detrás de la cámara": {"perfil": [0, -2, 0, 1, 1, 0, 0, -1], "tipos": ["Fantasma", "Bicho"]},
        "D) Donde sea, ya haciendo alguna tontería para salir raro": {"perfil": [0, 1, 2, 0, 0, 0, 3, -1], "tipos": ["Eléctrico", "Normal"]}}},

    # BLOQUE 8: Miedos, presión y últimas
    {"pregunta": "36. Estás a solas en casa y escuchas un ruido raro que no sabes de dónde viene.", "opciones": {
        "A) Voy directo a investigar, prefiero saber ya qué es": {"perfil": [2, 0, 1, 1, 0, 0, 0, 1], "tipos": ["Lucha", "Roca"]},
        "B) Me quedo quieto escuchando con atención antes de moverme": {"perfil": [0, -1, -1, 2, 1, 0, 0, 0], "tipos": ["Psíquico", "Hielo"]},
        "C) Pongo música o la tele más alta para no pensar en ello": {"perfil": [-1, 0, 0, 0, 0, 0, 1, 0], "tipos": ["Normal", "Eléctrico"]},
        "D) Le escribo a alguien contándole lo que está pasando, por si acaso": {"perfil": [0, 2, 0, 0, 1, 1, 0, -1], "tipos": ["Fantasma", "Hada"]}}},
    {"pregunta": "37. Tienes que hablar en público sobre algo que dominas poco.", "opciones": {
        "A) Improviso con seguridad, el tono lo es casi todo": {"perfil": [1, 2, 1, 0, 0, 0, 1, 1], "tipos": ["Fuego", "Volador"]},
        "B) Preparo cada palabra por escrito para no dejar nada al azar": {"perfil": [0, -1, 0, 4, 0, 0, -2, 0], "tipos": ["Acero", "Psíquico"]},
        "C) Empiezo reconociendo que no soy experto, y de ahí para arriba": {"perfil": [-1, 1, 0, 1, 0, 1, 0, -2], "tipos": ["Agua", "Normal"]},
        "D) Lo evito como sea, delego en quien pueda hacerlo si existe opción": {"perfil": [0, -2, -1, 0, 0, 0, 0, 0], "tipos": ["Bicho", "Hielo"]}}},
    {"pregunta": "38. Si tuvieras que desaparecer literalmente un día entero sin dar explicaciones, ¿qué harías con ese tiempo?", "opciones": {
        "A) Algo físico y fuerte, correr, escalar, quemar energía": {"perfil": [2, -1, 4, 0, 0, 0, 0, 0], "tipos": ["Lucha", "Volador"]},
        "B) Meterme en un sitio nuevo que no conozca nadie de mi vida": {"perfil": [0, 0, 1, 0, 2, -1, 1, 0], "tipos": ["Fantasma", "Volador"]},
        "C) Quedarme en algún sitio tranquilo leyendo o pensando": {"perfil": [-2, -1, -2, 2, 1, 0, 0, 0], "tipos": ["Psíquico", "Agua"]},
        "D) Buscar a alguien de mi pasado con quien perdí el contacto": {"perfil": [0, 1, 0, 0, 0, 3, 0, 0], "tipos": ["Hada", "Normal"]}}},
    {"pregunta": "39. Notas que estás empezando a caerle mal a alguien sin motivo claro.", "opciones": {
        "A) Se lo pregunto directamente, prefiero saberlo cuanto antes": {"perfil": [1, 1, 0, 1, 0, 0, 0, 0], "tipos": ["Lucha", "Normal"]},
        "B) Le doy vueltas mentalmente sin decir nada durante días": {"perfil": [0, -1, 0, 2, 1, 0, 0, 1], "tipos": ["Psíquico", "Fantasma"]},
        "C) Sigo a lo mío, no todo el mundo tiene que quererme": {"perfil": [0, -2, 0, 0, 0, 0, -1, 2], "tipos": ["Roca", "Dragón"]},
        "D) Intento agradarle un poco más de lo normal, me incomoda esa idea": {"perfil": [-2, 2, 0, 0, 0, 1, 0, -2], "tipos": ["Hada", "Agua"]}}},
    {"pregunta": "40. Si mañana te despertaras con una habilidad nueva pero al azar, ¿qué prefieres que te toque?", "opciones": {
        "A) Una fuerza descomunal, aunque no supiera controlarla del todo": {"perfil": [4, 0, 1, -1, 0, 0, 1, 0], "tipos": ["Lucha", "Dragón"]},
        "B) Saber exactamente qué siente la gente a mi alrededor": {"perfil": [-1, 3, 0, 0, 1, 1, 0, 0], "tipos": ["Hada", "Psíquico"]},
        "C) Poder ver patrones y conexiones que nadie más ve": {"perfil": [0, -1, 0, 4, 1, 0, 0, 0], "tipos": ["Psíquico", "Acero"]},
        "D) Volverme completamente invisible a voluntad cuando quisiera": {"perfil": [0, -2, 0, 0, 3, 0, 2, -1], "tipos": ["Fantasma", "Siniestro"]}}},

    # BLOQUE 9: Cierre, un poco más personal
    {"pregunta": "41. Cuando alguien llora delante de ti, ¿cuál es tu primer impulso, no el que crees que deberías tener?", "opciones": {
        "A) Abrazar o tocar, necesito hacer algo físico ya": {"perfil": [-1, 3, 1, 0, 0, 2, 0, 0], "tipos": ["Hada", "Agua"]},
        "B) Buscar una solución práctica al problema que lo causó": {"perfil": [0, 0, 0, 3, 0, 1, 0, 0], "tipos": ["Acero", "Tierra"]},
        "C) Quedarme un poco paralizado sin saber muy bien qué hacer": {"perfil": [0, -1, -1, 0, 1, 0, 0, 0], "tipos": ["Hielo", "Normal"]},
        "D) Sentir la emoción como si fuera un poco mía también": {"perfil": [0, 1, 0, 0, 3, 1, 0, -1], "tipos": ["Psíquico", "Fantasma"]}}},
    {"pregunta": "42. Ganas un premio pequeño en un sorteo random.", "opciones": {
        "A) Me pongo eufórico un rato, aunque sea una tontería sin valor": {"perfil": [0, 1, 3, 0, 0, 0, 1, 0], "tipos": ["Eléctrico", "Normal"]},
        "B) Pienso enseguida a quién se lo puedo regalar": {"perfil": [-2, 2, 0, 0, 0, 2, 0, -1], "tipos": ["Hada", "Planta"]},
        "C) Sospecho un poco, nada es gratis del todo": {"perfil": [0, -1, 0, 2, 0, 0, 1, 0], "tipos": ["Siniestro", "Veneno"]},
        "D) Ni me inmuto demasiado, ya se me pasa el efecto rápido": {"perfil": [0, -1, -1, 0, 0, 0, 0, 1], "tipos": ["Roca", "Acero"]}}},
    {"pregunta": "43. Un plan que llevabas semanas montando se cae en el último momento por algo fuera de tu control.", "opciones": {
        "A) Monto uno nuevo en el momento, no me quedo parado": {"perfil": [1, 0, 2, 1, 0, 0, 1, 0], "tipos": ["Eléctrico", "Volador"]},
        "B) Me frustro bastante, me cuesta soltar lo que ya tenía planeado": {"perfil": [1, 0, 0, 0, 0, 0, 0, 1], "tipos": ["Roca", "Tierra"]},
        "C) Lo dejo estar y aprovecho el hueco para no hacer nada": {"perfil": [-2, 0, -2, 0, 0, 0, 0, 0], "tipos": ["Agua", "Normal"]},
        "D) Repaso mentalmente qué falló para que no vuelva a pasar": {"perfil": [0, -1, 0, 3, 0, 0, -1, 0], "tipos": ["Acero", "Psíquico"]}}},
    {"pregunta": "44. Te piden que elijas entre trabajar solo con total libertad o en equipo con más apoyo pero menos control.", "opciones": {
        "A) Solo, sin duda, decido yo cómo y cuándo": {"perfil": [1, -2, 0, 1, 0, -1, 0, 2], "tipos": ["Roca", "Acero"]},
        "B) En equipo, se rinde más y se pasa mejor entre varios": {"perfil": [-1, 3, 0, 0, 0, 2, 0, -1], "tipos": ["Hada", "Normal"]},
        "C) Depende totalmente de la tarea, no tengo una postura fija": {"perfil": [0, 0, 0, 2, 0, 0, 0, 0], "tipos": ["Psíquico", "Normal"]},
        "D) Solo, pero en secreto me gustaría que alguien notara el resultado": {"perfil": [0, -1, 0, 1, 0, 0, 0, 3], "tipos": ["Dragón", "Siniestro"]}}},
    {"pregunta": "45. Alguien te copia claramente una idea o forma de hacer algo tuyo.", "opciones": {
        "A) Se lo digo directo, sin darle más vueltas al asunto": {"perfil": [1, 0, 0, 1, 0, 0, 0, 1], "tipos": ["Lucha", "Roca"]},
        "B) Me halaga un poco en el fondo, imitar es un cumplido": {"perfil": [-1, 1, 0, 0, 0, 0, 0, -1], "tipos": ["Agua", "Hada"]},
        "C) Me guardo el enfado y sigo innovando para ir un paso por delante": {"perfil": [0, -1, 1, 2, 0, 0, 0, 2], "tipos": ["Acero", "Dragón"]},
        "D) Empiezo a vigilar de cerca a esa persona, con desconfianza": {"perfil": [1, -1, 0, 1, 0, -1, 2, 0], "tipos": ["Siniestro", "Veneno"]}}},

    # BLOQUE 10: Vuelta a lo cotidiano
    {"pregunta": "46. En una comida familiar sale un tema políticamente delicado.", "opciones": {
        "A) Doy mi opinión aunque sepa que va a generar roce": {"perfil": [1, 0, 0, 0, 0, 0, 0, 2], "tipos": ["Dragón", "Lucha"]},
        "B) Cambio de tema con disimulo hacia algo más ligero": {"perfil": [-1, 1, 0, 1, 0, 0, 0, 0], "tipos": ["Normal", "Hada"]},
        "C) Escucho todos los puntos de vista sin posicionarme aún": {"perfil": [0, 0, 0, 3, 0, 0, 0, 0], "tipos": ["Psíquico", "Agua"]},
        "D) Me quedo callado pero por dentro ya estoy con la sangre hirviendo": {"perfil": [1, -1, 0, 0, 0, 0, 0, 0], "tipos": ["Fuego", "Siniestro"]}}},
    {"pregunta": "47. Vas a coger vacaciones. ¿Qué pesa más al elegir destino?", "opciones": {
        "A) Naturaleza salvaje, cuanto menos gente mejor": {"perfil": [0, -2, 1, 0, 2, 0, 0, 0], "tipos": ["Planta", "Tierra"]},
        "B) Ciudad con mucha vida, gente, ruido y planes": {"perfil": [0, 3, 2, 0, 0, 0, 0, 0], "tipos": ["Eléctrico", "Normal"]},
        "C) Un sitio con historia, museos, cosas que aprender de verdad": {"perfil": [0, -1, 0, 4, 0, 0, 0, 1], "tipos": ["Acero", "Psíquico"]},
        "D) Da igual dónde, con tal de que sea algo distinto a lo que conozco": {"perfil": [0, 0, 1, 0, 2, 0, 2, 0], "tipos": ["Dragón", "Volador"]}}},
    {"pregunta": "48. Te devuelven de más en un cambio en una tienda pequeña de barrio.", "opciones": {
        "A) Lo digo al momento, aunque sea una moneda suelta": {"perfil": [0, 1, 0, 0, 0, 3, 0, 1], "tipos": ["Lucha", "Acero"]},
        "B) Lo pienso un segundo mientras salgo, y decido si vuelvo o no": {"perfil": [0, -1, 0, 2, 0, 0, 1, 0], "tipos": ["Psíquico", "Siniestro"]},
        "C) Me lo quedo, tampoco es tanto dinero al final": {"perfil": [0, -1, 0, 0, 0, -2, 2, 0], "tipos": ["Veneno", "Normal"]},
        "D) Me sabe mal para el dueño de la tienda pequeña y vuelvo a devolverlo": {"perfil": [-1, 1, 0, 0, 0, 3, -1, 0], "tipos": ["Hada", "Planta"]}}},
    {"pregunta": "49. Estás enseñando algo a alguien y no lo capta a la primera.", "opciones": {
        "A) Lo repito con paciencia las veces que hagan falta": {"perfil": [-2, 1, 0, 1, 0, 2, 0, 0], "tipos": ["Hada", "Planta"]},
        "B) Busco otra forma distinta de explicarlo hasta dar con la que funcione": {"perfil": [0, 0, 0, 4, 0, 0, 0, 0], "tipos": ["Psíquico", "Acero"]},
        "C) Me impaciento un poco, aunque intente disimularlo": {"perfil": [2, -1, 1, 0, 0, 0, 0, 1], "tipos": ["Fuego", "Roca"]},
        "D) Acabo haciéndolo yo mismo, es más rápido que explicar": {"perfil": [1, -1, 0, 1, 0, 0, -1, 1], "tipos": ["Lucha", "Acero"]}}},
    {"pregunta": "50. Para cerrar: si tu forma de ser fuera un mensaje pegado en la puerta de tu cuarto, ¿cuál sería?", "opciones": {
        "A) 'Toca antes de entrar, luego lo que quieras'": {"perfil": [1, 0, 0, 0, 0, 1, 0, 1], "tipos": ["Roca", "Normal"]},
        "B) 'Puertas abiertas, aquí cabe quien quiera venir'": {"perfil": [-2, 4, 0, 0, 0, 2, 0, -1], "tipos": ["Hada", "Normal"]},
        "C) 'Entra bajo tu propio riesgo'": {"perfil": [1, -1, 0, 0, 2, 0, 2, 0], "tipos": ["Siniestro", "Fantasma"]},
        "D) 'Todo tiene su sitio, no lo desordenes'": {"perfil": [0, -1, 0, 4, 0, 0, -2, 1], "tipos": ["Acero", "Psíquico"]}}}
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
                
            # --- BLINDAJE ANTI-CEROS PARA LA BASE DE DATOS ---
            # Si algún Pokémon tiene todo a 0, le damos un 0.1 general para que exista en el mapa
            df_pokemon[columnas_dimensiones] = df_pokemon[columnas_dimensiones].replace(0, 0.1)

            # --- AJUSTE DE ESCALA (NORMALIZACIÓN) ---
            # El usuario puede acumular 30-40 puntos, pero los Pokémon del JSON están del 0 al 10.
            # Normalizamos el vector del usuario a una escala de 0 a 10 para que jueguen en la misma liga.
            max_puntos_usuario = np.max(perfil_personalidad)
            if max_puntos_usuario > 0:
                vector_usuario_norm = (perfil_personalidad / max_puntos_usuario) * 10
            else:
                vector_usuario_norm = perfil_personalidad
                
            vector_usuario_norm = vector_usuario_norm.reshape(1, -1)

            # 1. MATEMÁTICAS DEL POKÉMON (Distancia Euclidiana)
            # Mide la distancia real entre los puntos. Cuanto MENOR sea la distancia, MEJOR es el match.
            distancias = euclidean_distances(vector_usuario_norm, df_pokemon[columnas_dimensiones].values)[0]
            
            # Ordenamos de menor a mayor (los más cercanos van primero)
            indices_top = np.argsort(distancias)[:6]
            mejor_pokemon = df_pokemon.iloc[indices_top[0]]
            
            # Función para convertir la "distancia" geométrica en un porcentaje del 0 al 100%
            def calcular_afinidad(dist):
                # Restamos la distancia a 100 con un multiplicador de ajuste (4.5 suele dar resultados realistas)
                afinidad = 100 - (dist * 4.5)
                return round(max(0.1, afinidad), 1)

            porcentaje_afinidad = calcular_afinidad(distancias[indices_top[0]])
            
            # 2. LÓGICA DE GIMNASIO (Dominante y Secundario con Puntuación)
            tipos_ordenados = sorted(puntuacion_tipos.items(), key=lambda x: x[1], reverse=True)
            tipo_primario = tipos_ordenados[0][0]
            puntos_1 = tipos_ordenados[0][1]
            tipo_secundario = tipos_ordenados[1][0]
            puntos_2 = tipos_ordenados[1][1]
            
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
                
                # --- NUEVO RENDERIZADO DE GIMNASIO ---
                st.write(f"🏆 **Tipo Dominante:** {tipo_primario} ({puntos_1} pts)")
                st.write(f"🥈 **Tipo Secundario:** {tipo_secundario} ({puntos_2} pts)")
                
                st.write("---")
                st.markdown("**Tus 5 Pokémon más cercanos:**")
                # Bucle para mostrar del 2º al 6º clasificado (CORREGIDO)
                for i in indices_top[1:6]:
                    poke_cercano = df_pokemon.iloc[i]
                    # Usamos la nueva función calcular_afinidad y la variable distancias
                    afinidad_cercana = calcular_afinidad(distancias[i])
                    st.write(f"- **{poke_cercano['nombre']}** ({afinidad_cercana}%)")
                
            st.divider()
            
            st.subheader("El Por qué de tu Resultado (Tus Stats)")
            
            # Mostramos el Top 3 de dimensiones para que la gente compare
            st.write(f"1. **{estadisticas[0][0]}** dominante ({estadisticas[0][1]:.1f}%)")
            st.write(f"2. Fuerte presencia de **{estadisticas[1][0]}** ({estadisticas[1][1]:.1f}%)")
            if estadisticas[2][1] > 10.0:
                st.write(f"3. Matices de **{estadisticas[2][0]}** ({estadisticas[2][1]:.1f}%)")
                
            st.write(f"\nEsta combinación exacta en tu mapa de 8 dimensiones es la que el algoritmo ha emparejado milimétricamente con el *lore* y mecánicas internas de {mejor_pokemon['nombre']}.")