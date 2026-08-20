import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
from sklearn.metrics.pairwise import euclidean_distances

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Test Definitivo: Pokémon y Gimnasio", page_icon="🔮", layout="centered")

st.title("Escáner Psicológico Pokémon")
st.markdown("Responde a estas 50 situaciones. El algoritmo analizará 8 dimensiones de tu personalidad y cruzará tus mecánicas de juego para encontrar tu especie exacta y tu especialidad como Líder de Gimnasio.")

# --- 2. CARGAR BASE DE DATOS (.JSON) ---
@st.cache_data
def cargar_datos():
    return pd.read_json("vectores_pokemon_absoluto.json")

df_pokemon = cargar_datos()
columnas_dimensiones = ['Agresividad', 'Sociabilidad', 'Energia', 'Intelecto', 'Misticismo', 'Lealtad', 'Caos', 'Orgullo']

# --- 3. POOLS DE PREGUNTAS POR TEMÁTICA ---
# Orden del array: [Agresividad, Sociabilidad, Energia, Intelecto, Misticismo, Lealtad, Caos, Orgullo]
POOLS_PREGUNTAS = {
    "Gustos": [
        {"pregunta": "Eliges música para un viaje largo en coche con más gente.", "opciones": {
            "A) Pongo lo que a mí me gusta, ya se acostumbrarán": {"perfil": [1, -1, 0, 0, 0, -1, 0, 2], "tipos": ["Roca", "Dragón"]},
            "B) Voy alternando gustos para que todos tengan su rato": {"perfil": [-1, 3, 0, 0, 0, 1, 0, -1], "tipos": ["Normal", "Hada"]},
            "C) Hago una lista rara con cosas que nadie conoce, a ver qué opinan": {"perfil": [0, 0, 0, 1, 2, 0, 1, 0], "tipos": ["Fantasma", "Psíquico"]},
            "D) Me da igual, con el silencio o el ruido de fondo voy bien": {"perfil": [-2, -1, -1, 0, 0, 0, 0, 0], "tipos": ["Agua", "Normal"]}}},
        {"pregunta": "Vas a comprar o personalizar ropa nueva. ¿Qué criterio pesa más?", "opciones": {
            "A) Que sea cómoda, lo demás me da bastante igual": {"perfil": [-1, -1, 0, 0, 0, 0, 0, 0], "tipos": ["Normal", "Planta"]},
            "B) Aplicar mis propios diseños a mano con pintura de tela, que sea única": {"perfil": [0, 0, 1, 2, 0, 0, 2, 1], "tipos": ["Veneno", "Fuego"]},
            "C) Que combine bien con todo lo que ya tengo, tipo cápsula": {"perfil": [0, 0, 0, 3, 0, 0, -2, 1], "tipos": ["Acero", "Hielo"]},
            "D) Algo con estética Y2K, cyber-grunge o urbana que destaque": {"perfil": [0, 1, 0, 0, 2, 0, 2, 0], "tipos": ["Eléctrico", "Siniestro"]}}},
        {"pregunta": "Entras a un garito o festival. ¿Qué ritmo o vibra te hace quedarte?", "opciones": {
            "A) Algo contundente, tipo techno, hardstyle o psytrance que retumbe en el pecho": {"perfil": [2, 0, 3, 0, 0, 0, 2, 1], "tipos": ["Eléctrico", "Siniestro"]},
            "B) Melodías más envolventes para dejar la mente en blanco": {"perfil": [0, -1, -1, 0, 3, 0, 1, 0], "tipos": ["Psíquico", "Fantasma"]},
            "C) Lo que sea que la gente esté bailando, vengo a pasarlo bien con los míos": {"perfil": [0, 3, 2, 0, 0, 1, 0, 0], "tipos": ["Normal", "Fuego"]},
            "D) Me fijo más en la estética visual y acústica del sitio, si suena limpio me quedo": {"perfil": [0, 0, 0, 2, 0, 0, -1, 1], "tipos": ["Acero", "Roca"]}}},
        {"pregunta": "Montando tu setup ideal en casa (PC, estudio, etc.), ¿en qué inviertes más tiempo?", "opciones": {
            "A) En que el cableado y las interfaces de audio/MIDI estén ruteados a la perfección": {"perfil": [0, 0, 0, 4, 0, 0, -2, 1], "tipos": ["Acero", "Eléctrico"]},
            "B) En la estética, luces y decoración para que sea un espacio donde apetezca estar": {"perfil": [0, 1, 1, 0, 2, 0, 1, 0], "tipos": ["Hada", "Psíquico"]},
            "C) En puro rendimiento, potencia bruta para que nada me dé tirones": {"perfil": [1, -1, 2, 1, 0, 0, 0, 2], "tipos": ["Fuego", "Lucha"]},
            "D) Lo voy montando a trozos según lo necesito, sin mucha planificación inicial": {"perfil": [0, 0, -1, 0, 0, 0, 3, -1], "tipos": ["Bicho", "Veneno"]}}},
        {"pregunta": "Escoge sin pensarlo mucho: ¿qué franja horaria sientes más tuya?", "opciones": {
            "A) La madrugada, cuando no hay ni un alma despierta": {"perfil": [0, -2, 0, 1, 3, 0, 1, 0], "tipos": ["Siniestro", "Fantasma"]},
            "B) Primera hora de la mañana, con todo por estrenar": {"perfil": [0, 1, 2, 0, 0, 0, -1, 0], "tipos": ["Volador", "Normal"]},
            "C) El mediodía, cuando todo está en marcha y hay movimiento": {"perfil": [1, 3, 2, 0, 0, 0, 0, 0], "tipos": ["Fuego", "Eléctrico"]},
            "D) El atardecer, ese rato tranquilo antes de que caiga la noche": {"perfil": [-2, 0, -2, 0, 1, 0, 0, 0], "tipos": ["Agua", "Planta"]}}},
        {"pregunta": "En un restaurante, el plato te llega mal hecho o frío.", "opciones": {
            "A) Lo devuelvo sin problema, para eso pago": {"perfil": [1, 0, 0, 0, 0, 0, 0, 1], "tipos": ["Fuego", "Roca"]},
            "B) Me lo como igual, no quiero dar la lata": {"perfil": [-2, 1, 0, 0, 0, 0, 0, -2], "tipos": ["Agua", "Planta"]},
            "C) Se lo comento al camarero pero de forma suave y con una sonrisa": {"perfil": [-1, 2, 0, 1, 0, 1, 0, 0], "tipos": ["Hada", "Normal"]},
            "D) No digo nada pero no vuelvo nunca más a ese sitio": {"perfil": [0, -1, 0, 1, 0, 0, 1, 2], "tipos": ["Hielo", "Siniestro"]}}},
        {"pregunta": "Vas a coger vacaciones. ¿Qué pesa más al elegir destino?", "opciones": {
            "A) Naturaleza salvaje, cuanto menos gente mejor": {"perfil": [0, -2, 1, 0, 2, 0, 0, 0], "tipos": ["Planta", "Tierra"]},
            "B) Ciudad con mucha vida, gente, ruido y planes": {"perfil": [0, 3, 2, 0, 0, 0, 0, 0], "tipos": ["Eléctrico", "Normal"]},
            "C) Un sitio con historia o cultura distinta (como un viaje a Cuba)": {"perfil": [0, 0, 1, 2, 1, 0, 1, 0], "tipos": ["Acero", "Volador"]},
            "D) Organizar un buen itinerario en una capital grande con mi grupo de amigos": {"perfil": [0, 2, 1, 2, 0, 1, -1, 0], "tipos": ["Normal", "Roca"]}}},
        {"pregunta": "Estás decorando o montando algo para tu casa/habitación.", "opciones": {
            "A) Sigo un plano exacto, medido con regla y todo": {"perfil": [0, 0, 0, 4, 0, 0, -2, 0], "tipos": ["Acero", "Roca"]},
            "B) Voy improvisando según lo que me gusta al verlo puesto": {"perfil": [0, 0, 1, -1, 1, 0, 2, 0], "tipos": ["Planta", "Hada"]},
            "C) Pido opinión a todo el mundo aunque luego haga lo mío": {"perfil": [0, 2, 0, 0, 0, 0, 0, 0], "tipos": ["Normal", "Agua"]},
            "D) Lo dejo a medias, ya lo terminaré otro día que me apetezca más": {"perfil": [0, 0, -2, 0, 0, -1, 1, 0], "tipos": ["Fantasma", "Bicho"]}}},
        {"pregunta": "Eliges tu nuevo móvil. ¿Qué especificación es innegociable?", "opciones": {
            "A) La cámara y sus funciones inteligentes, para fotos perfectas sin esfuerzo": {"perfil": [0, 1, 0, 2, 1, 0, 0, 0], "tipos": ["Psíquico", "Hada"]},
            "B) Batería extrema y carga rápida, vivo pegado a la pantalla": {"perfil": [0, 0, 3, 0, 0, 0, 1, 0], "tipos": ["Eléctrico", "Acero"]},
            "C) Capacidad para instalar emuladores raros y toquetear el sistema": {"perfil": [1, -1, 0, 3, 0, 0, 2, 1], "tipos": ["Veneno", "Bicho"]},
            "D) Que sea duro y resistente, se me cae cada dos por tres": {"perfil": [-1, 0, -1, 0, 0, 0, 0, 2], "tipos": ["Roca", "Tierra"]}}},
        {"pregunta": "Un olor muy fuerte y desagradable inunda de repente la habitación donde estás.", "opciones": {
            "A) Abro ventanas y me pongo a buscar el origen del problema": {"perfil": [0, 0, 1, 2, 0, 0, -1, 0], "tipos": ["Volador", "Acero"]},
            "B) Aguanto ahí como si no pasara nada especial": {"perfil": [0, 0, -1, 0, 0, 0, 0, 1], "tipos": ["Veneno", "Roca"]},
            "C) Me quejo alto para que alguien más se encargue": {"perfil": [1, 1, 0, 0, 0, 0, 0, 0], "tipos": ["Eléctrico", "Normal"]},
            "D) Me voy de la habitación sin dar más explicaciones": {"perfil": [0, -2, 1, 0, 1, 0, 0, 0], "tipos": ["Fantasma", "Hielo"]}}},
        {"pregunta": "Viendo código o interfaces de desarrollo (como CSS o un backend en Python)...", "opciones": {
            "A) Tiene que estar todo tabulado, limpio y estructurado visualmente": {"perfil": [0, -1, 0, 3, 0, 0, -3, 1], "tipos": ["Acero", "Hielo"]},
            "B) Me da igual si parece un monstruo de Frankenstein mientras compile y funcione": {"perfil": [1, 0, 1, 1, 0, 0, 4, 0], "tipos": ["Veneno", "Eléctrico"]},
            "C) Me gusta usar frameworks que me faciliten la vida rápido": {"perfil": [-1, 1, 0, 2, 0, 0, 1, -1], "tipos": ["Volador", "Agua"]},
            "D) Disfruto rompiéndome la cabeza buscando el error oculto durante horas": {"perfil": [0, -2, 0, 4, 1, 0, 0, 1], "tipos": ["Psíquico", "Bicho"]}}},
        {"pregunta": "A la hora de jugar a un videojuego clásico, ¿qué método prefieres?", "opciones": {
            "A) Hardware original, cartucho y consola de toda la vida": {"perfil": [0, 0, 0, 1, 1, 2, -1, 1], "tipos": ["Roca", "Fantasma"]},
            "B) Emulación cruzada, sincronizando partidas en la nube entre el móvil y el PC": {"perfil": [0, 0, 1, 3, 0, 0, 1, 1], "tipos": ["Acero", "Psíquico"]},
            "C) Remakes modernos con gráficos de última generación": {"perfil": [0, 1, 1, 0, 0, 0, 0, 0], "tipos": ["Hada", "Fuego"]},
            "D) Buscar mods y flashcards para exprimir al máximo lo que el juego puede hacer": {"perfil": [1, 0, 0, 2, 0, 0, 3, 0], "tipos": ["Bicho", "Siniestro"]}}}
    ],
    "Situaciones": [
        {"pregunta": "Llevas tres días seguidos durmiendo mal y hoy tienes reunión importante. ¿Cómo llegas?", "opciones": {
            "A) Como si nada, funciono igual de bien reventado que descansado": {"perfil": [1, 0, 1, 0, 0, 0, 0, 3], "tipos": ["Roca", "Acero"]},
            "B) Con un café doble y quejándome a todo el que se cruce": {"perfil": [2, 1, 2, 0, 0, 0, 1, 0], "tipos": ["Eléctrico", "Fuego"]},
            "C) Voy pero aviso desde ya que hoy rindo la mitad": {"perfil": [0, 2, -2, 0, 0, 2, 0, -1], "tipos": ["Normal", "Agua"]},
            "D) Cancelo lo que pueda esperar, mi cuerpo manda hoy": {"perfil": [-1, -1, -3, 1, 0, -1, 1, 1], "tipos": ["Planta", "Hielo"]}}},
        {"pregunta": "Un compañero de piso lleva semanas sin fregar sus platos. Un día explota la cosa...", "opciones": {
            "A) Los meto en una bolsa y se los dejo delante de su puerta": {"perfil": [2, -1, 0, 1, 0, -1, 2, 1], "tipos": ["Veneno", "Siniestro"]},
            "B) Hablo con él tranquilo, seguro que tiene un mal momento": {"perfil": [-2, 3, 0, 0, 0, 3, 0, -1], "tipos": ["Hada", "Agua"]},
            "C) Hago una tabla de turnos y la pego en la nevera": {"perfil": [0, 0, 0, 4, 0, 1, -2, 0], "tipos": ["Acero", "Psíquico"]},
            "D) Los friego yo, total ya estoy de mal humor": {"perfil": [-3, 1, -1, 0, 0, 2, 0, -2], "tipos": ["Planta", "Normal"]}}},
        {"pregunta": "En el súper hay una cola larguísima y solo una caja abierta.", "opciones": {
            "A) Me pongo a hablar con el de delante para matar el rato": {"perfil": [0, 3, 0, 0, 0, 0, 0, -1], "tipos": ["Normal", "Volador"]},
            "B) Calculo mentalmente si me sale a cuenta ir a otra tienda": {"perfil": [0, -1, 0, 3, 0, 0, 0, 0], "tipos": ["Acero", "Tierra"]},
            "C) Suspiro fuerte esperando que alguien abra otra caja por presión social": {"perfil": [1, 0, 1, 0, 0, 0, 1, 1], "tipos": ["Fuego", "Eléctrico"]},
            "D) Dejo el carro y me voy, ya compraré online": {"perfil": [0, -2, -1, 1, 0, -1, 0, 2], "tipos": ["Hielo", "Dragón"]}}},
        {"pregunta": "Te avisan quince minutos antes de que va a venir gente a tu casa y está hecha un desastre.", "opciones": {
            "A) Limpio a toda velocidad como si me fuera la vida en ello": {"perfil": [0, 0, 3, 0, 0, 1, 0, 1], "tipos": ["Eléctrico", "Lucha"]},
            "B) Escondo el desorden en cualquier armario y listo": {"perfil": [0, 0, 1, 0, 0, -1, 3, 0], "tipos": ["Veneno", "Normal"]},
            "C) Aviso de que va a estar así, que no se asusten": {"perfil": [-1, 2, 0, 0, 0, 0, 0, -2], "tipos": ["Hada", "Agua"]},
            "D) Ni me inmuto, el que venga que venga como está": {"perfil": [0, -1, -1, 0, 0, 0, 0, 3], "tipos": ["Roca", "Dragón"]}}},
        {"pregunta": "Vas conduciendo o en bici y un coche te corta el paso de mala manera.", "opciones": {
            "A) Toco el pito y suelto un par de cosas por la ventana": {"perfil": [3, 0, 1, 0, 0, 0, 0, 0], "tipos": ["Fuego", "Eléctrico"]},
            "B) Freno, respiro y sigo, no vale la pena el disgusto": {"perfil": [-2, 0, -1, 1, 0, 0, -1, 0], "tipos": ["Agua", "Acero"]},
            "C) Me acuerdo de la matrícula por si acaso algún día coincide otra vez": {"perfil": [0, 0, 0, 2, 0, 0, 1, 2], "tipos": ["Dragón", "Siniestro"]},
            "D) Me da un vuelco el corazón y voy con miedo el resto del camino": {"perfil": [0, 0, 0, 0, 1, 0, 0, -2], "tipos": ["Hielo", "Agua"]}}},
        {"pregunta": "Notas que una araña grande se pasea por tu habitación de noche.", "opciones": {
            "A) La observo un rato con curiosidad antes de decidir qué hacer": {"perfil": [0, 0, 0, 2, 2, 0, 0, 0], "tipos": ["Bicho", "Psíquico"]},
            "B) Grito y salgo pitando de la habitación": {"perfil": [0, 1, 3, -1, 0, 0, 1, -1], "tipos": ["Volador", "Normal"]},
            "C) Cojo un vaso y la saco fuera con cuidado de no hacerle daño": {"perfil": [-2, 1, 0, 0, 0, 2, 0, 0], "tipos": ["Hada", "Planta"]},
            "D) La aplasto sin pensarlo dos veces, no hay negociación posible": {"perfil": [3, -1, 0, 0, 0, 0, 0, 1], "tipos": ["Lucha", "Roca"]}}},
        {"pregunta": "Un amigo viene a tu ciudad para disfrutar de un festival importante (como las Fallas).", "opciones": {
            "A) Le monto un itinerario milimétrico para que vea todo lo importante": {"perfil": [0, 2, 1, 3, 0, 2, -1, 1], "tipos": ["Acero", "Eléctrico"]},
            "B) Improvisamos sobre la marcha según la zona donde estemos": {"perfil": [0, 1, 1, 0, 0, 0, 2, 0], "tipos": ["Fuego", "Volador"]},
            "C) Le llevo a mis rincones favoritos huyendo un poco de las masas": {"perfil": [0, 1, -1, 0, 1, 2, 0, 0], "tipos": ["Fantasma", "Planta"]},
            "D) Le digo que se apunte a lo que yo ya iba a hacer con mi grupo": {"perfil": [1, 2, 0, 0, 0, 1, 0, 1], "tipos": ["Lucha", "Normal"]}}},
        {"pregunta": "Te escriben a las 23:00 pidiéndote un favor urgente que no te apetece nada hacer.", "opciones": {
            "A) Digo que sí aunque por dentro esté maldiciendo un poco": {"perfil": [0, 2, 0, 0, 0, 4, 0, -1], "tipos": ["Normal", "Hada"]},
            "B) Pregunto primero qué gano yo antes de mover un dedo": {"perfil": [1, -1, 0, 2, 0, -2, 2, 1], "tipos": ["Siniestro", "Veneno"]},
            "C) Contesto al día siguiente como si no hubiera visto nada": {"perfil": [0, -2, -1, 1, 0, -2, 1, 1], "tipos": ["Fantasma", "Hielo"]},
            "D) Le doy la solución rápida sin ni siquiera moverme del sitio": {"perfil": [0, 0, 0, 3, 1, 1, 0, 0], "tipos": ["Psíquico", "Acero"]}}},
        {"pregunta": "En una comida familiar sale un tema políticamente delicado.", "opciones": {
            "A) Doy mi opinión aunque sepa que va a generar roce": {"perfil": [1, 0, 0, 0, 0, 0, 0, 2], "tipos": ["Dragón", "Lucha"]},
            "B) Cambio de tema con disimulo hacia algo más ligero": {"perfil": [-1, 1, 0, 1, 0, 0, 0, 0], "tipos": ["Normal", "Hada"]},
            "C) Escucho todos los puntos de vista sin posicionarme aún": {"perfil": [0, 0, 0, 3, 0, 0, 0, 0], "tipos": ["Psíquico", "Agua"]},
            "D) Me quedo callado pero por dentro ya estoy con la sangre hirviendo": {"perfil": [1, -1, 0, 0, 0, 0, 0, 0], "tipos": ["Fuego", "Siniestro"]}}},
        {"pregunta": "Te pisan sin querer en el metro y ni se disculpan.", "opciones": {
            "A) Lo dejo pasar, seguro ni se ha dado cuenta": {"perfil": [-2, 0, 0, 0, 0, 0, 0, -1], "tipos": ["Agua", "Normal"]},
            "B) Carraspeo fuerte a ver si capta la indirecta": {"perfil": [1, 0, 0, 0, 0, 0, 1, 1], "tipos": ["Eléctrico", "Roca"]},
            "C) Se lo digo directamente pero sin mala leche": {"perfil": [1, 1, 0, 1, 0, 0, -1, 0], "tipos": ["Lucha", "Normal"]},
            "D) Me quedo rumiándolo el resto del trayecto en silencio": {"perfil": [0, -2, -1, 1, 0, 0, 0, 2], "tipos": ["Fantasma", "Hielo"]}}},
        {"pregunta": "Un desconocido te pide dinero por la calle.", "opciones": {
            "A) Le doy lo que llevo suelto sin pensarlo mucho": {"perfil": [-1, 2, 0, 0, 0, 1, 0, -1], "tipos": ["Hada", "Agua"]},
            "B) Le pregunto para qué lo necesita antes de decidir": {"perfil": [0, 1, 0, 2, 0, 0, 0, 0], "tipos": ["Psíquico", "Normal"]},
            "C) Sigo andando, prefiero no involucrarme en la calle": {"perfil": [0, -2, 0, 1, 0, -1, 0, 1], "tipos": ["Acero", "Hielo"]},
            "D) Le ofrezco comprarle algo de comer en vez de dinero": {"perfil": [0, 1, 0, 1, 0, 2, -1, 0], "tipos": ["Planta", "Tierra"]}}}
    ],
    "Psicologicas": [
        {"pregunta": "Alguien te corrige en público sobre algo que dijiste mal.", "opciones": {
            "A) Le doy la razón al instante, prefiero eso a discutir por tonterías": {"perfil": [-2, 1, 0, 0, 0, 1, 0, -2], "tipos": ["Agua", "Normal"]},
            "B) Busco el matiz por el que en realidad no estaba tan equivocado": {"perfil": [0, 0, 0, 3, 0, 0, 1, 2], "tipos": ["Psíquico", "Roca"]},
            "C) Me sienta fatal aunque no lo demuestre en la cara": {"perfil": [0, -1, -1, 0, 1, 0, 0, 3], "tipos": ["Fantasma", "Hielo"]},
            "D) Me río y lo uso de chiste antes de que lo haga otro": {"perfil": [0, 3, 2, 0, 0, 0, 2, -1], "tipos": ["Eléctrico", "Normal"]}}},
        {"pregunta": "Tienes que elegir entre decir una mentira piadosa o una verdad que duele.", "opciones": {
            "A) Verdad siempre, aunque sea incómoda para todos": {"perfil": [1, -1, 0, 1, 0, 1, 0, 2], "tipos": ["Roca", "Lucha"]},
            "B) Mentira si eso evita hacer daño a quien quiero": {"perfil": [-2, 2, 0, 0, 0, 3, 0, -1], "tipos": ["Hada", "Agua"]},
            "C) Busco una tercera opción que no sea ni una cosa ni otra": {"perfil": [0, 0, 0, 3, 0, 0, 1, 0], "tipos": ["Psíquico", "Acero"]},
            "D) Depende de si me conviene más a mí, siendo sincero": {"perfil": [0, -1, 0, 1, 0, -2, 2, 0], "tipos": ["Siniestro", "Veneno"]}}},
        {"pregunta": "Sientes que llevas un tiempo estancado, sin avanzar en nada importante.", "opciones": {
            "A) Lo cambio todo de golpe, necesito una sacudida fuerte": {"perfil": [1, 0, 3, 0, 0, -1, 2, 0], "tipos": ["Eléctrico", "Dragón"]},
            "B) Me paro a analizar en qué punto exacto me quedé atascado": {"perfil": [0, -1, 0, 4, 0, 0, -1, 0], "tipos": ["Acero", "Psíquico"]},
            "C) Espero, confío en que estas cosas pasan solas con tiempo": {"perfil": [-2, 0, -2, 0, 1, 0, 0, 0], "tipos": ["Agua", "Planta"]},
            "D) Me machaco bastante por dentro, aunque por fuera no se note": {"perfil": [0, -1, 0, 0, 1, 0, 0, -3], "tipos": ["Fantasma", "Hielo"]}}},
        {"pregunta": "¿Qué te cuesta más perdonar de verdad, aunque digas que ya lo has superado?", "opciones": {
            "A) Que me mientan directamente a la cara": {"perfil": [1, 0, 0, 0, 0, 2, -1, 1], "tipos": ["Lucha", "Acero"]},
            "B) Que me dejen de lado sin motivo aparente": {"perfil": [0, 1, 0, 0, 0, 1, 0, -2], "tipos": ["Hada", "Fantasma"]},
            "C) Que no valoren el esfuerzo que he puesto en algo": {"perfil": [0, 0, 0, 1, 0, 0, 0, 3], "tipos": ["Roca", "Dragón"]},
            "D) Que hablen sin saber y aun así opinen como si supieran": {"perfil": [0, -1, 0, 3, 0, 0, 0, 1], "tipos": ["Psíquico", "Acero"]}}},
        {"pregunta": "Cuando alguien llora delante de ti, ¿cuál es tu primer impulso?", "opciones": {
            "A) Abrazar o tocar, necesito hacer algo físico ya": {"perfil": [-1, 3, 1, 0, 0, 2, 0, 0], "tipos": ["Hada", "Agua"]},
            "B) Buscar una solución práctica al problema que lo causó": {"perfil": [0, 0, 0, 3, 0, 1, 0, 0], "tipos": ["Acero", "Tierra"]},
            "C) Quedarme un poco paralizado sin saber muy bien qué hacer": {"perfil": [0, -1, -1, 0, 1, 0, 0, 0], "tipos": ["Hielo", "Normal"]},
            "D) Sentir la emoción como si fuera un poco mía también": {"perfil": [0, 1, 0, 0, 3, 1, 0, -1], "tipos": ["Psíquico", "Fantasma"]}}},
        {"pregunta": "Alguien te copia claramente una idea o forma de hacer algo tuyo.", "opciones": {
            "A) Se lo digo directo, sin darle más vueltas al asunto": {"perfil": [1, 0, 0, 1, 0, 0, 0, 1], "tipos": ["Lucha", "Roca"]},
            "B) Me halaga un poco en el fondo, imitar es un cumplido": {"perfil": [-1, 1, 0, 0, 0, 0, 0, -1], "tipos": ["Agua", "Hada"]},
            "C) Me guardo el enfado y sigo innovando para ir un paso por delante": {"perfil": [0, -1, 1, 2, 0, 0, 0, 2], "tipos": ["Acero", "Dragón"]},
            "D) Empiezo a vigilar de cerca a esa persona, con desconfianza": {"perfil": [1, -1, 0, 1, 0, -1, 2, 0], "tipos": ["Siniestro", "Veneno"]}}},
        {"pregunta": "Notas que estás empezando a caerle mal a alguien sin motivo claro.", "opciones": {
            "A) Se lo pregunto directamente, prefiero saberlo cuanto antes": {"perfil": [1, 1, 0, 1, 0, 0, 0, 0], "tipos": ["Lucha", "Normal"]},
            "B) Le doy vueltas mentalmente sin decir nada durante días": {"perfil": [0, -1, 0, 2, 1, 0, 0, 1], "tipos": ["Psíquico", "Fantasma"]},
            "C) Sigo a lo mío, no todo el mundo tiene que quererme": {"perfil": [0, -2, 0, 0, 0, 0, -1, 2], "tipos": ["Roca", "Dragón"]},
            "D) Intento agradarle un poco más de lo normal, me incomoda esa idea": {"perfil": [-2, 2, 0, 0, 0, 1, 0, -2], "tipos": ["Hada", "Agua"]}}},
        {"pregunta": "Metes la pata en algo delante de gente que te importa impresionar.", "opciones": {
            "A) Lo reconozco en el momento y me río de mí mismo": {"perfil": [0, 2, 1, 0, 0, 0, 1, -2], "tipos": ["Normal", "Eléctrico"]},
            "B) Improviso una excusa creíble sobre la marcha": {"perfil": [0, 1, 0, 2, 0, -1, 2, 0], "tipos": ["Siniestro", "Volador"]},
            "C) Se me queda grabado días, aunque nadie más se acuerde": {"perfil": [0, -1, -1, 0, 1, 0, 0, 2], "tipos": ["Hielo", "Fantasma"]},
            "D) Sigo como si nada hubiera pasado, sin dar explicaciones": {"perfil": [0, -1, 0, 0, 0, 0, -1, 3], "tipos": ["Roca", "Dragón"]}}},
        {"pregunta": "En un grupo, dos personas que te caen bien empiezan a discutir fuerte.", "opciones": {
            "A) Me meto en medio a poner paz aunque no me lo pidan": {"perfil": [-1, 3, 1, 0, 0, 2, -1, 0], "tipos": ["Hada", "Normal"]},
            "B) Me quedo a un lado observando cómo se desarrolla todo": {"perfil": [0, -1, 0, 2, 1, 0, 0, 0], "tipos": ["Fantasma", "Psíquico"]},
            "C) Suelto una broma para bajar la tensión de golpe": {"perfil": [0, 2, 1, 0, 0, 0, 1, -1], "tipos": ["Eléctrico", "Normal"]},
            "D) Me quedo con quien tenga razón, sin más": {"perfil": [1, -1, 0, 2, 0, -1, 0, 1], "tipos": ["Acero", "Roca"]}}},
        {"pregunta": "Te piden opinión sobre algo que la otra persona ha hecho con mucho esfuerzo pero no te gusta.", "opciones": {
            "A) Destaco lo bueno y suavizo lo que no me convence": {"perfil": [-1, 2, 0, 1, 0, 1, 0, -1], "tipos": ["Hada", "Planta"]},
            "B) Digo exactamente lo que pienso, para eso me pregunta": {"perfil": [1, 0, 0, 1, 0, 0, 0, 2], "tipos": ["Roca", "Dragón"]},
            "C) Le pregunto qué opina él primero para no comprometerme": {"perfil": [0, 1, 0, 2, 0, -1, 1, 0], "tipos": ["Psíquico", "Siniestro"]},
            "D) Cambio de tema con disimulo antes de que me insista": {"perfil": [0, 0, 0, 0, 0, 0, 3, -1], "tipos": ["Fantasma", "Veneno"]}}},
        {"pregunta": "Si tu forma de ser fuera un cartel pegado en la puerta de tu cuarto, ¿cuál sería?", "opciones": {
            "A) 'Toca antes de entrar, luego lo que quieras'": {"perfil": [1, 0, 0, 0, 0, 1, 0, 1], "tipos": ["Roca", "Normal"]},
            "B) 'Puertas abiertas, aquí cabe quien quiera venir'": {"perfil": [-2, 4, 0, 0, 0, 2, 0, -1], "tipos": ["Hada", "Normal"]},
            "C) 'Entra bajo tu propio riesgo'": {"perfil": [1, -1, 0, 0, 2, 0, 2, 0], "tipos": ["Siniestro", "Fantasma"]},
            "D) 'Todo tiene su sitio, no lo desordenes'": {"perfil": [0, -1, 0, 4, 0, 0, -2, 1], "tipos": ["Acero", "Psíquico"]}}}
    ],
    "Aspiraciones": [
        {"pregunta": "Te regalan una tarde libre sin planes ni obligaciones. ¿Qué haces de verdad?", "opciones": {
            "A) Me tiro en el sofá a hacer scroll sin sentido durante horas": {"perfil": [0, 0, -3, -1, 0, 0, 1, 0], "tipos": ["Normal", "Agua"]},
            "B) Aprovecho para adelantar algo pendiente o programar mis proyectos web": {"perfil": [1, 0, 2, 2, 0, 0, -1, 1], "tipos": ["Acero", "Eléctrico"]},
            "C) Salgo a caminar sin rumbo y ver a dónde llego": {"perfil": [0, 0, 1, 0, 3, 0, 1, 0], "tipos": ["Volador", "Fantasma"]},
            "D) Llamo a alguien para que me acompañe, solo se me hace raro": {"perfil": [0, 4, 1, 0, 0, 1, 0, 0], "tipos": ["Hada", "Normal"]}}},
        {"pregunta": "Te piden que elijas entre trabajar solo con total libertad o en equipo con más apoyo.", "opciones": {
            "A) Solo, sin duda, decido yo cómo y cuándo": {"perfil": [1, -2, 0, 1, 0, -1, 0, 2], "tipos": ["Roca", "Acero"]},
            "B) En equipo, se rinde más y se pasa mejor entre varios": {"perfil": [-1, 3, 0, 0, 0, 2, 0, -1], "tipos": ["Hada", "Normal"]},
            "C) Depende totalmente de la tarea, no tengo una postura fija": {"perfil": [0, 0, 0, 2, 0, 0, 0, 0], "tipos": ["Psíquico", "Normal"]},
            "D) Solo, pero en secreto me gustaría que alguien notara el resultado": {"perfil": [0, -1, 0, 1, 0, 0, 0, 3], "tipos": ["Dragón", "Siniestro"]}}},
        {"pregunta": "En el trabajo o clase, ¿cómo prefieres que te reconozcan un buen resultado?", "opciones": {
            "A) En privado, no necesito que se entere todo el mundo": {"perfil": [-1, -1, 0, 0, 0, 0, 0, 1], "tipos": ["Agua", "Acero"]},
            "B) Delante de todos, que se sepa el curro que ha costado": {"perfil": [1, 2, 1, 0, 0, 0, 0, 3], "tipos": ["Fuego", "Dragón"]},
            "C) Con algo concreto, un aumento o un beneficio real": {"perfil": [0, -1, 0, 2, 0, 0, 1, 0], "tipos": ["Siniestro", "Acero"]},
            "D) La verdad, ni lo necesito, sé lo que he hecho yo mismo": {"perfil": [0, -2, 0, 1, 0, 0, -1, 2], "tipos": ["Hielo", "Roca"]}}},
        {"pregunta": "Un plan que llevabas semanas montando (como un evento o proyecto) se cae en el último momento.", "opciones": {
            "A) Monto uno nuevo en el momento, no me quedo parado": {"perfil": [1, 0, 2, 1, 0, 0, 1, 0], "tipos": ["Eléctrico", "Volador"]},
            "B) Me frustro bastante, me cuesta soltar lo que ya tenía planeado": {"perfil": [1, 0, 0, 0, 0, 0, 0, 1], "tipos": ["Roca", "Tierra"]},
            "C) Lo dejo estar y aprovecho el hueco para no hacer nada": {"perfil": [-2, 0, -2, 0, 0, 0, 0, 0], "tipos": ["Agua", "Normal"]},
            "D) Repaso mentalmente qué falló para que no vuelva a pasar en el futuro": {"perfil": [0, -1, 0, 3, 0, 0, -1, 0], "tipos": ["Acero", "Psíquico"]}}},
        {"pregunta": "Si tuvieras que desaparecer literalmente un día entero sin dar explicaciones, ¿qué harías?", "opciones": {
            "A) Algo físico y fuerte, correr, quemar energía": {"perfil": [2, -1, 4, 0, 0, 0, 0, 0], "tipos": ["Lucha", "Volador"]},
            "B) Meterme en un sitio nuevo que no conozca nadie": {"perfil": [0, 0, 1, 0, 2, -1, 1, 0], "tipos": ["Fantasma", "Volador"]},
            "C) Quedarme en algún sitio tranquilo leyendo o estructurando mis ideas": {"perfil": [-2, -1, -2, 2, 1, 0, 0, 0], "tipos": ["Psíquico", "Agua"]},
            "D) Buscar a alguien de mi pasado con quien perdí el contacto": {"perfil": [0, 1, 0, 0, 0, 3, 0, 0], "tipos": ["Hada", "Normal"]}}},
        {"pregunta": "Tienes que hablar en público o presentar algo sobre lo que dominas poco.", "opciones": {
            "A) Improviso con seguridad, el tono lo es casi todo": {"perfil": [1, 2, 1, 0, 0, 0, 1, 1], "tipos": ["Fuego", "Volador"]},
            "B) Preparo cada palabra por escrito para no dejar nada al azar": {"perfil": [0, -1, 0, 4, 0, 0, -2, 0], "tipos": ["Acero", "Psíquico"]},
            "C) Empiezo reconociendo que no soy experto, y de ahí para arriba": {"perfil": [-1, 1, 0, 1, 0, 1, 0, -2], "tipos": ["Agua", "Normal"]},
            "D) Lo evito como sea, delego en quien pueda hacerlo si existe opción": {"perfil": [0, -2, -1, 0, 0, 0, 0, 0], "tipos": ["Bicho", "Hielo"]}}},
        {"pregunta": "Decides empezar a estudiar algo por tu cuenta (por ejemplo, Machine Learning o un idioma nuevo).", "opciones": {
            "A) Voy directo a hacer proyectos, me salto la teoría inicial": {"perfil": [1, 0, 2, 0, 0, 0, 2, 1], "tipos": ["Fuego", "Eléctrico"]},
            "B) Empiezo por los fundamentos teóricos pesados hasta entender la base matemática": {"perfil": [0, 0, 0, 4, 0, 0, -2, 0], "tipos": ["Acero", "Psíquico"]},
            "C) Busco tutoriales en vídeo y voy copiando hasta que me sale": {"perfil": [0, 1, 0, 1, 0, 1, 0, -1], "tipos": ["Normal", "Agua"]},
            "D) Lo empiezo con muchas ganas pero a la semana se me olvida que existía": {"perfil": [0, 0, 1, 0, 0, -1, 3, 0], "tipos": ["Fantasma", "Volador"]}}},
        {"pregunta": "Descubres una serie o videojuego que te engancha muchísimo.", "opciones": {
            "A) Lo devoro en un fin de semana entero sin salir de casa": {"perfil": [0, -1, 1, 0, 0, 0, 1, 0], "tipos": ["Bicho", "Fantasma"]},
            "B) Lo voy dosificando para que dure y no se acabe pronto": {"perfil": [-1, 0, -1, 2, 0, 1, -1, 1], "tipos": ["Hielo", "Planta"]},
            "C) Necesito comentarlo con alguien mientras lo veo, si no pierde la gracia": {"perfil": [0, 3, 0, 0, 0, 0, 0, 0], "tipos": ["Normal", "Volador"]},
            "D) Me pongo a leer teorías y análisis en vez de seguir viéndolo": {"perfil": [0, -1, 0, 4, 1, 0, 0, 0], "tipos": ["Psíquico", "Acero"]}}},
        {"pregunta": "Estás organizando un evento local y toca gestionar la venta de entradas.", "opciones": {
            "A) Hago una tabla estricta de precios, tiers y comisiones para cuadrar todo": {"perfil": [0, 0, 0, 3, 0, 1, -1, 0], "tipos": ["Acero", "Tierra"]},
            "B) Me apoyo en plataformas automáticas, no quiero dolores de cabeza": {"perfil": [0, 0, 0, 2, 0, 0, 1, -1], "tipos": ["Volador", "Agua"]},
            "C) Lo uso como excusa para hablar con toda la gente que conozco y hacer ruido": {"perfil": [0, 3, 2, 0, 0, 1, 0, 1], "tipos": ["Hada", "Fuego"]},
            "D) Suelto las entradas a la vez y que se peleen por ellas, cero estrés": {"perfil": [1, -1, 0, 0, 0, 0, 2, 0], "tipos": ["Siniestro", "Dragón"]}}},
        {"pregunta": "Ganas un premio pequeño en un sorteo random.", "opciones": {
            "A) Me pongo eufórico un rato, aunque sea una tontería sin valor": {"perfil": [0, 1, 3, 0, 0, 0, 1, 0], "tipos": ["Eléctrico", "Normal"]},
            "B) Pienso enseguida a quién se lo puedo regalar": {"perfil": [-2, 2, 0, 0, 0, 2, 0, -1], "tipos": ["Hada", "Planta"]},
            "C) Sospecho un poco, nada es gratis del todo": {"perfil": [0, -1, 0, 2, 0, 0, 1, 0], "tipos": ["Siniestro", "Veneno"]},
            "D) Ni me inmuto demasiado, ya se me pasa el efecto rápido": {"perfil": [0, -1, -1, 0, 0, 0, 0, 1], "tipos": ["Roca", "Acero"]}}}
    ],
    "Fantasia_Rol": [
        {"pregunta": "En una partida de rol, el máster (Dungeon Master) te ofrece un objeto mágico misterioso.", "opciones": {
            "A) Lo uso ya mismo, ya se verá qué pasa": {"perfil": [1, 0, 2, -1, 1, 0, 2, 0], "tipos": ["Dragón", "Eléctrico"]},
            "B) Investigo primero preguntando o buscando pistas dentro del juego": {"perfil": [0, 0, 0, 3, 1, 0, -1, 0], "tipos": ["Psíquico", "Bicho"]},
            "C) Se lo regalo a otro jugador, que se arriesgue él": {"perfil": [-1, 1, 0, 0, 0, -1, 1, 0], "tipos": ["Normal", "Veneno"]},
            "D) Lo guardo sin usarlo, prefiero tenerlo por si acaso": {"perfil": [0, -1, 0, 1, 0, 1, -1, 1], "tipos": ["Acero", "Roca"]}}},
        {"pregunta": "Diriges una partida de D&D y los jugadores ignoran tu trama principal para ir a un bar.", "opciones": {
            "A) Los castigo con una emboscada para que vuelvan al redil": {"perfil": [2, -1, 0, 0, 0, 0, 0, 2], "tipos": ["Dragón", "Roca"]},
            "B) Improviso mecánicas de juegos de taberna y me río con ellos": {"perfil": [0, 3, 1, 0, 0, 0, 2, -1], "tipos": ["Hada", "Normal"]},
            "C) Uso a un NPC en el bar para reconducirlos sutilmente a la misión": {"perfil": [0, 1, 0, 3, 0, 0, -1, 0], "tipos": ["Psíquico", "Siniestro"]},
            "D) Me enfado un poco internamente porque había preparado 10 páginas de lore": {"perfil": [0, -2, -1, 2, 0, -1, 0, 1], "tipos": ["Acero", "Fantasma"]}}},
        {"pregunta": "Si pudieras tener un poder pero con una desventaja fea incluida, ¿cuál eliges?", "opciones": {
            "A) Fuerza sobrehumana, aunque se me agote la energía enseguida": {"perfil": [3, 0, 1, -1, 0, 0, 0, 1], "tipos": ["Lucha", "Tierra"]},
            "B) Leer mentes, aunque no pueda desconectarlo nunca": {"perfil": [0, 0, 0, 3, 1, 0, 0, 0], "tipos": ["Psíquico", "Fantasma"]},
            "C) Invisibilidad, aunque nadie vuelva a acordarse de mí": {"perfil": [0, -3, 0, 0, 2, 0, 1, -1], "tipos": ["Fantasma", "Siniestro"]},
            "D) Curar a otros, aunque a mí me cueste el doble sanar": {"perfil": [-2, 2, 0, 0, 0, 3, 0, 0], "tipos": ["Hada", "Planta"]}}},
        {"pregunta": "Construyendo tu base o fortaleza ideal en un videojuego tipo Terraria o Minecraft.", "opciones": {
            "A) La lleno de trampas, foso de lava y defensas por todos lados": {"perfil": [2, -1, 0, 1, 0, 0, 1, 0], "tipos": ["Veneno", "Roca"]},
            "B) La hago acogedora, con sitio para que quepan mis amigos y los NPCs": {"perfil": [-1, 3, 0, 0, 0, 1, 0, 0], "tipos": ["Hada", "Normal"]},
            "C) La escondo bien camuflada bajo tierra, que nadie sepa que existe": {"perfil": [0, -2, 0, 2, 1, 0, 0, 0], "tipos": ["Fantasma", "Bicho"]},
            "D) La hago enorme y vistosa, una torre inmensa que se vea desde lejos": {"perfil": [0, 0, 1, 0, 0, 0, 0, 4], "tipos": ["Dragón", "Roca"]}}},
        {"pregunta": "Vas a nombrar a tu personaje en un CRPG denso como Baldur's Gate 3.", "opciones": {
            "A) Algo gracioso o absurdo que me haga reír en las cinemáticas serias": {"perfil": [0, 2, 1, 0, 0, 0, 2, 0], "tipos": ["Normal", "Eléctrico"]},
            "B) Un nombre serio y con peso en el lore, que suene a alguien importante": {"perfil": [1, 0, 0, 1, 0, 0, -1, 2], "tipos": ["Dragón", "Roca"]},
            "C) El nombre que uso en todas partes, sin complicarme": {"perfil": [-1, 0, 0, 0, 0, 1, 0, -1], "tipos": ["Agua", "Acero"]},
            "D) Algo oscuro sacado de mitología antigua o de un idioma muerto": {"perfil": [0, -1, 0, 1, 3, 0, 0, 1], "tipos": ["Fantasma", "Siniestro"]}}},
        {"pregunta": "Encuentras una puerta cerrada con un cartel de NO PASAR en un sitio abandonado.", "opciones": {
            "A) La abro, la curiosidad me puede siempre": {"perfil": [1, 0, 1, 0, 2, 0, 1, 0], "tipos": ["Fantasma", "Siniestro"]},
            "B) Paso de largo, un cartel es un cartel": {"perfil": [-1, 0, 0, 1, 0, 1, -1, 0], "tipos": ["Acero", "Normal"]},
            "C) Busco otra forma de entrar, tipo ventana, por si acaso": {"perfil": [1, -1, 0, 2, 0, 0, 1, 0], "tipos": ["Siniestro", "Bicho"]},
            "D) Se lo cuento a alguien más para ir juntos a mirar": {"perfil": [0, 2, 1, 0, 1, 0, 0, 0], "tipos": ["Volador", "Normal"]}}},
        {"pregunta": "Estás a solas en casa y escuchas un ruido raro que no sabes de dónde viene.", "opciones": {
            "A) Voy directo a investigar, prefiero saber ya qué es": {"perfil": [2, 0, 1, 1, 0, 0, 0, 1], "tipos": ["Lucha", "Roca"]},
            "B) Me quedo quieto escuchando con atención antes de moverme": {"perfil": [0, -1, -1, 2, 1, 0, 0, 0], "tipos": ["Psíquico", "Hielo"]},
            "C) Pongo música o la tele más alta para no pensar en ello": {"perfil": [-1, 0, 0, 0, 0, 0, 1, 0], "tipos": ["Normal", "Eléctrico"]},
            "D) Le escribo a alguien contándole lo que está pasando, por si acaso": {"perfil": [0, 2, 0, 0, 1, 1, 0, -1], "tipos": ["Fantasma", "Hada"]}}},
        {"pregunta": "Si mañana te despertaras con una habilidad nueva pero al azar, ¿qué prefieres?", "opciones": {
            "A) Una fuerza descomunal, aunque no supiera controlarla del todo": {"perfil": [4, 0, 1, -1, 0, 0, 1, 0], "tipos": ["Lucha", "Dragón"]},
            "B) Saber exactamente qué siente la gente a mi alrededor": {"perfil": [-1, 3, 0, 0, 1, 1, 0, 0], "tipos": ["Hada", "Psíquico"]},
            "C) Poder ver patrones y algoritmos que nadie más ve": {"perfil": [0, -1, 0, 4, 1, 0, 0, 0], "tipos": ["Psíquico", "Acero"]},
            "D) Volverme completamente invisible a voluntad cuando quisiera": {"perfil": [0, -2, 0, 0, 3, 0, 2, -1], "tipos": ["Fantasma", "Siniestro"]}}},
        {"pregunta": "Al diseñar un mundo de fantasía (por ejemplo, piratas malditos o vampiros)...", "opciones": {
            "A) Escribo páginas de lore detallado, cronologías y facciones políticas": {"perfil": [0, -1, 0, 4, 1, 0, -1, 1], "tipos": ["Acero", "Psíquico"]},
            "B) Me centro en que el mundo sea brutal y peligroso para los habitantes": {"perfil": [2, 0, 0, 1, 2, 0, 1, 1], "tipos": ["Siniestro", "Veneno"]},
            "C) Prefiero diseñar los personajes clave y dejar el mundo en un segundo plano": {"perfil": [-1, 3, 0, 0, 0, 1, 0, 0], "tipos": ["Normal", "Hada"]},
            "D) Dejo zonas en blanco para inventarlas sobre la marcha según juegue": {"perfil": [0, 1, 1, 0, 0, 0, 4, 0], "tipos": ["Volador", "Caos"]}}},
        {"pregunta": "En un videojuego cooperativo, tu personaje muere por un error tuyo estúpido.", "opciones": {
            "A) Me río el primero, la cago fuerte pero se puede reír": {"perfil": [0, 2, 1, 0, 0, 0, 1, -2], "tipos": ["Normal", "Eléctrico"]},
            "B) Le echo la culpa al lag antes de asumir nada": {"perfil": [1, 0, 0, 0, 0, -1, 2, 1], "tipos": ["Veneno", "Siniestro"]},
            "C) Me quedo callado un rato dándole vueltas a cómo evitarlo la próxima": {"perfil": [0, -1, 0, 3, 0, 0, -1, 1], "tipos": ["Acero", "Psíquico"]},
            "D) Me frustro en serio, odio quedar mal delante de otros": {"perfil": [2, -1, 1, 0, 0, 0, 0, 3], "tipos": ["Fuego", "Dragón"]}}}
    ]
}

# --- 4. MOTOR DE GENERACIÓN ALEATORIA CON SESSION STATE ---
# Esto garantiza que al recargar la web o darle a un botón, las preguntas no bailen solas
if "test_actual" not in st.session_state:
    test_seleccionado = []
    
    # Bloque 1 (Obligatorio 10 preguntas): Gustos
    preguntas_gustos = random.sample(POOLS_PREGUNTAS["Gustos"], 10)
    for p in preguntas_gustos:
        p["peso"] = 2.0  # El multiplicador que querías, valen el doble.
        p["tema"] = "Gustos"
    test_seleccionado.extend(preguntas_gustos)
    
    # Bloques 2 al 5: Situaciones, Psicológicas, Aspiraciones, Fantasía
    # Cogemos 10 de cada uno de los 4 bloques restantes (10 + 40 = 50 preguntas en total)
    tematicas_secundarias = ["Situaciones", "Psicologicas", "Aspiraciones", "Fantasia_Rol"]
    for tematica in tematicas_secundarias:
        preguntas_bloque = random.sample(POOLS_PREGUNTAS[tematica], 10)
        for p in preguntas_bloque:
            p["peso"] = 1.0  # Peso estándar
            p["tema"] = tematica
        test_seleccionado.extend(preguntas_bloque)
        
    st.session_state.test_actual = test_seleccionado

# --- 5. FORMULARIO STREAMLIT ---
with st.form("formulario_test"):
    respuestas = []
    tema_actual = ""
    
    for idx, p in enumerate(st.session_state.test_actual):
        # Para hacer la separación visual por temáticas sin que quede cutre
        if p["tema"] != tema_actual:
            tema_actual = p["tema"]
            st.markdown(f"### Sección: {tema_actual.replace('_', ' ')}")
            
        st.markdown(f"**{idx + 1}. {p['pregunta']}**")
        opcion = st.radio(f"Opciones {idx}", list(p["opciones"].keys()), index=None, label_visibility="collapsed", key=f"radio_{idx}")
        respuestas.append(opcion)
        st.write("---")
        
    enviado = st.form_submit_button("Analizar mi Psique Completa", type="primary")

# --- 6. MOTOR DE CÁLCULO ABSOLUTO (CON PESOS) ---
if enviado:
    preguntas_faltantes = [str(i + 1) for i, resp in enumerate(respuestas) if resp is None]
    
    if preguntas_faltantes:
        st.error(f"⚠️ ¡Faltan datos! Responde estas preguntas para ajustar las matemáticas: {', '.join(preguntas_faltantes)}.")
    else:
        with st.spinner('Procesando 8 dimensiones e interceptando PokéAPI...'):
            perfil_personalidad = np.zeros(8)
            puntuacion_tipos = {t: 0 for t in ['Normal', 'Fuego', 'Agua', 'Eléctrico', 'Planta', 'Hielo', 'Lucha', 'Veneno', 'Tierra', 'Volador', 'Psíquico', 'Bicho', 'Roca', 'Fantasma', 'Dragón', 'Siniestro', 'Acero', 'Hada']}
            
            # Acumulación Cruzada multiplicada por el peso del bloque
            for index, respuesta_usuario in enumerate(respuestas):
                pregunta = st.session_state.test_actual[index]
                opciones_pregunta = pregunta["opciones"]
                datos = opciones_pregunta[respuesta_usuario]
                
                peso = pregunta["peso"]
                
                # Sumamos/restamos los pesos exactos de la dimensión multiplicados
                perfil_personalidad += (np.array(datos["perfil"]) * peso)
                # Asignamos el tipo de Gimnasio multiplicado
                for tipo in datos["tipos"]:
                    puntuacion_tipos[tipo] += (1 * peso)

            # Evitamos negativos absolutos
            perfil_personalidad = np.clip(perfil_personalidad, 0, None)
            if np.sum(perfil_personalidad) == 0:
                perfil_personalidad += 0.1 
                
            df_pokemon[columnas_dimensiones] = df_pokemon[columnas_dimensiones].replace(0, 0.1)

            # --- NORMALIZACIÓN ---
            max_puntos_usuario = np.max(perfil_personalidad)
            if max_puntos_usuario > 0:
                vector_usuario_norm = (perfil_personalidad / max_puntos_usuario) * 10
            else:
                vector_usuario_norm = perfil_personalidad
                
            vector_usuario_norm = vector_usuario_norm.reshape(1, -1)

            # 1. MATEMÁTICAS DEL POKÉMON (Distancia Euclidiana)
            distancias = euclidean_distances(vector_usuario_norm, df_pokemon[columnas_dimensiones].values)[0]
            indices_top = np.argsort(distancias)[:6]
            mejor_pokemon = df_pokemon.iloc[indices_top[0]]
            
            def calcular_afinidad(dist):
                afinidad = 100 - (dist * 4.5)
                return round(max(0.1, afinidad), 1)

            porcentaje_afinidad = calcular_afinidad(distancias[indices_top[0]])
            
            # 2. LÓGICA DE GIMNASIO 
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
                
                st.write(f"**Tipo Dominante:** {tipo_primario} ({puntos_1:.1f} pts)")
                st.write(f"**Tipo Secundario:** {tipo_secundario} ({puntos_2:.1f} pts)")
                
                st.write("---")
                st.markdown("**Tus 5 Pokémon más cercanos:**")
                for i in indices_top[1:6]:
                    poke_cercano = df_pokemon.iloc[i]
                    afinidad_cercana = calcular_afinidad(distancias[i])
                    st.write(f"- **{poke_cercano['nombre']}** ({afinidad_cercana}%)")
                
            st.divider()
            
            st.subheader("El Por qué de tu Resultado (Tus Stats)")
            
            st.write(f"1. **{estadisticas[0][0]}** dominante ({estadisticas[0][1]:.1f}%)")
            st.write(f"2. Fuerte presencia de **{estadisticas[1][0]}** ({estadisticas[1][1]:.1f}%)")
            if estadisticas[2][1] > 10.0:
                st.write(f"3. Matices de **{estadisticas[2][0]}** ({estadisticas[2][1]:.1f}%)")
                
            st.write(f"\nEsta combinación exacta en tu mapa de 8 dimensiones es la que el algoritmo ha emparejado milimétricamente con el *lore* y mecánicas internas de {mejor_pokemon['nombre']}.")