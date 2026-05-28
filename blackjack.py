"""
El problema del blackjack simplificado como un problema de aprendizaje por refuerzo

"""

from RL import MDPsim, SARSA, Q_learning, PoliticaGreedy
from random import random, randint, choices

class BlackJack(MDPsim):
    """
    Clase que representa un MDP simulación de Blackjack para Aprendizaje por Refuerzo.
    
    """
    def __init__(self, gama):
       
        # estados:
        # (suma_jugador, carta_visible_dealer, as_usable)

        self.estados = [
            (suma, carta, usable)
            for suma in range(12, 22)
            for carta in range(1, 11)
            for usable in (True, False)
        ]

        self.gama = gama

        # guardar mano del dealer
        self.mano_dealer = []

        # blackjack natural
        self.blackjack_natural = False

        super().__init__(self.estados, gama)

    def estado_inicial(self):
        """
        reinicia el juego y reparte dos cartas al jugador y dos al dealer.
        devuelve el estado inicial s0.
        si el jugador tiene menos de 12, pide cartas hasta llegar al rango.
        """
        self.mano_dealer = []
        self.blackjack_natural = False

        # cartas del jugador
        carta1 = self.reparte_carta()
        carta2 = self.reparte_carta()

        suma_jugador = 0
        as_usable = False

        suma_jugador, as_usable = self.ajustar_as(
            suma_jugador,
            as_usable,
            carta1
        )

        suma_jugador, as_usable = self.ajustar_as(
            suma_jugador,
            as_usable,
            carta2
        )

        # blackjack natural
        if suma_jugador == 21 and as_usable:
            self.blackjack_natural = True

        # si tiene menos de 12 pide automatico
        while suma_jugador < 12:

            nueva = self.reparte_carta()

            suma_jugador, as_usable = self.ajustar_as(
                suma_jugador,
                as_usable,
                nueva
            )

            # ya no es blackjack natural
            self.blackjack_natural = False

        # cartas dealer
        visible = self.reparte_carta()
        oculta = self.reparte_carta()

        self.mano_dealer = [visible, oculta]

        return (suma_jugador, visible, as_usable)

    def acciones_legales(self, s):
        """
        devuelve las acciones legales en el estado s.
        0 = stand (plantarse)
        1 = hit (pedir carta)
        """
        return [0, 1]

    def transicion(self, s, a):
        """
        devuelve el estado siguiente s' dado el estado s y la accion a.
        si hit: reparte carta y actualiza suma. si se pasa de 21 devuelve None.
        si stand: devuelve None (el dealer juega en recompensa).
        """
        suma_jugador, carta_visible, as_usable = s

        # hit
        if a == 1:

            nueva = self.reparte_carta()

            suma_jugador, as_usable = self.ajustar_as(
                suma_jugador,
                as_usable,
                nueva
            )

            # si se pasa termina
            if suma_jugador > 21:
                return None

            return (
                suma_jugador,
                carta_visible,
                as_usable
            )

        # stand
        elif a == 0:

            return None

    def recompensa(self, s, a, s_):
        """
        calcula la recompensa de la transicion (s, a, s').
        si no es terminal devuelve 0.
        si hit y se paso de 21 devuelve -1.
        si stand compara con el dealer y devuelve +1, 0 o -1.
        blackjack natural vale +1.5 (o 0 si el dealer tambien tiene 21).
        """
        # si no termina recompensa 0
        if s_ is not None:
            return 0

        # si pidio y se paso
        if a == 1:
            return -1

        # stand
        elif a == 0:

            suma_jugador, _, _ = s

            suma_dealer, as_dealer = self.calcular_mano(
                self.mano_dealer
            )

            # dealer pide hasta llegar a 17
            while suma_dealer < 17:

                carta = self.reparte_carta()

                suma_dealer, as_dealer = self.ajustar_as(
                    suma_dealer,
                    as_dealer,
                    carta
                )

            # comparar manos
            if suma_dealer > 21:
                recompensa = 1

            elif suma_jugador > suma_dealer:
                recompensa = 1

            elif suma_jugador < suma_dealer:
                recompensa = -1

            else:
                recompensa = 0

            # blackjack natural vale 1.5
            if self.blackjack_natural:

                suma_inicial_dealer, _ = self.calcular_mano(
                    self.mano_dealer
                )

                # empate de blackjack
                if suma_inicial_dealer == 21:
                    recompensa = 0

                else:
                    recompensa = 1.5

            return recompensa

    def es_terminal(self, s):
        """
        un estado es terminal cuando es None.
        """
        return s is None

    def reparte_carta(self):
        """
        devuelve una carta aleatoria de una baraja infinita.
        el 10 tiene probabilidad 4/13 porque representa 10, J, Q, K.
        """
        cartas = list(range(1, 11))

        pesos = [1] * 9 + [4]

        return choices(
            cartas,
            weights=pesos,
            k=1
        )[0]

    def ajustar_as(self, suma_actual, as_usable, nueva_carta):
        """
        añade nueva_carta a suma_actual respetando la logica del as.
        el as se cuenta como 11 si no excede 21, sino como 1.
        si la suma supera 21 y hay un as usable, se convierte a 1.
        """
        # si es as
        if nueva_carta == 1:

            # usar como 11
            if suma_actual + 11 <= 21:

                return suma_actual + 11, True

            # usar como 1
            else:

                return suma_actual + 1, as_usable

        # cartas normales
        else:

            suma = suma_actual + nueva_carta

            # convertir as de 11 a 1
            if suma > 21 and as_usable:

                suma -= 10

                return suma, False

            return suma, as_usable

    def calcular_mano(self, mano):
        """
        calcula la suma optima de una lista de cartas.
        """
        suma = 0
        as_usable = False

        for carta in mano:

            suma, as_usable = self.ajustar_as(
                suma,
                as_usable,
                carta
            )

        return suma, as_usable

if __name__ == "__main__":

    blackjack = BlackJack(gama=1)

    Q_sarsa = SARSA(blackjack, alfa=0.01, epsilon=0.5, n_ep=500_000, n_iter=100)
    Q_learning = Q_learning(blackjack, alfa=0.01, epsilon=0.5, n_ep=500_000, n_iter=100)

    # Encuentra las políticas óptimas para cada algoritmo
    pi_s = PoliticaGreedy(Q_sarsa)
    pi_q = PoliticaGreedy(Q_learning)

    # Imprime las políticas óptimas para cada estado no terminal
    print("Estado".center(30) + '|' + "SARSA".center(12) + '|' + "Q-learning".center(12))
    print("-" * 30 + '|' + "-" * 12 + '|' + "-" * 12)
    for s in blackjack.estados:
        accion_s = "Hit" if pi_s(s) == 1 else "Stand"
        accion_q = "Hit" if pi_q(s) == 1 else "Stand"
        print(str(s).center(30) + '|' + accion_s.center(12) + '|' + accion_q.center(12))
    print("-" * 30 + '|' + "-" * 12 + '|' + "-" * 12)


"""
****************************************************************************************
Responde las siguientes preguntas:

1. ¿Cuáles son los estados, acciones, recompensas y transiciones en el problema del 
    blackjack?  

    los estados son tuplas (suma_jugador, carta_visible_dealer, as_usable) donde la suma
    va de 12 a 21, la carta del dealer de 1 a 10, y el as_usable es True o False. Esto
    da un total de 10 * 10 * 2 = 200 estados posibles.
    las acciones son dos: 0 para Stand (plantarse) y 1 para Hit (pedir carta).
    las recompensas son 0 mientras el juego continua, +1 si ganas, -1 si pierdes, 0 en
    empate, y +1.5 si ganas con blackjack natural.
    las transiciones dependen de la accion: si haces Hit, recibes una carta y el estado
    se actualiza con la nueva suma. Si te pasas de 21 o haces Stand, el estado se vuelve
    terminal (None) y se calcula la recompensa final.

2. ¿Cómo se pueden representar los estados del blackjack de manera eficiente para el 
    aprendizaje por refuerzo?

    la representacion como tupla (suma_jugador, carta_visible_dealer, as_usable) es eficiente porque solo
    guarda la informacion relevante para tomar decisiones. No necesitamos saber todas las
    cartas que se han repartido, solo la suma actual y si tenemos un as flexible. Ademas,
    limitamos la suma a partir de 12 porque con menos siempre conviene pedir carta, lo que
    reduce el espacio de estados de forma inteligente.

3. ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?

    si epsilon es muy alto (cercano a 1), el agente explora mucho y toma acciones aleatorias
    con frecuencia, lo que hace que aprenda mas lento y la politica final sea menos optima.
    si epsilon es muy bajo (cercano a 0), el agente explota mas lo que ya sabe y explora
    poco, lo que puede hacer que se quede atascado en una politica suboptima porque no
    prueba suficientes alternativas.
    un valor intermedio permite balancear exploracion y explotacion, permitiendo que el
    agente aprenda bien sin desperdiciar demasiados episodios en acciones aleatorias.

4. ¿Cómo afecta el valor de alfa en la convergencia de los algoritmos SARSA y Q-learning?

    alfa es la tasa de aprendizaje. Si alfa es muy alto (cercano a 1), el agente actualiza
    los valores Q de forma agresiva, lo que puede causar oscilaciones y que no converja bien
    porque le da demasiado peso a las experiencias recientes.
    si alfa es muy bajo (cercano a 0), el agente aprende muy lento porque cada actualizacion
    cambia poco los valores Q, necesitando muchos mas episodios para converger.
    un valor como 0.01 funciona bien para este problema porque permite aprendizaje gradual
    y estable. Con 500,000 episodios hay suficiente tiempo para que los valores converjan
    incluso con un alfa pequeño.

5. ¿Cuál de los dos algoritmos, SARSA o Q-learning, consideras que es más adecuado para 
   el problema del blackjack y por qué?

    Q-learning es mas adecuado para blackjack porque es off-policy y aprende la politica
    optima directamente, sin importar la exploracion que haga durante el entrenamiento.
    SARSA es on-policy y aprende la politica que realmente sigue (incluyendo la exploracion),
    lo que lo hace mas conservador. Esto puede ser util en problemas donde las acciones
    exploratorias son peligrosas, pero en blackjack no hay consecuencias reales durante el
    entrenamiento.
    en los resultados se ve que Q-learning tiende a ser un poco mas agresivo y consistente,
    mientras que SARSA a veces toma decisiones mas cautelosas. Para encontrar la estrategia
    optima de blackjack, Q-learning es la mejor opcion.

6. ¿Se puede explicar con cierta lógica del juego la política óptima encontrada por cada 
   algoritmo? ¿Qué acciones se toman en cada estado y por qué?

   síp, cuando el jugador tiene valores altos como 19, 20 o 21, casi siempre conviene quedarse porque pedir 
   otra carta tiene mucho riesgo. En cambio, con manos bajas o medias, normalmente conviene pedir otra carta
   sobre todo si el dealer muestra cartas altas.
   se observa que en valores entre 12 y 16 muchas veces conviene quedarse si el dealer muestra cartas
   bajas, ya que existe una mayor probabilidad de que el dealer se pase de 21
   también se puede ver que cuando hay un as usable, los algoritmos juegan más agresivo porque el as puede cambiar
   de valor y reducir el riesgo de pasarse de 21. Las decisiones aprendidas se parecen bastante a las 
   estrategias que se suelen seguir realmente en el blackjack.
   en estados como (18, 9, True) o (18, 10, True), los algoritmos a veces prefieren pedir carta 
   porque el as usable reduce el riesgo de pasarse de 21 y todavía existe posibilidad de mejorar la mano.

****************************************************************************************
"""
