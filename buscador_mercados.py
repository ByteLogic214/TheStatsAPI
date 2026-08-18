#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buscador_mercados.py
================================================================================
Script de procesamiento de eventos deportivos en tiempo real con algoritmo
probabilístico agnóstico. Filtra mercados con probabilidad >= 85%.

PROVEEDOR: TheStatsAPI (https://thestatsapi.com)
  - Live realtime odds y match stats
  - xG (Expected Goals) integrado nativamente
  - 150+ competiciones, 84,000+ jugadores, 10 años histórico

REQUISITOS:
    pip install requests python-dotenv

CONFIGURACIÓN:
    Exportar variable de entorno: export THESTATS_API_KEY="tu_token_aqui"

ENDPOINTS DOCUMENTADOS (TheStatsAPI):
    GET /v1/football/matches/live          -> Partidos en vivo
    GET /v1/football/matches/{id}/stats    -> Estadísticas detalladas
    GET /v1/football/matches/{id}/events   -> Eventos del partido
    GET /v1/football/matches/{id}/shotmap  -> Mapa de tiros con xG por disparo
================================================================================
"""

import os
import sys
import json
import math
import random
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import requests
from requests.adapters import HTTPAdapter
from urllib.parse import urlencode

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BuscadorMercados")

# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

# ── URL BASE REAL DE THESTATSAPI ──
BASE_URL = "https://api.thestatsapi.com"
API_VERSION = "v1"

# ── VARIABLE DE ENTORNO ──
ENV_API_KEY = "THESTATS_API_KEY"

# ── UMBRAL DE PROBABILIDAD ──
UMBRAL_PROBABILIDAD = 85.0  # %

# ── ARCHIVO DE SALIDA ──
OUTPUT_FILE = "partidos_alta_probabilidad.md"

# ── TIMEOUTS Y RETRIES ──
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3

# ── LÍMITES DE RATE (TheStatsAPI: 30 req/min en Starter) ──
RATE_LIMIT_DELAY = 2.1  # segundos entre requests

# ── FACTORES DE PESO PARA EL ALGORITMO ──
# Calibrables con backtesting. Se usan para ajustar el modelo Poisson
# cuando xG nativo no está disponible o como complemento.
PESOS_MOMENTO = {
    "goles": 0.25,
    "tiros_puerta": 0.20,
    "ataques_peligrosos": 0.15,
    "posesion": 0.10,
    "corners": 0.10,
    "tarjetas": 0.05,
    "momento_partido": 0.15,
}


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class EstadisticasVivo:
    """Contenedor de estadísticas en vivo de un equipo."""
    goles: int = 0
    tiros_puerta: int = 0
    tiros_totales: int = 0
    tiros_fuera: int = 0
    ataques_peligrosos: int = 0
    ataques_totales: int = 0
    posesion: float = 50.0
    corners: int = 0
    corners_contra: int = 0
    tarjetas_amarillas: int = 0
    tarjetas_rojas: int = 0
    faltas: int = 0
    paradas: int = 0

    # xG nativo de TheStatsAPI (cuando disponible)
    xg: Optional[float] = None
    xg_against: Optional[float] = None

    def xg_estimado(self) -> float:
        """
        Calcula Expected Goals. Prioriza xG nativo de TheStatsAPI.
        Si no está disponible, estima desde tiros a puerta.
        """
        if self.xg is not None:
            return self.xg
        # Fallback: ratio conservador ~0.32 xG por tiro a puerta
        return self.tiros_puerta * 0.32


@dataclass
class PartidoEnVivo:
    """Representación estructurada de un partido en tiempo real."""
    match_id: str
    liga: str
    liga_id: Optional[str] = None
    pais: str = ""

    equipo_local: str
    equipo_local_id: Optional[str] = None
    equipo_visitante: str
    equipo_visitante_id: Optional[str] = None

    minuto: int = 0
    minuto_display: str = "0'"
    estado: str = "SCHEDULED"  # SCHEDULED, LIVE, HT, FT, POSTPONED, etc.
    periodo: str = ""  # 1H, 2H, ET, P

    stats_local: EstadisticasVivo = field(default_factory=EstadisticasVivo)
    stats_visitante: EstadisticasVivo = field(default_factory=EstadisticasVivo)

    # Odds en vivo (si vienen en el payload)
    odds: Dict = field(default_factory=dict)

    # Datos históricos/clasificación
    ranking_local: Optional[int] = None
    ranking_visitante: Optional[int] = None
    forma_local: List[str] = field(default_factory=list)
    forma_visitante: List[str] = field(default_factory=list)

    def goles_total(self) -> int:
        return self.stats_local.goles + self.stats_visitante.goles

    def diferencia_goles(self) -> int:
        return self.stats_local.goles - self.stats_visitante.goles

    def tiempo_restante_estimado(self) -> int:
        """Estima minutos restantes basado en estado actual."""
        if self.estado in ("FT", "FINISHED", "POSTPONED", "CANCELLED"):
            return 0
        if self.estado == "HT":
            return 45
        # LIVE o ET
        return max(90 - self.minuto, 1)


@dataclass
class MercadoCalculado:
    """Resultado de probabilidad para un mercado específico."""
    tipo_mercado: str
    seleccion: str
    probabilidad: float
    cuota_implicita: Optional[float] = None
    confianza: str = "alta"

    def __post_init__(self):
        # Clamp probabilidad a [0, 100]
        self.probabilidad = max(0.0, min(100.0, self.probabilidad))


# =============================================================================
# CLIENTE API - THESTATSAPI
# =============================================================================

class TheStatsAPIClient:
    """
    Cliente HTTP para TheStatsAPI.
    Documentación: https://thestatsapi.com/docs

    Endpoints principales:
      - GET /v1/football/matches/live
      - GET /v1/football/matches/{id}/stats
      - GET /v1/football/matches/{id}/events
    """

    def __init__(self):
        self.api_key = self._cargar_api_key()
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=MAX_RETRIES)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "BuscadorMercados/1.0",
            "X-API-Key": self.api_key,  # TheStatsAPI usa header auth
        })
        self._last_request_time = 0.0

    def _cargar_api_key(self) -> str:
        """Lee la API key desde variables de entorno de forma segura."""
        key = os.environ.get(ENV_API_KEY)
        if not key:
            logger.error(
                f"Variable de entorno '{ENV_API_KEY}' no encontrada.\n"
                f"Ejecuta: export {ENV_API_KEY}=\"tu_token_aqui\""
            )
            sys.exit(1)
        return key.strip()

    def _rate_limit(self):
        """Respeta el rate limit de TheStatsAPI (30 req/min)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Request HTTP con manejo robusto de errores."""
        self._rate_limit()
        url = f"{BASE_URL.rstrip('/')}/{API_VERSION}/{endpoint.lstrip('/')}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            else:
                response = self.session.request(method, url, json=params, timeout=REQUEST_TIMEOUT)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            logger.error(f"No se pudo conectar a {BASE_URL}. Verifica tu conexión.")
            return None
        except requests.exceptions.Timeout:
            logger.error("Timeout al consultar la API.")
            return None
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status == 401:
                logger.error("API Key inválida o expirada. Verifica THESTATS_API_KEY.")
            elif status == 429:
                logger.error("Rate limit excedido. Esperando antes de reintentar...")
                time.sleep(10)
            else:
                logger.error(f"Error HTTP {status}: {e.response.text}")
            return None
        except json.JSONDecodeError:
            logger.error("La respuesta no es JSON válido.")
            return None

    def obtener_partidos_en_vivo(self) -> List[Dict]:
        """
        Obtiene partidos en vivo desde TheStatsAPI.
        Endpoint: GET /v1/football/matches/live
        """
        logger.info("Consultando partidos en vivo desde TheStatsAPI...")
        data = self._request("GET", "/football/matches/live")

        if data is None:
            return []

        # TheStatsAPI devuelve: {"data": [...]} o directamente [...]
        if isinstance(data, dict):
            matches = data.get("data", data.get("matches", data.get("response", [])))
        elif isinstance(data, list):
            matches = data
        else:
            matches = []

        logger.info(f"Partidos en vivo recibidos: {len(matches)}")
        return matches

    def obtener_estadisticas_partido(self, match_id: str) -> Optional[Dict]:
        """
        Obtiene estadísticas detalladas de un partido.
        Endpoint: GET /v1/football/matches/{id}/stats
        """
        logger.debug(f"Obteniendo stats para partido {match_id}")
        return self._request("GET", f"/football/matches/{match_id}/stats")

    def obtener_eventos_partido(self, match_id: str) -> Optional[Dict]:
        """
        Obtiene eventos (goles, tarjetas, sustituciones) de un partido.
        Endpoint: GET /v1/football/matches/{id}/events
        """
        return self._request("GET", f"/football/matches/{match_id}/events")


# =============================================================================
# PARSER DE DATOS - THESTATSAPI
# =============================================================================

class TheStatsParser:
    """
    Transforma el JSON de TheStatsAPI en objetos PartidoEnVivo tipados.

    Estructura esperada de TheStatsAPI (basada en documentación pública):
    {
      "match_id": "12345",
      "league": {"name": "Premier League", "id": "39", "country": "England"},
      "home_team": {"name": "Liverpool", "id": "40"},
      "away_team": {"name": "Man City", "id": "50"},
      "status": "LIVE",
      "minute": 67,
      "minute_display": "67'",
      "period": "2H",
      "score": {"home": 2, "away": 1},
      "statistics": {
        "home": {
          "shots_on_target": 5, "shots_total": 12, "shots_off_target": 7,
          "possession": 55.2, "corners": 4, "yellow_cards": 1, "red_cards": 0,
          "dangerous_attacks": 45, "attacks": 120,
          "xg": 1.85, "xg_against": 0.92,
          "fouls": 8, "saves": 3
        },
        "away": { ... }
      },
      "odds": { ... }
    }
    """

    @staticmethod
    def parsear(json_partido: Dict) -> Optional[PartidoEnVivo]:
        try:
            match_id = str(json_partido.get("match_id", json_partido.get("id", "unknown")))

            # Liga
            league = json_partido.get("league", {})
            if isinstance(league, str):
                liga_nombre = league
                liga_id = None
                pais = ""
            else:
                liga_nombre = league.get("name", "Desconocida")
                liga_id = str(league.get("id", "")) if league.get("id") else None
                pais = league.get("country", "")

            # Equipos
            home = json_partido.get("home_team", json_partido.get("home", {}))
            away = json_partido.get("away_team", json_partido.get("away", {}))

            equipo_local = home.get("name", "Local") if isinstance(home, dict) else str(home)
            equipo_local_id = str(home.get("id", "")) if isinstance(home, dict) and home.get("id") else None
            equipo_visitante = away.get("name", "Visitante") if isinstance(away, dict) else str(away)
            equipo_visitante_id = str(away.get("id", "")) if isinstance(away, dict) and away.get("id") else None

            # Estado y tiempo
            estado = json_partido.get("status", "SCHEDULED")
            minuto_raw = json_partido.get("minute", 0)
            minuto = int(minuto_raw) if minuto_raw else 0
            minuto_display = json_partido.get("minute_display", f"{minuto}'")
            periodo = json_partido.get("period", "")

            # Marcador
            score = json_partido.get("score", json_partido.get("goals", {}))
            goles_local = int(score.get("home", score.get("local", 0))) if isinstance(score, dict) else 0
            goles_visitante = int(score.get("away", score.get("visitor", 0))) if isinstance(score, dict) else 0

            # Estadísticas
            stats = json_partido.get("statistics", json_partido.get("stats", {}))

            def extraer_stats(equipo_stats: Dict, goles: int) -> EstadisticasVivo:
                if not isinstance(equipo_stats, dict):
                    equipo_stats = {}
                return EstadisticasVivo(
                    goles=goles,
                    tiros_puerta=int(equipo_stats.get("shots_on_target", equipo_stats.get("shotsOnTarget", 0))),
                    tiros_totales=int(equipo_stats.get("shots_total", equipo_stats.get("shotsTotal", 0))),
                    tiros_fuera=int(equipo_stats.get("shots_off_target", equipo_stats.get("shotsOffTarget", 0))),
                    ataques_peligrosos=int(equipo_stats.get("dangerous_attacks", equipo_stats.get("dangerousAttacks", 0))),
                    ataques_totales=int(equipo_stats.get("attacks", equipo_stats.get("total_attacks", 0))),
                    posesion=float(equipo_stats.get("possession", equipo_stats.get("ball_possession", 50.0))),
                    corners=int(equipo_stats.get("corners", 0)),
                    corners_contra=int(equipo_stats.get("corners_against", 0)),
                    tarjetas_amarillas=int(equipo_stats.get("yellow_cards", equipo_stats.get("yellowCards", 0))),
                    tarjetas_rojas=int(equipo_stats.get("red_cards", equipo_stats.get("redCards", 0))),
                    faltas=int(equipo_stats.get("fouls", 0)),
                    paradas=int(equipo_stats.get("saves", equipo_stats.get("goalkeeper_saves", 0))),
                    xg=float(equipo_stats.get("xg", equipo_stats.get("expected_goals", None))) if equipo_stats.get("xg") or equipo_stats.get("expected_goals") else None,
                    xg_against=float(equipo_stats.get("xg_against", None)) if equipo_stats.get("xg_against") else None,
                )

            stats_local = extraer_stats(stats.get("home", stats.get("local", {})), goles_local)
            stats_visitante = extraer_stats(stats.get("away", stats.get("visitor", {})), goles_visitante)

            # Odds
            odds = json_partido.get("odds", {})

            return PartidoEnVivo(
                match_id=match_id,
                liga=liga_nombre,
                liga_id=liga_id,
                pais=pais,
                equipo_local=equipo_local,
                equipo_local_id=equipo_local_id,
                equipo_visitante=equipo_visitante,
                equipo_visitante_id=equipo_visitante_id,
                minuto=minuto,
                minuto_display=minuto_display,
                estado=estado,
                periodo=periodo,
                stats_local=stats_local,
                stats_visitante=stats_visitante,
                odds=odds,
            )

        except Exception as e:
            logger.warning(f"Error parseando partido: {e}")
            return None


# =============================================================================
# MOTOR PROBABILÍSTICO
# =============================================================================

class MotorProbabilistico:
    """
    Algoritmo agnóstico que calcula probabilidades para múltiples mercados.

    Modelos estadísticos utilizados:
    - Distribución de Poisson para goles (con xG nativo de TheStatsAPI como lambda)
    - Expected Goals (xG) integrado nativamente desde la API
    - Momentum del partido (posesión, ataques peligrosos, corners)
    - Factor tiempo (minuto actual vs. minutos restantes)
    - Simulación Monte Carlo para mercados complejos (1X2, hándicaps)
    - Ajuste por tarjetas rojas (ventaja numérica)
    """

    def __init__(self):
        self.pesos = PESOS_MOMENTO

    # -------------------------------------------------------------------------
    # Modelos base
    # -------------------------------------------------------------------------

    def _lambda_poisson(self, stats: EstadisticasVivo, minuto: int, es_local: bool) -> float:
        """
        Calcula el parámetro lambda (tasa de goles esperada restante) para Poisson.

        Prioriza xG nativo de TheStatsAPI. Si no está disponible, usa estimación
        basada en tiros a puerta y otros indicadores de momento.
        """
        if minuto <= 0:
            minuto = 1

        minutos_restantes = max(90 - minuto, 1)
        factor_tiempo = minutos_restantes / 90.0

        # xG base: nativo o estimado
        xg_base = stats.xg_estimado()

        # Ajuste por posesión
        factor_posesion = 1.0 + ((stats.posesion - 50.0) / 100.0)

        # Ajuste por ataques peligrosos (normalizado)
        factor_ataques = 1.0 + (stats.ataques_peligrosos / 50.0)

        # Ajuste por corners (indicador de presión ofensiva)
        factor_corners = 1.0 + (stats.corners / 10.0)

        # Ajuste por paradas del portero rival (menos paradas = más vulnerable)
        # Nota: paradas son del equipo propio, usamos proxy inverso
        factor_porteria = 1.0

        # Ajuste por tarjetas rojas (desventaja numérica reduce lambda)
        factor_tarjetas = 1.0 - (stats.tarjetas_rojas * 0.35)

        # Lambda ajustado para tiempo restante
        lambda_ajustado = (
            xg_base * factor_tiempo * factor_posesion * 
            factor_ataques * factor_corners * factor_tarjetas * factor_porteria
        )

        return max(lambda_ajustado, 0.05)

    def _prob_poisson(self, k: int, lambda_val: float) -> float:
        """Probabilidad P(X=k) con distribución de Poisson."""
        return (math.exp(-lambda_val) * (lambda_val ** k)) / math.factorial(k)

    def _prob_mas_de_n_goles(self, lambda_local: float, lambda_visit: float, n: int) -> float:
        """P(Total > n) = 1 - P(Total <= n) usando convolución de Poisson."""
        prob_acum = 0.0
        for i in range(n + 1):
            for j in range(n + 1 - i):
                prob_acum += self._prob_poisson(i, lambda_local) * self._prob_poisson(j, lambda_visit)
        return min(1.0 - prob_acum, 1.0)

    def _prob_exactamente_n_goles(self, lambda_local: float, lambda_visit: float, n: int) -> float:
        """P(Total = n)."""
        prob = 0.0
        for i in range(n + 1):
            j = n - i
            prob += self._prob_poisson(i, lambda_local) * self._prob_poisson(j, lambda_visit)
        return prob

    def _poisson_sample(self, lambda_val: float) -> int:
        """Muestreo de Poisson mediante método de Knuth."""
        L = math.exp(-lambda_val)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= random.random()
            if p <= L:
                break
        return k - 1

    # -------------------------------------------------------------------------
    # Cálculo de mercados
    # -------------------------------------------------------------------------

    def calcular_todos(self, partido: PartidoEnVivo) -> List[MercadoCalculado]:
        """Calcula probabilidades para todos los mercados soportados."""
        mercados = []

        lamb_local = self._lambda_poisson(partido.stats_local, partido.minuto, True)
        lamb_visit = self._lambda_poisson(partido.stats_visitante, partido.minuto, False)

        minutos_restantes = partido.tiempo_restante_estimado()
        factor_urgencia = 1.0 + (1.0 / max(minutos_restantes, 1))

        # ── 1. OVER/UNDER GOLES ──
        mercados.extend(self._mercados_over_under(lamb_local, lamb_visit, partido))

        # ── 2. AMBOS MARCAN (BTTS) ──
        mercados.extend(self._mercado_btts(lamb_local, lamb_visit, partido))

        # ── 3. DOBLE OPORTUNIDAD ──
        mercados.extend(self._mercado_doble_oportunidad(lamb_local, lamb_visit, partido))

        # ── 4. HÁNDICAPS ──
        mercados.extend(self._mercado_handicaps(lamb_local, lamb_visit, partido))

        # ── 5. GANADOR DEL ENCUENTRO (1X2) ──
        mercados.extend(self._mercado_1x2(lamb_local, lamb_visit, partido))

        # ── 6. SIGUIENTE EQUIPO EN ANOTAR ──
        mercados.extend(self._mercado_siguiente_gol(lamb_local, lamb_visit, partido, factor_urgencia))

        return mercados

    def _mercados_over_under(self, ll: float, lv: float, p: PartidoEnVivo) -> List[MercadoCalculado]:
        """Over/Under 0.5, 1.5, 2.5, 3.5, 4.5 goles."""
        resultados = []
        goles_actuales = p.goles_total()

        for linea in [0.5, 1.5, 2.5, 3.5, 4.5]:
            linea_ajustada = linea - goles_actuales

            if linea_ajustada <= 0:
                # Ya se superó la línea
                prob_over = 99.9
                prob_under = 0.1
            else:
                n = int(linea_ajustada)
                prob_over = self._prob_mas_de_n_goles(ll, lv, n) * 100
                prob_under = (1.0 - self._prob_mas_de_n_goles(ll, lv, n)) * 100

            resultados.append(MercadoCalculado(
                tipo_mercado=f"Over/Under {linea} Goles",
                seleccion=f"Over {linea}",
                probabilidad=round(prob_over, 2)
            ))
            resultados.append(MercadoCalculado(
                tipo_mercado=f"Over/Under {linea} Goles",
                seleccion=f"Under {linea}",
                probabilidad=round(prob_under, 2)
            ))

        return resultados

    def _mercado_btts(self, ll: float, lv: float, p: PartidoEnVivo) -> List[MercadoCalculado]:
        """Ambos Equipos Marcan (BTTS)."""
        # P(local marca al menos 1 gol más) = 1 - P(0 goles restantes)
        p_local_marca_resto = (1.0 - self._prob_poisson(0, ll))
        p_visit_marca_resto = (1.0 - self._prob_poisson(0, lv))

        # P(ambos marcan en el resto del partido)
        prob_btts_resto = p_local_marca_resto * p_visit_marca_resto

        # Ajuste por goles ya marcados
        local_ya_marco = p.stats_local.goles > 0
        visit_ya_marco = p.stats_visitante.goles > 0

        if local_ya_marco and visit_ya_marco:
            prob_btts_si = 99.9
            prob_btts_no = 0.1
        elif local_ya_marco or visit_ya_marco:
            # Uno ya marcó, la probabilidad de que el otro también lo haga
            # es simplemente P(el otro marca en el resto)
            if local_ya_marco:
                prob_btts_si = p_visit_marca_resto * 100
            else:
                prob_btts_si = p_local_marca_resto * 100
            prob_btts_no = 100.0 - prob_btts_si
        else:
            # Ninguno ha marcado aún
            prob_btts_si = prob_btts_resto * 100
            prob_btts_no = (1.0 - prob_btts_resto) * 100

        return [
            MercadoCalculado("Ambos Marcan", "Sí", round(prob_btts_si, 2)),
            MercadoCalculado("Ambos Marcan", "No", round(prob_btts_no, 2)),
        ]

    def _mercado_doble_oportunidad(self, ll: float, lv: float, p: PartidoEnVivo) -> List[MercadoCalculado]:
        """1X, 12, X2."""
        p_local, p_empate, p_visit = self._calcular_1x2(ll, lv, p)

        return [
            MercadoCalculado("Doble Oportunidad", "1X (Local o Empate)", round(min(p_local + p_empate, 99.9), 2)),
            MercadoCalculado("Doble Oportunidad", "12 (Local o Visitante)", round(min(p_local + p_visit, 99.9), 2)),
            MercadoCalculado("Doble Oportunidad", "X2 (Empate o Visitante)", round(min(p_empate + p_visit, 99.9), 2)),
        ]

    def _mercado_handicaps(self, ll: float, lv: float, p: PartidoEnVivo) -> List[MercadoCalculado]:
        """Hándicaps asiáticos del momento (-2, -1, 0, +1, +2)."""
        resultados = []

        for handicap in [-2, -1, 0, +1, +2]:
            victorias = 0
            empates = 0
            n_sim = 500

            for _ in range(n_sim):
                gl = p.stats_local.goles + self._poisson_sample(ll)
                gv = p.stats_visitante.goles + self._poisson_sample(lv)
                dif = (gl - gv) + handicap

                if dif > 0:
                    victorias += 1
                elif dif == 0:
                    empates += 1

            # En hándicap asiático, empate = push (devolución de stake)
            # Probabilidad de victoria efectiva
            prob = (victorias / n_sim) * 100

            signo = "+" if handicap >= 0 else ""
            resultados.append(MercadoCalculado(
                "Hándicap Asiático",
                f"Local {signo}{handicap}",
                round(prob, 2)
            ))

        return resultados

    def _mercado_1x2(self, ll: float, lv: float, p: PartidoEnVivo) -> List[MercadoCalculado]:
        """Ganador del encuentro (1X2)."""
        p_local, p_empate, p_visit = self._calcular_1x2(ll, lv, p)
        return [
            MercadoCalculado("Ganador Encuentro", "1 (Local)", round(p_local, 2)),
            MercadoCalculado("Ganador Encuentro", "X (Empate)", round(p_empate, 2)),
            MercadoCalculado("Ganador Encuentro", "2 (Visitante)", round(p_visit, 2)),
        ]

    def _mercado_siguiente_gol(self, ll: float, lv: float, p: PartidoEnVivo, urgencia: float) -> List[MercadoCalculado]:
        """Próximo equipo en anotar."""
        if p.estado in ("FT", "FINISHED", "POSTPONED", "CANCELLED"):
            return []
        if p.minuto >= 90 and p.periodo not in ("ET", "P"):
            return []

        # La tasa de gol instantánea es proporcional a lambda * urgencia
        tasa_local = ll * urgencia
        tasa_visit = lv * urgencia
        tasa_total = tasa_local + tasa_visit

        if tasa_total <= 0:
            return []

        prob_local = (tasa_local / tasa_total) * 100
        prob_visit = (tasa_visit / tasa_total) * 100

        return [
            MercadoCalculado("Siguiente Gol", f"{p.equipo_local}", round(prob_local, 2)),
            MercadoCalculado("Siguiente Gol", f"{p.equipo_visitante}", round(prob_visit, 2)),
        ]

    def _calcular_1x2(self, ll: float, lv: float, p: PartidoEnVivo) -> Tuple[float, float, float]:
        """Calcula probabilidades 1X2 mediante simulación Monte Carlo."""
        n_sim = 1000
        loc_wins = empates = vis_wins = 0

        for _ in range(n_sim):
            goles_loc = p.stats_local.goles + self._poisson_sample(ll)
            goles_vis = p.stats_visitante.goles + self._poisson_sample(lv)

            if goles_loc > goles_vis:
                loc_wins += 1
            elif goles_loc == goles_vis:
                empates += 1
            else:
                vis_wins += 1

        total = n_sim
        return (
            (loc_wins / total) * 100,
            (empates / total) * 100,
            (vis_wins / total) * 100,
        )


# =============================================================================
# FILTRO DE SEGURIDAD (>= 85%)
# =============================================================================

class FiltroSeguridad:
    """
    Filtra mercados con probabilidad >= UMBRAL_PROBABILIDAD.
    Ordena de mayor a menor probabilidad.
    """

    @staticmethod
    def filtrar(mercados: List[MercadoCalculado]) -> List[MercadoCalculado]:
        filtrados = [m for m in mercados if m.probabilidad >= UMBRAL_PROBABILIDAD]
        filtrados.sort(key=lambda x: x.probabilidad, reverse=True)
        return filtrados


# =============================================================================
# GENERADOR MARKDOWN PARA LECTURA MÓVIL
# =============================================================================

class MarkdownGenerator:
    """Genera el archivo Markdown formateado para lectura móvil."""

    @staticmethod
    def generar(hallazgos: List[Tuple[PartidoEnVivo, MercadoCalculado]]) -> str:
        ahora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        lines = [
            "# ⚽ Partidos de Alta Probabilidad",
            "",
            f"> **Actualizado:** {ahora} | **Umbral:** ≥ {UMBRAL_PROBABILIDAD}% | **Fuente:** TheStatsAPI",
            "",
            "---",
            "",
        ]

        if not hallazgos:
            lines.extend([
                "## 📭 Sin resultados",
                "",
                "**No se detectaron mercados de alta probabilidad en este momento.**",
                "",
                "Prueba actualizar en unos minutos.",
            ])
        else:
            lines.append(f"## 🎯 {len(hallazgos)} Mercados Detectados\n")

            for partido, mercado in hallazgos:
                # Emoji según probabilidad
                if mercado.probabilidad >= 95:
                    emoji_prob = "🔥"
                elif mercado.probabilidad >= 90:
                    emoji_prob = "🟢"
                else:
                    emoji_prob = "🟡"

                # Estado visual
                estado_emoji = "⏱️" if partido.estado == "LIVE" else "🛑"

                lines.extend([
                    f"### {emoji_prob} {partido.equipo_local} vs {partido.equipo_visitante}",
                    "",
                    f"| Campo | Valor |",
                    f"|-------|-------|",
                    f"| **Liga** | {partido.liga} ({partido.pais}) |",
                    f"| **Estado** | {estado_emoji} {partido.estado} — {partido.minuto_display} |",
                    f"| **Marcador** | {partido.stats_local.goles} - {partido.stats_visitante.goles} |",
                    f"| **Mercado** | `{mercado.tipo_mercado}` |",
                    f"| **Selección** | **{mercado.seleccion}** |",
                    f"| **Probabilidad** | `{mercado.probabilidad}%` |",
                    f"| **Confianza** | {mercado.confianza.upper()} |",
                    "",
                    "---",
                    "",
                ])

        lines.extend([
            "",
            "---",
            "",
            "*Generado automáticamente por BuscadorMercados v1.0*",
            "",
            f"*Proveedor de datos: TheStatsAPI | {ahora}*",
        ])

        return "\n".join(lines)

    @staticmethod
    def guardar(contenido: str, ruta: str = OUTPUT_FILE) -> None:
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(contenido)
        logger.info(f"Archivo guardado: {ruta}")


# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================

class BuscadorMercados:
    """
    Orquesta todo el pipeline:
    TheStatsAPI → Parse → Calcular → Filtrar (>=85%) → Exportar Markdown
    """

    def __init__(self):
        self.cliente = TheStatsAPIClient()
        self.parser = TheStatsParser()
        self.motor = MotorProbabilistico()
        self.filtro = FiltroSeguridad()
        self.formatter = MarkdownGenerator()

    def ejecutar(self) -> None:
        """Pipeline completo de ejecución."""
        logger.info("=" * 65)
        logger.info("INICIANDO BÚSQUEDA DE MERCADOS DE ALTA PROBABILIDAD")
        logger.info(f"Umbral: {UMBRAL_PROBABILIDAD}% | Fuente: TheStatsAPI")
        logger.info("=" * 65)

        # 1. Obtener datos crudos de partidos en vivo
        datos_crudos = self.cliente.obtener_partidos_en_vivo()

        if not datos_crudos:
            self._escribir_vacio()
            return

        # 2. Parsear a objetos tipados
        partidos: List[PartidoEnVivo] = []
        for raw in datos_crudos:
            p = self.parser.parsear(raw)
            if p and p.estado in ("LIVE", "HT", "ET"):
                partidos.append(p)

        logger.info(f"Partidos en vivo parseados: {len(partidos)}")

        if not partidos:
            self._escribir_vacio()
            return

        # 3. Calcular probabilidades y filtrar >= 85%
        hallazgos: List[Tuple[PartidoEnVivo, MercadoCalculado]] = []

        for partido in partidos:
            mercados = self.motor.calcular_todos(partido)
            mercados_filtrados = self.filtro.filtrar(mercados)

            for m in mercados_filtrados:
                hallazgos.append((partido, m))
                logger.info(
                    f"[ALTA PROB] {partido.equipo_local} vs {partido.equipo_visitante} | "
                    f"{m.tipo_mercado}: {m.seleccion} @ {m.probabilidad}%"
                )

        # 4. Generar Markdown
        markdown = self.formatter.generar(hallazgos)
        self.formatter.guardar(markdown)

        logger.info("=" * 65)
        logger.info(f"PROCESO COMPLETADO. Hallazgos >= {UMBRAL_PROBABILIDAD}%: {len(hallazgos)}")
        logger.info("=" * 65)

    def _escribir_vacio(self) -> None:
        """Escribe archivo indicando que no hay mercados activos."""
        markdown = self.formatter.generar([])
        self.formatter.guardar(markdown)
        logger.info("No se detectaron mercados de alta probabilidad en este momento.")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    buscador = BuscadorMercados()
    buscador.ejecutar()
