Reinf Semesterprojekt "Musikempfehlung"
=======================================

Beschreibung:
-------------
In diesem Semesterprojekt wird ein Musikempfehlungssystem entwickelt. Dieses soll auf Basis von Reinforcement Learning erstellt werden.

Installation:
-------------
1. Repository klonen
2. Virtual Environment erstellen:
    - `python3 -m venv venv`
    - Aktivieren des Virtual Environments:
        - Windows: `venv\Scripts\activate.ps1`
        - Linux: `source venv/bin/activate`
3. Abhängigkeiten installieren:
    - `pip install -r requirements.txt`

Ordnerstruktur:
---------------
- 'data-preprocessing': Enthält die Skripte zum Laden und Aufbereiten der Daten von Spotify
- 'RL-Model': Enthält das Reinforcement Learning Model und die Skripte zum Trainieren und Testen

Dateien:
--------
- `agent.py`: Implementation des Reinforcement Learning Agenten
- `config.py`: Konfigurationsparameter für das RL-Projekt
- `environment.py`: Das Musikempfehlungs-Environment inklusive der Rewards
- `main.py`: Hauptskript zum Trainieren des RL-Agents
- `test.py`: Skript zum Testen/Evaluieren des RL-Agents
- `visualizations.py`: Tools zum Plotten und Visualisieren


