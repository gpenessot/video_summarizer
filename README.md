# Video Transcription and Summary

Ce projet offre un outil pour transcrire le contenu audio d'une vidéo et générer un résumé concis du texte transcrit. Il utilise des technologies de pointe en traitement du langage naturel et en reconnaissance vocale pour fournir des résultats précis et pertinents.

## Fonctionnalités

- Extraction de l'audio à partir d'une vidéo
- Transcription de l'audio en texte
- Génération d'un résumé du texte transcrit
- Journalisation détaillée du processus

## Prérequis

- Python 3.10
- FFmpeg (pour l'extraction audio)
- GPU recommandé pour de meilleures performances (mais non obligatoire)

## Installation

1. Clonez ce dépôt :
   ```
   git clone https://github.com/votre-username/video-transcription-summary.git
   cd video-transcription-summary
   ```

2. Créez un environnement virtuel et activez-le :
   ```
   python -m venv venv
   source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
   ```

3. Installez les dépendances :
   ```
   pip install -r requirements.txt
   ```

4. Assurez-vous que FFmpeg est installé sur votre système. Si ce n'est pas le cas, suivez les instructions d'installation pour votre système d'exploitation sur [le site officiel de FFmpeg](https://ffmpeg.org/download.html).

## Utilisation

1. Placez votre fichier vidéo dans le répertoire du projet ou notez son chemin complet.

2. Modifiez la variable `video_path` dans le fichier `src/main.py` pour qu'elle pointe vers votre fichier vidéo :
   ```python
   video_path = "chemin/vers/votre/video.mp4"
   ```

3. Exécutez le script :
   ```
   python src/main.py
   ```

4. Les résultats (transcription et résumé) seront affichés dans la console et enregistrés dans le fichier `video_processing.log`.

## Structure du projet

```
video_transcription_summary/
│
├── src/
│   ├── __init__.py
│   └── main.py
│
├── tests/
│
├── docs/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Fonctionnement du code

Le script `main.py` contient plusieurs fonctions clés :

1. `extract_audio()` : Extrait l'audio de la vidéo.
2. `transcribe_audio()` : Utilise Whisper pour transcrire l'audio en texte.
3. `summarize_text()` : Utilise Mistral 7B pour générer un résumé du texte transcrit.
4. `main()` : Orchestre le processus complet et gère les erreurs.

Le script utilise `loguru` pour une journalisation détaillée, ce qui facilite le débogage et le suivi du processus.

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
