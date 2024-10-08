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
   git clone git@github.com:gpenessot/video_summarizer.git
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

Exécutez le script en spécifiant le chemin de votre fichier vidéo :

```
python main.py chemin/vers/votre/video.mp4
```

Les résultats (transcription et résumé) seront affichés dans la console et enregistrés dans le fichier `video_processing.log`.

## Structure du projet

```
video_transcription_summary/
│
├── src/
│   ├── __init__.py
│   └── main.py
│
├── tests/
│   └── test_main.py
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

## Tests

Ce projet utilise pytest pour les tests unitaires. Pour exécuter les tests :

1. Assurez-vous d'avoir installé les dépendances de développement :
   ```
   pip install -r requirements.txt
   ```

2. Exécutez les tests :
   ```
   pytest tests/
   ```

## CI/CD

Ce projet utilise GitHub Actions pour l'intégration continue et le déploiement continu (CI/CD). Le workflow est configuré pour :

- Se déclencher manuellement via l'interface GitHub Actions
- Formater le code avec Ruff
- Exécuter les tests avec pytest

Pour déclencher le workflow manuellement :

1. Allez dans l'onglet "Actions" du repository GitHub
2. Sélectionnez le workflow "Python CI"
3. Cliquez sur "Run workflow"
4. Optionnellement, spécifiez une branche, un tag ou un commit spécifique

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
