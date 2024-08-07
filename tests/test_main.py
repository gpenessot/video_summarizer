import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.main import extract_audio, transcribe_audio, summarize_text, process_video


@pytest.fixture
def temp_video():
    # Créer un fichier vidéo temporaire factice pour les tests
    temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp.close()
    yield temp.name
    # Supprimer le fichier vidéo temporaire après les tests
    os.unlink(temp.name)

@patch('moviepy.editor.VideoFileClip')
def test_extract_audio(mock_video_file_clip, temp_video):
    # Simuler l'extraction audio
    mock_audio = MagicMock()
    mock_video_file_clip.return_value.audio = mock_audio

    audio_path = "test_audio.wav"
    extract_audio(temp_video, audio_path)

    # Vérifier que la méthode write_audiofile a été appelée
    mock_audio.write_audiofile.assert_called_once_with(audio_path)

@patch('whisper.load_model')
def test_transcribe_audio(mock_load_model):
    # Simuler la transcription
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "Ceci est un test de transcription."}
    mock_load_model.return_value = mock_model

    result = transcribe_audio("test_audio.wav")

    assert result == "Ceci est un test de transcription."

@patch('transformers.AutoTokenizer.from_pretrained')
@patch('transformers.AutoModelForCausalLM.from_pretrained')
def test_summarize_text(mock_model, mock_tokenizer):
    # Simuler la génération de résumé
    mock_tokenizer.return_value.return_value = MagicMock()
    mock_model.return_value.generate.return_value = [MagicMock()]
    mock_tokenizer.return_value.decode.return_value = "Résumé : Ceci est un résumé de test."

    result = summarize_text("Ceci est un long texte à résumer.")

    assert result == "Ceci est un résumé de test."

@patch('main.extract_audio')
@patch('main.transcribe_audio')
@patch('main.summarize_text')
def test_process_video(mock_summarize, mock_transcribe, mock_extract, temp_video):
    # Simuler le processus complet
    mock_transcribe.return_value = "Ceci est une transcription de test."
    mock_summarize.return_value = "Ceci est un résumé de test."

    result = process_video(temp_video)

    assert result is not None
    transcription, summary = result
    assert transcription == "Ceci est une transcription de test."
    assert summary == "Ceci est un résumé de test."

@patch('main.extract_audio', side_effect=Exception("Erreur d'extraction"))
def test_process_video_error(mock_extract, temp_video):
    # Tester la gestion des erreurs
    result = process_video(temp_video)

    assert result is None

def test_main_cli(capsys, temp_video):
    # Test de l'interface en ligne de commande
    with patch('sys.argv', ['main.py', temp_video]), \
         patch('main.process_video', return_value=("Transcription test", "Résumé test")):
        from main import main
        main()
        captured = capsys.readouterr()
        assert "Transcription test" in captured.out
        assert "Résumé test" in captured.out