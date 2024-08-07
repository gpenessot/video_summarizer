import moviepy.editor as mp
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from loguru import logger
from typing import Optional
import argparse

logger.add("video_processing.log", rotation="1 day")

def extract_audio(video_path: str, audio_path: str) -> None:
    """
    Extrait l'audio d'une vidéo et le sauvegarde dans un fichier.

    Args:
        video_path (str): Chemin vers le fichier vidéo d'entrée.
        audio_path (str): Chemin où sauvegarder le fichier audio extrait.

    Raises:
        Exception: Si l'extraction de l'audio échoue.
    """
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction de l'audio: {e}")
        raise

def transcribe_audio(audio_path: str) -> str:
    """
    Transcrit un fichier audio en texte.

    Args:
        audio_path (str): Chemin vers le fichier audio à transcrire.

    Returns:
        str: Texte transcrit.

    Raises:
        Exception: Si la transcription échoue.
    """
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Erreur lors de la transcription: {e}")
        raise

def summarize_text(text: str) -> str:
    """
    Génère un résumé du texte donné en utilisant Mistral 7B.

    Args:
        text (str): Texte à résumer.

    Returns:
        str: Résumé généré.

    Raises:
        Exception: Si la génération du résumé échoue.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, device_map="auto")

        prompt = f"Résume le texte suivant en français en 2-3 phrases : {text}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.split(":", 1)[-1].strip()
    except Exception as e:
        logger.error(f"Erreur lors de la génération du résumé: {e}")
        raise

def process_video(video_path: str) -> Optional[tuple[str, str]]:
    """
    Fonction principale qui gère le processus de traitement de la vidéo.

    Args:
        video_path (str): Chemin vers le fichier vidéo à traiter.

    Returns:
        Optional[tuple[str, str]]: Un tuple contenant la transcription et le résumé,
                                   ou None si une erreur se produit.
    """
    audio_path = "temp_audio.wav"
    
    try:
        logger.info("Début du traitement de la vidéo")
        
        logger.info("Extraction de l'audio...")
        extract_audio(video_path, audio_path)
        
        logger.info("Transcription de l'audio avec Whisper...")
        transcription = transcribe_audio(audio_path)
        
        logger.info("Génération de la synthèse avec Mistral 7B...")
        summary = summarize_text(transcription)
        
        logger.info("Transcription et résumé générés avec succès")
        logger.info(f"Transcription: {transcription}")
        logger.info(f"Synthèse: {summary}")
        
        return transcription, summary
    
    except Exception as e:
        logger.error(f"Une erreur est survenue lors du traitement: {e}")
        return None
    
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info("Fichier audio temporaire supprimé")

def main():
    parser = argparse.ArgumentParser(description="Transcrit et résume le contenu d'une vidéo.")
    parser.add_argument("video_path", help="Chemin vers le fichier vidéo à traiter")
    args = parser.parse_args()

    result = process_video(args.video_path)
    if result:
        transcription, summary = result
        print("\nTranscription:")
        print(transcription)
        print("\nSynthèse:")
        print(summary)
    else:
        print("Le traitement de la vidéo a échoué. Consultez les logs pour plus de détails.")

if __name__ == "__main__":
    main()