#!/usr/bin/env python3
import argparse
import asyncio
import shutil
import torch
from faster_whisper import WhisperModel
from rich.table import Table

from src.ui import console
from src.models import TEMP_DIR, OUTPUT_DIR
from src.youtube import download_project
from src.media import smart_chunk, generate_srt, render_video, mix_dubbing
from src.googlev4 import GoogleTranslator
from src.tts import get_voice, tts

async def run_pipeline(args):
    base_lang = args.lang or "en"
    final_lang_sub = args.lang_sub if args.lang_sub else base_lang
    final_lang_dub = args.lang_dub if args.lang_dub else base_lang
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = args.whisper_model or ("base" if device == "cpu" else "small")

    # --- UI: HEADER & MINIMAL TABLE ---
    console.header("YouTube Auto Dub")

    console.header("Configuration", center=False)
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Video URL", f"[#e5e7eb]{args.url}[/#e5e7eb]")
    table.add_row("Mode", f"[#e5e7eb]{args.mode.upper()}[/#e5e7eb]")
    if args.mode in ["sub", "both"]: table.add_row("Subtitles Language", f"[#e5e7eb]{final_lang_sub.upper()}[/#e5e7eb]")
    if args.mode in ["dub", "both"]: table.add_row("Dubbing Language", f"[#e5e7eb]{final_lang_dub.upper()}[/#e5e7eb]")
    if args.mode in ["dub", "both"]: table.add_row("Gender", f"[#e5e7eb]{'Male' if args.gender == 'male' else 'Female'}[/#e5e7eb]")
    table.add_row("Whisper Model", f"[#e5e7eb]{model_size.upper()} ({device.upper()})[/#e5e7eb]")
    
    console.print(table)
    console.print()

    # --- UI: EXECUTION FLOW ---
    with console.status("Processing...") as status:
        
        console.info("Downloading media")
        project = download_project(args.url, args.browser)

        console.info(f"Transcribing speech ({model_size})")
        model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
        segs, info = model.transcribe(str(project.audio_path), word_timestamps=False)
        detected_lang = info.language if info else None
        raw_segments = [{'start': s.start, 'end': s.end, 'text': s.text.strip()} for s in segs]
        
        del model
        if device == "cuda": torch.cuda.empty_cache()
        console.success(f"Transcription complete (detected: {detected_lang})")

        console.info("Analyzing silence gaps")
        project.segments = smart_chunk(raw_segments)
        texts = [seg.source_text for seg in project.segments]
        
        console.info("Connecting to translation engine")
        translator = GoogleTranslator()
        
        if args.mode in ["sub", "both"]:
            if detected_lang and detected_lang == final_lang_sub:
                console.step(f"Source language matches target ({final_lang_sub}), skipping translation")
                sub_texts = texts
            else:
                console.step(f"Translating {len(texts)} subtitle segments -> {final_lang_sub.upper()}")
                sub_texts = await translator.translate_batch(texts, source=detected_lang or "auto", target=final_lang_sub)
            for i, seg in enumerate(project.segments):
                res_sub = sub_texts[i].strip() if i < len(sub_texts) else ""
                seg.translated_text_sub = res_sub if res_sub else seg.source_text

        if args.mode in ["dub", "both"]:
            if args.mode == "both" and final_lang_dub == final_lang_sub:
                dub_texts = sub_texts
            elif detected_lang and detected_lang == final_lang_dub:
                console.step(f"Source language matches target ({final_lang_dub}), skipping translation")
                dub_texts = texts
            else:
                console.step(f"Translating {len(texts)} dubbing scripts -> {final_lang_dub.upper()}")
                dub_texts = await translator.translate_batch(texts, source=detected_lang or "auto", target=final_lang_dub)
                
            for i, seg in enumerate(project.segments):
                res_dub = dub_texts[i].strip() if i < len(dub_texts) else ""
                seg.translated_text_dub = res_dub if res_dub else seg.source_text

        await translator.close()
        console.success("Translation complete")

        if args.mode in ["dub", "both"]:
            voice = get_voice(final_lang_dub, args.gender)
            console.info(f"Generating text-to-speech ({voice})")
            tasks = []
            for i, seg in enumerate(project.segments):
                seg.tts_audio_path = TEMP_DIR / f"tts_{i}.mp3"
                tasks.append(tts(seg.translated_text_dub, voice, seg.tts_audio_path))

            await asyncio.gather(*tasks)
                
            project.dub_audio_path = TEMP_DIR / "final_dub.wav"
            mix_dubbing(project.audio_path, project.segments, project.dub_audio_path)
            console.success("Dubbing & mixing complete")

        if args.mode in ["sub", "both"]:
            console.info("Generating SRT subtitles")
            project.subtitle_path = TEMP_DIR / "subtitles.srt"
            generate_srt(project.segments, project.subtitle_path)
            console.success("Subtitles saved")
        
        console.info("Rendering final video")
        lang_info = f"L-{base_lang}"
        if args.lang_sub: lang_info += f"_S-{final_lang_sub}"
        if args.lang_dub: lang_info += f"_D-{final_lang_dub}"
        
        output_name = f"Output_{args.mode}_{lang_info}_{project.video_id}.mp4"
        project.output_path = OUTPUT_DIR / output_name
        
        render_video(
            video_path=project.video_path,
            subtitle_path=project.subtitle_path if args.mode in ["sub", "both"] else None,
            dub_audio_path=project.dub_audio_path if args.mode in ["dub", "both"] else None,
            output_path=project.output_path
        )
        console.success("Video rendered")

    console.print()
    console.print(f"[bold #38bdf8]Output: {project.output_path.resolve()}[/bold #38bdf8]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Auto Sub/Dub Studio")
    parser.add_argument("url", help="YouTube video URL")
    
    parser.add_argument("--lang", "-l", help="General target language (Default: vi)")
    parser.add_argument("--lang_sub", "-ls", help="Subtitle language (Overrides --lang)")
    parser.add_argument("--lang_dub", "-ld", help="Dubbing language (Overrides --lang)")
    
    parser.add_argument("--mode", "-m", choices=["sub", "dub", "both"], default="both", help="Processing mode")
    parser.add_argument("--gender", "-g", choices=["male", "female"], default="female", help="Voice gender")
    
    parser.add_argument("--browser", "-b", help="Browser to extract cookies from (chrome, edge, firefox)")
    parser.add_argument("--whisper_model", "-wm", help="Whisper model (tiny, base, small, medium)")
    args = parser.parse_args()

    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR, ignore_errors=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"\n[red]System Error: {e}[/red]")