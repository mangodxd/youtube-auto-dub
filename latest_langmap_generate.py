import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import edge_tts

BASE_DIR = Path(__file__).resolve().parent
LANG_MAP_FILE = BASE_DIR / "youtube_auto_dub" / "language_map.json"

async def generate_lang_map() -> None:
    print("[*] Connecting to Microsoft Edge TTS API...")

    try:
        #fetch all available voices
        voices = await edge_tts.list_voices()
    except Exception as e:
        print(f"[!] CRITICAL: Failed to fetch voices: {e}")
        return

    print(f"[*] Processing {len(voices)} raw voice entries...")

    # Structure: { "vi": { "name": "vi-VN", "voices": { "male": [], "female": [] } } }
    lang_map: Dict[str, Any] = {}

    for v in voices:
        if "Neural" not in v["ShortName"]:
            continue

        short_name = v["ShortName"]      # e.g., "vi-VN-NamMinhNeural"
        locale = v["Locale"]             # e.g., "vi-VN"
        gender = v["Gender"].lower()     # "male" or "female"
        lang_code = locale.split('-')[0] # ISO Language Code (e.g., 'vi' from 'vi-VN')

        if lang_code not in lang_map:
            lang_map[lang_code] = {
                "name": locale,
                "voices": {"male": [], "female": []}
            }

        if gender in lang_map[lang_code]["voices"]:
            lang_map[lang_code]["voices"][gender].append(short_name)


    final_map = {
        k: v for k, v in lang_map.items()
        if v["voices"]["male"] or v["voices"]["female"]
    }

    try:
        with open(LANG_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(final_map, f, ensure_ascii=False, indent=2)

        print(f"\n[+] SUCCESS! Generated configuration for {len(final_map)} languages.")
        print(f"    File saved to -> {LANG_MAP_FILE}")

        if "vi" in final_map:
            print("\n[*] Preview (Vietnamese):")
            print(json.dumps(final_map["vi"], indent=2))

    except Exception as e:
        print(f"[!] ERROR: Failed to write JSON file: {e}")

if __name__ == "__main__":
    asyncio.run(generate_lang_map())
