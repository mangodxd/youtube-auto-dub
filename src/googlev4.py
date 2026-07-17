import json
import re
import httpx
from typing import List
from bs4 import BeautifulSoup
from .ui import console

class GoogleTranslator:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)
        self.base_url_rpc = "https://translate.google.com/_/TranslateWebserverUi/data/batchexecute"
        self.base_url_scrape = "https://translate.google.com/m"
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        self.bl = None

    async def _refreshRpcToken(self):
        try:
            response = await self.client.get("https://translate.google.com/", headers=self.headers)
            bl_match = re.search(r'"cfb2h":"(.*?)"', response.text)
            if bl_match:
                self.bl = bl_match.group(1)
            else:
                self.bl = "boq_translate-webserver_20251215.06_p0"
        except Exception as e:
            console.warning(f"Token refresh failed: {e}. Using fallback.")
            self.bl = "boq_translate-webserver_20251215.06_p0"

    async def _parseRpcResponse(self, raw_text):
        try:
            match = re.search(r'\["wrb.fr","MkEWBc","(.*?)",null,null,null,"generic"\]', raw_text, re.DOTALL)
            if not match: raise ValueError("Could not find translation data in RPC response.")
            inner_json_str = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            data = json.loads(inner_json_str)
            translation_parts = data[1][0][0][5]
            final_text = " ".join([part[0] for part in translation_parts if part[0]])
            return final_text
        except Exception as e:
            raise ValueError(f"RPC Parse Error: {e}")

    async def _translateRpc(self, text, source, target):
        """Method 1: fake browser api requests
        
        Args:
            text: Text to translate.
            source: Source language code.
            target: Target language code.
            
        Returns:
            translated text string.
            
        Raises:
            exception: if translation fails.
        """
        if not self.bl:
            await self._refreshRpcToken()
        
        rpc_arg = json.dumps([[text, source, target, True, [1]]], ensure_ascii=False)
        f_req = json.dumps([["MkEWBc", rpc_arg, None, "generic"]])

        params = {
            "rpcids": "MkEWBc",
            "bl": self.bl,
            "hl": "en",
            "rt": "c"
        }
        
        response = await self.client.post(
            self.base_url_rpc, 
            headers=self.headers, 
            params=params, 
            data={"f.req": f_req}
        )

        if response.status_code != 200:
            raise Exception(f"RPC HTTP Error: {response.status_code}")
        
        return self._parseRpcResponse(response.text)

    async def _translateScrape(self, text, source, target):
        """method 2: Web Scraping. Simple fallback.
        
        Args:
            text: Text to translate.
            source: Source language code.
            target: Target language code.
            
        Returns:
            translated text string.
            
        Raises:
            exception: if translation fails.
        """
        params = {
            "sl": source,
            "tl": target,
            "q": text
        }
        
        response = await self.client.get(self.base_url_scrape, params=params, headers=self.headers)
        
        if response.status_code == 429:
            raise Exception("Too Many Requests (429)")
        if response.status_code != 200:
            raise Exception(f"Scrape HTTP Error: {response.text}")

        soup = BeautifulSoup(response.text, "html.parser")
        
        element = soup.find("div", {"class": "t0"})

        if not element:
            element = soup.find("div", {"class": "result-container"})
        if not element:
            raise Exception("Could not find translation element in HTML.")
            
        return element.get_text(strip=True)

    async def translate(self, text, source="auto", target="vi"):
        """Main interface. Tries API first, falls back to Scraping.
        
        Args:
            text: Text to translate.
            source: Source language code. Default 'auto'.
            target: Target language code. Default 'vi'.
            
        Returns:
            translated text string or error message.
        """
        if not text:
            return ""

        # Nếu ngôn ngữ nguồn trùng với ngôn ngữ đích, không cần dịch
        if source != "auto" and source == target:
            return text

        # Thử RPC API trước
        try:
            return await self._translateRpc(text, source, target)
        except Exception:
            pass

        # Thử scrape fallback
        try:
            return await self._translateScrape(text, source, target)
        except Exception as e:
            console.error(f"All translation methods failed: {e}")
            return text

    async def translate_batch(self, texts: List[str], target: str, source: str = "auto") -> List[str]:
        delimiter = "\n\n|||\n\n"
        combined = delimiter.join([t if t.strip() else " " for t in texts])

        translated_combined = await self.translate(combined, source=source, target=target)
        results = [t.strip() for t in translated_combined.split(delimiter.strip())]

        if len(results) != len(texts):
            results = [await self.translate(t, source=source, target=target) for t in texts]

        return results

    async def close(self):
        await self.client.aclose()