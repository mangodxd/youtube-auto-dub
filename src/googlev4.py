"""
Google Translate Integration Module for YouTube Auto Dub.

This module provides robust translation capabilities by implementing a dual-strategy 
approach: the internal 'batchexecute' RPC API for high-quality results, and a 
mobile web scraping fallback for maximum reliability.

Acknowlegement: 
This implementation is inspired by and adapted from the logic found in the 
'deep-translator' library (nidhaloff/deep-translator). Optimized and 
refactored for the YouTube Auto Dub pipeline requirements.

Author: Nguyen Cong Thuan Huy (mangodxd)
Version: 1.0.0
"""

import json
import re
import httpx
from urllib.parse import quote
from bs4 import BeautifulSoup
from browserforge.headers import HeaderGenerator

class GoogleTranslator:
    """
    A unified Google Translator that attempts to use the internal 'batchexecute' API (RPC)
    first, and falls back to web scraping the mobile site if that fails.
    """
    def __init__(self, proxy=None):
        self.base_url_rpc = "https://translate.google.com/_/TranslateWebserverUi/data/batchexecute"
        self.base_url_scrape = "https://translate.google.com/m"
        
        # Browserforge headers are crucial for the RPC method to look legitimate
        self.headers = HeaderGenerator().generate()
        
        # HTTP client config
        self.proxy = proxy
        self.client = httpx.Client(proxy=self.proxy, timeout=10)
        
        # State for RPC method
        self.bl = None  # Build label token

    def _refresh_rpc_token(self):
        """Refreshes the 'cfb2h' token required for the RPC interface."""
        try:
            response = self.client.get("https://translate.google.com/", headers=self.headers)
            bl_match = re.search(r'"cfb2h":"(.*?)"', response.text)
            if bl_match:
                self.bl = bl_match.group(1)
            else:
                # Fallback to a hardcoded known-working build label if regex fails
                self.bl = "boq_translate-webserver_20251215.06_p0"
        except Exception as e:
            print(f"[Warning] Token refresh failed: {e}. Using fallback.")
            self.bl = "boq_translate-webserver_20251215.06_p0"

    def _parse_rpc_response(self, raw_text):
        """Parses the nested JSON response from the RPC endpoint."""
        try:
            # 1. Isolate the specific array containing the translation data
            match = re.search(r'\["wrb.fr","MkEWBc","(.*?)",null,null,null,"generic"\]', raw_text, re.DOTALL)
            if not match:
                raise ValueError("Could not find translation data in RPC response.")

            # 2. Unescape double-serialized JSON string
            inner_json_str = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            data = json.loads(inner_json_str)
            
            # 3. Navigate deep nested structure: [1] Result -> [0] Group -> [0] Entry -> [5] Sentences
            translation_parts = data[1][0][0][5]
            
            # 4. Join parts (handles multiline/long text)
            final_text = " ".join([part[0] for part in translation_parts if part[0]])
            return final_text
        except Exception as e:
            raise ValueError(f"RPC Parse Error: {e}")

    def _translate_rpc(self, text, source, target):
        """Method 1: Internal API (batchexecute). Higher quality, mimics browser app."""
        if not self.bl:
            self._refresh_rpc_token()
        
        # Prepare RPC arguments
        rpc_arg = json.dumps([[text, source, target, True, [1]]], ensure_ascii=False)
        f_req = json.dumps([[["MkEWBc", rpc_arg, None, "generic"]]])

        params = {
            "rpcids": "MkEWBc",
            "bl": self.bl,
            "hl": "en", # GUI language
            "rt": "c"
        }
        
        response = self.client.post(
            self.base_url_rpc, 
            headers=self.headers, 
            params=params, 
            data={"f.req": f_req}
        )

        if response.status_code != 200:
            raise Exception(f"RPC HTTP Error: {response.status_code}")
        
        return self._parse_rpc_response(response.text)

    def _translate_scrape(self, text, source, target):
        """Method 2: Web Scraping (Mobile Site). Simple fallback."""
        params = {
            "sl": source,
            "tl": target,
            "q": text
        }
        
        # Use existing client but ensure headers allow scraping (sometimes strict User-Agents fail on mobile site)
        # We try with the existing generated headers first.
        response = self.client.get(self.base_url_scrape, params=params, headers=self.headers)
        
        if response.status_code == 429:
            raise Exception("Too Many Requests (429)")
        if response.status_code != 200:
            raise Exception(f"Scrape HTTP Error: {response.status_code}")

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Try standard result container
        element = soup.find("div", {"class": "t0"})
        if not element:
            # Try alternate container
            element = soup.find("div", {"class": "result-container"})
        
        if not element:
            raise Exception("Could not find translation element in HTML.")
            
        return element.get_text(strip=True)

    def translate(self, text, source="auto", target="vi"):
        """
        Main interface. Tries RPC first, falls back to Scraping.
        """
        if not text:
            return ""
            
        # Strategy 1: RPC
        try:
            return self._translate_rpc(text, source, target)
        except Exception as e:
            # Silent fallback to scraping
            pass

        # Strategy 2: Scrape Fallback
        try:
            return self._translate_scrape(text, source, target)
        except Exception as e:
            return f"Error: All translation methods failed. Last error: {e}"

    def close(self):
        self.client.close()

# ==========================================
# USAGE
# ==========================================
if __name__ == "__main__":
    t = GoogleTranslator()
    
    sample_text = "Programming is about logic, not just syntax."
    
    print(f"Original: {sample_text}")
    
    # Test 1: English to Vietnamese
    res_vi = t.translate(sample_text, source="auto", target="vi")
    print(f"Vietnamese: {res_vi}")
    
    # Test 2: English to Japanese
    res_ja = t.translate(sample_text, source="en", target="ja")
    print(f"Japanese:   {res_ja}")
    
    t.close()