import logging
import time

import pandas as pd
import requests
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CSV_FILES, OLLAMA_BASE_URL, MODELS, LLM_OUTPUT_LIMITS, HTTP_RETRY

logger = logging.getLogger(__name__)


class DisasterCSVTool:
    def __init__(self):
        self.dataframes = {}
        self._load_csvs()

    def _load_csvs(self):
        for name, path in CSV_FILES.items():
            if Path(path).exists():
                df = pd.read_csv(path, low_memory=False)
                df.columns = [c.strip().lower().replace(" ", "_").replace("'", "") for c in df.columns]
                self.dataframes[name] = df

    def get_schema_summary(self) -> str:
        lines = []
        for name, df in self.dataframes.items():
            lines.append(f"\nDataset: {name} ({len(df)} rows)")
            lines.append(f"Columns: {', '.join(df.columns.tolist())}")
            for col in df.columns[:10]:
                dtype = str(df[col].dtype)
                sample = df[col].dropna().head(3).tolist()
                lines.append(f"  - {col} ({dtype}): {sample}")
        return "\n".join(lines)

    def _ask_ollama_for_code(self, question: str) -> str:
        schema = self.get_schema_summary()
        prompt = f"""You are a data analyst. Given these pandas DataFrames, write Python code to answer the question.

Available DataFrames (already loaded as self.dataframes[name]):
{schema}

Question: {question}

Rules:
- Use self.dataframes['global_response'], self.dataframes['emdat_1970'], or self.dataframes['emdat_1900']
- Return ONLY a Python expression or short code block that produces the answer
- The result should be assigned to a variable called `result`
- Use pandas operations only
- Do NOT import anything
- Do NOT use print()
- Keep it simple and safe - no exec, eval, os, or system calls

Example: result = self.dataframes['global_response'].groupby('disaster_type')['casualties'].sum().sort_values(ascending=False).head(5)

Code:"""

        last_exc = None
        for attempt in range(1, HTTP_RETRY["attempts"] + 1):
            try:
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": MODELS["chat"]["name"],
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.1,
                        "options": {"num_predict": LLM_OUTPUT_LIMITS["chat_num_predict"]},
                    },
                    timeout=HTTP_RETRY["request_timeout"],
                )
                response.raise_for_status()
                return response.json()["response"]
            except requests.RequestException as e:
                last_exc = e
                sleep = HTTP_RETRY["backoff_seconds"] * (2 ** (attempt - 1))
                logger.warning("ollama code-gen attempt=%d/%d failed: %s — backoff %.1fs",
                               attempt, HTTP_RETRY["attempts"], e, sleep)
                if attempt < HTTP_RETRY["attempts"]:
                    time.sleep(sleep)
        raise last_exc

    def _extract_code(self, llm_response: str) -> str:
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", llm_response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        lines = []
        for line in llm_response.strip().split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("```"):
                lines.append(stripped)
        return "\n".join(lines)

    def _safe_execute(self, code: str) -> str:
        forbidden = ["import ", "exec(", "eval(", "os.", "sys.", "subprocess", "__", "open(", "file"]
        for f in forbidden:
            if f in code:
                return f"Error: unsafe operation detected ({f})"

        safe_builtins = {
            "len": len, "sum": sum, "min": min, "max": max, "abs": abs,
            "round": round, "sorted": sorted, "list": list, "dict": dict,
            "str": str, "int": int, "float": float, "bool": bool,
            "range": range, "enumerate": enumerate, "zip": zip,
            "True": True, "False": False, "None": None,
            "print": lambda *a, **k: None,
        }
        local_vars = {"self": self, "pd": pd}
        try:
            exec(code, {"__builtins__": safe_builtins}, local_vars)
            result = local_vars.get("result", "No result variable found")
            if isinstance(result, pd.DataFrame):
                return result.to_string(max_rows=20)
            elif isinstance(result, pd.Series):
                return result.to_string()
            return str(result)
        except Exception as e:
            return f"Error executing query: {e}"

    def query(self, question: str) -> str:
        t0 = time.time()
        try:
            llm_response = self._ask_ollama_for_code(question)
            code = self._extract_code(llm_response)
            result = self._safe_execute(code)
            logger.info("csv_tool.query ok latency_ms=%d code_len=%d",
                        int((time.time() - t0) * 1000), len(code))
            return f"Query: {question}\nGenerated Code:\n{code}\n\nResult:\n{result}"
        except Exception as e:
            logger.exception("csv_tool.query failed: %s", e)
            return f"Error processing query: {e}"

    def get_available_datasets(self) -> str:
        info = []
        for name, df in self.dataframes.items():
            info.append(f"- {name}: {len(df)} rows, {len(df.columns)} columns")
        return "\n".join(info)

    def get_disaster_stats(self, dataset: str = "global_response") -> str:
        if dataset not in self.dataframes:
            return f"Dataset '{dataset}' not found. Available: {list(self.dataframes.keys())}"
        df = self.dataframes[dataset]
        stats = []
        if "disaster_type" in df.columns:
            stats.append(f"Disaster types: {df['disaster_type'].nunique()}")
            stats.append(f"Type distribution:\n{df['disaster_type'].value_counts().head(10).to_string()}")
        if "country" in df.columns:
            stats.append(f"Countries: {df['country'].nunique()}")
        if "year" in df.columns:
            stats.append(f"Year range: {df['year'].min()} - {df['year'].max()}")
        elif "date" in df.columns:
            stats.append(f"Date range: {df['date'].min()} - {df['date'].max()}")
        return "\n".join(stats)
