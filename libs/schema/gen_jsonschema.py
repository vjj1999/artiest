"""生成 ResultEnvelope 的 JSON Schema 文件"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from libs.schema.result_envelope import ResultEnvelope

schema = ResultEnvelope.model_json_schema()
out_path = Path(__file__).parent / "result_envelope.jsonschema"
out_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"已生成: {out_path}")
