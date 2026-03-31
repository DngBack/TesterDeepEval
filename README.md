# TesterDeepEval

Bo tool Python de danh gia chat luong RAG va Agent bang DeepEval.
Muc tieu:
- Nhap du lieu tu CSV de de map voi du lieu production.
- Xuat ket qua dang so (0.0 -> 1.0) cho tung test case.
- Co file tong hop de theo doi trend chat luong.

---

## 1) Yeu cau he thong

- Python: khuyen nghi `3.10+` (da test voi Python 3.12).
- OS: Linux / macOS / Windows.
- Co API key cho LLM judge (vi du OpenAI), vi DeepEval cham diem theo LLM-as-a-judge.

---

## 2) Cai Python (neu may chua co)

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
python3 --version
```

### macOS (Homebrew)

```bash
brew install python
python3 --version
```

### Windows (PowerShell)

```powershell
winget install Python.Python.3.12
python --version
```

Neu lenh `python` khong nhan, dung `py` hoac mo terminal moi.

---

## 3) Cai dat project (chi tiet tung buoc)

Chay tai thu muc project:

```bash
cd /path/to/TesterDeepEval
```

### Buoc 1: Tao virtual environment

Linux/macOS:
```bash
python3 -m venv .venv
```

Windows:
```powershell
python -m venv .venv
```

### Buoc 2: Kich hoat virtual environment

Linux/macOS:
```bash
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

### Buoc 3: Cai dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Buoc 4: Cau hinh API key

Project da co:
- `.env.example` (mau)
- `.env` (file de ban dien key)

Ban sua `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Luu y:
- Script tu dong load `.env`, khong bat buoc phai `export` thu cong.
- Cung co the set tam thoi qua env:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

---

## 4) Cau truc file quan trong

- `scripts/eval_rag_deepeval.py`: danh gia RAG.
- `scripts/eval_agent_deepeval.py`: danh gia Agent.
- `scripts/run_all_evals.py`: chay ca RAG + Agent.
- `data/rag_input_sample.csv`: du lieu mau RAG.
- `data/agent_input_sample.csv`: du lieu mau Agent.
- `outputs/`: noi luu ket qua.

---

## 5) Danh gia RAG

### 5.1 Input CSV bat buoc

File mau: `data/rag_input_sample.csv`

Cot bat buoc:
- `input`: cau hoi cua user.
- `actual_output`: cau tra loi thuc te cua he thong.
- `expected_output`: cau tra loi ky vong.
- `retrieval_context`: danh sach context truy hoi (JSON list string).

Cot tuy chon:
- `id`: dinh danh test case.

Vi du `retrieval_context` dung:
- `["Doan context 1", "Doan context 2"]`

### 5.2 Lenh chay

```bash
python scripts/eval_rag_deepeval.py \
  --input data/rag_input_sample.csv \
  --output outputs/rag_scores.csv \
  --summary outputs/rag_summary.csv \
  --threshold 0.5
```

### 5.3 Cac metrics RAG (chi tiet)

- `answer_relevancy`
  - Do muc do cau tra loi co lien quan den cau hoi hay khong.
  - Cao khi tra loi dung trong tam cau hoi, thap khi lan man/lech de.

- `faithfulness`
  - Do muc do cau tra loi co trung thuc voi `retrieval_context`.
  - Phat hien hallucination: output co thong tin khong co trong context.

- `contextual_precision`
  - Danh gia chat luong context duoc retrieve theo huong "precision".
  - Cao khi context lien quan duoc dua len uu tien, it nhieu.

- `contextual_recall`
  - Danh gia muc do context retrieve bao phu du thong tin can cho cau tra loi dung.
  - Cao khi context khong bo sot thong tin quan trong.

- `contextual_relevancy`
  - Danh gia tong quan do lien quan cua bo context retrieve voi query.
  - Cao khi phan lon context phuc vu truc tiep cho cau hoi.

- `overall_score`
  - Trung binh cac metric co diem hop le tren moi dong.

---

## 6) Danh gia Agent

### 6.1 Input CSV bat buoc

File mau: `data/agent_input_sample.csv`

Cot bat buoc:
- `input`: yeu cau cua user.
- `actual_output`: output thuc te cua agent.
- `expected_output`: output ky vong.
- `tools_called`: danh sach tool agent da goi (JSON list string).
- `expected_tools`: danh sach tool agent nen goi (JSON list string).

Cot tuy chon:
- `id`: dinh danh test case.

Vi du:
- `tools_called = ["WeatherAPI", "Calculator"]`
- `expected_tools = ["WeatherAPI"]`

### 6.2 Lenh chay

```bash
python scripts/eval_agent_deepeval.py \
  --input data/agent_input_sample.csv \
  --output outputs/agent_scores.csv \
  --summary outputs/agent_summary.csv \
  --threshold 0.5
```

### 6.3 Cac metrics Agent (chi tiet)

- `answer_relevancy`
  - Output cuoi cung cua agent co tra loi dung de bai user khong.

- `tool_correctness`
  - Agent co goi dung tool can thiet khong.
  - So sanh `tools_called` va `expected_tools`.
  - Phat hien goi thieu tool hoac goi sai tool.

- `task_completion` (GEval)
  - Danh gia muc do hoan thanh task dua tren:
    - `input`
    - `actual_output`
    - `expected_output`
  - Cao khi ket qua dap ung dung yeu cau va muc tieu.

- `overall_score`
  - Trung binh cac metric co diem hop le tren moi dong.

---

## 7) File output va cach doc ket qua

### 7.1 File chi tiet: `outputs/*_scores.csv`

Moi dong la 1 test case, gom:
- Diem metric: `0.0 -> 1.0`
- Cot pass/fail: `*_passed` (dua theo `--threshold`)
- Giai thich: `*_reason`
- `overall_score`

Vi du cach hieu nhanh:
- `>= 0.8`: tot
- `0.5 - 0.79`: tam on, can xem `reason`
- `< 0.5`: yeu, nen uu tien fix

### 7.2 File tong hop: `outputs/*_summary.csv`

Chi 1 dong tong hop:
- `rows`: tong so test case
- `avg_<metric>`: diem trung binh metric
- `avg_overall_score`
- `pass_rate_<metric>`: ty le pass theo threshold

---

## 8) Chay nhanh ca 2 bo danh gia

```bash
python scripts/run_all_evals.py \
  --rag-input data/rag_input_sample.csv \
  --agent-input data/agent_input_sample.csv \
  --threshold 0.5
```

Co the them model cham diem:

```bash
python scripts/run_all_evals.py \
  --rag-input data/rag_input_sample.csv \
  --agent-input data/agent_input_sample.csv \
  --threshold 0.5 \
  --model gpt-4.1
```

---

## 9) Quy uoc du lieu CSV de tranh loi

- Nen dung JSON list cho cac cot list:
  - `retrieval_context`
  - `tools_called`
  - `expected_tools`
- Script co fallback parse theo dau `||` neu du lieu khong phai JSON.
- Neu 1 metric loi o 1 dong, script van chay tiep:
  - Diem metric do se de trong (`None`)
  - `*_reason` se ghi `metric_error: ...`

---

## 10) Troubleshooting nhanh

- Loi `OpenAI API key is not configured`
  - Kiem tra `.env` da co `OPENAI_API_KEY=...`
  - Hoac `export OPENAI_API_KEY=...` roi chay lai.

- Loi `python: command not found`
  - Dung `python3` tren Linux/macOS.

- Loi khong kich hoat duoc venv tren Windows
  - Dung PowerShell voi:
  - `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
  - roi chay `.venv\Scripts\Activate.ps1`.

- Loi thieu cot CSV
  - Doi ten header dung theo muc 5.1 (RAG) va 6.1 (Agent).

---

## 11) Custom prompt khi danh gia (DeepEval)

Da ho tro custom prompt/criteria theo tung metric qua file JSON:
- `scripts/eval_rag_deepeval.py`: them `--prompt-config`
- `scripts/eval_agent_deepeval.py`: them `--prompt-config`
- `scripts/run_all_evals.py`: them `--prompt-config` (ap dung cho ca 2), hoac rieng:
  - `--rag-prompt-config`
  - `--agent-prompt-config`

File mau: `data/prompt_config_sample.json`

Format:
- Co the dat theo section `rag` va `agent`.
- Moi metric co the la:
  - String: dung lam `criteria`.
  - Object:
    - `criteria`: chuoi tieu chi cham diem.
    - `evaluation_steps`: list cac buoc cham diem (tuy chon).

Vi du chay RAG voi prompt custom:

```bash
python scripts/eval_rag_deepeval.py \
  --input data/rag_input_sample.csv \
  --output outputs/rag_scores.csv \
  --summary outputs/rag_summary.csv \
  --threshold 0.5 \
  --prompt-config data/prompt_config_sample.json
```

Vi du chay ca RAG + Agent voi 1 file config:

```bash
python scripts/run_all_evals.py \
  --rag-input data/rag_input_sample.csv \
  --agent-input data/agent_input_sample.csv \
  --threshold 0.5 \
  --prompt-config data/prompt_config_sample.json
```
