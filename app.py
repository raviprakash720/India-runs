import os
import json
import uuid
import threading
import traceback
from flask import Flask, request, jsonify, render_template, send_from_directory
from rank import main as run_ranking

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Global state to keep track of tasks
tasks = {}


def process_candidates_task(task_id, filepath, csv_out, xlsx_out):
    tasks[task_id]["status"] = "processing"

    def log_progress(msg):
        tasks[task_id]["logs"].append(msg)
        if len(tasks[task_id]["logs"]) > 80:
            tasks[task_id]["logs"].pop(0)

    try:
        run_ranking(
            candidates_path=filepath,
            output_csv_path=csv_out,
            output_xlsx_path=xlsx_out,
            progress_callback=log_progress,
        )
        tasks[task_id]["status"] = "completed"
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tb = traceback.format_exc()
        tasks[task_id]["logs"].append(f"ERROR: {str(e)}")
        tasks[task_id]["logs"].append(tb)
        print(f"[TASK {task_id}] FAILED: {e}")
        print(tb)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    task_id = str(uuid.uuid4())
    filename = f"{task_id}_candidates.jsonl"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    csv_out = os.path.join(app.config["OUTPUT_FOLDER"], f"{task_id}_submission.csv")
    xlsx_out = os.path.join(app.config["OUTPUT_FOLDER"], f"{task_id}_submission.xlsx")

    tasks[task_id] = {
        "status": "queued",
        "logs": ["File uploaded successfully. Initializing pipeline..."],
        "csv_filename": f"{task_id}_submission.csv",
        "xlsx_filename": f"{task_id}_submission.xlsx",
        "detail_filename": f"{task_id}_submission_detail.json",
    }

    t = threading.Thread(
        target=process_candidates_task,
        args=(task_id, filepath, csv_out, xlsx_out),
    )
    t.daemon = True
    t.start()

    return jsonify({"task_id": task_id})


@app.route("/status/<task_id>", methods=["GET"])
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(tasks[task_id])


@app.route("/results/<task_id>", methods=["GET"])
def get_results(task_id):
    """Return the enriched top-100 candidate data for the interactive table."""
    if task_id not in tasks:
        return jsonify({"error": "Task not found"}), 404
    if tasks[task_id]["status"] != "completed":
        return jsonify({"error": "Task not completed yet"}), 202

    detail_path = os.path.join(
        app.config["OUTPUT_FOLDER"], tasks[task_id]["detail_filename"]
    )
    if not os.path.exists(detail_path):
        return jsonify({"error": "Results file not found"}), 404

    with open(detail_path, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    return send_from_directory(
        app.config["OUTPUT_FOLDER"], filename, as_attachment=True
    )


if __name__ == "__main__":
    # use_reloader=False prevents Flask from restarting the server when
    # output files (CSV/XLSX/JSON) are written during background processing.
    # Without this, the watchdog kills the ranking thread mid-run and wipes
    # the in-memory tasks dict, causing the frontend to poll a dead task ID.
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
