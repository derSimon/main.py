from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import uuid
import asyncio
import os
import json
import tempfile
import httpx

from openai import AsyncOpenAI
import anthropic
import cloudinary
import cloudinary.uploader

# ─── API Clients ───────────────────────────────
whisper_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = "youtube-media-downloader.p.rapidapi.com"

cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}

class VideoRequest(BaseModel):
    url: str

def is_valid_youtube_url(url: str) -> bool:
    pattern = r"(https?://)?(www\.)?(youtube\.com|youtu\.be)(/|$)"
    return bool(re.match(pattern, url))

def extract_video_id(url: str) -> str:
    patterns = [
        r"youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
        r"youtu\.be/([a-zA-Z0-9_-]+)",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise Exception("Video ID konnte nicht extrahiert werden")


# ─────────────────────────────────────────────
# SCHRITT 1: Video-Details + Download-URLs holen
# ─────────────────────────────────────────────
async def get_video_details(video_id: str) -> dict:
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    params = {
        "videoId": video_id,
        "urlAccess": "normal",
        "videos": "auto",
        "audios": "auto"
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"https://{RAPIDAPI_HOST}/v2/video/details",
            headers=headers,
            params=params
        )
        data = response.json()
        print(f"[get_video_details] status={response.status_code}")
        print(f"[get_video_details] keys={list(data.keys())}")
        return data


# ─────────────────────────────────────────────
# SCHRITT 1a: Audio herunterladen
# ─────────────────────────────────────────────
async def download_audio(video_details: dict, output_dir: str) -> str:
    audio_url = None
    audios = video_details.get("audios", [])
    print(f"[download_audio] audios type={type(audios)}, count={len(audios)}")
    if audios:
        first = audios[0]
        print(f"[download_audio] first audio type={type(first)}, value={str(first)[:200]}")
        for audio in audios:
            # Kann ein Dict oder ein String sein
            if isinstance(audio, dict):
                url = audio.get("url")
            elif isinstance(audio, str):
                url = audio
            else:
                continue
            if url and url.startswith("http"):
                audio_url = url
                break

    if not audio_url:
        raise Exception(f"Keine Audio-URL gefunden. audios={str(audios)[:300]}")

    print(f"[download_audio] Downloading from URL...")
    audio_path = os.path.join(output_dir, "audio.mp3")

    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        async with client.stream("GET", audio_url) as stream:
            with open(audio_path, "wb") as f:
                async for chunk in stream.aiter_bytes(chunk_size=8192):
                    f.write(chunk)

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise Exception("Audio-Download fehlgeschlagen – Datei leer")

    print(f"[download_audio] Done: {os.path.getsize(audio_path)} bytes")
    return audio_path


# ─────────────────────────────────────────────
# SCHRITT 1b: Video herunterladen
# ─────────────────────────────────────────────
async def download_video(video_details: dict, output_dir: str) -> str:
    video_url = None
    videos = video_details.get("videos", [])
    print(f"[download_video] videos type={type(videos)}, count={len(videos)}")
    if videos:
        first = videos[0]
        print(f"[download_video] first video type={type(first)}, value={str(first)[:200]}")
        # Versuche nach Qualität zu sortieren wenn möglich
        try:
            sorted_videos = sorted(
                videos,
                key=lambda x: x.get("height", 0) if isinstance(x, dict) else 0,
                reverse=True
            )
        except Exception:
            sorted_videos = videos
        for video in sorted_videos:
            if isinstance(video, dict):
                url = video.get("url")
                height = video.get("height", "?")
            elif isinstance(video, str):
                url = video
                height = "?"
            else:
                continue
            if url and url.startswith("http"):
                video_url = url
                print(f"[download_video] Selected quality: {height}p")
                break

    if not video_url:
        raise Exception(f"Keine Video-URL gefunden. videos={str(videos)[:300]}")

    print(f"[download_video] Downloading from URL...")
    video_path = os.path.join(output_dir, "video.mp4")

    async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
        async with client.stream("GET", video_url) as stream:
            with open(video_path, "wb") as f:
                async for chunk in stream.aiter_bytes(chunk_size=8192):
                    f.write(chunk)

    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        raise Exception("Video-Download fehlgeschlagen – Datei leer")

    print(f"[download_video] Done: {os.path.getsize(video_path)} bytes")
    return video_path


# ─────────────────────────────────────────────
# SCHRITT 2: Transkription via Whisper
# ─────────────────────────────────────────────
async def transcribe_audio(audio_path: str) -> list:
    with open(audio_path, "rb") as f:
        transcript = await whisper_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return transcript.segments


# ─────────────────────────────────────────────
# SCHRITT 3: Beste Momente via Claude
# ─────────────────────────────────────────────
async def find_best_moments(segments: list) -> list:
    transcript_text = "\n".join(
        [f"[{s.start:.1f}s - {s.end:.1f}s]: {s.text}" for s in segments]
    )
    prompt = f"""Du bist ein Experte für virale Kurzvideos (TikTok, Reels, YouTube Shorts).

Analysiere dieses Transkript und wähle die 3 besten Momente für 60-Sekunden-Shorts aus.
Achte auf: starke Aussagen, emotionale Momente, klare Tipps, überraschende Fakten, gute Hooks.

Transkript:
{transcript_text}

Antworte NUR mit einem JSON-Array, ohne weitere Erklärungen, ohne Markdown:
[
  {{"start": 12.5, "end": 68.0, "reason": "Starker Hook über..."}},
  {{"start": 145.0, "end": 203.0, "reason": "Emotionaler Moment..."}},
  {{"start": 312.0, "end": 371.0, "reason": "Klarer Tipp der..."}}
]"""

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: claude_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


# ─────────────────────────────────────────────
# SCHRITT 4: Clips schneiden (FFmpeg 9:16 Smart Crop)
# ─────────────────────────────────────────────
async def cut_clips(video_path: str, moments: list, output_dir: str) -> list:
    clip_paths = []
    for i, moment in enumerate(moments):
        start = moment["start"]
        duration = moment["end"] - moment["start"]
        out_path = os.path.join(output_dir, f"clip_{i+1}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-vf", "scale='if(gt(iw/ih,9/16),1080,trunc(oh*a/2)*2)':'if(gt(iw/ih,9/16),trunc(ow/a/2)*2,1920)',crop=1080:1920:(iw-1080)/2:(ih-1920)/2",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            out_path
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        if os.path.exists(out_path):
            clip_paths.append(out_path)
    return clip_paths


# ─────────────────────────────────────────────
# SCHRITT 5: Clips zu Cloudinary hochladen
# ─────────────────────────────────────────────
async def upload_to_cloudinary(clip_paths: list, moments: list) -> list:
    loop = asyncio.get_event_loop()
    clips_data = []
    for i, clip_path in enumerate(clip_paths):
        result = await loop.run_in_executor(
            None,
            lambda p=clip_path, idx=i: cloudinary.uploader.upload(
                p,
                resource_type="video",
                folder="shorts_generator",
                public_id=f"clip_{uuid.uuid4().hex[:8]}",
            )
        )
        clips_data.append({
            "clip_number": i + 1,
            "url": result["secure_url"],
            "public_id": result["public_id"],
            "reason": moments[i]["reason"] if i < len(moments) else "",
            "duration": round(moments[i]["end"] - moments[i]["start"], 1) if i < len(moments) else 0
        })
    return clips_data


# ─────────────────────────────────────────────
# HAUPT-WORKFLOW
# ─────────────────────────────────────────────
async def process_video(job_id: str, url: str):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:

            jobs[job_id]["status"] = "downloading"
            jobs[job_id]["message"] = "⬇️ Video-Details werden geladen..."
            video_id = extract_video_id(url)
            video_details = await get_video_details(video_id)

            jobs[job_id]["message"] = "⬇️ Audio wird heruntergeladen..."
            audio_path = await download_audio(video_details, tmpdir)

            jobs[job_id]["message"] = "⬇️ Video wird heruntergeladen..."
            video_path = await download_video(video_details, tmpdir)

            jobs[job_id]["status"] = "transcribing"
            jobs[job_id]["message"] = "🎙️ Audio wird transkribiert (Whisper)..."
            segments = await transcribe_audio(audio_path)

            jobs[job_id]["status"] = "analyzing"
            jobs[job_id]["message"] = "🤖 Claude analysiert beste Momente..."
            moments = await find_best_moments(segments)

            jobs[job_id]["status"] = "cutting"
            jobs[job_id]["message"] = f"✂️ {len(moments)} Clips werden geschnitten..."
            clips = await cut_clips(video_path, moments, tmpdir)

            jobs[job_id]["status"] = "uploading"
            jobs[job_id]["message"] = "☁️ Clips werden zu Cloudinary hochgeladen..."
            clips_data = await upload_to_cloudinary(clips, moments)

            jobs[job_id]["status"] = "done"
            jobs[job_id]["message"] = f"✅ {len(clips_data)} Clips bereit zur Vorschau!"
            jobs[job_id]["clips"] = clips_data

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = f"❌ Fehler: {str(e)}"


HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Shorts Generator</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{--bg:#0a0a0a;--surface:#111;--border:#1e1e1e;--accent:#ff3d3d;--accent2:#ff8c42;--text:#f0ede8;--muted:#666;--success:#3dff8f;--warning:#ffd60a}
  body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;display:flex;align-items:flex-start;justify-content:center;padding:40px 24px}
  .container{position:relative;z-index:1;width:100%;max-width:680px}
  .logo{font-family:'Syne',sans-serif;font-weight:800;font-size:13px;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);margin-bottom:48px}
  h1{font-family:'Syne',sans-serif;font-weight:800;font-size:clamp(32px,7vw,52px);line-height:1.0;margin-bottom:12px;letter-spacing:-.02em}
  h1 span{display:block;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
  .subtitle{color:var(--muted);font-size:15px;font-weight:300;margin-bottom:40px;line-height:1.6}
  .input-group{display:flex;flex-direction:column;gap:12px;margin-bottom:24px}
  .input-wrapper{position:relative;display:flex;align-items:center}
  .yt-icon{position:absolute;left:16px;width:20px;height:20px;opacity:.5;color:var(--text)}
  input[type=text]{width:100%;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 20px 18px 48px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:15px;outline:none;transition:border-color .2s,box-shadow .2s}
  input[type=text]::placeholder{color:var(--muted)}
  input[type=text]:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(255,61,61,.1)}
  .btn-primary{width:100%;padding:18px;background:var(--accent);color:#fff;border:none;border-radius:12px;font-family:'Syne',sans-serif;font-weight:700;font-size:15px;letter-spacing:.05em;text-transform:uppercase;cursor:pointer;transition:transform .15s,box-shadow .15s}
  .btn-primary:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 8px 32px rgba(255,61,61,.35)}
  .btn-primary:disabled{opacity:.5;cursor:not-allowed}
  .status-card{margin-top:24px;background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:24px;display:none;animation:slideUp .3s ease}
  .status-card.visible{display:block}
  @keyframes slideUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
  .status-label{font-size:11px;font-weight:500;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
  .status-message{font-family:'Syne',sans-serif;font-size:18px;font-weight:700;margin-bottom:20px;min-height:28px}
  .progress-track{height:3px;background:var(--border);border-radius:99px;overflow:hidden;margin-bottom:16px}
  .progress-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:99px;width:0%;transition:width .6s ease}
  .steps{display:flex;flex-direction:column;gap:8px}
  .step{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--muted);transition:color .3s}
  .step.active{color:var(--text)}.step.done{color:var(--success)}
  .step-dot{width:6px;height:6px;border-radius:50%;background:var(--border);flex-shrink:0;transition:background .3s}
  .step.active .step-dot{background:var(--accent);box-shadow:0 0 8px var(--accent)}
  .step.done .step-dot{background:var(--success)}
  .preview-section{margin-top:32px;display:none}
  .preview-section.visible{display:block;animation:slideUp .4s ease}
  .preview-title{font-family:'Syne',sans-serif;font-weight:800;font-size:22px;margin-bottom:6px}
  .preview-subtitle{color:var(--muted);font-size:13px;margin-bottom:24px}
  .clips-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}
  .clip-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;overflow:hidden;transition:border-color .2s}
  .clip-card:hover{border-color:#333}
  .video-wrapper{position:relative;background:#000;aspect-ratio:9/16}
  .video-wrapper video{width:100%;height:100%;object-fit:cover}
  .clip-info{padding:14px}
  .clip-label{font-family:'Syne',sans-serif;font-weight:700;font-size:13px;margin-bottom:4px}
  .clip-duration{font-size:11px;color:var(--muted);margin-bottom:8px}
  .clip-reason{font-size:12px;color:var(--muted);line-height:1.5;margin-bottom:14px;min-height:36px}
  .clip-actions{display:flex;gap:8px}
  .btn-approve{flex:1;padding:10px;background:var(--success);color:#000;border:none;border-radius:8px;font-family:'Syne',sans-serif;font-weight:700;font-size:12px;letter-spacing:.04em;text-transform:uppercase;cursor:pointer;transition:opacity .2s}
  .btn-approve:hover{opacity:.85}
  .btn-approve:disabled{opacity:.4;cursor:not-allowed}
  .btn-reject{padding:10px 12px;background:transparent;color:var(--muted);border:1px solid var(--border);border-radius:8px;font-family:'Syne',sans-serif;font-weight:700;font-size:12px;cursor:pointer;transition:all .2s}
  .btn-reject:hover{border-color:var(--accent);color:var(--accent)}
  .btn-reject:disabled{opacity:.4;cursor:not-allowed}
  .clip-status{font-size:11px;font-weight:500;text-align:center;padding:6px;border-radius:6px;margin-top:8px;display:none}
  .clip-status.uploading{display:block;background:rgba(255,214,10,.1);color:var(--warning)}
  .clip-status.uploaded{display:block;background:rgba(61,255,143,.1);color:var(--success)}
  .clip-status.rejected{display:block;background:rgba(255,61,61,.1);color:var(--accent)}
  .error-msg{color:var(--accent);font-size:13px;margin-top:8px;display:none}
  .footer{margin-top:48px;font-size:12px;color:#333;text-align:center;padding-bottom:40px}
</style>
</head>
<body>
<div class="container">
  <div class="logo">◆ Shorts Generator</div>
  <h1>Dreh jedes <span>Video in Shorts.</span></h1>
  <p class="subtitle">YouTube-Link einfügen – Claude findet die besten Momente, du entscheidest welche gepostet werden.</p>
  <div class="input-group">
    <div class="input-wrapper">
      <svg class="yt-icon" viewBox="0 0 24 24" fill="none"><path d="M22.54 6.42C22.42 5.95 22.18 5.51 21.84 5.16C21.5 4.81 21.07 4.55 20.6 4.42C18.88 4 12 4 12 4C12 4 5.12 4 3.4 4.46C2.93 4.59 2.5 4.85 2.16 5.2C1.82 5.55 1.58 5.99 1.46 6.46C1.15 8.21 0.99 9.98 1 11.75C0.99 13.54 1.14 15.32 1.46 17.08C1.59 17.54 1.84 17.96 2.18 18.29C2.52 18.63 2.94 18.87 3.4 19C5.12 19.46 12 19.46 12 19.46C12 19.46 18.88 19.46 20.6 19C21.07 18.87 21.5 18.61 21.84 18.26C22.18 17.91 22.42 17.47 22.54 17C22.85 15.27 23.01 13.51 23 11.75C23.01 9.96 22.85 8.18 22.54 6.42Z" fill="currentColor"/><path d="M9.75 15.02L15.5 11.75L9.75 8.48V15.02Z" fill="#0a0a0a"/></svg>
      <input type="text" id="urlInput" placeholder="https://youtube.com/watch?v=..." />
    </div>
    <button class="btn-primary" id="startBtn" onclick="startWorkflow()">Workflow starten →</button>
    <p class="error-msg" id="errorMsg">⚠️ Bitte eine gültige YouTube-URL eingeben.</p>
  </div>
  <div class="status-card" id="statusCard">
    <div class="status-label">Status</div>
    <div class="status-message" id="statusMessage">🚀 Workflow gestartet!</div>
    <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
    <div class="steps">
      <div class="step" id="step-started"><span class="step-dot"></span>Workflow gestartet</div>
      <div class="step" id="step-downloading"><span class="step-dot"></span>Video herunterladen</div>
      <div class="step" id="step-transcribing"><span class="step-dot"></span>Audio transkribieren (Whisper)</div>
      <div class="step" id="step-analyzing"><span class="step-dot"></span>Claude analysiert Momente</div>
      <div class="step" id="step-cutting"><span class="step-dot"></span>Clips schneiden (FFmpeg)</div>
      <div class="step" id="step-uploading"><span class="step-dot"></span>Zu Cloudinary hochladen</div>
      <div class="step" id="step-done"><span class="step-dot"></span>Vorschau bereit!</div>
    </div>
  </div>
  <div class="preview-section" id="previewSection">
    <div class="preview-title">🎬 Deine Clips</div>
    <p class="preview-subtitle">Schau dir jeden Clip an und entscheide selbst was gepostet wird.</p>
    <div class="clips-grid" id="clipsGrid"></div>
  </div>
  <div class="footer">Powered by FastAPI · Railway · Whisper · Claude · Cloudinary</div>
</div>
<script>
  const STEPS=['started','downloading','transcribing','analyzing','cutting','uploading','done'];
  const PROGRESS={started:5,downloading:20,transcribing:40,analyzing:60,cutting:78,uploading:90,done:100};
  let pollInterval=null;
  async function startWorkflow(){
    const url=document.getElementById('urlInput').value.trim();
    const errorMsg=document.getElementById('errorMsg');
    const btn=document.getElementById('startBtn');
    errorMsg.style.display='none';
    if(!url||(!url.includes('youtube')&&!url.includes('youtu.be'))){errorMsg.style.display='block';return;}
    btn.disabled=true;btn.textContent='Startet...';
    document.getElementById('previewSection').classList.remove('visible');
    document.getElementById('clipsGrid').innerHTML='';
    try{
      const res=await fetch('/process',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})});
      if(!res.ok){const err=await res.json();errorMsg.textContent='⚠️ '+(err.error||'Fehler.');errorMsg.style.display='block';btn.disabled=false;btn.textContent='Workflow starten →';return;}
      const data=await res.json();
      document.getElementById('statusCard').classList.add('visible');
      pollStatus(data.job_id);
    }catch(e){errorMsg.textContent='⚠️ Server nicht erreichbar.';errorMsg.style.display='block';btn.disabled=false;btn.textContent='Workflow starten →';}
  }
  function updateUI(status){
    document.getElementById('statusMessage').textContent=status.message;
    document.getElementById('progressFill').style.width=(PROGRESS[status.status]||5)+'%';
    STEPS.forEach(s=>{
      const el=document.getElementById('step-'+s);if(!el)return;
      el.className='step';
      const idx=STEPS.indexOf(s),curIdx=STEPS.indexOf(status.status);
      if(idx<curIdx)el.classList.add('done');else if(idx===curIdx)el.classList.add('active');
    });
    if(status.status==='done'&&status.clips){
      clearInterval(pollInterval);
      document.getElementById('startBtn').disabled=false;
      document.getElementById('startBtn').textContent='Weiteres Video →';
      showPreviews(status.clips);
    }
    if(status.status==='error'){
      clearInterval(pollInterval);
      document.getElementById('startBtn').disabled=false;
      document.getElementById('startBtn').textContent='Erneut versuchen →';
    }
  }
  function showPreviews(clips){
    const grid=document.getElementById('clipsGrid');
    grid.innerHTML='';
    clips.forEach((clip,i)=>{
      grid.innerHTML+=`
        <div class="clip-card" id="card-${i}">
          <div class="video-wrapper">
            <video controls preload="metadata" playsinline>
              <source src="${clip.url}" type="video/mp4">
            </video>
          </div>
          <div class="clip-info">
            <div class="clip-label">Clip ${clip.clip_number}</div>
            <div class="clip-duration">⏱ ${clip.duration}s</div>
            <div class="clip-reason">${clip.reason}</div>
            <div class="clip-actions">
              <button class="btn-approve" onclick="approveClip(${i},'${clip.url}','${clip.public_id}')">✓ Posten</button>
              <button class="btn-reject" onclick="rejectClip(${i})">✕</button>
            </div>
            <div class="clip-status" id="status-${i}"></div>
          </div>
        </div>`;
    });
    document.getElementById('previewSection').classList.add('visible');
    window.scrollTo({top:document.getElementById('previewSection').offsetTop-20,behavior:'smooth'});
  }
  function approveClip(idx,url,publicId){
    const statusEl=document.getElementById('status-'+idx);
    const card=document.getElementById('card-'+idx);
    statusEl.className='clip-status uploading';
    statusEl.textContent='⏳ Wird vorbereitet...';
    card.querySelectorAll('button').forEach(b=>b.disabled=true);
    setTimeout(()=>{
      statusEl.className='clip-status uploaded';
      statusEl.textContent='✓ Bereit zum Download';
      const a=document.createElement('a');
      a.href=url;a.download=`clip_${idx+1}.mp4`;a.target='_blank';a.click();
    },1000);
  }
  function rejectClip(idx){
    const statusEl=document.getElementById('status-'+idx);
    const card=document.getElementById('card-'+idx);
    statusEl.className='clip-status rejected';
    statusEl.textContent='✕ Abgelehnt';
    card.querySelectorAll('button').forEach(b=>b.disabled=true);
    card.style.opacity='0.4';
  }
  function pollStatus(jobId){
    pollInterval=setInterval(async()=>{
      try{const res=await fetch('/status/'+jobId);const data=await res.json();updateUI(data);}
      catch(e){}
    },1500);
  }
  document.addEventListener('DOMContentLoaded',()=>{
    document.getElementById('urlInput').addEventListener('keydown',e=>{if(e.key==='Enter')startWorkflow();});
  });
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML

@app.post("/process")
async def start_process(request: VideoRequest, background_tasks: BackgroundTasks):
    if not is_valid_youtube_url(request.url):
        return JSONResponse(status_code=400, content={"error": "Ungültige YouTube-URL"})
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "started", "message": "🚀 Workflow gestartet!", "url": request.url}
    background_tasks.add_task(process_video, job_id, request.url)
    return {"job_id": job_id, "message": "Workflow wurde gestartet!"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job nicht gefunden"})
    return jobs[job_id]