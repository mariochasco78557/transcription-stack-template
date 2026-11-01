"""Single-page HTML UI: `GET /` (ARCHITECTURE.md §6, §14 P5).

Spanish, mobile-first, no JS framework, no external assets/CDN. Every string that comes from
config or the user is escaped before it lands in the HTML (`html.escape` server-side for the
client <select> options; the recent-jobs table is built client-side with `textContent`, never
`innerHTML`, so job data returned by `/jobs` can never inject markup).
"""

from __future__ import annotations

import html

from transcription247.config import Config

_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITULO__</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 16px; font-family: -apple-system, system-ui, sans-serif;
    max-width: 640px; margin-inline: auto; line-height: 1.4;
  }
  h1 { font-size: 1.25rem; margin: 0 0 1rem; }
  h2 { font-size: 1rem; margin: 2rem 0 0.5rem; }
  label { display: block; margin-top: 0.75rem; font-size: 0.9rem; }
  input, select, button {
    width: 100%; font-size: 16px; padding: 0.5rem; margin-top: 0.25rem;
    border: 1px solid #888; border-radius: 6px; background: Field; color: FieldText;
  }
  button {
    margin-top: 1rem; background: #2563eb; color: #fff; border: none; font-weight: 600;
    cursor: pointer;
  }
  button:disabled { opacity: 0.6; }
  progress { width: 100%; margin-top: 0.5rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem; }
  th, td { text-align: left; padding: 0.35rem 0.25rem; border-bottom: 1px solid #ccc; }
  #estado { margin-top: 0.5rem; font-size: 0.9rem; }
  .oculto { display: none; }
</style>
</head>
<body>
<h1>__TITULO__</h1>

<div id="token-box">
  <label for="token">Token de acceso</label>
  <input id="token" type="password" autocomplete="off" placeholder="Pega aquí el token">
  <button id="guardar-token" type="button">Guardar token</button>
</div>

<form id="form-subida">
  <label for="file">Audio o vídeo</label>
  <input id="file" name="file" type="file" accept="audio/*,video/*" required>

  <label for="title">Título (opcional)</label>
  <input id="title" name="title" type="text" maxlength="120">

  <label for="client">Cliente (opcional)</label>
  <select id="client" name="client">
    <option value="">— Detectar automáticamente —</option>
__OPCIONES_CLIENTE__
  </select>

  <label for="language">Idioma</label>
  <select id="language" name="language">
    <option value="">Automático</option>
    <option value="es">Español</option>
    <option value="en">English</option>
  </select>

  <label for="speakers">Nº de hablantes (opcional)</label>
  <input id="speakers" name="speakers" type="number" min="1" max="10" step="1">

  <button id="btn-subir" type="submit">Subir grabación</button>
  <progress id="progreso" value="0" max="100" class="oculto"></progress>
  <div id="estado"></div>
</form>

<h2>Últimas transcripciones</h2>
<table id="tabla-jobs">
  <thead><tr><th>Fecha</th><th>Fichero</th><th>Estado</th><th>Nota</th></tr></thead>
  <tbody></tbody>
</table>

<script>
(function () {
  "use strict";
  var TOKEN_KEY = "t247_token";

  function getToken() { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch (e) { return ""; } }
  function setToken(t) { try { localStorage.setItem(TOKEN_KEY, t); } catch (e) {} }

  var tokenBox = document.getElementById("token-box");
  var tokenInput = document.getElementById("token");
  var existing = getToken();
  if (existing) { tokenInput.value = existing; tokenBox.classList.add("oculto"); }

  document.getElementById("guardar-token").addEventListener("click", function () {
    var value = tokenInput.value.trim();
    if (!value) { return; }
    setToken(value);
    tokenBox.classList.add("oculto");
  });

  var form = document.getElementById("form-subida");
  var estado = document.getElementById("estado");
  var progreso = document.getElementById("progreso");
  var btn = document.getElementById("btn-subir");

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    var token = getToken();
    if (!token) {
      estado.textContent = "Falta el token de acceso.";
      tokenBox.classList.remove("oculto");
      return;
    }
    var fileInput = document.getElementById("file");
    if (!fileInput.files.length) { return; }

    var data = new FormData();
    data.append("file", fileInput.files[0]);
    var title = document.getElementById("title").value.trim();
    if (title) { data.append("title", title); }
    var client = document.getElementById("client").value;
    if (client) { data.append("client", client); }
    var language = document.getElementById("language").value;
    if (language) { data.append("language", language); }
    var speakers = document.getElementById("speakers").value;
    if (speakers) { data.append("speakers", speakers); }

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload");
    xhr.setRequestHeader("Authorization", "Bearer " + token);
    btn.disabled = true;
    progreso.classList.remove("oculto");
    progreso.value = 0;
    estado.textContent = "Subiendo…";

    xhr.upload.addEventListener("progress", function (ev) {
      if (ev.lengthComputable) { progreso.value = Math.round((ev.loaded / ev.total) * 100); }
    });

    xhr.addEventListener("load", function () {
      btn.disabled = false;
      progreso.classList.add("oculto");
      var body = {};
      try { body = JSON.parse(xhr.responseText); } catch (e) {}
      if (xhr.status === 200) {
        estado.textContent = body.duplicate
          ? "Ya existía (job " + body.job_id + ")."
          : "Subida correcta (job " + body.job_id + ").";
        form.reset();
        cargarJobs();
      } else {
        estado.textContent = "Error " + xhr.status + ": " + (body.error || xhr.statusText);
      }
    });

    xhr.addEventListener("error", function () {
      btn.disabled = false;
      progreso.classList.add("oculto");
      estado.textContent = "Error de red al subir.";
    });

    xhr.send(data);
  });

  function celda(texto) {
    var td = document.createElement("td");
    td.textContent = texto === null || texto === undefined ? "" : texto;
    return td;
  }

  function cargarJobs() {
    var token = getToken();
    if (!token) { return; }
    fetch("/jobs?limit=20", { headers: { Authorization: "Bearer " + token } })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (data) {
        var tbody = document.querySelector("#tabla-jobs tbody");
        tbody.innerHTML = "";
        (data.jobs || []).forEach(function (job) {
          var tr = document.createElement("tr");
          tr.appendChild(celda(job.created_at));
          tr.appendChild(celda(job.source_name));
          tr.appendChild(celda(job.state));
          tr.appendChild(celda(job.note_path));
          tbody.appendChild(tr);
        });
      })
      .catch(function () { /* silent: network blip, next poll retries */ });
  }

  cargarJobs();
  setInterval(cargarJobs, 10000);
})();
</script>
</body>
</html>
"""


def index_html(cfg: Config) -> str:
    """Self-contained HTML for `GET /` (§6). No framework, no CDN, mobile-first, Spanish UI."""
    titulo = html.escape(cfg.api.titulo_web)
    opciones = "\n".join(
        f'    <option value="{html.escape(key)}">{html.escape(cliente.nombre)}</option>'
        for key, cliente in sorted(cfg.archivo.clientes.items())
    )
    return _TEMPLATE.replace("__TITULO__", titulo).replace("__OPCIONES_CLIENTE__", opciones)
