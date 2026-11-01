# Clientes iPhone y Mac — Transcripción 24/7

Cómo mandar audio desde el iPhone y el Mac al sistema, y cómo saber si ha llegado. Todo lo que
aparece aquí depende de valores reales de `config.toml` y `src/transcription247/api.py` — si algún
día cambias el puerto, el token o las carpetas en el VPS, actualiza esta página.

Datos fijos usados en toda la guía:

- VPS en la tailnet: **`<VPS_HOSTNAME>`** — IP Tailscale `<TAILSCALE_IP>`
- API: `http://<VPS_HOSTNAME>:8247` o `http://<TAILSCALE_IP>:8247` (puerto de `config.toml` → `[api] puerto`)
- Carpeta bandeja de entrada en Drive: `Mi unidad/Mi organizacion/Transcripciones_24_7/_INBOX_AUDIO/`
- Resultados: `Mi organizacion/Transcripciones_24_7/<Cliente>/AAAA-MM/`
- El token de la API vive **solo** en el VPS: `~/.config/transcription247/api_token`. Nunca se copia
  a este documento ni a ningún sitio; se muestra una vez por SSH y se pega directamente donde toque.

> **Nota:** los nombres de acciones de Atajos están verificados contra documentación y tutoriales
> de 2026 (iOS/iPadOS 26); si algo no coincide al 100 % en tu pantalla, búscalo por la palabra clave
> en negrita (p. ej. "URL", "Formulario", "Menú") en el buscador de acciones del editor.

---

## 1. Resumen en 30 segundos

**iPhone, uso normal:**
1. Graba con **Notas de Voz** (funciona con la pantalla bloqueada, sin límite de duración).
2. Toca **Compartir** → elige el Atajo **"Transcribir 24/7"**.
3. Responde dos preguntas rápidas (título opcional, cliente) y espera la notificación de "subida
   completada". El Atajo hace el resto.
4. ~20-25 min después por cada hora de audio, la nota Markdown aparece en Drive y (si lo activas)
   llega un Telegram. (Medido: 20 min de audio → 6 min; 1 h → 20 min de transcripción +
   diarización, más el resumen. El tiempo crece de forma proporcional a la duración, y la memoria
   **no** crece: una grabación de 2 h consume la misma RAM que una de 20 min.)

**Mac, uso normal:**
1. Exporta o arrastra el archivo de audio a la carpeta `_INBOX_AUDIO` sincronizada por
   **Google Drive para ordenador**.
2. Ya está — el sistema recoge el archivo solo en menos de un minuto y lo borra de la bandeja al
   terminar.

**Si Tailscale falla o el Atajo da error:** usa la vía de Drive (§3) desde cualquier dispositivo;
es más lenta pero no depende de la VPN.

---

## 2. iPhone (método principal): Notas de Voz + Atajo por la API

### 2.1 Por qué esta vía

Sube el audio directamente al VPS por la tailnet, sin pasar por la sincronización de Drive: el job
aparece en la cola en segundos y puedes ver su estado en la web (§8). Requiere Tailscale conectado.

### 2.2 Antes de empezar: Tailscale conectado

1. Abre **Tailscale** en el iPhone y confirma "Conectado" con `<VPS_HOSTNAME>` en la lista.
2. Para no tener que abrirlo a mano cada vez: perfil (arriba a la derecha) → **VPN On Demand** →
   actívalo. Wi-Fi y Datos móviles en **"Siempre"** conecta en cualquier red; o **"No hacer nada"**
   + coincidencia MagicDNS para `*.ts.net`, que conecta solo cuando el Atajo llama a `<VPS_HOSTNAME>`
   (ficha oficial: Tailscale — *Using VPN On Demand for iOS and macOS*).
3. Comprobación rápida: Safari → `http://<TAILSCALE_IP>:8247/healthz` debe devolver `{"ok":true,...}`.
   Si no carga, Tailscale no está conectado o el servicio `t247-api` está caído.

### 2.3 Construir el Atajo "Transcribir 24/7", acción por acción

Abre la app **Atajos** → pestaña **Mis Atajos** → **+** (nuevo atajo). Nómbralo **Transcribir 24/7**.

**Paso 0 — Hacerlo aparecer en Compartir.** Toca el icono ⓘ (ajustes del atajo) →
activa **"Mostrar en la hoja de Compartir"** (o **"Usar con la hoja de Compartir"** según versión).
En **Tipos de entrada aceptados**, deja marcado **Audio** (y opcionalmente **Archivos**) y desmarca
el resto, para que el Atajo solo aparezca cuando compartes algo que sea audio.

Ahora añade las acciones en este orden (busca cada una por su nombre en el buscador de acciones,
abajo):

1. **Texto** — pega aquí el token de la API (lo obtienes por SSH, ver más abajo). Déjala arriba del
   todo para poder editarla sin rehacer el resto del Atajo.
2. **Solicitar entrada** (*Ask for Input*) — tipo **Texto**, mensaje "Título (opcional, deja vacío
   para que lo genere el resumen)". Guarda en la variable `Título`.
3. **Elegir del menú** (*Choose from Menu*) — "¿Cliente?" con un botón por cada clave de
   `config.toml → archivo.clientes`: `Centro Ejemplo` (Texto = `centro_ejemplo`),
   `Cliente Alfa` (`cliente_alfa`), `Cliente Beta` (`cliente_beta`), `Sin clasificar / no lo sé` (vacío, deja que
   el resumen lo adivine). Guarda cada rama en la variable `Cliente`.
4. **Solicitar entrada** — tipo **Número**, mensaje "Nº de hablantes (opcional, 1-10)", con
   **"Permitir que quede vacío"** activada si tu versión la ofrece. Guarda en `Hablantes`.

5. **Obtener contenido de URL** (*Get Contents of URL*) — esta es la acción que hace la subida:
   - URL: `http://<TAILSCALE_IP>:8247/upload`
   - Método: **POST**
   - Cabeceras (Headers): añade una con clave `Authorization` y valor `Bearer ` seguido de la
     variable `Texto` del paso 1 (debe quedar literalmente `Bearer eyJ...` sin comillas ni saltos
     de línea — cuidado con el espacio entre `Bearer` y el token).
   - Cuerpo de la solicitud (Request Body): **Formulario** (*Form*).
   - Añade un campo nuevo, nómbralo `file`, y como valor arrastra la variable **Entrada del atajo**
     (el audio recibido desde Notas de Voz). Al soltar un archivo/audio en un campo de formulario,
     Atajos lo trata automáticamente como archivo (tipo **Archivo**), no como texto.
   - Añade un campo `title` con valor = variable `Título` (déjalo vacío si `Título` está vacío;
     Atajos omite campos con valor vacío en un Formulario, así que no pasa nada si queda en blanco).
   - Añade un campo `client` con valor = variable `Cliente` (mismo criterio: si está vacío, el
     servidor decide el cliente con el resumen automático).
   - Añade un campo `speakers` con valor = variable `Hablantes` (solo si diste un número; **debe
     ser un entero entre 1 y 10** o la API devuelve error 400).

6. **Obtener valor de diccionario** (*Get Dictionary Value*) — sobre el resultado del paso 5, clave
   `job_id`.
7. **Mostrar notificación** (*Show Notification*) — título "Transcripción 24/7", cuerpo
   `Job: [job_id] — subida completa`. Es la única confirmación de que el archivo salió bien: si la
   subida falla, el paso 5 devuelve un error y no verás un `job_id` válido aquí.

**Obtener el token por SSH (una sola vez, para pegarlo en el paso 1):**

```bash
ssh USER@<TAILSCALE_IP>
cat ~/.config/transcription247/api_token
```

Copia el valor mostrado y pégalo en la acción **Texto** del paso 1 del Atajo. No lo guardes en
notas ni capturas de pantalla — es la única credencial de escritura de este proyecto, sin
relación con el token de solo-lectura de otros sistemas.

### 2.4 Tamaños esperados y consejos para archivos grandes

- 1 hora de audio en Notas de Voz (m4a comprimido) ronda los **30 MB**; el límite del servidor es
  **1 GiB** (`api.max_upload_bytes`), no vas a rozarlo en uso normal.
- **Mantén la app Atajos en primer plano** hasta la notificación del paso 7. iOS puede pausar la
  subida en segundo plano en reuniones muy largas (2-3 h) si cambias de app o se apaga la pantalla
  a mitad; enchufa el iPhone para esos casos.
- Si la subida se corta a medias, reintenta desde el Atajo (vuelve a compartir la misma nota de
  voz): la API detecta duplicados por hash de archivo, así que un reintento tras una subida que sí
  llegó completa no crea trabajo doble (§9, "duplicado").

### 2.5 Variante: grabar directamente dentro del Atajo (solo notas cortas)

Para audios cortos y espontáneos, un segundo Atajo, **"Transcribir — grabar ahora"**: acción
**Grabar audio** (*Record Audio*, calidad "Normal") en vez de recibir de Compartir, y luego los
mismos pasos 1-3 y 5-7 de §2.3 usando su salida como archivo.

**Advertencia:** esta acción **se detiene si bloqueas la pantalla o el Atajo pasa a segundo
plano** — a diferencia de Notas de Voz. Úsala solo para apuntes de menos de un par de minutos con
el teléfono en la mano; para cualquier reunión real sigue usando Notas de Voz → Compartir (§2.3).

---

## 3. iPhone — vía alternativa sin Tailscale (o si la subida falla)

Cuando no tengas Tailscale a mano, o la acción **Obtener contenido de URL** del Atajo dé error:

1. Notas de Voz → abre la grabación → **Compartir**.
2. Elige **Guardar en Archivos** (*Save to Files*).
3. Navega hasta **Google Drive** → `Mi unidad` → `Mi organizacion` → `Transcripciones_24_7` →
   `_INBOX_AUDIO` → **Guardar**.
4. El poller revisa esa carpeta cada 15 s y, tras comprobar que el tamaño no cambia durante 30 s,
   lo encola. El archivo **desaparece de `_INBOX_AUDIO`** al recogerse — es normal.

**Limitaciones:** requiere la app **Google Drive** instalada con el proveedor de **Archivos**
habilitado (Ajustes → Archivos → Ubicaciones); es manual, sin confirmación de "subida completa" en
el propio teléfono (consulta la web §8 o espera el Telegram); y más lenta que la API, porque
depende de que Drive sincronice antes de que el poller lo vea.

---

## 4. Opcional: app de terceros "Webhook Audio Recorder" — *sin verificar*

Existe una app gratuita de terceros, **Webhook Audio Recorder** (App Store, iOS y Android,
`rec.codejungle.pl`): graba y sube a un webhook con **cola sin conexión y reintentos automáticos**
si el móvil no tiene red o VPN en ese momento. Su ficha de la App Store menciona "personaliza
cabeceras, añade tokens de autenticación", lo que encajaría con `Authorization: Bearer`.

**Configuración orientativa (sin probar en el dispositivo real):** webhook =
`http://<TAILSCALE_IP>:8247/upload`, cabecera `Authorization: Bearer <token>`. **Sin verificar**
si permite renombrar el campo del archivo a `file` (la API lo exige literal, o falla con `400`) ni
si admite campos extra (`title`/`client`/`speakers`). Es una app de terceros no auditada por este
proyecto — trátala como respaldo, no como vía principal.

---

## 5. Mac (método principal, sin código): Google Drive para ordenador

1. Instala/abre **Google Drive para ordenador** y asegúrate de que `Mi unidad` está disponible —
   **"Transmitir archivos"** o **"Reflejar archivos"**, cualquiera vale mientras `_INBOX_AUDIO` sea
   una carpeta normal del Finder.
2. Localiza `Mi unidad/Mi organizacion/Transcripciones_24_7/_INBOX_AUDIO/` (arrástrala a la barra
   lateral del Finder para tenerla a mano).
3. Arrastra el audio ahí — desde QuickTime (**Archivo → Exportar como**) o desde Notas de Voz
   sincronizada por iCloud (**Compartir** → **Guardar en carpeta**).
4. Mismo comportamiento que en iPhone (§3): se recoge en ≤15 s tras 30 s de tamaño estable.

Es la vía recomendada en Mac: cero código, cero Atajos, solo tener Drive instalado.

### 5.1 Mac alternativa: `curl` + Acción rápida de Finder (Automator)

Para subir directamente por la API (job visible al instante en la web, sin esperar sincronización
de Drive), sin guardar el token en un fichero de texto plano:

**Paso 1 — guarda el token en el llavero (Keychain), una vez** (bórralo después del historial con
`history -d` si tu shell no lo ignora al llevar un espacio inicial):

```bash
security add-generic-password -a "$USER" -s "t247-api-token" -w "PEGA_AQUI_EL_TOKEN"
```

**Paso 2 — Automator → Acción rápida:**
1. **Automator** → **Nuevo documento** → **Acción rápida** (*Quick Action*).
2. Arriba: **El flujo de trabajo recibe** → **archivos o carpetas**, en **Finder.app**.
3. Añade **Ejecutar script de shell** (*Run Shell Script*), Shell `/bin/zsh`, **Entrada del
   proceso**: "como argumentos" (*as arguments*), y pega:

```bash
#!/bin/zsh
TOKEN=$(security find-generic-password -a "$USER" -s "t247-api-token" -w)
for f in "$@"; do
  curl -sS -H "Authorization: Bearer $TOKEN" \
    -F "file=@${f}" \
    http://<TAILSCALE_IP>:8247/upload
done
```

4. Guarda la Acción rápida como **"Enviar a Transcribir 24/7"**.

Uso: clic derecho sobre un audio en el Finder → **Acciones rápidas** → **Enviar a Transcribir
24/7**. Requiere Tailscale conectado (`tailscale status`). Sin `-F "title=…"` ni `-F "client=…"`
el resumen automático decide; añádelos igual que en `ARCHITECTURE.md` §6 si quieres fijarlos.

### 5.2 Opcional: watcher `launchd` en `~/Transcribir/`

Sube solo cualquier archivo soltado en una carpeta local, sin Drive ni clics en Automator.
**Opcional** — la vía de Drive (§5) ya cubre el caso "sin tocar nada"; monta esto solo si de
verdad lo vas a usar.

`~/Transcribir/subir.sh` (`chmod +x`), reutiliza el token del llavero de §5.1 y **borra el
archivo local solo si la API responde 200**, para que un fallo de red no pierda nada:
```bash
#!/bin/zsh
TOKEN=$(security find-generic-password -a "$USER" -s "t247-api-token" -w)
for f in "$HOME/Transcribir"/*.{m4a,mp3,wav,aac,ogg,opus,flac,mp4,mov,webm,caf}(N); do
  [ -f "$f" ] || continue
  curl -sS -o /tmp/t247-last.json -w "%{http_code}" \
       -H "Authorization: Bearer $TOKEN" -F "file=@${f}" \
       http://<TAILSCALE_IP>:8247/upload | grep -q '^200$' && rm -f "$f"
done
```

`~/Library/LaunchAgents/com.example.t247watch.plist` — dispara el script anterior cada vez que
`~/Transcribir/` cambia (clave `WatchPaths`):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.example.t247watch</string>
  <key>ProgramArguments</key>
  <array><string>/bin/zsh</string><string>-c</string><string>~/Transcribir/subir.sh</string></array>
  <key>WatchPaths</key><array><string>/Users/USER/Transcribir</string></array>
  <key>RunAtLoad</key><false/>
</dict></plist>
```
Cargar: `launchctl load ~/Library/LaunchAgents/com.example.t247watch.plist`. Descargar:
`launchctl unload …` (mismo path).

---

## 6. Grabar reuniones online en el Mac (Meet / Zoom / Teams), gratis

No hay integración directa con estas plataformas; hay que capturar el audio del sistema y guardarlo
como cualquier otro archivo (§5).

**Ruta recomendada — BlackHole + Dispositivo Agregado/Multisalida + QuickTime:**
1. Instala **BlackHole 2ch** (gratis, open source, `existential.audio/blackhole`); la versión
   0.7.1 está confirmada compatible con macOS 26 "Tahoe" (julio de 2026).
2. **Configuración de Audio y MIDI** (Utilidades) → **+** → **Crear dispositivo multisalida**:
   marca tus altavoces reales **y** BlackHole 2ch, para seguir escuchando mientras graba.
3. En Meet/Zoom/Teams, selecciona ese **Dispositivo Multisalida** como salida de audio (o
   **BlackHole 2ch** directo si no necesitas escuchar en vivo).
4. QuickTime → **Archivo → Nueva grabación de audio** → flechita junto al botón rojo → elige
   **BlackHole 2ch** como entrada → graba → exporta y súbelo por §5. (Para incluir tu propia voz
   además de la reunión, usa un **Dispositivo Agregado** con BlackHole + tu micrófono.)

**Alternativa:** grabadores de código abierto basados en **ScreenCaptureKit** (la API de captura de
audio de sistema de Apple desde macOS 13), que graban el audio de una app sin instalar un driver
virtual. **Sin verificar** cuál es la herramienta concreta más recomendable hoy (2026-09-19) —
busca "screen capture kit audio recorder macOS 26 open source" antes de instalar nada.

**Nota legal (RGPD y organización):** informa antes a los participantes de que la reunión se graba y
para qué (transcripción/acta interna), igual que con una grabadora física. Con menores de por
medio, sé especialmente cuidadoso: no grabes sesiones con participantes sin autorización expresa, y
guarda la nota resultante en las mismas carpetas ya protegidas que usas hoy.

---

## 7. Notificaciones por Telegram (opcional)

El worker manda un mensaje de éxito (título, duración, cliente, resumen corto, ruta en Drive) o de
fallo (job, archivo, etapa, error) usando `src/transcription247/notify.py`. Si no configuras nada,
sigue funcionando en silencio (`NullNotifier`) — es puramente opcional.

**1.** En Telegram, habla con **@BotFather** → `/newbot` → sigue las instrucciones → te da un
**token** (`123456789:AA...`).

**2.** Consigue tu `chat_id`: manda un mensaje a tu bot y visita
`https://api.telegram.org/bot<TOKEN>/getUpdates` — busca `"chat":{"id":...}` en la respuesta JSON.

**3.** En el VPS, crea el fichero de secretos (ruta fijada por `config.toml` →
`[paths] secrets_dir`, hoy `~/.config/transcription247`):

```bash
mkdir -p ~/.config/transcription247 && chmod 700 ~/.config/transcription247
cat > ~/.config/transcription247/telegram.json <<'EOF'
{"token": "TU_TOKEN_DE_BOTFATHER", "chat_id": "TU_CHAT_ID"}
EOF
chmod 600 ~/.config/transcription247/telegram.json
```

`config.py` **rechaza el fichero si no tiene permisos `0600`** — no te saltes el `chmod`.

**4.** Reinicia el worker (es quien manda notificaciones, no la API) para que recoja el fichero:

```bash
systemctl --user restart t247-worker.service
```

---

## 8. Página web: subir a mano y ver el estado de los trabajos

Con el iPhone o el Mac conectados a la tailnet, abre:

```
http://<VPS_HOSTNAME>:8247/
```

(o `http://<TAILSCALE_IP>:8247/` si el nombre `<VPS_HOSTNAME>` no resuelve — hace falta MagicDNS
activado en la cuenta de Tailscale para que el nombre corto funcione; la IP siempre funciona).

Página HTML sola, en español y pensada para el móvil: selector de archivo, campos de
título/cliente/idioma, un botón que sube por `/upload`, y una tabla con los últimos 20 trabajos
refrescada cada 10 s. La primera vez pide el token (se guarda en el propio navegador, nunca sale de
tu dispositivo). Útil para subir un archivo grande sin terminal, comprobar el estado de un job
(`received`, `summary_pending`, `delivered`, `failed`…) sin esperar al Telegram, y ver `note_path`
en cuanto termina para ir directo a la nota en Drive.

---

## 9. Solución de problemas

Comandos útiles en el VPS (todos dentro de `~/transcription-stack-template`):

```bash
uv run t247 status              # cola: cuántos jobs en cada estado, últimos jobs
uv run t247 doctor               # comprobaciones de entorno (Drive montado, modelos, claude, etc.)
uv run t247 retry <job_id>       # reintenta un job en 'failed'
uv run t247 resummarize <job_id> # regenera el resumen de un job en 'summary_pending'
journalctl --user -u t247-worker -f   # log en vivo del worker
journalctl --user -u t247-api -f      # log en vivo de la API
```

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| No llega la transcripción y no hay error visible | Job aún en cola/procesando (medido: ~20 min de proceso por hora de audio, más el resumen; una grabación de 2 h tarda ~40-50 min) | `uv run t247 status` para ver el estado real; si lleva horas en el mismo estado, revisa `journalctl -u t247-worker -f` |
| `401 unauthorized` al subir | Token mal copiado, con espacio de más, o caducado tras reinstalar el token | Vuelve a leer el token con `cat ~/.config/transcription247/api_token` por SSH y pégalo de nuevo en la acción **Texto** del Atajo (o en el llavero del Mac) |
| `413 too_large` | Archivo por encima de 1 GiB (`api.max_upload_bytes`) | No debería pasar en uso normal (1 h ≈ 30 MB); si pasa, revisa que no sea un vídeo sin comprimir |
| `415 unsupported_media_type` | Extensión fuera de la lista permitida (`m4a mp3 wav aac ogg opus flac mp4 mov webm caf`) | Comprueba la extensión real del archivo, no solo el nombre |
| Tailscale desconectado / no carga `/healthz` | VPN On Demand no disparó, o el móvil está en modo avión/ahorro de batería | Abre Tailscale a mano y comprueba "Conectado"; revisa la config de VPN On Demand (§2.2) |
| "Resumen pendiente" en la nota | El resumen falló dos veces (ver §13 de `ARCHITECTURE.md`, `claude` no logueado o límite de uso) | `uv run t247 resummarize <job_id>` una vez resuelto el problema con `claude`; la transcripción en sí ya está completa y entregada |
| Los hablantes se separan mal (mezclados o divididos de más) | El sistema no conocía el número real de hablantes y usó el modo automático | Vuelve a subir con el campo `speakers` (Atajo) o añade `[Np]`/`(2p)` al nombre del archivo antes de soltarlo en `_INBOX_AUDIO`, con `N` = número real de personas (1-10) |
| Aparece como "duplicado" / no se crea un job nuevo | Mismo archivo (mismo hash SHA-256) ya subido antes | Es el comportamiento esperado — la respuesta trae el `job_id` y la nota ya existentes; no hace falta volver a subirlo |
| `503`/`insufficient_storage` (507) | DB inaccesible o poco disco libre en el VPS | `uv run t247 doctor`; si es disco, libera espacio — el worker también poda artefactos con más de `conservar_artefactos_dias` (14 días) automáticamente |

---

**Recuerda:** este documento no contiene el token ni ningún secreto — solo instrucciones para
obtenerlos tú mismo por SSH cuando los necesites.
