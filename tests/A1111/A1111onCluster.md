# AUTOMATIC1111 - stable-diffusion-webui auf HPC Cluster ausführen

## Venv erstellen
1. 

## Einrichtung A1111
1. `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`
2. Datei "webui-user.sh" anpassen: `install_dir="/nfs/scratch/students/$(whoami)"`
3. Datei "webui-user.sh" anpassen: clone_dir einkommentieren
4. Datei "webui-user.sh" anpassen: `export COMMANDLINE_ARGS="--ckpt-dir '/nfs/scratch/students/$(whoami)/content'"`
5. Datei "webui-user.sh" anpassen: `venv_dir="/nfs/scratch/students/$(whoami)/venv"`
6. `source /nfs/scratch/students/$(whoami)/venv/bin/activate`
7. `pip install --upgrade pip`
8. `deactivate`
9. runA1111Job.sh hochladen

## Ausführung
1. `sbatch runA1111Job.sh` 
   1. Hinweis: Der Job läuft vor allem beim ersten Mal starten deutlich länger. Der Log kann sich live mit `tail -f <FILE>` angesehen werden.
2. Lokal: `ssh -N -L 127.0.0.1:7860:127.0.0.1:7860 <USER>@<NODE>.informatik.fh-nuernberg.de -i ~/.ssh/<KEY>`

Anschließend kann A1111 lokal über 127.0.0.1:7860 aufgerufen werden.

## Falls Modelle noch nicht vorhanden:
1. `mkdir /nfs/scratch/students/$(whoami)/content`
2. Gute Qualität: `wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true`
3. Sehr gute Qualität (in Verbindung mit Base): `wget https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors?download=true`
4. Extrem schnell: `wget https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors?download=true`
5. Umbenennen der Dateien (ohne "?download=true")

## Konfiguration der Modelle (abweichend vom Standard):
1. SDXL Base: Bildgröße = 1024x1024
2. SDXL Base + Refiner: Bildgröße = 1024x1024, bei Refiner das Modell einstellen, Sampling-Schritte = 30, Switch = 0,6
3. SDXL Turbo: Sampling Methode = Euler a, Sampling-Schritte = 1, CFG-Scale = 1 

## Dreambooth-Extension installieren
1. In Web-UI: "Extensions" -> "Available" -> "Load from:" -> Nach "Dreambooth" suchen -> "Install"
2. `scancel <JobId>`
3. `source /nfs/scratch/students/$(whoami)/venv/bin/activate`
4. BitsAndBytes installieren:
   1. `srun --qos=interactive --pty --partition=p2 bash -i`
   2. `source /nfs/scratch/students/$(whoami)/venv/bin/activate`
   3. `module load cuda/cuda-11.8.0`
   4. `python -m pip install bitsandbytes==0.41.2.post2  --prefer-binary`
   5. Tritt hier ein Fehler auf, folgendes vorher installieren:
      1. `pip install nvidia-pyindex` 
      2. `pip install nvidia-cuda-nvcc`
   6. `exit`
5. **WORKAROUND Bug in V0.24.0:** In Datei "stable-diffusion-webui/extensions/sd_dreambooth_extension/requirements.txt": `diffusers==0.23.1`
6. **WORKAROUND Bug in V0.24.0:** `pip uninstall diffusers`
7. Xformers installieren:
   1. `cd /nfs/scratch/students/$(whoami)/stable-diffusion-webui`
   2. `git clone https://github.com/facebookresearch/xformers.git`
   3. `cd xformers`
   4. `git submodule update --init --recursive`
   5. `pip install -r requirements.txt`
   6. `pip install -e .`
8. `deactivate`
9. `sbatch runA1111Job.sh` 

## Training
Referenz: https://github.com/d8ahazard/sd_dreambooth_extension/wiki/ELI5-Training
1. Modell erstellen:
   1. "Dreambooth" -> "Model" -> "Create"
   2. Name vergeben
   3. Checkpoint angeben und Modellart wählen
   4. "Create Model"
   5. Modell im Select-Tab auswählen
2. Training konfigurieren
3. Concept anlegen