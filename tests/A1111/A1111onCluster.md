# AUTOMATIC1111 - stable-diffusion-webui auf HPC Cluster ausführen
1. `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`
2. Datei "webui-user.sh" anpassen: install_dir einkommentieren
3. Datei "webui-user.sh" anpassen: clone_dir einkommentieren
3. Datei "webui-user.sh" anpassen: `export COMMANDLINE_ARGS="--ckpt-dir '/nfs/scratch/students/$(whoami)/content'"`
3. Datei "webui-user.sh" anpassen: TORCH_COMMAND einkommentieren
4. sbatchJobSD.sh hochladen
5. `sbatch sbatchJobSD.sh` 
   1. Hinweis: Der Job läuft vor allem beim ersten Mal starten deutlich länger. Der Log kann sich live mit `tail -f <FILE>` angesehen werden.
6. Lokal: `ssh -N -L 127.0.0.1:7860:127.0.0.1:7860 <USER>@<NODE>.informatik.fh-nuernberg.de -i ~/.ssh/<KEY>`

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