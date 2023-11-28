# Tuning mit LoRA
Low-Rank Adaptation of Large Language Models (LoRA) ist eine Trainingsmethode, die das Training großer Modelle beschleunigt und dabei weniger Speicherplatz verbraucht. Sie fügt Paare von Rangzerlegungs-Gewichtungsmatrizen (so genannte Aktualisierungsmatrizen) zu bestehenden Gewichten hinzu und trainiert nur die neu hinzugefügten Gewichte.

## Vorarbeiten
- Ordner *data* auf Cluster hochladen
- Datei *sBatchJobSD* auf Cluster hochladen
- Datei *train_text_to_image_lora.py* auf Cluster hochladen

## Schritte auf dem HPC-Cluster
1. Virtuelle Umgebung erstellen: `python3 -m venv venv`
2. Virtuelle Umgebung aktivieren: `source venv/bin/activate`
3. Abhängigkeiten installieren: `pip install accelerate torchvision transformers datasets ftfy tensorboard`
4. Diffusers installieren: `pip install git+https://github.com/huggingface/diffusers`
5. Accelerate konfigurieren: `accelerate config`
   1. Ich habe die genauen Einstellungen noch nicht herausgefunden. Funktionieren tut es mit diesen:
   2. Which type of machine are you using? *No distributed training*
   3. Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? *no*
   4. Do you wish to optimize your script with torch dynamo? *no*
   5. Do you want to use DeepSpeed? *no*
   6. What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? *all*
6. Ausführen des Batch-Jobs **"sbatchJobSD"**