import os
from PIL import Image

root_directory = "/Users/luca/Downloads/data_challenge_gathered"

# Mapping der Ordnerbezeichner
folder_mapping = {
    'Celebration': 'same',
    'CrossedArms-45deg-l': 'CrossedArms-45deg-r',
    'CrossedArms-45deg-r': 'CrossedArms-45deg-l',
    'CrossedArms-90deg-l': 'CrossedArms-90deg-r',
    'CrossedArms-90deg-r': 'CrossedArms-90deg-l',
    'CrossedArms-frontal': 'same',
    'Full Body': 'same',
    'Half Body': 'same',
    'HandsOnHips-45deg-l': 'HandsOnHips-45deg-r',
    'HandsOnHips-45deg-r': 'HandsOnHips-45deg-l',
    'HandsOnHips-90-deg-l': 'HandsOnHips-90deg-r',
    'HandsOnHips-90deg-r': 'HandsOnHips-90-deg-l',
    'Head Shot': 'same',
    'Hero': 'same',
    'HoldingBall': 'same',
    'HoldingBall-45deg-l': 'HoldingBall-45deg-r',
    'HoldingBall-45deg-r': 'HoldingBall-45deg-l',
}

# Funktion zum Spiegeln eines Bildes
def mirror_image(image_path):
    try:
        with Image.open(image_path) as img:
            mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            print(f"Bild gespiegelt: {image_path}")  # Debugging: Bild wurde gespiegelt
            return mirrored_img
    except Exception as e:
        print(f"Fehler beim Spiegeln des Bildes {image_path}: {e}")
        return None

# Funktion, um alle Bilder im Ordner und den Unterordnern zu spiegeln
def mirror_images_in_directory(root_dir):
    # Gehe durch alle Unterordner und Dateien im angegebenen Ordner
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # Überprüfe, ob die Datei eine Bilddatei ist und ob sie noch nicht mit '_mirrored' endet
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and '_mirrored' not in file:
                image_path = os.path.join(subdir, file)
                
                # Debugging: Überprüfe, ob das Bild richtig gefunden wird
                print(f"Bild gefunden: {image_path}")

                # Bestimme den Zielordner anhand des Ordnernamens
                folder_name = os.path.basename(subdir)
                target_folder = folder_mapping.get(folder_name, 'same')

                # Debugging: Überprüfe, welchen Zielordner wir verwenden
                print(f"Zielordner für {folder_name}: {target_folder}")

                # Zielordner erstellen, wenn nötig
                if target_folder != 'same':
                    target_dir = os.path.join(root_dir, target_folder)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                        print(f"Zielordner erstellt: {target_dir}")
                else:
                    target_dir = subdir

                # Spiegel das Bild
                mirrored_img = mirror_image(image_path)
                
                if mirrored_img is None:
                    print(f"Bild {image_path} konnte nicht gespiegelt werden.")
                    continue

                # Erstelle den neuen Dateinamen
                new_file_name = f"{os.path.splitext(file)[0]}_mirrored{os.path.splitext(file)[1]}"
                
                # Verhindern, dass der Name bereits '_mirrored' enthält
                new_image_path = os.path.join(target_dir, new_file_name)
                
                # Debugging: Zeige, wo das Bild gespeichert wird
                print(f"Gespiegeltes Bild wird gespeichert unter: {new_image_path}")
                
                # Speichere das gespiegelte Bild
                try:
                    mirrored_img.save(new_image_path)
                    print(f"Bild gespeichert: {new_image_path}")
                except Exception as e:
                    print(f"Fehler beim Speichern des Bildes {new_image_path}: {e}")
                continue  # Weiter mit dem nächsten Bild

mirror_images_in_directory(root_directory)