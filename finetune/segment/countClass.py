import os
import numpy as np
from skimage.transform import resize

def resize_image_array(image_array, target_shape):
    resized = np.zeros(target_shape, dtype=image_array.dtype)
    for c in range(target_shape[0]):
        resized[c] = resize(
            image_array[c] if c < image_array.shape[0] else np.zeros_like(image_array[0]),
            target_shape[1:],
            mode='constant',
            preserve_range=True
        )
    return resized

def resize_label_array(label_array, target_shape):
    resized = resize(
        label_array[0],
        target_shape[1:],
        mode='constant',
        preserve_range=True
    )
    return resized[np.newaxis, ...]

def process_and_count_labels(filepath, counters=None, label_frequencies=None):
    """
    Processa um arquivo .npz, conta todos os rótulos e atualiza os contadores globais.

    Args:
        filepath (str): Caminho do arquivo .npz.
        counters (dict): Dicionário para rastrear contagem total e imagens processadas.
        label_frequencies (dict): Dicionário para rastrear a frequência de cada rótulo.
    """
    # Carrega os dados do arquivo .npz
    data = np.load(filepath)
    image_array = data['array']

    # Separar a última banda como rótulo e as demais como imagem
    class_array = image_array[-1:, :, :]  # Rótulo (1, H, W)

    # Atualiza contador de total de imagens
    if counters is not None:
        counters["total"] += 1

    # Conta todos os rótulos na máscara
    unique_labels, counts = np.unique(class_array, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label_frequencies is not None:
            if label in label_frequencies:
                label_frequencies[label] += count
            else:
                label_frequencies[label] = count

    #print(f"Arquivo {os.path.basename(filepath)} processado. Rótulos encontrados: {dict(zip(unique_labels, counts))}")


# Configuração inicial
data_dir = '../../dataset_palmls4claymodel/data/raw_train'
npz_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.npz')]

# Dicionários para contagem
counters = {"total": 0}
label_frequencies = {}

# Processa todos os arquivos e conta rótulos
for file in npz_files:
    process_and_count_labels(file, counters=counters, label_frequencies=label_frequencies)

# Exibe as estatísticas finais
print(f"\nProcessamento concluído!")
print(f"Total de imagens: {counters['total']}")
print("\nFrequência de cada rótulo:")
for label, count in sorted(label_frequencies.items()):
    print(f"Rótulo {label}: {count} pixels")
