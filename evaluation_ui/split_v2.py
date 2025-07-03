import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

def analyze_and_print_report(data: list):
    """
    Analiza los datos para contar los dominios por dimensión e imprime un informe.

    Args:
        data: Una lista de diccionarios, donde se espera que cada uno tenga
              las claves 'dimension_code' y 'domain'.
    """
    dimension_domain_counts = defaultdict(lambda: defaultdict(int))

    for item in data:
        dimension = item.get("dimension_code")
        domain = item.get("domain")
        if dimension and domain:
            dimension_domain_counts[dimension][domain] += 1

    logging.info("\n--- Informe de Conteo (Después de Limitar) ---")
    if not dimension_domain_counts:
        logging.info("No hay datos para informar.")
        return

    grand_total = 0
    for dimension, domain_counts in sorted(dimension_domain_counts.items()):
        logging.info(f"\nDimensión: {dimension}")
        total_for_dimension = sum(domain_counts.values())
        grand_total += total_for_dimension
        
        for domain, count in sorted(domain_counts.items()):
            logging.info(f"  - {domain}: {count}")
        
        logging.info(f"  --------------------")
        logging.info(f"  Total para la dimensión: {total_for_dimension}")
    
    logging.info(f"\nGRAN TOTAL DE ELEMENTOS: {grand_total}")
    logging.info("------------------------------------------")

def main():
    # --- 1. Configuración y Carga ---
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    parser = argparse.ArgumentParser(
        description="Filtra, limita y divide datos JSON en partes para diferentes personas.")
    parser.add_argument(
        "input", type=Path,
        help="Ruta al archivo JSON de entrada que contiene un array.")
    args = parser.parse_args()

    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error(f"Error al leer o analizar el archivo JSON: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        logging.error(f"Se esperaba un array JSON en el nivel superior, pero se obtuvo {type(data).__name__}.")
        sys.exit(1)

    original_count = len(data)
    logging.info(f"Se cargaron {original_count} elementos de {args.input}.")

    # --- 2. Filtrar Dominios ---
    domains_to_remove = {"Family", "Lifestyle"}
    filtered_data = [
        item for item in data if item.get("domain") not in domains_to_remove
    ]
    logging.info(f"Se eliminaron {original_count - len(filtered_data)} elementos con dominio 'Family' o 'Lifestyle'.")

    # --- 3. Limitar a 30 por Grupo (Dimensión, Dominio) ---
    limit_per_group = 30
    group_counts = defaultdict(int)
    limited_data = []

    for item in filtered_data:
        dimension = item.get("dimension_code")
        domain = item.get("domain")

        if not (dimension and domain):
            continue  # Omitir si faltan claves esenciales

        group_key = (dimension, domain)
        if group_counts[group_key] < limit_per_group:
            limited_data.append(item)
            group_counts[group_key] += 1
    
    logging.info(f"Los datos se han limitado a un máximo de {limit_per_group} elementos por cada par dimensión-dominio.")
    logging.info(f"Número total de elementos después de limitar: {len(limited_data)}")

    # --- 4. Analizar y Verificar ---
    if not limited_data:
        logging.warning("No quedan datos después de filtrar y limitar. Saliendo.")
        sys.exit(0)
    analyze_and_print_report(limited_data)

    # --- 5. Dividir en 4 Partes ---
    num_parts = 5
    total_items = len(limited_data)
    base_size = total_items // num_parts
    remainder = total_items % num_parts

    logging.info(f"Dividiendo {total_items} elementos en {num_parts} partes.")
    
    parts = []
    current_index = 0
    for i in range(num_parts):
        # Los primeros 'remainder' partes obtienen un elemento extra para una división equitativa
        part_size = base_size + (1 if i < remainder else 0)
        end_index = current_index + part_size
        part_data = limited_data[current_index:end_index]
        parts.append(part_data)
        current_index = end_index
        logging.info(f"La Parte {i+1} tendrá {len(part_data)} elementos.")

    # --- 6. Escribir Archivos de Salida ---
    output_dir = Path(".") # Guardar en el directorio actual
    
    for i, part_data in enumerate(parts, 1):
        output_path = output_dir / f"part_{i}.json"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Usar indent=2 para que el JSON sea legible
                json.dump(part_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Se escribieron correctamente {len(part_data)} elementos en {output_path}")
        except Exception as e:
            logging.error(f"No se pudo escribir en {output_path}: {e}")


if __name__ == "__main__":
    # Necesitamos importar argparse dentro de la función o aquí
    import argparse
    main()
