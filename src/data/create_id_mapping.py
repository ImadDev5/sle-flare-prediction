import pandas as pd
import logging
import gzip
import os
from pathlib import Path
import requests
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('id_mapping.log'),
        logging.StreamHandler()
    ]
)

# --- Configuration ---
# Platform ID for the microarray
PLATFORM_ID = "GPL10558"

# Local and remote paths
EXPRESSION_FILE = r"D:\SLE_data\processed\expression_normalized.csv"
STRING_INFO_FILE = r"D:\SLE_data\raw\STRING\protein.info.v12.0.txt.gz"
OUTPUT_DIR = Path(r"D:\SLE_data\processed\mapping")

# URL to fetch GPL annotation if not found locally
# Using a reliable mirror for GEO data
GPL_ANNOTATION_URL = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={PLATFORM_ID}&targ=gpl&view=data&form=text"


def get_gpl_annotation(platform_id: str) -> pd.DataFrame:
    """Fetches and parses the GPL annotation file from GEO."""
    logging.info(f"Fetching annotation for platform {platform_id} from GEO...")
    try:
        response = requests.get(GPL_ANNOTATION_URL)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Find the start of the data table
        lines = response.text.split('\n')
        data_start_line = -1
        for i, line in enumerate(lines):
            if line.startswith('ID\t'):
                data_start_line = i
                break
        
        if data_start_line == -1:
            raise ValueError("Could not find data table in GPL annotation file.")

        # Read the table into a pandas DataFrame
        table_text = '\n'.join(lines[data_start_line:])
        annot_df = pd.read_csv(StringIO(table_text), sep='\t')
        logging.info(f"Successfully parsed annotation data. Shape: {annot_df.shape}")
        return annot_df

    except requests.RequestException as e:
        logging.error(f"Failed to download annotation file: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to parse annotation file: {e}")
        raise

def create_mapping():
    """Creates and saves a mapping from Illumina Probe IDs to STRING Protein IDs."""
    logging.info("--- Starting ID Mapping Process ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Expression Data to get target Probe IDs
    logging.info(f"Loading expression data from {EXPRESSION_FILE}")
    expr_df = pd.read_csv(EXPRESSION_FILE)
    probe_ids = expr_df['Unnamed: 0'].unique()
    logging.info(f"Found {len(probe_ids)} unique probe IDs in expression data.")

    # 2. Get GPL10558 Annotation (Probe ID -> Gene Symbol)
    annot_df = get_gpl_annotation(PLATFORM_ID)
    # We need the 'ID' (probe) and 'Symbol' (gene symbol) columns
    probe_to_symbol = annot_df[['ID', 'Symbol']].dropna().drop_duplicates()
    probe_to_symbol = probe_to_symbol.set_index('ID')['Symbol']
    logging.info(f"Created mapping for {len(probe_to_symbol)} probes to gene symbols.")

    # 3. Load STRING Info (Gene Symbol -> STRING ID)
    logging.info(f"Loading STRING info file from {STRING_INFO_FILE}")
    with gzip.open(STRING_INFO_FILE, 'rt') as f:
        string_info_df = pd.read_csv(f, sep='\t')
    string_info_df = string_info_df[['#string_protein_id', 'preferred_name']]
    symbol_to_string_id = string_info_df.set_index('preferred_name')['#string_protein_id']
    logging.info(f"Created mapping for {len(symbol_to_string_id)} gene symbols to STRING IDs.")

    # 4. Create the final mapping: Probe ID -> STRING ID
    logging.info("Joining mappings to create final Probe ID -> STRING ID map...")
    mapped_probes = []
    for probe_id in probe_ids:
        gene_symbol = probe_to_symbol.get(probe_id)
        if gene_symbol:
            string_id = symbol_to_string_id.get(gene_symbol)
            if string_id:
                mapped_probes.append({'ProbeID': probe_id, 'GeneSymbol': gene_symbol, 'StringID': string_id})
    
    final_mapping_df = pd.DataFrame(mapped_probes)
    logging.info(f"Successfully mapped {len(final_mapping_df)} out of {len(probe_ids)} probes to STRING IDs.")

    # 5. Save the mapping
    output_file = OUTPUT_DIR / 'probe_to_string_id_mapping.csv'
    final_mapping_df.to_csv(output_file, index=False)
    logging.info(f"Final mapping saved to {output_file}")

    logging.info("--- ID Mapping Process Finished ---")

if __name__ == "__main__":
    create_mapping()
