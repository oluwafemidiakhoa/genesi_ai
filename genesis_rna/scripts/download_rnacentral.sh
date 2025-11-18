#!/bin/bash
# Download RNA sequences from RNAcentral database
# RNAcentral is a comprehensive database of non-coding RNA sequences

set -e

DATA_DIR="${1:-./data/rnacentral}"
mkdir -p "$DATA_DIR"

echo "Downloading RNAcentral data to $DATA_DIR"

# RNAcentral FTP server
FTP_BASE="ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral"

# Download active sequences (FASTA format)
echo "Downloading active sequences..."
wget -c "$FTP_BASE/current_release/sequences/rnacentral_active.fasta.gz" \
     -O "$DATA_DIR/rnacentral_active.fasta.gz"

# Download species-specific sequences (optional)
# Uncomment for specific organisms:

# Human
# echo "Downloading human sequences..."
# wget -c "$FTP_BASE/current_release/sequences/by-database/ena/homo_sapiens.fasta.gz" \
#      -O "$DATA_DIR/homo_sapiens.fasta.gz"

# Mouse
# echo "Downloading mouse sequences..."
# wget -c "$FTP_BASE/current_release/sequences/by-database/ena/mus_musculus.fasta.gz" \
#      -O "$DATA_DIR/mus_musculus.fasta.gz"

# Extract compressed files
echo "Extracting files..."
gunzip -k "$DATA_DIR/rnacentral_active.fasta.gz" || true

echo "Download complete!"
echo "Data location: $DATA_DIR"
echo ""
echo "Next steps:"
echo "1. Run preprocess_rna.py to convert FASTA to training format"
echo "2. python scripts/preprocess_rna.py --input $DATA_DIR/rnacentral_active.fasta --output $DATA_DIR/processed"
