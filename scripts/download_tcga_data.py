#!/usr/bin/env python3
"""
TCGA (The Cancer Genome Atlas) Data Integration for Genesis RNA

This script helps download and process TCGA breast cancer RNA-seq data
for training and evaluating the Genesis RNA model.

TCGA is the premier cancer genomics program with data from 11,000+ patients.
For breast cancer (BRCA), TCGA has:
- ~1,100 breast cancer samples
- RNA-seq data (gene expression)
- Clinical annotations
- Mutation data

Data Access:
1. Public data: Available via GDC Data Portal (https://portal.gdc.cancer.gov/)
2. Controlled data: Requires dbGaP authorization

Usage:
    # List available TCGA-BRCA samples
    python download_tcga_data.py --list

    # Download public RNA-seq data
    python download_tcga_data.py --download --cancer_type BRCA --output data/tcga/brca

Note: This script focuses on public data. For full raw sequencing data,
      you'll need to apply for dbGaP access.
"""

import argparse
import json
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional
import time


class TCGADownloader:
    """
    Download and process TCGA data for cancer research

    Uses the GDC (Genomic Data Commons) API to access TCGA data.
    """

    def __init__(self):
        self.base_url = "https://api.gdc.cancer.gov"
        self.data_endpoint = f"{self.base_url}/data"
        self.files_endpoint = f"{self.base_url}/files"
        self.cases_endpoint = f"{self.base_url}/cases"

    def search_cases(
        self,
        cancer_type: str = "BRCA",
        data_type: str = "Gene Expression Quantification",
        workflow_type: str = "STAR - Counts"
    ) -> List[Dict]:
        """
        Search for TCGA cases matching criteria

        Args:
            cancer_type: TCGA project code (e.g., BRCA, LUAD, COAD)
            data_type: Type of data to download
            workflow_type: Data processing workflow

        Returns:
            List of file metadata
        """
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [f"TCGA-{cancer_type}"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_type",
                        "value": [data_type]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.analysis.workflow_type",
                        "value": [workflow_type]
                    }
                }
            ]
        }

        # Build query
        params = {
            "filters": json.dumps(filters),
            "format": "JSON",
            "size": "10000"  # Maximum allowed
        }

        print(f"🔍 Searching TCGA-{cancer_type} for {data_type}...")

        try:
            response = requests.get(self.files_endpoint, params=params)
            response.raise_for_status()

            data = response.json()
            files = data['data']['hits']

            print(f"   Found {len(files)} files")

            return files

        except Exception as e:
            print(f"❌ Error searching TCGA: {e}")
            return []

    def download_file(self, file_id: str, output_dir: str) -> bool:
        """
        Download a specific file from TCGA

        Args:
            file_id: GDC file UUID
            output_dir: Output directory

        Returns:
            True if successful
        """
        try:
            url = f"{self.data_endpoint}/{file_id}"

            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Get filename from content disposition
            content_disp = response.headers.get('content-disposition', '')
            if 'filename=' in content_disp:
                filename = content_disp.split('filename=')[1].strip('"')
            else:
                filename = f"{file_id}.txt"

            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True

        except Exception as e:
            print(f"   ❌ Error downloading {file_id}: {e}")
            return False

    def download_manifest(
        self,
        cancer_type: str,
        output_dir: str,
        max_files: Optional[int] = None
    ):
        """
        Download TCGA data files

        Args:
            cancer_type: TCGA cancer type code
            output_dir: Output directory
            max_files: Maximum number of files to download (None = all)
        """
        # Search for files
        files = self.search_cases(cancer_type)

        if not files:
            print("No files found.")
            return

        if max_files:
            files = files[:max_files]
            print(f"📥 Downloading first {max_files} files...")
        else:
            print(f"📥 Downloading {len(files)} files...")

        successful = 0
        failed = 0

        for i, file_info in enumerate(files):
            file_id = file_info['file_id']
            print(f"   [{i+1}/{len(files)}] Downloading {file_id}...")

            if self.download_file(file_id, output_dir):
                successful += 1
            else:
                failed += 1

            # Rate limiting
            time.sleep(0.5)

        print(f"\n✅ Download complete:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")


def create_tcga_info_file(output_dir: str):
    """Create information file about TCGA data access"""
    info = """
# TCGA Data Access Guide

## Overview
The Cancer Genome Atlas (TCGA) is the premier cancer genomics resource with
data from 11,000+ patients across 33 cancer types.

## Breast Cancer (TCGA-BRCA) Data
- ~1,100 tumor samples
- ~100 normal tissue samples
- RNA-seq gene expression data
- Mutation calls (SNVs, indels)
- Clinical data (survival, treatment, subtype)

## Data Levels

### 1. Open Access Data (No Authorization Required)
- Gene expression quantification
- Somatic mutation calls
- Clinical/biospecimen data

Access via:
- GDC Data Portal: https://portal.gdc.cancer.gov/
- This script: `python download_tcga_data.py --download --cancer_type BRCA`

### 2. Controlled Access Data (dbGaP Authorization Required)
- Raw sequencing reads (BAM/FASTQ)
- Genotype data
- Protected clinical data

Apply for access:
1. Visit: https://dbgap.ncbi.nlm.nih.gov/
2. Apply for TCGA project access
3. Download data using GDC Transfer Tool

## Data Processing for Genesis RNA

1. **Download RNA-seq counts**:
   ```bash
   python download_tcga_data.py --download --cancer_type BRCA --max_files 100
   ```

2. **Extract RNA sequences** (requires reference transcriptome):
   ```bash
   python process_tcga_rna.py --input data/tcga/brca --output data/tcga/processed
   ```

3. **Train model**:
   ```bash
   python -m genesis_rna.train_pretrain --data_path data/tcga/processed
   ```

## Citation

If you use TCGA data, please cite:

> The Cancer Genome Atlas Research Network. Comprehensive molecular portraits
> of human breast tumours. Nature 490, 61–70 (2012).
> https://doi.org/10.1038/nature11412

## Useful Resources

- GDC Data Portal: https://portal.gdc.cancer.gov/
- GDC API Documentation: https://docs.gdc.cancer.gov/API/
- TCGA Publications: https://www.cancer.gov/tcga/publications
- GDC Support: support@nci-gdc.datacommons.io
"""

    info_path = Path(output_dir) / "TCGA_DATA_ACCESS.md"
    info_path.parent.mkdir(parents=True, exist_ok=True)

    with open(info_path, 'w') as f:
        f.write(info)

    print(f"📄 Created TCGA info file: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download TCGA cancer genomics data for Genesis RNA'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available TCGA data files'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download TCGA data files'
    )
    parser.add_argument(
        '--cancer_type',
        type=str,
        default='BRCA',
        help='TCGA cancer type code (e.g., BRCA, LUAD, COAD). Default: BRCA'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/tcga',
        help='Output directory for downloaded data'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        help='Maximum number of files to download (default: all)'
    )
    parser.add_argument(
        '--create_info',
        action='store_true',
        help='Create TCGA data access information file'
    )

    args = parser.parse_args()

    downloader = TCGADownloader()

    if args.create_info:
        create_tcga_info_file(args.output)
        return

    if args.list:
        print("🔍 Listing TCGA data files...")
        files = downloader.search_cases(args.cancer_type)

        if files:
            print(f"\n📋 Found {len(files)} files for TCGA-{args.cancer_type}:")
            for i, f in enumerate(files[:20]):  # Show first 20
                print(f"   {i+1}. {f['file_name']} ({f['file_size'] / 1024 / 1024:.1f} MB)")

            if len(files) > 20:
                print(f"   ... and {len(files) - 20} more files")

            print(f"\nTo download: python {__file__} --download --cancer_type {args.cancer_type}")
        else:
            print("No files found.")

    elif args.download:
        print("="*70)
        print("TCGA DATA DOWNLOAD")
        print("="*70)
        print(f"\nCancer type: TCGA-{args.cancer_type}")
        print(f"Output directory: {args.output}")
        if args.max_files:
            print(f"Max files: {args.max_files}")

        print("\n⚠️  NOTE: This downloads public gene expression data.")
        print("   For raw sequencing data, you need dbGaP authorization.")
        print("   See: https://dbgap.ncbi.nlm.nih.gov/\n")

        downloader.download_manifest(args.cancer_type, args.output, args.max_files)

        # Create info file
        create_tcga_info_file(args.output)

        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"\nData saved to: {args.output}")
        print("\nNext steps:")
        print("1. Review TCGA_DATA_ACCESS.md for detailed information")
        print("2. Process RNA-seq data for Genesis RNA training")
        print("3. Train model on real cancer data")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # List available files:")
        print(f"  python {__file__} --list --cancer_type BRCA")
        print("\n  # Download first 10 files:")
        print(f"  python {__file__} --download --cancer_type BRCA --max_files 10")
        print("\n  # Create access information:")
        print(f"  python {__file__} --create_info --output data/tcga")


if __name__ == '__main__':
    main()
