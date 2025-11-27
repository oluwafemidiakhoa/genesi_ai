#!/usr/bin/env python3
"""
HGVS Parser for Variant Notation
Parses HGVS nomenclature (e.g., c.5266dupC) and applies mutations to sequences.
"""

import re
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ParsedVariant:
    """Parsed HGVS variant"""
    variant_type: str  # 'substitution', 'deletion', 'insertion', 'duplication', 'frameshift'
    position: int
    reference: Optional[str] = None
    alternate: Optional[str] = None
    end_position: Optional[int] = None


class HGVSParser:
    """
    Parse HGVS notation and apply variants to reference sequences.

    Supports:
    - Substitutions: c.123A>T
    - Deletions: c.123del, c.123_125del
    - Insertions: c.123_124insAT
    - Duplications: c.123dup, c.123_125dup
    - Frameshifts: c.123delinsAT
    """

    # HGVS patterns
    SUBSTITUTION_PATTERN = r'c\.(\d+)([ACGT])>([ACGT])'
    DELETION_PATTERN = r'c\.(\d+)(?:_(\d+))?del([ACGT]*)'
    INSERTION_PATTERN = r'c\.(\d+)_(\d+)ins([ACGT]+)'
    DUPLICATION_PATTERN = r'c\.(\d+)(?:_(\d+))?dup([ACGT]*)'
    DELINS_PATTERN = r'c\.(\d+)(?:_(\d+))?delins([ACGT]+)'

    @staticmethod
    def parse(hgvs_string: str) -> ParsedVariant:
        """
        Parse HGVS notation into structured variant.

        Args:
            hgvs_string: HGVS nomenclature (e.g., "c.5266dupC")

        Returns:
            ParsedVariant object

        Raises:
            ValueError: If HGVS format is invalid
        """
        # Substitution: c.123A>T
        match = re.match(HGVSParser.SUBSTITUTION_PATTERN, hgvs_string)
        if match:
            position = int(match.group(1))
            reference = match.group(2)
            alternate = match.group(3)
            return ParsedVariant(
                variant_type='substitution',
                position=position,
                reference=reference,
                alternate=alternate
            )

        # Deletion: c.123del or c.123_125del
        match = re.match(HGVSParser.DELETION_PATTERN, hgvs_string)
        if match:
            position = int(match.group(1))
            end_position = int(match.group(2)) if match.group(2) else position
            reference = match.group(3) if match.group(3) else None
            return ParsedVariant(
                variant_type='deletion',
                position=position,
                end_position=end_position,
                reference=reference
            )

        # Insertion: c.123_124insAT
        match = re.match(HGVSParser.INSERTION_PATTERN, hgvs_string)
        if match:
            position = int(match.group(1))
            alternate = match.group(3)
            return ParsedVariant(
                variant_type='insertion',
                position=position,
                alternate=alternate
            )

        # Duplication: c.123dup or c.123_125dup
        match = re.match(HGVSParser.DUPLICATION_PATTERN, hgvs_string)
        if match:
            position = int(match.group(1))
            end_position = int(match.group(2)) if match.group(2) else position
            reference = match.group(3) if match.group(3) else None
            return ParsedVariant(
                variant_type='duplication',
                position=position,
                end_position=end_position,
                reference=reference
            )

        # Deletion-Insertion: c.123delinsAT
        match = re.match(HGVSParser.DELINS_PATTERN, hgvs_string)
        if match:
            position = int(match.group(1))
            end_position = int(match.group(2)) if match.group(2) else position
            alternate = match.group(3)
            return ParsedVariant(
                variant_type='delins',
                position=position,
                end_position=end_position,
                alternate=alternate
            )

        raise ValueError(f"Unable to parse HGVS notation: {hgvs_string}")

    @staticmethod
    def apply_variant(reference_seq: str, variant: ParsedVariant, convert_to_rna: bool = True) -> str:
        """
        Apply parsed variant to reference sequence.

        Args:
            reference_seq: Reference DNA sequence
            variant: Parsed variant from parse()
            convert_to_rna: Convert T to U for RNA (default: True)

        Returns:
            Mutant sequence with variant applied
        """
        # Convert to uppercase
        seq = reference_seq.upper()

        # Convert 1-based HGVS position to 0-based Python index
        pos = variant.position - 1

        if variant.variant_type == 'substitution':
            # Verify reference base
            if variant.reference and seq[pos] != variant.reference:
                raise ValueError(
                    f"Reference mismatch at position {variant.position}: "
                    f"expected {variant.reference}, found {seq[pos]}"
                )
            # Apply substitution
            mutant_seq = seq[:pos] + variant.alternate + seq[pos+1:]

        elif variant.variant_type == 'deletion':
            end_pos = variant.end_position - 1 if variant.end_position else pos
            # Verify reference if provided
            if variant.reference:
                deleted_bases = seq[pos:end_pos+1]
                if deleted_bases != variant.reference:
                    raise ValueError(
                        f"Reference mismatch for deletion: "
                        f"expected {variant.reference}, found {deleted_bases}"
                    )
            # Apply deletion
            mutant_seq = seq[:pos] + seq[end_pos+1:]

        elif variant.variant_type == 'insertion':
            # Insert after position
            mutant_seq = seq[:pos+1] + variant.alternate + seq[pos+1:]

        elif variant.variant_type == 'duplication':
            end_pos = variant.end_position - 1 if variant.end_position else pos
            duplicated_bases = seq[pos:end_pos+1]
            # Verify reference if provided
            if variant.reference and duplicated_bases != variant.reference:
                raise ValueError(
                    f"Reference mismatch for duplication: "
                    f"expected {variant.reference}, found {duplicated_bases}"
                )
            # Apply duplication (insert duplicated bases after original)
            mutant_seq = seq[:end_pos+1] + duplicated_bases + seq[end_pos+1:]

        elif variant.variant_type == 'delins':
            end_pos = variant.end_position - 1 if variant.end_position else pos
            # Apply deletion-insertion
            mutant_seq = seq[:pos] + variant.alternate + seq[end_pos+1:]

        else:
            raise ValueError(f"Unknown variant type: {variant.variant_type}")

        # Convert to RNA if requested
        if convert_to_rna:
            mutant_seq = mutant_seq.replace('T', 'U')

        return mutant_seq


def apply_hgvs_variant(reference_seq: str, hgvs_string: str, convert_to_rna: bool = True) -> Dict[str, str]:
    """
    Convenience function to parse and apply HGVS variant in one step.

    Args:
        reference_seq: Reference DNA sequence
        hgvs_string: HGVS notation (e.g., "c.5266dupC")
        convert_to_rna: Convert to RNA (default: True)

    Returns:
        Dictionary with 'wildtype' and 'mutant' sequences
    """
    parser = HGVSParser()
    variant = parser.parse(hgvs_string)
    mutant_seq = parser.apply_variant(reference_seq, variant, convert_to_rna)

    # Convert reference to RNA if requested
    wildtype_seq = reference_seq.upper()
    if convert_to_rna:
        wildtype_seq = wildtype_seq.replace('T', 'U')

    return {
        'wildtype': wildtype_seq,
        'mutant': mutant_seq,
        'variant_type': variant.variant_type
    }


if __name__ == '__main__':
    # Test examples
    test_cases = [
        ("c.5266dupC", "ATCGATCG"),  # Duplication
        ("c.123A>T", "GGGAAACCC"),  # Substitution
        ("c.5del", "ATCGATCG"),  # Deletion
        ("c.3_5del", "ATCGATCG"),  # Range deletion
        ("c.3_4insGG", "ATCGATCG"),  # Insertion
    ]

    parser = HGVSParser()

    for hgvs, ref_seq in test_cases:
        print(f"\nTest: {hgvs}")
        print(f"Reference: {ref_seq}")
        try:
            result = apply_hgvs_variant(ref_seq, hgvs)
            print(f"Wildtype:  {result['wildtype']}")
            print(f"Mutant:    {result['mutant']}")
            print(f"Type:      {result['variant_type']}")
        except Exception as e:
            print(f"Error: {e}")
