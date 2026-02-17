#!/usr/bin/env python3
"""
Direct test for LEAK #7: XML ElementTree Objects Not Cleared

Tests if XML parsing without clear() causes memory accumulation.
"""

import sys
import tracemalloc
import xml.etree.ElementTree as ET

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<MediaContainer size="20">
    <Directory ratingKey="12345" key="/library/metadata/12345" type="season" title="Season 1">
        <Video ratingKey="67890" title="Test Episode 1" index="1">
            <Part id="1" file="/media/show/s01e01.mp4" size="1000000" />
        </Video>
        <Video ratingKey="67891" title="Test Episode 2" index="2">
            <Part id="2" file="/media/show/s01e02.mp4" size="2000000" />
        </Video>
    </Directory>
</MediaContainer>"""


def test_xml_parsing_without_clear():
    """Test if parsing XML repeatedly without clear() leaks."""
    tracemalloc.start()
    baseline = tracemalloc.take_snapshot()

    # Parse XML many times without clearing
    trees = []
    for i in range(500):
        root = ET.fromstring(SAMPLE_XML)
        for video in root.findall(".//Video"):
            _ = video.get("ratingKey")
        trees.append(root)  # Keep reference

    current = tracemalloc.take_snapshot()
    stats = current.compare_to(baseline, "lineno")
    total_memory = sum(stat.size_diff for stat in stats) / 1024 / 1024

    print(f"\n{'=' * 60}")
    print(f"LEAK #7 TEST: XML Parsing Without clear()")
    print(f"{'=' * 60}")
    print(f"XML parses: 500")
    print(f"Trees held: {len(trees)}")
    print(f"Memory growth: {total_memory:.2f} MB")
    print(f"Per tree: {total_memory / 500:.3f} MB")
    print(f"{'=' * 60}")

    # Now test if clear() helps
    before_clear = tracemalloc.take_snapshot()
    for root in trees:
        root.clear()
    trees.clear()

    after_clear = tracemalloc.take_snapshot()
    stats2 = after_clear.compare_to(before_clear, "lineno")
    freed = abs(sum(stat.size_diff for stat in stats2)) / 1024 / 1024

    print(f"Memory freed by clear(): {freed:.2f} MB")
    print(f"{'=' * 60}")

    if total_memory > 10.0:
        print(f"❌ SIGNIFICANT LEAK: {total_memory:.2f} MB for 500 parses")
        return False
    elif total_memory > 2.0:
        print(f"⚠️  MINOR LEAK: {total_memory:.2f} MB (GC will clean but delayed)")
        return False
    else:
        print(f"✅ NO SIGNIFICANT LEAK: {total_memory:.2f} MB is acceptable")
        return True


def test_xml_parsing_with_clear():
    """Test if parsing with immediate clear() prevents leak."""
    tracemalloc.start()
    baseline = tracemalloc.take_snapshot()

    # Parse and immediately clear
    for i in range(500):
        root = ET.fromstring(SAMPLE_XML)
        try:
            for video in root.findall(".//Video"):
                _ = video.get("ratingKey")
        finally:
            root.clear()

    current = tracemalloc.take_snapshot()
    stats = current.compare_to(baseline, "lineno")
    total_memory = sum(stat.size_diff for stat in stats) / 1024 / 1024

    print(f"\n{'=' * 60}")
    print(f"LEAK #7 TEST: XML Parsing WITH clear()")
    print(f"{'=' * 60}")
    print(f"XML parses: 500")
    print(f"Memory growth: {total_memory:.2f} MB")
    print(f"{'=' * 60}")

    if total_memory < 1.0:
        print(f"✅ clear() helps: memory stays low")
        return True
    else:
        print(f"⚠️  Even with clear(), {total_memory:.2f} MB growth")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING LEAK #7: XML ElementTree Objects")
    print("=" * 60)

    test1_pass = test_xml_parsing_without_clear()
    test2_pass = test_xml_parsing_with_clear()

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    if not test1_pass:
        print(f"❌ LEAK #7 IS REAL - XML trees accumulate without clear()")
        print(f"   clear() helps: {'YES' if test2_pass else 'NO'}")
        sys.exit(1)
    else:
        print(f"✅ NO SIGNIFICANT LEAK - GC handles it adequately")
        sys.exit(0)
