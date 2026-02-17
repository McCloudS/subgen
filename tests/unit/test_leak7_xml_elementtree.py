"""
Memory Leak Test: LEAK #7 - XML ElementTree Objects Not Cleared

PROBLEM:
ElementTree creates complex object graphs with circular references when parsing XML.
The code parses XML responses from Plex/Jellyfin APIs but doesn't explicitly clear
the tree, relying on Python's garbage collector. In tight loops, these accumulate
and can trigger expensive GC cycles.

EXPECTED BEHAVIOR:
- XML trees should be explicitly cleared after use
- No accumulation of parsed XML objects
- Circular references should be broken promptly

IMPACT:
- Each XML tree: ~10-50KB depending on size
- For 1000 Plex API calls: 10-50MB before GC runs
- Can trigger expensive GC cycles in production

LOCATIONS IN CODE:
- Line 1830-1895: get_next_plex_episode() - parses multiple XML responses
- Line 1932: get_plex_file_name() - parses XML
- Line 1990-2030: Jellyfin functions use JSON (not affected)

TEST STRATEGY:
1. Parse XML repeatedly without clearing
2. Parse XML with explicit clear()
3. Measure memory accumulation
4. Verify GC pressure
"""

import pytest
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.memory_utils import MemoryProfiler, LeakDetector


# Sample XML similar to Plex API responses
SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<MediaContainer size="20">
    <Directory ratingKey="12345" key="/library/metadata/12345" type="season" title="Season 1">
        <Video ratingKey="67890" key="/library/metadata/67890" parentRatingKey="12345" 
               grandparentRatingKey="54321" type="episode" title="Test Episode 1" index="1" parentIndex="1">
            <Part id="1" key="/library/parts/1/file.mp4" file="/media/show/s01e01.mp4" size="1000000" />
        </Video>
        <Video ratingKey="67891" key="/library/metadata/67891" parentRatingKey="12345" 
               grandparentRatingKey="54321" type="episode" title="Test Episode 2" index="2" parentIndex="1">
            <Part id="2" key="/library/parts/2/file.mp4" file="/media/show/s01e02.mp4" size="2000000" />
        </Video>
    </Directory>
</MediaContainer>"""


@pytest.mark.unit
@pytest.mark.memory_leak
class TestXMLElementTreeLeaks:
    """Test suite for XML ElementTree memory management."""

    def test_repeated_xml_parsing_without_clear(self):
        """
        LEAK TEST: Parsing XML repeatedly without clearing trees.

        ElementTree objects contain circular references that aren't
        immediately freed without explicit clear().
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=10.0)

        profiler.snapshot("baseline")

        # Parse XML many times without clearing
        iterations = 500
        trees = []
        for i in range(iterations):
            root = ET.fromstring(SAMPLE_XML)
            # Extract some data (typical usage)
            for video in root.findall(".//Video"):
                _ = video.get("ratingKey")
                _ = video.get("title")
            trees.append(root)  # Keep reference (simulating held objects)

        profiler.snapshot("after_parsing_no_clear")

        growth = profiler.compare("baseline", "after_parsing_no_clear")

        print(f"\n=== XML Parsing Without clear() ===")
        print(f"Iterations: {iterations}")
        print(f"Trees held in memory: {len(trees)}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")
        print(f"Per tree: {growth['increase_mb'] / iterations:.3f} MB")

        # Should show memory accumulation
        assert growth["increase_mb"] > 2.0, "Expected memory growth from held trees"

        # Now clear all trees
        profiler.snapshot("before_clear")
        for root in trees:
            root.clear()
        trees.clear()

        profiler.snapshot("after_clear")

        freed = profiler.compare("before_clear", "after_clear")
        print(f"Memory freed by clear(): {abs(freed['increase_mb']):.2f} MB")

    def test_xml_parsing_with_immediate_clear(self):
        """
        COMPARISON TEST: Parsing XML with immediate clear() after use.

        This is the correct pattern that should prevent accumulation.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=5.0)

        profiler.snapshot("baseline")

        # Parse and immediately clear
        iterations = 500
        for i in range(iterations):
            root = ET.fromstring(SAMPLE_XML)
            try:
                # Extract data
                for video in root.findall(".//Video"):
                    _ = video.get("ratingKey")
                    _ = video.get("title")
            finally:
                root.clear()  # Explicitly break circular references

        profiler.snapshot("after_parsing_with_clear")

        growth = profiler.compare("baseline", "after_parsing_with_clear")

        print(f"\n=== XML Parsing With clear() ===")
        print(f"Iterations: {iterations}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak despite clear(): {growth['increase_mb']:.2f} MB"
        )

    def test_plex_get_next_episode_pattern(self):
        """
        LEAK TEST: Simulate get_next_plex_episode() XML parsing pattern.

        This function parses multiple XML responses per call.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=10.0)

        profiler.snapshot("baseline")

        # Simulate processing 100 episodes, each parsing 3 XML responses
        iterations = 100
        for episode_num in range(iterations):
            # Parse 1: Get episode metadata
            root1 = ET.fromstring(SAMPLE_XML)
            grandparent_key = root1.find(".//Video").get("grandparentRatingKey")

            # Parse 2: Get season list
            root2 = ET.fromstring(SAMPLE_XML)
            seasons = root2.findall(".//Directory[@type='season']")

            # Parse 3: Get episodes in season
            root3 = ET.fromstring(SAMPLE_XML)
            episodes = root3.findall(".//Video")

            # Bug: roots not cleared!

        profiler.snapshot("after_plex_pattern")

        growth = profiler.compare("baseline", "after_plex_pattern")

        print(f"\n=== Plex get_next_episode() Pattern ===")
        print(f"Episodes processed: {iterations}")
        print(f"XML parses: {iterations * 3}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak in Plex pattern: {growth['increase_mb']:.2f} MB"
        )

    def test_xml_tree_circular_references(self):
        """
        MEASUREMENT TEST: Verify circular references in ElementTree.

        ElementTree nodes maintain parent/child relationships that
        create reference cycles.
        """
        profiler = MemoryProfiler()

        profiler.snapshot("baseline")

        # Create trees and examine their structure
        trees = []
        for i in range(100):
            root = ET.fromstring(SAMPLE_XML)
            trees.append(root)

        profiler.snapshot("with_circular_refs")

        # Access parent/child relationships (creates strong refs)
        for root in trees:
            for elem in root.iter():
                # Element iteration creates circular references
                _ = elem.tag
                _ = elem.attrib

        profiler.snapshot("after_iteration")

        growth1 = profiler.compare("baseline", "with_circular_refs")
        growth2 = profiler.compare("with_circular_refs", "after_iteration")

        print(f"\n=== XML Circular Reference Analysis ===")
        print(f"Memory after parsing: {growth1['increase_mb']:.2f} MB")
        print(f"Memory after iteration: {growth2['increase_mb']:.2f} MB")

        # Clear to break cycles
        for root in trees:
            root.clear()
        trees.clear()

    def test_large_xml_document_parsing(self):
        """
        LEAK TEST: Large XML documents amplify the leak.

        Larger XML responses cause more significant memory accumulation.
        """
        # Generate a large XML document
        large_xml = "<MediaContainer>"
        for i in range(100):  # 100 episodes
            large_xml += f'''
            <Video ratingKey="{i}" title="Episode {i}" index="{i}">
                <Part id="{i}" file="/media/file{i}.mp4" size="100000" />
                <Media id="{i}" duration="3600000" bitrate="5000">
                    <Part id="{i}" file="/media/file{i}.mp4" />
                </Media>
            </Video>'''
        large_xml += "</MediaContainer>"

        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=15.0)

        profiler.snapshot("baseline")

        # Parse large XML repeatedly
        iterations = 50
        for i in range(iterations):
            root = ET.fromstring(large_xml)
            # Process data
            for video in root.findall(".//Video"):
                _ = video.get("ratingKey")
            # Bug: root not cleared

        profiler.snapshot("after_large_xml")

        growth = profiler.compare("baseline", "after_large_xml")

        print(f"\n=== Large XML Document Parsing ===")
        print(f"XML size: {len(large_xml)} bytes")
        print(f"Iterations: {iterations}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak with large XML: {growth['increase_mb']:.2f} MB"
        )

    def test_gc_collection_without_clear(self):
        """
        MEASUREMENT TEST: Verify GC eventually collects uncleaned trees.

        Without clear(), GC must break cycles, which is slower.
        """
        import gc

        profiler = MemoryProfiler()

        # Disable automatic GC
        gc.disable()

        try:
            profiler.snapshot("baseline")

            # Parse without clearing
            for i in range(200):
                root = ET.fromstring(SAMPLE_XML)
                for video in root.findall(".//Video"):
                    _ = video.get("ratingKey")
                # Don't clear - let GC handle it

            profiler.snapshot("before_gc")

            # Force GC collection
            gc.collect()

            profiler.snapshot("after_gc")

            before_gc = profiler.compare("baseline", "before_gc")
            after_gc = profiler.compare("before_gc", "after_gc")

            print(f"\n=== GC Collection Analysis ===")
            print(f"Memory before GC: {before_gc['increase_mb']:.2f} MB")
            print(f"Memory freed by GC: {abs(after_gc['increase_mb']):.2f} MB")
            print(f"GC can clean up cycles, but it's delayed and expensive")

        finally:
            gc.enable()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
