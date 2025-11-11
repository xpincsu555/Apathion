"""Test CLI algorithm name handling for A* variants."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apathion.cli import ApathionCLI
from apathion.pathfinding.astar import AStarPathfinder


def test_algorithm_name_variants():
    """Test that different algorithm names create correct pathfinder types."""
    print("=" * 60)
    print("Testing CLI Algorithm Name Handling")
    print("=" * 60)
    
    cli = ApathionCLI()
    
    # Test 1: "astar" should create enhanced A*
    print("\n1. Testing 'astar' (should default to enhanced):")
    pathfinder = cli._create_pathfinder("astar")
    assert isinstance(pathfinder, AStarPathfinder)
    assert pathfinder.use_enhanced == True
    print(f"   ✓ Name: {pathfinder.name}")
    print(f"   ✓ use_enhanced: {pathfinder.use_enhanced}")
    print(f"   ✓ Type: {type(pathfinder).__name__}")
    
    # Test 2: "astar_enhanced" should create enhanced A*
    print("\n2. Testing 'astar_enhanced' (explicit enhanced):")
    pathfinder = cli._create_pathfinder("astar_enhanced")
    assert isinstance(pathfinder, AStarPathfinder)
    assert pathfinder.use_enhanced == True
    print(f"   ✓ Name: {pathfinder.name}")
    print(f"   ✓ use_enhanced: {pathfinder.use_enhanced}")
    print(f"   ✓ Type: {type(pathfinder).__name__}")
    
    # Test 3: "astar_basic" should create basic A*
    print("\n3. Testing 'astar_basic' (basic mode):")
    pathfinder = cli._create_pathfinder("astar_basic")
    assert isinstance(pathfinder, AStarPathfinder)
    assert pathfinder.use_enhanced == False
    print(f"   ✓ Name: {pathfinder.name}")
    print(f"   ✓ use_enhanced: {pathfinder.use_enhanced}")
    print(f"   ✓ Type: {type(pathfinder).__name__}")
    
    # Test 4: Verify algorithm names are correct for logging
    print("\n4. Verifying algorithm names for logging:")
    basic = cli._create_pathfinder("astar_basic")
    enhanced = cli._create_pathfinder("astar_enhanced")
    default = cli._create_pathfinder("astar")
    
    print(f"   astar_basic -> {basic.name}")
    print(f"   astar_enhanced -> {enhanced.name}")
    print(f"   astar (default) -> {default.name}")
    
    assert basic.name == "A*-Basic"
    assert enhanced.name == "A*-Enhanced"
    assert default.name == "A*-Enhanced"  # Default is enhanced
    
    print("\n" + "=" * 60)
    print("✅ All algorithm name tests passed!")
    print("=" * 60)
    
    # Print usage examples
    print("\n" + "=" * 60)
    print("Usage Examples:")
    print("=" * 60)
    print("\nCommand-line usage:")
    print("  apathion play --algorithm=astar_basic")
    print("  apathion play --algorithm=astar_enhanced")
    print("  apathion play --algorithm=astar  # defaults to enhanced")
    print("\nEvaluation with both variants:")
    print("  apathion evaluate --algorithms=astar_basic,astar_enhanced")
    print("\nIn logs, these will appear as:")
    print("  - astar_basic  → logged as 'A*-Basic'")
    print("  - astar_enhanced → logged as 'A*-Enhanced'")
    print("  - astar → logged as 'A*-Enhanced' (default)")
    print()


if __name__ == "__main__":
    test_algorithm_name_variants()

