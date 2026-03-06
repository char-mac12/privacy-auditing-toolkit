import sys
sys.path.append('.')

from attack_executor.range_sampler import RangeSampler

# Test configuration
config = {
    "num_masks": 2,
    "mask_model": "bert-base-uncased",
    "device": "cpu"  # Use CPU for testing
}

# Create sampler
sampler = RangeSampler(
    range_fn="word_replace",
    sample_size=5,
    config=config
)

# Test text
test_text = "The quick brown fox jumps over the lazy dog"

print("Original text:")
print(f"  {test_text}")
print(f"\nGenerating {sampler.sample_size} variants with {config['num_masks']} masks:\n")

# Generate neighborhood
neighborhood = sampler.sample(test_text)

# Print results
for i, variant in enumerate(neighborhood, 1):
    print(f"{i}. {variant}")

# Verify output
print("\n" + "="*60)
print("VERIFICATION:")
print(f"✓ Expected {sampler.sample_size} variants")
print(f"✓ Got {len(neighborhood)} variants")
print(f"✓ All are strings: {all(isinstance(v, str) for v in neighborhood)}")
print(f"✓ All different: {len(set(neighborhood)) == len(neighborhood)}")
print("="*60)