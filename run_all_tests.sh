#!/bin/bash

# Option Calculator Test Suite
# This script runs all test configurations and saves results

echo "=========================================="
echo "Option Calculator - Test Suite"
echo "=========================================="
echo ""

# Create results directory
mkdir -p test_results

# Array of test configurations
tests=(
    "european_call.json:European ATM Call"
    "european_put.json:European ATM Put"
    "american_call.json:American ATM Call"
    "american_put.json:American ITM Put"
    "test_american_call_itm.json:American ITM Call"
    "test_asian_arithmetic_call.json:Asian Arithmetic Call"
    "test_asian_geometric_call.json:Asian Geometric Call"
    "test_barrier_down_out_call.json:Down-and-Out Barrier Call"
    "test_barrier_up_out_call.json:Up-and-Out Barrier Call"
)

# Run each test
for test_info in "${tests[@]}"; do
    # Split into filename and description
    IFS=':' read -r filename description <<< "$test_info"
    
    echo "=========================================="
    echo "Test: $description"
    echo "Config: $filename"
    echo "=========================================="
    
    # Run the test
    python3.12 main.py --config "config/$filename"
    
    # Also save to file
    python3.12 main.py --config "config/$filename" --output "test_results/${filename%.json}_results.json" --format json
    
    echo ""
    echo "Results saved to: test_results/${filename%.json}_results.json"
    echo ""
    
    # Small delay between tests
    sleep 1
done

echo "=========================================="
echo "All tests completed!"
echo "Results saved in test_results/ directory"
echo "=========================================="
