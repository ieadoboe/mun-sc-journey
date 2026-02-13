# Simple script to compute powers of pi
import math
import argparse

# Use argparse to get input from command line,
#   by default compute pi**2, otherwise compute given integer powers
parser = argparse.ArgumentParser(description="Computes powers of pi")
parser.add_argument("--power", type=int, default=2, help="Integer power of pi to compute")
args = parser.parse_args()

# Print the value of pi**power
print(math.pow(math.pi, args.power))
