#!/usr/bin/env bash
# Usage: ./build.sh /path/to/november  -> ./november_uci
set -euo pipefail
SRC="${1:?path to the folder holding module1..module12 .cpp files}"
HERE="$(cd "$(dirname "$0")" && pwd)"
sed 's/^int main() {/int main_module12_standalone() {/' "$SRC/module12_strengthened.cpp" > "$HERE/module12_lib.cpp"
g++ -std=c++17 -O2 -w -I "$SRC" "$HERE/november_uci.cpp" -o "$HERE/november_uci"
rm -f "$HERE/module12_lib.cpp"
echo "built $HERE/november_uci"
