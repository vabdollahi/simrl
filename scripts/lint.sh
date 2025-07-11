#!/usr/bin/env bash
ß
# Just lint
# ./scripts/lint.sh

# Lint and fix issues where safe
#./scripts/lint.sh --fix

set -e

echo "🔍 Checking for clang-format and clang-tidy..."

# ---------------------------
# 🔧 Detect Shell Config
# ---------------------------
detect_shell_config() {
    if [[ $SHELL == *"zsh" ]]; then
        echo "$HOME/.zshrc"
    elif [[ $SHELL == *"bash" ]]; then
        [[ -f "$HOME/.bashrc" ]] && echo "$HOME/.bashrc" || echo "$HOME/.bash_profile"
    else
        echo "$HOME/.profile"
    fi
}

# ---------------------------
# 🧪 Install Missing Tools
# ---------------------------
install_if_missing() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ $1 not found."
        if [ "$1" = "clang-tidy" ]; then
            echo "Installing via Homebrew's llvm..."
            if ! command -v brew &> /dev/null; then
                echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh"
                exit 1
            fi
            brew install llvm
            config_file=$(detect_shell_config)
            echo "⚠️  Add LLVM to your PATH:"
            echo "    echo 'export PATH=\"/opt/homebrew/opt/llvm/bin:\$PATH\"' >> $config_file"
            echo "Then run: source $config_file"
            exit 1
        else
            brew install "$1"
        fi
    else
        echo "✅ $1 found"
    fi
}

install_if_missing clang-format
install_if_missing clang-tidy

# ---------------------------
# 🧼 Run clang-format
# ---------------------------
echo "🎯 Running clang-format..."
find src include tests -name "*.cpp" -o -name "*.hpp" -print0 | xargs -0 clang-format -i

# ---------------------------
# 🧠 Run clang-tidy
# ---------------------------
FIX_FLAG=""
if [[ "$1" == "--fix" ]]; then
    echo "🛠️  Auto-fix mode enabled"
    FIX_FLAG="-fix"
fi

# Ensure compile_commands.json exists
if [ ! -f build/compile_commands.json ]; then
    echo "⚙️  Generating compile_commands.json..."
    mkdir -p build
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -S . -B build
fi

echo "🔎 Running clang-tidy..."
find src include tests -name "*.cpp" | while read -r file; do
    echo "🔍 Analyzing $file"
    clang-tidy $FIX_FLAG "$file" -- -Iinclude
done

echo "✅ Linting complete!"
