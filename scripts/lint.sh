#!/usr/bin/env bash

# Just lint:
#   ./scripts/lint.sh
# Lint and auto-fix issues where safe:
#   ./scripts/lint.sh --fix

set -e

echo "🔍 Checking for clang-format and clang-tidy..."

# ---------------------------
# 🔧 Detect Shell Config (used only for local PATH hints)
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
# 🧪 Install Missing Tools (local only)
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

# Skip install in CI
if [[ -z "$CI" ]]; then
    install_if_missing clang-format
    install_if_missing clang-tidy
fi

# ---------------------------
# 🔍 Determine Lint Mode
# ---------------------------
FIX_CLANG=0
if [[ "$1" == "--fix" ]]; then
    FIX_CLANG=1
    echo "🛠️  Auto-fix mode enabled"
fi

# ---------------------------
# 🧼 Run clang-format
# ---------------------------
if [[ $FIX_CLANG -eq 1 ]]; then
    FORMAT_FLAG="-i"
else
    FORMAT_FLAG="-n --Werror"
fi

echo "🎯 Running clang-format with flags: $FORMAT_FLAG"
find src include tests -name "*.cpp" -o -name "*.hpp" -print0 | xargs -0 clang-format $FORMAT_FLAG --style=file

# ---------------------------
# 🧠 Run clang-tidy
# ---------------------------
if [[ ! -f build/compile_commands.json ]]; then
    echo "⚙️  Generating compile_commands.json..."
    mkdir -p build
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -S . -B build
fi

TIDY_FIX_FLAG=""
if [[ $FIX_CLANG -eq 1 ]]; then
    TIDY_FIX_FLAG="-fix"
fi

echo "🔎 Running clang-tidy with flags: $TIDY_FIX_FLAG"
find src include tests -name "*.cpp" | while read -r file; do
    echo "🔍 Analyzing $file"
    clang-tidy $TIDY_FIX_FLAG "$file" -- -Iinclude
done

echo "✅ Linting complete!"
