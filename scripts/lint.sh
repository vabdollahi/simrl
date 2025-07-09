#!/usr/bin/env bash

echo "🔍 Checking for clang-format and clang-tidy..."

# Determine shell config file
detect_shell_config() {
    if [[ $SHELL == *"zsh" ]]; then
        echo "$HOME/.zshrc"
    elif [[ $SHELL == *"bash" ]]; then
        if [ -f "$HOME/.bashrc" ]; then
            echo "$HOME/.bashrc"
        else
            echo "$HOME/.bash_profile"
        fi
    else
        echo "$HOME/.profile"
    fi
}

install_if_missing() {
    if ! command -v "$1" &> /dev/null; then
        if [ "$1" = "clang-tidy" ]; then
            echo "❌ clang-tidy not found. Installing via Homebrew's llvm..."
            if ! command -v brew &> /dev/null; then
                echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh"
                exit 1
            fi
            brew install llvm
            config_file=$(detect_shell_config)
            echo ""
            echo "⚠️  Add LLVM to your PATH in your shell config:"
            echo "    echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> $config_file"
            echo "Then run: source $config_file"
            exit 1
        else
            echo "❌ $1 not found. Installing via Homebrew..."
            brew install "$1"
        fi
    else
        echo "✅ $1 found"
    fi
}

install_if_missing clang-format
install_if_missing clang-tidy

echo "🎯 Running clang-format on source and test files..."
find src include tests -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i

echo "🧠 Running clang-tidy..."
# Ensure compile_commands.json exists
if [ ! -f build/compile_commands.json ]; then
    echo "⚙️  Generating compile_commands.json..."
    mkdir -p build
    cd build
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
    cd ..
fi

# Run clang-tidy on all cpp files in src/, include/, and tests/
find src include tests -name "*.cpp" | while read -r file; do
    echo "🔍 Analyzing $file"
    clang-tidy "$file" -- -Iinclude
done

echo "✅ Linting complete!"
