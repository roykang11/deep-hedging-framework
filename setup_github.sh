#!/bin/bash
# GitHub Setup Script for Deep Hedging Framework
# Run this script after creating the repository on GitHub

echo "Setting up GitHub repository for Deep Hedging Framework..."

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "Error: Please run this script from the deep_hedging_framework directory"
    exit 1
fi

# Get GitHub username
echo "Please enter your GitHub username:"
read GITHUB_USERNAME

# Add remote origin
echo "Adding remote origin..."
git remote add origin https://github.com/$GITHUB_USERNAME/deep-hedging-framework.git

# Set main branch
echo "Setting main branch..."
git branch -M main

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main

echo "Success! Your repository is now on GitHub:"
echo "https://github.com/$GITHUB_USERNAME/deep-hedging-framework"
echo ""
echo "Next steps:"
echo "1. Add a description to your repository"
echo "2. Add topics: deep-learning, quantitative-finance, hedging, neural-networks, pytorch"
echo "3. Enable GitHub Pages if you want to host documentation"
echo "4. Add a license badge to your README"
echo ""
echo "Your project is now ready for your resume!"
