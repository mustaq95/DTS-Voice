#!/bin/bash
set -e

echo "🎨 Starting Next.js frontend..."

cd "$(dirname "$0")/../frontend"

npm run dev
