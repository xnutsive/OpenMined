#!/bin/sh

UNITY_EXECUTABLE="/Applications/Unity/Unity.app/Contents/MacOS/Unity"
PROJECT_FOLDER="$(pwd)/UnityProject"
BUILD_FOLDER="$(pwd)/openmined"
EDITOR_LOG_FILEPATH=" $(echo ~/Library/Logs/Unity/Editor.log)"

echo $PROJECT_FOLDER

## Run Unity Editor tests
echo "Testing Unity project: $PROJECT_FOLDER"
echo "Building Player..."

# Build the player
"$UNITY_EXECUTABLE" \
-projectPath "$PROJECT_FOLDER" \
-nographics \
-batchmode \
-buildOSXUniversalPlayer "$BUILD_FOLDER" \
-quit

echo "Player build successfully"
echo "Running playing in background"

# Run the player
./openmined.app/Contents/MacOS/openmined \
-nographics \
-batchmode \
&

# Player process ID so we can kill later
PLAYER_PID=$!

echo "Player began to run"
echo "Running integration tests"

# Run the tests themselves
pytest integration/*

kill -9 PLAYER_PID
