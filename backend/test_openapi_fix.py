#!/usr/bin/env python3
"""Test script to verify OpenAPI schema generation works after fixing SearchResponse."""

import sys
import os

# Add the parent directory of backend to the path so we can import backend as a package
# We are currently in /c/sovereign-ai/backend, so parent is /c/sovereign-ai
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_import_and_openapi():
    """Test that we can import the app and generate OpenAPI schema."""
    try:
        # Import the FastAPI app
        from backend.app.main import app

        # Try to generate the OpenAPI schema
        openapi_schema = app.openapi()

        # Check that the schema was generated successfully
        if openapi_schema and 'paths' in openapi_schema:
            print("SUCCESS: OpenAPI schema generated successfully")
            print(f"Found {len(openapi_schema['paths'])} paths")

            # Check specifically for knowledge endpoints
            knowledge_paths = [path for path in openapi_schema['paths'] if '/knowledge' in path]
            print(f"Knowledge endpoints found: {knowledge_paths}")

            return True
        else:
            print("ERROR: OpenAPI schema generation failed or returned empty schema")
            return False

    except Exception as e:
        print(f"ERROR: Failed to generate OpenAPI schema: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_import_and_openapi()
    sys.exit(0 if success else 1)