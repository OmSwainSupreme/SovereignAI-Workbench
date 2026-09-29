#!/usr/bin/env python3
"""Focused test for knowledge router OpenAPI schema generation."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_knowledge_router_import():
    """Test that knowledge router can be imported without errors."""
    try:
        from backend.app.api.v1.knowledge_router import SearchResponse, SearchRequest, SearchResultItem
        print("SUCCESS: Knowledge router imports correctly")

        # Test that we can instantiate the models
        request = SearchRequest(query="test query")
        print(f"SUCCESS: SearchRequest instantiated: {request}")

        result_item = SearchResultItem(
            chunk_id="test-id",
            text="test text",
            metadata={},
            score=0.9,
            rank=1
        )
        print(f"SUCCESS: SearchResultItem instantiated: {result_item}")

        response = SearchResponse(
            query="test query",
            total_available=1,
            results=[result_item],
            count=1
        )
        print(f"SUCCESS: SearchResponse instantiated: {response}")

        return True
    except Exception as e:
        print(f"ERROR: Failed to import or instantiate knowledge router models: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_openapi_schema_generation():
    """Test that OpenAPI schema can be generated successfully."""
    try:
        from backend.app.main import app

        # Generate OpenAPI schema
        openapi_schema = app.openapi()

        if not openapi_schema:
            print("ERROR: OpenAPI schema is None")
            return False

        if 'paths' not in openapi_schema:
            print("ERROR: No paths in OpenAPI schema")
            return False

        print(f"SUCCESS: OpenAPI schema generated with {len(openapi_schema['paths'])} paths")

        # Check specifically for knowledge endpoints
        knowledge_paths = [path for path in openapi_schema['paths'] if '/knowledge' in path]
        expected_paths = ['/api/v1/knowledge/search', '/api/v1/knowledge/collections']

        for expected_path in expected_paths:
            if expected_path not in knowledge_paths:
                print(f"ERROR: Expected path {expected_path} not found in schema")
                print(f"Available knowledge paths: {knowledge_paths}")
                return False
            else:
                print(f"SUCCESS: Found expected path {expected_path}")

        # Check that the search endpoint has a POST operation
        search_path_info = openapi_schema['paths'].get('/api/v1/knowledge/search', {})
        if 'post' not in search_path_info:
            print("ERROR: POST operation not found for /api/v1/knowledge/search")
            return False

        print("SUCCESS: POST operation found for /api/v1/knowledge/search")

        # Check that the response model is referenced
        post_operation = search_path_info['post']
        if 'responses' not in post_operation:
            print("ERROR: No responses defined for POST /api/v1/knowledge/search")
            return False

        print("SUCCESS: Responses defined for POST /api/v1/knowledge/search")

        return True

    except Exception as e:
        print(f"ERROR: Failed to generate OpenAPI schema: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Knowledge Router Import ===")
    import_success = test_knowledge_router_import()

    print("\n=== Testing OpenAPI Schema Generation ===")
    schema_success = test_openapi_schema_generation()

    if import_success and schema_success:
        print("\n=== ALL TESTS PASSED ===")
        sys.exit(0)
    else:
        print("\n=== SOME TESTS FAILED ===")
        sys.exit(1)