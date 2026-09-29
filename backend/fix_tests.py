import re

with open('tests/test_agent.py', 'r') as f:
    content = f.read()

# Pattern to find: model_router=router, followed by newline, then exactly 12 spaces, then tool_executor=executor,
pattern = r'(model_router=router,\n)(\s{12})(tool_executor=executor,)'

# Replace with: model_router=router, newline, 12 spaces, model_gateway=FakeModelGateway(), newline, 12 spaces, tool_executor=executor,
replacement = r'\1\2model_gateway=FakeModelGateway(),\n\2\3'

new_content = re.sub(pattern, replacement, content)

with open('tests/test_agent.py', 'w') as f:
    f.write(new_content)
