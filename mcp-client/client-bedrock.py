import asyncio
import sys
import json
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import boto3
from dotenv import load_dotenv

load_dotenv()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # Initialize Bedrock client for Claude
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"  # Update to your region if needed
        )
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Claude 3.5 Sonnet

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        response = await self.session.list_tools()

        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        final_text = []
        assistant_message_content = []

        request_body = {
            "messages": messages,
            "max_tokens": 1000,
            "anthropic_version": "bedrock-2023-05-31",
            "tools": available_tools
        }

        # First call to Bedrock Claude
        bedrock_response = self.bedrock.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )

        response_json = json.loads(bedrock_response['body'].read())
        for content in response_json.get("content", []):
            if content["type"] == "text":
                final_text.append(content["text"])
                assistant_message_content.append(content)
            elif content["type"] == "tool_use":
                tool_name = content["name"]
                tool_args = content["input"]

                result = await self.session.call_tool(tool_name, tool_args)

                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                assistant_message_content.append(content)

                messages.append({"role": "assistant", "content": assistant_message_content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": content["id"],
                        "content": str(result.content)
                    }]
                })

                request_body["messages"] = messages

                bedrock_response = self.bedrock.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body)
                )
                response_json = json.loads(bedrock_response['body'].read())
                final_text.append(response_json["content"][0]["text"])

        return "\n".join(final_text)

    async def chat_loop(self):
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())